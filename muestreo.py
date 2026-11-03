"""Pregunta 2: muestreo progresivo y particiones sin reemplazo.

python muestreo.py ../pregunta1_completa/resultados resultados
Los parametros se guardan antes de calcular las curvas. No se entrena un MLP.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

import preprocesamiento as prep

CLASES = prep.reglas.CLASES
SEMILLAS = [17, 42, 73]
PASO = 10_000
TOLERANCIA = 0.01
CONSECUTIVOS = 3
MARGEN = 1.0
FRACCIONES = [0.10, 0.15, 0.20, 0.25]


def asignar(n, capacidades):
    """Reparto proporcional entero por restos mayores; suma exacta."""
    capacidades_disponibles = np.asarray(capacidades, dtype=int)
    if not 0 <= n <= capacidades_disponibles.sum():
        raise ValueError("Tamano fuera de la reserva disponible")
    ideales = n * capacidades_disponibles / capacidades_disponibles.sum()
    cuotas = np.floor(ideales).astype(int)
    orden = np.argsort(-(ideales - cuotas), kind="stable")
    cuotas[orden[: n - cuotas.sum()]] += 1
    assert cuotas.sum() == n and (cuotas <= capacidades_disponibles).all()
    return cuotas


def restaurar_marcas(df, bitacora):
    """Copia previa a imputacion, conservando las correcciones del dominio."""
    salida = df.copy()
    if not salida[prep.COLUMNA_ID].is_unique:
        raise ValueError("IDs repetidos")
    indice = pd.Series(salida.index, index=salida[prep.COLUMNA_ID])
    if bitacora.duplicated([prep.COLUMNA_ID, "columna"]).any():
        raise ValueError("Marcas repetidas")
    for columna, datos_grupo in bitacora.groupby("columna"):
        if columna not in prep.PENDIENTES:
            raise ValueError("Marca ajena al Bloque 3")
        posiciones = indice.loc[datos_grupo[prep.COLUMNA_ID]].to_numpy()
        if columna in prep.NUMERICAS:
            salida[columna] = salida[columna].astype(float)
        salida.loc[posiciones, columna] = np.nan
    return salida


def descriptores():
    columnas_estadisticas = [
        (columna, estadistico, "")
        for columna in prep.NUMERICAS
        for estadistico in ["media", "desviacion"]
    ]
    columnas_estadisticas += [
        (columna, "proporcion", categoria)
        for columna in prep.CATEGORICAS
        for categoria in prep.reglas.DOMINIOS_CATEGORICOS[columna]
    ]
    columnas_estadisticas += [
        (columna, "tasa_ausencia", "") for columna in prep.PENDIENTES
    ]
    return columnas_estadisticas


def resumen_prefijos(datos_grupo, tamanos):
    """Suficientes acumulados: cada fila aparece una vez en cada orden."""
    numericos = datos_grupo[prep.NUMERICAS].to_numpy(float)
    observados = np.isfinite(numericos)
    numericos_sin_ausencias = np.where(observados, numericos, 0.0)
    posiciones_finales = np.asarray(tamanos) - 1
    cantidades = np.cumsum(observados, axis=0)[posiciones_finales]
    sumas = np.cumsum(numericos_sin_ausencias, axis=0)[posiciones_finales]
    cuadrados = np.cumsum(numericos_sin_ausencias * numericos_sin_ausencias, axis=0)[
        posiciones_finales
    ]
    conteos_categorias, conocidos = [], []
    for columna in prep.CATEGORICAS:
        conteos_categorias.extend(
            np.cumsum(datos_grupo[columna].eq(categoria).to_numpy())[posiciones_finales]
            for categoria in prep.reglas.DOMINIOS_CATEGORICOS[columna]
        )
        conocidos.extend(
            [np.cumsum(datos_grupo[columna].notna().to_numpy())[posiciones_finales]]
            * len(prep.reglas.DOMINIOS_CATEGORICOS[columna])
        )
    faltantes = np.stack(
        [
            np.cumsum(datos_grupo[columna].isna().to_numpy())[posiciones_finales]
            for columna in prep.PENDIENTES
        ],
        axis=1,
    )
    return [
        cantidades,
        sumas,
        cuadrados,
        np.stack(conteos_categorias, axis=1),
        np.stack(conocidos, axis=1),
        faltantes,
    ]


def estadisticas(acumulados, n):
    (
        conteos,
        suma,
        sumas_cuadrados,
        conteos_categorias,
        conocidos_categoricos,
        faltantes,
    ) = acumulados
    media = np.divide(suma, conteos, out=np.full_like(suma, np.nan), where=conteos > 0)
    varianza = np.divide(
        sumas_cuadrados - suma * media,
        conteos - 1,
        out=np.full_like(suma, np.nan),
        where=conteos > 1,
    )
    desviacion = np.sqrt(np.maximum(0, varianza))
    resumen_numerico = np.stack([media, desviacion], axis=2).reshape(len(n), -1)
    proporciones = np.divide(
        conteos_categorias,
        conocidos_categoricos,
        out=np.full(conteos_categorias.shape, np.nan),
        where=conocidos_categoricos > 0,
    )
    return np.concatenate(
        [resumen_numerico, proporciones, faltantes / np.asarray(n)[:, None]], axis=1
    )


def escalas_piloto(df, carpeta):
    """Piloto 10000, semilla independiente 2026; IQR > std > magnitud > 1."""
    rng = np.random.default_rng(2026)
    grupos = [df.index[df[prep.OBJETIVO].eq(columna)].to_numpy() for columna in CLASES]
    cuotas = asignar(PASO, [len(grupo) for grupo in grupos])
    indices = np.concatenate(
        [rng.permutation(grupo)[:cuota] for grupo, cuota in zip(grupos, cuotas)]
    )
    piloto = df.loc[indices]
    filas, mapas = [], {}
    for alcance in ["Global"] + CLASES:
        datos_grupo = (
            piloto
            if alcance == "Global"
            else piloto.loc[piloto[prep.OBJETIVO].eq(alcance)]
        )
        mapa = {}
        for columna in prep.NUMERICAS:
            valores = datos_grupo[columna].dropna()
            if len(valores) < 2:
                raise ValueError("Piloto sin soporte: " + alcance + "/" + columna)
            iqr = float(valores.quantile(0.75) - valores.quantile(0.25))
            desviacion = float(valores.std())
            if iqr > 1e-12:
                escala, metodo = iqr, "IQR"
            elif desviacion > 1e-12:
                escala, metodo = desviacion, "desviacion"
            elif abs(float(valores.median())) > 1e-12:
                escala, metodo = abs(float(valores.median())), "magnitud_mediana"
            else:
                escala, metodo = 1.0, "uno_variable_constante_en_piloto"
            mapa[columna] = escala
            filas.append(
                {
                    "alcance": alcance,
                    "variable": columna,
                    "observados": len(valores),
                    "escala": escala,
                    "metodo": metodo,
                }
            )
        mapas[alcance] = mapa
    pd.DataFrame(filas).to_csv(carpeta / "escalas_piloto.csv", index=False)
    piloto[[prep.COLUMNA_ID, prep.OBJETIVO]].to_csv(
        carpeta / "ids_piloto.csv", index=False
    )
    return mapas


def detectar_minimo(cambios, tamanos, consecutivos=CONSECUTIVOS, confirmar=False):
    """Fin de tres cambios estables; confirmar exige estabilidad posterior."""
    bien = np.asarray(cambios) < TOLERANCIA
    for indice in range(consecutivos, len(bien)):
        if bien[indice - consecutivos + 1 : indice + 1].all() and (
            not confirmar or bien[indice + 1 :].all()
        ):
            return int(tamanos[indice])
    return None


def generar_sensibilidad(curvas, carpeta):
    """Contraste descriptivo; no cambia el protocolo ni aprueba particiones."""
    filas = []
    for nombre, mascara, confirmar in [
        ("con_desviaciones_diagnostico", np.ones(len(curvas), dtype=bool), False),
        ("confirmacion_estricta", np.ones(len(curvas), dtype=bool), True),
        (
            "solo_medias_proporciones_ausencias",
            curvas.estadistico.ne("desviacion"),
            False,
        ),
    ]:
        registros = []
        for (semilla, version, alcance), datos_grupo in curvas.loc[mascara].groupby(
            ["semilla", "version", "alcance"]
        ):
            pasos = (
                datos_grupo.groupby(["n_total", "n_alcance"])
                .cambio_normalizado.max()
                .reset_index()
                .sort_values("n_total")
            )
            n_estable = detectar_minimo(
                pasos.cambio_normalizado.to_numpy(),
                pasos.n_total.to_numpy(),
                confirmar=confirmar,
            )
            soporte = int(pasos.loc[pasos.n_total.eq(n_estable), "n_alcance"].iloc[0])
            registros.append((alcance, n_estable, soporte))
        tabla = pd.DataFrame(registros, columns=["alcance", "n_total", "n_alcance"])
        fila = {
            "criterio": nombre,
            "n_min_operativo": int(tabla.n_total.max()),
            "n_min_agregado_global": int(
                tabla.loc[tabla.alcance.eq("Global"), "n_total"].max()
            ),
            "usado_para_dimensionar": nombre == "solo_medias_proporciones_ausencias",
        }
        fila.update(
            {
                clase: int(tabla.loc[tabla.alcance.eq(clase), "n_alcance"].max())
                for clase in CLASES
            }
        )
        filas.append(fila)
    pd.DataFrame(filas).to_csv(Path(carpeta) / "sensibilidad_criterio.csv", index=False)


def analizar_alcance(
    semilla,
    version,
    alcance,
    acumulados,
    soportes,
    tamanos,
    escalas,
    descriptores_estadisticos,
):
    """Calcula estadisticas, cambios y estabilidad para un unico alcance."""
    curvas, cambios = [], []
    valores = estadisticas(acumulados, soportes)
    escala = np.array(
        [
            (
                escalas[alcance][variable]
                if estadistico in ["media", "desviacion"]
                else 1.0
            )
            for variable, estadistico, _ in descriptores_estadisticos
        ]
    )
    delta = np.full_like(valores, np.nan)
    delta[1:] = abs(np.diff(valores, axis=0)) / escala
    if not np.isfinite(valores).all():
        raise ValueError("Estadistico indefinido; revisar soportes")
    criterio = np.array(
        [estadistico != "desviacion" for _, estadistico, _ in descriptores_estadisticos]
    )
    maximos = np.r_[np.nan, delta[1:, criterio].max(axis=1)]
    n_estable = detectar_minimo(maximos, tamanos)
    n_confirmado = detectar_minimo(maximos, tamanos, confirmar=True)
    if n_estable is None:
        raise ValueError(f"No hubo estabilidad: {semilla}/{version}/{alcance}")
    posicion = int(np.where(tamanos == n_estable)[0][0])
    minimo = {
        "semilla": semilla,
        "version": version,
        "alcance": alcance,
        "n_total_estable": n_estable,
        "n_alcance_estable": int(soportes[posicion]),
        "n_total_confirmacion_estricta": n_confirmado,
        "n_alcance_confirmacion_estricta": int(
            soportes[np.where(tamanos == n_confirmado)[0][0]]
        ),
        "fallos_posteriores": int(np.sum(maximos[posicion + 1 :] >= TOLERANCIA)),
    }
    for indice_descriptor, (variable, estadistico, categoria) in enumerate(
        descriptores_estadisticos
    ):
        curvas.append(
            pd.DataFrame(
                {
                    "semilla": semilla,
                    "version": version,
                    "alcance": alcance,
                    "n_total": tamanos,
                    "n_alcance": soportes,
                    "variable": variable,
                    "estadistico": estadistico,
                    "categoria": categoria,
                    "valor": valores[:, indice_descriptor],
                    "cambio_normalizado": delta[:, indice_descriptor],
                }
            )
        )
    for indice, tamano in enumerate(tamanos):
        cambios.append(
            {
                "semilla": semilla,
                "version": version,
                "alcance": alcance,
                "n_total": int(tamano),
                "n_alcance": int(soportes[indice]),
                "max_cambio_criterio_clase": float(maximos[indice]),
                "max_cambio_con_desviaciones": (
                    float(np.nanmax(delta[indice])) if indice else float("nan")
                ),
                "min_observados_numericos": int(acumulados[0][indice].min()),
                "min_observados_categoricos": int(acumulados[4][indice].min()),
            }
        )
    return minimo, curvas, cambios


def analizar_version(
    semilla,
    version,
    tabla,
    permutaciones,
    cuotas,
    tamanos,
    escalas,
    descriptores_estadisticos,
):

    acumulados = [
        resumen_prefijos(tabla.loc[permutacion], cuotas[:, indice])
        for indice, permutacion in enumerate(permutaciones)
    ]
    total = [sum(acumulado[indice] for acumulado in acumulados) for indice in range(6)]
    alcances = [("Global", total, tamanos)] + [
        (clase, acumulado, cuotas[:, indice])
        for indice, (clase, acumulado) in enumerate(zip(CLASES, acumulados))
    ]
    curvas, cambios, minimos = [], [], []
    for alcance, acumulado, soportes in alcances:
        minimo, curvas_alcance, cambios_alcance = analizar_alcance(
            semilla,
            version,
            alcance,
            acumulado,
            soportes,
            tamanos,
            escalas,
            descriptores_estadisticos,
        )
        minimos.append(minimo)
        curvas.extend(curvas_alcance)
        cambios.extend(cambios_alcance)
    return curvas, cambios, minimos


def construir_minimos(minimos, grupos):
    capacidades = [len(grupo) for grupo in grupos]
    # Maximo comun; cuotas por clase en ese tamano como condicion conservadora.
    n_min = int(minimos.n_total_estable.max())
    minimos_por_clase = {
        clase: int(minimos.loc[minimos.alcance.eq(clase), "n_alcance_estable"].max())
        for clase in CLASES
    }
    minimos_medidos = dict(minimos_por_clase)
    minimos_por_clase = dict(zip(CLASES, map(int, asignar(n_min, capacidades))))
    resultado = {
        "n_min": n_min,
        "n_min_por_clase_medido": minimos_medidos,
        "n_min_global_sin_condicionar_clase": int(
            minimos.loc[minimos.alcance.eq("Global"), "n_total_estable"].max()
        ),
        "n_min_por_clase": minimos_por_clase,
        "n_min_confirmacion_estricta_del_criterio_vigente": int(
            minimos.n_total_confirmacion_estricta.max()
        ),
        "n_min_por_clase_confirmacion_estricta_del_criterio_vigente": {
            clase: int(
                minimos.loc[
                    minimos.alcance.eq(clase), "n_alcance_confirmacion_estricta"
                ].max()
            )
            for clase in CLASES
        },
        "definicion": (
            "n_min es el mayor tamano total al completar tres cambios estables en los"
            " seis alcances, tres semillas y dos versiones; n_min,c adopta las cuotas"
            " en el tamano comun; se exportan tambien los puntos individuales medidos"
        ),
        "control_adicional": (
            "La confirmacion estricta sin fluctuaciones posteriores se informa como"
            " diagnostico complementario al criterio de tres incrementos consecutivos"
        ),
        "limite": (
            "Minimo empirico condicionado a estas estadisticas y tolerancias; se"
            " registran fallos posteriores. No garantiza representatividad universal ni"
            " desempeno de un MLP"
        ),
    }
    return resultado


def progresivo(observados, corregidos, carpeta):
    """Coordina ordenes anidados y tres semillas, siempre sobre la reserva train."""
    tamanos = np.unique(
        np.r_[np.arange(PASO, len(observados) + 1, PASO), len(observados)]
    )
    grupos = [
        observados.index[observados[prep.OBJETIVO].eq(clase)].to_numpy()
        for clase in CLASES
    ]
    cuotas = np.stack(
        [asignar(int(n), [len(grupo) for grupo in grupos]) for n in tamanos]
    )
    assert (np.diff(cuotas, axis=0) >= 0).all(), "Las cuotas deben ser anidadas"
    escalas = escalas_piloto(observados, carpeta)
    descriptores_estadisticos = descriptores()
    curvas, cambios, minimos, ordenes = [], [], [], []
    for semilla in SEMILLAS:
        azar = np.random.default_rng(semilla)
        permutaciones = [azar.permutation(grupo) for grupo in grupos]
        for clase, permutacion in zip(CLASES, permutaciones):
            ordenes.append(
                pd.DataFrame(
                    {
                        "semilla": semilla,
                        "clase_cliente": clase,
                        "id_cliente": (
                            observados.loc[permutacion, prep.COLUMNA_ID].to_numpy()
                        ),
                        "orden_clase": np.arange(len(permutacion)),
                    }
                )
            )
        for version, tabla in [("observados", observados), ("corregidos", corregidos)]:
            curvas_version, cambios_version, minimos_version = analizar_version(
                semilla,
                version,
                tabla,
                permutaciones,
                cuotas,
                tamanos,
                escalas,
                descriptores_estadisticos,
            )
            curvas.extend(curvas_version)
            cambios.extend(cambios_version)
            minimos.extend(minimos_version)
            print(f"Curvas listas: semilla {semilla}, {version}", flush=True)
            gc.collect()
    pd.concat(ordenes, ignore_index=True).to_csv(
        carpeta / "ordenes_incrementales.csv.gz", index=False, compression="gzip"
    )
    tabla_curvas = pd.concat(curvas, ignore_index=True)
    tabla_curvas.to_csv(carpeta / "curvas_estadisticas.csv", index=False)
    generar_sensibilidad(tabla_curvas, carpeta)
    pd.DataFrame(cambios).to_csv(carpeta / "cambios_por_paso.csv", index=False)
    minimos = pd.DataFrame(minimos)
    minimos.to_csv(carpeta / "estabilizacion_por_semilla.csv", index=False)
    resultado = construir_minimos(minimos, grupos)
    (carpeta / "minimos.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultado


def dimensionar(minimos, capacidades):
    """Conserva las reservas completas; verifica k=5 y parada interna .25."""
    from crossVal import evaluar_candidato

    tamanos = {grupo: int(sum(columna)) for grupo, columna in capacidades.items()}
    cuotas = {
        grupo: np.asarray(columna).tolist() for grupo, columna in capacidades.items()
    }
    externo = all(
        cumple(tamanos[grupo], capacidades[grupo], minimos, MARGEN)
        for grupo in capacidades
    )
    etiquetas = np.concatenate(
        [
            np.repeat(columna, capacidades["entrenamiento"][indice])
            for indice, columna in enumerate(CLASES)
        ]
    )
    resumen_cv, _ = evaluar_candidato(etiquetas, 5, 0.25, minimos, margen=MARGEN)
    return {
        "estado": "factible" if externo and resumen_cv["cumple"] else "no_factible",
        "n_total": sum(tamanos.values()),
        "tamanos": tamanos,
        "cuotas": cuotas,
        "k_planificacion": 5,
        "validation_fraction": 0.25,
        "margen_dimensionamiento": MARGEN,
        "motivo": (
            "Se conservan las reservas completas; se comprueban todas las clases y la"
            " reserva interna al 25%."
        ),
    }


def cumple(n, conteos, minimos, margen=1.0):
    return n >= np.ceil(margen * minimos["n_min"]) and all(
        conteo >= np.ceil(margen * minimos["n_min_por_clase"][columna])
        for columna, conteo in zip(CLASES, conteos)
    )


def construir_protocolo():
    protocolo = {
        "paso": PASO,
        "semillas": SEMILLAS,
        "piloto": {"n": PASO, "semilla": 2026},
        "tolerancia_estricta": TOLERANCIA,
        "incrementos_consecutivos": CONSECUTIVOS,
        "criterio_principal": (
            "Tres incrementos consecutivos con variacion menor a la tolerancia;"
            " incluye extremo final"
        ),
        "control_adicional": (
            "Se informa como diagnostico complementario la confirmacion estricta"
            " sin fluctuaciones posteriores"
        ),
        "estadisticas_decision": [
            "medias",
            "proporciones categoricas entre conocidos",
            "tasas de ausencia",
        ],
        "estadisticas_complementarias": ["desviaciones muestrales"],
        "versiones": [
            "observados antes de imputar",
            "corregidos con el ajuste inicial de entrenamiento",
        ],
        "dimensionamiento": {
            "reparto": [0.70, 0.15, 0.15],
            "margen": MARGEN,
            "k_candidato": 5,
            "fracciones_internas": FRACCIONES,
            "seleccion": "Conservar reservas completas; comprobar fraccion interna .25",
        },
        "advertencia": (
            "Umbrales operativos definidos para evaluar la convergencia empirica"
            " del muestreo incremental estratificado"
        ),
    }
    return protocolo


def seleccionar_particiones(df, reservas, bitacora, decision):
    """Selecciona sin reemplazo y recupera solo las marcas de esas filas."""
    azar = np.random.default_rng(101)
    seleccion = {}
    asignaciones = []
    nombres = {"entrenamiento": "train", "validacion": "validation", "prueba": "test"}
    for grupo in nombres:
        indices = []
        for clase, cuota in zip(CLASES, decision["cuotas"][grupo]):
            candidatos = reservas.index[
                reservas.grupo.eq(grupo) & reservas[prep.OBJETIVO].eq(clase)
            ].to_numpy()
            indices.extend(azar.permutation(candidatos)[:cuota])
        indices = np.sort(indices)
        particion = df.loc[indices].copy().reset_index(drop=True)
        marcas = bitacora.loc[
            bitacora[prep.COLUMNA_ID].isin(particion[prep.COLUMNA_ID])
        ]
        seleccion[grupo] = restaurar_marcas(particion, marcas)
        asignaciones.append(
            particion[[prep.COLUMNA_ID, prep.OBJETIVO]].assign(
                particion=nombres[grupo], reserva_original=grupo
            )
        )
    return seleccion, asignaciones


def exportar_particiones(seleccion, asignaciones, minimos, salida):
    """Reaprende la imputacion solo con train y exporta los tres CSV completos."""
    nombres = {"entrenamiento": "train", "validacion": "validation", "prueba": "test"}
    # Reajuste obligatorio en el train efectivo; no usar las medianas de 700001.
    estado = prep.ajustar_imputacion(seleccion["entrenamiento"][prep.PREDICTORES])
    estado["huella_ids_entrenamiento"] = prep.huella_ids(
        seleccion["entrenamiento"][prep.COLUMNA_ID]
    )
    bitacoras = []
    tablas = []
    for grupo, particion in seleccion.items():
        predictores_imputados, bitacora = prep.aplicar_imputacion(
            particion[prep.PREDICTORES], estado
        )
        bitacora.insert(
            0,
            prep.COLUMNA_ID,
            particion.loc[bitacora.indice_fila, prep.COLUMNA_ID].to_numpy(),
        )
        bitacora["particion"] = nombres[grupo]
        bitacoras.append(bitacora.drop(columns="indice_fila"))
        particion[prep.PREDICTORES] = predictores_imputados
        prep.formatear_enteras(particion)
        assert not particion.isna().any().any()
        assert not any(prep.reglas.comprobar_dependencias(particion).values())
        particion.to_csv(salida / (nombres[grupo] + ".csv"), index=False)
        conteos = particion[prep.OBJETIVO].value_counts().reindex(CLASES)
        for clase, filas_clase in conteos.items():
            tablas.append(
                {
                    "particion": nombres[grupo],
                    "clase_cliente": clase,
                    "filas": int(filas_clase),
                    "minimo_clase": minimos["n_min_por_clase"][clase],
                    "razon_sobre_minimo_clase": (
                        filas_clase / minimos["n_min_por_clase"][clase]
                    ),
                    "cumple_clase": bool(
                        filas_clase >= minimos["n_min_por_clase"][clase]
                    ),
                    "cumple_clase_con_margen": bool(
                        filas_clase >= np.ceil(
                            MARGEN * minimos["n_min_por_clase"][clase]
                        )
                    ),
                }
            )
    pd.concat(bitacoras, ignore_index=True).to_csv(
        salida / "bitacora_imputacion_particiones.csv", index=False
    )
    asignacion = pd.concat(asignaciones, ignore_index=True)
    assert asignacion[prep.COLUMNA_ID].is_unique
    asignacion.to_csv(salida / "indices_particiones.csv", index=False)
    pd.DataFrame(tablas).to_csv(salida / "conteos_particiones.csv", index=False)
    (salida / "estadisticas_train_efectivo.json").write_text(
        json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return asignacion, bitacoras


def construir_reporte(decision, minimos, entrada, df, asignacion, bitacoras):
    resumen = {
        "estado": decision["estado"],
        "particiones_aprobadas": decision["estado"] == "factible",
        "minimos": minimos,
        "decision": decision,
        "semilla_seleccion": 101,
        "ids_disjuntos": True,
        "reservas_respetadas": True,
        "imputacion_reajustada_solo_train": True,
        "csv": (
            "Particiones completas, sin codificar ni escalar. Restaurar las marcas de"
            " su bitacora y reaprender la imputacion en cada fold."
        ),
        "entradas_sha256": {
            n: prep.huella_archivo(entrada / n)
            for n in [
                "clientes_corregidos_1000000.csv",
                "reservas_identificadores.csv",
                "bitacora_imputacion.csv",
            ]
        },
        "filas_no_seleccionadas": len(df) - len(asignacion),
        "imputaciones_seleccionadas": sum(len(x) for x in bitacoras),
    }
    return resumen


def ejecutar(entrada, salida):
    """Coordina curvas, dimensionamiento y exportacion de la Pregunta 2."""
    entrada, salida = Path(entrada), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    protocolo = construir_protocolo()
    (salida / "protocolo.json").write_text(
        json.dumps(protocolo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df = pd.read_csv(
        entrada / "clientes_corregidos_1000000.csv", dtype={prep.COLUMNA_ID: str}
    )
    reservas = pd.read_csv(
        entrada / "reservas_identificadores.csv", dtype={prep.COLUMNA_ID: str}
    )
    pd.testing.assert_frame_equal(
        df[[prep.COLUMNA_ID, prep.OBJETIVO]], reservas[[prep.COLUMNA_ID, prep.OBJETIVO]]
    )
    bitacora = pd.read_csv(
        entrada / "bitacora_imputacion.csv", dtype={prep.COLUMNA_ID: str}
    )
    train_mask = reservas.grupo.eq("entrenamiento")
    train_corregido = df.loc[train_mask].copy()
    bitacora_train = bitacora.loc[
        bitacora[prep.COLUMNA_ID].isin(train_corregido[prep.COLUMNA_ID])
    ]
    observados = restaurar_marcas(train_corregido, bitacora_train)
    minimos = progresivo(observados, train_corregido, salida)
    del observados, train_corregido
    gc.collect()
    capacidades = {
        grupo: (
            reservas.loc[reservas.grupo.eq(grupo), prep.OBJETIVO]
            .value_counts()
            .reindex(CLASES)
            .to_numpy()
        )
        for grupo in ["entrenamiento", "validacion", "prueba"]
    }
    decision = dimensionar(minimos, capacidades)
    (salida / "dimensionamiento.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    seleccion, asignaciones = seleccionar_particiones(df, reservas, bitacora, decision)
    asignacion, bitacoras = exportar_particiones(
        seleccion, asignaciones, minimos, salida
    )
    resumen = construir_reporte(decision, minimos, entrada, df, asignacion, bitacoras)
    (salida / "diagnostico_muestreo.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(resumen, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("entrada")
    a.add_argument("salida")
    args = a.parse_args()
    ejecutar(args.entrada, args.salida)
