"""Pregunta 1: reglas, reservas, analisis, imputacion y constructor de transformacion.

Uso con la salida del Bloque 2:
    python3 preprocesamiento.py clientes_reglas_dominio.csv resultados

Uso desde el CSV original (incluye los Bloques 1 y 2):
    python3 preprocesamiento.py original.csv resultados --desde-original

Las funciones ajustar_imputacion y aplicar_imputacion estan separadas para
volver a ajustarlas en cada entrenamiento futuro. No se entrena una red aqui.
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import reglas_dominio as reglas
from resumir_correlaciones import resumir
from transformaciones import construir_transformador as _construir_transformador
from transformaciones import seleccionar_predictores as _seleccionar_predictores


SEMILLA = 42
COLUMNA_ID = reglas.COLUMNA_ID
OBJETIVO = reglas.OBJETIVO
NUMERICAS = reglas.NUMERICAS
CATEGORICAS = reglas.CATEGORICAS
PREDICTORES = NUMERICAS + CATEGORICAS
PENDIENTES = ["edad", "paginas_por_sesion", "duracion_sesion_min",
              "satisfaccion_promedio", "region", "metodo_pago_preferido"]
NUMERICAS_PENDIENTES = PENDIENTES[:4]
POR_GRUPOS = {"paginas_por_sesion": "sesiones_365d",
              "duracion_sesion_min": "sesiones_365d",
              "satisfaccion_promedio": "compras_365d"}
MIN_DONANTES_GRUPO = 100  # Soporte minimo de donantes observados por estrato.
ENTERAS = ["edad", "antiguedad_dias", "compras_365d", "dias_ultima_compra",
           "unidades_365d", "categorias_compradas_365d", "sesiones_365d",
           "carritos_abandonados_365d", "compras_con_descuento_365d",
           "emails_enviados_365d", "clics_anuncios_365d", "puntos_lealtad",
           "miembro_lealtad", "devoluciones_365d", "reclamos_365d", "emails_abiertos_365d"]


def construir_transformador(escalador="robust", incluir_gasto=True):
    """Construye sin ajustar; usa el diccionario unico de reglas_dominio.

    Cada entrenamiento o fold necesita su propio objeto. Aplicar antes la
    imputacion aprendida solamente con las filas de ese entrenamiento.
    """
    return _construir_transformador(NUMERICAS, CATEGORICAS,
        reglas.DOMINIOS_CATEGORICOS, escalador, incluir_gasto)


def seleccionar_predictores(df, incluir_gasto=True):
    """Selecciona las entradas ya imputadas, excluyendo ID y clase."""
    return _seleccionar_predictores(df, NUMERICAS, CATEGORICAS, incluir_gasto)


def huella_archivo(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(bloque)
    return sha.hexdigest()


def huella_ids(ids):
    texto = "\n".join(sorted(ids.astype(str)))
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def reservar_grupos(df):
    """Baraja cada clase sin reemplazo; redondea 70%/15% y reserva el resto.

    Son grupos provisionales, no una muestra minima ya justificada. Solo se
    usan ID/etiqueta para organizar la reserva, nunca para calcular imputaciones.
    """
    grupo = pd.Series("", index=df.index, name="grupo")
    azar = np.random.default_rng(SEMILLA)
    for etiqueta in reglas.CLASES:
        indices = azar.permutation(df.index[df[OBJETIVO].eq(etiqueta)])
        n_train = round(0.70 * len(indices))
        n_val = round(0.15 * len(indices))
        grupo.loc[indices[:n_train]] = "entrenamiento"
        grupo.loc[indices[n_train:n_train+n_val]] = "validacion"
        grupo.loc[indices[n_train+n_val:]] = "prueba"
    if grupo.eq("").any():
        raise ValueError("Hay filas sin reserva; revisar las clases.")
    reservas = df[[COLUMNA_ID, OBJETIVO]].copy()
    reservas["grupo"] = grupo
    return reservas


def ajustar_imputacion(x_train, min_donantes=MIN_DONANTES_GRUPO):
    """Aprende estadisticas exclusivamente de los predictores de entrenamiento.

    La eleccion de columnas y agrupadores se justifica en el analisis adjunto.
    El umbral controla grupos escasos; no demuestra optimalidad predictiva.
    """
    if COLUMNA_ID in x_train or OBJETIVO in x_train:
        raise ValueError("El ajuste no debe recibir identificador ni etiqueta.")
    estado = {"min_donantes_grupo": min_donantes, "filas_ajuste": len(x_train),
              "medianas": {}, "valores_respaldo": {}, "modas": {}, "por_grupos": {}}
    for columna in NUMERICAS_PENDIENTES:
        mediana = x_train[columna].median()
        if pd.isna(mediana):
            raise ValueError("No hay donantes para " + columna)
        estado["medianas"][columna] = float(mediana)
        estado["valores_respaldo"][columna] = float(np.round(mediana, 0 if columna == "edad" else 2))
    for columna in ["region", "metodo_pago_preferido"]:
        modas = x_train[columna].mode()
        if modas.empty:
            raise ValueError("No hay donantes para " + columna)
        # En un empate se usa el orden alfabetico que devuelve Pandas.
        estado["modas"][columna] = str(modas.iloc[0])
    for columna, agrupador in POR_GRUPOS.items():
        tabla = x_train.groupby(agrupador)[columna].agg(["count", "median"])
        tabla = tabla.loc[tabla["count"].ge(min_donantes)]
        registros = [{"grupo": int(g), "donantes": int(r["count"]),
                      "mediana": float(r["median"]),
                      "valor_imputacion": float(np.round(r["median"], 2))} for g, r in tabla.iterrows()]
        estado["por_grupos"][columna] = {"agrupador": agrupador, "valores": registros}
    return estado


def aplicar_imputacion(x, estado):
    """Usa estadisticas ya aprendidas. No calcula medianas o modas nuevas."""
    if COLUMNA_ID in x or OBJETIVO in x:
        raise ValueError("La transformacion recibe solo predictores.")
    salida = x.copy()
    registros = []
    for columna in PENDIENTES:
        faltantes = x[columna].isna()
        indices = x.index[faltantes]
        if len(indices) == 0:
            continue
        if columna in estado["modas"]:
            valores = pd.Series(estado["modas"][columna], index=indices)
            metodos = pd.Series("moda_entrenamiento", index=indices)
            grupos = pd.Series("", index=indices)
        else:
            base = estado["valores_respaldo"][columna]
            valores = pd.Series(base, index=indices, dtype=float)
            metodos = pd.Series("mediana_entrenamiento", index=indices)
            grupos = pd.Series("", index=indices)
            if columna in estado["por_grupos"]:
                regla = estado["por_grupos"][columna]
                mapa = {r["grupo"]: r["valor_imputacion"] for r in regla["valores"]}
                claves = x.loc[indices, regla["agrupador"]]
                locales = claves.map(mapa)
                tienen_grupo = locales.notna()
                valores.loc[tienen_grupo] = locales.loc[tienen_grupo]
                metodos.loc[tienen_grupo] = "mediana_por_" + regla["agrupador"]
                metodos.loc[~tienen_grupo] = "mediana_entrenamiento_respaldo"
                grupos = claves.astype("string")
            # Solo se redondean las estimaciones nuevas, no las observaciones.
            valores = valores.round(0 if columna == "edad" else 2)
            minimo, maximo = reglas.RANGOS_VALIDOS[columna]
            if not valores.between(minimo, maximo).all():
                raise ValueError("Imputacion fuera de rango en " + columna)
        salida.loc[indices, columna] = valores
        registros.append(pd.DataFrame({"indice_fila": indices, "columna": columna,
                                      "valor_imputado": valores.to_numpy(),
                                      "metodo": metodos.to_numpy(), "grupo_donantes": grupos.to_numpy()}))
    columnas = ["indice_fila", "columna", "valor_imputado", "metodo", "grupo_donantes"]
    bitacora = pd.concat(registros, ignore_index=True) if registros else pd.DataFrame(columns=columnas)
    return salida, bitacora


def formatear_enteras(df):
    """Recupera el tipo entero solo despues de comprobar integridad y ausencias."""
    for columna in ENTERAS:
        if df[columna].isna().any() or not df[columna].eq(df[columna].round()).all():
            raise ValueError("La columna no admite conversion a entero: " + columna)
        df[columna] = df[columna].astype("int64")


def analizar_entrenamiento(x_train, carpeta):
    """Descriptivos y asociaciones; nunca recibe validacion, prueba ni etiqueta."""
    carpeta.mkdir(parents=True, exist_ok=True)
    descriptivos = x_train[NUMERICAS].describe(percentiles=[.25, .5, .75]).T
    descriptivos["asimetria"] = x_train[NUMERICAS].skew()
    descriptivos.to_csv(carpeta / "descriptivos_numericos.csv")
    for metodo in ["pearson", "spearman"]:
        x_train[NUMERICAS].corr(method=metodo).to_csv(carpeta / ("correlaciones_" + metodo + ".csv"))
    tasas = pd.DataFrame({"faltantes": x_train[PENDIENTES].isna().sum()})
    tasas["tasa"] = tasas["faltantes"] / len(x_train)
    tasas.to_csv(carpeta / "tasas_ausencia.csv")
    ausencias_num, ausencias_cat, grupos_numericos, modas_grupos = [], [], [], []
    for columna in PENDIENTES:
        ausente = x_train[columna].isna()
        for predictor in NUMERICAS:
            if predictor == columna:
                continue
            valores = x_train[predictor]
            disponibles = valores.notna()
            correlacion = valores.loc[disponibles].corr(ausente.loc[disponibles].astype(int))
            media_si = valores.loc[ausente].mean()
            media_no = valores.loc[~ausente].mean()
            escala = valores.std()
            ausencias_num.append({"faltante": columna, "predictor": predictor,
                "n_con_faltante": int((ausente & disponibles).sum()),
                "n_sin_faltante": int((~ausente & disponibles).sum()),
                "correlacion_indicador": correlacion,
                "media_con_faltante": media_si, "media_sin_faltante": media_no,
                "diferencia_en_desviaciones": (media_si-media_no)/escala if escala else 0})
        for predictor in CATEGORICAS:
            if predictor == columna:
                continue
            si = x_train.loc[ausente, predictor].value_counts(normalize=True)
            no = x_train.loc[~ausente, predictor].value_counts(normalize=True)
            for categoria in sorted(set(si.index) | set(no.index)):
                ausencias_cat.append({"faltante": columna, "predictor": predictor,
                    "categoria": categoria, "proporcion_con_faltante": float(si.get(categoria, 0)),
                    "proporcion_sin_faltante": float(no.get(categoria, 0))})
    for columna, agrupador in POR_GRUPOS.items():
        tabla = x_train.groupby(agrupador)[columna].agg(["count", "mean", "median", "std"])
        for g, r in tabla.iterrows():
            grupos_numericos.append({"columna": columna, "agrupador": agrupador, "grupo": int(g),
                "donantes": int(r["count"]), "media": r["mean"], "mediana": r["median"],
                "desviacion": r["std"], "admitido": bool(r["count"] >= MIN_DONANTES_GRUPO)})
    for columna in ["region", "metodo_pago_preferido"]:
        for agrupador in [c for c in CATEGORICAS if c != columna] + ["miembro_lealtad"]:
            for g, sub in x_train.groupby(agrupador):
                frecuencias = sub[columna].value_counts()
                if not frecuencias.empty:
                    modas_grupos.append({"columna": columna, "agrupador": agrupador, "grupo": g,
                        "donantes": int(frecuencias.sum()), "moda": frecuencias.index[0],
                        "proporcion_moda": float(frecuencias.iloc[0]/frecuencias.sum())})
    pd.DataFrame(ausencias_num).to_csv(carpeta / "ausencias_vs_numericas.csv", index=False)
    pd.DataFrame(ausencias_cat).to_csv(carpeta / "ausencias_vs_categoricas.csv", index=False)
    pd.DataFrame(grupos_numericos).to_csv(carpeta / "medianas_por_grupo.csv", index=False)
    pd.DataFrame(modas_grupos).to_csv(carpeta / "modas_condicionales.csv", index=False)
    frecuencias = pd.concat([x_train[c].value_counts().rename_axis("categoria").reset_index(name="donantes").assign(columna=c)
                            for c in ["region", "metodo_pago_preferido"]], ignore_index=True)
    frecuencias.to_csv(carpeta / "frecuencias_categoricas.csv", index=False)
    max_corr = pd.DataFrame(ausencias_num)["correlacion_indicador"].abs().max()
    max_diff = pd.DataFrame(ausencias_num)["diferencia_en_desviaciones"].abs().max()
    return {"max_correlacion_indicador": float(max_corr), "max_diferencia_medias_en_desviaciones": float(max_diff),
            "interpretacion": "Estadisticas descriptivas de asociacion; no identifican ni prueban MCAR, MAR o MNAR. No se ejecutaron pruebas formales de mecanismo."}


def procesar(archivo_entrada, carpeta_salida):
    entrada, carpeta = Path(archivo_entrada), Path(carpeta_salida)
    if entrada.resolve() == (carpeta / "clientes_corregidos_1000000.csv").resolve():
        raise ValueError("La salida no puede reemplazar la entrada.")
    df = pd.read_csv(entrada, dtype={COLUMNA_ID: str})
    if set(df.columns) != set(PREDICTORES + [COLUMNA_ID, OBJETIVO]):
        raise ValueError("Columnas distintas del diccionario.")
    if df[COLUMNA_ID].isna().any() or not df[COLUMNA_ID].is_unique or not df[OBJETIVO].isin(reglas.CLASES).all():
        raise ValueError("Identificadores o etiquetas invalidos.")
    if df[[c for c in PREDICTORES if c not in PENDIENTES]].isna().any().any():
        raise ValueError("Hay ausencias fuera de las seis columnas previstas.")
    por_fila = df[PENDIENTES].isna().sum(axis=1)
    if por_fila.gt(1).any():
        raise ValueError("No se cumple la observacion de una sola ausencia por fila.")
    if any(m.any() for m in reglas.detectar_rangos(df).values()):
        raise ValueError("La entrada debe tener aplicadas las reglas de dominio.")
    if any(reglas.comprobar_dependencias(df).values()):
        raise ValueError("Hay dependencias incumplidas en la entrada.")
    reservas = reservar_grupos(df)
    entrenamiento = reservas["grupo"].eq("entrenamiento")
    x_train = df.loc[entrenamiento, PREDICTORES]
    carpeta.mkdir(parents=True, exist_ok=True)
    diagnostico_ausencias = analizar_entrenamiento(x_train, carpeta / "analisis_entrenamiento")
    diagnostico_correlaciones = resumir(carpeta / "analisis_entrenamiento",
                                       carpeta / "cierre_pregunta1")
    estado = ajustar_imputacion(x_train)
    estado["huella_ids_entrenamiento"] = huella_ids(df.loc[entrenamiento, COLUMNA_ID])
    x_completo, bitacora = aplicar_imputacion(df[PREDICTORES], estado)
    bitacora.insert(0, COLUMNA_ID, df.loc[bitacora["indice_fila"], COLUMNA_ID].to_numpy())
    bitacora["reserva"] = reservas.loc[bitacora["indice_fila"], "grupo"].to_numpy()
    bitacora = bitacora.drop(columns="indice_fila")
    salida = df.copy()
    salida[PREDICTORES] = x_completo[PREDICTORES]
    formatear_enteras(salida)
    if salida.isna().any().any() or not np.isfinite(salida[NUMERICAS].to_numpy()).all():
        raise ValueError("Quedaron ausencias o valores no finitos.")
    if any(m.any() for m in reglas.detectar_rangos(salida).values()) or any(reglas.comprobar_dependencias(salida).values()):
        raise ValueError("La salida no satisface el diccionario.")
    for columna in CATEGORICAS:
        if not salida[columna].isin(reglas.DOMINIOS_CATEGORICOS[columna]).all():
            raise ValueError("Categoria fuera de dominio.")
    conteos = reservas.groupby(["grupo", OBJETIVO]).size().rename("filas").reset_index()
    reporte = {"avance": "Bloque 3: imputacion estadistica; base completa sin codificar ni escalar",
        "filas": len(df), "columnas": len(df.columns), "semilla_reservas": SEMILLA,
        "reservas_provisionales": reservas["grupo"].value_counts().to_dict(),
        "imputaciones_por_columna": df[PENDIENTES].isna().sum().astype(int).to_dict(),
        "total_imputaciones": len(bitacora), "filas_con_mas_de_una_ausencia_entrada": int(por_fila.gt(1).sum()),
        "faltantes_salida": int(salida.isna().sum().sum()),
        "valores_no_finitos_salida": int((~np.isfinite(salida[NUMERICAS].to_numpy())).sum()),
        "metodos_aplicados": bitacora.groupby(["columna", "metodo"]).size().rename("celdas").reset_index().to_dict("records"),
        "medianas_entrenamiento": estado["medianas"], "modas_entrenamiento": estado["modas"],
        "donantes_solo_entrenamiento": True, "id_y_etiqueta_fuera_del_ajuste": True,
        "analisis_ausencias": diagnostico_ausencias,
        "tipos_enteros_salida": {c: str(salida[c].dtype) for c in ENTERAS},
        "versiones": {"pandas": pd.__version__, "numpy": np.__version__},
        "entrada_sha256": huella_archivo(entrada),
        "huella_ids_entrenamiento": estado["huella_ids_entrenamiento"],
        "pendiente": "Muestreo incremental, tamanos minimos, particiones finales, k y modelos. Constructor de codificacion/escalado disponible: ajustar junto con la imputacion en cada entrenamiento efectivo y fold.",
        "limite": "Valores estimados, no recuperacion del ground truth. Grupos y minimo de 100 donantes son decisiones iniciales justificadas por descriptivos, no una comparacion de precision de imputadores."}
    salida.to_csv(carpeta / "clientes_corregidos_1000000.csv", index=False)
    reservas.to_csv(carpeta / "reservas_identificadores.csv", index=False)
    conteos.to_csv(carpeta / "conteos_reservas.csv", index=False)
    bitacora.to_csv(carpeta / "bitacora_imputacion.csv", index=False)
    bitacora.groupby(["columna", "metodo"], sort=False).head(2).to_csv(carpeta / "ejemplos_imputacion.csv", index=False)
    (carpeta / "estadisticas_imputacion.json").write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding="utf-8")
    (carpeta / "diagnostico_bloque3.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    cierre = {"estado": "Preprocesamiento y resumen de correlaciones ejecutados; constructor disponible sin ajuste global",
        "correlaciones": diagnostico_correlaciones,
        "predictores_base": len(PREDICTORES),
        "entradas_transformadas_esperadas": len(NUMERICAS) + sum(len(reglas.DOMINIOS_CATEGORICOS[c]) for c in CATEGORICAS),
        "constructor": "preprocesamiento.construir_transformador",
        "csv": "Base completa en unidades originales; no se sustituye por la matriz del modelo",
        "limite": "No entrena modelos ni determina tamanos minimos. El constructor requiere fit solo en entrenamiento."}
    (carpeta / "cierre_pregunta1" / "diagnostico_pregunta1.json").write_text(
        json.dumps(cierre, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))
    return reporte


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4] or (len(sys.argv) == 4 and sys.argv[3] != "--desde-original"):
        raise SystemExit("Uso: python3 preprocesamiento.py entrada.csv resultados [--desde-original]")
    if len(sys.argv) == 4:
        with tempfile.TemporaryDirectory(prefix="bloque2_") as temporal:
            reglas.procesar(sys.argv[1], temporal)
            procesar(Path(temporal) / reglas.ARCHIVO_SALIDA, sys.argv[2])
            for nombre in ["diagnostico_bloque2.json", "bitacora_bloque2.csv"]:
                shutil.copyfile(Path(temporal) / nombre, Path(sys.argv[2]) / nombre)
    else:
        procesar(sys.argv[1], sys.argv[2])
