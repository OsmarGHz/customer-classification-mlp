"""Pregunta 3: compara k por tamanos y costo; no compara puntuaciones de redes.

python crossVal.py resultados resultados/crossval
Cada CSV de indices usa la posicion (base cero) de train.csv y el ID del cliente.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import preprocesamiento as prep

CLASES = prep.reglas.CLASES


def resumir_candidato(detalle, k, fraccion, n_train):
    resumen = {
        "k": k,
        "validation_fraction": fraccion,
        "cumple": bool(detalle.cumple_margen.all()),
        "cumple_sin_margen": bool(detalle.cumple_minimo.all()),
        "menor_razon_sobre_minimo": float(detalle.razon_sobre_minimo.min()),
        "min_evaluacion_total": int(
            detalle.loc[
                detalle.parte.eq("evaluacion") & detalle.alcance.eq("Global"),
                "filas_o_cota",
            ].min()
        ),
        "min_evaluacion_alto_valor": int(
            detalle.loc[
                detalle.parte.eq("evaluacion") & detalle.alcance.eq("Alto_valor"),
                "filas_o_cota",
            ].min()
        ),
        "min_parada_total": int(
            detalle.loc[
                detalle.parte.eq("parada_interna_cota") & detalle.alcance.eq("Global"),
                "filas_o_cota",
            ].min()
        ),
        "min_parada_alto_valor_cota": int(
            detalle.loc[
                detalle.parte.eq("parada_interna_cota")
                & detalle.alcance.eq("Alto_valor"),
                "filas_o_cota",
            ].min()
        ),
        "ajustes_por_configuracion": k,
        "costo_filas_relativo_k5": (k - 1) / 4,
        "filas_entrenamiento_acumuladas_por_configuracion": int((k - 1) * n_train),
    }
    return resumen


def evaluar_candidato(etiquetas, k, fraccion, minimos, margen=1.0):
    """Evalua todos los folds con conteos reales y cotas internas conservadoras.

    MLP inicializa pesos antes de su split interno: no se atribuyen al estimador
    los indices de un train_test_split aislado. Las cotas floor son validas para
    cualquier asignacion estratificada con esos tamanos y esa version de sklearn.
    """
    etiquetas = np.asarray(etiquetas)
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    filas = []
    for fold, (indices_entrenamiento, indices_evaluacion) in enumerate(
        cv.split(np.zeros(len(etiquetas)), etiquetas), 1
    ):
        filas_entrenamiento = len(indices_entrenamiento)
        filas_parada = int(np.ceil(fraccion * filas_entrenamiento))
        filas_pesos = filas_entrenamiento - filas_parada
        conteos_entrenamiento = (
            pd.Series(etiquetas[indices_entrenamiento])
            .value_counts()
            .reindex(CLASES, fill_value=0)
            .to_numpy()
        )
        conteos_evaluacion = (
            pd.Series(etiquetas[indices_evaluacion])
            .value_counts()
            .reindex(CLASES, fill_value=0)
            .to_numpy()
        )
        for parte, filas_parte, cuentas in [
            ("evaluacion", len(indices_evaluacion), conteos_evaluacion),
            ("entrenamiento_fold", filas_entrenamiento, conteos_entrenamiento),
            (
                "parada_interna_cota",
                filas_parada,
                np.floor(
                    conteos_entrenamiento * filas_parada / filas_entrenamiento
                ).astype(int),
            ),
            (
                "ajuste_pesos_cota",
                filas_pesos,
                np.floor(
                    conteos_entrenamiento * filas_pesos / filas_entrenamiento
                ).astype(int),
            ),
        ]:
            for alcance, observado, minimo in [
                ("Global", filas_parte, minimos["n_min"])
            ] + [
                (clase, int(conteo), minimos["n_min_por_clase"][clase])
                for clase, conteo in zip(CLASES, cuentas)
            ]:
                filas.append(
                    {
                        "k": k,
                        "validation_fraction": fraccion,
                        "fold": fold,
                        "parte": parte,
                        "alcance": alcance,
                        "filas_o_cota": observado,
                        "minimo": minimo,
                        "minimo_con_margen": int(np.ceil(margen * minimo)),
                        "razon_sobre_minimo": observado / minimo,
                        "cumple_minimo": observado >= minimo,
                        "cumple_margen": observado >= np.ceil(margen * minimo),
                    }
                )
    detalle = pd.DataFrame(filas)
    return resumir_candidato(detalle, k, fraccion, len(etiquetas)), detalle


def exportar_indices(train, etiquetas, k, salida):
    """Cada fila se usa exactamente una vez como evaluacion externa."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    asignacion = np.zeros(len(train), dtype=int)
    for fold, (_, indices_evaluacion) in enumerate(
        cv.split(np.zeros(len(etiquetas)), etiquetas), 1
    ):
        assert (asignacion[indices_evaluacion] == 0).all()
        asignacion[indices_evaluacion] = fold
    assert (asignacion > 0).all()
    pd.DataFrame(
        {
            "posicion_train": np.arange(len(train)),
            prep.COLUMNA_ID: train[prep.COLUMNA_ID],
            prep.OBJETIVO: etiquetas,
            "fold_evaluacion": asignacion,
        }
    ).to_csv(salida / f"indices_k{k}.csv", index=False)


def construir_decision(elegido, fraccion, train, dimension, elegible):
    decision = {
        "estado": "factible" if elegido else "no_factible",
        "k_elegido": elegido,
        "validation_fraction": fraccion,
        "semilla_folds": 42,
        "n_train": len(train),
        "margen": dimension["margen_dimensionamiento"],
        "justificacion": (
            "Se elige k=5 solo si cumple todo y las particiones externas estan"
            " aprobadas. En otro caso no se selecciona k. Se exportan los tres"
            " candidatos para diagnostico. El costo por filas acumuladas es"
            " (k-1)*N_train; no se afirma superioridad predictiva sin entrenar."
        ),
        "k_factibles_con_fraccion_elegida": elegible.k.astype(int).tolist(),
        "indices": (
            "indices_k5.csv es la asignacion aprobada; k=3 y k=10 quedan como"
            " comparaciones"
        ),
        "cota_interna": (
            "MLP estratifica su reserva interna. Se usa floor(n_clase * n_interno /"
            " n_fit) como cota inferior; no se exportan falsos indices exactos de la"
            " reserva interna del MLP."
        ),
        "preprocesamiento": (
            "Restaurar bitacora_imputacion_particiones.csv y ajustar imputacion,"
            " codificacion y escalado con el entrenamiento de cada fold; usar transform"
            " en su evaluacion. MLP crea despues la reserva interna para accuracy."
        ),
        "limite": (
            "Este archivo compara dimensionamiento y costo estructural de k. Los"
            " resultados predictivos de k=5 se documentan en entrenamiento."
        ),
    }
    return decision


def ejecutar(entrada, salida):
    """Compara soportes y costo para k=3, 5 y 10; no entrena redes."""
    entrada, salida = Path(entrada), Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(entrada / "train.csv", dtype={prep.COLUMNA_ID: str})
    minimos = json.loads((entrada / "minimos.json").read_text())
    dimension = json.loads((entrada / "dimensionamiento.json").read_text())
    etiquetas = train[prep.OBJETIVO].to_numpy()
    fraccion = dimension["validation_fraction"]
    resumenes = []
    detalles = []
    # Contrasta la fraccion 10% inicial y todas las alternativas prefijadas.
    for k in [3, 5, 10]:
        for fraccion_candidata in [0.10, 0.15, 0.20, 0.25]:
            resumen, detalle_candidato = evaluar_candidato(
                etiquetas,
                k,
                fraccion_candidata,
                minimos,
                margen=dimension["margen_dimensionamiento"],
            )
            resumenes.append(resumen)
            detalles.append(detalle_candidato)
        exportar_indices(train, etiquetas, k, salida)
    tabla = pd.DataFrame(resumenes)
    detalle = pd.concat(detalles, ignore_index=True)
    tabla.to_csv(salida / "comparacion_k_fracciones.csv", index=False)
    detalle.to_csv(salida / "conteos_todos_folds.csv", index=False)
    elegible = tabla.loc[tabla.validation_fraction.eq(fraccion) & tabla.cumple]
    elegido = (
        5 if 5 in elegible.k.to_list() and dimension["estado"] == "factible" else None
    )
    decision = construir_decision(elegido, fraccion, train, dimension, elegible)
    (salida / "diagnostico_crossval.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(tabla.to_string(index=False), flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("entrada")
    a.add_argument("salida")
    args = a.parse_args()
    ejecutar(args.entrada, args.salida)
