"""Preguntas 4-6: MLP por etapas, preprocesamiento por fold y evaluacion final.

python entrenamiento.py preparar resultados /ruta/cache_temporal
python entrenamiento.py piloto resultados /ruta/cache_temporal
python entrenamiento.py ejecutar resultados /ruta/cache_temporal
Las matrices de cache son temporales y no forman parte de la entrega.
"""

import argparse
import gc
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

import preprocesamiento as prep
from muestreo import restaurar_marcas

CLASES = sorted(prep.reglas.CLASES)
BASE = {
    "hidden_layer_sizes": [32],
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": 512,
    "max_iter": 150,
    "n_iter_no_change": 15,
    "tol": 0.001,
    "early_stopping": True,
    "validation_fraction": 0.25,
    "shuffle": True,
}


def guardar_json(ruta, contenido):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Path(ruta).write_text(
        json.dumps(contenido, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def configuracion(parametros):
    configuracion_red = dict(BASE)
    configuracion_red.update(parametros)
    configuracion_red["hidden_layer_sizes"] = list(
        configuracion_red["hidden_layer_sizes"]
    )
    if configuracion_red["solver"] != "sgd":
        for parametro in ["momentum", "learning_rate", "nesterovs_momentum"]:
            configuracion_red.pop(parametro, None)
    return configuracion_red


def clave(configuracion_red):
    return hashlib.sha256(
        json.dumps(configuracion_red, sort_keys=True).encode()
    ).hexdigest()[:12]


def cargar_train(ruta_resultados):
    datos = pd.read_csv(ruta_resultados / "train.csv", dtype={prep.COLUMNA_ID: str})
    bitacora = pd.read_csv(
        ruta_resultados / "bitacora_imputacion_particiones.csv",
        dtype={prep.COLUMNA_ID: str},
    )
    return restaurar_marcas(datos, bitacora.loc[bitacora.particion.eq("train")])


def preparar_fold(datos, indices_folds, fold, ruta_resultados, ruta_cache):
    """Ajusta con el entrenamiento del fold y solo transforma su evaluacion."""
    carpeta_fold = ruta_cache / f"fold{fold}"
    carpeta_fold.mkdir(exist_ok=True)
    es_entrenamiento = indices_folds.fold_evaluacion.ne(fold)
    es_evaluacion = ~es_entrenamiento
    estado, transformador, matriz_entrenamiento = prep.ajustar_preprocesador(
        datos.loc[es_entrenamiento]
    )
    estado["huella_ids_entrenamiento"] = prep.huella_ids(
        datos.loc[es_entrenamiento, prep.COLUMNA_ID]
    )
    estado["huella_ids_evaluacion"] = prep.huella_ids(
        datos.loc[es_evaluacion, prep.COLUMNA_ID]
    )
    matriz_evaluacion = prep.transformar_particion(
        datos.loc[es_evaluacion], estado, transformador
    )
    assert (
        matriz_entrenamiento.shape[1] == 47
        and matriz_evaluacion.shape[1] == 47
        and np.isfinite(matriz_entrenamiento).all()
        and np.isfinite(matriz_evaluacion).all()
    )
    np.save(carpeta_fold / "x_fit.npy", matriz_entrenamiento)
    np.save(carpeta_fold / "x_eval.npy", matriz_evaluacion)
    np.save(
        carpeta_fold / "y_fit.npy",
        datos.loc[es_entrenamiento, prep.OBJETIVO].to_numpy(dtype=str),
    )
    np.save(
        carpeta_fold / "y_eval.npy",
        datos.loc[es_evaluacion, prep.OBJETIVO].to_numpy(dtype=str),
    )
    auditoria = {
        "fold": fold,
        "filas_fit": int(es_entrenamiento.sum()),
        "filas_eval": int(es_evaluacion.sum()),
        "entradas": 47,
        "estado_imputacion": estado,
        "huella_train_csv": prep.huella_archivo(ruta_resultados / "train.csv"),
        "escalador_centros": (
            transformador.named_transformers_["numericas"].center_.tolist()
        ),
        "escalador_escalas": (
            transformador.named_transformers_["numericas"].scale_.tolist()
        ),
        "id_y_clase_fuera_del_ajuste": True,
    }
    guardar_json(
        ruta_resultados / "entrenamiento/preprocesamiento_folds" / f"fold{fold}.json",
        auditoria,
    )
    return {k: auditoria[k] for k in ["fold", "filas_fit", "filas_eval", "entradas"]}


def preparar(ruta_resultados, ruta_cache):
    """Prepara cinco pares de matrices y deja la auditoria de cada ajuste."""
    ruta_resultados, ruta_cache = Path(ruta_resultados), Path(ruta_cache)
    ruta_cache.mkdir(parents=True, exist_ok=True)
    datos_json = json.loads(
        (ruta_resultados / "crossval/diagnostico_crossval.json").read_text()
    )
    if (
        datos_json["k_elegido"] != 5
        or not json.loads((ruta_resultados / "diagnostico_muestreo.json").read_text())[
            "particiones_aprobadas"
        ]
    ):
        raise ValueError("Muestreo y CV deben estar aprobados antes de modelar")
    datos = cargar_train(ruta_resultados)
    indices_folds = pd.read_csv(
        ruta_resultados / "crossval/indices_k5.csv", dtype={prep.COLUMNA_ID: str}
    )
    if not indices_folds[prep.COLUMNA_ID].equals(datos[prep.COLUMNA_ID]):
        raise ValueError("IDs/folds fuera de orden")
    auditorias = []
    for fold in range(1, 6):
        auditorias.append(
            preparar_fold(datos, indices_folds, fold, ruta_resultados, ruta_cache)
        )
        gc.collect()
        print("Preprocesamiento listo", fold, flush=True)
    guardar_json(
        ruta_cache / "manifest.json",
        {
            "train_sha256": prep.huella_archivo(ruta_resultados / "train.csv"),
            "folds": auditorias,
        },
    )


def codificar(etiquetas):
    """Indices de las cinco clases del diccionario; no son predictores ordinales.

    Evita un error de MLP 1.8.0 con etiquetas string y early_stopping. No se
    aprende de evaluacion: el catalogo esta definido por el problema.
    """
    if not np.isin(etiquetas, CLASES).all():
        raise ValueError("Clase fuera de catalogo")
    return np.searchsorted(CLASES, np.asarray(etiquetas)).astype("int64")


def cargar_fold(ruta_cache, fold):
    carpeta_fold = Path(ruta_cache) / f"fold{fold}"
    (
        matriz_entrenamiento,
        matriz_evaluacion,
        etiquetas_entrenamiento,
        etiquetas_evaluacion,
    ) = tuple(
        np.load(carpeta_fold / f"{nombre}.npy", mmap_mode="r")
        for nombre in ["x_fit", "x_eval", "y_fit", "y_eval"]
    )
    return (
        matriz_entrenamiento,
        matriz_evaluacion,
        codificar(etiquetas_entrenamiento),
        codificar(etiquetas_evaluacion),
    )


def piloto(ruta_resultados, ruta_cache):
    carpeta_entrenamiento = Path(ruta_resultados) / "entrenamiento"
    carpeta_entrenamiento.mkdir(exist_ok=True)
    (
        matriz_entrenamiento,
        matriz_evaluacion,
        etiquetas_entrenamiento,
        etiquetas_evaluacion,
    ) = cargar_fold(ruta_cache, 1)
    configuracion_red = configuracion({"max_iter": 8, "n_iter_no_change": 15})
    inicio = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        modelo = MLPClassifier(**configuracion_red, random_state=42).fit(
            matriz_entrenamiento, etiquetas_entrenamiento
        )
    tiempo = time.perf_counter() - inicio
    resultado = {
        "objetivo": (
            "Medir costo en un fold real; no seleccionar modelo con este piloto"
            " truncado"
        ),
        "filas_fit": len(etiquetas_entrenamiento),
        "filas_eval": len(etiquetas_evaluacion),
        "epocas": modelo.n_iter_,
        "segundos": tiempo,
        "segundos_por_epoca": tiempo / modelo.n_iter_,
        "hilos_blas": 1,
        "configuracion": configuracion_red,
        "avisos": [str(aviso.message) for aviso in avisos],
    }
    guardar_json(carpeta_entrenamiento / "piloto_recursos.json", resultado)
    print(json.dumps(resultado, ensure_ascii=False, indent=2), flush=True)


def metricas(etiquetas_reales, predicciones, probabilidades=None):
    resultados = {
        "accuracy": float(accuracy_score(etiquetas_reales, predicciones)),
        "balanced_accuracy": float(
            balanced_accuracy_score(etiquetas_reales, predicciones)
        ),
        "f1_macro": float(
            f1_score(etiquetas_reales, predicciones, average="macro", zero_division=0)
        ),
        "f1_ponderado": float(
            f1_score(
                etiquetas_reales, predicciones, average="weighted", zero_division=0
            )
        ),
    }
    if probabilidades is not None:
        if not np.isfinite(probabilidades).all():
            raise ValueError("Probabilidades no finitas")
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1, atol=2e-6)
        resultados["roc_auc_ovr_macro"] = float(
            roc_auc_score(
                etiquetas_reales,
                probabilidades,
                multi_class="ovr",
                average="macro",
                labels=np.arange(5),
            )
        )
        resultados["log_loss"] = float(
            log_loss(etiquetas_reales, probabilidades, labels=np.arange(5))
        )
    return resultados


def ajustar_cv(configuracion_red, fold, ruta_resultados, ruta_cache, semilla=42):
    ruta = (
        Path(ruta_resultados)
        / "entrenamiento/ajustes"
        / f"{clave(configuracion_red)}_fold{fold}_s{semilla}.json"
    )
    if ruta.exists():
        return json.loads(ruta.read_text())
    (
        matriz_entrenamiento,
        matriz_evaluacion,
        etiquetas_entrenamiento,
        etiquetas_evaluacion,
    ) = cargar_fold(ruta_cache, fold)
    inicio = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = MLPClassifier(**configuracion_red, random_state=semilla)
        modelo.fit(matriz_entrenamiento, etiquetas_entrenamiento)
    segundos = time.perf_counter() - inicio
    predicciones = modelo.predict(matriz_evaluacion)
    metricas_train = metricas(
        etiquetas_entrenamiento, modelo.predict(matriz_entrenamiento)
    )
    metricas_evaluacion = metricas(etiquetas_evaluacion, predicciones)
    resultado = {
        "id": clave(configuracion_red),
        "fold": fold,
        "semilla": semilla,
        "configuracion": configuracion_red,
        "estado": "correcto",
        "segundos_fit": segundos,
        "epocas": int(modelo.n_iter_),
        "filas_fit": len(etiquetas_entrenamiento),
        "filas_eval": len(etiquetas_evaluacion),
        "parametros_red": int(
            sum(parametro.size for parametro in modelo.coefs_)
            + sum(parametro.size for parametro in modelo.intercepts_)
        ),
        "train": metricas_train,
        "evaluacion": metricas_evaluacion,
        "brecha_f1": metricas_train["f1_macro"] - metricas_evaluacion["f1_macro"],
        "loss_curve": list(map(float, modelo.loss_curve_)),
        "accuracy_parada_interna": list(map(float, modelo.validation_scores_)),
        "mejor_accuracy_interna": float(modelo.best_validation_score_),
        "avisos": [str(parametro.message) for parametro in avisos],
    }
    guardar_json(ruta, resultado)
    print(
        f"CV {resultado['id']} fold {fold}: F1={metricas_evaluacion['f1_macro']:.4f},"
        f" epocas={resultado['epocas']}, s={segundos:.1f}",
        flush=True,
    )
    return resultado


def resumen_cv(resultados):
    filas = []
    for identificador in sorted({resultado["id"] for resultado in resultados}):
        resultados_configuracion = [
            resultado for resultado in resultados if resultado["id"] == identificador
        ]
        # El paralelismo cambia el orden de llegada. Fijarlo evita diferencias
        # de redondeo al reducir los mismos cinco valores de punto flotante.
        resultados_configuracion.sort(key=lambda resultado: resultado["fold"])
        assert len(resultados_configuracion) == 5
        valores_f1 = np.array(
            [
                resultado["evaluacion"]["f1_macro"]
                for resultado in resultados_configuracion
            ]
        )
        filas.append(
            {
                "id": identificador,
                "configuracion": json.dumps(
                    resultados_configuracion[0]["configuracion"], sort_keys=True
                ),
                "f1_macro_media": float(valores_f1.mean()),
                "f1_macro_desv": float(valores_f1.std(ddof=1)),
                "f1_macro_varianza": float(valores_f1.var(ddof=1)),
                "balanced_accuracy_media": float(
                    np.mean(
                        [
                            resultado["evaluacion"]["balanced_accuracy"]
                            for resultado in resultados_configuracion
                        ]
                    )
                ),
                "accuracy_media": float(
                    np.mean(
                        [
                            resultado["evaluacion"]["accuracy"]
                            for resultado in resultados_configuracion
                        ]
                    )
                ),
                "f1_train_media": float(
                    np.mean(
                        [
                            resultado["train"]["f1_macro"]
                            for resultado in resultados_configuracion
                        ]
                    )
                ),
                "brecha_f1_media": float(
                    np.mean(
                        [
                            resultado["brecha_f1"]
                            for resultado in resultados_configuracion
                        ]
                    )
                ),
                "epocas_media": float(
                    np.mean(
                        [resultado["epocas"] for resultado in resultados_configuracion]
                    )
                ),
                "segundos_fit_suma": float(
                    sum(
                        resultado["segundos_fit"]
                        for resultado in resultados_configuracion
                    )
                ),
                "parametros_red": resultados_configuracion[0]["parametros_red"],
            }
        )
    return pd.DataFrame(filas).sort_values(
        ["f1_macro_media", "f1_macro_desv", "parametros_red", "id"],
        ascending=[False, True, True, True],
    )


def etapa(nombre, candidatos, ruta_resultados, ruta_cache):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unicos = {
        clave(configuracion_candidata): configuracion_candidata
        for configuracion_candidata in candidatos
    }
    guardar_json(
        Path(ruta_resultados) / "entrenamiento" / f"rejilla_{nombre}.json",
        list(unicos.values()),
    )
    resultados = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=3) as executor:
        tareas = [
            executor.submit(
                ajustar_cv, configuracion_candidata, fold, ruta_resultados, ruta_cache
            )
            for configuracion_candidata in unicos.values()
            for fold in range(1, 6)
        ]
        for tarea in as_completed(tareas):
            resultados.append(tarea.result())
    resumen = resumen_cv(resultados)
    resumen.to_csv(
        Path(ruta_resultados) / "entrenamiento" / f"resumen_{nombre}.csv", index=False
    )
    mejor = json.loads(resumen.iloc[0].configuracion)
    print(
        "ETAPA TERMINADA",
        nombre,
        "mejor",
        resumen.iloc[0].id,
        "F1",
        resumen.iloc[0].f1_macro_media,
        flush=True,
    )
    return mejor


def entrenamiento_completo(configuracion_red, semilla, datos_train, matriz_train):
    # Validacion externa solo se usa tras cerrar las rejillas de CV.
    modelo = MLPClassifier(**configuracion_red, random_state=semilla)
    inicio = time.perf_counter()
    with warnings.catch_warnings(record=True) as avisos, threadpool_limits(limits=1):
        warnings.simplefilter("always")
        modelo.fit(matriz_train, codificar(datos_train[prep.OBJETIVO]))
    return modelo, {
        "semilla": semilla,
        "configuracion": configuracion_red,
        "segundos_fit": time.perf_counter() - inicio,
        "epocas": int(modelo.n_iter_),
        "loss_curve": list(map(float, modelo.loss_curve_)),
        "accuracy_parada_interna": list(map(float, modelo.validation_scores_)),
        "avisos": [str(aviso.message) for aviso in avisos],
    }


def construir_protocolo():
    protocolo = {
        "seleccion": (
            "F1 macro medio de CV estratificada k=5; empate exacto por menor desviacion"
            " y menor red"
        ),
        "semilla_red_cv": 42,
        "semillas_replicas": [42, 17, 73],
        "preprocesamiento": (
            "Imputacion y RobustScaler aprendidos por fold; OneHot catalogo fijo"
        ),
        "parada": {
            "validation_fraction": 0.25,
            "tol": 0.001,
            "max_iter_inicial": 150,
            "n_iter_no_change_inicial": 15,
            "monitor": "accuracy",
        },
        "etapas": {
            "A": {
                "hidden_layer_sizes": [[32], [64], [64, 32]],
                "activation": ["relu", "tanh"],
            },
            "B": (
                "Ganador A: Adam LR .001/.003; SGD LR .001/.01, momentum 0/.9,"
                " constante, Nesterov desactivado"
            ),
            "C": (
                "Ganador B: alpha .00001/.0001/.001, batch 256/512/1024, subconjunto"
                " 2x2 mas referencia"
            ),
            "D": "Ganador C: max_iter 100/200 y paciencia 10/20 mas referencia",
        },
        "presupuesto": (
            "Hasta 19 configuraciones nuevas x 5 folds; se reutilizan ajustes de"
            " configuraciones identicas entre etapas. Rejilla reducida respecto al plan"
            " inicial tras piloto de recursos."
        ),
        "validacion_externa": (
            "Compara los dos finalistas de CV; F1 macro; sin consultar test"
        ),
        "final": (
            "Semilla 42 fijada antes de ver test; dos replicas 17/73 evaluadas solo en"
            " validacion como sensibilidad"
        ),
        "test": (
            "Una sola evaluacion con decision congelada; no selecciona configuracion ni"
            " semilla"
        ),
        "recursos": {
            "trabajadores": 3,
            "hilos_blas_por_ajuste": 1,
            "matrices_modelo": "float32",
            "entradas": 47,
        },
        "versiones": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "etiquetas": dict(enumerate(CLASES)),
        "aclaracion_etiquetas": (
            "Indices del diccionario para evitar TypeError de early_stopping con"
            " strings en MLP 1.8.0; nunca entran en X."
        ),
    }
    return protocolo


def ejecutar_busqueda(ruta_resultados, ruta_cache, carpeta_entrenamiento):
    """Explora las mismas 19 configuraciones por etapas y conserva dos finalistas."""
    candidatos_a = [
        configuracion(parametros_a)
        for parametros_a in ParameterGrid(
            {
                "hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "activation": ["relu", "tanh"],
            }
        )
    ]
    ganador_a = etapa("A", candidatos_a, ruta_resultados, ruta_cache)
    candidatos_b = [
        ganador_a,
        configuracion(dict(ganador_a, solver="adam", learning_rate_init=0.003)),
    ]
    candidatos_b += [
        configuracion(
            dict(
                ganador_a,
                solver="sgd",
                learning_rate="constant",
                nesterovs_momentum=False,
                **parametros_candidatos,
            )
        )
        for parametros_candidatos in ParameterGrid(
            {"learning_rate_init": [0.001, 0.01], "momentum": [0.0, 0.9]}
        )
    ]
    ganador_b = etapa("B", candidatos_b, ruta_resultados, ruta_cache)
    candidatos_c = [ganador_b] + [
        configuracion(dict(ganador_b, **parametros_candidatos))
        for parametros_candidatos in ParameterGrid(
            {"alpha": [0.00001, 0.001], "batch_size": [256, 1024]}
        )
    ]
    ganador_c = etapa("C", candidatos_c, ruta_resultados, ruta_cache)
    candidatos_d = [ganador_c] + [
        configuracion(dict(ganador_c, **parametros_candidatos))
        for parametros_candidatos in ParameterGrid(
            {"max_iter": [100, 200], "n_iter_no_change": [10, 20]}
        )
    ]
    etapa("D", candidatos_d, ruta_resultados, ruta_cache)
    resultados = [
        json.loads(archivo.read_text())
        for archivo in (carpeta_entrenamiento / "ajustes").glob("*.json")
    ]
    resumen = resumen_cv(resultados)
    resumen.to_csv(carpeta_entrenamiento / "resumen_cv_todas.csv", index=False)
    finalistas = [
        json.loads(texto_configuracion)
        for texto_configuracion in resumen.head(2).configuracion
    ]
    guardar_json(
        carpeta_entrenamiento / "finalistas_antes_validacion.json",
        {
            "configs": finalistas,
            "criterio": (
                "Dos mayores F1 macro medios de CV entre las configuraciones exploradas"
            ),
        },
    )
    return resumen, resultados, finalistas


def preparar_ajuste_final(ruta_resultados):
    """Ajusta una vez sobre train; validation recibe solo transformaciones."""
    # Reajuste final solo con train; no se agregan filas de validation.
    datos_train = cargar_train(ruta_resultados)
    estado, transformador, matriz_train = prep.ajustar_preprocesador(datos_train)
    estado["huella_ids_entrenamiento"] = prep.huella_ids(datos_train[prep.COLUMNA_ID])
    datos_validacion = pd.read_csv(
        ruta_resultados / "validation.csv", dtype={prep.COLUMNA_ID: str}
    )
    bitacora = pd.read_csv(
        ruta_resultados / "bitacora_imputacion_particiones.csv",
        dtype={prep.COLUMNA_ID: str},
    )
    validacion_observada = restaurar_marcas(
        datos_validacion, bitacora.loc[bitacora.particion.eq("validation")]
    )
    matriz_validacion = prep.transformar_particion(
        validacion_observada, estado, transformador
    )
    etiquetas_validacion = codificar(datos_validacion[prep.OBJETIVO])
    return (
        datos_train,
        estado,
        transformador,
        matriz_train,
        matriz_validacion,
        etiquetas_validacion,
        bitacora,
    )


def seleccionar_finalista(
    finalistas,
    datos_train,
    ruta_cache,
    carpeta_entrenamiento,
    matriz_train,
    matriz_validacion,
    etiquetas_validacion,
):
    """Decide con validation y guarda la decision antes de abrir test."""
    candidatos = []
    for configuracion_red in finalistas:
        modelo, resultado = entrenamiento_completo(
            configuracion_red, 42, datos_train, matriz_train
        )
        resultado["id"] = clave(configuracion_red)
        resultado["validacion"] = metricas(
            etiquetas_validacion,
            modelo.predict(matriz_validacion),
            modelo.predict_proba(matriz_validacion),
        )
        guardar_json(
            carpeta_entrenamiento / "finalistas" / f'{resultado["id"]}_s42.json',
            resultado,
        )
        joblib.dump(modelo, ruta_cache / f'modelo_{resultado["id"]}.joblib', compress=3)
        candidatos.append(resultado)
        print("FINALISTA", resultado["id"], resultado["validacion"], flush=True)
    candidatos.sort(
        key=lambda resultado: (-resultado["validacion"]["f1_macro"], resultado["id"])
    )
    mejor = candidatos[0]
    configuracion_red = mejor["configuracion"]
    modelo = joblib.load(ruta_cache / f'modelo_{mejor["id"]}.joblib')
    decision = {
        "configuracion": configuracion_red,
        "id": mejor["id"],
        "semilla_final": 42,
        "validacion": mejor["validacion"],
        "etiquetas": CLASES,
        "criterio": "F1 macro validacion entre dos finalistas; semilla 42 prefijada",
        "test_aun_sin_evaluar": True,
        "entrenamiento_filas": len(datos_train),
        "validation_fraction": 0.25,
    }
    guardar_json(carpeta_entrenamiento / "decision_antes_test.json", decision)
    return modelo, mejor, decision


def evaluar_replicas(
    configuracion_red,
    mejor,
    datos_train,
    carpeta_entrenamiento,
    matriz_train,
    matriz_validacion,
    etiquetas_validacion,
):
    """Las semillas 17 y 73 son sensibilidad; no sustituyen la semilla final 42."""
    # Replicas descriptivas: mismo cfg, mismo train; no se elige la mejor semilla.
    replicas = [dict(mejor)]
    for semilla in [17, 73]:
        modelo, resultado = entrenamiento_completo(
            configuracion_red, semilla, datos_train, matriz_train
        )
        resultado["id"] = clave(configuracion_red)
        resultado["validacion"] = metricas(
            etiquetas_validacion,
            modelo.predict(matriz_validacion),
            modelo.predict_proba(matriz_validacion),
        )
        guardar_json(carpeta_entrenamiento / "replicas" / f"s{semilla}.json", resultado)
        replicas.append(resultado)
        print("REPLICA", semilla, resultado["validacion"], flush=True)
    pd.DataFrame(
        [
            dict(
                semilla=resultado["semilla"],
                epocas=resultado["epocas"],
                segundos=resultado["segundos_fit"],
                **resultado["validacion"],
            )
            for resultado in replicas
        ]
    ).to_csv(carpeta_entrenamiento / "replicas_validacion.csv", index=False)


def exportar_clasificacion(etiquetas_test, predicciones, carpeta_entrenamiento):
    reporte = classification_report(
        etiquetas_test,
        predicciones,
        labels=np.arange(5),
        target_names=CLASES,
        output_dict=True,
        zero_division=0,
    )
    tabla_clases = pd.DataFrame(
        {k: v for k, v in reporte.items() if isinstance(v, dict)}
    ).T
    tabla_clases["support"] = tabla_clases["support"].astype(int)
    tabla_clases.to_csv(carpeta_entrenamiento / "metricas_por_clase_test.csv")
    conf = confusion_matrix(etiquetas_test, predicciones, labels=np.arange(5))
    pd.DataFrame(conf, index=CLASES, columns=CLASES).to_csv(
        carpeta_entrenamiento / "matriz_confusion_test.csv",
        index_label="real_vs_predicha",
    )


def exportar_predicciones(
    datos_test, predicciones, matriz_probabilidades, carpeta_entrenamiento
):
    """Escribe primero un temporal y verifica que salgan todas las filas."""
    probabilidades = pd.DataFrame(
        matriz_probabilidades, columns=["prob_" + clase for clase in CLASES]
    )
    probabilidades.insert(0, "prediccion", np.asarray(CLASES)[predicciones])
    probabilidades.insert(0, "clase_real", datos_test[prep.OBJETIVO])
    probabilidades.insert(0, "id_cliente", datos_test[prep.COLUMNA_ID])
    temporal = carpeta_entrenamiento / "predicciones_test.tmp"
    with temporal.open("w", encoding="utf-8", newline="") as archivo:
        probabilidades.to_csv(archivo, index=False)
        archivo.flush()
        os.fsync(archivo.fileno())
    os.replace(temporal, carpeta_entrenamiento / "predicciones_test.csv")
    if len(
        pd.read_csv(
            carpeta_entrenamiento / "predicciones_test.csv", usecols=["id_cliente"]
        )
    ) != len(datos_test):
        raise ValueError("La exportacion de predicciones esta incompleta")


def exportar_roc(etiquetas_test, probabilidades, carpeta_entrenamiento):
    roc = []
    for indice_clase, clase in enumerate(CLASES):
        fpr, tpr, umbrales = roc_curve(
            etiquetas_test == indice_clase, probabilidades[:, indice_clase]
        )
        roc.append(
            pd.DataFrame({"clase": clase, "fpr": fpr, "tpr": tpr, "umbral": umbrales})
        )
    pd.concat(roc, ignore_index=True).to_csv(
        carpeta_entrenamiento / "curvas_roc_test.csv", index=False
    )


def evaluar_test(
    modelo,
    datos_train,
    ruta_resultados,
    carpeta_entrenamiento,
    estado,
    transformador,
    matriz_train,
    matriz_validacion,
    etiquetas_validacion,
    bitacora,
):
    """Evalua la decision congelada y exporta metricas, predicciones y curvas."""
    # Evaluacion test, exclusivamente despues de congelar configuracion y semilla.
    datos_test = pd.read_csv(ruta_resultados / "test.csv", dtype={prep.COLUMNA_ID: str})
    test_observado = restaurar_marcas(
        datos_test, bitacora.loc[bitacora.particion.eq("test")]
    )
    matriz_test = prep.transformar_particion(test_observado, estado, transformador)
    etiquetas_test = codificar(datos_test[prep.OBJETIVO])
    with threadpool_limits(limits=1):
        probabilidades = modelo.predict_proba(matriz_test)
        predicciones = np.argmax(probabilidades, axis=1)
        metricas_train = metricas(
            codificar(datos_train[prep.OBJETIVO]), modelo.predict(matriz_train)
        )
    metricas_validacion = metricas(
        etiquetas_validacion,
        modelo.predict(matriz_validacion),
        modelo.predict_proba(matriz_validacion),
    )
    metricas_test = metricas(etiquetas_test, predicciones, probabilidades)
    exportar_clasificacion(etiquetas_test, predicciones, carpeta_entrenamiento)
    exportar_predicciones(
        datos_test, predicciones, probabilidades, carpeta_entrenamiento
    )
    exportar_roc(etiquetas_test, probabilidades, carpeta_entrenamiento)
    return metricas_train, metricas_validacion, metricas_test


def construir_resumen_final(
    decision,
    metricas_train,
    metricas_validacion,
    metricas_test,
    resumen,
    resultados,
    mejor,
):
    resumen_final = {
        "estado": "ejecutado",
        "decision": decision,
        "train": metricas_train,
        "validation": metricas_validacion,
        "test": metricas_test,
        "configuraciones_exploradas": len(resumen),
        "ajustes_cv": len(resultados),
        "replicas_validacion": 3,
        "modelo_final_epocas": mejor["epocas"],
        "test_evaluado_una_vez_para_seleccion_congelada": True,
        "limitaciones": [
            (
                "Busqueda por etapas, no rejilla cartesiana exhaustiva de todos los"
                " parametros"
            ),
            (
                "Desviacion/varianza entre folds descriptivas; no intervalos de"
                " confianza de folds independientes"
            ),
            "Parada interna usa accuracy; seleccion externa F1 macro",
            "Metricas train incluyen la reserva interna de parada del estimador",
            (
                "No se afirma que estabilidad de medias garantice representatividad"
                " universal"
            ),
        ],
    }
    return resumen_final


def ejecutar(ruta_resultados, ruta_cache):
    """Coordina la busqueda, seleccion, sensibilidad y evaluacion final."""
    ruta_resultados, ruta_cache = Path(ruta_resultados), Path(ruta_cache)
    carpeta_entrenamiento = ruta_resultados / "entrenamiento"
    carpeta_entrenamiento.mkdir(exist_ok=True)
    if json.loads((ruta_cache / "manifest.json").read_text())[
        "train_sha256"
    ] != prep.huella_archivo(ruta_resultados / "train.csv"):
        raise ValueError("Cache de otra version de train")
    guardar_json(
        carpeta_entrenamiento / "protocolo_entrenamiento.json", construir_protocolo()
    )
    resumen, resultados, finalistas = ejecutar_busqueda(
        ruta_resultados, ruta_cache, carpeta_entrenamiento
    )
    (
        datos_train,
        estado,
        transformador,
        matriz_train,
        matriz_validacion,
        etiquetas_validacion,
        bitacora,
    ) = preparar_ajuste_final(ruta_resultados)
    modelo, mejor, decision = seleccionar_finalista(
        finalistas,
        datos_train,
        ruta_cache,
        carpeta_entrenamiento,
        matriz_train,
        matriz_validacion,
        etiquetas_validacion,
    )
    configuracion_red = mejor["configuracion"]
    evaluar_replicas(
        configuracion_red,
        mejor,
        datos_train,
        carpeta_entrenamiento,
        matriz_train,
        matriz_validacion,
        etiquetas_validacion,
    )
    metricas_train, metricas_validacion, metricas_test = evaluar_test(
        modelo,
        datos_train,
        ruta_resultados,
        carpeta_entrenamiento,
        estado,
        transformador,
        matriz_train,
        matriz_validacion,
        etiquetas_validacion,
        bitacora,
    )
    resumen_final = construir_resumen_final(
        decision,
        metricas_train,
        metricas_validacion,
        metricas_test,
        resumen,
        resultados,
        mejor,
    )
    guardar_json(
        carpeta_entrenamiento / "diagnostico_entrenamiento.json", resumen_final
    )
    guardar_json(carpeta_entrenamiento / "estado_imputacion_final.json", estado)
    joblib.dump(
        {
            "modelo": modelo,
            "transformador": transformador,
            "imputacion": estado,
            "clases": CLASES,
            "configuracion": configuracion_red,
        },
        carpeta_entrenamiento / "modelo_final.joblib",
        compress=3,
    )
    print("FINAL", json.dumps(resumen_final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("modo", choices=["preparar", "piloto", "ejecutar"])
    a.add_argument("resultados")
    a.add_argument("cache")
    args = a.parse_args()
    if args.modo == "preparar":
        preparar(args.resultados, args.cache)
    elif args.modo == "piloto":
        piloto(args.resultados, args.cache)
    else:
        ejecutar(args.resultados, args.cache)
