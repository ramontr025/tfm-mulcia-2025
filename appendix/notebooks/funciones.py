import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from interpret.glassbox import ExplainableBoostingClassifier
from hyperparameters_config import MODELS_HYPERPARAMS, GRID_SEARCH_CONFIG

def obtener_info_valores_unicos(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    soporte_map = {}
    
    for col in cat_cols:
        iloc_col = df.columns.get_loc(col)
        valores_unicos = frozenset(df[col].dropna().unique())
        
        if valores_unicos not in soporte_map:
            soporte_map[valores_unicos] = []
            
        soporte_map[valores_unicos].append(iloc_col)
            
    return pd.DataFrame([
        {"Indices_iloc": indices, "Soporte": list(valores_unicos)}
        for valores_unicos, indices in soporte_map.items()
    ])

def extraccion_columnas(dataset, tabla_soportes, soporte=[[]]):
    columnas = tabla_soportes[tabla_soportes["Soporte"].isin(soporte)]["Indices_iloc"]
    columnas_indices = list({item for sublist in columnas for item in sublist})
    columnas_nombres = dataset.columns[columnas_indices]
    return columnas_nombres


# ============================================================================
# BÚSQUEDA DE HIPERPARÁMETROS CON GRIDSEARCH
# ============================================================================

def construir_modelo(nombre_modelo):
    """
    Construye una instancia del modelo especificado.
    
    Parámetros:
    -----------
    nombre_modelo : str
        Nombre del modelo: 'RandomForest', 'XGBoost', 'MLPClassifier', 
        'EBM', 'NaiveBayes', 'SVM'
    
    Retorna:
    --------
    modelo : estimador de scikit-learn compatible
    """
    if nombre_modelo == 'RandomForest':
        return RandomForestClassifier(random_state=42)
    elif nombre_modelo == 'XGBoost':
        return xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    elif nombre_modelo == 'MLPClassifier':
        return MLPClassifier(random_state=42)
    elif nombre_modelo == 'EBM':
        return ExplainableBoostingClassifier(random_state=42)
    elif nombre_modelo == 'NaiveBayes':
        return GaussianNB()
    elif nombre_modelo == 'SVM':
        return SVC(random_state=42)
    else:
        raise ValueError(f"Modelo no reconocido: {nombre_modelo}")


def buscar_hiperparametros(modelo, X_train, y_train, nombre_modelo, 
                           param_grid=None, cv=5, scoring='f1_weighted', 
                           n_jobs=-1, verbose=1, preprocessing_pipeline=None):
    """
    Realiza búsqueda de hiperparámetros mediante GridSearchCV con validación cruzada.
    
    Parámetros:
    -----------
    modelo : estimador
        Modelo de scikit-learn a tunear
    X_train : array-like
        Características de entrenamiento
    y_train : array-like
        Variables objetivo de entrenamiento
    nombre_modelo : str
        Nombre del modelo para obtener parámetros de configuración
    param_grid : dict, opcional
        Diccionario de parámetros. Si es None, usa MODELS_HYPERPARAMS
    cv : int (default=5)
        Número de folds para validación cruzada
    scoring : str (default='f1_weighted')
        Métrica para seleccionar el mejor modelo. Opciones:
        - 'f1': F1-score estándar
        - 'f1_weighted': F1 ponderado (recomendado para clases desbalanceadas)
        - 'f1_macro': F1 macro
        - 'f1_micro': F1 micro
    n_jobs : int (default=-1)
        Número de procesos. -1 usa todos disponibles
    verbose : int (default=1)
        Nivel de detalle de la salida
    preprocessing_pipeline : Pipeline, opcional
        Pipeline de preprocesamiento. Si se proporciona, se integra con el modelo
        en una Pipeline completa para evitar data leakage en CV
    
    Retorna:
    --------
    dict : con las claves:
        - 'mejor_modelo': modelo ajustado con mejores parámetros
        - 'mejores_parametros': diccionario con parámetros óptimos
        - 'mejor_score': mejor puntuación F1 obtenida
        - 'grid_search': objeto GridSearchCV completo con todos los resultados
        - 'resultados_cv': DataFrame con todos los resultados de la búsqueda
    """
    from sklearn.pipeline import Pipeline as SKPipeline
    
    # Usar parámetros de configuración si no se proporcionan
    if param_grid is None:
        if nombre_modelo not in MODELS_HYPERPARAMS:
            raise ValueError(f"Configuración no encontrado para {nombre_modelo}")
        param_grid = MODELS_HYPERPARAMS[nombre_modelo]
    
    # Si se proporciona un pipeline de preprocesamiento, integrarlo con el modelo
    if preprocessing_pipeline is not None:
        # Crear pipeline completo: preprocesamiento + modelo
        modelo_completo = SKPipeline([
            ("preprocesamiento", preprocessing_pipeline),
            ("modelo", modelo)
        ])
        # Adaptar param_grid: agregar prefijo 'modelo__' a todos los parámetros
        param_grid_adaptado = {f"modelo__{k}": v for k, v in param_grid.items()}
    else:
        modelo_completo = modelo
        param_grid_adaptado = param_grid
    
    print(f"\n{'='*70}")
    print(f"Búsqueda de hiperparámetros para: {nombre_modelo}")
    if preprocessing_pipeline is not None:
        print(f"✓ Con pipeline de preprocesamiento integrado")
    print(f"{'='*70}")
    print(f"Métrica de evaluación: {scoring}")
    print(f"Validación cruzada: {cv}-fold")
    print(f"Número de combinaciones de parámetros: {_calcular_combinaciones(param_grid_adaptado)}")
    print(f"{'='*70}\n")
    
    # Crear GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo_completo,
        param_grid=param_grid_adaptado,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )
    
    # Ejecutar búsqueda
    grid_search.fit(X_train, y_train)
    
    # Preparar resultados
    resultados_df = pd.DataFrame(grid_search.cv_results_)
    
    # Seleccionar columnas relevantes
    columnas_interes = [col for col in resultados_df.columns 
                        if 'param_' in col or 'mean_test_score' in col 
                        or 'std_test_score' in col or 'rank_test_score' in col]
    resultados_df = resultados_df[columnas_interes]
    
    # Mostrar mejores resultados
    print(f"\n✓ Búsqueda completada")
    print(f"\nMejor puntuación ({scoring}): {grid_search.best_score_:.4f}")
    print(f"\nMejores parámetros:")
    for param, valor in grid_search.best_params_.items():
        print(f"  - {param}: {valor}")
    
    return {
        'mejor_modelo': grid_search.best_estimator_,
        'mejores_parametros': grid_search.best_params_,
        'mejor_score': grid_search.best_score_,
        'grid_search': grid_search,
        'resultados_cv': resultados_df.sort_values('rank_test_score')
    }


def tunear_multiples_modelos(modelos, X_train, y_train, scoring='f1_weighted', 
                             cv=5, n_jobs=-1, verbose=1, preprocessing_pipeline=None):
    """
    Tunea múltiples modelos usando la configuración centralizada.
    
    Parámetros:
    -----------
    modelos : list de str o dict
        Lista de nombres de modelos: ['RandomForest', 'XGBoost', 'MLPClassifier', 'EBM', 'NaiveBayes', 'SVM']
        O diccionario {nombre_modelo: param_grid_personalizado}
    X_train : array-like
        Características de entrenamiento (datos crudos si usa preprocessing_pipeline)
    y_train : array-like
        Variables objetivo de entrenamiento
    scoring : str (default='f1_weighted')
        Métrica para seleccionar el mejor modelo
    cv : int (default=5)
        Número de folds para validación cruzada
    n_jobs : int (default=-1)
        Número de procesos
    verbose : int (default=1)
        Nivel de detalle
    preprocessing_pipeline : Pipeline, opcional
        Pipeline de preprocesamiento. Si se proporciona, se integra con cada modelo
        para evitar data leakage en validación cruzada
    
    Retorna:
    --------
    dict : con los resultados de cada modelo
        Ejemplo de uso:
        
        resultados = tunear_multiples_modelos(
            modelos=['RandomForest', 'SVM', 'XGBoost'],
            X_train=X_train_sin_procesar,
            y_train=y_train,
            preprocessing_pipeline=mi_pipeline_preprocesamiento,
            cv=5,
            scoring='f1_weighted'
        )
        
        mejor_modelo = resultados['RandomForest']['mejor_modelo']
        mejor_modelo.predict(X_test_sin_procesar)  # El pipeline está incluido
    """
    
    resultados = {}
    
    # Convertir lista a diccionario si es necesario
    if isinstance(modelos, list):
        modelos_dict = {modelo: None for modelo in modelos}
    else:
        modelos_dict = modelos
    
    for nombre_modelo, param_grid in modelos_dict.items():
        try:
            modelo = construir_modelo(nombre_modelo)
            resultados[nombre_modelo] = buscar_hiperparametros(
                modelo=modelo,
                X_train=X_train,
                y_train=y_train,
                nombre_modelo=nombre_modelo,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=verbose,
                preprocessing_pipeline=preprocessing_pipeline  # NUEVO
            )
        except Exception as e:
            print(f"\n✗ Error al tunear {nombre_modelo}: {str(e)}")
            resultados[nombre_modelo] = {'error': str(e)}
    
    return resultados


def _calcular_combinaciones(param_grid):
    """Calcula el número total de combinaciones de parámetros."""
    import math
    total = 1
    for valores in param_grid.values():
        total *= len(valores)
    return total


def resumen_tuning(resultados_tuning):
    """
    Genera un resumen comparativo de todos los modelos tuneados.
    
    Parámetros:
    -----------
    resultados_tuning : dict
        Retorno de tunear_multiples_modelos()
    
    Retorna:
    --------
    pd.DataFrame : tabla comparativa de modelos
    """
    
    resumen = []
    errores = []
    
    for nombre_modelo, resultado in resultados_tuning.items():
        if 'error' not in resultado:
            resumen.append({
                'Modelo': nombre_modelo,
                'F1-Score': resultado['mejor_score'],
                'Mejores Parámetros': resultado['mejores_parametros']
            })
        else:
            errores.append({'Modelo': nombre_modelo, 'Error': resultado['error']})
    
    # Manejar caso cuando no hay resultados válidos
    if not resumen:
        print(f"\n{'='*70}")
        print("RESUMEN COMPARATIVO DE MODELOS TUNEADOS")
        print(f"{'='*70}\n")
        print("⚠️ No hay resultados válidos. Errores encontrados:")
        for err in errores:
            print(f"  - {err['Modelo']}: {err['Error']}")
        print(f"\n{'='*70}\n")
        return pd.DataFrame()
    
    resumen_df = pd.DataFrame(resumen).sort_values('F1-Score', ascending=False)
    
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO DE MODELOS TUNEADOS")
    print(f"{'='*70}\n")
    print(resumen_df.to_string(index=False))
    
    if errores:
        print(f"\n⚠️ Modelos con errores:")
        for err in errores:
            print(f"  - {err['Modelo']}: {err['Error']}")
    
    print(f"\n{'='*70}\n")
    
    return resumen_df
