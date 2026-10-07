"""
Configuración centralizada de hiperparámetros para modelos de ML.
Define los espacios de búsqueda para GridSearchCV.
"""

# ============================================================================
# RANDOM FOREST
# ============================================================================
RANDOM_FOREST_PARAMS = {
    'n_estimators': [100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt'],
    'bootstrap': [True, False],
}

# ============================================================================
# XGBOOST
# ============================================================================
XGBOOST_PARAMS = {
    'n_estimators': [100],
    'max_depth': [3, 7],
    'learning_rate': [0.01],
    'subsample': [ 1.0],
    'colsample_bytree': [0.7, 0.8],
    'reg_lambda': [0.1, 1],
    'reg_alpha': [0, 0.1],
}

# ============================================================================
# MLP CLASSIFIER (Neural Network)
# ============================================================================
MLP_CLASSIFIER_PARAMS = {
    'hidden_layer_sizes': [(64,), (128,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [1000],
}

# ============================================================================
# EXPLAINABLE BOOSTING MACHINE (EBM)
# ============================================================================
EBM_PARAMS = {
    'max_bins': [32, 64],
    'max_interaction_bins': [32],
    'interactions': [10, 20],
    'outer_bags': [8, 16],
    'inner_bags': [0],
}

# ============================================================================
# NAIVE BAYES
# ============================================================================
NAIVE_BAYES_PARAMS = {
    'var_smoothing': [1e-10, 1e-8],
}

# ============================================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================================
SVM_PARAMS = {
    'C': [0.1, 1],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3],
    'probability': [True],
}

# ============================================================================
# DICCIONARIO CENTRAL DE CONFIGURACIÓN
# ============================================================================
MODELS_HYPERPARAMS = {
    'RandomForest': RANDOM_FOREST_PARAMS,
    'XGBoost': XGBOOST_PARAMS,
    'MLPClassifier': MLP_CLASSIFIER_PARAMS,
    'EBM': EBM_PARAMS,
    'NaiveBayes': NAIVE_BAYES_PARAMS,
    'SVM': SVM_PARAMS,
}

# ============================================================================
# CONFIGURACIÓN DE GRID SEARCH
# ============================================================================
GRID_SEARCH_CONFIG = {
    'cv': 2,  # Validación cruzada: 2-fold
    'scoring': 'f1_weighted',  # Métrica F1 (weighted para datasets desbalanceados)
    'n_jobs': -1,  # Usar todos los procesadores disponibles
    'verbose': 1,  # Mostrar progreso
}
