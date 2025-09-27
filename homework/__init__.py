"""Módulo de ajuste de hiperparámetros para predicción de calidad de vino."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


def load_data():
    """Carga los datos de calidad de vino desde UCI ML Repository."""
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
    
    y = df["quality"]
    x = df.copy()
    x.pop("quality")
    
    return x, y


def make_train_test_split(x, y):
    """Divide los datos en conjuntos de entrenamiento y prueba."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=0
    )
    return x_train, x_test, y_train, y_test


def create_model_grids():
    """Define grillas de hiperparámetros para diferentes algoritmos."""
    
    # Random Forest - grilla más pequeña para velocidad
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    
    # Gradient Boosting - grilla más pequeña para velocidad
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.2],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    # ElasticNet con escalado - grilla simplificada
    en_params = {
        'model__alpha': [0.01, 0.1, 1.0],
        'model__l1_ratio': [0.3, 0.5, 0.7]
    }
    
    # Ridge con escalado - grilla simplificada
    ridge_params = {
        'model__alpha': [1.0, 10.0, 100.0]
    }
    
    # SVR con escalado - grilla simplificada
    svr_params = {
        'model__C': [1.0, 10.0],
        'model__gamma': ['scale', 'auto'],
        'model__kernel': ['rbf']
    }
    
    models = [
        (Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(random_state=0))]), en_params),
        (Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=0))]), ridge_params),
        (Pipeline([('scaler', StandardScaler()), ('model', SVR())]), svr_params),
        (RandomForestRegressor(random_state=0), rf_params),
        (GradientBoostingRegressor(random_state=0), gb_params)
    ]
    
    return models


def hyperparameter_tuning(x_train, y_train):
    """Realiza ajuste de hiperparámetros usando GridSearchCV."""
    
    models = create_model_grids()
    best_score = -np.inf
    best_estimator = None
    
    print("Iniciando ajuste de hiperparámetros...")
    
    for i, (model, params) in enumerate(models):
        print(f"Evaluando modelo {i+1}/5: {type(model.steps[-1][1] if hasattr(model, 'steps') else model).__name__}")
        
        # GridSearchCV con validación cruzada
        grid_search = GridSearchCV(
            model, 
            params, 
            cv=5, 
            scoring='r2', 
            n_jobs=-1, 
            verbose=0
        )
        
        grid_search.fit(x_train, y_train)
        
        # Evaluar con validación cruzada adicional
        cv_scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=5, scoring='r2')
        mean_cv_score = cv_scores.mean()
        
        print(f"  Mejor R² CV: {mean_cv_score:.4f}")
        
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_estimator = grid_search.best_estimator_
            print(f"  ¡Nuevo mejor modelo! R²: {best_score:.4f}")
    
    print(f"\nMejor modelo encontrado con R² CV: {best_score:.4f}")
    return best_estimator


def save_estimator(estimator, filename="estimator.pickle"):
    """Guarda el mejor estimador en un archivo pickle."""
    with open(filename, "wb") as file:
        pickle.dump(estimator, file)
    print(f"Estimador guardado en {filename}")


def main():
    """Función principal que ejecuta todo el pipeline."""
    
    # Cargar datos
    print("Cargando datos...")
    x, y = load_data()
    print(f"Datos cargados: {x.shape[0]} muestras, {x.shape[1]} características")
    
    # Dividir datos
    print("Dividiendo datos en entrenamiento y prueba...")
    x_train, x_test, y_train, y_test = make_train_test_split(x, y)
    print(f"Entrenamiento: {x_train.shape[0]} muestras")
    print(f"Prueba: {x_test.shape[0]} muestras")
    
    # Ajuste de hiperparámetros
    best_estimator = hyperparameter_tuning(x_train, y_train)
    
    # Evaluar en conjunto de prueba
    print("\nEvaluando en conjunto de prueba...")
    y_pred = best_estimator.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Rendimiento en conjunto de prueba:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Guardar el mejor estimador
    save_estimator(best_estimator)
    
    # Verificar que cumple con el requerimiento
    if r2 > 0.3450:
        print(f"\n✅ ¡Éxito! El modelo cumple con el requerimiento R² > 0.3450")
        print(f"   R² obtenido: {r2:.4f}")
    else:
        print(f"\n❌ El modelo no cumple con el requerimiento R² > 0.3450")
        print(f"   R² obtenido: {r2:.4f}")
    
    return best_estimator, r2


if __name__ == "__main__":
    main()
