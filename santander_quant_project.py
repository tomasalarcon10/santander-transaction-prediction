
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
santander_quant_project.py

Proyecto completo de analítica predictiva para el problema
"Santander Customer Transaction Prediction" (estilo Kaggle).
Genera datos sintéticos, realiza EDA, entrena modelos de ML y compara resultados.
Sin dependencia de TensorFlow; usa MLPClassifier de scikit-learn.
"""

# =============================================================================
# 1. IMPORTACIONES
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys

# Configuración de warnings
warnings.filterwarnings('ignore')

# Scikit-learn y utilidades
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                             confusion_matrix, accuracy_score, precision_recall_curve)

# Para balanceo de clases
from imblearn.over_sampling import SMOTE

# Modelos avanzados
import xgboost as xgb
import lightgbm as lgb

# Estilo de gráficos
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Semilla para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# 2. GENERACIÓN O CARGA DE DATOS SINTÉTICOS
# =============================================================================
print("="*60)
print("1. GENERACIÓN DE DATOS SINTÉTICOS")
print("="*60)

DATA_FILE = 'santander_synthetic_dataset.csv.gz'

if os.path.exists(DATA_FILE):
    print(f"Cargando datos existentes desde {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)
    print(f"Shape: {df.shape}")
else:
    print("Generando nuevo dataset sintético...")
    # Parámetros del dataset (ajustables)
    n_samples = 200_000          # Número de clientes
    n_features_informative = 20   # Variables realmente informativas
    n_features_redundant = 40     # Variables redundantes (combinaciones lineales)
    n_features_noise = 140        # Variables de ruido puro
    n_features_total = n_features_informative + n_features_redundant + n_features_noise
    weights = [0.9, 0.1]          # Desbalance: 10% de positivos

    # Generar features informativas + redundantes con make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features_informative + n_features_redundant,
        n_informative=n_features_informative,
        n_redundant=n_features_redundant,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=weights,
        flip_y=0.01,
        class_sep=1.0,
        shuffle=False,
        random_state=RANDOM_STATE
    )

    # Añadir ruido puro (gaussiano)
    X_noise = np.random.normal(0, 1, (n_samples, n_features_noise))
    X_full = np.hstack([X, X_noise])

    # Mezclar columnas para que las informativas no estén juntas
    indices = np.random.permutation(n_features_total)
    X_full = X_full[:, indices]

    # Escalar y aplicar transformaciones para simular datos reales
    scaler_temp = StandardScaler()
    X_scaled = scaler_temp.fit_transform(X_full)

    # Desplazar y reescalar cada columna aleatoriamente
    for i in range(n_features_total):
        shift = np.random.choice([-2, -1, 0, 1, 2]) * np.random.uniform(0.5, 2)
        scale = np.random.uniform(0.5, 3)
        X_scaled[:, i] = X_scaled[:, i] * scale + shift
        # Añadir outliers en algunas columnas (10% de las columnas, 1% de filas)
        if np.random.rand() < 0.1:
            outlier_idx = np.random.choice(n_samples, size=int(n_samples*0.01), replace=False)
            X_scaled[outlier_idx, i] += np.random.choice([-5, 5]) * np.random.uniform(2, 5)

    # Crear DataFrame con nombres var_0 ... var_199
    column_names = [f'var_{i}' for i in range(n_features_total)]
    df = pd.DataFrame(X_scaled, columns=column_names)
    df['target'] = y

    # Guardar a CSV comprimido
    df.to_csv(DATA_FILE, index=False, compression='gzip')
    print(f"Dataset generado y guardado en {DATA_FILE} (shape: {df.shape})")

# Mostrar distribución del target
print("\nDistribución de la variable objetivo:")
print(df['target'].value_counts(normalize=True))

# =============================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n" + "="*60)
print("2. ANÁLISIS EXPLORATORIO DE DATOS")
print("="*60)

# Información general
print("\n--- Información del dataset ---")
print(df.info())

# Estadísticas descriptivas de las primeras 5 variables
print("\n--- Estadísticas descriptivas (primeras 5 variables) ---")
print(df.iloc[:, :5].describe())

# Verificar valores faltantes
print("\n--- Valores faltantes ---")
print(df.isnull().sum().sum())

# Análisis de la variable objetivo
fig, axes = plt.subplots(1, 2, figsize=(12,4))
df['target'].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
axes[0].set_title('Distribución de target')
axes[0].set_xticklabels(['No transacción (0)', 'Transacción (1)'], rotation=0)
df['target'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
axes[1].set_title('Proporción de clases')
plt.tight_layout()
plt.savefig('target_distribucion.png')
plt.show()

# Distribución de 6 variables aleatorias comparando target=0 y target=1
np.random.seed(RANDOM_STATE)
sample_vars = np.random.choice(df.columns[:-1], 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.ravel()
for i, var in enumerate(sample_vars):
    axes[i].hist(df[df['target']==0][var], bins=50, alpha=0.5, label='target=0', density=True)
    axes[i].hist(df[df['target']==1][var], bins=50, alpha=0.5, label='target=1', density=True)
    axes[i].set_title(f'Distribución de {var}')
    axes[i].legend()
plt.tight_layout()
plt.savefig('distribuciones_variables.png')
plt.show()

# Boxplots para detectar outliers
fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.ravel()
for i, var in enumerate(sample_vars):
    df.boxplot(column=var, by='target', ax=axes[i])
    axes[i].set_title(f'Boxplot de {var} por target')
plt.suptitle('')
plt.tight_layout()
plt.savefig('boxplots_outliers.png')
plt.show()

# Matriz de correlación (primeras 50 variables)
corr_matrix = df.iloc[:, :50].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False, cbar=True)
plt.title('Matriz de correlación (primeras 50 variables)')
plt.savefig('matriz_correlacion.png')
plt.show()

# Correlación con el target (top 10)
target_corr = df.iloc[:, :50].corrwith(df['target']).sort_values(ascending=False)
print("\n--- Correlación de las primeras 50 variables con target (top 10) ---")
print(target_corr.head(10))

# =============================================================================
# 4. PREPROCESAMIENTO Y DIVISIÓN DE DATOS
# =============================================================================
print("\n" + "="*60)
print("3. PREPROCESAMIENTO")
print("="*60)

# Separar features y target
X = df.drop('target', axis=1)
y = df['target']

# Dividir en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=RANDOM_STATE)
print(f"Tamaño de entrenamiento: {X_train.shape}")
print(f"Tamaño de prueba: {X_test.shape}")

# Escalar características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar SMOTE para balancear las clases en entrenamiento
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Distribución después de SMOTE: {pd.Series(y_train_resampled).value_counts().to_dict()}")

# =============================================================================
# 5. MODELADO
# =============================================================================
print("\n" + "="*60)
print("4. ENTRENAMIENTO DE MODELOS")
print("="*60)

# Diccionario para guardar resultados
results = {}
models = {}

# 5.1 Regresión Logística
print("\n--- Entrenando Regresión Logística ---")
lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_resampled, y_train_resampled)  # Usamos datos balanceados
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
roc_lr = roc_auc_score(y_test, y_proba_lr)
results['Logistic Regression'] = roc_lr
models['Logistic Regression'] = lr
print(f"ROC-AUC: {roc_lr:.4f}")

# 5.2 Random Forest
print("\n--- Entrenando Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE,
                            class_weight='balanced', n_jobs=-1)
rf.fit(X_train_scaled, y_train)  # Usamos datos originales (el modelo maneja desbalance)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]
roc_rf = roc_auc_score(y_test, y_proba_rf)
results['Random Forest'] = roc_rf
models['Random Forest'] = rf
print(f"ROC-AUC: {roc_rf:.4f}")

# 5.3 XGBoost
print("\n--- Entrenando XGBoost ---")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=scale_pos_weight,
                              random_state=RANDOM_STATE, use_label_encoder=False,
                              eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train)
y_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
roc_xgb = roc_auc_score(y_test, y_proba_xgb)
results['XGBoost'] = roc_xgb
models['XGBoost'] = xgb_model
print(f"ROC-AUC: {roc_xgb:.4f}")

# 5.4 LightGBM
print("\n--- Entrenando LightGBM ---")
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               class_weight='balanced', random_state=RANDOM_STATE,
                               verbose=-1)
lgb_model.fit(X_train_scaled, y_train)
y_proba_lgb = lgb_model.predict_proba(X_test_scaled)[:, 1]
roc_lgb = roc_auc_score(y_test, y_proba_lgb)
results['LightGBM'] = roc_lgb
models['LightGBM'] = lgb_model
print(f"ROC-AUC: {roc_lgb:.4f}")

# 5.5 Red Neuronal (MLPClassifier de scikit-learn)
print("\n--- Entrenando Red Neuronal (MLP) ---")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    solver='adam', alpha=0.001, batch_size=2048,
                    max_iter=50, random_state=RANDOM_STATE, early_stopping=True,
                    validation_fraction=0.1, verbose=False)
# Entrenamos con datos balanceados por SMOTE para mejor rendimiento
mlp.fit(X_train_resampled, y_train_resampled)
y_proba_mlp = mlp.predict_proba(X_test_scaled)[:, 1]
roc_mlp = roc_auc_score(y_test, y_proba_mlp)
results['Neural Network (MLP)'] = roc_mlp
models['Neural Network (MLP)'] = mlp
print(f"ROC-AUC: {roc_mlp:.4f}")

# =============================================================================
# 6. COMPARACIÓN DE MODELOS
# =============================================================================
print("\n" + "="*60)
print("5. COMPARACIÓN DE MODELOS")
print("="*60)

# Tabla comparativa
results_df = pd.DataFrame(list(results.items()), columns=['Modelo', 'ROC-AUC'])
results_df = results_df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
print("\n--- Rendimiento (ROC-AUC) en test ---")
print(results_df)

# Gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x='ROC-AUC', y='Modelo', data=results_df, palette='viridis')
plt.xlim(0.5, 1.0)
plt.title('Comparación de Modelos - ROC-AUC')
plt.tight_layout()
plt.savefig('comparacion_modelos.png')
plt.show()

# Curvas ROC de todos los modelos
plt.figure(figsize=(10,8))
for name, model in models.items():
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.3f})')

plt.plot([0,1], [0,1], 'k--', label='Azar')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curvas ROC')
plt.legend(loc='lower right')
plt.savefig('curvas_roc.png')
plt.show()

# Evaluación detallada del mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_model = models[best_model_name]
print(f"\n--- Evaluación detallada del mejor modelo: {best_model_name} ---")

y_pred_best = best_model.predict(X_test_scaled)
y_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_best))

# Matriz de confusión gráfica
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['No Transacción', 'Transacción'],
            yticklabels=['No Transacción', 'Transacción'])
plt.title(f'Matriz de Confusión - {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicción')
plt.savefig('matriz_confusion_mejor_modelo.png')
plt.show()

# Feature Importance si el mejor modelo lo soporta
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    plt.figure(figsize=(10,6))
    plt.barh(range(20), importances[indices], align='center')
    plt.yticks(range(20), [X.columns[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Top 20 Características más Importantes - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()
elif best_model_name == 'Neural Network (MLP)' and hasattr(best_model, 'coefs_'):
    # Para MLP podríamos mostrar la magnitud de los pesos de la primera capa
    first_layer_weights = np.abs(best_model.coefs_[0]).mean(axis=1)
    indices = np.argsort(first_layer_weights)[::-1][:20]
    plt.figure(figsize=(10,6))
    plt.barh(range(20), first_layer_weights[indices], align='center')
    plt.yticks(range(20), [X.columns[i] for i in indices])
    plt.xlabel('Importancia (pesos promedio primera capa)')
    plt.title(f'Top 20 Características - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_mlp.png')
    plt.show()

# =============================================================================
# 7. CONCLUSIONES
# =============================================================================
print("\n" + "="*60)
print("6. CONCLUSIONES")
print("="*60)

print(f"""
En este proyecto se ha desarrollado un pipeline completo de análisis predictivo
para un problema de clasificación binaria desbalanceada, similar al desafío
"Santander Customer Transaction Prediction" de Kaggle.

Principales hallazgos:
- Los datos sintéticos generados imitan la estructura del problema real con 200
  variables anónimas y un target desbalanceado (~10% de positivos).
- El análisis exploratorio reveló diferencias en las distribuciones de algunas
  variables entre clases, así como la presencia de outliers.
- Se aplicaron técnicas de balanceo (SMOTE) y se entrenaron múltiples modelos.
- El mejor modelo en términos de ROC-AUC fue {best_model_name} con un valor de {results_df.iloc[0]['ROC-AUC']:.4f}.
- Las curvas ROC y matrices de confusión muestran un rendimiento sólido,
  superando ampliamente al azar.

Este proyecto demuestra habilidades en generación de datos sintéticos, análisis
exploratorio, preprocesamiento, modelado avanzado (incluyendo redes neuronales
con scikit-learn) y evaluación comparativa. Es un excelente punto de partida
para abordar problemas reales de predicción de transacciones bancarias.
""")

print("\n¡Proyecto completado exitosamente!")
print("Revisa las imágenes PNG generadas en el directorio actual.")