import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import requests
import zipfile
import os

# Función para descargar archivos
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Descargado: {filename}")

# Función para extraer zip
def extract_zip(zip_path, extract_to="."):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraído: {zip_path}")

# Descargar y cargar datos desde mi Github
print("Descargando datos de entrenamiento...")
download_file("https://raw.githubusercontent.com/OswaldGutierrez/contenidoTrain/main/train.zip", "train.zip")
extract_zip("train.zip")
df_train = pd.read_csv("train.csv")

print("Descargando datos de prueba...")
download_file("https://raw.githubusercontent.com/OswaldGutierrez/contenidoTest/main/test.zip", "test.zip")
extract_zip("test.zip")
df_test = pd.read_csv("test.csv")

print("Shape of train dataframe:", df_train.shape)
print("Shape of test dataframe:", df_test.shape)

# DIAGNÓSTICO: Verificar columnas en test set
print("\n" + "="*50)
print("DIAGNÓSTICO DE COLUMNAS EN TEST SET")
print("="*50)
print("Columnas en df_test:")
print(df_test.columns.tolist())
print(f"\nPrimeras 5 filas de df_test:")
print(df_test.head())

# Preparamos los datos
X = df_train.drop(columns=['RENDIMIENTO_GLOBAL'])
y = df_train['RENDIMIENTO_GLOBAL']

# Identificamos las columnas categóricas para usarlas en nuestro modelo: CatBoost
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Columnas categóricas encontradas: {len(categorical_features)}")
print(f"Columnas categóricas: {categorical_features}")

# Limpiamos los valores NaN (vacíos) en las columnas categóricas
print("\nLimpiando valores NaN en columnas categóricas...")
for col in categorical_features:
    # Contar NaN antes de limpiar
    nan_count = X[col].isna().sum()
    if nan_count > 0:
        print(f"  - {col}: {nan_count} valores NaN encontrados")
        X[col] = X[col].fillna('Unknown')
    else:
        print(f"  - {col}: Sin valores NaN")

# También limpiamos los valores NaN en columnas numéricas
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nLimpiando valores NaN en {len(numeric_features)} columnas numéricas...")
for col in numeric_features:
    nan_count = X[col].isna().sum()
    if nan_count > 0:
        print(f"  - {col}: {nan_count} valores NaN → rellenando con mediana")
        X[col] = X[col].fillna(X[col].median())

# Obtener índices de columnas categóricas después de la limpieza
categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

# Split de datos
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo CatBoost optimizado para velocidad
model = CatBoostClassifier(
    iterations=500,           # Reducido para velocidad
    learning_rate=0.1,        # Aumentado para convergencia más rápida
    depth=6,                  # Profundidad moderada
    cat_features=categorical_indices,  # CatBoost maneja automáticamente las categóricas
    random_seed=42,
    verbose=20,               # Mostrar progreso cada 20 iteraciones
    early_stopping_rounds=50, # Parar si no mejora en 50 iteraciones
    use_best_model=True,      # Usar el mejor modelo encontrado
    eval_metric='Accuracy',   # Métrica de evaluación
    task_type='CPU',          # Usar CPU (cambiar a 'GPU' si tienes GPU)
    bootstrap_type='Bayesian', # Más rápido que 'Bernoulli'
    od_type='Iter'            # Tipo de overfitting detection
)

# Entrenamento del modelo
print("Iniciando entrenamiento...")
model.fit(
    X_train, 
    y_train,
    eval_set=(X_val, y_val),  # Conjunto de validación para early stopping
    plot=False                # No mostrar gráficos durante entrenamiento
)

# Predicciones y evaluación
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\nAccuracy en validación: {accuracy:.4f}")
print(f"Mejor iteración: {model.best_iteration_}")
print(f"Mejor score: {model.best_score_}")

# Reporte de clasificación
print("\n" + "="*50)
print("REPORTE DE CLASIFICACIÓN")
print("="*50)
print(classification_report(y_val, y_pred))

# Importancia de características (top 10)
feature_importance = model.get_feature_importance(prettified=True)
print("\n" + "="*50)
print("TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES")
print("="*50)
print(feature_importance.head(10))

# Predicciones en test set
print("\nPreparando datos de test...")
X_test = df_test.drop(columns=['RENDIMIENTO_GLOBAL'], errors='ignore')

# Limpiar valores NaN en test set (igual que en training)
print("Limpiando valores NaN en test set...")
for col in categorical_features:
    if col in X_test.columns:
        nan_count = X_test[col].isna().sum()
        if nan_count > 0:
            print(f"  - {col}: {nan_count} valores NaN en test")
            X_test[col] = X_test[col].fillna('Unknown')

for col in numeric_features:
    if col in X_test.columns:
        nan_count = X_test[col].isna().sum()
        if nan_count > 0:
            print(f"  - {col}: {nan_count} valores NaN en test → rellenando con mediana")
            # Usar la mediana del conjunto de entrenamiento
            X_test[col] = X_test[col].fillna(X[col].median())

print("\nGenerando predicciones para el conjunto de test...")
test_predictions = model.predict(X_test)

# Asegurar que las predicciones sean un array 1D
if test_predictions.ndim > 1:
    test_predictions = test_predictions.flatten()

# SECCIÓN MEJORADA: Detectar automáticamente la columna ID correcta
print("\n" + "="*50)
print("DETECTANDO COLUMNA ID PARA KAGGLE")
print("="*50)

# Posibles nombres de columnas ID en Kaggle
possible_id_columns = ['id', 'Id', 'ID', 'index', 'Index', 'row_id', 'Row_ID']
id_column_name = None

# Buscar la columna ID en el test set
for col in possible_id_columns:
    if col in df_test.columns:
        id_column_name = col
        print(f"✅ Columna ID encontrada: '{col}'")
        break

# Si no se encuentra columna ID, crear una
if id_column_name is None:
    print("⚠️  No se encontró columna ID en el test set")
    print("Creando columna ID basada en índices...")
    id_column_name = 'ID'  # Usar 'ID' como nombre por defecto para Kaggle
    id_values = range(len(test_predictions))
else:
    id_values = df_test[id_column_name]

# Crear DataFrame con predicciones en formato Kaggle
submission = pd.DataFrame({
    id_column_name: id_values,
    'RENDIMIENTO_GLOBAL': test_predictions
})

# Verificar que no haya valores NaN en la submission
if submission.isnull().any().any():
    print("⚠️  Detectados valores NaN en submission, limpiando...")
    submission = submission.fillna(0)  # O el valor que sea apropiado

# Guardar archivo de submission para Kaggle
submission.to_csv('submission.csv', index=False)
print(f"\n✅ Archivo 'submission.csv' creado exitosamente!")
print(f"📊 Predicciones generadas: {len(test_predictions)}")
print(f"📈 Distribución de predicciones:")
print(pd.Series(test_predictions).value_counts().sort_index())

# Mostrar primeras filas del archivo de submission
print(f"\n📋 Primeras 5 filas del archivo de submission:")
print(submission.head())

# Verificar formato del archivo final
print(f"\n🔍 VERIFICACIÓN FINAL DEL FORMATO:")
print(f"   - Columnas: {list(submission.columns)}")
print(f"   - Forma: {submission.shape}")
print(f"   - Tipos de datos:")
print(submission.dtypes)

# Mostrar el archivo tal como se verá en Kaggle
print(f"\n📄 Contenido del archivo submission.csv:")
print(submission.head(10).to_string(index=False))

# Guardar modelo y otros archivos
model.save_model('catboost_model.cbm')
print("\n💾 Modelo guardado como 'catboost_model.cbm'")

# Guardar feature
feature_importance = model.get_feature_importance(prettified=True)
feature_importance.to_csv('feature_importance.csv', index=False)
print("📊 Importancia de características guardada como 'feature_importance.csv'")