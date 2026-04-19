# 🎓 Predicción de Rendimiento Académico — CatBoost

Modelo de Machine Learning que clasifica el **rendimiento global de estudiantes** a partir de variables académicas y socioeconómicas. Desarrollado para una competencia en **Kaggle**, utilizando el algoritmo **CatBoost** con manejo automático de variables categóricas, limpieza de datos y early stopping.

---

## 🛠️ Tecnologías utilizadas

| Herramienta | Uso |
|-------------|-----|
| Python | Lenguaje principal |
| CatBoost | Modelo de clasificación |
| pandas + NumPy | Carga y limpieza de datos |
| scikit-learn | Evaluación y split de datos |

---

## ✅ Características del modelo

- Descarga automática del dataset desde GitHub (train/test en `.zip`)
- Limpieza de valores nulos en columnas categóricas y numéricas
- Manejo nativo de variables categóricas con CatBoost (sin necesidad de encoding manual)
- Early stopping para evitar sobreajuste (para si no mejora en 50 iteraciones)
- Reporte de accuracy, F1-score y matriz de confusión
- Genera automáticamente `submission.csv` en el formato requerido por Kaggle
- Guarda el modelo entrenado como `catboost_model.cbm` para reutilizarlo
- Exporta la importancia de características a `feature_importance.csv`

---

## 📊 Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy en validación | Ver salida del script |
| Mejor iteración | Determinada por early stopping |
| Variables más importantes | Ver `feature_importance.csv` |

---


## 📚 Lo que aprendí

Este proyecto me introdujo al flujo completo de una competencia en Kaggle: desde la descarga y limpieza del dataset hasta la generación del archivo de submission. Aprendí a usar CatBoost, un algoritmo de gradient boosting que maneja variables categóricas de forma nativa sin necesidad de codificarlas manualmente, lo cual simplifica mucho el preprocesamiento. También implementé técnicas como early stopping y uso del mejor modelo encontrado durante el entrenamiento, que ayudan a evitar el sobreajuste. Analizar la importancia de características me permitió entender qué variables tienen mayor influencia sobre el rendimiento académico.

---

## 👤 Autor

**Oswald David Gutiérrez Cortina**  
Estudiante de Ingeniería — Universidad  
[LinkedIn](https://www.linkedin.com/in/oswald-david-gutierrez-1a452939a) · [GitHub](https://github.com/OswaldGutierrez)
