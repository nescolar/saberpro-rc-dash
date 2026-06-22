# Dashboard Saber Pro — Razonamiento Cuantitativo (Atlántico, 2023)

Dashboard interactivo (Dash/Plotly) para la predicción de bajo desempeño académico en el
módulo de Razonamiento Cuantitativo del examen Saber Pro 2023, departamento del Atlántico.
Proyecto de tesis de maestría en Estadística (profundización), Universidad del Norte.

## Estructura del repositorio

```
saberpro_rc_dash/
├── app_final.py              # Dashboard interactivo (Dash) — punto de entrada
├── requirements.txt          # Dependencias del proyecto
├── assets/
│   └── portada_bg.jpg        # Imagen de fondo de la portada (cargada automáticamente por Dash)
└── data/
    ├── processed/
    │   ├── data_modelo_final.csv
    │   ├── df_mapa.csv
    │   ├── metadata_umbral.csv
    │   ├── rf_pipeline.pkl       # Pipeline entrenado — Random Forest (modelo ganador)
    │   └── xgboost_pipeline.pkl  # Pipeline entrenado — XGBoost (modelo de contraste)
    └── tables/
        └── ...                  # Métricas, curvas ROC/PR, pliegues de validación cruzada,
                                  # matrices de confusión, importancias y resultados SHAP
```

Los datos crudos del ICFES, los shapefiles geográficos (GADM) y los notebooks de
entrenamiento/EDA no forman parte de este repositorio: no son necesarios para ejecutar
el dashboard. Los shapefiles se descargan automáticamente al iniciar la aplicación
(función `descargar_gadm()`).

## Cómo ejecutar

1. Instalar dependencias (se recomienda un entorno virtual):

```
pip install -r requirements.txt
```

2. Configurar la variable de entorno `GEMINI_API_KEY` (necesaria para la pestaña del
   Asistente IA, que usa la API de Gemini):

```
export GEMINI_API_KEY="tu_api_key_aquí"
```

   Si la variable no está configurada, el resto del dashboard funciona con normalidad;
   únicamente la pestaña del Asistente IA mostrará un aviso indicando que falta la key.

3. Ejecutar el dashboard:

```
python app_final.py
```

El dashboard usa rutas relativas a la carpeta del proyecto, por lo que corre en cualquier
equipo sin modificaciones adicionales.

## Resumen metodológico

* **Imputación**: MICE para variables numéricas; moda para ordinales/nominales.
* **Modelos comparados**: Random Forest, XGBoost, Regresión Logística, Ridge, Lasso, KNN
  (y un baseline Dummy).
* **Criterio principal**: Recall sobre la clase de bajo desempeño (clase 1), para minimizar
  falsos negativos en la identificación de estudiantes en riesgo.
* **Modelo seleccionado**: Random Forest, con umbral óptimo de 0,445.
* **Validación**: prueba de Wilcoxon pareada e intervalos de confianza bootstrap
  (B = 1.000, método del percentil) para Recall₁ y AUC-ROC.
* **Interpretabilidad**: importancia por permutación y análisis SHAP (TreeExplainer).

## Asistente IA analítico

El dashboard incluye una pestaña de chatbot, integrada mediante el SDK oficial
`google-genai`, que permite realizar consultas en lenguaje natural sobre las variables,
el modelo Random Forest y los hallazgos del estudio. Requiere la variable de entorno
`GEMINI_API_KEY`.
