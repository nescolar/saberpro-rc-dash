# Predicción del desempeño en Razonamiento Cuantitativo — Saber Pro 2023 (Atlántico)

Proyecto de maestría en Estadística. Clasificación de estudiantes en desempeño alto/bajo
en la competencia de Razonamiento Cuantitativo (Saber Pro 2023, departamento del Atlántico),
con énfasis en la identificación de estudiantes en riesgo.

## Estructura

```
saberpro_rc_dash/
├── EDA_corregido_25_04.ipynb        # Análisis exploratorio de datos
├── Modelos_corregido_25_04.ipynb    # Entrenamiento y comparación de los 6 modelos
├── Experimento_GridSearchCV.ipynb   # Variante experimental del tuning (GridSearchCV)
├── Modelos/                         # Scripts auxiliares (preprocess, train, evaluate, predict)
├── app/
│   └── app_estetico.py              # Dashboard interactivo (Dash)
├── data/
│   ├── raw/                         # Datos originales ICFES (2023-1 y 2023-2)
│   ├── processed/                   # Datos procesados y pipeline entrenado (.pkl)
│   ├── tables/                      # Tablas de resultados (métricas, pliegues, ROC, PR…)
│   ├── figures/                     # Figuras de error por pliegue
│   └── geo/                         # Shapefiles GADM para los mapas
├── figures/                         # Figuras del EDA y de los modelos
├── tables/                          # Tabla descriptiva
└── requirements.txt                 # Dependencias del proyecto
```

En la raíz del proyecto, la carpeta `mapa_shp/` contiene los límites municipales
(IGAC) que utiliza el notebook de EDA.

## Cómo ejecutar

1. Instalar dependencias (se recomienda un entorno virtual):

   ```bash
   pip install -r requirements.txt
   ```

2. **Notebooks**: abrir con Jupyter en el orden EDA → Modelos.
   Nota: las rutas de lectura de datos en los notebooks son absolutas
   (`/Users/nataliangaritaescolar/Desktop/Trabajo_final/...`). Para ejecutarlos en
   otro equipo, ubicar la carpeta en esa misma ruta o ajustar la ruta base al inicio
   de cada notebook.

3. **Dashboard**:

   ```bash
   python app/app_estetico.py
   ```

   El dashboard usa rutas relativas a la carpeta del proyecto, por lo que corre
   en cualquier equipo sin modificaciones.

## Resumen metodológico

- Imputación: MICE para variables numéricas; moda para ordinales/nominales.
- Modelos comparados: Random Forest, Regresión Logística, Ridge, Lasso, XGBoost y KNN.
- Criterio principal: Recall sobre la clase de bajo desempeño.
- Modelo seleccionado: **Random Forest**, con umbral óptimo de 0,445 (maximización de F1).
- Validación: prueba de Wilcoxon pareada e intervalos bootstrap (B = 1.000, percentil).
- Interpretabilidad: análisis SHAP (TreeExplainer).
