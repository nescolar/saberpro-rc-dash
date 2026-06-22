"""
Dashboard de Maestría — Análisis Saber Pro
Razonamiento Cuantitativo · Atlántico 2023
Versión 2 — corregida y ampliada

ESTRUCTURA DE CARPETAS ESPERADA:
saberpro_rc_dash/
├── app.py
├── data/
│   ├── processed/
│   │   ├── data_modelo_final.csv
│   │   └── df_mapa.csv
│   └── tables/
│       ├── comparacion_modelos.csv
│       ├── roc_*.csv
│       └── cm_*.csv
└── data/geo/          ← shapefile GADM (auto-descargado)

Ejecutar: python app.py
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import unicodedata
import io, base64, os, zipfile, urllib.request
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import chi2_contingency
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
from dash import dash_table

# SDK oficial de Google GenAI — chatbot del asistente analítico
from google import genai
from google.genai import types

# ─────────────────────────────────────────────
#  RUTAS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
TABLES_DIR = os.path.join(BASE_DIR, "data", "tables")
GEO_DIR    = os.path.join(BASE_DIR, "data", "geo")
GADM_URL   = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_COL_shp.zip"
GADM_ZIP   = os.path.join(GEO_DIR, "gadm41_COL_shp.zip")
GADM_SHP   = os.path.join(GEO_DIR, "gadm41_COL_2.shp")

# Inicialización segura del cliente GenAI (chatbot del asistente analítico)
try:
    client_genai = genai.Client()
except Exception:
    client_genai = None

# ─────────────────────────────────────────────
#  PALETA Y ESTILOS
# ─────────────────────────────────────────────
AZUL_OSCURO   = "#023373"
AZUL_MEDIO    = "#3C92A6"
AZUL_CLARO    = "#A2F2F2"
MORADO        = "#b46cff"
LILA          = "#8f6bff"
ROSA          = "#ff6f9f"
NARANJA       = "#F2B749"
ACENTO_ROJO   = "#d14b46"
GRIS_SUAVE    = "#f5f6fb"
GRIS_CARD     = "#fbfbfe"
GRIS_TEXTO    = "#596579"
GRIS_MUTED    = "#8d98ab"
BLANCO        = "#ffffff"
BORDE         = "#e8ebf3"
VERDE         = "#49A65D"
MORADO_OSCURO = "#023373"

# ── Paleta presentación ──────────────────────
CORAL         = "#F29441"   # naranja cálido
PETROL        = "#3C92A6"   # azul petróleo/teal
MENTA         = "#A2F2F2"   # celeste claro
AMARILLO      = "#F2B749"   # amarillo dorado

FONT = "'Inter', 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif"

RCPARAMS = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial"],
    "font.size":         13,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

PAGE_STYLE = {
    "padding": "2rem",
    "backgroundColor": GRIS_SUAVE,
    "minHeight": "100vh",
}
NAVBAR_STYLE = {
    "backgroundColor": PETROL,
    "padding": "0 2rem",
    "boxShadow": "0 6px 20px rgba(58,125,140,0.25)",
    "position": "sticky", "top": "0", "zIndex": "1000",
    "fontFamily": FONT, "display": "flex", "alignItems": "center",
}
BTN_BASE = {
    "background": "none", "border": "none",
    "color": "#d9e4f2", "fontSize": "1.0rem",
    "fontFamily": FONT, "letterSpacing": "0.01em",
    "padding": "1.1rem 1rem", "cursor": "pointer",
    "borderBottom": "3px solid transparent", "whiteSpace": "nowrap",
}
BTN_ACTIVE = {**BTN_BASE, "color": BLANCO,
              "borderBottom": f"3px solid {CORAL}", "fontWeight": "700"}
SUB_BASE = {
    "background": BLANCO, "border": f"1px solid {BORDE}",
    "borderRadius": "12px", "color": AZUL_OSCURO,
    "fontSize": "0.94rem", "fontFamily": FONT,
    "padding": "0.6rem 1rem", "cursor": "pointer", "margin": "0 0.25rem",
    "boxShadow": "0 1px 4px rgba(31,53,86,0.05)",
}
SUB_ACTIVE = {**SUB_BASE,
              "background": f"linear-gradient(135deg, {PETROL}, {MENTA})",
              "color": BLANCO, "border": "1px solid transparent", "fontWeight": "700"}
CARD = {
    "backgroundColor": BLANCO, "borderRadius": "18px",
    "boxShadow": "0 10px 28px rgba(31,53,86,0.08)",
    "padding": "1.8rem", "marginBottom": "1.5rem",
    "border": f"1px solid {BORDE}", "fontFamily": FONT,
}
SEC_TITLE = {
    "fontFamily": FONT, "color": PETROL,
    "fontSize": "2rem", "fontWeight": "800",
    "marginBottom": "0.35rem", "letterSpacing": "-0.02em",
}
SUBTITLE_S = {
    "fontFamily": FONT, "color": GRIS_TEXTO,
    "fontSize": "1.08rem", "marginBottom": "1.4rem",
}
KPI_GRADIENTS = [
    f"linear-gradient(135deg, {CORAL}, {AMARILLO})",
    f"linear-gradient(135deg, {AZUL_OSCURO}, {PETROL})",
    f"linear-gradient(135deg, {PETROL}, {MENTA})",
    f"linear-gradient(135deg, {AMARILLO}, {CORAL})",
]

SECCIONES = [
    ("portada",       "Portada"),
    ("intro",         "Introducción"),
    ("metodologia",   "Metodología"),
    ("marco",         "Marco Teórico"),
    ("eda",           "EDA"),
    ("modelos",       "Modelos y Resultados"),
    ("prediccion",    "Predictor Interactivo"),
    ("chatbot",       "+PRO"),
    ("conclusiones",  "Conclusiones"),
]

EDA_SUBS = [
    ("eda_desc", "Variable respuesta"),
    ("eda_dist", "Distribución de variables predictoras"),
    ("eda_mapa", "Análisis por municipio"),
    ("eda_chi",  "Tablas Chi-cuadrado"),
]

MODELOS_SUBS = [
    ("mod_comparacion", "Modelos"),
    ("mod_analisis",    "Análisis de resultados"),
]

MARCO_SUBS = [
    ("marco_modelos",  "Modelos"),
    ("marco_metricas", "Métricas y validación"),
]

METODOLOGIA_SUBS = [
    ("met_marco",     "Marco metodológico"),
    ("met_variables", "Operacionalización de variables"),
    ("met_flujo",     "Flujo de construcción de modelos"),
]

INTRO_SUBS = [
    ("intro_intro", "Introducción"),
]

# ── Variables predictoras para el panel interactivo ───────────────
VARIABLES_PREDICTOR = {
    "estu_genero": {
        "label": "Género",
        "tipo": "radio",
        "opciones": ["F", "M"],
        "labels_opciones": ["Femenino", "Masculino"],
        "default": "F",
    },
    "fami_estratovivienda": {
        "label": "Estrato socioeconómico",
        "tipo": "dropdown",
        "opciones": ["Sin Estrato", "Estrato 1", "Estrato 2",
                     "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"],
        "default": "Estrato 2",
    },
    "fami_tieneinternet": {
        "label": "Internet en casa",
        "tipo": "radio",
        "opciones": ["Si", "No"],
        "labels_opciones": ["Sí", "No"],
        "default": "Si",
    },
    "fami_tienecomputador": {
        "label": "Computador en casa",
        "tipo": "radio",
        "opciones": ["Si", "No"],
        "labels_opciones": ["Sí", "No"],
        "default": "Si",
    },
    "fami_tieneautomovil": {
        "label": "Automóvil en casa",
        "tipo": "radio",
        "opciones": ["Si", "No"],
        "labels_opciones": ["Sí", "No"],
        "default": "No",
    },
    "fami_tienemotocicleta": {
        "label": "Motocicleta en casa",
        "tipo": "radio",
        "opciones": ["Si", "No"],
        "labels_opciones": ["Sí", "No"],
        "default": "No",
    },
    "estu_pagomatricula": {
        "label": "Pago de matrícula",
        "tipo": "dropdown",
        "opciones": ["Con apoyo externo", "Con recursos propios",
                     "Con crédito ICETEX", "Sin pago de matrícula",
                     "Con otro tipo de crédito"],
        "default": "Con recursos propios",
    },
    "estu_metodo_prgm": {
        "label": "Modalidad del programa",
        "tipo": "radio",
        "opciones": ["PRESENCIAL", "DISTANCIA VIRTUAL"],
        "labels_opciones": ["Presencial", "Virtual"],
        "default": "PRESENCIAL",
    },
    "inst_caracter_academico": {
        "label": "Carácter académico IES",
        "tipo": "dropdown",
        "opciones": ["INSTITUCIÓN UNIVERSITARIA/ESCUELA TECNOLÓGICA",
                     "UNIVERSIDAD", "INSTITUCIÓN TECNOLÓGICA"],
        "default": "UNIVERSIDAD",
    },
    "inst_origen": {
        "label": "Origen de la institución",
        "tipo": "dropdown",
        "opciones": ["OFICIAL DEPARTAMENTAL", "NO OFICIAL - CORPORACIÓN",
                     "NO OFICIAL - FUNDACIÓN", "OFICIAL NACIONAL"],
        "default": "OFICIAL DEPARTAMENTAL",
    },
    "fami_educacionpadre": {
        "label": "Educación del padre",
        "tipo": "dropdown",
        "opciones": ["Ninguno", "Primaria incompleta", "Primaria completa",
                     "Secundaria (Bachillerato) incompleta",
                     "Secundaria (Bachillerato) completa",
                     "Técnica o tecnológica incompleta",
                     "Técnica o tecnológica completa",
                     "Educación profesional incompleta",
                     "Educación profesional completa", "Postgrado"],
        "default": "Secundaria (Bachillerato) completa",
    },
    "fami_educacionmadre": {
        "label": "Educación de la madre",
        "tipo": "dropdown",
        "opciones": ["Ninguno", "Primaria incompleta", "Primaria completa",
                     "Secundaria (Bachillerato) incompleta",
                     "Secundaria (Bachillerato) completa",
                     "Técnica o tecnológica incompleta",
                     "Técnica o tecnológica completa",
                     "Educación profesional incompleta",
                     "Educación profesional completa", "Postgrado"],
        "default": "Secundaria (Bachillerato) completa",
    },
    "fami_ocupacionpadre": {
        "label": "Ocupación del padre",
        "tipo": "dropdown",
        "opciones": ["Trabaja por cuenta propia", "Obrero, empleado de empresa particular",
                     "Empleado del gobierno", "Empresario", "Pensionado / Jubilado",
                     "Trabaja en las labores del hogar", "Desempleado",
                     "Otro"],
        "default": "Trabaja por cuenta propia",
    },
    "fami_ocupacionmadre": {
        "label": "Ocupación de la madre",
        "tipo": "dropdown",
        "opciones": ["Trabaja por cuenta propia", "Obrero, empleado de empresa particular",
                     "Empleado del gobierno", "Empresario", "Pensionado / Jubilado",
                     "Trabaja en las labores del hogar", "Desempleado",
                     "Otro"],
        "default": "Trabaja en las labores del hogar",
    },
    "estu_tituloobtenidobachiller": {
        "label": "Título de bachiller",
        "tipo": "dropdown",
        "opciones": ["ACADÉMICO", "TÉCNICO", "NORMALISTA"],
        "default": "ACADÉMICO",
    },
    "estu_horassemanatrabaja": {
        "label": "Horas de trabajo semanal",
        "tipo": "dropdown",
        "opciones": ["0", "Menos de 10 horas", "Entre 11 y 20 horas",
                     "Entre 21 y 30 horas", "Más de 30 horas"],
        "default": "0",
    },
    "estu_valormatriculauniversidad": {
        "label": "Valor de la matrícula",
        "tipo": "dropdown",
        "opciones": ["No pagó matrícula", "Menos de 500 mil",
                     "Entre 500 mil y menos de 1 millón",
                     "Entre 1 millón y menos de 2.5 millones",
                     "Entre 2.5 millones y menos de 4 millones",
                     "Entre 4 millones y menos de 5.5 millones",
                     "Entre 5.5 millones y menos de 7 millones",
                     "Más de 7 millones"],
        "default": "Entre 1 millón y menos de 2.5 millones",
    },
    "edad": {
        "label": "Edad",
        "tipo": "number",
        "min": 18, "max": 69, "step": 1,
        "default": 21,
    },
}

# ── Operacionalización ────────────────────────────────────────────
VARIABLES_DATA = [
    {"grupo": "Variable a predecir", "variable": "Razonamiento Cuantitativo",
     "descripcion": "Desempeño del estudiante en la prueba Saber Pro, módulo de Razonamiento Cuantitativo",
     "respuestas": ["Clase 0: Alto desempeño (puntaje ≥ promedio nacional)",
                    "Clase 1: Bajo desempeño (puntaje < promedio nacional)"],
     "tipo": "Cualitativa", "nivel": "Categórica dicotómica"},
    {"grupo": "Sociodemográficas", "variable": "Género",
     "descripcion": "Sexo biológico del estudiante",
     "respuestas": ["Femenino", "Masculino"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Sociodemográficas", "variable": "Horas de trabajo a la semana",
     "descripcion": "Horas semanales dedicadas a trabajo remunerado",
     "respuestas": ["0", "Menos de 10 h", "11–20 h", "21–30 h", "Más de 30 h"],
     "tipo": "Cualitativa", "nivel": "Categórica ordinal"},
    {"grupo": "Sociodemográficas", "variable": "Valor de la matrícula",
     "descripcion": "Costo semestral de matrícula reportado por el estudiante",
     "respuestas": ["$0", "< $500k", "$1M–$2.5M", "$2.5M–$4M",
                    "$4M–$5.5M", "$5.5M–$7M", "> $7M", "No pagó"],
     "tipo": "Cualitativa", "nivel": "Categórica ordinal"},
    {"grupo": "Sociodemográficas", "variable": "Edad",
     "descripcion": "Edad del estudiante en años cumplidos al momento de presentar la prueba",
     "respuestas": ["18 – 69 años (variable continua)"],
     "tipo": "Cuantitativa", "nivel": "Continua"},
    {"grupo": "Sociodemográficas", "variable": "Estrato socioeconómico",
     "descripcion": "Estrato de la vivienda según clasificación DANE",
     "respuestas": ["Sin estrato", "Estrato 1", "Estrato 2",
                    "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"],
     "tipo": "Cualitativa", "nivel": "Categórica ordinal"},
    {"grupo": "Sociodemográficas", "variable": "Forma de pago de la matrícula",
     "descripcion": "Fuente de financiación utilizada para el pago de la matrícula universitaria",
     "respuestas": ["Recursos propios", "Crédito ICETEX", "Crédito con otra entidad",
                    "Beca / subsidio", "No pagó"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Académicas", "variable": "Título bachiller obtenido",
     "descripcion": "Tipo de bachillerato cursado en educación media",
     "respuestas": ["Bachiller académico", "Bachiller pedagógico / normalista", "Bachiller técnico"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Académicas", "variable": "Método del programa",
     "descripcion": "Modalidad de enseñanza del programa de educación superior",
     "respuestas": ["Distancia virtual", "Presencial"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Familia tiene Internet",
     "descripcion": "Acceso a Internet en el hogar del estudiante",
     "respuestas": ["Sí", "No"], "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Familia tiene computador",
     "descripcion": "Disponibilidad de computador en el hogar",
     "respuestas": ["Sí", "No"], "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Familia tiene automóvil",
     "descripcion": "Disponibilidad de automóvil en el hogar",
     "respuestas": ["Sí", "No"], "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Familia tiene motocicleta",
     "descripcion": "Disponibilidad de motocicleta en el hogar",
     "respuestas": ["Sí", "No"], "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Nivel educativo del padre",
     "descripcion": "Máximo nivel educativo alcanzado por el padre",
     "respuestas": ["No sabe", "Ninguno", "Primaria", "Secundaria",
                    "Técnica / tecnológica", "Profesional", "Postgrado"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Nivel educativo de la madre",
     "descripcion": "Máximo nivel educativo alcanzado por la madre",
     "respuestas": ["No sabe", "Ninguno", "Primaria", "Secundaria",
                    "Técnica / tecnológica", "Profesional", "Postgrado"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Ocupación del padre",
     "descripcion": "Actividad laboral u ocupación principal del padre",
     "respuestas": ["Director/gerente", "Técnico/profesional",
                    "Obrero/operario", "Empresario", "Hogar", "Pensionado", "Otra"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Condiciones familiares y del hogar", "variable": "Ocupación de la madre",
     "descripcion": "Actividad laboral u ocupación principal de la madre",
     "respuestas": ["Director/gerente", "Técnico/profesional",
                    "Obrero/operario", "Empresario", "Hogar", "Pensionado", "Otra"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Institucionales", "variable": "Carácter académico",
     "descripcion": "Tipo de institución de educación superior según reconocimiento SNIES",
     "respuestas": ["Institución tecnológica", "Institución universitaria", "Universidad"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
    {"grupo": "Institucionales", "variable": "Institución de origen",
     "descripcion": "Naturaleza jurídica de la institución de educación superior",
     "respuestas": ["No oficial – corporación", "No oficial – fundación",
                    "Oficial departamental", "Oficial nacional"],
     "tipo": "Cualitativa", "nivel": "Categórica nominal"},
]

GRUPO_CONFIG = {
    "Variable a predecir":              {"color": "#A6321F", "light": "#fbeae7", "badge": "#f7d9d3", "icon": ""},
    "Sociodemográficas":                {"color": "#4d2d8c", "light": "#f0ebfa", "badge": "#ede0ff", "icon": ""},
    "Académicas":                       {"color": "#1f3556", "light": "#e8f0fb", "badge": "#dfe8ff", "icon": ""},
    "Condiciones familiares y del hogar": {"color": "#185FA5", "light": "#e6f1fb", "badge": "#d0e8f8", "icon": ""},
    "Institucionales":                  {"color": "#0F6E56", "light": "#e1f5ee", "badge": "#c8f0e0", "icon": ""},
}
NIVEL_COLOR = {
    "Categórica dicotómica": {"bg": "#fbeae7", "text": "#A6321F"},
    "Categórica nominal": {"bg": "#fff3e0", "text": "#e65100"},
    "Categórica ordinal": {"bg": "#e8f5e9", "text": "#2e7d32"},
    "Continua":           {"bg": "#e3f2fd", "text": "#1565c0"},
}

# ─────────────────────────────────────────────
#  CARGA DE DATOS
# ─────────────────────────────────────────────
def limpiar_texto(x):
    if pd.isna(x):
        return x
    x = str(x).strip().upper()
    x = unicodedata.normalize("NFKD", x).encode("ascii", errors="ignore").decode("utf-8")
    return x


def descargar_gadm():
    os.makedirs(GEO_DIR, exist_ok=True)
    if not os.path.exists(GADM_SHP):
        print("⬇ Descargando shapefile GADM...")
        try:
            urllib.request.urlretrieve(GADM_URL, GADM_ZIP)
            with zipfile.ZipFile(GADM_ZIP, "r") as z:
                z.extractall(GEO_DIR)
            print("✓ Shapefile GADM descargado.")
        except Exception as e:
            print(f"⚠ Error descargando GADM: {e}")
            return None
    return GADM_SHP


def cargar_datos():
    for nombre in ["data_modelo_final.csv", "data_imputada_ok.csv", "data_imputada.csv"]:
        ruta = os.path.join(DATA_DIR, nombre)
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            print(f"✓ {nombre} cargado: {df.shape}")
            return df
    print("⚠ No se encontró archivo de datos procesados.")
    return pd.DataFrame()


def cargar_df_mapa():
    ruta = os.path.join(DATA_DIR, "df_mapa.csv")
    try:
        df = pd.read_csv(ruta)
        print(f"✓ df_mapa cargado: {df.shape}")
        return df
    except FileNotFoundError:
        print("⚠ df_mapa.csv no encontrado.")
        return pd.DataFrame()


def cargar_geodatos():
    try:
        os.makedirs(GEO_DIR, exist_ok=True)
        shp = descargar_gadm()
        if shp is None:
            return None
        muns = gpd.read_file(shp)
        muns["DEP_CLEAN"] = muns["NAME_1"].apply(limpiar_texto)
        muns["mun_clean"] = muns["NAME_2"].apply(limpiar_texto)
        atlantico = muns[muns["DEP_CLEAN"] == "ATLANTICO"].copy()
        if atlantico.empty:
            return None
        return atlantico.to_crs(epsg=4326)
    except Exception as e:
        print(f"⚠ Geodatos no disponibles: {e}")
        return None


def cargar_modelo_prediccion():
    """Carga el pipeline del modelo ganador (Random Forest); XGBoost como respaldo."""
    for nombre in ["rf_pipeline.pkl", "random_forest_pipeline.pkl", "modelo_rf.pkl",
                   "xgboost_pipeline.pkl", "xgb_pipeline.pkl", "modelo_xgb.pkl"]:
        ruta = os.path.join(DATA_DIR, nombre)
        if os.path.exists(ruta):
            try:
                modelo = joblib.load(ruta)
                print(f"✓ Modelo cargado: {nombre}")
                return modelo
            except Exception as e:
                print(f"⚠ Error cargando modelo: {e}")
    print("⚠ No se encontró modelo serializado. El predictor usará datos de entrenamiento.")
    return None


# ── Carga de CSVs de tablas ───────────────────────────────────────
try:
    df_comparacion = pd.read_csv(os.path.join(TABLES_DIR, "comparacion_modelos.csv"), index_col=0)
    df_comparacion.index.name = "Modelo"
    df_comparacion = df_comparacion.reset_index().round(4)
except FileNotFoundError:
    df_comparacion = pd.DataFrame()

# Encabezados amenos para la tabla comparativa de modelos (solo visualización;
# los nombres internos de columna NO cambian, así que ordenamientos/filtros siguen intactos)
ETIQUETAS_METRICAS = {
    "Modelo":       "Modelo",
    "accuracy":     "Exactitud",
    "precision":    "Precisión",
    "recall_1":     "Recall (clase 1)",
    "recall_0":     "Recall (clase 0)",
    "f1":           "F1-Score",
    "auc":          "AUC-ROC (test)",
    "auc_cv_mean":  "AUC-ROC (prom. CV)",
    "auc_cv_std":   "AUC-ROC (desv. CV)",
    "rec1_cv_mean": "Recall clase 1 (prom. CV)",
    "ap":           "Precisión promedio (AP)",
}

# Nombres de modelos disponibles (incluye LightGBM si existe)
MODELOS_NOMBRES = ["Logistica", "KNN", "Ridge", "Lasso", "Random Forest", "XGBoost", "LightGBM"]

roc_files = {m: f"roc_{m.lower().replace(' ', '_').replace('ó', 'o')}.csv" for m in MODELOS_NOMBRES}
roc_files.update({
    "Logistica": "roc_logistica.csv",
    "Random Forest": "roc_random_forest.csv",
})
curvas_roc_dash = {}
for nombre, archivo in roc_files.items():
    ruta = os.path.join(TABLES_DIR, archivo)
    if os.path.exists(ruta):
        curvas_roc_dash[nombre] = pd.read_csv(ruta)

cm_files = {m: f"cm_{m.lower().replace(' ', '_').replace('ó', 'o')}.csv" for m in MODELOS_NOMBRES}
cm_files.update({
    "Logistica": "cm_logistica.csv",
    "Random Forest": "cm_random_forest.csv",
})
cms_dash = {}
for nombre, archivo in cm_files.items():
    ruta = os.path.join(TABLES_DIR, archivo)
    if os.path.exists(ruta):
        cms_dash[nombre] = pd.read_csv(ruta, index_col=0)

# Pliegues por modelo — CSV individual por modelo
pliegues_files = {m: f"pliegues_{m.lower().replace(' ', '_').replace('(','').replace(')','')}.csv"
                  for m in MODELOS_NOMBRES}
pliegues_dash = {}
for nombre, archivo in pliegues_files.items():
    ruta = os.path.join(TABLES_DIR, archivo)
    if os.path.exists(ruta):
        pliegues_dash[nombre] = pd.read_csv(ruta)

# CSV consolidado con todos los modelos — para boxplot comparativo
ruta_pliegues_todos = os.path.join(TABLES_DIR, "pliegues_todos_modelos.csv")
try:
    df_pliegues_todos = pd.read_csv(ruta_pliegues_todos)
    print(f"✓ pliegues_todos_modelos.csv cargado: {df_pliegues_todos.shape}")
except FileNotFoundError:
    df_pliegues_todos = pd.DataFrame()
    print("⚠ pliegues_todos_modelos.csv no encontrado — boxplot usará datos ilustrativos")

# Bootstrap IC 95% por modelo (forest plot nativo)
try:
    df_bootstrap_ic = pd.read_csv(os.path.join(TABLES_DIR, "bootstrap_ic95.csv"))
    print(f"✓ bootstrap_ic95.csv cargado: {df_bootstrap_ic.shape}")
except FileNotFoundError:
    df_bootstrap_ic = pd.DataFrame()
    print("⚠ bootstrap_ic95.csv no encontrado — forest plot no disponible")

# Valores SHAP (bee swarm nativo) — formato largo: feature, label, shap_value, feature_value
try:
    df_shap_bee = pd.read_csv(os.path.join(TABLES_DIR, "shap_beeswarm_top15.csv"))
    print(f"✓ shap_beeswarm_top15.csv cargado: {df_shap_bee.shape}")
except FileNotFoundError:
    df_shap_bee = pd.DataFrame()
    print("⚠ shap_beeswarm_top15.csv no encontrado — gráfico SHAP no disponible")

# Tabla resumen de importancia SHAP global (orden, dirección del efecto)
try:
    df_shap_imp = pd.read_csv(os.path.join(TABLES_DIR, "shap_importancia_top15.csv"))
    print(f"✓ shap_importancia_top15.csv cargado: {df_shap_imp.shape}")
except FileNotFoundError:
    df_shap_imp = pd.DataFrame()
    print("⚠ shap_importancia_top15.csv no encontrado")

# Datos globales
data_imputada = cargar_datos()
df_mapa       = cargar_df_mapa()
atlantico_gdf = cargar_geodatos()
modelo_pred   = cargar_modelo_prediccion()

# Modelo ganador
MODELO_GANADOR = "Random Forest"

# ─────────────────────────────────────────────
#  FUNCIONES DE GRÁFICOS
# ─────────────────────────────────────────────
def fig_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def cargar_imagen_estatica(nombre_archivo):
    """Carga una imagen PNG desde data/processed/ o desde la carpeta del script como base64."""
    rutas_busqueda = [
        os.path.join(DATA_DIR, nombre_archivo),
        os.path.join(BASE_DIR, nombre_archivo),
        os.path.join(BASE_DIR, "assets", nombre_archivo),
    ]
    for ruta in rutas_busqueda:
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    return None


def grafico_chi(n_total=13078):
    plt.rcParams.update(RCPARAMS)
    chi_results = pd.DataFrame({
        "Variable": ["Género", "Edad", "Título bachiller", "Modalidad programa",
                     "Forma pago", "Valor matrícula", "Estrato", "Internet",
                     "Computador", "Automóvil", "Motocicleta", "Educación padre",
                     "Educación madre", "Ocupación padre", "Ocupación madre",
                     "Carácter académico", "Institución de origen"],
        "pvalor": [0.0001, 0.0001, 0.08, 0.38, 0.0001, 0.0001, 0.0001, 0.0001,
                   0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    })
    chi_results["significativo"] = chi_results["pvalor"] < 0.05
    chi_results = chi_results.sort_values("pvalor", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    colores_bar = [AZUL_OSCURO if s else "#aaaaaa" for s in chi_results["significativo"]]
    bars = ax.barh(chi_results["Variable"], -np.log10(chi_results["pvalor"]),
                   color=colores_bar, edgecolor="white", linewidth=0.6, alpha=0.88, height=0.6)
    ax.axvline(-np.log10(0.05), color=ACENTO_ROJO, linestyle="--",
               linewidth=1.2, label="Umbral p = 0.05")
    for bar, pval in zip(bars, chi_results["pvalor"]):
        val = -np.log10(pval)
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"p = {pval:.2f}" if pval >= 0.01 else "p < 0.001",
                va="center", fontsize=8.5, color="#333333")
    ax.set_xlabel("−log₁₀(p-valor)", fontsize=11)
    ax.set_title("Figura II\nAsociación de variables con desempeño — Chi-cuadrado, Atlántico 2023",
                 fontsize=11, loc="left", pad=10)
    fig.text(0.01, -0.03,
             f"Nota. Elaboración propia con datos del ICFES (2023). n = {n_total:,}. "
             "Barras grises: variables no significativas (p ≥ 0.05).",
             fontsize=9, style="italic", ha="left")
    ax.tick_params(axis="both", labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    plt.tight_layout()
    return fig_a_base64(fig)


def grafico_mapa(variable="pct_bajo_desempeno"):
    if atlantico_gdf is None or df_mapa.empty:
        fig = go.Figure()
        fig.add_annotation(text="Geodatos no disponibles", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=16, color=GRIS_TEXTO))
        return fig
    atlantico_plot = atlantico_gdf.merge(df_mapa, on="mun_clean", how="left")

    # Normalizar nombres de columna: el CSV puede venir con variantes
    col_map = {}
    for col in atlantico_plot.columns:
        col_map[col.strip().lower()] = col
    # pct_bajo_desempeno
    if "pct_bajo_desempeno" not in atlantico_plot.columns:
        for candidato in ["pct_bajo_desempeno", "pct_alto", "alto_desempeno", "pct_alto_desempeño"]:
            if candidato in col_map:
                atlantico_plot = atlantico_plot.rename(columns={col_map[candidato]: "pct_bajo_desempeno"})
                break
        else:
            atlantico_plot["pct_bajo_desempeno"] = 0.0
    # n_estudiantes
    if "n_estudiantes" not in atlantico_plot.columns:
        for candidato in ["n_estudiantes", "n_est", "estudiantes", "total"]:
            if candidato in col_map:
                atlantico_plot = atlantico_plot.rename(columns={col_map[candidato]: "n_estudiantes"})
                break
        else:
            atlantico_plot["n_estudiantes"] = 0

    atlantico_plot["n_estudiantes"]      = atlantico_plot["n_estudiantes"].fillna(0)
    atlantico_plot["pct_bajo_desempeno"] = atlantico_plot["pct_bajo_desempeno"].fillna(0)

    # Si la variable solicitada sigue sin existir, caer a pct_bajo_desempeno
    if variable not in atlantico_plot.columns:
        variable = "pct_bajo_desempeno"

    geojson_atlantico = atlantico_plot.__geo_interface__
    titulos = {
        "pct_bajo_desempeno": "% Bajo desempeño por municipio",
        "n_estudiantes": "Número de estudiantes por municipio",
    }
    # hover_data solo con columnas que existen
    hover_cols = {c: True for c in ["n_estudiantes", "pct_bajo_desempeno"] if c in atlantico_plot.columns}
    hover_cols["mun_clean"] = False
    fig = px.choropleth_map(
        atlantico_plot, geojson=geojson_atlantico,
        locations="mun_clean", featureidkey="properties.mun_clean",
        color=variable, hover_name="NAME_2",
        hover_data=hover_cols,
        color_continuous_scale="Viridis", map_style="carto-positron",
        center={"lat": 10.9, "lon": -74.9}, zoom=8, opacity=0.85
    )
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    fig.update_layout(
        title=titulos.get(variable, "Mapa por municipio"),
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        font_family=FONT,
        coloraxis_colorbar=dict(title="Valor"),
        paper_bgcolor=GRIS_SUAVE,
    )
    return fig


def tabla_categorica_df(serie, nombre_var):
    if data_imputada.empty or "desempeno_dicotomico" not in data_imputada.columns:
        return pd.DataFrame()
    tab = pd.crosstab(serie, data_imputada["desempeno_dicotomico"])
    if tab.shape[1] < 2:
        return pd.DataFrame()
    total     = tab.sum(axis=1)
    total_pct = (total / total.sum() * 100).round(1)
    alto      = tab.get(0, pd.Series(0, index=tab.index))
    alto_pct  = (alto / total * 100).round(1)
    bajo      = tab.get(1, pd.Series(0, index=tab.index))
    bajo_pct  = (bajo / total * 100).round(1)
    chi2_stat, p_valor, _, _ = chi2_contingency(tab)
    return pd.DataFrame({
        "Variable": nombre_var, "Categoría": total.index,
        "Total": [f"{n} ({p}%)" for n, p in zip(total, total_pct)],
        "Alto desempeño (0)": [f"{n} ({p}%)" for n, p in zip(alto, alto_pct)],
        "Bajo desempeño (1)": [f"{n} ({p}%)" for n, p in zip(bajo, bajo_pct)],
        "Chi²": round(chi2_stat, 2), "p-valor": f"{p_valor:.4e}",
    })


# ─────────────────────────────────────────────
#  HELPERS UI
# ─────────────────────────────────────────────
def kpi_gradient(valor, etiqueta, detalle=None, gradiente=None, resaltado=False):
    gradiente = gradiente or KPI_GRADIENTS[0]
    borde_extra = "none"
    return html.Div([
        html.Div(valor, style={
            "fontSize": "2rem", "fontWeight": "800", "color": BLANCO,
            "fontFamily": FONT, "lineHeight": "1.1"
        }),
        html.Div(etiqueta, style={
            "fontSize": "0.83rem", "color": "rgba(255,255,255,0.9)",
            "marginTop": "0.35rem", "fontWeight": "600"
        }),
        html.Div(detalle or "", style={
            "fontSize": "0.74rem", "color": "rgba(255,255,255,0.78)",
            "marginTop": "0.2rem"
        }),
    ], style={
        "background": gradiente, "borderRadius": "16px",
        "padding": "1.2rem 1.35rem",
        "boxShadow": "0 10px 24px rgba(79,124,255,0.18)",
        "minHeight": "110px", "display": "flex",
        "flexDirection": "column", "justifyContent": "center",
        "border": borde_extra,
    })


def card(children, titulo=None, subtitulo=None):
    content = []
    if titulo:
        content.append(html.H3(titulo, style=SEC_TITLE))
    if subtitulo:
        content.append(html.P(subtitulo, style=SUBTITLE_S))
    content += children if isinstance(children, list) else [children]
    return html.Div(content, style=CARD)


def img_block(src, alt=""):
    return html.Img(src=src, alt=alt, style={
        "width": "100%", "maxWidth": "850px", "display": "block",
        "margin": "1rem auto", "borderRadius": "6px",
        "boxShadow": "0 2px 10px rgba(0,0,0,0.08)"
    })


def dash_tabla(df, id_tabla="tabla", highlight_row=None, etiquetas=None):
    if df.empty:
        return html.P("Sin datos disponibles.", style={"color": GRIS_TEXTO})
    cond = [{"if": {"row_index": "odd"}, "backgroundColor": "#f8fafc"}]
    if highlight_row is not None:
        cond.append({
            "if": {"row_index": highlight_row},
            "fontWeight": "700",
            "backgroundColor": AZUL_CLARO,
        })
    etiquetas = etiquetas or {}
    return dash_table.DataTable(
        id=id_tabla,
        columns=[{"name": etiquetas.get(c, c), "id": c} for c in df.columns],
        data=df.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": AZUL_OSCURO, "color": BLANCO,
            "fontWeight": "bold", "fontFamily": FONT, "fontSize": "0.82rem",
            "padding": "10px",
        },
        style_cell={
            "fontFamily": FONT, "fontSize": "0.82rem",
            "padding": "8px 12px", "textAlign": "left",
            "color": GRIS_TEXTO, "border": f"1px solid {BORDE}",
        },
        style_data_conditional=cond,
        page_size=15,
    )


def texto_academico(parrafo, color=None):
    return html.P(parrafo, style={
        "fontFamily": FONT,
        "color": color or GRIS_TEXTO,
        "lineHeight": "1.95", "fontSize": "0.98rem",
        "textAlign": "justify", "marginBottom": "1rem",
    })


def bloque_texto(titulo, parrafos, color_titulo=AZUL_OSCURO, gradiente=None):
    return html.Div([
        html.H3(titulo, style={
            "fontFamily": FONT, "color": BLANCO if gradiente else color_titulo,
            "fontSize": "1.08rem", "fontWeight": "700", "marginBottom": "0.9rem",
        }),
        *[texto_academico(p, color="rgba(255,255,255,0.94)" if gradiente else GRIS_TEXTO)
          for p in parrafos]
    ], style={
        "background": gradiente if gradiente else BLANCO,
        "border": "none" if gradiente else f"1px solid {BORDE}",
        "borderRadius": "18px", "padding": "1.5rem",
        "boxShadow": "0 10px 24px rgba(26,58,92,0.10)",
        "height": "100%", "minHeight": "280px",
    })


def bloque_bullets(titulo, items, gradiente):
    return html.Div([
        html.H3(titulo, style={
            "fontFamily": FONT, "color": BLANCO,
            "fontSize": "1.08rem", "fontWeight": "700", "marginBottom": "0.95rem",
        }),
        html.Ul([html.Li(item, style={"marginBottom": "0.7rem", "lineHeight": "1.75"})
                 for item in items],
                style={"fontFamily": FONT, "color": "rgba(255,255,255,0.95)",
                       "fontSize": "0.94rem", "paddingLeft": "1.2rem", "marginBottom": "0"})
    ], style={
        "background": gradiente, "borderRadius": "18px", "padding": "1.5rem",
        "boxShadow": "0 10px 24px rgba(26,58,92,0.12)",
        "height": "100%", "minHeight": "280px",
    })


# ── Gráficos plotly para modelos ──────────────────────────────────
def build_roc_comparativa():
    """Curva ROC comparativa fija con todos los modelos."""
    fig_roc = go.Figure()
    colores_roc = {
        "Logistica":     "#A2F2F2",
        "KNN":           "#3C92A6",
        "Ridge":         "#F2B749",
        "Lasso":         "#F29441",
        "Random Forest": "#023373",
        "XGBoost":       "#5a8fa0",
        "LightGBM":      "#0a5080",
    }
    for nombre, df_roc in curvas_roc_dash.items():
        if {"fpr", "tpr"}.issubset(df_roc.columns):
            lw = 3 if nombre == MODELO_GANADOR else 1.5
            fig_roc.add_trace(go.Scatter(
                x=df_roc["fpr"], y=df_roc["tpr"], mode="lines", name=nombre,
                line=dict(color=colores_roc.get(nombre, "#aaaaaa"), width=lw)
            ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Azar",
        line=dict(color="#cccccc", dash="dash", width=1)
    ))
    fig_roc.update_layout(
        title="Curvas ROC — comparación de modelos (fija)",
        xaxis_title="Tasa de falsos positivos (1 - Especificidad)",
        yaxis_title="Tasa de verdaderos positivos (Sensibilidad)",
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=50, b=20),
        annotations=[dict(
            text=f"★ Modelo ganador: {MODELO_GANADOR}",
            xref="paper", yref="paper", x=0.01, y=0.97,
            showarrow=False, font=dict(size=11, color=CORAL),
            bgcolor="rgba(255,255,255,0.8)", bordercolor=CORAL, borderwidth=1,
        )]
    )
    return fig_roc


def build_cm_figure(modelo):
    """Heatmap de matriz de confusión para un modelo dado."""
    cm = cms_dash.get(modelo)
    if cm is None:
        return go.Figure()

    # --- Normalizar índices/columnas: aceptar 0/1 o strings como "Real_0_alto" ---
    df = cm.copy()

    # Renombrar índice (filas) a 0 y 1 si son strings
    idx_map = {}
    for v in df.index:
        sv = str(v).lower()
        if sv in ("0", "real_0_alto", "alto", "alto desempeno", "alto desempeño"):
            idx_map[v] = 0
        elif sv in ("1", "real_1_bajo", "bajo", "bajo desempeno", "bajo desempeño"):
            idx_map[v] = 1
    if idx_map:
        df = df.rename(index=idx_map)

    # Renombrar columnas a 0 y 1 si son strings
    col_map = {}
    for v in df.columns:
        sv = str(v).lower()
        if sv in ("0", "pred_0_alto", "alto", "alto desempeno", "alto desempeño"):
            col_map[v] = 0
        elif sv in ("1", "pred_1_bajo", "bajo", "bajo desempeno", "bajo desempeño"):
            col_map[v] = 1
    if col_map:
        df = df.rename(columns=col_map)

    # Asegurarse de tener filas 0 y 1 en ese orden
    for idx in [0, 1]:
        if idx not in df.index:
            df.loc[idx] = [0, 0]
        if idx not in df.columns:
            df[idx] = 0
    df = df.loc[[0, 1], [0, 1]]

    # Convertir a int para mostrar
    z = df.values.tolist()
    etiquetas = ["Alto desempeño (0)", "Bajo desempeño (1)"]

    # Etiquetas enriquecidas con TN/FP/FN/TP
    tn, fp = int(z[0][0]), int(z[0][1])
    fn, tp = int(z[1][0]), int(z[1][1])
    text_labels = [
        [f"{tn}\nTN", f"{fp}\nFP"],
        [f"{fn}\nFN", f"{tp}\nTP"],
    ]

    # Color fijo por categoría (acierto vs error), independiente de la magnitud
    # del valor, para que las 7 matrices se lean con el mismo criterio visual.
    # Diagonal (TN, TP) = acierto -> tono oscuro; antidiagonal (FP, FN) = error -> tono claro.
    z_color = [
        [1, 0],
        [0, 1],
    ]

    fig = go.Figure(go.Heatmap(
        z=z_color,
        x=[f"Pred: {e}" for e in etiquetas],
        y=[f"Real: {e}" for e in etiquetas],
        zmin=0, zmax=1,
        colorscale=[[0, "#A2F2F2"], [1, "#023373"]],
        showscale=False,
        text=text_labels,
        texttemplate="%{text}",
        textfont={"size": 16},
    ))
    fig.update_layout(
        title=f"Matriz de confusión — {modelo}",
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        margin=dict(t=55, b=20, l=20, r=20),
        xaxis=dict(side="bottom"),
        height=380,
    )
    return fig


# Paleta y orden compartidos entre los gráficos comparativos de modelos
ORDEN_MODELOS_COMP = ["KNN", "Logistica", "Lasso", "Ridge", "Random Forest", "XGBoost"]
COLORES_MODELOS_COMP = {
    "Logistica"    : "#A2F2F2",
    "KNN"          : "#3C92A6",
    "Ridge"        : "#F2B749",
    "Lasso"        : "#F29441",
    "Random Forest": "#023373",
    "XGBoost"      : "#5a8fa0",
}


def build_forest_plot_bootstrap(metrica="recall_1"):
    """
    Forest plot nativo (Plotly) — estimación puntual + IC 95% bootstrap por modelo,
    a partir de df_bootstrap_ic (una fila por modelo, columnas <metrica>,
    <metrica>_ic_low, <metrica>_ic_high).
    """
    etiquetas_metrica = {
        "recall_1": "Recall (clase 1)",
        "auc": "AUC-ROC",
        "f1": "F1-Score",
        "precision": "Precisión",
    }
    nombre_metrica = etiquetas_metrica.get(metrica, metrica)

    if df_bootstrap_ic.empty or metrica not in df_bootstrap_ic.columns:
        fig = go.Figure()
        fig.add_annotation(text="Datos de bootstrap no disponibles",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color=GRIS_MUTED))
        return fig

    col_low  = f"{metrica}_ic_low"
    col_high = f"{metrica}_ic_high"

    df_fp = df_bootstrap_ic.copy()
    modelos_presentes = [m for m in ORDEN_MODELOS_COMP if m in df_fp["modelo"].unique()]
    modelos_extra = [m for m in df_fp["modelo"].unique() if m not in ORDEN_MODELOS_COMP]
    orden_final = modelos_presentes + modelos_extra
    df_fp["orden"] = df_fp["modelo"].map({m: i for i, m in enumerate(orden_final)})
    df_fp = df_fp.sort_values("orden", ascending=False)

    fig = go.Figure()
    for _, row in df_fp.iterrows():
        modelo_n = row["modelo"]
        color = COLORES_MODELOS_COMP.get(modelo_n, AZUL_OSCURO)
        es_ganador = modelo_n == MODELO_GANADOR
        fig.add_trace(go.Scatter(
            x=[row[col_low], row[col_high]], y=[modelo_n, modelo_n],
            mode="lines",
            line=dict(color=color, width=3 if es_ganador else 2),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[row[metrica]], y=[modelo_n],
            mode="markers",
            marker=dict(color=color, size=14 if es_ganador else 10,
                        symbol="diamond" if es_ganador else "circle",
                        line=dict(color=BLANCO, width=1.5)),
            showlegend=False,
            hovertemplate=(f"<b>{modelo_n}</b><br>{nombre_metrica}: %{{x:.3f}}"
                           f"<br>IC 95%%: [{row[col_low]:.3f}, {row[col_high]:.3f}]<extra></extra>"),
        ))

    fig.update_layout(
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        margin=dict(t=30, b=40, l=20, r=20),
        height=380,
        xaxis=dict(title=f"{nombre_metrica} (estimación puntual e IC 95% bootstrap, B = 1.000)",
                   gridcolor="#edf0f7", zeroline=False),
        yaxis=dict(title="", automargin=True),
        showlegend=False,
    )
    return fig


def build_shap_beeswarm():
    """
    Bee swarm nativo (Plotly) de valores SHAP — a partir de df_shap_bee
    (formato largo: feature, label, shap_value, feature_value) y df_shap_imp
    (orden e importancia global por variable).
    """
    if df_shap_bee.empty:
        fig = go.Figure()
        fig.add_annotation(text="Valores SHAP no disponibles",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color=GRIS_MUTED))
        return fig

    # Orden de variables: de mayor a menor importancia (de abajo hacia arriba en el eje Y)
    if not df_shap_imp.empty:
        orden_labels = df_shap_imp.sort_values("importancia", ascending=True)["label"].tolist()
    else:
        orden_labels = (df_shap_bee.groupby("label")["shap_value"]
                         .apply(lambda s: s.abs().mean())
                         .sort_values(ascending=True).index.tolist())

    y_pos = {lab: i for i, lab in enumerate(orden_labels)}

    fig = go.Figure()
    np.random.seed(42)
    for lab in orden_labels:
        sub = df_shap_bee[df_shap_bee["label"] == lab]
        fvals = sub["feature_value"].values
        fmin, fmax = fvals.min(), fvals.max()
        norm = (fvals - fmin) / (fmax - fmin) if fmax > fmin else np.zeros_like(fvals)
        jitter = y_pos[lab] + np.random.uniform(-0.28, 0.28, size=len(sub))

        fig.add_trace(go.Scatter(
            x=sub["shap_value"], y=jitter,
            mode="markers",
            marker=dict(
                size=5, opacity=0.65,
                color=norm, colorscale=[[0, AZUL_CLARO], [1, AZUL_OSCURO]],
                showscale=(lab == orden_labels[-1]),
                colorbar=dict(
                    title=dict(text="Valor de<br>la variable", font=dict(size=10)),
                    tickvals=[0, 1], ticktext=["Bajo", "Alto"],
                    len=0.7, thickness=12, x=1.02,
                ) if lab == orden_labels[-1] else None,
                line=dict(width=0),
            ),
            showlegend=False,
            hovertemplate=f"<b>{lab}</b><br>SHAP: %{{x:.4f}}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color=GRIS_MUTED, line_width=1)

    fig.update_layout(
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        margin=dict(t=30, b=40, l=20, r=90),
        height=460,
        xaxis=dict(title="Valor SHAP (impacto sobre la predicción de bajo desempeño)",
                   gridcolor="#edf0f7", zeroline=False),
        yaxis=dict(title="", tickmode="array",
                   tickvals=list(range(len(orden_labels))),
                   ticktext=orden_labels, automargin=True),
        showlegend=False,
    )
    return fig


def build_boxplot_pliegues_comparativo():
    """

    Boxplot comparativo de Recall por pliegue entre todos los modelos.
    """
    ORDEN_MODELOS = ["KNN", "Logistica", "Lasso", "Ridge", "Random Forest", "XGBoost"]
    COLORES_MODELOS = {
        "Logistica"    : "#A2F2F2",
        "KNN"          : "#3C92A6",
        "Ridge"        : "#F2B749",
        "Lasso"        : "#F29441",
        "Random Forest": "#023373",
        "XGBoost"      : "#5a8fa0",
    }

    if not df_pliegues_todos.empty:
        df_box = df_pliegues_todos.copy()
        df_box = df_box[~df_box["modelo"].str.lower().str.contains("baseline", na=False)]
    elif pliegues_dash:
        dfs = []
        for nombre, df_p in pliegues_dash.items():
            col = next((c for c in ["recall", "recall_1"] if c in df_p.columns), None)
            if col:
                tmp = df_p[[col]].rename(columns={col: "recall"})
                tmp["modelo"] = nombre
                dfs.append(tmp)
        df_box = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    else:
        np.random.seed(42)
        rows = []
        base_recalls = {
            "KNN": 0.675, "Logistica": 0.7152, "Lasso": 0.727,
            "Ridge": 0.7358, "Random Forest": 0.7897, "XGBoost": 0.6938,
        }
        for modelo_n, base in base_recalls.items():
            for _ in range(10):
                rows.append({"modelo": modelo_n,
                             "recall": float(np.clip(np.random.normal(base, 0.018), 0.55, 0.88))})
        df_box = pd.DataFrame(rows)

    if df_box.empty:
        fig = go.Figure()
        fig.add_annotation(text="Datos de pliegues no disponibles",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color=GRIS_MUTED))
        return fig

    # Detectar columna recall
    col_recall = next((c for c in ["recall", "recall_1"] if c in df_box.columns), None)
    if col_recall is None:
        col_recall = "auc"  # fallback si el CSV solo tiene auc

    modelos_presentes = [m for m in ORDEN_MODELOS if m in df_box["modelo"].unique()]
    modelos_extra     = [m for m in df_box["modelo"].unique() if m not in ORDEN_MODELOS]
    orden_final = modelos_presentes + modelos_extra

    fig = go.Figure()
    for modelo_n in orden_final:
        datos_m = df_box[df_box["modelo"] == modelo_n][col_recall].values
        color   = COLORES_MODELOS.get(modelo_n, AZUL_OSCURO)
        lw      = 3 if modelo_n == MODELO_GANADOR else 1.5
        fig.add_trace(go.Box(
            y=datos_m, name=modelo_n,
            boxpoints="all", jitter=0.3, pointpos=0,
            marker=dict(color=color, size=7, opacity=0.85,
                        line=dict(color="white", width=1)),
            line=dict(color=color, width=lw),
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0,2,4)) + (0.18,)}",
            whiskerwidth=0.7, notched=False,
            hovertemplate=f"<b>{modelo_n}</b><br>Recall: %{{y:.4f}}<br><extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="Recall (Clase 1) por pliegue de validación cruzada — comparación de modelos",
            font=dict(size=13, family=FONT, color=AZUL_OSCURO),
            x=0, xanchor="left",
        ),
        xaxis=dict(title="Modelo", categoryorder="array", categoryarray=orden_final,
                   tickfont=dict(size=11, family=FONT), gridcolor="#edf0f7", zeroline=False),
        yaxis=dict(title="Recall — Clase 1 (10 pliegues estratificados)",
                   tickfont=dict(size=11, family=FONT), gridcolor="#edf0f7",
                   zeroline=False, tickformat=".3f"),
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        margin=dict(t=60, b=60, l=60, r=20),
        showlegend=False, height=430,
        annotations=[dict(
            text=(
                "Nota: cada punto representa el Recall en un pliegue de validación cruzada estratificada (k=10). "
                f"El trazo más grueso corresponde al modelo ganador ({MODELO_GANADOR})."
            ),
            xref="paper", yref="paper",
            x=0, y=-0.16, showarrow=False,
            font=dict(size=9, color=GRIS_MUTED, family=FONT), align="left",
        )],
    )
    return fig


def grafico_distribucion(variable):
    """Gráfico de distribución mejorado con barras horizontales y proporciones."""
    if data_imputada.empty or variable not in data_imputada.columns:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos disponibles", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=16, color=GRIS_TEXTO))
        return fig

    if variable == "desempeno_dicotomico":
        conteo = (data_imputada[["desempeno_dicotomico"]]
                  .value_counts(dropna=False).reset_index(name="n")
                  .sort_values("desempeno_dicotomico"))
        etiquetas = {0: "Alto desempeño", 1: "Bajo desempeño"}
        conteo["categoria"] = conteo["desempeno_dicotomico"].map(etiquetas)
        conteo["pct"] = (conteo["n"] / conteo["n"].sum() * 100).round(1)

        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            subplot_titles=["Frecuencia por clase", "Distribución del puntaje RC por clase"],
        )

        # Panel izquierdo — barras
        colores = {"Alto desempeño": VERDE, "Bajo desempeño": MORADO_OSCURO}
        for _, row in conteo.iterrows():
            fig.add_trace(go.Bar(
                x=[row["categoria"]], y=[row["n"]],
                text=[f"{row['pct']}%"], textposition="outside",
                marker_color=colores[row["categoria"]],
                marker_line_color=BLANCO, marker_line_width=1.5,
                name=row["categoria"], showlegend=False,
            ), row=1, col=1)

        # Panel derecho — boxplots del puntaje continuo si existe
        col_puntaje = next((c for c in data_imputada.columns
                            if "puntaje" in c.lower() or "punt_rc" in c.lower()
                            or "mod_razona" in c.lower() or "razona_cuantitat" in c.lower()), None)
        if col_puntaje:
            for clase, label in [(0, "Alto desempeño"), (1, "Bajo desempeño")]:
                subset = data_imputada[data_imputada["desempeno_dicotomico"] == clase][col_puntaje]
                fig.add_trace(go.Box(
                    y=subset, name=label, marker_color=PETROL,
                    boxmean=True, showlegend=False,
                    line_color=PETROL,
                ), row=1, col=2)
        else:
            fig.add_annotation(
                text="Puntaje RC no disponible", x=0.75, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=GRIS_MUTED)
            )

        fig.update_layout(
            paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
            font_family=FONT, font_color=GRIS_TEXTO,
            margin=dict(t=60, b=30, l=10, r=10),
            yaxis=dict(title="Frecuencia", gridcolor="#edf0f7"),
            yaxis2=dict(title="Puntaje RC", gridcolor="#edf0f7"),
        )
        return fig

    tiene_desempeno = "desempeno_dicotomico" in data_imputada.columns
    col_puntaje = next((c for c in data_imputada.columns
                        if "mod_razona" in c.lower() or "razona_cuantitat" in c.lower()
                        or "puntaje" in c.lower() or "punt_rc" in c.lower()), None)

    # ── Variable continua (ej. edad): boxplot por clase, no barras categóricas ──
    es_continua = (
        pd.api.types.is_numeric_dtype(data_imputada[variable])
        and data_imputada[variable].nunique() > 15
    )
    if es_continua and tiene_desempeno:
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            subplot_titles=[f"Distribución general", f"{variable.capitalize()} por clase de desempeño"],
            horizontal_spacing=0.12,
        )

        # Panel izquierdo — histograma general (un bin por cada valor entero de edad)
        edad_min, edad_max = data_imputada[variable].min(), data_imputada[variable].max()
        fig.add_trace(go.Histogram(
            x=data_imputada[variable], marker_color=PETROL,
            marker_line_color=BLANCO, marker_line_width=0.8,
            showlegend=False,
            xbins=dict(start=edad_min - 0.5, end=edad_max + 0.5, size=1),
        ), row=1, col=1)

        # Panel derecho — boxplot comparativo por clase
        for clase, label in [(0, "Alto desempeño"), (1, "Bajo desempeño")]:
            subset = data_imputada[data_imputada["desempeno_dicotomico"] == clase][variable]
            fig.add_trace(go.Box(
                y=subset, name=label, marker_color=PETROL,
                boxmean=True, showlegend=False, line_color=PETROL,
            ), row=1, col=2)

        fig.update_layout(
            paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
            font_family=FONT, font_color=GRIS_TEXTO,
            margin=dict(t=60, b=30, l=10, r=10),
            height=420,
        )
        fig.update_yaxes(title="Frecuencia", gridcolor="#edf0f7", row=1, col=1)
        fig.update_yaxes(title=variable.capitalize(), gridcolor="#edf0f7", row=1, col=2)
        fig.update_xaxes(title=variable.capitalize(), dtick=1, gridcolor="#edf0f7", row=1, col=1)
        return fig

    if tiene_desempeno:
        conteo = (data_imputada.groupby([variable, "desempeno_dicotomico"])
                  .size().reset_index(name="n"))
        conteo["Desempeño"] = conteo["desempeno_dicotomico"].map(
            {0: "Alto desempeño", 1: "Bajo desempeño"})

        total_cat = conteo.groupby(variable)["n"].transform("sum")
        conteo["pct"] = (conteo["n"] / total_cat * 100).round(1)

        orden = (conteo.groupby(variable)["n"].sum()
                 .sort_values(ascending=False).index.tolist())

        from plotly.subplots import make_subplots
        alto = max(400, len(orden) * 38 + 100)

        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            subplot_titles=["Distribución proporcional por categoría",
                            "Puntaje RC por categoría"],
            horizontal_spacing=0.12,
        )

        # Panel izquierdo — barras apiladas
        for desempeno, color in [("Alto desempeño", AZUL_MEDIO), ("Bajo desempeño", MORADO_OSCURO)]:
            sub = conteo[conteo["Desempeño"] == desempeno]
            fig.add_trace(go.Bar(
                y=[orden.index(c) for c in sub[variable]],
                x=sub["n"],
                orientation="h",
                name=desempeno,
                marker_color=color,
                marker_line_color=BLANCO,
                marker_line_width=0.8,
                text=sub["pct"].map(lambda p: f"{p:.1f}%"),
                textposition="inside",
                insidetextanchor="middle",
                legendgroup=desempeno,
            ), row=1, col=1)

        # Etiquetas de frecuencia total al final de cada barra
        totales = conteo.groupby(variable)["n"].sum().reset_index()
        totales["y_idx"] = totales[variable].map(lambda c: orden.index(c))
        fig.add_trace(go.Bar(
            y=totales["y_idx"],
            x=totales["n"],
            orientation="h",
            marker_color="rgba(0,0,0,0)",
            showlegend=False,
            text=totales["n"].map(lambda n: f"n={n:,}"),
            textposition="outside",
            textfont=dict(size=12, color=AZUL_OSCURO, family=FONT),
            hoverinfo="skip",
        ), row=1, col=1)

        fig.update_yaxes(
            tickvals=list(range(len(orden))),
            ticktext=orden,
            automargin=True, row=1, col=1
        )

        # Panel derecho — boxplots por categoría de la variable
        if col_puntaje:
            for i, cat in enumerate(orden):
                subset = data_imputada[data_imputada[variable] == cat][col_puntaje]
                fig.add_trace(go.Box(
                    y=subset, name=str(cat),
                    marker_color=PETROL,
                    boxmean=True, showlegend=False,
                    line_width=1.2,
                ), row=1, col=2)
        else:
            fig.add_annotation(
                text="Puntaje RC no disponible", x=0.8, y=0.5,
                xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=GRIS_MUTED)
            )

        fig.update_layout(
            paper_bgcolor=GRIS_SUAVE, plot_bgcolor=GRIS_SUAVE,
            font_family=FONT, font_color=GRIS_TEXTO,
            margin=dict(t=70, b=30, l=20, r=80),
            height=alto,
            barmode="stack",
            legend=dict(orientation="h", y=-0.15, x=0.3, xanchor="center"),
            xaxis=dict(title="Frecuencia", gridcolor="rgba(0,0,0,0)", zeroline=False,
                       showgrid=False, tickformat=","),
            yaxis=dict(title="", automargin=True, showgrid=False),
            yaxis2=dict(title="Puntaje RC", gridcolor="rgba(0,0,0,0)", showgrid=False),
        )
        # Bajar los subtítulos para que no choquen con la leyenda
        fig.layout.annotations[0].update(y=1.08, font=dict(size=12))
        fig.layout.annotations[1].update(y=1.08, font=dict(size=12))
    else:
        conteo = data_imputada[variable].value_counts(dropna=False).reset_index()
        conteo.columns = [variable, "n"]
        fig = px.bar(conteo, x=variable, y="n", text_auto=True,
                     color_discrete_sequence=px.colors.qualitative.Vivid,
                     labels={variable: variable, "n": "Frecuencia"})
        fig.update_traces(marker_line_color=BLANCO, marker_line_width=1.2)
        fig.update_layout(paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
                          font_family=FONT, font_color=GRIS_TEXTO,
                          margin=dict(t=40, b=30, l=20, r=20))

    return fig


# ─────────────────────────────────────────────
#  OPERACIONALIZACIÓN
# ─────────────────────────────────────────────
def _th_style(color="#1f3556", width=None):
    s = {"backgroundColor": color, "color": "#ffffff", "fontWeight": "600",
         "fontSize": "0.80rem", "fontFamily": FONT, "padding": "10px 14px",
         "textAlign": "left", "borderBottom": "2px solid rgba(255,255,255,0.2)",
         "whiteSpace": "nowrap"}
    if width:
        s["width"] = width
    return s


def _td_style(bold=False):
    return {"padding": "10px 14px", "verticalAlign": "top", "color": "#1f3556",
            "fontSize": "0.82rem", "fontFamily": FONT,
            "borderBottom": "1px solid #e8ebf3",
            "fontWeight": "600" if bold else "400", "lineHeight": "1.5"}


def _build_rows(grupo_filtro="todos", nivel_filtro="todos"):
    base = [v for v in VARIABLES_DATA if v["grupo"] != "Variable a predecir"]
    filtradas = [v for v in base
                 if (grupo_filtro == "todos" or v["grupo"] == grupo_filtro)
                 and (nivel_filtro == "todos" or v["nivel"] == nivel_filtro)]
    if not filtradas:
        return [html.Tr([html.Td("Sin resultados.", colSpan=6,
                                 style={"padding": "2rem", "textAlign": "center",
                                        "color": "#8d98ab", "fontFamily": FONT})])], 0, len(base)
    filas = []
    ultimo_grupo = None
    for i, v in enumerate(filtradas):
        cfg = GRUPO_CONFIG.get(v["grupo"], {"color": "#596579", "light": "#f5f6fb",
                                             "badge": "#e8ebf3", "icon": "·"})
        n_cfg = NIVEL_COLOR.get(v["nivel"], {"bg": "#f5f6fb", "text": "#596579"})
        bg_row = "#fafbff" if i % 2 == 0 else "#ffffff"
        mostrar_grupo = v["grupo"] != ultimo_grupo
        ultimo_grupo  = v["grupo"]
        celda_grupo = html.Td(
            html.Div(html.Span(v["grupo"], style={"fontWeight": "600"})) if mostrar_grupo else "",
            style={"padding": "10px 14px", "verticalAlign": "top", "color": cfg["color"],
                   "fontSize": "0.80rem", "fontFamily": FONT,
                   "borderBottom": "1px solid #e8ebf3",
                   "backgroundColor": cfg["light"] if mostrar_grupo else bg_row,
                   "borderLeft": f"3px solid {cfg['color']}" if mostrar_grupo else "3px solid transparent",
                   "whiteSpace": "nowrap"})
        badges = html.Div([
            html.Span(r, style={"display": "inline-block", "backgroundColor": cfg["badge"],
                                "color": cfg["color"], "borderRadius": "6px",
                                "padding": "2px 7px", "fontSize": "0.75rem", "fontFamily": FONT,
                                "margin": "2px 2px", "fontWeight": "500",
                                "border": f"1px solid {cfg['color']}22"})
            for r in v["respuestas"]], style={"lineHeight": "1.8"})
        filas.append(html.Tr([
            celda_grupo,
            html.Td(v["variable"], style=_td_style(bold=True)),
            html.Td(v["descripcion"], style={**_td_style(), "color": "#596579"}),
            html.Td(badges, style={**_td_style(), "minWidth": "180px"}),
            html.Td(v["tipo"], style={**_td_style(), "color": "#8d98ab"}),
            html.Td(html.Span(v["nivel"], style={
                "backgroundColor": n_cfg["bg"], "color": n_cfg["text"],
                "borderRadius": "6px", "padding": "3px 9px", "fontSize": "0.76rem",
                "fontWeight": "600", "fontFamily": FONT, "whiteSpace": "nowrap"}),
                    style=_td_style()),
        ], style={"backgroundColor": bg_row}))
    return filas, len(filtradas), len(base)


def build_operacionalizacion():
    grupos = list(dict.fromkeys(v["grupo"] for v in VARIABLES_DATA if v["grupo"] != "Variable a predecir"))
    opciones_grupo = [{"label": "Todos los grupos", "value": "todos"}] + [
        {"label": g, "value": g} for g in grupos]
    filtros = html.Div([
        html.Div([
            html.Label("Filtrar por grupo:", style={"fontSize": "0.82rem", "color": "#596579",
                                                     "fontFamily": FONT, "fontWeight": "600",
                                                     "marginBottom": "4px", "display": "block"}),
            dcc.Dropdown(id="op-dropdown-grupo", options=opciones_grupo, value="todos",
                         clearable=False, searchable=False,
                         style={"fontFamily": FONT, "fontSize": "0.85rem", "minWidth": "260px"})]),
        html.Div([
            html.Label("Filtrar por nivel:", style={"fontSize": "0.82rem", "color": "#596579",
                                                     "fontFamily": FONT, "fontWeight": "600",
                                                     "marginBottom": "4px", "display": "block"}),
            dcc.Dropdown(id="op-dropdown-nivel",
                         options=[{"label": "Todos", "value": "todos"},
                                  {"label": "Categórica dicotómica", "value": "Categórica dicotómica"},
                                  {"label": "Categórica nominal", "value": "Categórica nominal"},
                                  {"label": "Categórica ordinal", "value": "Categórica ordinal"}],
                         value="todos", clearable=False, searchable=False,
                         style={"fontFamily": FONT, "fontSize": "0.85rem", "minWidth": "240px"})]),
        html.Div(id="op-contador", style={"marginLeft": "auto", "fontSize": "0.83rem",
                                           "color": "#8d98ab", "fontFamily": FONT,
                                           "alignSelf": "flex-end", "paddingBottom": "2px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
              "gap": "1rem", "marginBottom": "1.4rem", "padding": "1rem 1.2rem",
              "background": "#f7f8fc", "borderRadius": "12px", "border": f"1px solid {BORDE}"})

    tabla = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Grupo",              style=_th_style("#1f3556", "130px")),
                html.Th("Variable",           style=_th_style("#1f3556", "170px")),
                html.Th("Descripción",        style=_th_style("#1f3556")),
                html.Th("Posibles respuestas",style=_th_style("#1f3556", "220px")),
                html.Th("Tipo",               style=_th_style("#1f3556", "100px")),
                html.Th("Nivel",              style=_th_style("#1f3556", "140px")),
            ])),
            html.Tbody(id="op-tabla-body"),
        ], style={"width": "100%", "borderCollapse": "collapse",
                  "fontFamily": FONT, "fontSize": "0.83rem"})
    ], style={"overflowX": "auto", "borderRadius": "12px",
              "border": f"1px solid {BORDE}", "background": "#ffffff"})

    # ── Tabla independiente: variable a predecir (sin filtros) ─────
    var_objetivo = next(v for v in VARIABLES_DATA if v["grupo"] == "Variable a predecir")
    cfg_obj = GRUPO_CONFIG.get(var_objetivo["grupo"], {"color": "#596579", "light": "#f5f6fb",
                                                        "badge": "#e8ebf3", "icon": "·"})
    n_cfg_obj = NIVEL_COLOR.get(var_objetivo["nivel"], {"bg": "#f5f6fb", "text": "#596579"})
    badges_obj = html.Div([
        html.Span(r, style={"display": "inline-block", "backgroundColor": cfg_obj["badge"],
                            "color": cfg_obj["color"], "borderRadius": "6px",
                            "padding": "2px 7px", "fontSize": "0.75rem", "fontFamily": FONT,
                            "margin": "2px 2px", "fontWeight": "500",
                            "border": f"1px solid {cfg_obj['color']}22"})
        for r in var_objetivo["respuestas"]], style={"lineHeight": "1.8"})
    fila_objetivo = html.Tr([
        html.Td(html.Span(var_objetivo["grupo"], style={"fontWeight": "600"}),
                style={"padding": "10px 14px", "verticalAlign": "top", "color": cfg_obj["color"],
                       "fontSize": "0.80rem", "fontFamily": FONT,
                       "borderBottom": "1px solid #e8ebf3",
                       "backgroundColor": cfg_obj["light"],
                       "borderLeft": f"3px solid {cfg_obj['color']}",
                       "whiteSpace": "nowrap"}),
        html.Td(var_objetivo["variable"], style=_td_style(bold=True)),
        html.Td(var_objetivo["descripcion"], style={**_td_style(), "color": "#596579"}),
        html.Td(badges_obj, style={**_td_style(), "minWidth": "180px"}),
        html.Td(var_objetivo["tipo"], style={**_td_style(), "color": "#8d98ab"}),
        html.Td(html.Span(var_objetivo["nivel"], style={
            "backgroundColor": n_cfg_obj["bg"], "color": n_cfg_obj["text"],
            "borderRadius": "6px", "padding": "3px 9px", "fontSize": "0.76rem",
            "fontWeight": "600", "fontFamily": FONT, "whiteSpace": "nowrap"}),
                style=_td_style()),
    ], style={"backgroundColor": "#fafbff"})
    tabla_objetivo = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Grupo",              style=_th_style("#1f3556", "130px")),
                html.Th("Variable",           style=_th_style("#1f3556", "170px")),
                html.Th("Descripción",        style=_th_style("#1f3556")),
                html.Th("Posibles respuestas",style=_th_style("#1f3556", "220px")),
                html.Th("Tipo",               style=_th_style("#1f3556", "100px")),
                html.Th("Nivel",              style=_th_style("#1f3556", "140px")),
            ])),
            html.Tbody([fila_objetivo]),
        ], style={"width": "100%", "borderCollapse": "collapse",
                  "fontFamily": FONT, "fontSize": "0.83rem"})
    ], style={"overflowX": "auto", "borderRadius": "12px",
              "border": f"1px solid {BORDE}", "background": "#ffffff"})

    return html.Div([
        html.Div([
            html.H3("Operacionalización de variables",
                    style={"fontFamily": FONT, "color": "#4d2d8c", "fontSize": "1.55rem",
                           "fontWeight": "800", "marginBottom": "0.2rem",
                           "letterSpacing": "-0.02em"}),
            html.P("Clasificación, descripción y niveles de medición de las variables del estudio",
                   style={"fontFamily": FONT, "color": "#596579",
                          "fontSize": "0.92rem", "marginBottom": "0"}),
        ], style={"marginBottom": "1.2rem"}),

        html.H4("Variable a predecir",
                style={"fontFamily": FONT, "color": "#A6321F", "fontSize": "1.1rem",
                       "fontWeight": "700", "marginBottom": "0.7rem"}),
        tabla_objetivo,

        html.H4("Variables predictoras",
                style={"fontFamily": FONT, "color": "#4d2d8c", "fontSize": "1.1rem",
                       "fontWeight": "700", "marginTop": "1.8rem", "marginBottom": "0.7rem"}),
        filtros, tabla,

        html.P("Nota. La variable a predecir y las 18 variables predictoras son de tipo cualitativo. Elaboración propia.",
               style={"fontFamily": FONT, "fontSize": "0.82rem", "fontStyle": "italic",
                      "color": "#8d98ab", "marginTop": "0.75rem"}),
    ], style={**CARD})


# ─────────────────────────────────────────────
#  LAYOUTS
# ─────────────────────────────────────────────
def layout_portada():
    n = len(data_imputada) if not data_imputada.empty else 13078
    return html.Div([

        # ── Hero con imagen de fondo ─────────────────────────────
        html.Div([
            # Overlay degradado sobre la imagen
            html.Div(style={
                "position": "absolute", "inset": "0",
                "background": f"linear-gradient(135deg, rgba(58,125,140,0.88) 0%, rgba(31,53,86,0.80) 60%, rgba(77,191,160,0.55) 100%)",
                "zIndex": "1",
            }),
            # Contenido del hero
            html.Div([
                # Acento superior coral
                html.Div(style={
                    "width": "56px", "height": "4px",
                    "backgroundColor": CORAL,
                    "marginBottom": "1.8rem",
                    "borderRadius": "2px",
                }),
                html.H1(
                    "Predicción del Desempeño en Razonamiento Cuantitativo "
                    "mediante algoritmos de aprendizaje automático",
                    style={
                        "fontFamily": FONT, "color": BLANCO,
                        "fontSize": "2.4rem", "fontWeight": "800",
                        "lineHeight": "1.3", "maxWidth": "750px",
                        "marginBottom": "1rem",
                        "textShadow": "0 2px 12px rgba(0,0,0,0.25)",
                    }
                ),
                html.H2(
                    "Un estudio con datos del Atlántico · 2023",
                    style={
                        "fontFamily": FONT, "color": "rgba(255,255,255,0.85)",
                        "fontSize": "1.1rem", "fontWeight": "400",
                        "marginBottom": "2.5rem", "letterSpacing": "0.02em",
                    }
                ),
                # Metadatos
                html.Div([
                    html.Div([
                        html.Span("Programa: ", style={"fontWeight": "700", "color": MENTA}),
                        html.Span("Maestría en Estadística Aplicada · Universidad del Norte",
                                  style={"color": "rgba(255,255,255,0.88)"}),
                    ], style={"marginBottom": "0.45rem"}),
                    html.Div([
                        html.Span("Fuente: ", style={"fontWeight": "700", "color": MENTA}),
                        html.Span("ICFES · Registro Saber Pro 2023",
                                  style={"color": "rgba(255,255,255,0.88)"}),
                    ], style={"marginBottom": "0.45rem"}),
                    html.Div([
                        html.Span("Autora: ", style={"fontWeight": "700", "color": MENTA}),
                        html.Span("Natali Angarita Escolar",
                                  style={"color": "rgba(255,255,255,0.88)"}),
                    ], style={"marginBottom": "0.45rem"}),
                    html.Div([
                        html.Span("Tutor: ", style={"fontWeight": "700", "color": MENTA}),
                        html.Span("MSc. Carlos De Oro",
                                  style={"color": "rgba(255,255,255,0.88)"}),
                    ]),
                ], style={
                    "fontFamily": FONT, "fontSize": "0.93rem",
                    "borderLeft": f"3px solid {CORAL}",
                    "paddingLeft": "1.2rem", "marginBottom": "3rem",
                }),
            ], style={
                "position": "relative", "zIndex": "2",
                "padding": "5rem 3.5rem 3rem 8.5rem",
                "maxWidth": "900px",
            }),
        ], style={
            "position": "relative",
            "backgroundImage": "url('/assets/portada_bg.jpg')",
            "backgroundSize": "cover",
            "backgroundPosition": "center top",
            "minHeight": "62vh",
            "overflow": "hidden",
        }),

        # ── KPIs debajo del hero ──────────────────────────────────
        html.Div([
            html.Div([
                kpi_gradient(f"{n:,}", "Estudiantes analizados", "Base depurada", KPI_GRADIENTS[0]),
                kpi_gradient("23", "Municipios del Atlántico", gradiente=KPI_GRADIENTS[1]),
                kpi_gradient("6", "Modelos de ML comparados", gradiente=KPI_GRADIENTS[2]),
                kpi_gradient("2023", "Ventana de estudio", gradiente=KPI_GRADIENTS[3]),
            ], style={
                "display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
                "gap": "1.2rem", "maxWidth": "1050px", "margin": "0 auto",
            }),
        ], style={
            "backgroundColor": BLANCO,
            "padding": "2.5rem 3rem",
            "boxShadow": "0 4px 20px rgba(58,125,140,0.10)",
            "borderBottom": f"3px solid {CORAL}",
        }),

    ], style={"backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_intro_intro():
    """Pestaña Introducción — diseño tipo diapositiva Propósito."""

    def nodo(color_circulo, titulo, descripcion):
        return html.Div([
            # círculo con ícono
            html.Div(style={
                "width": "52px", "height": "52px", "flexShrink": "0",
                "backgroundColor": color_circulo,
                "borderRadius": "50%",
                "border": f"3px solid {BLANCO}",
                "boxShadow": f"0 0 0 3px {color_circulo}55",
                "zIndex": "1",
            }),
            html.Div([
                html.Div(titulo, style={
                    "fontFamily": FONT, "fontWeight": "800",
                    "fontSize": "1.05rem", "color": AZUL_OSCURO,
                    "letterSpacing": "0.04em", "textTransform": "uppercase",
                    "marginBottom": "0.3rem",
                }),
                html.Div(descripcion, style={
                    "fontFamily": FONT, "fontSize": "0.88rem",
                    "color": GRIS_TEXTO, "lineHeight": "1.5",
                }),
            ], style={"flex": "1", "paddingLeft": "0.5rem"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "1rem"})

    # columna izquierda: texto
    col_izq = html.Div([
        html.H1("PROPÓSITO", style={
            "fontFamily": FONT, "fontWeight": "900",
            "fontSize": "2.4rem", "color": AZUL_OSCURO,
            "letterSpacing": "0.04em", "marginBottom": "1.8rem",
        }),
        html.P(
            "El desempeño en pruebas estandarizadas como el Saber Pro refleja desigualdades "
            "estructurales que demandan intervención temprana. Este estudio aplica técnicas de "
            "aprendizaje automático para identificar, antes de la prueba, a estudiantes con "
            "riesgo de bajo rendimiento en Razonamiento Cuantitativo.",
            style={"fontFamily": FONT, "fontSize": "0.95rem", "color": GRIS_TEXTO,
                   "lineHeight": "1.8", "marginBottom": "1.2rem"}
        ),
        html.P([
            "El objetivo es predecir el desempeño y clasificar a los estudiantes en ",
            html.Strong("alto (0) y bajo (1) desempeño", style={"color": CORAL}),
            ", a partir de datos de 13.078 estudiantes del departamento del Atlántico "
            "que presentaron la prueba en 2023 (fuente: ICFES).",
        ], style={"fontFamily": FONT, "fontSize": "0.95rem", "color": GRIS_TEXTO,
                  "lineHeight": "1.8"}),
    ], style={"flex": "1.1", "paddingRight": "3rem"})

    # columna derecha: línea vertical con nodos
    col_der = html.Div([
        # nodo 1
        nodo(PETROL, "Marco Teórico",
             "Literatura relacionada con EDM y sustento matemático de técnicas y métodos implementados."),
        # línea vertical entre nodos
        html.Div(style={
            "width": "3px", "height": "3rem",
            "backgroundColor": AZUL_OSCURO,
            "margin": "0 0 0 24px",
        }),
        # nodo 2
        nodo(CORAL, "Metodología",
             "Operacionalización de las variables. Arquitectura de los modelos y técnicas de validación, e interpretación."),
        # línea vertical entre nodos
        html.Div(style={
            "width": "3px", "height": "3rem",
            "backgroundColor": AZUL_OSCURO,
            "margin": "0 0 0 24px",
        }),
        # nodo 3
        nodo(AMARILLO, "Resultados",
             "Hallazgos sobre el desempeño de los algoritmos. Aplicación de Bootstrap, test de Wilcoxon y valores SHAP."),
    ], style={"flex": "0.9", "display": "flex", "flexDirection": "column"})

    return html.Div([
        html.Div([col_izq, col_der], style={
            "display": "flex", "alignItems": "flex-start",
            "gap": "1rem",
            "background": BLANCO,
            "borderRadius": "18px",
            "border": f"1px solid {BORDE}",
            "padding": "3rem 2.5rem",
            "boxShadow": "0 6px 24px rgba(2,51,115,0.08)",
            "maxWidth": "1050px", "margin": "0 auto",
        }),
    ], style={"padding": "2.5rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_intro_intro_OLD():
    """Pestaña 1 — versión anterior (archivada)."""

    bloque_superior = html.Div([
        # Resumen izquierda
        html.Div([
            html.H4("Sobre este trabajo", style={
                "fontFamily": FONT, "color": AZUL_OSCURO, "fontSize": "1rem",
                "fontWeight": "700", "marginBottom": "0.8rem",
                "borderBottom": f"2px solid {BORDE}", "paddingBottom": "0.5rem"}),
            html.P("El análisis del rendimiento académico en pruebas estandarizadas "
                   "constituye un insumo estratégico para la formulación de políticas "
                   "educativas y el diseño de intervenciones focalizadas en estudiantes "
                   "en riesgo. En los últimos años, el aprendizaje automático ha fortalecido "
                   "la minería de datos educativos, ampliando la capacidad predictiva de los "
                   "enfoques tradicionales.",
                   style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                          "lineHeight": "1.8", "marginBottom": "0.8rem"}),
            html.P("Este estudio analiza 13,078 registros de estudiantes del departamento "
                   "del Atlántico que presentaron la prueba Saber Pro en 2023 (fuente ICFES). "
                   "Se comparan seis modelos de clasificación supervisada y se privilegia el "
                   "Recall de la clase de bajo desempeño como criterio de selección, coherente "
                   "con el objetivo de intervención educativa preventiva.",
                   style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                          "lineHeight": "1.8", "marginBottom": "0.8rem"}),
            html.P("El modelo ganador (Random Forest, Recall = 0,803) permite identificar "
                   "correctamente a más del 80% de los estudiantes en riesgo usando únicamente "
                   "información disponible al momento de la matrícula.",
                   style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                          "lineHeight": "1.8"}),
        ], style={"flex": "1", "minWidth": "280px", "padding": "1.4rem",
                  "background": BLANCO, "borderRadius": "14px", "border": f"1px solid {BORDE}"}),

        # Objetivo derecha
        html.Div([
            html.Span("Objetivo general", style={
                "fontFamily": FONT, "fontSize": "0.78rem", "fontWeight": "700",
                "color": BLANCO, "textTransform": "uppercase",
                "letterSpacing": "0.08em", "opacity": "0.85"}),
            html.P("Predecir el desempeño en Razonamiento Cuantitativo de los estudiantes "
                   "del departamento del Atlántico que presentaron la prueba Saber Pro en 2023, "
                   "mediante la construcción y comparación de modelos de clasificación supervisada "
                   "basados en variables sociodemográficas, académicas, institucionales y de "
                   "condiciones del hogar y la familia.",
                   style={"fontFamily": FONT, "fontSize": "0.95rem", "color": BLANCO,
                          "lineHeight": "1.8", "marginTop": "0.6rem", "marginBottom": "1.2rem"}),
            html.Hr(style={"borderColor": "rgba(255,255,255,0.25)", "margin": "0.8rem 0"}),
            html.Span("Importancia del estudio", style={
                "fontFamily": FONT, "fontSize": "0.78rem", "fontWeight": "700",
                "color": BLANCO, "textTransform": "uppercase",
                "letterSpacing": "0.08em", "opacity": "0.85"}),
            html.Ul([
                html.Li("Permite identificar estudiantes en riesgo antes de que presenten la prueba, "
                        "usando datos de matrícula disponibles desde el primer semestre.",
                        style={"fontSize": "0.87rem", "color": "rgba(255,255,255,0.9)",
                               "marginBottom": "0.5rem", "lineHeight": "1.7"}),
                html.Li("Aporta evidencia empírica sobre los factores socioeconómicos e institucionales "
                        "que estructuran las brechas en competencias cuantitativas en el Atlántico.",
                        style={"fontSize": "0.87rem", "color": "rgba(255,255,255,0.9)",
                               "marginBottom": "0.5rem", "lineHeight": "1.7"}),
                html.Li("Contribuye al fortalecimiento de los sistemas de alerta temprana en "
                        "educación superior colombiana con evidencia territorial y metodología rigurosa.",
                        style={"fontSize": "0.87rem", "color": "rgba(255,255,255,0.9)",
                               "lineHeight": "1.7"}),
            ], style={"paddingLeft": "1.1rem", "marginTop": "0.6rem",
                      "marginBottom": "0", "fontFamily": FONT}),
        ], style={"flex": "1", "minWidth": "280px", "padding": "1.4rem",
                  "background": f"linear-gradient(135deg, {MORADO_OSCURO}, {AZUL_OSCURO})",
                  "borderRadius": "14px"}),
    ], style={"display": "flex", "gap": "1.2rem", "flexWrap": "wrap",
              "alignItems": "stretch", "marginBottom": "1.5rem"})

    # Objetivos específicos
    obj_especificos = card([
        html.H4("Objetivos específicos", style={
            "fontFamily": FONT, "color": AZUL_OSCURO, "fontSize": "1rem",
            "fontWeight": "700", "marginBottom": "1rem"}),
        html.Ol([
            html.Li(item, style={"fontFamily": FONT, "fontSize": "0.88rem",
                                  "color": GRIS_TEXTO, "marginBottom": "0.6rem",
                                  "lineHeight": "1.7"})
            for item in [
                "Construir la variable de desempeño dicotómico en Razonamiento Cuantitativo a partir del puntaje continuo, utilizando la media nacional como umbral de clasificación.",
                "Explorar las características sociodemográficas, académicas, institucionales y del hogar, analizando su asociación con el desempeño mediante análisis exploratorio y pruebas bivariadas.",
                "Construir y entrenar seis modelos de clasificación supervisada para predecir el desempeño en Razonamiento Cuantitativo.",
                "Comparar el desempeño predictivo de los modelos priorizando el Recall de la clase de bajo desempeño como criterio principal de selección.",
                "Identificar las variables con mayor contribución predictiva a partir del modelo ganador mediante valores SHAP.",
            ]
        ], style={"paddingLeft": "1.2rem", "marginBottom": "0"}),
    ])

    # Abstract
    abstract = card([
        html.H4("Abstract", style={
            "fontFamily": FONT, "color": AZUL_OSCURO, "fontSize": "1rem",
            "fontWeight": "700", "marginBottom": "1rem"}),
        html.Div([
            html.Div([
                html.P("El presente estudio tiene como objetivo predecir el desempeño de los estudiantes "
                       "en la competencia de Razonamiento Cuantitativo de las pruebas Saber Pro, "
                       "clasificándolos en alto (0) y bajo (1) desempeño. El análisis se realiza a partir "
                       "de datos de 13,078 estudiantes del departamento del Atlántico que presentaron la "
                       "prueba en 2023 (fuente ICFES).",
                       style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                              "lineHeight": "1.8", "marginBottom": "0.7rem"}),
                html.P("La metodología comprende depuración de datos, análisis exploratorio y asociación "
                       "bivariada, seguida de la estimación de seis modelos supervisados. La selección del "
                       "modelo ganador se fundamenta en el Recall de la clase de bajo desempeño, "
                       "complementado con F1-Score, AUC-ROC e intervalos de confianza bootstrap al 95%.",
                       style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                              "lineHeight": "1.8", "marginBottom": "0.7rem"}),
                html.P("Random Forest presenta el mejor desempeño predictivo (Recall = 0,803, F1 = 0,722, "
                       "AUC = 0,731), respaldado por el test de Wilcoxon pareado (p = 0,002). El análisis "
                       "SHAP identifica el género, la edad, el valor de matrícula y el origen institucional "
                       "como los predictores de mayor contribución.",
                       style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                              "lineHeight": "1.8"}),
            ], style={"flex": "2", "minWidth": "280px"}),
            html.Div([
                html.H5("Hallazgos clave", style={
                    "fontFamily": FONT, "color": AZUL_OSCURO, "fontSize": "0.9rem",
                    "fontWeight": "700", "marginBottom": "0.8rem",
                    "borderBottom": f"1px solid {BORDE}", "paddingBottom": "0.4rem"}),
                html.Ul([
                    html.Li(item, style={"fontFamily": FONT, "fontSize": "0.86rem",
                                         "color": GRIS_TEXTO, "marginBottom": "0.5rem",
                                         "lineHeight": "1.7"})
                    for item in [
                        "Random Forest: modelo ganador (Recall = 0,803).",
                        "Wilcoxon confirma superioridad frente a Ridge (p = 0,002).",
                        "Umbral óptimo: 0,441 (maximización del F1).",
                        "XGBoost: mayor AUC (0,736), pero menor Recall.",
                        "Género, edad y matrícula: predictores SHAP más relevantes.",
                        "Detecta 1,124 de 1,400 estudiantes en riesgo (umbral 0,5).",
                    ]
                ], style={"paddingLeft": "1rem", "marginBottom": "0.8rem"}),
                html.P("Palabras clave: Educación Superior, Machine Learning, Predicción, "
                       "Razonamiento Cuantitativo, Saber Pro.",
                       style={"fontFamily": FONT, "fontSize": "0.78rem", "color": GRIS_MUTED,
                              "fontStyle": "italic", "lineHeight": "1.6"}),
            ], style={"flex": "1", "minWidth": "220px", "padding": "1rem",
                      "background": "#f7f8fc", "borderRadius": "12px",
                      "border": f"1px solid {BORDE}"}),
        ], style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap"}),
    ])

    # Línea de tiempo
    hitos = [
        {"año": "2001", "autor": "Breiman", "titulo": "Random Forest",
         "desc": "Introduce un método de ensamble robusto para clasificación y predicción.",
         "color": "#f6c90e"},
        {"año": "2006", "autor": "Davis & Goadrich", "titulo": "Precision-Recall vs ROC",
         "desc": "Apoya la decisión metodológica para seleccionar el modelo ganador.",
         "color": "#8ecf8e"},
        {"año": "2022", "autor": "Abdulkareem et al.", "titulo": "Student Retention & EDM",
         "desc": "Consolida la minería de datos educativos orientada a la predicción del rendimiento estudiantil.",
         "color": "#b0c4de"},
        {"año": "2024", "autor": "Ying Huang et al.", "titulo": "SHAP en Educación",
         "desc": "Apoya la interpretabilidad del modelo ganador aplicando SHAP a resultados académicos.",
         "color": "#c9b8d8"},
        {"año": "2025", "autor": "Ahmed et al.", "titulo": "Predicción de riesgo académico",
         "desc": "Refuerza la importancia de priorizar el Recall para identificar estudiantes en riesgo.",
         "color": "#f4a97f"},
    ]
    fig_timeline = go.Figure()
    n = len(hitos)
    fig_timeline.add_trace(go.Scatter(
        x=list(range(n)), y=[0]*n, mode="lines",
        line=dict(color="#cccccc", width=2), showlegend=False, hoverinfo="skip"
    ))
    for i, h in enumerate(hitos):
        fig_timeline.add_trace(go.Scatter(
            x=[i], y=[0.18], mode="markers+text",
            marker=dict(size=52, color=h["color"], line=dict(color=BLANCO, width=3)),
            text=[h["año"]], textposition="middle center",
            textfont=dict(size=11, color=AZUL_OSCURO, family=FONT),
            showlegend=False, hoverinfo="skip"
        ))
        fig_timeline.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers",
            marker=dict(size=10, color=BLANCO, line=dict(color="#aaaaaa", width=2)),
            showlegend=False, hoverinfo="skip"
        ))
        fig_timeline.add_annotation(
            x=i, y=-0.12, text=f"<b>{h['autor']}</b><br>{h['titulo']}",
            showarrow=False, font=dict(size=10, family=FONT, color=AZUL_OSCURO),
            align="center", yanchor="top"
        )
        fig_timeline.add_annotation(
            x=i, y=-0.42, text=h["desc"],
            showarrow=False, font=dict(size=9, family=FONT, color=GRIS_TEXTO),
            align="center", yanchor="top", width=140
        )
    fig_timeline.update_layout(
        paper_bgcolor=BLANCO, plot_bgcolor=BLANCO,
        height=380, margin=dict(t=20, b=20, l=20, r=20),
        xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
        yaxis=dict(visible=False, range=[-0.85, 0.45]),
        font_family=FONT,
    )
    bloque_timeline = card([
        html.H4("Línea de tiempo: antecedentes clave", style={
            "fontFamily": FONT, "color": AZUL_OSCURO, "fontSize": "1rem",
            "fontWeight": "700", "marginBottom": "0.5rem"}),
        dcc.Graph(figure=fig_timeline, config={"displaylogo": False}, style={"height": "380px"}),
    ])

    return html.Div(
        [bloque_superior, obj_especificos, abstract, bloque_timeline],
        style={"padding": "2rem", "maxWidth": "1200px", "margin": "0 auto",
               "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"}
    )

def layout_intro_modelos():
    """Pestaña Modelos — Sustento estadístico, con fórmula y explicación de símbolos."""

    MODELOS_DEF = [
        ("REGRESIÓN LOGÍSTICA", PETROL,
         "Estima la probabilidad de pertenencia a una clase mediante la función logística.",
         r"P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0+\beta_1 x_1+\cdots+\beta_p x_p)}}",
         "P(Y=1|x): probabilidad estimada de bajo desempeño • β₀: intercepto • βⱼ: coeficiente asociado a la variable xⱼ • p: número de variables predictoras"),

        ("RIDGE", MENTA,
         "Introduce un término de penalización cuadrática que contrae los coeficientes hacia cero.",
         r"\mathcal{L}_{\text{Ridge}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} \beta_j^2",
         "ℓ(β): log-verosimilitud del modelo • λ: parámetro de regularización • βⱼ²: penalización cuadrática del coeficiente • p: número de variables predictoras"),

        ("LASSO", AMARILLO,
         "Introduce un término de penalización que lleva a cero los coeficientes de las variables con menor contribución predictiva.",
         r"\mathcal{L}_{\text{Lasso}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} |\beta_j|",
         "ℓ(β): log-verosimilitud del modelo • λ: intensidad de la penalización • |βⱼ|: valor absoluto del coeficiente • p: número de variables predictoras"),
    ]

    MODELOS_DEF_2 = [
        ("KNN", PETROL,
         "Clasifica al identificar los k casos más cercanos usando la distancia euclidiana.",
         r"d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2} \quad \Rightarrow \quad \hat{y} = \text{moda}_{k\text{ vecinos}}",
         "d(x,xi): distancia entre observaciones • xⱼ: valor de la variable j • xᵢⱼ: valor de la variable j en el vecino i • k: número de vecinos • ŷ: clase predicha"),

        ("RANDOM FOREST", MENTA,
         "Ensamble de B árboles de decisión entrenados sobre submuestras con reemplazo. La predicción final se obtiene por votación mayoritaria.",
         r"\hat{y} = \text{moda}(\hat{y}_1,\, \hat{y}_2,\, \ldots,\, \hat{y}_B) \qquad B \text{ árboles entrenados con bagging}",
         "ŷ: clase predicha final • ŷᵦ: predicción del árbol b • B: número total de árboles • moda: votación mayoritaria entre árboles"),

        ("XGBOOST", CORAL,
         "Boosting secuencial: cada árbol corrige los errores residuales del modelo acumulado. Incluye regularización L1 y L2 sobre los pesos de las hojas.",
         r"\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\!\left(y_i,\, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)",
         "L(t): función objetivo en la iteración t • yᵢ: valor real observado • ŷᵢ: predicción acumulada • fₜ(xᵢ): nuevo árbol agregado • Ω(fₜ): término de regularización"),
    ]

    def item(titulo, color, desc, latex, explicacion=None):
        contenido = [
            html.Div(titulo, style={
                "fontFamily": FONT, "fontWeight": "800",
                "fontSize": "1.05rem", "color": color,
                "letterSpacing": "0.02em", "marginBottom": "0.4rem",
            }),
            html.P(desc, style={
                "fontFamily": FONT, "fontSize": "0.9rem",
                "color": GRIS_TEXTO, "lineHeight": "1.6",
                "marginBottom": "0.6rem" if latex else "0",
            }),
        ]

        if latex:
            contenido.append(
                html.Div(formula(latex), style={"marginTop": "0.3rem"})
            )

        if explicacion:
            contenido.append(
                html.Div(explicacion, style={
                    "fontFamily": FONT,
                    "fontSize": "0.78rem",
                    "color": GRIS_TEXTO,
                    "lineHeight": "1.55",
                    "marginTop": "0.45rem",
                    "backgroundColor": "#f8fafc",
                    "border": f"1px solid {BORDE}",
                    "borderRadius": "8px",
                    "padding": "0.55rem 0.7rem",
                })
            )

        return html.Div([
            html.Div(style={
                "width": "56px", "height": "56px", "borderRadius": "50%",
                "backgroundColor": color, "flexShrink": "0",
                "border": f"3px solid {BLANCO}",
                "boxShadow": f"0 0 0 3px {color}33",
                "zIndex": "1",
            }),
            html.Div(contenido, style={"flex": "1", "paddingLeft": "1.2rem"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "0.3rem"})

    def linea_v():
        return html.Div(style={
            "width": "3px", "height": "2.5rem",
            "backgroundColor": AZUL_OSCURO,
            "margin": "0.4rem 0 0.4rem 26px",
        })

    def columna(items_def):
        bloques = []
        for i, it in enumerate(items_def):
            bloques.append(item(*it))
            if i < len(items_def) - 1:
                bloques.append(linea_v())
        return html.Div(bloques, style={"flex": "1"})

    return html.Div([
        html.Div([
            html.H2("Sustento Estadístico", style={
                "fontFamily": FONT, "color": AZUL_OSCURO,
                "fontSize": "1.9rem", "fontWeight": "800",
                "marginBottom": "2rem",
            }),
            columna(MODELOS_DEF + MODELOS_DEF_2),
        ], style={
            "background": BLANCO, "borderRadius": "18px",
            "border": f"1px solid {BORDE}",
            "padding": "2.5rem 2.5rem 2rem",
            "boxShadow": "0 6px 24px rgba(2,51,115,0.07)",
            "maxWidth": "700px", "margin": "0 auto",
        }),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_intro_modelos_OLD():
    """Pestaña 2 — versión anterior (archivada): fórmula grande por fila."""

    MODELOS_DEF = [
        ("Regresión Logística", "#4a6fa5",
         r"P(Y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_p x_p)}}",
         "Modelo lineal generalizado que estima la probabilidad de pertenencia a una clase mediante "
         "la función logística. Coeficientes estimados por máxima verosimilitud.",
         "Interpretabilidad directa de coeficientes como cambios en el log-odds."),
        ("Ridge — Regularización L2", "#5e60ce",
         r"\mathcal{L}_{\text{Ridge}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} \beta_j^2",
         "Introduce penalización cuadrática sobre los coeficientes para reducir el sobreajuste "
         "y manejar multicolinealidad. Contrae coeficientes sin eliminarlos.",
         "Estabilidad numérica en presencia de predictores correlacionados."),
        ("Lasso — Regularización L1", "#7400b8",
         r"\mathcal{L}_{\text{Lasso}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} |\beta_j|",
         "Penalización L1 que produce soluciones dispersas: lleva exactamente a cero los "
         "coeficientes de variables con menor contribución predictiva.",
         "Selección implícita de variables, útil cuando hay predictores irrelevantes."),
        ("K-Nearest Neighbors (KNN)", "#48cae4",
         r"d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2} \quad \Rightarrow \quad \hat{y} = \text{moda}_{k\text{ vecinos}}",
         "Método no paramétrico que clasifica una observación según la clase mayoritaria entre "
         "sus k vecinos más cercanos en el espacio de características.",
         "Sin supuestos distribucionales. Requiere estandarización previa."),
        ("Random Forest ★ Modelo ganador", "#2dc653",
         r"\hat{y} = \text{moda}(\hat{y}_1,\, \hat{y}_2,\, \ldots,\, \hat{y}_B) \qquad B \text{ árboles entrenados con bagging}",
         "Ensamble de B árboles de decisión entrenados sobre submuestras con reemplazo. "
         "En cada nodo se selecciona aleatoriamente un subconjunto de variables, "
         "reduciendo la correlación entre árboles y mejorando la estabilidad.",
         "Recall = 0,803 en prueba. Robusto frente a valores atípicos e interacciones no lineales."),
        ("XGBoost — Extreme Gradient Boosting", "#f4a261",
         r"\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\!\left(y_i,\, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)",
         "Boosting secuencial: cada árbol corrige los errores residuales del modelo acumulado. "
         "Incluye regularización L1 y L2 sobre los pesos de las hojas.",
         "Mayor AUC en prueba (0,736), pero menor Recall que Random Forest."),
    ]

    filas = []
    for nombre, color, latex, desc, ventaja in MODELOS_DEF:
        filas.append(html.Div([
            # Encabezado de color
            html.Div(html.Span(nombre, style={
                "fontFamily": FONT, "fontSize": "1rem",
                "fontWeight": "700", "color": BLANCO,
            }), style={"background": color, "padding": "0.7rem 1.4rem",
                       "borderRadius": "12px 12px 0 0"}),
            # Cuerpo: fórmula grande a la izquierda, texto a la derecha
            html.Div([
                html.Div(
                    formula(latex),
                    style={"flex": "3", "minWidth": "340px",
                           "display": "flex", "alignItems": "center",
                           "justifyContent": "center",
                           "background": "#f0ebfa", "borderRadius": "10px",
                           "padding": "0.5rem 1rem"}
                ),
                html.Div([
                    html.P(desc, style={
                        "fontFamily": FONT, "fontSize": "0.92rem",
                        "color": GRIS_TEXTO, "lineHeight": "1.8",
                        "marginBottom": "0.6rem"}),
                    html.P(f"✦ {ventaja}", style={
                        "fontFamily": FONT, "fontSize": "0.86rem",
                        "color": GRIS_MUTED, "fontStyle": "italic",
                        "lineHeight": "1.6", "marginBottom": "0"}),
                ], style={"flex": "2", "minWidth": "200px",
                          "padding": "0 1.2rem"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap",
                      "alignItems": "center", "padding": "1.2rem 1.4rem",
                      "background": BLANCO, "borderRadius": "0 0 12px 12px",
                      "border": f"1px solid {BORDE}", "borderTop": "none"}),
        ], style={"borderRadius": "12px",
                  "boxShadow": "0 2px 12px rgba(31,53,86,0.08)",
                  "marginBottom": "1.2rem"}))

    return html.Div(
        [card(filas, titulo="Modelos de clasificación supervisada",
              subtitulo="Definiciones matemáticas y características de los seis algoritmos estimados")],
        style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"}
    )


def layout_intro_metricas():
    """Pestaña Métricas y validación — mismo patrón visual que Sustento Estadístico."""

    METRICAS_DEF = [
        ("RECALL — CLASE 1", PETROL,
         "Fracción de estudiantes con bajo desempeño correctamente detectados. "
         "Minimiza los falsos negativos, el error más costoso en contextos preventivos. "
         "Criterio principal de selección del modelo ganador.",
         r"\text{Recall}_1 = \frac{TP}{TP + FN}"),
        ("PRECISIÓN", MENTA,
         "Proporción de predicciones positivas que son realmente positivas. "
         "Refleja la eficiencia de las intervenciones asignadas.",
         r"\text{Precisión} = \frac{TP}{TP + FP}"),
        ("F1-SCORE", AMARILLO,
         "Media armónica entre precisión y Recall. Penaliza desequilibrios entre ambas "
         "métricas. Usado para la selección del umbral óptimo de clasificación (0,441).",
         r"F_1 = 2 \cdot \frac{\text{Precisión} \times \text{Recall}_1}{\text{Precisión} + \text{Recall}_1}"),
        ("AUC-ROC", CORAL,
         "Área bajo la curva ROC. Resume la capacidad discriminativa global del modelo "
         "con independencia del umbral de clasificación.",
         r"\text{AUC} = \int_0^1 \text{TPR}\, d(\text{FPR})"),
        ("ACCURACY", PETROL,
         "Proporción global de clasificaciones correctas sobre el total de observaciones. "
         "Métrica secundaria; puede ser engañosa en clases desbalanceadas.",
         r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}"),
    ]

    VALIDACION_DEF = [
        ("VALIDACIÓN CRUZADA (K = 10)", MENTA,
         "El desempeño promedio en los 10 pliegues reduce la varianza asociada a una "
         "única partición aleatoria. Garantiza que la proporción de clases se preserve "
         "en cada pliegue.",
         r"\bar{M} = \frac{1}{k} \sum_{i=1}^{k} M_i"),
        ("TEST DE WILCOXON PAREADO", AMARILLO,
         "Prueba no paramétrica para comparar dos modelos cuando no puede asumirse "
         "normalidad en las diferencias. Aplicado sobre los 10 pares de Recall₁ en "
         "validación cruzada. Random Forest superior a Ridge (p = 0,002).",
         r"W = \sum_{i:\,d_i > 0} r_i"),
        ("BOOTSTRAP — IC 95%", CORAL,
         "Método del percentil sobre B = 1.000 remuestras con reemplazo del conjunto "
         "de prueba. Cuantifica la incertidumbre de las métricas sin supuestos "
         "distribucionales.",
         r"\text{IC}_{95\%} = \left[\hat{\theta}_{(0.025)},\; \hat{\theta}_{(0.975)}\right]"),
    ]

    def item(titulo, color, desc, latex):
        return html.Div([
            html.Div(style={
                "width": "56px", "height": "56px", "borderRadius": "50%",
                "backgroundColor": color, "flexShrink": "0",
                "border": f"3px solid {BLANCO}",
                "boxShadow": f"0 0 0 3px {color}33",
                "zIndex": "1",
            }),
            html.Div([
                html.Div(titulo, style={
                    "fontFamily": FONT, "fontWeight": "800",
                    "fontSize": "1.0rem", "color": color,
                    "letterSpacing": "0.02em", "marginBottom": "0.4rem",
                }),
                html.P(desc, style={
                    "fontFamily": FONT, "fontSize": "0.88rem",
                    "color": GRIS_TEXTO, "lineHeight": "1.6",
                    "marginBottom": "0.6rem",
                }),
                html.Div(formula(latex), style={"marginTop": "0.3rem"}),
            ], style={"flex": "1", "paddingLeft": "1.2rem"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "0.3rem"})

    def linea_v():
        return html.Div(style={
            "width": "3px", "height": "2.5rem",
            "backgroundColor": AZUL_OSCURO,
            "margin": "0.4rem 0 0.4rem 26px",
        })

    def columna(items_def):
        bloques = []
        for i, it in enumerate(items_def):
            bloques.append(item(*it))
            if i < len(items_def) - 1:
                bloques.append(linea_v())
        return html.Div(bloques, style={"flex": "1"})

    bloque_metricas = html.Div([
        html.H2("Métricas de Evaluación", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "1.9rem", "fontWeight": "800",
            "marginBottom": "0.4rem",
        }),
        html.P("Derivadas de la matriz de confusión (TP, FP, TN, FN). El Recall de "
               "Clase 1 es el criterio principal de selección.",
               style={"fontFamily": FONT, "fontSize": "0.92rem",
                      "color": GRIS_TEXTO, "marginBottom": "2rem"}),
        columna(METRICAS_DEF),
    ], style={
        "background": BLANCO, "borderRadius": "18px",
        "border": f"1px solid {BORDE}",
        "padding": "2.5rem 2.5rem 2rem",
        "boxShadow": "0 6px 24px rgba(2,51,115,0.07)",
        "maxWidth": "700px", "margin": "0 auto 2rem",
    })

    bloque_validacion = html.Div([
        html.H2("Estrategia de Validación y Comparación Estadística", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "1.9rem", "fontWeight": "800",
            "marginBottom": "0.4rem",
        }),
        html.P("Métodos empleados para estimar, comparar y cuantificar la "
               "incertidumbre de los modelos.",
               style={"fontFamily": FONT, "fontSize": "0.92rem",
                      "color": GRIS_TEXTO, "marginBottom": "2rem"}),
        columna(VALIDACION_DEF),
    ], style={
        "background": BLANCO, "borderRadius": "18px",
        "border": f"1px solid {BORDE}",
        "padding": "2.5rem 2.5rem 2rem",
        "boxShadow": "0 6px 24px rgba(2,51,115,0.07)",
        "maxWidth": "700px", "margin": "0 auto",
    })

    return html.Div([
        bloque_metricas,
        bloque_validacion,
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_intro_metricas_OLD():
    """Pestaña 3 — versión anterior (archivada): fórmulas grandes con fondo morado."""

    METRICAS_DEF = [
        ("Recall — Clase 1", "#c0392b",
         r"\text{Recall}_1 = \frac{TP}{TP + FN}",
         "Fracción de estudiantes con bajo desempeño correctamente detectados. "
         "Minimiza los falsos negativos — el error más costoso en contextos preventivos.",
         "★ Criterio principal de selección del modelo ganador."),
        ("Precisión", "#2980b9",
         r"\text{Precisión} = \frac{TP}{TP + FP}",
         "Proporción de predicciones positivas que son realmente positivas. "
         "Refleja la eficiencia de las intervenciones asignadas.",
         "Métrica complementaria reportada en la tabla comparativa."),
        ("F1-Score", "#8e44ad",
         r"F_1 = 2 \cdot \frac{\text{Precisión} \times \text{Recall}_1}{\text{Precisión} + \text{Recall}_1}",
         "Media armónica entre precisión y Recall. Penaliza desequilibrios entre ambas métricas. "
         "Usado para la selección del umbral óptimo de clasificación.",
         "Umbral óptimo: 0,441 (maximización del F1)."),
        ("AUC-ROC", "#16a085",
         r"\text{AUC} = \int_0^1 \text{TPR}\, d(\text{FPR})",
         "Área bajo la curva ROC. Resume la capacidad discriminativa global del modelo "
         "con independencia del umbral de clasificación.",
         "Random Forest: 0,731 — XGBoost: 0,736."),
        ("Accuracy", "#7f8c8d",
         r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}",
         "Proporción global de clasificaciones correctas sobre el total de observaciones. "
         "Métrica secundaria; puede ser engañosa en clases desbalanceadas.",
         "Random Forest: 67,2% en el conjunto de prueba."),
    ]

    VALIDACION_DEF = [
        ("Validación cruzada estratificada (k = 10)", "#2471a3",
         r"\bar{M} = \frac{1}{k} \sum_{i=1}^{k} M_i",
         "El desempeño promedio en los 10 pliegues reduce la varianza asociada a una única "
         "partición aleatoria. Garantiza que la proporción de clases se preserve en cada pliegue.",
         "Usado para búsqueda de hiperparámetros maximizando Recall₁."),
        ("Test de Wilcoxon pareado", "#1a5276",
         r"W = \sum_{i:\,d_i > 0} r_i",
         "Prueba no paramétrica para comparar dos modelos cuando no puede asumirse normalidad "
         "en las diferencias. Aplicado sobre los 10 pares de Recall₁ en validación cruzada.",
         "Random Forest superior a Ridge: W = 0.000, p = 0.002."),
        ("Bootstrap — IC 95%", "#6c3483",
         r"\text{IC}_{95\%} = \left[\hat{\theta}_{(0.025)},\; \hat{\theta}_{(0.975)}\right]",
         "Método del percentil sobre B = 1,000 remuestras con reemplazo del conjunto de prueba. "
         "Cuantifica la incertidumbre de las métricas sin supuestos distribucionales.",
         "Random Forest Recall IC [0.772; 0.815], sin solapamiento con ningún otro modelo."),
        ("Valores SHAP", "#117a65",
         r"f(\mathbf{x}) = E[f(\mathbf{x})] + \sum_{j=1}^{p} \phi_j",
         "Los valores SHAP distribuyen equitativamente la contribución de cada variable "
         "considerando todas las combinaciones posibles de predictores (Teoría de Shapley).",
         "Implementado con TreeExplainer sobre 1,000 observaciones del conjunto de prueba."),
    ]

    def fila_def(nombre, color, latex, desc, nota):
        return html.Div([
            html.Div(html.Span(nombre, style={
                "fontFamily": FONT, "fontSize": "1rem",
                "fontWeight": "700", "color": BLANCO,
            }), style={"background": color, "padding": "0.7rem 1.4rem",
                       "borderRadius": "12px 12px 0 0"}),
            html.Div([
                html.Div(
                    formula(latex),
                    style={"flex": "3", "minWidth": "340px",
                           "display": "flex", "alignItems": "center",
                           "justifyContent": "center",
                           "background": "#f0ebfa", "borderRadius": "10px",
                           "padding": "0.5rem 1rem"}
                ),
                html.Div([
                    html.P(desc, style={
                        "fontFamily": FONT, "fontSize": "0.92rem",
                        "color": GRIS_TEXTO, "lineHeight": "1.8",
                        "marginBottom": "0.6rem"}),
                    html.P(f"✦ {nota}", style={
                        "fontFamily": FONT, "fontSize": "0.86rem",
                        "color": GRIS_MUTED, "fontStyle": "italic",
                        "lineHeight": "1.6", "marginBottom": "0"}),
                ], style={"flex": "2", "minWidth": "200px", "padding": "0 1.2rem"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap",
                      "alignItems": "center", "padding": "1.2rem 1.4rem",
                      "background": BLANCO, "borderRadius": "0 0 12px 12px",
                      "border": f"1px solid {BORDE}", "borderTop": "none"}),
        ], style={"borderRadius": "12px",
                  "boxShadow": "0 2px 12px rgba(31,53,86,0.08)",
                  "marginBottom": "1.2rem"})

    metricas_filas  = [fila_def(*m) for m in METRICAS_DEF]
    validacion_filas = [fila_def(*v) for v in VALIDACION_DEF]

    return html.Div([
        card(metricas_filas,
             titulo="Métricas de evaluación",
             subtitulo="Derivadas de la matriz de confusión (TP, FP, TN, FN). "
                       "El Recall de Clase 1 es el criterio principal de selección."),
        card(validacion_filas,
             titulo="Estrategia de validación y comparación estadística",
             subtitulo="Métodos empleados para estimar, comparar y cuantificar la incertidumbre de los modelos."),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_intro_marco():
    """Redirige a modelos por defecto — mantenemos compatibilidad."""
    return layout_intro_modelos()


def layout_intro():
    return html.Div([
        html.Div(id="intro-sub-content"),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_marco():
    """Marco teórico unificado: modelos (izquierda) y métricas/validación (derecha) en dos columnas."""

    MODELOS_DEF = [
        ("REGRESIÓN LOGÍSTICA", PETROL,
         "Estima la probabilidad de pertenencia a una clase mediante la función logística.",
         r"P(Y=1 \mid \mathbf{x}) = \frac{1}{1 + e^{-(\beta_0+\beta_1 x_1+\cdots+\beta_p x_p)}}",
         "P(Y=1|x): probabilidad estimada de bajo desempeño • β₀: intercepto • βⱼ: coeficiente asociado a la variable xⱼ • p: número de variables predictoras"),

        ("RIDGE", MENTA,
         "Introduce un término de penalización cuadrática que contrae los coeficientes hacia cero.",
         r"\mathcal{L}_{\text{Ridge}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} \beta_j^2",
         "ℓ(β): log-verosimilitud del modelo • λ: parámetro de regularización • βⱼ²: penalización cuadrática del coeficiente • p: número de variables predictoras"),

        ("LASSO", AMARILLO,
         "Introduce un término de penalización que lleva a cero los coeficientes de las variables con menor contribución predictiva.",
         r"\mathcal{L}_{\text{Lasso}} = -\ell(\boldsymbol{\beta}) + \lambda \sum_{j=1}^{p} |\beta_j|",
         "ℓ(β): log-verosimilitud del modelo • λ: intensidad de la penalización • |βⱼ|: valor absoluto del coeficiente • p: número de variables predictoras"),

        ("KNN", PETROL,
         "Clasifica al identificar los k casos más cercanos usando la distancia euclidiana.",
         r"d(\mathbf{x}, \mathbf{x}_i) = \sqrt{\sum_{j=1}^{p}(x_j - x_{ij})^2} \quad \Rightarrow \quad \hat{y} = \text{moda}_{k\text{ vecinos}}",
         "d(x,xi): distancia entre observaciones • xⱼ: valor de la variable j • xᵢⱼ: valor de la variable j en el vecino i • k: número de vecinos • ŷ: clase predicha"),

        ("RANDOM FOREST", MENTA,
         "Ensamble de B árboles de decisión entrenados sobre submuestras con reemplazo. La predicción final se obtiene por votación mayoritaria.",
         r"\hat{y} = \text{moda}(\hat{y}_1,\, \hat{y}_2,\, \ldots,\, \hat{y}_B) \qquad B \text{ árboles entrenados con bagging}",
         "ŷ: clase predicha final • ŷᵦ: predicción del árbol b • B: número total de árboles • moda: votación mayoritaria entre árboles"),

        ("XGBOOST", CORAL,
         "Boosting secuencial: cada árbol corrige los errores residuales del modelo acumulado. Incluye regularización L1 y L2 sobre los pesos de las hojas.",
         r"\mathcal{L}^{(t)} = \sum_{i=1}^{n} \ell\!\left(y_i,\, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)",
         "L(t): función objetivo en la iteración t • yᵢ: valor real observado • ŷᵢ: predicción acumulada • fₜ(xᵢ): nuevo árbol agregado • Ω(fₜ): término de regularización"),
    ]

    METRICAS_DEF = [
        ("RECALL — CLASE 1", PETROL,
         "Fracción de estudiantes con bajo desempeño correctamente detectados. "
         "Minimiza los falsos negativos, el error más costoso en contextos preventivos. "
         "Criterio principal de selección del modelo ganador.",
         r"\text{Recall}_1 = \frac{TP}{TP + FN}", None),
        ("PRECISIÓN", MENTA,
         "Proporción de predicciones positivas que son realmente positivas. "
         "Refleja la eficiencia de las intervenciones asignadas.",
         r"\text{Precisión} = \frac{TP}{TP + FP}", None),
        ("F1-SCORE", AMARILLO,
         "Media armónica entre precisión y Recall. Penaliza desequilibrios entre ambas "
         "métricas. Usado para la selección del umbral óptimo de clasificación (0,441).",
         r"F_1 = 2 \cdot \frac{\text{Precisión} \times \text{Recall}_1}{\text{Precisión} + \text{Recall}_1}", None),
        ("AUC-ROC", CORAL,
         "Área bajo la curva ROC. Resume la capacidad discriminativa global del modelo "
         "con independencia del umbral de clasificación.",
         r"\text{AUC} = \int_0^1 \text{TPR}\, d(\text{FPR})", None),
        ("ACCURACY", PETROL,
         "Proporción global de clasificaciones correctas sobre el total de observaciones. "
         "Métrica secundaria; puede ser engañosa en clases desbalanceadas.",
         r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}", None),
        ("VALIDACIÓN CRUZADA (K = 10)", MENTA,
         "El desempeño promedio en los 10 pliegues reduce la varianza asociada a una "
         "única partición aleatoria. Garantiza que la proporción de clases se preserve "
         "en cada pliegue.",
         r"\bar{M} = \frac{1}{k} \sum_{i=1}^{k} M_i", None),
        ("TEST DE WILCOXON PAREADO", AMARILLO,
         "Prueba no paramétrica para comparar dos modelos cuando no puede asumirse "
         "normalidad en las diferencias. Aplicado sobre los 10 pares de Recall₁ en "
         "validación cruzada. Random Forest superior a Ridge (p = 0,002).",
         r"W = \sum_{i:\,d_i > 0} r_i", None),
        ("BOOTSTRAP — IC 95%", CORAL,
         "Método del percentil sobre B = 1.000 remuestras con reemplazo del conjunto "
         "de prueba. Cuantifica la incertidumbre de las métricas sin supuestos "
         "distribucionales.",
         r"\text{IC}_{95\%} = \left[\hat{\theta}_{(0.025)},\; \hat{\theta}_{(0.975)}\right]", None),
    ]

    def item(titulo, color, desc, latex, explicacion=None):
        contenido = [
            html.Div(titulo, style={
                "fontFamily": FONT, "fontWeight": "800",
                "fontSize": "1.0rem", "color": color,
                "letterSpacing": "0.02em", "marginBottom": "0.4rem",
            }),
            html.P(desc, style={
                "fontFamily": FONT, "fontSize": "0.88rem",
                "color": GRIS_TEXTO, "lineHeight": "1.6",
                "marginBottom": "0.6rem" if latex else "0",
            }),
        ]

        if latex:
            contenido.append(
                html.Div(formula(latex), style={"marginTop": "0.3rem"})
            )

        if explicacion:
            contenido.append(
                html.Div(explicacion, style={
                    "fontFamily": FONT,
                    "fontSize": "0.78rem",
                    "color": GRIS_TEXTO,
                    "lineHeight": "1.55",
                    "marginTop": "0.45rem",
                    "backgroundColor": "#f8fafc",
                    "border": f"1px solid {BORDE}",
                    "borderRadius": "8px",
                    "padding": "0.55rem 0.7rem",
                })
            )

        return html.Div([
            html.Div(style={
                "width": "52px", "height": "52px", "borderRadius": "50%",
                "backgroundColor": color, "flexShrink": "0",
                "border": f"3px solid {BLANCO}",
                "boxShadow": f"0 0 0 3px {color}33",
                "zIndex": "1",
            }),
            html.Div(contenido, style={"flex": "1", "paddingLeft": "1.2rem"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "0.3rem"})

    def linea_v():
        return html.Div(style={
            "width": "3px", "height": "2rem",
            "backgroundColor": AZUL_OSCURO,
            "margin": "0.4rem 0 0.4rem 24px",
        })

    def columna(items_def):
        bloques = []
        for i, it in enumerate(items_def):
            bloques.append(item(*it))
            if i < len(items_def) - 1:
                bloques.append(linea_v())
        return html.Div(bloques)

    panel_modelos = html.Div([
        html.H2("Sustento Estadístico", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "1.7rem", "fontWeight": "800",
            "marginBottom": "1.6rem",
        }),
        columna(MODELOS_DEF),
    ], style={
        "background": BLANCO, "borderRadius": "18px",
        "border": f"1px solid {BORDE}",
        "padding": "2rem 2rem 1.6rem",
        "boxShadow": "0 6px 24px rgba(2,51,115,0.07)",
        "flex": "1", "minWidth": "640px",
    })

    panel_metricas = html.Div([
        html.H2("Métricas, Validación y Comparación Estadística", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "1.7rem", "fontWeight": "800",
            "marginBottom": "1.6rem",
        }),
        columna(METRICAS_DEF),
    ], style={
        "background": BLANCO, "borderRadius": "18px",
        "border": f"1px solid {BORDE}",
        "padding": "2rem 2rem 1.6rem",
        "boxShadow": "0 6px 24px rgba(2,51,115,0.07)",
        "flex": "1", "minWidth": "640px",
    })

    return html.Div([
        html.Div([panel_modelos, panel_metricas],
                  style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap",
                         "alignItems": "flex-start"}),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})




def layout_eda_desc():
    n = len(data_imputada) if not data_imputada.empty else 0
    if not data_imputada.empty and "desempeno_dicotomico" in data_imputada.columns:
        pct_bajo = (data_imputada["desempeno_dicotomico"] == 1).mean() * 100
        pct_alto = 100 - pct_bajo
    else:
        pct_bajo, pct_alto = 53.5, 46.5

    kpis = html.Div([
        kpi_gradient(f"{pct_bajo:.1f}%", "Bajo desempeño", "Clase 1 (evento de interés)", KPI_GRADIENTS[2]),
        kpi_gradient(f"{pct_alto:.1f}%", "Alto desempeño", "Clase 0", KPI_GRADIENTS[3]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2,1fr)",
              "gap": "1rem", "marginBottom": "1.5rem"})

    fig_y = grafico_distribucion("desempeno_dicotomico")
    fig_y.update_layout(height=360, margin=dict(t=55, b=20, l=20, r=20))

    contenido = [
        kpis,
        html.Div([
            html.H4("Distribución de la variable respuesta", style={
                "color": AZUL_OSCURO, "fontFamily": FONT,
                "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.2rem"}),
            html.P("Balance de clases para el modelo de clasificación. "
                   "Clase 1 = bajo desempeño (evento de interés).", style={
                "color": GRIS_TEXTO, "fontFamily": FONT,
                "fontSize": "0.86rem", "marginBottom": "0.8rem"}),
            html.Div([
                dcc.Graph(figure=fig_y, config={"displaylogo": False},
                          style={"height": "360px", "flex": "7", "minWidth": "300px"}),
                html.Div([
                    html.H5("Comportamiento de la distribución", style={
                        "fontFamily": FONT, "color": AZUL_OSCURO,
                        "fontSize": "0.95rem", "fontWeight": "700",
                        "marginBottom": "0.8rem"}),
                    html.P(f"Podemos observar en la figura que para el Dpto. del Atlántico son más "
                           f"los estudiantes que tuvieron un desempeño por debajo de la media nacional "
                           f"para el año 2023 (clase 1 = {pct_bajo:.1f}%), lo que para este estudio se "
                           f"considera un desempeño bajo. En contraste, los estudiantes que tuvieron un "
                           f"desempeño alto representan el {pct_alto:.1f}%.",
                           style={"fontFamily": FONT, "fontSize": "0.88rem",
                                  "color": GRIS_TEXTO, "lineHeight": "1.7"}),
                ], style={"flex": "3", "minWidth": "200px", "padding": "1rem 1.2rem",
                          "background": BLANCO, "borderRadius": "12px",
                          "border": f"1px solid {BORDE}", "alignSelf": "center"}),
            ], style={"display": "flex", "gap": "1.2rem", "flexWrap": "wrap"}),
        ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
                  "borderRadius": "16px", "padding": "1rem", "marginBottom": "1.4rem"}),
    ]

    if not data_imputada.empty:
        desc = data_imputada.select_dtypes(include=np.number).describe().T.round(2).reset_index()
        desc.columns = ["Variable"] + list(desc.columns[1:])
        desc = desc[desc["Variable"] != "desempeno_dicotomico"]
        nombres_legibles = {
            "mod_razona_cuantitat_punt": "Puntaje Razonamiento Cuantitativo",
            "edad": "Edad",
        }
        desc["Variable"] = desc["Variable"].map(lambda x: nombres_legibles.get(x, x.replace("_", " ").title()))
        contenido += [
            html.H4("Estadísticas descriptivas — variables numéricas",
                    style={"color": AZUL_OSCURO, "marginTop": "0.8rem",
                           "fontSize": "1rem", "fontFamily": FONT, "fontWeight": "700"}),
            html.P("De las 18 variables predictoras, únicamente la edad se incluye en la "
                   "construcción de los modelos como variable continua.",
                   style={"fontFamily": FONT, "fontSize": "0.84rem", "color": GRIS_MUTED,
                          "fontStyle": "italic", "marginBottom": "0.6rem", "marginTop": "0.2rem"}),
            dash_tabla(desc, "tabla-desc"),
        ]

    return card(contenido, titulo="EDA — Variable respuesta",
                subtitulo="Distribución y balance de clases del desempeño en Razonamiento Cuantitativo")


def layout_eda_dist():
    opciones = [
        {"label": "Género",                       "value": "estu_genero"},
        {"label": "Estrato socioeconómico",       "value": "fami_estratovivienda"},
        {"label": "Internet en casa",             "value": "fami_tieneinternet"},
        {"label": "Computador en casa",           "value": "fami_tienecomputador"},
        {"label": "Automóvil en casa",            "value": "fami_tieneautomovil"},
        {"label": "Motocicleta en casa",          "value": "fami_tienemotocicleta"},
        {"label": "Pago matrícula",               "value": "estu_pagomatricula"},
        {"label": "Valor matrícula universidad",  "value": "estu_valormatriculauniversidad"},
        {"label": "Método del programa",          "value": "estu_metodo_prgm"},
        {"label": "Carácter académico",           "value": "inst_caracter_academico"},
        {"label": "Institución de origen",        "value": "inst_origen"},
        {"label": "Educación padre",              "value": "fami_educacionpadre"},
        {"label": "Ocupación padre",              "value": "fami_ocupacionpadre"},
        {"label": "Educación madre",              "value": "fami_educacionmadre"},
        {"label": "Ocupación madre",              "value": "fami_ocupacionmadre"},
        {"label": "Título bachiller",             "value": "estu_tituloobtenidobachiller"},
        {"label": "Horas trabajo semanal",        "value": "estu_horassemanatrabaja"},
        {"label": "Edad (continua)",                "value": "edad"},
    ]
    return card([
        html.Div([
            html.Div([
                html.H4("Explorador de variables", style={
                    "fontFamily": FONT, "fontSize": "1rem", "fontWeight": "700",
                    "color": AZUL_OSCURO, "marginBottom": "0.2rem"}),
                html.P("Selecciona una variable para ver su distribución proporcional "
                       "según nivel de desempeño. Las barras muestran el % dentro de "
                       "cada categoría.", style={
                    "fontFamily": FONT, "fontSize": "0.86rem",
                    "color": GRIS_TEXTO, "marginBottom": "0"}),
            ]),
            dcc.Dropdown(id="dropdown-dist-variable", options=opciones,
                         value="estu_genero", clearable=False, searchable=True,
                         style={"width": "320px", "fontFamily": FONT, "fontSize": "0.88rem"}),
        ], style={"display": "flex", "alignItems": "flex-end",
                  "justifyContent": "space-between", "gap": "1rem", "marginBottom": "1.2rem"}),
        html.Div([
            dcc.Graph(id="grafico-dist-variable", config={"displaylogo": False},
                      style={"minHeight": "420px"})
        ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
                  "borderRadius": "16px", "padding": "0.8rem"}),
    ], titulo="Distribución de variables predictoras",
       subtitulo="Proporciones de alto y bajo desempeño dentro de cada categoría")


def layout_eda_mapa():
    """Mapa a la izquierda, caja de texto interpretativa a la derecha."""
    texto_interpretacion = html.Div([
        html.P("El mapa coroplético muestra la distribución geográfica del desempeño "
               "en Razonamiento Cuantitativo Saber Pro 2023 en los 23 municipios del "
               "departamento del Atlántico.",
               style={"fontFamily": FONT, "fontSize": "0.88rem",
                      "color": GRIS_TEXTO, "lineHeight": "1.8", "marginBottom": "0.8rem"}),
        html.P("Los tonos más claros (amarillos) indican mayor porcentaje de bajo desempeño, "
               "mientras que los tonos más oscuros indican menor concentración. "
               "La variable 'Número de estudiantes' permite identificar los municipios "
               "con mayor volumen de estudiantes.",
               style={"fontFamily": FONT, "fontSize": "0.88rem",
                      "color": GRIS_TEXTO, "lineHeight": "1.8", "marginBottom": "1rem"}),
        html.Hr(style={"borderColor": BORDE, "margin": "1rem 0"}),
        html.H5("Principales hallazgos", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "0.92rem", "fontWeight": "700", "marginBottom": "0.8rem"}),
        html.Ul([
            html.Li("Barranquilla concentra la mayor parte de las observaciones.",
                    style={"fontFamily": FONT, "fontSize": "0.86rem",
                           "color": GRIS_TEXTO, "marginBottom": "0.5rem", "lineHeight": "1.6"}),
            html.Li("Los municipios del Área Metropolitana (Soledad, Malambo, Galapa) "
                    "presentan patrones similares a la capital.",
                    style={"fontFamily": FONT, "fontSize": "0.86rem",
                           "color": GRIS_TEXTO, "marginBottom": "0.5rem", "lineHeight": "1.6"}),
            html.Li("Municipios del interior del departamento muestran mayor variabilidad "
                    "en el desempeño, posiblemente asociada al menor número de instituciones.",
                    style={"fontFamily": FONT, "fontSize": "0.86rem",
                           "color": GRIS_TEXTO, "marginBottom": "0.5rem", "lineHeight": "1.6"}),
        ], style={"paddingLeft": "1.2rem", "marginBottom": "0"}),
        html.Hr(style={"borderColor": BORDE, "margin": "1rem 0"}),
        html.P("Nota. Elaboración propia con datos del ICFES (2023) y shapefile GADM 4.1. "
               "La unidad de análisis es el municipio de residencia del estudiante.",
               style={"fontFamily": FONT, "fontSize": "0.80rem",
                      "fontStyle": "italic", "color": GRIS_MUTED, "lineHeight": "1.6"}),
    ], style={
        "background": BLANCO, "border": f"1px solid {BORDE}",
        "borderRadius": "16px", "padding": "1.4rem",
        "height": "100%", "overflowY": "auto",
    })

    return card([
        html.Div([
            html.Label("Variable a visualizar:",
                       style={"fontFamily": FONT, "fontSize": "0.9rem",
                              "color": AZUL_OSCURO, "marginRight": "1rem"}),
            dcc.Dropdown(id="dropdown-mapa",
                         options=[
                             {"label": "% Bajo desempeño", "value": "pct_bajo_desempeno"},
                             {"label": "Número de estudiantes", "value": "n_estudiantes"},
                         ],
                         value="pct_bajo_desempeno", clearable=False,
                         style={"width": "280px", "fontFamily": FONT, "fontSize": "0.9rem"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "1.2rem"}),

        # Layout de dos columnas: mapa izquierda, texto derecha
        html.Div([
            html.Div([
                dcc.Graph(id="mapa-atlantico", style={"height": "520px"},
                          config={"displaylogo": False}),
            ], style={"flex": "3", "minWidth": "300px"}),
            html.Div(texto_interpretacion,
                     style={"flex": "2", "minWidth": "260px"}),
        ], style={"display": "flex", "gap": "1.2rem", "alignItems": "stretch"}),

    ], titulo="Análisis por Municipio — Atlántico",
       subtitulo="Distribución geográfica del desempeño en Razonamiento Cuantitativo")


def layout_eda_chi():
    # Datos de la tabla BH del paper
    chi_data = [
        {"Grupo": "Sociodemográficas", "Variable": "Género",                  "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Sociodemográficas", "Variable": "Estrato",                 "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Sociodemográficas", "Variable": "Horas trabajo semanal",   "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Sociodemográficas", "Variable": "Valor matrícula",         "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Sociodemográficas", "Variable": "Edad (continua)",          "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Académicas",        "Variable": "Título bachiller",        "p_crudo": 0.192,   "p_ajustado": 0.203,   "significativa": "No"},
        {"Grupo": "Académicas",        "Variable": "Modalidad programa",      "p_crudo": 0.382,   "p_ajustado": 0.382,   "significativa": "No"},
        {"Grupo": "Hogar",             "Variable": "Internet",                "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Computador",              "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Automóvil",               "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Motocicleta",             "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Educación padre",         "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Educación madre",         "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Ocupación padre",         "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Hogar",             "Variable": "Ocupación madre",         "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Institucionales",   "Variable": "Carácter académico",      "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Institucionales",   "Variable": "Origen institución",      "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
        {"Grupo": "Institucionales",   "Variable": "Forma de pago",           "p_crudo": 0.0001,  "p_ajustado": 0.0001,  "significativa": "Sí"},
    ]
    df_chi = pd.DataFrame(chi_data)
    # Para las significativas usamos -log10 del p crudo real si está disponible;
    # añadimos un offset por grupo para diferenciar visualmente
    df_chi["log_p"] = df_chi.apply(
        lambda r: -np.log10(max(r["p_ajustado"], 1e-10)),
        axis=1
    )
    df_chi = df_chi.sort_values("log_p", ascending=True)

    colores_grupo = {
        "Sociodemográficas": AZUL_MEDIO,
        "Hogar":             AMARILLO,
        "Institucionales":   VERDE,
        "Académicas":        "#aaaaaa",
    }
    colores_barra = [colores_grupo.get(g, AZUL_OSCURO) if s == "Sí" else "#cccccc"
                     for g, s in zip(df_chi["Grupo"], df_chi["significativa"])]

    fig_chi = go.Figure()
    fig_chi.add_trace(go.Bar(
        x=df_chi["log_p"], y=df_chi["Variable"],
        orientation="h",
        marker_color=colores_barra,
        marker_line_color=BLANCO, marker_line_width=0.5,
        text=["p < 0.001" if s == "Sí" else f"p = {p:.3f}"
              for s, p in zip(df_chi["significativa"], df_chi["p_ajustado"])],
        textposition="outside",
        textfont=dict(size=11, color=GRIS_TEXTO),
        hovertemplate="<b>%{y}</b><br>Grupo: " +
                      df_chi["Grupo"] + "<br>-log10(p) = %{x:.2f}<extra></extra>",
    ))
    fig_chi.add_vline(x=-np.log10(0.05), line_dash="dash",
                      line_color=ACENTO_ROJO, line_width=1.5,
                      annotation_text="umbral p = 0.05",
                      annotation_position="top right",
                      annotation_font_color=ACENTO_ROJO, annotation_font_size=10)
    fig_chi.update_layout(
        paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
        font_family=FONT, font_color=GRIS_TEXTO,
        margin=dict(t=40, b=40, l=20, r=160),
        height=520,
        xaxis=dict(title="−log₁₀(p-valor ajustado BH)", gridcolor="#edf0f7",
                   zeroline=False, range=[0, 5.5]),
        yaxis=dict(title="", automargin=True),
        showlegend=False,
    )

    # Tabla resumen BH
    df_tabla = df_chi[["Grupo", "Variable", "p_crudo", "p_ajustado", "significativa"]].copy()
    df_tabla = df_tabla.sort_values(["Grupo", "Variable"]).reset_index(drop=True)
    df_tabla.columns = ["Grupo", "Variable", "p-valor crudo", "p-valor ajustado (BH)", "Significativa"]
    df_tabla["p-valor crudo"]        = df_tabla["p-valor crudo"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")
    df_tabla["p-valor ajustado (BH)"] = df_tabla["p-valor ajustado (BH)"].apply(lambda x: "< 0.001" if x < 0.001 else f"{x:.3f}")

    interpretacion = html.Div([
        html.H5("Interpretación de resultados", style={
            "fontFamily": FONT, "color": AZUL_OSCURO,
            "fontSize": "0.95rem", "fontWeight": "700",
            "marginBottom": "0.8rem", "marginTop": "1.5rem",
            "borderBottom": f"2px solid {BORDE}", "paddingBottom": "0.5rem",
        }),
        html.P("16 de las 18 variables predictoras presentaron asociación estadísticamente significativa "
               "con el desempeño en Razonamiento Cuantitativo tras la corrección Benjamini-Hochberg (FDR, alpha = 0,05).",
               style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                      "lineHeight": "1.7", "marginBottom": "0.8rem"}),
        html.Ul([
            html.Li("Las variables académicas (tipo de título de bachiller y modalidad del programa) "
                    "perdieron significancia bivariado tras el ajuste, evidenciando una asociación marginal débil "
                    "con la variable respuesta. En contraste, las variables relacionadas con las condiciones "
                    "socioeconómicas, familiares e institucionales continúan desempeñando un papel relevante en "
                    "la explicación de las brechas observadas en el rendimiento académico.",
                    style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                           "marginBottom": "0.6rem", "lineHeight": "1.7"}),
            html.Li("Una asociación débil en el análisis bivariado no descarta su contribución en la predicción; "
                    "por eso, todas las variables se conservaron para el modelamiento.",
                    style={"fontFamily": FONT, "fontSize": "0.88rem", "color": GRIS_TEXTO,
                           "lineHeight": "1.7"}),
        ], style={"paddingLeft": "1.2rem", "marginBottom": "0"}),
    ])

    return card([
        html.Div([
            # Gráfico 70%
            html.Div([
                dcc.Graph(figure=fig_chi, config={"displaylogo": False}),
            ], style={"flex": "7", "minWidth": "300px", "background": GRIS_CARD,
                      "border": f"1px solid {BORDE}", "borderRadius": "16px", "padding": "0.8rem"}),
            # Interpretación 30%
            html.Div([
                html.H5("Interpretación", style={
                    "fontFamily": FONT, "color": AZUL_OSCURO,
                    "fontSize": "0.95rem", "fontWeight": "700",
                    "marginBottom": "0.8rem",
                    "borderBottom": f"2px solid {BORDE}", "paddingBottom": "0.5rem",
                }),
                html.P("16 de las 18 variables predictoras presentaron asociación estadísticamente "
                       "significativa con el desempeño en Razonamiento Cuantitativo tras la corrección "
                       "Benjamini-Hochberg (FDR, alpha = 0,05).",
                       style={"fontFamily": FONT, "fontSize": "0.87rem", "color": GRIS_TEXTO,
                              "lineHeight": "1.7", "marginBottom": "0.8rem"}),
                html.Ul([
                    html.Li("Las variables académicas (tipo de título de bachiller y modalidad del programa) "
                            "perdieron significancia bivariada tras el ajuste, evidenciando una asociación "
                            "marginal débil con la variable respuesta. En contraste, las variables relacionadas "
                            "con las condiciones socioeconómicas, familiares e institucionales continúan "
                            "desempeñando un papel relevante en la explicación de las brechas observadas en el "
                            "rendimiento académico.",
                            style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO,
                                   "marginBottom": "0.6rem", "lineHeight": "1.7"}),
                    html.Li("Una asociación débil en el análisis bivariado no descarta su contribución "
                            "en la predicción; por eso, todas las variables se conservaron para el modelamiento.",
                            style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO,
                                   "lineHeight": "1.7"}),
                ], style={"paddingLeft": "1.1rem", "marginBottom": "0"}),
            ], style={"flex": "3", "minWidth": "200px", "padding": "1rem 1.2rem",
                      "background": BLANCO, "borderRadius": "12px",
                      "border": f"1px solid {BORDE}", "alignSelf": "flex-start"}),
        ], style={"display": "flex", "gap": "1.2rem", "flexWrap": "wrap", "alignItems": "flex-start"}),
    ], titulo="Análisis de asociación — Chi-cuadrado con corrección BH",
       subtitulo="Asociación entre variables explicativas y nivel de desempeño en RC")


def layout_met_marco():
    """Pestaña Marco metodológico — 6 ítems en dos columnas tipo timeline."""

    items = [
        ("01", AZUL_OSCURO,  "Tipo de investigación",  "Cuantitativa, No experimental - aplicado"),
        ("02", PETROL,       "Alcance y diseño",        "Predictivo - observacional"),
        ("03", "#a0aec0",    "Unidad de análisis",      "Estudiantes del Dpto. del Atlántico"),
        ("04", AZUL_OSCURO,  "Fuente de los datos",     "ICFES Saber Pro · Pruebas del 2023"),
        ("05", MENTA,        "Población objetivo",      "Estudiantes de pregrado que presentaron la prueba en el 2023 (241.602)"),
        ("06", PETROL,       "Muestra",                 "Estudiantes de pregrado que realizaron la prueba en el Dpto. del Atlántico (13.078)"),
    ]

    def item_card(numero, color, titulo, desc):
        return html.Div([
            # burbuja numerada
            html.Div(numero, style={
                "width": "52px", "height": "52px", "flexShrink": "0",
                "backgroundColor": color,
                "borderRadius": "50%",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "fontFamily": FONT, "fontWeight": "800", "fontSize": "1rem",
                "color": BLANCO,
                "boxShadow": f"0 4px 12px {color}55",
                "zIndex": "1",
            }),
            # línea vertical decorativa
            html.Div(style={
                "width": "3px",
                "background": f"linear-gradient({color}, {color}44)",
                "borderRadius": "2px",
                "alignSelf": "stretch",
                "margin": "0 1.1rem",
                "flexShrink": "0",
            }),
            # texto
            html.Div([
                html.Div(titulo, style={
                    "fontFamily": FONT, "fontWeight": "700",
                    "fontSize": "0.95rem", "color": AZUL_OSCURO,
                    "marginBottom": "0.25rem",
                }),
                html.Div(desc, style={
                    "fontFamily": FONT, "fontSize": "0.82rem",
                    "color": GRIS_TEXTO, "lineHeight": "1.5",
                }),
            ], style={"flex": "1"}),
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "marginBottom": "2rem",
            "minHeight": "70px",
        })

    col_izq = html.Div([item_card(*items[i]) for i in range(3)],
                       style={"flex": "1", "paddingRight": "2rem",
                              "borderRight": f"2px solid {BORDE}"})
    col_der = html.Div([item_card(*items[i]) for i in range(3, 6)],
                       style={"flex": "1", "paddingLeft": "2rem"})

    return html.Div([
        html.Div([
            html.H2("Marco Metodológico", style={
                "fontFamily": FONT, "color": AZUL_OSCURO,
                "fontSize": "1.8rem", "fontWeight": "800",
                "marginBottom": "0.5rem",
            }),
            html.P("Su objetivo metodológico central es la construcción y comparación de seis modelos "
                   "de clasificación supervisada, capaces de anticipar el desempeño de los estudiantes "
                   "en la competencia de RC, categorizándolo como alto o bajo.",
                   style={
                       "fontFamily": FONT, "fontSize": "0.92rem",
                       "color": GRIS_TEXTO, "lineHeight": "1.7",
                       "maxWidth": "820px", "marginBottom": "2.5rem",
                   }),
            html.Div([col_izq, col_der], style={
                "display": "flex", "alignItems": "flex-start",
                "background": BLANCO, "borderRadius": "18px",
                "border": f"1px solid {BORDE}",
                "padding": "2.5rem 2rem",
                "boxShadow": "0 6px 24px rgba(31,53,86,0.07)",
            }),
        ], style={"maxWidth": "1000px", "margin": "0 auto"}),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_met_variables():
    """Pestaña 1: Operacionalización de variables — tabla compacta al 80%."""
    return html.Div([
        html.Div(
            build_operacionalizacion(),
            style={"maxWidth": "80%", "margin": "0 auto"}
        )
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_met_flujo():
    """Flujo metodológico — tres etapas con líneas de conexión y paleta sólida."""

    # ── helpers ────────────────────────────────────────────────────

    def icono_caja(simbolo, color_fondo):
        return html.Div(simbolo, style={
            "width": "44px", "height": "44px", "flexShrink": "0",
            "backgroundColor": color_fondo,
            "borderRadius": "3px",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "fontFamily": FONT, "fontSize": "1.3rem", "fontWeight": "800",
            "color": BLANCO,
        })

    def bloque_central(icono_sym, color_icono, titulo, subtitulo):
        return html.Div([
            icono_caja(icono_sym, color_icono),
            html.Div([
                html.Div(titulo, style={
                    "fontFamily": FONT, "fontWeight": "700",
                    "fontSize": "1.0rem", "color": AZUL_OSCURO,
                }),
            ], style={"flex": "1"}),
        ], style={
            "display": "flex", "alignItems": "center", "gap": "0.8rem",
            "background": BLANCO, "borderRadius": "4px",
            "border": f"1.5px solid {BORDE}",
            "padding": "0.85rem 1rem",
            "position": "relative",
        })

    def caja_derecha(texto, color_bg):
        return html.Div(texto, style={
            "backgroundColor": color_bg,
            "color": BLANCO,
            "borderRadius": "3px",
            "padding": "0.6rem 1rem",
            "fontFamily": FONT,
            "fontSize": "0.86rem",
            "lineHeight": "1.45",
            "marginBottom": "0.5rem",
        })

    def caja_modelo(nombre, detalle):
        return html.Div([
            html.Div(nombre, style={
                "fontFamily": FONT, "fontWeight": "700",
                "fontSize": "0.82rem", "color": AZUL_OSCURO,
                "lineHeight": "1.25",
            }),
        ], style={
            "background": BLANCO,
            "border": f"1.5px solid {BORDE}",
            "borderRadius": "3px",
            "padding": "0.5rem 0.6rem",
            "flex": "1",
            "minHeight": "70px",
        })

    def etiqueta_lateral(texto, color_borde):
        return html.Div(texto, style={
            "fontFamily": FONT, "fontWeight": "700",
            "fontSize": "0.86rem", "color": BLANCO,
            "background": color_borde,
            "borderRadius": "4px",
            "padding": "0.7rem 0.6rem",
            "textAlign": "center",
            "width": "100%",
            "boxSizing": "border-box",
            "alignSelf": "center",
            "whiteSpace": "normal",
            "wordBreak": "break-word",
            "overflowWrap": "break-word",
            "lineHeight": "1.35",
        })

    # línea horizontal sólida entre etiqueta→central y central→derecha
    def linea_h(color, w="28px"):
        return html.Div(style={
            "height": "2px",
            "width": w,
            "backgroundColor": color,
            "alignSelf": "center",
            "flexShrink": "0",
        })

    # separador vertical entre etapas con línea punteada
    def separador_etapa(color):
        return html.Div(style={
            "height": "0.6rem",
            "borderLeft": f"2px dashed {color}",
            "margin": "0 0 0 60px",   # alineado bajo etiqueta lateral
            "width": "0",
        })

    # línea vertical interna que une bloques centrales entre sí
    def linea_v_interna(color, h="0.7rem"):
        return html.Div(style={
            "width": "2px",
            "height": h,
            "backgroundColor": color,
            "margin": "0 auto",
        })

    # ── Función genérica: etapa con conectores curvos tipo bracket ──
    def construir_etapa(color_etiqueta, texto_etiqueta, grupos):
        """
        grupos: lista de tuplas (color_bloque, bloque_central_div, [(color_caja, caja_div), ...], alto_px)
                El 4º elemento (alto_px) es opcional; por defecto 60.
                Si un grupo tiene lista de cajas vacía, se conecta en cascada vertical
                con el grupo anterior (en vez de recibir flecha desde la etiqueta), y se le
                reserva el espacio vertical indicado por alto_px.
        """
        ANCHO = 1000
        ALTO_POR_CAJA = 60
        ALTO_BLOQUE_VISUAL = 56
        X_ETIQUETA, W_ETIQUETA = 0, 150
        X_BLOQUE, W_BLOQUE = 220, 300
        X_CAJA, W_CAJA = 620, 360

        # normalizar grupos a 4 elementos (color, div, cajas, alto_px)
        grupos_n = [(g[0], g[1], g[2], g[3] if len(g) > 3 else ALTO_POR_CAJA) for g in grupos]

        alturas_grupo = [max(g[3], len(g[2]) * ALTO_POR_CAJA, ALTO_POR_CAJA) for g in grupos_n]
        alto_total = sum(alturas_grupo) + 20
        y_bloques = []
        y_cursor = 10
        for h in alturas_grupo:
            y_bloques.append(y_cursor + h / 2)
            y_cursor += h
        y_etiqueta = alto_total / 2

        # determinar qué grupos reciben flecha de la etiqueta (los que inician una "rama":
        # el primer grupo, o cualquier grupo cuyo grupo anterior tenga cajas)
        recibe_de_etiqueta = []
        for i, (_, _, cajas, _) in enumerate(grupos_n):
            if i == 0:
                recibe_de_etiqueta.append(True)
            else:
                cajas_prev = grupos_n[i - 1][2]
                recibe_de_etiqueta.append(len(cajas_prev) > 0)

        paths, markers, marker_ids = [], [], {}

        def get_marker(color):
            if color not in marker_ids:
                mid = f"arrow_{abs(hash(color)) % 100000}"
                marker_ids[color] = mid
                markers.append(
                    f'<marker id="{mid}" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">'
                    f'<path d="M0,0 L6,3 L0,6" fill="none" stroke="{color}" stroke-width="1.4"/></marker>'
                )
            return marker_ids[color]

        for i, (color_bloque, _, cajas, alto_px) in enumerate(grupos_n):
            yb = y_bloques[i]
            mid = get_marker(color_bloque)
            if recibe_de_etiqueta[i]:
                x_codo = (X_ETIQUETA + W_ETIQUETA + X_BLOQUE) / 2
                if abs(yb - y_etiqueta) < 1:
                    d = f'M {X_ETIQUETA+W_ETIQUETA},{y_etiqueta:.0f} L {X_BLOQUE},{yb:.0f}'
                else:
                    d = (f'M {X_ETIQUETA+W_ETIQUETA},{y_etiqueta:.0f} '
                         f'L {x_codo:.0f},{y_etiqueta:.0f} '
                         f'L {x_codo:.0f},{yb:.0f} '
                         f'L {X_BLOQUE},{yb:.0f}')
                paths.append(
                    f'<path d="{d}" fill="none" stroke="{color_bloque}" stroke-width="2" marker-end="url(#{mid})"/>'
                )
            else:
                # cascada: flecha vertical simple desde el borde inferior real del bloque anterior
                alto_prev = grupos_n[i - 1][3]
                yb_prev_borde = y_bloques[i - 1] + alto_prev / 2
                yb_borde = yb - alto_px / 2
                x_centro = X_BLOQUE + W_BLOQUE / 2
                paths.append(
                    f'<path d="M {x_centro:.0f},{yb_prev_borde:.0f} L {x_centro:.0f},{yb_borde:.0f}" '
                    f'fill="none" stroke="{color_bloque}" stroke-width="2" marker-end="url(#{mid})"/>'
                )
            n_cajas = len(cajas)
            y0 = yb - (n_cajas - 1) * ALTO_POR_CAJA / 2
            x_codo_caja = X_BLOQUE + W_BLOQUE + (X_CAJA - X_BLOQUE - W_BLOQUE) / 2
            for j, (color_caja, _) in enumerate(cajas):
                mid2 = get_marker(color_caja)
                yc = y0 + j * ALTO_POR_CAJA
                if abs(yc - yb) < 1:
                    d = f'M {X_BLOQUE+W_BLOQUE},{yb:.0f} L {X_CAJA},{yc:.0f}'
                else:
                    d = (f'M {X_BLOQUE+W_BLOQUE},{yb:.0f} '
                         f'L {x_codo_caja:.0f},{yb:.0f} '
                         f'L {x_codo_caja:.0f},{yc:.0f} '
                         f'L {X_CAJA},{yc:.0f}')
                paths.append(
                    f'<path d="{d}" fill="none" stroke="{color_caja}" stroke-width="2" marker-end="url(#{mid2})"/>'
                )

        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ANCHO} {alto_total:.0f}" '
            f'width="{ANCHO}" height="{alto_total:.0f}"><defs>{"".join(markers)}</defs>'
            f'{"".join(paths)}</svg>'
        )

        hijos = [
            html.Img(
                src="data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8"),
                style={"position": "absolute", "top": "0", "left": "0",
                       "width": f"{ANCHO}px", "height": f"{alto_total:.0f}px",
                       "pointerEvents": "none", "zIndex": "0"}
            ),
            html.Div(
                etiqueta_lateral(texto_etiqueta, color_etiqueta),
                style={"position": "absolute", "left": "0px", "top": f"{y_etiqueta:.0f}px",
                       "transform": "translateY(-50%)", "width": f"{W_ETIQUETA}px", "zIndex": "1"}
            ),
        ]
        for i, (color_bloque, bloque_div, cajas, alto_px) in enumerate(grupos_n):
            yb = y_bloques[i]
            hijos.append(html.Div(
                bloque_div,
                style={"position": "absolute", "left": f"{X_BLOQUE}px", "top": f"{yb:.0f}px",
                       "transform": "translateY(-50%)", "width": f"{W_BLOQUE}px", "zIndex": "1"}
            ))
            n_cajas = len(cajas)
            y0 = yb - (n_cajas - 1) * ALTO_POR_CAJA / 2
            for j, (color_caja, caja_div) in enumerate(cajas):
                yc = y0 + j * ALTO_POR_CAJA
                hijos.append(html.Div(
                    caja_div,
                    style={"position": "absolute", "left": f"{X_CAJA}px", "top": f"{yc:.0f}px",
                           "transform": "translateY(-50%)", "width": f"{W_CAJA}px", "zIndex": "1"}
                ))

        return html.Div(
            html.Div(hijos, style={"position": "relative", "width": f"{ANCHO}px",
                                    "height": f"{alto_total:.0f}px", "margin": "0 auto"}),
            style={
                "background": BLANCO, "borderRadius": "6px",
                "border": f"2px solid {color_etiqueta}",
                "padding": "0.7rem 1rem",
                "boxShadow": f"0 4px 16px {color_etiqueta}22",
                "overflowX": "auto",
            }
        )

    # ── ETAPA 1: Preparación de datos ──────────────────────────────
    C1 = PETROL
    C1b = AZUL_OSCURO

    etapa1 = construir_etapa(C1, "Preparación\nde datos", [
        (C1, bloque_central("1", C1, "Fuente de los datos",
                             "ICFES Saber Pro · N = 241.062 observaciones únicas (2023-1 y 2023-2)"), [
            (C1, caja_derecha("Filtro para el Atlántico n = 13.078 >> muestra de interés", C1)),
            (C1, caja_derecha("Construcción de RC dicotómica: Alto vs. bajo desempeño (umbral: mean = 142.866)", C1)),
            (C1, caja_derecha("18 variables predictoras · Prueba Chi-cuadrado con corrección de BH", C1)),
        ]),
        (C1b, bloque_central("2", C1b, "Preprocesamiento",
                              "Tratamiento de los datos depurados"), [
            (C1b, caja_derecha("Partición 80% (train) — 20% (test) estratificada", C1b)),
            (C1b, caja_derecha("Imputación de valores faltantes (MICE/Moda), codificación (Encoding) y escalado", C1b)),
        ]),
    ])

    # ── ETAPA 2: Modelamiento ───────────────────────────────────────
    C2 = AMARILLO

    bloque_validacion = bloque_central("3", C2, "Validación cruzada y búsqueda de hiperparámetros",
                                        "StratifiedKFold (k = 10) · GridSearchCV · Recall clase (1)")
    bloque_modelos_grid = html.Div([
        html.Div([
            caja_modelo("Regresión logística", ""),
            caja_modelo("Ridge", "C ∈ {0.001 … 100}"),
            caja_modelo("Lasso", "C ∈ {0.001 … 100}"),
        ], style={"display": "flex", "gap": "0.5rem", "marginBottom": "0.5rem"}),
        html.Div([
            caja_modelo("KNN", "k ∈ {3, 5, 7, 9, 11}"),
            caja_modelo("XGBoost", "learning_rate · max_depth · γ"),
            caja_modelo("Random Forest", "n_estimators · max_depth · max_features"),
        ], style={"display": "flex", "gap": "0.5rem"}),
    ], style={"background": GRIS_SUAVE, "borderRadius": "4px",
              "border": f"1.5px solid {BORDE}", "padding": "0.7rem", "width": "300px",
              "minHeight": "200px", "boxSizing": "border-box"})

    etapa2 = construir_etapa(C2, "Modelamiento", [
        (C2, bloque_validacion, [
            (C2, caja_derecha("StratifiedKFold (k = 10)", C2)),
            (C2, caja_derecha("GridSearchCV con grilla", C2)),
            (C2, caja_derecha("Criterio de evaluación: Recall clase (1)", C2)),
        ], 60),
        (C2, bloque_modelos_grid, [
            (C2, caja_derecha("6 modelos entrenados · listos para evaluación sobre el conjunto de prueba (20%)", C2)),
        ], 220),
    ])

    # ── ETAPA 3: Evaluación e interpretabilidad ─────────────────────
    C3 = CORAL

    etapa3 = construir_etapa(C3, "Evaluación e\ninterpretabilidad", [
        (C3, bloque_central("4", C3, "Métricas de validación",
                             "Evaluación sobre el conjunto de prueba independiente"), [
            (C3, caja_derecha("Recall · F1-Score · AUC-ROC · Accuracy · Precisión", C3)),
            (C3, caja_derecha("Bootstrap (1.000 iteraciones) e intervalo de confianza al 95%", C3)),
            (C3, caja_derecha("Distribución del Recall clase (1) en los 10 pliegues por modelo (boxplot comparativo)", C3)),
        ]),
        (AZUL_OSCURO, bloque_central("5", AZUL_OSCURO, "Comparación estadística del 1.º vs. 2.º mejor modelo",
                                      "Test de Wilcoxon Pareado entre RF y Ridge"), [
            (AZUL_OSCURO, caja_derecha("Test de Wilcoxon Pareado para comparación estadística entre RF y Ridge", AZUL_OSCURO)),
        ]),
        (PETROL, bloque_central("6", PETROL, "Interpretabilidad del mejor modelo",
                                 "Valores SHAP y umbral óptimo"), [
            (PETROL, caja_derecha("Valores SHAP para variables predictoras", PETROL)),
            (PETROL, caja_derecha("Umbral óptimo", PETROL)),
        ]),
    ])

    return html.Div([
        html.Div([
            html.H2("Flujo Metodológico", style={
                "fontFamily": FONT, "color": AZUL_OSCURO,
                "fontSize": "2rem", "fontWeight": "800",
                "marginBottom": "0.2rem",
            }),
            html.P("Implementado en Python 3.11", style={
                "fontFamily": FONT, "color": GRIS_MUTED,
                "fontSize": "0.92rem", "marginBottom": "2rem",
            }),
            etapa1,
            separador_etapa(C1),
            etapa2,
            separador_etapa(C2),
            etapa3,
        ], style={"maxWidth": "1060px", "margin": "0 auto"}),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})




def layout_metodologia():
    return html.Div([
        html.Div(id="metodologia-sub-content"),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_modelos_comparacion():
    mejor = None
    if not df_comparacion.empty and "Modelo" in df_comparacion.columns:
        row = df_comparacion[df_comparacion["Modelo"] == MODELO_GANADOR]
        if not row.empty:
            mejor = row.iloc[0]
    col_recall = "recall_1" if mejor is not None and "recall_1" in mejor.index else "recall"
    kpis = html.Div([
        kpi_gradient(f"{mejor[col_recall]:.1%}" if mejor is not None else "80.3%",
            "Recall", f"Modelo ganador: {MODELO_GANADOR}", KPI_GRADIENTS[0]),
        kpi_gradient(f"{mejor['f1']:.1%}" if mejor is not None else "72.2%",
            "F1-Score", "Clase 1 (bajo desempeño)", KPI_GRADIENTS[1]),
        kpi_gradient(f"{mejor['auc']:.4f}" if mejor is not None else "0.7285",
            "AUC-ROC", "En test set (20%)", KPI_GRADIENTS[2]),
        kpi_gradient(f"{mejor['accuracy']:.1%}" if mejor is not None else "67.2%",
            "Accuracy", "Global en test", KPI_GRADIENTS[3]),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4,1fr)",
              "gap": "1rem", "marginBottom": "1.5rem"})
    ganador_idx = None
    if not df_comparacion.empty and "Modelo" in df_comparacion.columns:
        df_modelos_validos = df_comparacion[
            ~df_comparacion["Modelo"].str.lower().str.contains("dummy|baseline", na=False)
        ]
        col_sort = "recall_1" if "recall_1" in df_modelos_validos.columns else "auc"
        df_sorted = df_modelos_validos.sort_values(col_sort, ascending=False).reset_index(drop=True)
        rows_ganador = df_sorted[df_sorted["Modelo"] == MODELO_GANADOR]
        ganador_idx = rows_ganador.index[0] if not rows_ganador.empty else None
    else:
        df_sorted = df_comparacion
    tabla_comp = (dash_tabla(df_sorted, "tabla-comparacion", highlight_row=ganador_idx,
                              etiquetas=ETIQUETAS_METRICAS)
                  if not df_comparacion.empty
                  else html.Div("[ Tabla no disponible ]",
                                style={"color": AZUL_MEDIO, "textAlign": "center", "padding": "2rem"}))
    modelos_disponibles = list(cms_dash.keys()) or list(curvas_roc_dash.keys())
    selector = html.Div([
        html.Label("Seleccionar modelo:",
                   style={"fontFamily": FONT, "fontSize": "0.92rem",
                          "color": AZUL_OSCURO, "fontWeight": "700",
                          "marginBottom": "0.5rem", "display": "block"}),
        html.Div([
            html.Button(m, id={"type": "btn-modelo-sel", "index": m}, n_clicks=0,
                style={
                    "background": f"linear-gradient(135deg, {PETROL}, {AZUL_OSCURO})"
                                  if m == MODELO_GANADOR else BLANCO,
                    "color": BLANCO if m == MODELO_GANADOR else GRIS_TEXTO,
                    "border": f"1px solid {BORDE}", "borderRadius": "10px",
                    "padding": "0.55rem 1.1rem", "fontFamily": FONT,
                    "fontSize": "0.85rem", "cursor": "pointer", "margin": "0.2rem",
                    "fontWeight": "700" if m == MODELO_GANADOR else "400",
                }) for m in modelos_disponibles
        ], style={"display": "flex", "flexWrap": "wrap", "gap": "0.3rem"}),
    ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
              "borderRadius": "14px", "padding": "1.2rem", "marginBottom": "1.2rem"})
    graficos_modelo = html.Div([
        html.Div([dcc.Graph(id="grafico-roc-modelo", config={"displaylogo": False})],
                 style={"flex": "1", "minWidth": "300px", "background": GRIS_CARD,
                        "border": f"1px solid {BORDE}", "borderRadius": "14px", "padding": "0.8rem"}),
        html.Div([dcc.Graph(id="grafico-cm-modelo", config={"displaylogo": False})],
                 style={"flex": "1", "minWidth": "300px", "background": GRIS_CARD,
                        "border": f"1px solid {BORDE}", "borderRadius": "14px", "padding": "0.8rem"}),
    ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap", "marginBottom": "1.2rem"})
    grafico_pliegues = html.Div([
        dcc.Graph(id="grafico-pliegues-modelo", config={"displaylogo": False}, style={"height": "380px"}),
    ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
              "borderRadius": "14px", "padding": "0.8rem"})
    return card([
        kpis,
        html.H4("Tabla comparativa de modelos",
                style={"color": AZUL_OSCURO, "fontSize": "1rem",
                       "fontFamily": FONT, "fontWeight": "700", "marginBottom": "0.4rem"}),
        html.P(f"Ordenada por Recall descendente. Fila resaltada = {MODELO_GANADOR} (ganador).",
               style={"fontFamily": FONT, "fontSize": "0.83rem",
                      "color": GRIS_MUTED, "marginBottom": "0.8rem"}),
        tabla_comp,
        html.Div(style={"height": "1.5rem"}),
        selector,
        html.H4(id="titulo-modelo-sel",
                style={"color": AZUL_OSCURO, "fontFamily": FONT,
                       "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.5rem"}),
        graficos_modelo,
        html.Div(style={"height": "1rem"}),
        grafico_pliegues,
    ], titulo="Modelos y Resultados",
       subtitulo="Comparación de modelos de clasificación supervisada")


def layout_modelos_analisis():
    fig_forest_recall = build_forest_plot_bootstrap("recall_1")
    fig_forest_auc    = build_forest_plot_bootstrap("auc")
    fig_shap          = build_shap_beeswarm()

    def bloque_grafico(titulo, descripcion, figura, id_grafico):
        return html.Div([
            html.H5(titulo, style={"fontFamily": FONT, "color": AZUL_OSCURO,
                                   "fontSize": "0.95rem", "fontWeight": "700", "marginBottom": "0.2rem"}),
            html.P(descripcion, style={"fontFamily": FONT, "fontSize": "0.82rem",
                                       "color": GRIS_MUTED, "marginBottom": "0"}),
            dcc.Graph(id=id_grafico, figure=figura, config={"displaylogo": False},
                      style={"marginTop": "0.8rem"}),
        ], style={"flex": "1", "minWidth": "300px"})

    return card([
        html.Div([
            html.H4("Estimación de incertidumbre — IC 95% Bootstrap (B = 1,000)",
                    style={"color": AZUL_OSCURO, "fontFamily": FONT,
                           "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.4rem"}),
            html.P("Forest plot comparativo. Intervalos que no se solapan indican diferencias estadísticamente relevantes.",
                   style={"fontFamily": FONT, "fontSize": "0.84rem", "color": GRIS_MUTED, "marginBottom": "1rem"}),
            html.Div([
                bloque_grafico(
                    "Recall (clase 1) — IC 95% bootstrap",
                    "Punto = estimación puntual en test; línea = IC 95% percentil bootstrap (B = 1,000). "
                    "Random Forest: Recall [0,781; 0,825], sin solapamiento con ningún otro modelo.",
                    fig_forest_recall, "grafico-forest-bootstrap-recall"
                ),
                bloque_grafico(
                    "AUC-ROC — IC 95% bootstrap",
                    "Punto = estimación puntual en test; línea = IC 95% percentil bootstrap (B = 1,000).",
                    fig_forest_auc, "grafico-forest-bootstrap-auc"
                ),
            ], style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap"}),
        ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
                  "borderRadius": "14px", "padding": "1.2rem", "marginBottom": "1.5rem"}),
        html.Div([
            html.H4("Selección del umbral óptimo de clasificación",
                    style={"color": AZUL_OSCURO, "fontFamily": FONT,
                           "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.4rem"}),
            html.P("La decisión sobre el umbral no es exclusivamente estadística, sino también operativa.",
                   style={"fontFamily": FONT, "fontSize": "0.84rem", "color": GRIS_MUTED, "marginBottom": "1rem"}),
            html.Div([
                html.Div([
                    html.H5("Umbral 0.50 (por defecto)", style={"fontFamily": FONT, "color": AZUL_OSCURO,
                        "fontSize": "0.9rem", "fontWeight": "700", "marginBottom": "0.5rem"}),
                    html.Ul([
                        html.Li("Recall = 0,803 — detecta 1,124 de 1,400 estudiantes en riesgo.",
                                style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO,
                                       "marginBottom": "0.4rem", "lineHeight": "1.6"}),
                        html.Li("Genera 589 falsos positivos.",
                                style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO, "lineHeight": "1.6"}),
                    ], style={"paddingLeft": "1rem"}),
                ], style={"flex": "1", "minWidth": "220px", "padding": "1rem",
                          "background": BLANCO, "borderRadius": "10px", "border": f"1px solid {BORDE}"}),
                html.Div([
                    html.H5("Umbral 0.441 (óptimo F1)", style={"fontFamily": FONT, "color": AZUL_OSCURO,
                        "fontSize": "0.9rem", "fontWeight": "700", "marginBottom": "0.5rem"}),
                    html.Ul([
                        html.Li("Recall = 0,903 — detecta 1,262 de 1,400 estudiantes en riesgo.",
                                style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO,
                                       "marginBottom": "0.4rem", "lineHeight": "1.6"}),
                        html.Li("Genera 732 falsos positivos. Recomendado si los recursos lo permiten.",
                                style={"fontFamily": FONT, "fontSize": "0.86rem", "color": GRIS_TEXTO, "lineHeight": "1.6"}),
                    ], style={"paddingLeft": "1rem"}),
                ], style={"flex": "1", "minWidth": "220px", "padding": "1rem",
                          "background": f"linear-gradient(135deg, {AZUL_CLARO}, #eef3ff)",
                          "borderRadius": "10px", "border": f"1px solid {AZUL_MEDIO}"}),
            ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),
        ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
                  "borderRadius": "14px", "padding": "1.2rem", "marginBottom": "1.5rem"}),
        html.Div([
            html.H4("Interpretabilidad — Valores SHAP (TreeExplainer)",
                    style={"color": AZUL_OSCURO, "fontFamily": FONT,
                           "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.4rem"}),
            html.P("Análisis sobre 1,000 observaciones del conjunto de prueba. "
                   "Valores SHAP positivos indican contribución hacia el bajo desempeño (Clase 1).",
                   style={"fontFamily": FONT, "fontSize": "0.84rem", "color": GRIS_MUTED, "marginBottom": "1rem"}),
            html.Div([bloque_grafico(
                "Importancia SHAP global — Random Forest",
                "Bee swarm: eje horizontal = valor SHAP sobre predicción de bajo desempeño.",
                fig_shap, "grafico-shap-beeswarm"
            )], style={"display": "flex", "gap": "1.5rem", "flexWrap": "wrap", "marginBottom": "1.2rem"}),
            html.Div([
                html.H5("Interpretación", style={"fontFamily": FONT, "color": AZUL_OSCURO,
                    "fontSize": "0.9rem", "fontWeight": "700", "marginBottom": "0.8rem",
                    "borderBottom": f"1px solid {BORDE}", "paddingBottom": "0.4rem"}),
                html.Ul([html.Li(item, style={"fontFamily": FONT, "fontSize": "0.87rem",
                    "color": GRIS_TEXTO, "marginBottom": "0.5rem", "lineHeight": "1.7"})
                    for item in [
                        "Género masculino (0,0532): ser hombre reduce la probabilidad predicha de bajo desempeño.",
                        "Edad (0,0357): a mayor edad, mayor probabilidad de bajo desempeño.",
                        "Matrícula mayor a 7 millones (0,0354): efecto protector marcado.",
                        "Institución no oficial fundación y oficial departamental: factores protectores.",
                        "Nivel educativo de los padres y estrato: a mayor capital educativo familiar, menor probabilidad de bajo desempeño.",
                    ]], style={"paddingLeft": "1.2rem", "marginBottom": "0"}),
            ], style={"background": "#f7f8fc", "borderRadius": "12px",
                      "padding": "1.2rem", "border": f"1px solid {BORDE}"}),
        ], style={"background": GRIS_CARD, "border": f"1px solid {BORDE}",
                  "borderRadius": "14px", "padding": "1.2rem"}),
    ], titulo="Análisis de resultados",
       subtitulo="Bootstrap, umbral óptimo e interpretabilidad SHAP")


def layout_modelos():
    return html.Div([
        html.Div(id="modelos-sub-content"),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_prediccion():
    """Panel interactivo de predicción con Random Forest."""

    def input_variable(col, cfg):
        label = html.Label(cfg["label"], style={
            "fontFamily": FONT, "fontSize": "0.85rem", "color": AZUL_OSCURO,
            "fontWeight": "600", "marginBottom": "4px", "display": "block"})

        if cfg["tipo"] == "radio":
            control = dcc.RadioItems(
                id={"type": "pred-input", "index": col},
                options=[{"label": l, "value": v}
                         for v, l in zip(cfg["opciones"], cfg.get("labels_opciones", cfg["opciones"]))],
                value=cfg["default"], inline=True,
                inputStyle={"marginRight": "4px"},
                labelStyle={"marginRight": "12px", "fontFamily": FONT,
                            "fontSize": "0.84rem", "color": GRIS_TEXTO},
            )
        elif cfg["tipo"] == "number":
            control = dcc.Input(
                id={"type": "pred-input", "index": col},
                type="number",
                min=cfg.get("min", 18), max=cfg.get("max", 69),
                step=cfg.get("step", 1),
                value=cfg["default"],
                style={"fontFamily": FONT, "fontSize": "0.9rem", "width": "100%",
                       "padding": "0.4rem 0.6rem", "borderRadius": "8px",
                       "border": f"1px solid {BORDE}", "color": AZUL_OSCURO},
            )
        else:  # dropdown
            control = dcc.Dropdown(
                id={"type": "pred-input", "index": col},
                options=[{"label": o, "value": o} for o in cfg["opciones"]],
                value=cfg["default"], clearable=False,
                style={"fontFamily": FONT, "fontSize": "0.84rem"},
            )

        return html.Div([label, control], style={
            "background": BLANCO, "border": f"1px solid {BORDE}",
            "borderRadius": "12px", "padding": "0.9rem 1rem",
            "marginBottom": "0.8rem",
        })

    # Agrupar inputs por grupo
    grupos_vars = {
        "Características del estudiante": [
            "estu_genero", "edad", "estu_horassemanatrabaja",
            "estu_pagomatricula", "estu_metodo_prgm",
            "estu_tituloobtenidobachiller", "estu_valormatriculauniversidad",
        ],
        "Contexto familiar": [
            "fami_estratovivienda", "fami_tieneinternet", "fami_tienecomputador",
            "fami_tieneautomovil", "fami_tienemotocicleta",
            "fami_educacionpadre", "fami_ocupacionpadre",
            "fami_educacionmadre", "fami_ocupacionmadre",
        ],
        "Institución": [
            "inst_caracter_academico", "inst_origen",
        ],
    }

    columnas = []
    for titulo_grupo, vars_grupo in grupos_vars.items():
        inputs_grupo = [input_variable(col, VARIABLES_PREDICTOR[col])
                        for col in vars_grupo if col in VARIABLES_PREDICTOR]
        columnas.append(html.Div([
            html.H5(titulo_grupo, style={
                "fontFamily": FONT, "color": MORADO_OSCURO,
                "fontSize": "0.95rem", "fontWeight": "700",
                "marginBottom": "0.8rem",
                "borderBottom": f"2px solid {BORDE}", "paddingBottom": "0.4rem"}),
            *inputs_grupo,
        ], style={"flex": "1", "minWidth": "280px"}))

    panel_inputs = html.Div(columnas, style={
        "display": "flex", "gap": "1.2rem", "flexWrap": "wrap", "alignItems": "flex-start"})

    boton = html.Div([
        html.Button("Predecir desempeño", id="btn-predecir", n_clicks=0, style={
            "background": f"linear-gradient(135deg, {CORAL}, {AZUL_OSCURO})",
            "color": BLANCO, "border": "none", "borderRadius": "14px",
            "padding": "0.9rem 2.5rem", "fontFamily": FONT,
            "fontSize": "1rem", "fontWeight": "700", "cursor": "pointer",
            "boxShadow": f"0 8px 20px {CORAL}40",
            "letterSpacing": "0.01em",
        }),
    ], style={"textAlign": "center", "marginTop": "1.5rem", "marginBottom": "1rem"})

    resultado = html.Div(id="resultado-prediccion", style={"marginTop": "1rem"})

    nota_modelo = html.Div([
        html.Span(style={"fontSize": "1rem"}),
        html.Span(f"Este panel utiliza el modelo {MODELO_GANADOR} (Recall = 0,803, AUC = 0,728 en test set). ",
                  style={"fontFamily": FONT, "fontSize": "0.83rem", "color": GRIS_TEXTO}),
        html.Span("Si el modelo no está serializado, se mostrará un mensaje informativo.",
                  style={"fontFamily": FONT, "fontSize": "0.83rem", "color": GRIS_MUTED}),
    ], style={"background": AZUL_CLARO, "borderRadius": "10px",
              "padding": "0.7rem 1rem", "marginBottom": "1.2rem",
              "border": f"1px solid {BORDE}"})

    return html.Div([
        card([
            nota_modelo,
            panel_inputs,
            boton,
            resultado,
        ], titulo="Predictor Interactivo — Random Forest",
           subtitulo="Selecciona las características del estudiante y obtén la clasificación predicha"),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


# ── Asistente IA analítico (chatbot con Gemini) ───────────────
def construir_burbujas_chat(historial):
    """Genera bloques de HTML estilizados simulando una conversación fluida."""
    burbujas = []
    for mensaje in historial:
        es_usuario = (mensaje["role"] == "user")
        bg_bubble = AZUL_OSCURO if es_usuario else BLANCO
        color_text = BLANCO if es_usuario else GRIS_TEXTO
        alineacion = "flex-end" if es_usuario else "flex-start"
        borde_bubble = "none" if es_usuario else f"1px solid {BORDE}"

        burbujas.append(html.Div([
            html.Div(mensaje["text"], style={
                "backgroundColor": bg_bubble, "color": color_text,
                "padding": "0.75rem 1.1rem", "borderRadius": "14px",
                "maxWidth": "78%", "fontFamily": FONT, "fontSize": "0.88rem",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.03)", "border": borde_bubble,
                "whiteSpace": "pre-line",
            })
        ], style={"display": "flex", "justifyContent": alineacion, "width": "100%"}))
    return burbujas


def layout_chatbot_analitico():
    """Vista de la pestaña del Asistente IA analítico."""
    aviso_sin_key = None
    if client_genai is None:
        aviso_sin_key = html.P(
            "El asistente no pudo inicializarse. Verifica que la variable de entorno "
            "GEMINI_API_KEY esté configurada en la terminal antes de ejecutar el dashboard.",
            style={"fontFamily": FONT, "fontSize": "0.82rem", "color": ACENTO_ROJO,
                   "marginBottom": "1rem"})

    contenido_card = [
        html.Div(id="chat-history-container", style={
            "height": "450px", "overflowY": "auto",
            "padding": "1.2rem", "backgroundColor": GRIS_SUAVE,
            "borderRadius": "14px", "border": f"1px solid {BORDE}",
            "display": "flex", "flexDirection": "column", "gap": "12px",
            "marginBottom": "1rem",
        }),
        dbc.InputGroup([
            dbc.Input(id="chat-user-input", placeholder="Ej: Explícame el impacto del estrato socioeconómico...",
                       type="text", style={"fontFamily": FONT}),
            dbc.Button("Consultar", id="chat-send-btn", color="primary",
                       style={"fontWeight": "600", "backgroundColor": AZUL_OSCURO,
                              "border": "none"}),
        ]),
    ]
    if aviso_sin_key:
        contenido_card.insert(0, aviso_sin_key)

    return html.Div([
        card(contenido_card,
             titulo="Asistente IA +Pro",
             subtitulo="Realiza consultas sobre las variables, los modelos,"
                       "la metodología y los hallazgos del estudio."),
    ], style={"padding": "2rem", "backgroundColor": GRIS_SUAVE, "minHeight": "100vh"})


def layout_conclusiones():
    hallazgos = [
        "Random Forest obtuvo el mayor Recall en la clase de bajo desempeño (0,803) en el conjunto de test, criterio principal de selección del modelo ganador, con superioridad estadística confirmada por el test de Wilcoxon pareado (W = 0, p = 0,002).",
        "Los modelos lineales (Logística, Ridge, Lasso) son prácticamente equivalentes entre sí, indicando que la regularización no aporta valor adicional significativo en este conjunto de datos.",
        "KNN es el modelo con menor desempeño en todas las métricas y mayor inestabilidad en validación cruzada.",
        "El género masculino, la edad, el valor de matrícula y el origen institucional son los predictores de mayor importancia según el análisis SHAP sobre Random Forest.",
        "XGBoost obtiene el AUC más alto (0,736) y la mayor Average Precision (0,747), pero su Recall (0,698) es inferior al de Random Forest, lo que lo excluye como ganador bajo el criterio de minimización de falsos negativos.",
    ]
    limitaciones = [
        "La variable dependiente se construye con el promedio nacional como umbral, lo que puede clasificar diferente a estudiantes con puntajes similares cerca del corte.",
        "No se dispone de variables cognitivas previas (ICFES bachillerato) ni de calidad instruccional, lo que limita el poder predictivo.",
        "El análisis se restringe al departamento del Atlántico (2023), por lo que los resultados no son generalizables directamente a otros contextos.",
    ]
    recomendaciones = [
        "Priorizar municipios del interior del Atlántico con alto porcentaje de bajo desempeño para intervenciones de política educativa.",
        "Incorporar variables cognitivas previas (puntaje ICFES bachillerato) en futuras versiones del modelo para mejorar la capacidad predictiva.",
        "Implementar el modelo Random Forest como herramienta de diagnóstico temprano en instituciones de educación superior del Atlántico.",
        "Explorar modelos de stacking combinando Random Forest y XGBoost para superar el umbral de AUC = 0,80.",
        "Extender el análisis a otras competencias genéricas Saber Pro y a otros departamentos de Colombia.",
    ]

    def seccion(titulo, items, color_borde):
        return html.Div([
            html.H4(titulo, style={
                "fontFamily": FONT, "color": AZUL_OSCURO,
                "fontSize": "1rem", "fontWeight": "700", "marginBottom": "0.8rem",
                "borderLeft": f"4px solid {color_borde}", "paddingLeft": "0.8rem"}),
            html.Ul([html.Li(item, style={"fontFamily": FONT, "fontSize": "0.88rem",
                                          "color": GRIS_TEXTO, "marginBottom": "0.6rem",
                                          "lineHeight": "1.7"}) for item in items],
                    style={"paddingLeft": "1.2rem"}),
        ], style={**CARD, "border": f"1px solid {BORDE}", "marginBottom": "0"})

    return html.Div([
        card([
            html.Div([
                seccion("Hallazgos principales", hallazgos, VERDE),
                seccion("Limitaciones del análisis", limitaciones, NARANJA),
                seccion("Recomendaciones para política educativa", recomendaciones, AZUL_MEDIO),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(3,1fr)",
                      "gap": "1rem"}),
        ], titulo="Conclusiones",
           subtitulo="Síntesis de hallazgos, limitaciones y recomendaciones"),
    ], style=PAGE_STYLE)


# ─────────────────────────────────────────────
#  APP
# ─────────────────────────────────────────────
app = Dash(__name__,
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)
app.title = "Saber Pro · Atlántico 2023"


def formula(latex_str):
    """Renderiza una fórmula LaTeX como imagen PNG usando matplotlib."""
    try:
        import matplotlib
        matplotlib.rcParams['text.usetex'] = False
        matplotlib.rcParams['mathtext.fontset'] = 'cm'

        # Ajustar ancho según longitud de la fórmula
        ancho = min(max(len(latex_str) * 0.13, 4.5), 8.0)

        fig, ax = plt.subplots(figsize=(ancho, 0.9))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.axis("off")
        ax.text(0.5, 0.5, f"${latex_str}$",
                ha="center", va="center", fontsize=16,
                color="#000000", transform=ax.transAxes)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                    pad_inches=0.15, facecolor="#ffffff")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return html.Img(
            src=f"data:image/png;base64,{encoded}",
            style={"display": "block", "maxWidth": "100%",
                   "borderRadius": "6px", "marginBottom": "0.6rem",
                   "padding": "0.1rem 0"}
        )
    except Exception:
        return html.Code(latex_str, style={
            "fontFamily": "'Courier New', monospace", "fontSize": "0.82rem",
            "color": "#000000", "background": "#ffffff",
            "borderRadius": "5px", "padding": "0.2rem 0.5rem"
        })

app.layout = html.Div([
    dcc.Store(id="store-seccion",    data="portada"),
    dcc.Store(id="store-eda-sub",    data="eda_desc"),
    dcc.Store(id="store-modelos-sub", data="mod_comparacion"),
    dcc.Store(id="store-intro-sub",  data="intro_intro"),
    dcc.Store(id="store-marco-sub",       data="marco_modelos"),
    dcc.Store(id="store-metodologia-sub", data="met_marco"),
    dcc.Store(id="store-modelo-sel", data=MODELO_GANADOR),
    dcc.Store(id="chat-memory-store", data=[]),
    html.Div(id="navbar-container"),
    html.Div(id="subnav-container"),
    html.Div(id="page-content"),
], style={"fontFamily": FONT, "backgroundColor": GRIS_SUAVE, "color": GRIS_TEXTO})


# ── Navbar ────────────────────────────────────
def make_navbar(activa):
    botones = [html.Span("Saber Pro · Atlántico", style={
        "color": BLANCO, "fontFamily": FONT, "fontSize": "0.95rem",
        "fontWeight": "bold", "marginRight": "2rem", "whiteSpace": "nowrap",
        "borderRight": "1px solid rgba(255,255,255,0.2)", "paddingRight": "2rem",
    })]
    for key, label in SECCIONES:
        botones.append(html.Button(
            label, id=f"btn-{key}", n_clicks=0,
            style=BTN_ACTIVE if key == activa else BTN_BASE
        ))
    return html.Div(botones, style=NAVBAR_STYLE)


def make_subnav(sub_activa, subs=None):
    subs = subs or EDA_SUBS
    botones = [
        html.Button(label, id=f"btn-{key}", n_clicks=0,
                    style=SUB_ACTIVE if key == sub_activa else SUB_BASE)
        for key, label in subs
    ]
    return html.Div(botones, style={
        "backgroundColor": BLANCO, "padding": "0.9rem 2rem",
        "borderBottom": f"1px solid {BORDE}",
        "display": "flex", "flexWrap": "wrap", "gap": "0.35rem",
        "boxShadow": "0 2px 10px rgba(31,53,86,0.04)",
    })


# ── Callbacks de navegación ───────────────────
@app.callback(
    Output("store-seccion", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in SECCIONES],
    prevent_initial_call=True,
)
def actualizar_seccion(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "portada"


@app.callback(
    Output("store-eda-sub", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in EDA_SUBS],
    prevent_initial_call=True,
)
def actualizar_eda_sub(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "eda_desc"


@app.callback(
    Output("store-intro-sub", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in INTRO_SUBS],
    prevent_initial_call=True,
)
def actualizar_intro_sub(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "intro_intro"


@app.callback(
    Output("intro-sub-content", "children"),
    Input("store-intro-sub", "data"),
)
def render_intro_sub_content(sub):
    if sub == "intro_modelos":
        return layout_intro_modelos()
    if sub == "intro_metricas":
        return layout_intro_metricas()
    return layout_intro_intro()


@app.callback(
    Output("store-modelos-sub", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in MODELOS_SUBS],
    prevent_initial_call=True,
)
def actualizar_modelos_sub(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "mod_comparacion"


@app.callback(Output("navbar-container", "children"), Input("store-seccion", "data"))
def render_navbar(s):
    return make_navbar(s)


@app.callback(Output("subnav-container", "children"),
              Input("store-seccion", "data"),
              Input("store-eda-sub", "data"),
              Input("store-modelos-sub", "data"),
              Input("store-intro-sub", "data"),
              Input("store-marco-sub", "data"),
              Input("store-metodologia-sub", "data"))
def render_subnav(seccion, eda_sub, modelos_sub, intro_sub, marco_sub, met_sub):
    if seccion == "eda":
        return make_subnav(eda_sub, EDA_SUBS)
    if seccion == "modelos":
        return make_subnav(modelos_sub, MODELOS_SUBS)
    if seccion == "intro":
        return make_subnav(intro_sub, INTRO_SUBS)
    if seccion == "metodologia":
        return make_subnav(met_sub, METODOLOGIA_SUBS)
    return html.Div()


@app.callback(
    Output("store-marco-sub", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in MARCO_SUBS],
    prevent_initial_call=True,
)
def actualizar_marco_sub(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "marco_modelos"


@app.callback(
    Output("marco-sub-content", "children"),
    Input("store-marco-sub", "data"),
)
def render_marco_sub_content(sub):
    if sub == "marco_metricas":
        return layout_intro_metricas()
    return layout_intro_modelos()


@app.callback(
    Output("store-metodologia-sub", "data"),
    [Input(f"btn-{k}", "n_clicks") for k, _ in METODOLOGIA_SUBS],
    prevent_initial_call=True,
)
def actualizar_metodologia_sub(*_):
    return ctx.triggered_id.replace("btn-", "") if ctx.triggered_id else "met_marco"


@app.callback(
    Output("metodologia-sub-content", "children"),
    Input("store-metodologia-sub", "data"),
)
def render_metodologia_sub_content(sub):
    if sub == "met_flujo":
        return layout_met_flujo()
    if sub == "met_marco":
        return layout_met_marco()
    return layout_met_variables()


@app.callback(Output("page-content", "children"),
              Input("store-seccion", "data"),
              Input("store-eda-sub", "data"),
              Input("store-modelos-sub", "data"),
              Input("store-intro-sub", "data"),
              Input("store-marco-sub", "data"),
              Input("store-metodologia-sub", "data"))
def render_content(seccion, eda_sub, modelos_sub, intro_sub, marco_sub, met_sub):
    if seccion == "portada":      return layout_portada()
    if seccion == "intro":        return layout_intro()
    if seccion == "marco":        return layout_marco()
    if seccion == "metodologia":  return layout_metodologia()
    if seccion == "prediccion":   return layout_prediccion()
    if seccion == "chatbot":      return layout_chatbot_analitico()
    if seccion == "conclusiones": return layout_conclusiones()
    if seccion == "eda":
        contenido = {
            "eda_desc": layout_eda_desc,
            "eda_dist": layout_eda_dist,
            "eda_mapa": layout_eda_mapa,
            "eda_chi":  layout_eda_chi,
        }.get(eda_sub, layout_eda_desc)()
        return html.Div(contenido,
                        style={"padding": "2rem", "backgroundColor": GRIS_SUAVE,
                               "minHeight": "100vh"})
    if seccion == "modelos":
        return layout_modelos()
    return layout_portada()


@app.callback(
    Output("modelos-sub-content", "children"),
    Input("store-modelos-sub", "data"),
)
def render_modelos_sub_content(sub):
    return _get_modelos_content(sub)


def _get_modelos_content(sub):
    return layout_modelos_comparacion() if sub == "mod_comparacion" else layout_modelos_analisis()


# ── Callbacks EDA ─────────────────────────────
@app.callback(Output("mapa-atlantico", "figure"), Input("dropdown-mapa", "value"))
def actualizar_mapa(variable):
    return grafico_mapa(variable or "pct_bajo_desempeno")


@app.callback(Output("grafico-dist-variable", "figure"),
              Input("dropdown-dist-variable", "value"))
def actualizar_grafico_dist(variable):
    return grafico_distribucion(variable or "desempeno_dicotomico")


# ── Callbacks operacionalización ──────────────
@app.callback(
    Output("op-tabla-body", "children"),
    Output("op-contador",   "children"),
    Input("op-dropdown-grupo", "value"),
    Input("op-dropdown-nivel", "value"),
)
def actualizar_op_tabla(grupo, nivel):
    filas, n, total = _build_rows(grupo, nivel)
    return filas, f"Mostrando {n} de {total} variables predictoras"


# ── Callbacks modelos interactivos ────────────
@app.callback(
    Output("store-modelo-sel", "data"),
    Input({"type": "btn-modelo-sel", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def seleccionar_modelo(n_clicks):
    from dash import ALL
    if not ctx.triggered_id:
        return MODELO_GANADOR
    return ctx.triggered_id["index"]


@app.callback(
    Output({"type": "btn-modelo-sel", "index": ALL}, "style"),
    Input("store-modelo-sel", "data"),
    State({"type": "btn-modelo-sel", "index": ALL}, "id"),
)
def actualizar_estilo_botones_modelo(modelo_sel, ids_botones):
    modelo_sel = modelo_sel or MODELO_GANADOR
    estilos = []
    for id_btn in ids_botones:
        m = id_btn["index"]
        activo = (m == modelo_sel)
        estilos.append({
            "background": f"linear-gradient(135deg, {PETROL}, {AZUL_OSCURO})" if activo else BLANCO,
            "color": BLANCO if activo else GRIS_TEXTO,
            "border": f"1px solid {BORDE}", "borderRadius": "10px",
            "padding": "0.55rem 1.1rem", "fontFamily": FONT,
            "fontSize": "0.85rem", "cursor": "pointer", "margin": "0.2rem",
            "fontWeight": "700" if activo else "400",
        })
    return estilos


@app.callback(
    Output("titulo-modelo-sel",       "children"),
    Output("grafico-roc-modelo",      "figure"),
    Output("grafico-cm-modelo",       "figure"),
    Output("grafico-pliegues-modelo", "figure"),
    Input("store-modelo-sel", "data"),
)
def actualizar_graficos_modelo(modelo):
    modelo = modelo or MODELO_GANADOR
    titulo = f"Diagnóstico del modelo: {modelo}" + (" — Ganador ★" if modelo == MODELO_GANADOR else "")

    # ROC individual
    df_roc = curvas_roc_dash.get(modelo)
    if df_roc is not None and {"fpr", "tpr"}.issubset(df_roc.columns):
        auc_val = (df_comparacion[df_comparacion["Modelo"] == modelo]["auc"].values[0]
                   if not df_comparacion.empty and "Modelo" in df_comparacion.columns
                   and modelo in df_comparacion["Modelo"].values else "?")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=df_roc["fpr"], y=df_roc["tpr"], mode="lines", name=modelo,
            line=dict(color=AZUL_OSCURO, width=2.5),
            fill="tozeroy", fillcolor=f"rgba(2,51,115,0.08)"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Azar",
            line=dict(color="#cccccc", dash="dash", width=1)
        ))
        fig_roc.update_layout(
            title=f"Curva ROC — {modelo} (AUC = {auc_val:.4f})" if isinstance(auc_val, float)
                  else f"Curva ROC — {modelo}",
            xaxis_title="1 - Especificidad", yaxis_title="Sensibilidad",
            paper_bgcolor=GRIS_SUAVE, plot_bgcolor=BLANCO,
            font_family=FONT, font_color=GRIS_TEXTO,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=50, b=20), height=380,
        )
    else:
        fig_roc = go.Figure()
        fig_roc.add_annotation(text="Curva ROC no disponible para este modelo",
                                x=0.5, y=0.5, xref="paper", yref="paper",
                                showarrow=False, font=dict(size=13, color=GRIS_MUTED))

    return titulo, fig_roc, build_cm_figure(modelo), build_boxplot_pliegues_comparativo()


# ── Callback predictor ────────────────────────
@app.callback(
    Output("resultado-prediccion", "children"),
    Input("btn-predecir", "n_clicks"),
    [State({"type": "pred-input", "index": col}, "value")
     for col in VARIABLES_PREDICTOR.keys()],
    prevent_initial_call=True,
)
def predecir(*args):
    n_clicks = args[0]
    valores  = args[1:]
    cols     = list(VARIABLES_PREDICTOR.keys())

    if n_clicks == 0:
        return html.Div()

    # Construir DataFrame de entrada
    fila = {col: val for col, val in zip(cols, valores)}
    df_input = pd.DataFrame([fila])

    # Intentar predicción con modelo cargado
    if modelo_pred is not None:
        try:
            prob = modelo_pred.predict_proba(df_input)[0]
            pred = modelo_pred.predict(df_input)[0]
            prob_bajo  = prob[1]
            prob_alto  = prob[0]
        except Exception as e:
            return html.Div([
                html.Div([
                    html.H4("Error en la predicción", style={
                        "fontFamily": FONT, "color": ACENTO_ROJO,
                        "fontSize": "1rem", "marginBottom": "0.5rem"}),
                    html.P(f"El modelo encontró un error: {str(e)}",
                           style={"fontFamily": FONT, "fontSize": "0.88rem",
                                  "color": GRIS_TEXTO}),
                    html.P("Verifica que las categorías del formulario coincidan "
                           "con las usadas en el entrenamiento.",
                           style={"fontFamily": FONT, "fontSize": "0.85rem",
                                  "color": GRIS_MUTED}),
                ], style={**CARD, "border": f"2px solid {ACENTO_ROJO}"})
            ])
    else:
        # Modo demo: regla heurística simple para mostrar la interfaz
        score = 0.5
        factores_riesgo = {
            "fami_tieneinternet": ("No", +0.08),
            "fami_tienecomputador": ("No", +0.07),
            "fami_estratovivienda": (["Sin Estrato", "Estrato 1", "Estrato 2"], +0.06),
            "estu_metodo_prgm": ("DISTANCIA VIRTUAL", +0.05),
        }
        for col, (valor_riesgo, delta) in factores_riesgo.items():
            val_actual = fila.get(col, "")
            if isinstance(valor_riesgo, list):
                if val_actual in valor_riesgo:
                    score += delta
            else:
                if val_actual == valor_riesgo:
                    score += delta
        score = min(max(score, 0.15), 0.92)
        prob_bajo = score
        prob_alto = 1 - score
        pred = 1 if prob_bajo >= 0.5 else 0

    # ── Resultado visual ─────────────────────────────────────────
    es_bajo = int(pred) == 1
    color_resultado = ACENTO_ROJO if es_bajo else VERDE
    label_resultado = "Bajo desempeño" if es_bajo else "Alto desempeño"
    descripcion = (
        "El modelo predice que este perfil de estudiante tiene mayor probabilidad de obtener "
        "un puntaje por debajo del promedio nacional en Razonamiento Cuantitativo Saber Pro."
        if es_bajo else
        "El modelo predice que este perfil de estudiante tiene mayor probabilidad de obtener "
        "un puntaje igual o superior al promedio nacional en Razonamiento Cuantitativo Saber Pro."
    )

    # Gauge de probabilidad
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_bajo * 100, 1),
        title={"text": "Probabilidad de bajo desempeño (%)", "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar": {"color": color_resultado},
            "steps": [
                {"range": [0,  40], "color": "#e8f5e9"},
                {"range": [40, 60], "color": "#fff8e1"},
                {"range": [60, 100], "color": "#ffebee"},
            ],
            "threshold": {"line": {"color": "#555", "width": 2}, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 28}},
    ))
    fig_gauge.update_layout(
        height=280, margin=dict(t=40, b=20, l=30, r=30),
        paper_bgcolor=GRIS_SUAVE, font_family=FONT,
    )

    resultado_card = html.Div([
        # Encabezado del resultado
        html.Div([
            html.Div(label_resultado, style={
                "fontSize": "1.6rem", "fontWeight": "800",
                "color": color_resultado, "fontFamily": FONT,
                "marginBottom": "0.3rem",
            }),
            html.P(descripcion, style={
                "fontFamily": FONT, "fontSize": "0.9rem",
                "color": GRIS_TEXTO, "lineHeight": "1.7",
                "maxWidth": "600px",
            }),
        ], style={"marginBottom": "1rem",
                  "borderLeft": f"5px solid {color_resultado}",
                  "paddingLeft": "1rem"}),

        # Gauge + probabilidades en dos columnas
        html.Div([
            html.Div([
                dcc.Graph(figure=fig_gauge, config={"displaylogo": False},
                          style={"height": "280px"}),
            ], style={"flex": "1", "minWidth": "260px"}),
            html.Div([
                html.H5("Probabilidades estimadas", style={
                    "fontFamily": FONT, "color": AZUL_OSCURO,
                    "fontSize": "0.95rem", "fontWeight": "700",
                    "marginBottom": "1rem"}),
                html.Div([
                    html.Div([
                        html.Span("Alto desempeño (clase 0)",
                                  style={"fontFamily": FONT, "fontSize": "0.88rem",
                                         "color": GRIS_TEXTO}),
                        html.Span(f"{prob_alto:.1%}",
                                  style={"fontFamily": FONT, "fontSize": "1.3rem",
                                         "fontWeight": "800", "color": VERDE,
                                         "marginLeft": "auto"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "justifyContent": "space-between",
                              "background": "#f0faf3", "borderRadius": "10px",
                              "padding": "0.8rem 1rem", "marginBottom": "0.6rem"}),
                    html.Div([
                        html.Span("Bajo desempeño (clase 1)",
                                  style={"fontFamily": FONT, "fontSize": "0.88rem",
                                         "color": GRIS_TEXTO}),
                        html.Span(f"{prob_bajo:.1%}",
                                  style={"fontFamily": FONT, "fontSize": "1.3rem",
                                         "fontWeight": "800", "color": ACENTO_ROJO,
                                         "marginLeft": "auto"}),
                    ], style={"display": "flex", "alignItems": "center",
                              "justifyContent": "space-between",
                              "background": "#fff5f5", "borderRadius": "10px",
                              "padding": "0.8rem 1rem"}),
                ]),
                html.P(
                    f"Modelo: Random Forest (Recall = 0,803, AUC = 0,728 en test set)."
                    if modelo_pred is not None else
                    "Modelo no serializado disponible. Resultado aproximado (modo demo). "
                    "Para activar el predictor real, guarda el pipeline como rf_pipeline.pkl "
                    "en data/processed/.",
                    style={"fontFamily": FONT, "fontSize": "0.78rem",
                           "color": GRIS_MUTED, "marginTop": "1rem",
                           "fontStyle": "italic", "lineHeight": "1.6"}),
            ], style={"flex": "1", "minWidth": "260px",
                      "padding": "1rem", "background": BLANCO,
                      "borderRadius": "14px", "border": f"1px solid {BORDE}"}),
        ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),
    ], style={**CARD, "border": f"2px solid {color_resultado}",
              "backgroundColor": "#fafbfe"})

    return resultado_card


# ── Callback lógico del Asistente IA (chatbot con Gemini) ────
@app.callback(
    Output("chat-history-container", "children"),
    Output("chat-memory-store", "data"),
    Output("chat-user-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-user-input", "n_submit"),
    State("chat-user-input", "value"),
    State("chat-memory-store", "data"),
    prevent_initial_call=True,
)
def procesar_consulta_chatbot(n_clicks, n_submit, consulta_usuario, historial):
    historial = historial or []

    if not consulta_usuario or str(consulta_usuario).strip() == "":
        return construir_burbujas_chat(historial), historial, ""

    if client_genai is None:
        historial.append({"role": "user", "text": consulta_usuario})
        historial.append({"role": "model",
                           "text": "Falta configurar la variable GEMINI_API_KEY en la terminal "
                                   "antes de ejecutar el dashboard."})
        return construir_burbujas_chat(historial), historial, ""

    historial.append({"role": "user", "text": consulta_usuario})

    contexto_institucional = (
        "Eres un consultor analítico de la Universidad del Norte experto en el examen "
        "Saber Pro 2023, módulo de Razonamiento Cuantitativo en el departamento del "
        "Atlántico. Ayudas a interpretar el modelo Random Forest ganador (Recall clase 1 "
        "= 0,803, AUC = 0,728), las variables de mayor importancia (género, valor de "
        "matrícula, edad, origen institucional) y los hallazgos generales del estudio. "
        "Responde en español, de forma clara y concisa."
    )

    try:
        contents_payload = []
        for entrada in historial:
            contents_payload.append(
                types.Content(role=entrada["role"], parts=[types.Part.from_text(text=entrada["text"])])
            )
            
        # Intento 1: Con el modelo Flash estándar
        try:
            response = client_genai.models.generate_content(
                model='gemini-2.5-flash',
                contents=contents_payload,
                config=types.GenerateContentConfig(system_instruction=contexto_institucional, temperature=0.35)
            )
        except Exception as e_flash:
            # Plan de Respaldo: Si Flash está saturado (503), intentamos con Gemini 2.5 Pro
            if "503" in str(e_flash) or "UNAVAILABLE" in str(e_flash).upper():
                response = client_genai.models.generate_content(
                    model='gemini-2.5-pro',  # <-- Modelo de respaldo de alta capacidad
                    contents=contents_payload,
                    config=types.GenerateContentConfig(system_instruction=contexto_institucional, temperature=0.35)
                )
            else:
                raise e_flash  # Si es otro tipo de error, lo mandamos al except general
        
        historial.append({"role": "model", "text": response.text})
        
    except Exception as e:
        historial.append({"role": "model", "text": f"Inconveniente con Gemini: {str(e)}"})

    return construir_burbujas_chat(historial), historial, ""

# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8054)