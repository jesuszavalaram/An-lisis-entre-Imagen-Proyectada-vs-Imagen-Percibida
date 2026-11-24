from pathlib import Path
from html import escape
import argparse
import re
import unicodedata
import webbrowser
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.express as px
from plotly.offline import get_plotlyjs
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from pysentimiento import create_analyzer
 
CARPETA = Path(__file__).resolve().parent
IDIOMAS = {"es": "spanish", "en": "english", "fr": "french"}
 
VACIAS = {i: {"".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn") for w in stopwords.words(n)} for i, n in IDIOMAS.items()}
 
ENLACE_RE = re.compile(r"https?://\S+|www\.\S+")
MENCION_RE = re.compile(r"@\w+")
ETIQUETA_RE = re.compile(r"#(\w+)")
NO_LETRAS_RE = re.compile(r"[^a-z\s]")
ESPACIOS_RE = re.compile(r"\s+")
 
PALETAS = ["viridis", "cividis", "plasma", "inferno", "magma", "turbo"]
CONCEPTO_PRECIO = {"es": "precio costo valor caro barato pagar dinero cobran",
                   "en": "price cost value expensive cheap pay money charge",
                   "fr": "prix cout valeur cher payer argent tarif"}
def leer_parametros():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="?", default="cdmx_unificado.csv")
    p.add_argument("columna_texto", nargs="?", default="texto")
    p.add_argument("columna_fuente", nargs="?", default="fuente")
    p.add_argument("etiqueta_dmo", nargs="?", default="dmo")
    p.add_argument("idioma", nargs="?", choices=["es", "en"], default="es")
    p.add_argument("titulo", nargs="?", default="CDMX: imagen proyectada vs percibida")
    p.add_argument("paleta", nargs="?", choices=PALETAS, default="cividis")
    p.add_argument("--contaminacion", type=float, default=0.05)
    p.add_argument("--umbral-polaridad", type=float, default=0)
    p.add_argument("--columna-validacion", default="estrellas")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--minimo-topicos", type=int, default=50)
    p.add_argument("--umbral-precio", type=float, default=0.3)
    parametros = p.parse_args()
    return parametros
