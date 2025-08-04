# ------------- LIBRERÍAS -------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------- CONFIGURACIÓN DE STREAMLIT -------------
st.set_page_config(
    page_title="Análisis de Datos de Amazon",
    page_icon=":bar_chart:",
    layout="wide"
)
# ------------- TÍTULO DE LA APLICACIÓN -------------
st.title("Análisis de Datos de Amazon")
