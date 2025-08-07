# 📊 Amazon Sales Dashboard - Análisis Integral de Ventas

Un dashboard completo desarrollado en Streamlit para el análisis integral de datos de ventas de Amazon, que incluye preprocesamiento de datos, visualizaciones interactivas y métricas clave de rendimiento.

## 🎯 Objetivo del Proyecto

Crear una herramienta integral que permita analizar el rendimiento de ventas en Amazon a través de múltiples dimensiones: ventas generales, análisis por países, publicidad e inventario.

## 📁 Estructura del Proyecto

AMAZON PROYECTO FINAL/
├── data/
│   ├── raw/                    # Datos originales sin procesar
│   │   ├── fba/pedidos/       # Archivos de pedidos
│   │   └── inventario/        # Reportes de inventario
│   ├── clean/                 # Datos procesados y limpios
│   └── processed/             # Datos transformados para análisis
├── preprocessing/
│   ├── 01_Preprocesamiento_Ventas.ipynb  # Notebook de limpieza
│   ├── 02_EDA_Ventas.ipynb              # Análisis exploratorio
│   └── clean_pagos.py                   # Utilidades de limpieza
├── app/
│   ├── app_amazon.py          # Dashboard principal
│   └── app.py                 # Aplicación alternativa
├── notebooks/                 # Notebooks de análisis
├── scripts/                   # Scripts auxiliares
└── requirements.txt          # Dependencias del proyecto

### 🔧 Preprocesamiento de Datos

**Archivo:** `01_Preprocesamiento_Ventas.ipynb`  
El notebook de preprocesamiento realiza las siguientes transformaciones:

#### 🧹 Limpieza y Normalización

- Normalización de columnas: Eliminación de acentos, caracteres especiales y conversión a snake_case
- Estandarización de fechas: Conversión a formato datetime uniforme
- Limpieza de valores monetarios: Eliminación de símbolos de moneda y espacios

#### 📊 Procesamiento por Tipo de Datos

- **Pedidos:** Extracción de información de ventas, fechas y productos
- **Inventario:** Procesamiento de stock y disponibilidad
- **Publicidad:** Análisis de campañas y términos de búsqueda
- **Finanzas:** Tarifas de almacenamiento y costes

#### 💾 Generación de DataFrames Limpios

Los datos procesados se guardan en la carpeta `clean` con nombres estandarizados:

- `df_pedidos.csv` - Datos de ventas principales
- `df_inventario.csv` - Estado del inventario
- `df_publicidad_rendimiento.csv` - Métricas de publicidad
- `df_tarifas_almacenamiento.csv` - Costes de almacén

### 🖥️ Dashboard Interactivo

**Archivo principal:** `app_amazon.py`  
El dashboard principal desarrollado en Streamlit ofrece cuatro módulos de análisis:

#### 📈 TAB 1: Análisis General

- **KPIs Principales:** Comparativa mes actual vs año anterior y acumulados
- **Métricas Clave:**
  - Ingresos y unidades vendidas
  - Variaciones porcentuales año sobre año
- **Visualizaciones:**
  - Gráfico comparativo barras (ingresos) + líneas (unidades)
  - Treemap interactivo por países con conversión de divisas
  - Tendencia mensual de ventas
  - Tabla resumen de productos con nombres en español

#### 🇪🇸 TAB 2: España

- **Análisis Específico:** Foco exclusivo en Amazon.es
- **Filtros Avanzados:** Por fechas, ASIN y SKU
- **Visualizaciones:**
  - Gráfico mixto (barras + línea) de ingresos y unidades
  - Resumen de productos vendidos en España
  - Métricas: Ingresos totales destacados con formato monetario

#### 📢 TAB 3: Publicidad

- **KPIs Publicitarios:**
  - Gasto publicitario total
  - Ventas atribuidas
  - ACOS (Advertising Cost of Sales)
  - ROAS (Return on Advertising Spend)
- **Análisis Temporal:** Tendencia diaria de gasto y ventas
- **Rankings:**
  - Campañas más efectivas
  - Palabras clave con mejor rendimiento
- **Métricas Calculadas:** CPC, ACOS y ROAS por término

#### 📦 TAB 4: Inventario

- **Estado de Stock:** Análisis de inventario actual
- **Previsión Inteligente:** Cálculo de envíos necesarios basado en ventas últimos 2 meses
- **Costes de Almacenamiento:**
  - Filtros por país y fechas
  - Tarifas base y recargos
  - Coste total estimado
- **Métricas de Inventario:** Disponible, reservado, no vendible

## 🛠️ Funcionalidades Técnicas

### 🎨 Diseño y UX

- Look & Feel Power BI: Tarjetas métricas con colores corporativos
- Responsive Design: Adaptable a diferentes tamaños de pantalla
- Navegación Intuitiva: Tabs principales con iconos descriptivos

### 🔄 Procesamiento de Datos

- Cache Inteligente: Uso de `@st.cache_data` para optimizar rendimiento
- Conversión de Divisas: Manejo automático SEK → EUR
- Filtrado Dinámico: Filtros interconectados entre secciones

### 📊 Visualizaciones Avanzadas

- Plotly Interactive: Gráficos interactivos con tooltips personalizados
- Múltiples Ejes: Combinación de métricas en un solo gráfico
- Formatos Monetarios: Presentación consistente de valores económicos

## 🚀 Instalación y Uso

### Prerrequisitos

#### Principales Dependencias

- `streamlit` - Framework web
- `pandas` - Manipulación de datos
- `plotly` - Visualizaciones interactivas
- `numpy` - Cálculos numéricos

### Ejecución

(Agrega aquí instrucciones de ejecución si es necesario, por ejemplo: `streamlit run app_amazon.py`)

## 📋 Características Destacadas

### ✨ Análisis Inteligente

- Comparativas automáticas año sobre año
- Cálculo de variaciones porcentuales
- Previsiones basadas en históricos

### 🌍 Soporte Multi-País

- Manejo de múltiples monedas
- Conversión automática de divisas
- Filtros por mercado geográfico

### 📈 Métricas Empresariales

- KPIs estándar de e-commerce
- Métricas publicitarias especializadas
- Análisis de rentabilidad por producto

### 🎛️ Filtros Avanzados

- Filtrado por fechas, ASIN, SKU
- Filtros específicos por sección
- Mantenimiento de estado entre tabs

## 🔍 Casos de Uso

- **Análisis de Rendimiento Mensual:** Comparar ventas actuales con períodos anteriores
- **Optimización de Inventario:** Identificar productos con bajo stock
- **Análisis de Publicidad:** Evaluar ROI de campañas publicitarias
- **Expansión Geográfica:** Analizar rendimiento por países
- **Gestión de Costes:** Monitorear tarifas de almacenamiento

## 🎯 Valor de Negocio

- **Toma de Decisiones Data-Driven:** Métricas claras y visualizaciones intuitivas
- **Optimización de Recursos:** Identificación de oportunidades de mejora
- **Monitoreo en Tiempo Real:** Dashboard actualizable con nuevos datos
- **Análisis Integral:** Visión 360° del negocio Amazon

Desarrollado para proporcionar insights accionables y facilitar la toma de decisiones estratégicas en el ecosistema Amazon.