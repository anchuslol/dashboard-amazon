# app_amazon.py actualizado
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import unicodedata
import re
from streamlit_extras.metric_cards import style_metric_cards
import streamlit as st
import pandas as pd

# ---- Funciones de limpieza y estandarización ----
def limpiar_columnas(df):
    nuevas_columnas = []
    for col in df.columns:
        col = str(col)
        col_sin_acentos = unicodedata.normalize('NFKD', col).encode('ascii', errors='ignore').decode('utf-8')
        col_limpia = re.sub(r'[-:()\t]', ' ', col_sin_acentos)
        col_limpia = re.sub(r'\s+', ' ', col_limpia).strip()
        col_snake = col_limpia.replace(' ', '_').lower()
        nuevas_columnas.append(col_snake)
    df.columns = nuevas_columnas
    return df

def estandarizar_fechas(df):
    for col in df.columns:
        if 'fecha' in col or 'date' in col:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', utc=True).dt.tz_localize(None)
            except Exception:
                pass
    return df

# ---- Configuración de la aplicación ----
st.set_page_config(page_title='Dashboard Amazon', page_icon='🛒', layout='wide')

BASE_PATH = Path('C:/Users/Maria.Mezquita/OneDrive/Documentos/GitHub/AMAZON PROYECTO FINAL/data/clean')

# ------------- TABS PRINCIPALES -------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 General", "España", "Publicidad", "Inventario"
])

# --- Hacer más grandes los botones de las tabs, aumentar letra y poner en negrita ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.5rem !important;
        font-weight: bold !important;
        padding: 1.2em 2.2em !important;
        height: 3.5em !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_csv(name):
    df = pd.read_csv(BASE_PATH / f'{name}.csv')
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['fecha', 'date']):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# ------------- CABECERA ESTILIZADA -------------
with st.container():
    st.markdown("""
        <div style='text-align:center;'>
            <h1 style='color:#1f77b4;'>📈 Dashboard Amazon</h1>
            <p style='font-size:18px;'>Explora el performance mensual de ventas y publicidad.</p>
        </div>
    """, unsafe_allow_html=True)

# Cargar dataframes
df_asin             = load_csv('df_asin')
df_fecha            = load_csv('df_fecha')
df_fecha_child      = load_csv('df_fecha_child')
df_asin_child       = load_csv('df_asin_child')
df_pedidos          = load_csv('df_pedidos')
df_devoluciones     = load_csv('df_report_devoluciones')
df_tarifas          = load_csv('df_tarifas_almacenamiento')
df_reembolsos       = load_csv('df_reembolsos')
df_finanzas         = load_csv('df_finanzas')
df_envios           = load_csv('df_envios')
df_libro_mayor      = load_csv('df_libro_mayor')
df_inventario       = load_csv('df_inventario')
df_listings         = load_csv('df_listings_limpio')
df_campanas         = load_csv('df_campanas')
df_pub_rend         = load_csv('df_publicidad_rendimiento')
df_pub_terminos     = load_csv('df_publicidad_terminos')


# --- NORMALIZAR FECHAS ---
df_pedidos['purchase_date'] = df_pedidos['purchase_date'].dt.tz_localize(None)


# --- SELECCIÓN DE FECHAS ---
st.sidebar.header("Filtros")
start_date = st.sidebar.date_input("Fecha inicio", df_pedidos['purchase_date'].min().date())
end_date = st.sidebar.date_input("Fecha fin", df_pedidos['purchase_date'].max().date())

# --- FILTRADO ---
df_pedidos_f = df_pedidos[(df_pedidos['purchase_date'] >= pd.to_datetime(start_date)) & 
                          (df_pedidos['purchase_date'] <= pd.to_datetime(end_date))]


# ------------- SIDEBAR DE FILTROS -------------
st.sidebar.title("🎛️ Filtros del Dashboard")
st.sidebar.markdown("Filtra los datos para personalizar el análisis de campañas.")
st.sidebar.markdown("---")

min_date = df_pedidos['purchase_date'].min().date()
max_date = df_pedidos['purchase_date'].max().date()

st.sidebar.date_input("Fecha de inicio", min_value=min_date, max_value=max_date, value=min_date)
st.sidebar.date_input("Fecha de fin", min_value=min_date, max_value=max_date, value=max_date)

# ------------- SELECCIÓN DE ASIN -------------
st.sidebar.markdown("### Selección de ASIN")
asin_options = df_pedidos['asin'].dropna().unique().tolist()
asin_options.sort()
asin_options = ['Todos'] + asin_options

selected_asin = st.sidebar.multiselect(
    "Selecciona los ASIN",
    options=asin_options,
    default=['Todos']
)

with tab1:
    # Filtrar df_pedidos por los ASIN seleccionados si hay alguno seleccionado y no es 'Todos'
    if 'Todos' not in selected_asin and selected_asin:
        df_pedidos = df_pedidos[df_pedidos['asin'].isin(selected_asin)]

    st.sidebar.markdown("### Selección de SKU")
    sku_options = df_pedidos['sku'].dropna().unique().tolist()
    sku_options.sort()
    sku_options = ['Todos'] + sku_options

    selected_sku = st.sidebar.multiselect(
        "Selecciona los SKU",
        options=sku_options,
        default=['Todos']
    )

    # Filtrar df_pedidos por los SKU seleccionados si hay alguno seleccionado y no es 'Todos'
    if 'Todos' not in selected_sku and selected_sku:
        df_pedidos = df_pedidos[df_pedidos['sku'].isin(selected_sku)]


    # --- MÉTRICAS CLAVE CON LOOK&FEEL TIPO POWER BI Y LEYENDA DE COMPARATIVA ---

    # Asegurarse de que purchase_date es datetime
    df_pedidos['purchase_date'] = pd.to_datetime(df_pedidos['purchase_date'], errors='coerce')

    # Fechas de referencia
    hoy = pd.Timestamp.today()
    anio_actual = hoy.year
    anio_pasado = anio_actual - 1

    # Encontrar el último mes disponible en los datos filtrados
    if not df_pedidos.empty:
        ultima_fecha = df_pedidos['purchase_date'].max()
        ultimo_anio = ultima_fecha.year
        ultimo_mes = ultima_fecha.month
    else:
        ultimo_anio = anio_actual
        ultimo_mes = hoy.month

    # --- KPIs MES ACTUAL vs MISMO MES AÑO ANTERIOR ---
    df_mes_actual = df_pedidos[
        (df_pedidos['purchase_date'].dt.year == ultimo_anio) &
        (df_pedidos['purchase_date'].dt.month == ultimo_mes)
    ]
    df_mes_anterior = df_pedidos[
        (df_pedidos['purchase_date'].dt.year == (ultimo_anio - 1)) &
        (df_pedidos['purchase_date'].dt.month == ultimo_mes)
    ]

    ventas_mes_actual = df_mes_actual['item_price'].sum()
    ventas_mes_anterior = df_mes_anterior['item_price'].sum()
    unidades_mes_actual = df_mes_actual['quantity'].sum()
    unidades_mes_anterior = df_mes_anterior['quantity'].sum()

    var_ventas_mes = ((ventas_mes_actual - ventas_mes_anterior) / ventas_mes_anterior * 100) if ventas_mes_anterior else 0
    var_unidades_mes = ((unidades_mes_actual - unidades_mes_anterior) / unidades_mes_anterior * 100) if unidades_mes_anterior else 0

    # --- KPIs ACUMULADO AÑO ACTUAL vs ACUMULADO MISMO PERIODO AÑO ANTERIOR ---
    df_acum_actual = df_pedidos[
        (df_pedidos['purchase_date'].dt.year == ultimo_anio) &
        (df_pedidos['purchase_date'].dt.month <= ultimo_mes)
    ]
    df_acum_pasado = df_pedidos[
        (df_pedidos['purchase_date'].dt.year == (ultimo_anio - 1)) &
        (df_pedidos['purchase_date'].dt.month <= ultimo_mes)
    ]

    ventas_acum_actual = df_acum_actual['item_price'].sum()
    ventas_acum_pasado = df_acum_pasado['item_price'].sum()
    unidades_acum_actual = df_acum_actual['quantity'].sum()
    unidades_acum_pasado = df_acum_pasado['quantity'].sum()

    var_ventas_acum = ((ventas_acum_actual - ventas_acum_pasado) / ventas_acum_pasado * 100) if ventas_acum_pasado else 0
    var_unidades_acum = ((unidades_acum_actual - unidades_acum_pasado) / unidades_acum_pasado * 100) if unidades_acum_pasado else 0

    # --- LOOK&FEEL POWER BI: TARJETAS COLORIDAS Y LAYOUT SIMILAR ---
    st.markdown("""
        <style>
        .metric-card {
            background: #f7e7ef;
            border-radius: 16px;
            padding: 24px 16px 16px 16px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(31,119,180,0.07);
            border-left: 8px solid #e75480;
            min-width: 220px;
            text-align: center;
        }
        .metric-title {
            font-size: 1.1rem;
            color: #e75480;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .metric-value {
            font-size: 2.1rem;
            color: #22223b;
            font-weight: 700;
        }
        .metric-delta {
            font-size: 1.1rem;
            color: #1f77b4;
            font-weight: 500;
            margin-top: 4px;
        }
        .metric-sub {
            font-size: 0.95rem;
            color: #6c757d;
            margin-top: 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='color:#e75480; margin-bottom:0;'>INFORME DE VENTAS</h2>", unsafe_allow_html=True)

    # --- Tarjetas tipo Power BI ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MES ACTUAL</div>
                <div class="metric-value">{ventas_mes_actual:,.2f} €</div>
                <div class="metric-delta">{var_ventas_mes:+.2f}%<br>
                    <span style='color:#6c757d;font-size:0.95em;'>
                        ({ventas_mes_anterior:,.2f} € año anterior)
                    </span>
                </div>
                <div class="metric-sub">Ingresos<br><span style='color:#6c757d;font-size:0.9em;'>(vs mismo mes año anterior)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">MES ACTUAL</div>
                <div class="metric-value">{int(unidades_mes_actual):,}</div>
                <div class="metric-delta">{var_unidades_mes:+.2f}%<br>
                    <span style='color:#6c757d;font-size:0.95em;'>
                        ({int(unidades_mes_anterior):,} año anterior)
                    </span>
                </div>
                <div class="metric-sub">Unidades<br><span style='color:#6c757d;font-size:0.9em;'>(vs mismo mes año anterior)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card" style="background:#e7f7f3;border-left-color:#1f77b4;">
                <div class="metric-title" style="color:#1f77b4;">ACUMULADO {ultimo_anio}</div>
                <div class="metric-value">{ventas_acum_actual:,.2f} €</div>
                <div class="metric-delta">{var_ventas_acum:+.2f}%<br>
                    <span style='color:#6c757d;font-size:0.95em;'>
                        ({ventas_acum_pasado:,.2f} € año anterior)
                    </span>
                </div>
                <div class="metric-sub">Ingresos<br><span style='color:#6c757d;font-size:0.9em;'>(vs acumulado hasta mes actual año anterior)</span></div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="metric-card" style="background:#e7f7f3;border-left-color:#1f77b4;">
                <div class="metric-title" style="color:#1f77b4;">ACUMULADO {ultimo_anio}</div>
                <div class="metric-value">{int(unidades_acum_actual):,}</div>
                <div class="metric-delta">{var_unidades_acum:+.2f}%<br>
                    <span style='color:#6c757d;font-size:0.95em;'>
                        ({int(unidades_acum_pasado):,} año anterior)
                    </span>
                </div>
                <div class="metric-sub">Unidades<br><span style='color:#6c757d;font-size:0.9em;'>(vs acumulado hasta mes actual año anterior)</span></div>
            </div>
        """, unsafe_allow_html=True)

    # --- LEYENDA DE COMPARATIVA ---
    st.markdown("""
    <div style='margin-top:18px; margin-bottom:10px; color:#444; font-size:1.05em;'>
    <b>Leyenda:</b> 
    <ul>
      <li><b>Mes Actual</b>: compara el mes seleccionado con el mismo mes del año anterior.</li>
      <li><b>Acumulado</b>: compara el acumulado desde enero hasta el mes seleccionado con el mismo periodo del año anterior.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- TABLA DETALLADA OPCIONAL ---

    def format_currency(val):
        return f"{val:,.2f} €"

    def get_arrow(curr, prev):
        if prev == 0:
            return ""
        if curr == prev:
            return "➡️"
        elif curr > prev:
            return "🟢⬆️"
        else:
            return "🔴⬇️"

    with st.expander("Ver detalle de los periodos comparados"):
        st.markdown("#### Mes Actual vs Mismo Mes Año Anterior")
        tabla_mes = pd.DataFrame({
            "Periodo": [f"{ultimo_mes}/{ultimo_anio}", f"{ultimo_mes}/{ultimo_anio-1}"],
            "Ingresos (€)": [ventas_mes_actual, ventas_mes_anterior],
            "Unidades": [unidades_mes_actual, unidades_mes_anterior]
        })
        # Formatear ingresos y añadir flecha
        tabla_mes["Ingresos (€)"] = [
            f"{format_currency(tabla_mes['Ingresos (€)'][i])} {get_arrow(tabla_mes['Ingresos (€)'][i], tabla_mes['Ingresos (€)'][i+1] if i == 0 else tabla_mes['Ingresos (€)'][i-1])}"
            for i in range(len(tabla_mes))
        ]
        st.dataframe(tabla_mes, use_container_width=True)

        st.markdown(f"#### Acumulado {ultimo_anio} vs Acumulado {ultimo_anio-1} (hasta mes {ultimo_mes})")
        tabla_acum = pd.DataFrame({
            "Periodo": [f"Acum {ultimo_anio}", f"Acum {ultimo_anio-1}"],
            "Ingresos (€)": [ventas_acum_actual, ventas_acum_pasado],
            "Unidades": [unidades_acum_actual, unidades_acum_pasado]
        })
        tabla_acum["Ingresos (€)"] = [
            f"{format_currency(tabla_acum['Ingresos (€)'][i])} {get_arrow(tabla_acum['Ingresos (€)'][i], tabla_acum['Ingresos (€)'][i+1] if i == 0 else tabla_acum['Ingresos (€)'][i-1])}"
            for i in range(len(tabla_acum))
        ]
        st.dataframe(tabla_acum, use_container_width=True)

    # --- GRÁFICO COMPARATIVO: Ingresos (€) en barras y Unidades en línea por Mes (año actual vs anterior, hasta julio) ---

    import plotly.graph_objects as go

    # Limitar hasta julio
    mes_limite = 7

    # Filtrar y agrupar datos hasta julio para ambos años
    def resumen_mensual(df, anio):
        df = df[df['purchase_date'].dt.year == anio].copy()
        df['mes'] = df['purchase_date'].dt.month
        df = df[df['mes'] <= mes_limite]
        resumen = df.groupby('mes').agg({'item_price': 'sum', 'quantity': 'sum'}).reset_index()
        resumen['anio'] = anio
        return resumen

    tabla_actual = resumen_mensual(df_pedidos, ultimo_anio)
    tabla_anterior = resumen_mensual(df_pedidos, ultimo_anio - 1)

    # Unir ambos años para facilitar el gráfico
    tabla = pd.concat([tabla_actual, tabla_anterior], ignore_index=True)
    tabla['mes_label'] = tabla['mes'].apply(lambda x: f"{x:02d}")

    # Crear figura
    fig = go.Figure()

    # Barras de ingresos para ambos años (agrupadas por mes)
    for anio, color in zip([ultimo_anio, ultimo_anio-1], ['#e75480', '#b4c7e7']):
        df = tabla[tabla['anio'] == anio]
        fig.add_trace(go.Bar(
            x=df['mes_label'],
            y=df['item_price'],
            name=f'Ingresos {anio} (€)',
            marker_color=color,
            text=[f"{v:,.0f} €" for v in df['item_price']],
            textposition='outside',
            offsetgroup=f"ingresos_{anio}",
            yaxis='y1'
        ))

    # Línea de unidades para ambos años (eje secundario)
    for anio, color, dash in zip([ultimo_anio, ultimo_anio-1], ['#1f77b4', '#ff9900'], [None, 'dash']):
        df = tabla[tabla['anio'] == anio]
        fig.add_trace(go.Scatter(
            x=df['mes_label'],
            y=df['quantity'],
            name=f'Unidades {anio}',
            mode='lines+markers+text',
            marker=dict(color=color, size=8),
            line=dict(color=color, width=3, dash=dash),
            yaxis='y2',
            text=[f"{int(v):,}" for v in df['quantity']],
            textposition='bottom center'
        ))

    fig.update_layout(
        title=f'Ingresos (€) y Unidades por Mes (Enero-Julio) - {ultimo_anio} vs {ultimo_anio-1}',
        xaxis_title='Mes',
        yaxis=dict(
            title='Ingresos (€)',
            showgrid=True,
            zeroline=True
        ),
        yaxis2=dict(
            title='Unidades',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        legend_title='Métrica',
        bargap=0.15,
        plot_bgcolor='#fff',
        margin=dict(t=60, b=40, l=40, r=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.caption("Nota: Las unidades se muestran como línea (eje derecho) y los ingresos como barras. Comparativa hasta julio.")


    # --- ASIGNAR NOMBRES DE PRODUCTO EN ESPAÑOL POR ASIN + SKU ---
    df_productos = df_pedidos[['asin', 'sku', 'product_name', 'sales_channel']].dropna()
    df_productos['pais'] = df_productos['sales_channel'].str.extract(r'Amazon\.([a-z\.]+)', expand=False)

    # Filtrar nombres usados en Amazon España
    df_names_espana = df_productos[df_productos['pais'] == 'es'].copy()
    df_names_espana = df_names_espana.groupby(['asin', 'sku'])['product_name'].first().reset_index()

    # _________________________________________________________________________

    tab_dashboard, tab_por_pais = st.tabs(["📊 Dashboard General", "🌍 Análisis por País"])

    with tab_dashboard:

        # --- TREEMAP INTERACTIVO CON FILTROS PERSONALIZADOS ---
        st.markdown("## 🌍 Treemap por País")

        with st.expander("🎚️ Filtros para Treemap"):
            col1, col2 = st.columns(2)
            with col1:
                fecha_min = df_pedidos['purchase_date'].min().date()
                fecha_max = df_pedidos['purchase_date'].max().date()
                fecha_inicio_t = st.date_input("Fecha de inicio", fecha_min)
                fecha_fin_t = st.date_input("Fecha de fin", fecha_max)
            with col2:
                asin_t = st.multiselect("Filtrar ASIN", ['Todos'] + sorted(df_pedidos['asin'].dropna().unique()), default='Todos')
                sku_t = st.multiselect("Filtrar SKU", ['Todos'] + sorted(df_pedidos['sku'].dropna().unique()), default='Todos')

        # --- APLICAR FILTROS PERSONALIZADOS ---
        df_t = df_pedidos.copy()
        df_t = df_t[(df_t['purchase_date'] >= pd.to_datetime(fecha_inicio_t)) & (df_t['purchase_date'] <= pd.to_datetime(fecha_fin_t))]

        if 'Todos' not in asin_t and asin_t:
            df_t = df_t[df_t['asin'].isin(asin_t)]
        if 'Todos' not in sku_t and sku_t:
            df_t = df_t[df_t['sku'].isin(sku_t)]

        # Extraer código de país
        df_t['pais_codigo'] = df_t['sales_channel'].str.extract(r'Amazon\.([a-z\.]+)', expand=False)

        mapa_paises = {
            "fr": "Francia", "de": "Alemania", "it": "Italia", "es": "España",
            "co.uk": "Reino Unido", "uk": "Reino Unido", "nl": "Países Bajos",
            "se": "Suecia", "be": "Bélgica", "com.be": "Bélgica", "pl": "Polonia",
            "com.tr": "Turquía", "tr": "Turquía", "ie": "Irlanda", "fi": "Finlandia",
            "pt": "Portugal", "ch": "Suiza", "com": "Internacional"
        }

        df_t['pais'] = df_t['pais_codigo'].replace(mapa_paises).fillna('Otro')

        # Agrupación
        df_geo = df_t.groupby('pais', as_index=False).agg({
            'ingresos': 'sum',
            'quantity': 'sum'
        })
        df_geo['label'] = df_geo['pais'] + "<br>" + df_geo['ingresos'].round(0).astype(int).astype(str).str.replace('.', ',') + " €"

        # Ajustar moneda para Suecia (SEK) y el resto (€)
        df_geo['moneda'] = np.where(df_geo['pais'] == 'Suecia', 'SEK', '€')
        df_geo['ingresos_label'] = df_geo.apply(
            lambda row: f"{int(row['ingresos']):,} {row['moneda']}".replace(',', '.') if row['moneda'] == 'SEK' else f"{int(row['ingresos']):,} €".replace(',', '.'),
            axis=1
        )
        df_geo['label'] = df_geo['pais'] + "<br>" + df_geo['ingresos_label']

        # Convertir coronas suecas (SEK) a euros (€) usando un tipo de cambio aproximado
        SEK_TO_EUR = 0.087  # Puedes ajustar este valor según el tipo de cambio actual

        # Crear una columna de ingresos en euros para el treemap
        df_geo['ingresos_eur'] = np.where(
            df_geo['pais'] == 'Suecia',
            df_geo['ingresos'] * SEK_TO_EUR,
            df_geo['ingresos']
        )

        # Graficar Treemap mostrando valores en euros y país en la leyenda
        fig_geo = px.treemap(
            df_geo,
            path=['pais', 'label'],
            values='ingresos_eur',
            color='quantity',
            color_continuous_scale='Tealgrn',
            title='Treemap de Ingresos por País (Tamaño: ingresos en €)'
        )
        # Mostrar país y valor monetario en la etiqueta
        fig_geo.update_traces(texttemplate='<b>%{label}</b>', textinfo='label+value+percent entry')
        fig_geo.update_layout(margin=dict(t=30, l=0, r=0, b=0))

        st.plotly_chart(fig_geo, use_container_width=True)


        # _________________________________________________________________________

 # --- GRÁFICO DE TENDENCIA DE VENTAS POR MES ---
        st.markdown("## 📊 Gráfico de Tendencia de Ventas por Mes")
        # Agrupar por mes y año
        df_tendencia = df_pedidos.copy()
        df_tendencia['mes'] = df_tendencia['purchase_date'].dt.to_period('M')
        df_tendencia = df_tendencia.groupby('mes').agg({'item_price': 'sum', 'quantity': 'sum'}).reset_index()
        df_tendencia['mes'] = df_tendencia['mes'].dt.to_timestamp()

        # Crear gráfico de líneas con valores en formato moneda
        fig_tendencia = px.line(
            df_tendencia,
            x='mes',
            y='item_price',
            title='Tendencia de Ventas por Mes',
            labels={'mes': 'Mes', 'item_price': 'Ventas (€)'},
            markers=True
        )
        # Mostrar los valores en formato moneda en los puntos
        fig_tendencia.update_traces(
            text=df_tendencia['item_price'].apply(lambda x: f"{x:,.2f} €"),
            textposition='top center',
            mode='lines+markers+text'
        )
        fig_tendencia.update_layout(
            xaxis_title='Mes',
            yaxis_title='Ventas (€)',
            plot_bgcolor='#fff',
            margin=dict(t=40, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_tendencia, use_container_width=True)
    

        # _________________________________________________________________________

        # --- TABLA RESUMEN DE PRODUCTOS CON NOMBRE EN ESPAÑOL ---
        st.markdown("### 📋 Resumen de productos vendidos")

        # Agregar nombre en español según asin+sku
        df_geo_tabla = df_t.merge(df_names_espana, on=['asin', 'sku'], how='left', suffixes=('', '_es'))
        df_geo_tabla['nombre_producto'] = df_geo_tabla['product_name_es'].fillna(df_geo_tabla['product_name'])

        # Agrupación final
        tabla_productos = df_geo_tabla.groupby(['nombre_producto', 'asin', 'sku'], as_index=False).agg({
            'ingresos': 'sum',
            'quantity': 'sum'
        }).sort_values(by='ingresos', ascending=False)

        # Formateo de moneda
        def format_moneda(val):
            return f"{val:,.2f} €".replace(",", ".")

        tabla_productos['ingresos (€)'] = tabla_productos['ingresos'].apply(format_moneda)
        tabla_productos['unidades'] = tabla_productos['quantity'].astype(int)
        tabla_final = tabla_productos[['nombre_producto', 'asin', 'sku', 'ingresos (€)', 'unidades']]
        tabla_final.rename(columns={
            'nombre_producto': 'Producto',
            'asin': 'ASIN',
            'sku': 'SKU'
        }, inplace=True)

        # Mostrar tabla
        st.dataframe(tabla_final, use_container_width=True)

    with tab_por_pais:
        st.markdown("### 🌍 Análisis detallado por País")

        # Filtro de fechas para esta sección (aplica a todo menos tendencia mensual)
        fecha_min_pais = df_t['purchase_date'].min().date()
        fecha_max_pais = df_t['purchase_date'].max().date()
        col1, col2 = st.columns(2)
        with col1:
            fecha_inicio_pais = st.date_input("Fecha de inicio", fecha_min_pais, key='fecha_inicio_pais')
        with col2:
            fecha_fin_pais = st.date_input("Fecha de fin", fecha_max_pais, key='fecha_fin_pais')

        # Filtrado general por país (para todo menos tendencia mensual)
        df_por_pais = df_t[
            (df_t['purchase_date'] >= pd.to_datetime(fecha_inicio_pais)) &
            (df_t['purchase_date'] <= pd.to_datetime(fecha_fin_pais))
        ].copy()

        # Convertir ingresos de Suecia de coronas suecas (SEK) a euros (€)
        SEK_TO_EUR = 0.087  # Ajusta según el tipo de cambio actual
        df_por_pais['ingresos_monetario'] = np.where(
            df_por_pais['pais'] == 'Suecia',
            df_por_pais['ingresos'] * SEK_TO_EUR,
            df_por_pais['ingresos']
        )
        df_por_pais['moneda'] = np.where(df_por_pais['pais'] == 'Suecia', '€', '€')

        # --- RESUMEN POR PAÍS ---
        resumen_pais = df_por_pais.groupby('pais', as_index=False).agg({
            'ingresos_monetario': 'sum',
            'quantity': 'sum',
            'asin': pd.Series.nunique,
            'sku': pd.Series.nunique,
            'moneda': 'first'
        }).rename(columns={
            'ingresos_monetario': 'Ingresos Totales',
            'quantity': 'Unidades Vendidas',
            'asin': 'ASIN Únicos',
            'sku': 'SKU Únicos'
        })

        # Formatear ingresos en formato monetario
        resumen_pais['Ingresos Totales'] = resumen_pais.apply(
            lambda row: f"{row['Ingresos Totales']:,.2f} {row['moneda']}", axis=1
        )

        st.dataframe(resumen_pais.sort_values(by='Ingresos Totales', ascending=False), use_container_width=True)

        # --- GRÁFICO DE INGRESOS POR MES Y PAÍS ---
        # Para la tendencia mensual, usar todo el rango de df_t (sin filtro de fechas)
        df_t_trend = df_t.copy()
        df_t_trend['ingresos_monetario'] = np.where(
            df_t_trend['pais'] == 'Suecia',
            df_t_trend['ingresos'] * SEK_TO_EUR,
            df_t_trend['ingresos']
        )
        df_t_trend['moneda'] = np.where(df_t_trend['pais'] == 'Suecia', '€', '€')
        df_t_trend['mes'] = df_t_trend['purchase_date'].dt.to_period('M')
        df_mensual_pais = df_t_trend.groupby(['mes', 'pais', 'moneda'], as_index=False).agg({'ingresos_monetario': 'sum'})
        df_mensual_pais['mes'] = df_mensual_pais['mes'].dt.to_timestamp()

        # Formatear tooltip con moneda
        fig_line = px.line(
            df_mensual_pais,
            x='mes',
            y='ingresos_monetario',
            color='pais',
            title='Tendencia de Ingresos por Mes y País',
            labels={'mes': 'Mes', 'ingresos_monetario': 'Ingresos'},
            markers=True,
            custom_data=['moneda']
        )
        fig_line.update_traces(
            hovertemplate='%{x|%b %Y}<br>%{y:,.2f} %{customdata[0]}<extra></extra>'
        )
        fig_line.update_layout(
            plot_bgcolor='#fff',
            xaxis_title='Mes',
            yaxis_title='Ingresos (€)',
            margin=dict(t=40, b=20, l=40, r=40)
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # --- DISTRIBUCIÓN DE UNIDADES POR PAÍS ---
        fig_bar = px.bar(
            resumen_pais,
            x='pais',
            y='Unidades Vendidas',
            title='Distribución de Unidades Vendidas por País',
            labels={'pais': 'País', 'Unidades Vendidas': 'Unidades'},
            text='Unidades Vendidas'
        )
        fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='#fff',
            margin=dict(t=40, b=40, l=20, r=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- FILTRO DETALLADO POR PAÍS (Selector) ---
        paises_disponibles = resumen_pais['pais'].unique().tolist()
        pais_seleccionado = st.selectbox("Selecciona un país para detalle:", options=paises_disponibles)

        df_pais = df_por_pais[df_por_pais['pais'] == pais_seleccionado]
        resumen_producto_pais = df_pais.groupby(['asin', 'sku', 'moneda'], as_index=False).agg({
            'ingresos_monetario': 'sum',
            'quantity': 'sum'
        }).sort_values(by='ingresos_monetario', ascending=False)

        # Agregar nombre en español
        resumen_producto_pais = resumen_producto_pais.merge(df_names_espana, on=['asin', 'sku'], how='left')
        resumen_producto_pais['nombre_producto'] = resumen_producto_pais['product_name'].fillna('')

        # Formatear ingresos en formato monetario
        resumen_producto_pais['Ingresos'] = resumen_producto_pais.apply(
            lambda row: f"{row['ingresos_monetario']:,.2f} {row['moneda']}", axis=1
        )

        st.markdown(f"#### Productos vendidos en {pais_seleccionado}")
        st.dataframe(resumen_producto_pais[['nombre_producto', 'asin', 'sku', 'Ingresos', 'quantity']].rename(columns={
            'nombre_producto': 'Producto',
            'Ingresos': 'Ingresos',
            'quantity': 'Unidades'
        }), use_container_width=True)

# _____________TAB 2______________
with tab2:
    st.markdown("### 🇪🇸 Análisis para Amazon España")

    # Filtrar solo datos de España
    df_esp = df_pedidos[df_pedidos['sales_channel'].str.contains("Amazon\\.es", na=False)].copy()

    # Filtro de fechas y filtros adicionales (ASIN y SKU)
    st.markdown("#### 🎚️ Filtros para España")
    col1, col2, col3 = st.columns(3)
    with col1:
        fecha_inicio_esp = st.date_input("Desde", df_esp['purchase_date'].min().date(), key='fecha_inicio_esp')
    with col2:
        fecha_fin_esp = st.date_input("Hasta", df_esp['purchase_date'].max().date(), key='fecha_fin_esp')
    with col3:
        asin_esp = st.multiselect("Filtrar ASIN", ['Todos'] + sorted(df_esp['asin'].dropna().unique()), default='Todos', key='asin_esp')

    sku_esp = st.multiselect("Filtrar SKU", ['Todos'] + sorted(df_esp['sku'].dropna().unique()), default='Todos', key='sku_esp')

    df_esp = df_esp[(df_esp['purchase_date'] >= pd.to_datetime(fecha_inicio_esp)) &
                    (df_esp['purchase_date'] <= pd.to_datetime(fecha_fin_esp))]

    if 'Todos' not in asin_esp and asin_esp:
        df_esp = df_esp[df_esp['asin'].isin(asin_esp)]
    if 'Todos' not in sku_esp and sku_esp:
        df_esp = df_esp[df_esp['sku'].isin(sku_esp)]

    # Agregar código de país
    df_esp['pais'] = 'España'

    # --- RESUMEN ---
    resumen_esp = df_esp.groupby('pais', as_index=False).agg({
        'ingresos': 'sum',
        'quantity': 'sum',
        'asin': pd.Series.nunique,
        'sku': pd.Series.nunique
    }).rename(columns={
        'ingresos': 'Ingresos Totales',
        'quantity': 'Unidades Vendidas',
        'asin': 'ASIN Únicos',
        'sku': 'SKU Únicos'
    })

    # Formatear ingresos totales como moneda €
    resumen_esp['Ingresos Totales'] = resumen_esp['Ingresos Totales'].apply(lambda x: f"{x:,.2f} €")

    st.dataframe(resumen_esp, use_container_width=True)

    # --- TENDENCIA MENSUAL (BARRAS + LÍNEA) ---
    df_esp['mes'] = df_esp['purchase_date'].dt.to_period('M')
    df_mensual = df_esp.groupby('mes').agg({
        'ingresos': 'sum',
        'quantity': 'sum'
    }).reset_index()
    df_mensual['mes'] = df_mensual['mes'].dt.to_timestamp()

    import plotly.graph_objects as go

    fig_mix = go.Figure()
    fig_mix.add_trace(go.Bar(
        x=df_mensual['mes'],
        y=df_mensual['ingresos'],
        name='Ingresos (€)',
        marker_color='#e75480',
        text=df_mensual['ingresos'].apply(lambda x: f"{x:,.0f} €"),
        textposition='inside',
        yaxis='y1'
    ))
    fig_mix.add_trace(go.Scatter(
        x=df_mensual['mes'],
        y=df_mensual['quantity'],
        name='Unidades Vendidas',
        mode='lines+markers+text',
        text=df_mensual['quantity'],
        textposition='top center',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=7),
        yaxis='y2'
    ))

    fig_mix.update_layout(
        title="ESPAÑA  amazon.es",
        xaxis=dict(title='Mes'),
        yaxis=dict(title='Ingresos (€)', side='left', showgrid=True, tickformat=",.2f €"),
        yaxis2=dict(title='Unidades', overlaying='y', side='right', showgrid=False),
        barmode='group',
        plot_bgcolor='#fff',
        margin=dict(t=60, b=40, l=40, r=40),
        legend=dict(x=0.01, y=1.05, orientation='h')
    )

    st.plotly_chart(fig_mix, use_container_width=True)

    # Mostrar ingresos totales como moneda
    ingresos_totales_esp = df_esp['ingresos'].sum()
    st.markdown(f"**Ingresos totales en España:** <span style='font-size:1.5em;color:#e75480;font-weight:bold'>{ingresos_totales_esp:,.2f} €</span>", unsafe_allow_html=True)

    # --- PRODUCTOS EN ESPAÑA ---
    df_names_espana = df_esp.groupby(['asin', 'sku'])['product_name'].first().reset_index()
    resumen_prod_esp = df_esp.groupby(['asin', 'sku'], as_index=False).agg({
        'ingresos': 'sum',
        'quantity': 'sum'
    })
    resumen_prod_esp = resumen_prod_esp.merge(df_names_espana, on=['asin', 'sku'], how='left')
    resumen_prod_esp['nombre_producto'] = resumen_prod_esp['product_name'].fillna('')

    st.markdown("### 📦 Productos vendidos en España")
    st.dataframe(resumen_prod_esp[['nombre_producto', 'asin', 'sku', 'ingresos', 'quantity']].rename(columns={
        'nombre_producto': 'Producto',
        'ingresos': 'Ingresos (€)',
        'quantity': 'Unidades'
    }), use_container_width=True)



# ____________TAB3_________________________________

with tab3:
    st.markdown("### 📢 Análisis de Publicidad")

    col1, col2 = st.columns(2)
    with col1:
        fecha_min_pub = df_pub_rend['fecha'].min()
        fecha_max_pub = df_pub_rend['fecha'].max()
        fecha_inicio_pub = st.date_input("Desde", fecha_min_pub.date(), key='fecha_inicio_pub')
    with col2:
        fecha_fin_pub = st.date_input("Hasta", fecha_max_pub.date(), key='fecha_fin_pub')

    # Filtrar por fecha
    df_pub = df_pub_rend[(df_pub_rend['fecha'] >= pd.to_datetime(fecha_inicio_pub)) &
                         (df_pub_rend['fecha'] <= pd.to_datetime(fecha_fin_pub))].copy()

    # KPIs
    gasto_total = df_pub['gasto'].sum()
    ventas_atribuidas = df_pub_terminos[
        (df_pub_terminos['fecha'] >= pd.to_datetime(fecha_inicio_pub)) &
        (df_pub_terminos['fecha'] <= pd.to_datetime(fecha_fin_pub))
    ]['ventas_totales_de_7_dias'].sum()
    acos = (gasto_total / ventas_atribuidas) * 100 if ventas_atribuidas else 0
    roas = (ventas_atribuidas / gasto_total) if gasto_total else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gasto Publicitario (€)", f"{gasto_total:,.2f}")
    col2.metric("Ventas Atribuidas (€)", f"{ventas_atribuidas:,.2f}")
    col3.metric("ACOS (%)", f"{acos:.2f}%")
    col4.metric("ROAS", f"{roas:.2f}")

    # --- Tendencia diaria: Gasto y Ventas atribuidas por día ---
    df_pub['dia'] = df_pub['fecha'].dt.date
    df_terms = df_pub_terminos[
        (df_pub_terminos['fecha'] >= pd.to_datetime(fecha_inicio_pub)) &
        (df_pub_terminos['fecha'] <= pd.to_datetime(fecha_fin_pub))
    ].copy()
    df_terms['dia'] = df_terms['fecha'].dt.date

    # Agrupar por día
    df_gasto_dia = df_pub.groupby('dia', as_index=False)['gasto'].sum()
    df_ventas_dia = df_terms.groupby('dia', as_index=False)['ventas_totales_de_7_dias'].sum()

    # Unir ambos para alinear fechas
    df_diario = pd.merge(df_gasto_dia, df_ventas_dia, on='dia', how='outer').fillna(0)
    df_diario = df_diario.sort_values('dia')

    fig_diario = go.Figure()
    fig_diario.add_trace(go.Bar(
        x=df_diario['dia'],
        y=df_diario['gasto'],
        name='Gasto Publicitario (€)',
        marker_color='#e75480',
        text=df_diario['gasto'].apply(lambda x: f"{x:,.0f} €"),
        textposition='outside'
    ))
    fig_diario.add_trace(go.Bar(
        x=df_diario['dia'],
        y=df_diario['ventas_totales_de_7_dias'],
        name='Ventas Atribuidas (€)',
        marker_color='#1f77b4',
        text=df_diario['ventas_totales_de_7_dias'].apply(lambda x: f"{x:,.0f} €"),
        textposition='outside'
    ))

    fig_diario.update_layout(
        title="📊 Gasto Publicitario y Ventas Atribuidas por Día",
        xaxis_title='Día',
        yaxis_title='Euros (€)',
        barmode='group',
        plot_bgcolor='#fff',
        margin=dict(t=60, b=40, l=40, r=40),
        legend=dict(x=0.01, y=1.05, orientation='h')
    )

    st.plotly_chart(fig_diario, use_container_width=True)

    # --- Ranking campañas ---
    st.markdown("### 📋 Ranking de campañas")
    df_camp_ranking = df_terms.groupby('nombre_de_campana', as_index=False).agg({
        'gasto': 'sum',
        'ventas_totales_de_7_dias': 'sum'
    })
    df_camp_ranking['ACOS (%)'] = (df_camp_ranking['gasto'] / df_camp_ranking['ventas_totales_de_7_dias']) * 100
    df_camp_ranking['ROAS'] = df_camp_ranking['ventas_totales_de_7_dias'] / df_camp_ranking['gasto']
    st.dataframe(df_camp_ranking.sort_values(by='ventas_totales_de_7_dias', ascending=False), use_container_width=True)

    # --- Palabras clave ---
    st.markdown("### 🔑 Palabras clave con mejor rendimiento")
    df_keywords = df_terms.groupby('termino_de_busqueda_de_cliente', as_index=False).agg({
        'gasto': 'sum',
        'ventas_totales_de_7_dias': 'sum',
        'clics': 'sum'
    })
    df_keywords['ACOS (%)'] = (df_keywords['gasto'] / df_keywords['ventas_totales_de_7_dias']) * 100
    df_keywords['ROAS'] = df_keywords['ventas_totales_de_7_dias'] / df_keywords['gasto']
    df_keywords['CPC (€)'] = df_keywords.apply(
        lambda row: row['gasto'] / row['clics'] if row['clics'] else 0, axis=1
    )
    st.dataframe(
        df_keywords.sort_values(by='ventas_totales_de_7_dias', ascending=False)
        .head(20)[['termino_de_busqueda_de_cliente', 'gasto', 'ventas_totales_de_7_dias', 'clics', 'CPC (€)', 'ACOS (%)', 'ROAS']],
        use_container_width=True
    )



# _________tab4 _____________________________________

with tab4:
    st.markdown("### 📦 Estado del Inventario y Costes de Almacenamiento")

    # --- FILTRO POR SKU Y ASIN ---
    st.markdown("#### 🎯 Filtros de Inventario")
    col1, col2 = st.columns(2)
    with col1:
        skus_inv = st.multiselect("Filtrar SKU", ['Todos'] + sorted(df_inventario['sku'].dropna().unique()), default='Todos', key='sku_inv')
    with col2:
        asins_inv = st.multiselect("Filtrar ASIN", ['Todos'] + sorted(df_inventario['asin'].dropna().unique()), default='Todos', key='asin_inv')

    df_inv_filtrado = df_inventario.copy()
    if 'Todos' not in skus_inv:
        df_inv_filtrado = df_inv_filtrado[df_inv_filtrado['sku'].isin(skus_inv)]
    if 'Todos' not in asins_inv:
        df_inv_filtrado = df_inv_filtrado[df_inv_filtrado['asin'].isin(asins_inv)]

    # --- RESUMEN DE INVENTARIO ---
    st.markdown("#### 📋 Resumen del Inventario Actual")

    # Calcular ventas de los últimos 2 meses para cada ASIN+SKU
    hoy = pd.Timestamp.today()
    fecha_hace_2m = hoy - pd.DateOffset(months=2)

    # Filtrar pedidos de los últimos 2 meses
    df_ultimos_2m = df_pedidos[
        (df_pedidos['purchase_date'] >= fecha_hace_2m) &
        (df_pedidos['purchase_date'] <= hoy)
    ]

    ventas_2m = df_ultimos_2m.groupby(['asin', 'sku'])['quantity'].sum().reset_index()
    ventas_2m.rename(columns={'quantity': 'ventas_ultimos_2m'}, inplace=True)

    # Previsión para los próximos 2 meses = ventas últimos 2 meses
    ventas_2m['prevision_2m'] = ventas_2m['ventas_ultimos_2m']

    # Unir previsión al inventario filtrado
    resumen_inv = df_inv_filtrado.groupby(['asin', 'sku'], as_index=False).agg({
        'product_name': 'first',
        'afn_total_quantity': 'sum',
        'afn_fulfillable_quantity': 'sum',
        'afn_unsellable_quantity': 'sum',
        'afn_reserved_quantity': 'sum'
    }).rename(columns={
        'product_name': 'Producto',
        'afn_total_quantity': 'Total',  
        'afn_fulfillable_quantity': 'Disponible',
        'afn_unsellable_quantity': 'No Vendible',
        'afn_reserved_quantity': 'Reservado'
    })

    resumen_inv = resumen_inv.merge(ventas_2m[['asin', 'sku', 'prevision_2m']], on=['asin', 'sku'], how='left')
    resumen_inv['prevision_2m'] = resumen_inv['prevision_2m'].fillna(0).astype(int)
    resumen_inv.rename(columns={'prevision_2m': 'Enviar (próx. 2 meses)'}, inplace=True)

    st.dataframe(resumen_inv, use_container_width=True)

    

    # --- COSTES DE ALMACENAMIENTO ---
    st.markdown("#### 💰 Costes Estimados de Almacenamiento")

    # Filtro de país
    paises_disponibles_tarifa = ['Todos'] + sorted(df_tarifas['country_code'].dropna().unique())
    pais_seleccionado_tarifa = st.selectbox("Filtrar por País", options=paises_disponibles_tarifa, key='pais_tarifa')

    # Filtro de fecha por columna month_of_charge
    df_tarifas['month_of_charge'] = pd.to_datetime(df_tarifas['month_of_charge'], errors='coerce')
    fecha_min_tarifa = df_tarifas['month_of_charge'].min().date()
    fecha_max_tarifa = df_tarifas['month_of_charge'].max().date()
    col1, col2 = st.columns(2)
    with col1:
        fecha_inicio_tarifa = st.date_input("Desde (mes de cargo)", fecha_min_tarifa, key='fecha_inicio_tarifa')
    with col2:
        fecha_fin_tarifa = st.date_input("Hasta (mes de cargo)", fecha_max_tarifa, key='fecha_fin_tarifa')

    df_tarifas_filtrado = df_tarifas.copy()
    df_tarifas_filtrado = df_tarifas_filtrado[
        (df_tarifas_filtrado['month_of_charge'] >= pd.to_datetime(fecha_inicio_tarifa)) &
        (df_tarifas_filtrado['month_of_charge'] <= pd.to_datetime(fecha_fin_tarifa))
    ]
    if pais_seleccionado_tarifa != 'Todos':
        df_tarifas_filtrado = df_tarifas_filtrado[df_tarifas_filtrado['country_code'] == pais_seleccionado_tarifa]
    if 'Todos' not in skus_inv:
        df_tarifas_filtrado = df_tarifas_filtrado[df_tarifas_filtrado['fnsku'].isin(df_inv_filtrado['fnsku'])]
    if 'Todos' not in asins_inv:
        df_tarifas_filtrado = df_tarifas_filtrado[df_tarifas_filtrado['asin'].isin(df_inv_filtrado['asin'])]

    resumen_costes = df_tarifas_filtrado.groupby(['asin', 'fnsku', 'product_name'], as_index=False).agg({
        'estimated_monthly_storage_fee': 'sum',
        'base_rate': 'mean',
        'utilization_surcharge_rate': 'mean'
    }).rename(columns={
        'estimated_monthly_storage_fee': 'Costo Estimado (€)',
        'base_rate': 'Tarifa Base',
        'utilization_surcharge_rate': 'Recargo (%)'
    })

    # KPI pastilla arriba de la tabla
    coste_total = resumen_costes['Costo Estimado (€)'].sum()
    st.markdown("""
        <div class="metric-card" style="background:#e7f7f3;border-left-color:#1f77b4;display:inline-block;min-width:220px;">
            <div class="metric-title" style="color:#1f77b4;">Coste Total Filtrado</div>
            <div class="metric-value">{:,.2f} €</div>
            <div class="metric-sub">Coste de almacenamiento<br><span style='color:#6c757d;font-size:0.9em;'>(según filtros aplicados)</span></div>
        </div>
        <style>
        .metric-card {{
            background: #f7e7ef;
            border-radius: 16px;
            padding: 24px 16px 16px 16px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(31,119,180,0.07);
            border-left: 8px solid #e75480;
            min-width: 220px;
            text-align: center;
        }}
        .metric-title {{
            font-size: 1.1rem;
            color: #e75480;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 2.1rem;
            color: #22223b;
            font-weight: 700;
        }}
        .metric-delta {{
            font-size: 1.1rem;
            color: #1f77b4;
            font-weight: 500;
            margin-top: 4px;
        }}
        .metric-sub {{
            font-size: 0.95rem;
            color: #6c757d;
            margin-top: 2px;
        }}
        </style>
    """.format(coste_total), unsafe_allow_html=True)

    # Mostrar tabla con dos decimales y símbolo de moneda
    resumen_costes['Costo Estimado (€)'] = resumen_costes['Costo Estimado (€)'].apply(lambda x: f"{x:,.2f} €")
    resumen_costes['Tarifa Base'] = resumen_costes['Tarifa Base'].apply(lambda x: f"{x:,.2f} €")
    resumen_costes['Recargo (%)'] = resumen_costes['Recargo (%)'].apply(lambda x: f"{x:.2f} %")
    st.dataframe(resumen_costes, use_container_width=True)
