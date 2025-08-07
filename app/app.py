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

@st.cache_data
def load_csv(name):
    df = pd.read_csv(BASE_PATH / f'{name}.csv')
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['fecha', 'date']):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

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

# Lista de todos los DataFrames que quieres limpiar
dataframes = [
    df_asin, df_asin_child, df_campanas, df_envios, df_fecha, df_fecha_child, df_finanzas,
    df_inventario, df_libro_mayor, df_listings, df_pedidos, df_pub_rend,
    df_pub_terminos, df_reembolsos, df_devoluciones, df_tarifas
]

# Nombres para referencia opcional (solo si haces debug)
nombres_df = [
    "df_asin", "df_asin_child", "df_campanas", "df_envios", "df_fecha", "df_fecha_child", "df_finanzas",
    "df_inventario", "df_libro_mayor", "df_listings_limpio", "df_pedidos", "df_publicidad_rendimiento",
    "df_publicidad_terminos", "df_reembolsos", "df_report_devoluciones", "df_tarifas_almacenamiento"
]

# Aplicar limpieza a todos
for i, df in enumerate(dataframes):
    dataframes[i] = limpiar_columnas(df)
    dataframes[i] = estandarizar_fechas(dataframes[i])
    # print(f"✅ Limpieza aplicada a {nombres_df[i]}")  # Streamlit no recomienda print

# Reasignar los dataframes limpios
(
    df_asin, df_asin_child, df_campanas, df_envios, df_fecha, df_fecha_child, df_finanzas,
    df_inventario, df_libro_mayor, df_listings, df_pedidos, df_pub_rend,
    df_pub_terminos, df_reembolsos, df_devoluciones, df_tarifas
) = dataframes

# Diccionario asin -> sku
asin_to_sku = df_asin.set_index('asin_child')['sku'].to_dict()
asin_options = sorted(df_asin['asin_child'].dropna().unique().tolist())
min_date = df_fecha['fecha'].min().date()
max_date = df_fecha['fecha'].max().date()

# Sidebar
st.sidebar.title('Filtros')
date_range = st.sidebar.date_input('Rango de fechas', value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
selected_asin = st.sidebar.multiselect('ASIN', options=asin_options, default=asin_options)
if 'ship_country' in df_pedidos.columns:
    countries = sorted(df_pedidos['ship_country'].dropna().unique().tolist())
    selected_countries = st.sidebar.multiselect('País de destino', options=countries, default=countries)
else:
    selected_countries = None
st.sidebar.markdown('---')
st.sidebar.caption('Dashboard Amazon | Proyecto Final')

# Filtros
start_date, end_date = [pd.to_datetime(d) for d in date_range]
df_fecha_f = df_fecha[(df_fecha['fecha'] >= start_date) & (df_fecha['fecha'] <= end_date)]
df_fecha_child_f = df_fecha_child[(df_fecha_child['fecha'] >= start_date) & (df_fecha_child['fecha'] <= end_date)]
selected_skus = [asin_to_sku.get(asin) for asin in selected_asin if asin_to_sku.get(asin)]
df_asin_f = df_asin[df_asin['asin_child'].isin(selected_asin)]
df_asin_child_f = df_asin_child[df_asin_child['asin_child'].isin(selected_asin)]
df_tarifas_f = df_tarifas[df_tarifas['asin'].isin(selected_asin)] if 'asin' in df_tarifas.columns else df_tarifas
df_inventario_f = df_inventario[df_inventario['asin'].isin(selected_asin)] if 'asin' in df_inventario.columns else df_inventario
df_listings_f = df_listings[df_listings['sku_del_vendedor'].isin(selected_skus)] if 'sku_del_vendedor' in df_listings.columns else df_listings
df_ped_f = df_pedidos[
    (df_pedidos['purchase_date'] >= start_date) & 
    (df_pedidos['purchase_date'] <= end_date)
] if 'purchase_date' in df_pedidos.columns else df_pedidos
if selected_countries and 'ship_country' in df_ped_f.columns:
    df_ped_f = df_ped_f[df_ped_f['ship_country'].isin(selected_countries)]
df_env_f = df_envios[
    (df_envios['fecha_de_compra'] >= start_date) & 
    (df_envios['fecha_de_compra'] <= end_date)
] if 'fecha_de_compra' in df_envios.columns else df_envios
if selected_countries and 'codigo_del_pais_de_entrega' in df_env_f.columns:
    df_env_f = df_env_f[df_env_f['codigo_del_pais_de_entrega'].isin(selected_countries)]
df_devoluciones_f = df_devoluciones[(df_devoluciones['fecha'] >= start_date) & (df_devoluciones['fecha'] <= end_date)] if 'fecha' in df_devoluciones.columns else df_devoluciones
df_reembolsos_f   = df_reembolsos[(df_reembolsos['fecha'] >= start_date) & (df_reembolsos['fecha'] <= end_date)] if 'fecha' in df_reembolsos.columns else df_reembolsos
df_finanzas_f     = df_finanzas[(df_finanzas['fecha'] >= start_date) & (df_finanzas['fecha'] <= end_date)] if 'fecha' in df_finanzas.columns else df_finanzas
df_pub_rend_f     = df_pub_rend[(df_pub_rend['fecha'] >= start_date) & (df_pub_rend['fecha'] <= end_date)] if 'fecha' in df_pub_rend.columns else df_pub_rend
df_pub_terminos_f = df_pub_terminos[(df_pub_terminos['fecha'] >= start_date) & (df_pub_terminos['fecha'] <= end_date)] if 'fecha' in df_pub_terminos.columns else df_pub_terminos

# Tabs
tabs = st.tabs([
    'Resumen', 'Productos', 'Ventas & Pedidos', 'Logística', 'Publicidad', 
    'Finanzas', 'Analytics', 'Datos'
])

with tabs[0]:
    st.header('Resumen general')
    total_ventas = df_ped_f['item-price'].sum() if 'item-price' in df_ped_f.columns else 0
    unidades_totales = df_ped_f['quantity'].sum() if 'quantity' in df_ped_f.columns else 0
    devoluciones_tot = df_devoluciones_f['quantity'].sum() if 'quantity' in df_devoluciones_f.columns else 0
    reembolsos_tot = df_reembolsos_f['amount'].sum() if 'amount' in df_reembolsos_f.columns else 0
    gasto_pub = df_pub_rend_f['gasto'].sum() if 'gasto' in df_pub_rend_f.columns else 0
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric('Ventas (€)', f'{total_ventas:,.2f}')
    col2.metric('Unidades vendidas', int(unidades_totales))
    col3.metric('Devoluciones (unid.)', int(devoluciones_tot))
    col4.metric('Reembolsos (€)', f'{reembolsos_tot:,.2f}')
    col5.metric('Gasto publicitario (€)', f'{gasto_pub:,.2f}')
    if 'purchase_date' in df_ped_f.columns and 'item-price' in df_ped_f.columns:
        ventas_diarias = df_ped_f.groupby(df_ped_f['purchase_date'].dt.date)['item-price'].sum()
        fig_line = px.line(ventas_diarias, x=ventas_diarias.index, y=ventas_diarias.values,
                           labels={'x':'Fecha', 'y':'Ventas (€)'}, title='Evolución diaria de ventas')
        st.plotly_chart(fig_line, use_container_width=True)
    if 'ship_country' in df_ped_f.columns and 'item-price' in df_ped_f.columns:
        ventas_pais = df_ped_f.groupby('ship_country')['item-price'].sum().sort_values(ascending=False).reset_index()
        fig_pais = px.bar(ventas_pais, x='ship_country', y='item-price',
                          labels={'item-price':'Ventas (€)', 'ship_country':'País'},
                          title='Ventas por país')
        st.plotly_chart(fig_pais, use_container_width=True)

with tabs[1]:
    st.header('Análisis de Productos (ASIN)')
    cols_show = ['asin_child','titulo','unidades_encargadas','ventas_de_productos_encargados',
                 'porcentaje_de_ofertas_destacadas_buy_box','porcentaje_de_unidades_por_sesion']
    cols_show_exist = [c for c in cols_show if c in df_asin_f.columns]
    st.dataframe(df_asin_f[cols_show_exist], use_container_width=True)
    if {'asin_child','unidades_encargadas','ventas_de_productos_encargados'}.issubset(df_asin_f.columns):
        fig_asin = px.bar(df_asin_f, x='asin_child', y='unidades_encargadas',
                          color='ventas_de_productos_encargados',
                          labels={'unidades_encargadas':'Unidades','ventas_de_productos_encargados':'Ventas (€)'},
                          title='Unidades enc. y ventas por ASIN')
        st.plotly_chart(fig_asin, use_container_width=True)
    if {'unidades_encargadas','sesiones_total'}.issubset(df_fecha_f.columns):
        df_fecha_group = df_fecha_f.groupby('fecha').agg({
            'unidades_encargadas':'sum',
            'sesiones_total':'sum'
        })
        fig_fecha = px.line(df_fecha_group, x=df_fecha_group.index, y=['unidades_encargadas','sesiones_total'],
                            labels={'value':'Total','fecha':'Fecha'}, title='Evolución de unidades y sesiones')
        st.plotly_chart(fig_fecha, use_container_width=True)

with tabs[2]:
    st.header('Ventas y Pedidos')
    ped_cols = ['amazon_order_id','purchase_date','item-price','item-tax',
                'shipping-price','quantity','ship_country']
    ped_cols_exist = [c for c in ped_cols if c in df_ped_f.columns]
    st.dataframe(df_ped_f[ped_cols_exist].head(300), use_container_width=True)
    if 'reason' in df_devoluciones_f.columns:
        devol_motivo = df_devoluciones_f['reason'].value_counts().reset_index()
        devol_motivo.columns = ['motivo','devoluciones']
        fig_dev = px.bar(devol_motivo, x='motivo', y='devoluciones', 
                         title='Devoluciones por motivo')
        st.plotly_chart(fig_dev, use_container_width=True)
    if 'category' in df_reembolsos_f.columns and 'amount' in df_reembolsos_f.columns:
        reemb_cat = df_reembolsos_f.groupby('category')['amount'].sum().reset_index()
        fig_reemb = px.bar(reemb_cat, x='category', y='amount',
                           title='Reembolsos por categoría', labels={'amount':'Importe (€)'})
        st.plotly_chart(fig_reemb, use_container_width=True)

with tabs[3]:
    st.header('Logística: Envíos, Inventario y Almacenamiento')
    env_cols = ['numero_de_pedido_de_amazon','fecha_de_compra','fecha_de_envio','divisa',
                'precio_del_articulo','precio_de_la_entrega','transportista',
                'centro_logistico','canal_de_venta']
    env_cols_exist = [c for c in env_cols if c in df_env_f.columns]
    st.subheader('Envíos')
    st.dataframe(df_env_f[env_cols_exist].head(300), use_container_width=True)
    if {'fecha_de_pago','fecha_de_envio'}.issubset(df_env_f.columns):
        df_env_f = df_env_f.copy()
        df_env_f['tiempo_preparacion_h'] = (
            pd.to_datetime(df_env_f['fecha_de_envio']) -
            pd.to_datetime(df_env_f['fecha_de_pago'])
        ).dt.total_seconds() / 3600
        fig_hist = px.histogram(df_env_f, x='tiempo_preparacion_h', nbins=20,
                                title='Distribución del tiempo de preparación (h)')
        st.plotly_chart(fig_hist, use_container_width=True)
    st.subheader('Tarifas de almacenamiento')
    if not df_tarifas_f.empty and 'asin' in df_tarifas_f.columns and 'estimated_monthly_storage_fee' in df_tarifas_f.columns:
        fig_tarifas = px.bar(df_tarifas_f, x='asin', y='estimated_monthly_storage_fee',
                             title='Tarifa mensual estimada por ASIN', 
                             labels={'estimated_monthly_storage_fee':'Tarifa (€)'})
        st.plotly_chart(fig_tarifas, use_container_width=True)
    st.subheader('Inventario')
    st.dataframe(df_inventario_f.head(300), use_container_width=True)

with tabs[4]:
    st.header('Publicidad: campañas y términos')
    if not df_pub_rend_f.empty:
        st.subheader('Rendimiento por campaña')
        cols_pub = ['fecha','nombre_de_campana','gasto','ventas_totales','acos','roas']
        cols_pub_exist = [c for c in cols_pub if c in df_pub_rend_f.columns]
        st.dataframe(df_pub_rend_f[cols_pub_exist].head(300), use_container_width=True)
        if {'gasto','ventas_totales','acos','nombre_de_campana'}.issubset(df_pub_rend_f.columns):
            fig_scatter_camp = px.scatter(df_pub_rend_f, x='gasto', y='ventas_totales',
                                          size='acos', hover_name='nombre_de_campana',
                                          title='Gasto vs Ventas por campaña')
            st.plotly_chart(fig_scatter_camp, use_container_width=True)
    st.subheader('Términos de búsqueda')
    cols_term = ['fecha','termino_de_busqueda_de_cliente','impresiones','clics','gasto',
                 'ventas_totales_de_7_dias','coste_publicitario_de_las_ventas_acos_total']
    cols_term_exist = [c for c in cols_term if c in df_pub_terminos_f.columns]
    st.dataframe(df_pub_terminos_f[cols_term_exist].head(300), use_container_width=True)
    if {'gasto','ventas_totales_de_7_dias','impresiones','termino_de_busqueda_de_cliente'}.issubset(df_pub_terminos_f.columns):
        fig_scatter_terms = px.scatter(df_pub_terminos_f, x='gasto', y='ventas_totales_de_7_dias',
                                       size='impresiones',
                                       hover_name='termino_de_busqueda_de_cliente',
                                       title='Relación gasto vs ventas por término')
        st.plotly_chart(fig_scatter_terms, use_container_width=True)

with tabs[5]:
    st.header('Finanzas y libro mayor')
    st.subheader('Extracto financiero')
    st.dataframe(df_finanzas_f.head(300), use_container_width=True)
    st.subheader('Libro mayor')
    st.dataframe(df_libro_mayor.head(300), use_container_width=True)
    if {'ingresos','costes','fecha'}.issubset(df_finanzas_f.columns):
        fig_fin = px.bar(df_finanzas_f, x='fecha', y=['ingresos','costes'],
                         title='Ingresos y costes a lo largo del tiempo')
        st.plotly_chart(fig_fin, use_container_width=True)

with tabs[6]:
    st.header('Analytics avanzada')
    st.subheader('Matriz de correlación de métricas de producto')
    num_cols_asin = ['unidades_encargadas','ventas_de_productos_encargados',
   
                     'porcentaje_de_ofertas_destacadas_buy_box','porcentaje_de_unidades_por_sesion']
    num_cols_exist = [c for c in num_cols_asin if c in df_asin_f.columns]
    df_corr = df_asin_f[num_cols_exist].dropna()
    if not df_corr.empty:
        fig_corr, ax = plt.subplots(figsize=(5.5,4.5))
        sns.heatmap(df_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    square=True, linewidths=.5, cbar_kws={"shrink": 0.75}, ax=ax)
        ax.set_title('Matriz de correlación (productos)')
        st.pyplot(fig_corr)
    st.subheader('Clustering de productos')
    cluster_cols = ['unidades_encargadas','ventas_de_productos_encargados']
    cluster_cols_exist = [c for c in cluster_cols if c in df_asin_f.columns]
    df_cluster = df_asin_f[cluster_cols_exist].dropna()
    if len(df_cluster) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_cluster)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_cluster_plot = df_asin_f.loc[df_cluster.index].copy()
        df_cluster_plot['segmento'] = labels
        fig_cluster = px.scatter(df_cluster_plot, x=cluster_cols_exist[0], y=cluster_cols_exist[1],
                                 color='segmento', hover_name='asin_child' if 'asin_child' in df_cluster_plot.columns else None,
                                 title='Clustering de productos por unidades y ventas')
        st.plotly_chart(fig_cluster, use_container_width=True)

with tabs[7]:
    st.header('Datos brutos')
    st.markdown('Descarga los dataframes filtrados en CSV.')
    csv_ped = df_ped_f.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar pedidos filtrados', csv_ped,
                       file_name='pedidos_filtrados.csv', mime='text/csv')
    csv_asin = df_asin_f.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar datos ASIN filtrados', csv_asin,
                       file_name='asin_filtrado.csv', mime='text/csv')
    csv_env = df_env_f.to_csv(index=False).encode('utf-8')
    st.download_button('Descargar envíos filtrados', csv_env,
                       file_name='envios_filtrados.csv', mime='text/csv')
    st.dataframe(df_ped_f.head(200), use_container_width=True)
