import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import streamlit as st
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Andes Spirits IA — MVP",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_todo():
    df_skus = pd.read_csv("data/skus.csv")
    df_reg = pd.read_csv("data/regulatorio.csv")
    df_ss = pd.read_csv("data/smart_supply_reporte.csv")
    df_cli = pd.read_csv("data/clientes.csv")
    return df_skus, df_reg, df_ss, df_cli

df_skus, df_reg, df_ss, df_cli = cargar_todo()

st.sidebar.markdown("""
<div style="
    background: linear-gradient(135deg, #5C0A0A, #8B0000);
    padding: 18px 12px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 8px;
    border: 1px solid #C9A84C;
">
    <div style="font-size: 28px;">🍷</div>
    <div style="
        color: #C9A84C;
        font-size: 17px;
        font-weight: bold;
        letter-spacing: 3px;
        font-family: Georgia, serif;
    ">ANDES SPIRITS</div>
    <div style="
        color: #FAF6F0;
        font-size: 11px;
        letter-spacing: 5px;
        margin-top: 2px;
    ">S · A ·</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Andes Spirits S.A.")
st.sidebar.markdown("**MVP IA — 4 Modulos Integrados**")
st.sidebar.divider()
modulo = st.sidebar.radio("Navegar a:", [
    "Resumen Ejecutivo",
    "Smart Supply IA",
    "IA Regulatoria",
    "Marketing Digital IA",
    "Sommelier Digital IA",
    "Flujos Integrados",
    "Valor de Negocio"
])
st.sidebar.divider()
st.sidebar.markdown("Eco. Doménica Merino")
st.sidebar.markdown("AI Startup Challenge 2026")

if modulo == "Resumen Ejecutivo":
    st.title("Sistema IA Integrado — Andes Spirits")
    st.markdown("**4 Modulos · Dashboard Operativo · AI Startup Challenge 2026**")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    bloqueados = len(df_reg[df_reg["bloqueado"] == True])
    quiebre = len(df_ss[df_ss["prioridad"] == "URGENTE"])
    sobrestock = len(df_ss[df_ss["prioridad"] == "SOBRESTOCK"])
    ok_reg = len(df_reg[df_reg["estado_regulatorio"] == "OK"])
    col1.metric("SKUs en Riesgo Quiebre", quiebre)
    col2.metric("SKUs Sobrestock", sobrestock)
    col3.metric("SKUs Bloqueados", bloqueados)
    col4.metric("SKUs Regulatorio OK", ok_reg)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estado del Portafolio")
        conteo = df_ss["prioridad"].value_counts().reset_index()
        conteo.columns = ["Estado", "SKUs"]
        fig = px.bar(conteo, x="Estado", y="SKUs", color="Estado",
                    color_discrete_map={"URGENTE":"#e74c3c","SOBRESTOCK":"#f39c12",
                                        "ALERTA":"#e67e22","OK":"#2ecc71",
                                        "BLOQUEADO":"#95a5a6","BAJA_ROTACION":"#3498db"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Estado Regulatorio")
        reg_conteo = df_reg["estado_regulatorio"].value_counts().reset_index()
        reg_conteo.columns = ["Estado", "SKUs"]
        fig2 = px.pie(reg_conteo, values="SKUs", names="Estado",
                     color_discrete_map={"OK":"#2ecc71","ALERTA":"#f39c12","BLOQUEADO":"#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)
    st.divider()
    st.subheader("Los 4 Modulos del Sistema")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Smart Supply IA**\nForecast + Recomendacion de compra + Alertas")
    c2.warning("**IA Regulatoria**\nSemaforo normativo + Motor de reglas")
    c3.success("**Marketing Digital IA**\nRFM + K-means + MAB + Contenido")
    c4.error("**Sommelier Digital IA**\nRecomendador hibrido + Rotacion stock")

elif modulo == "Smart Supply IA":
    st.title("Smart Supply IA")
    st.markdown("Forecast mensual + Motor de recomendacion de compra")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs", len(df_ss))
    col2.metric("Riesgo Quiebre", len(df_ss[df_ss["prioridad"]=="URGENTE"]))
    col3.metric("Sobrestock", len(df_ss[df_ss["prioridad"]=="SOBRESTOCK"]))
    col4.metric("Bloqueados", len(df_ss[df_ss["prioridad"]=="BLOQUEADO"]))
    st.divider()

    st.subheader("Forecast Proyectado — Proximos 3 Meses")
    ventas_cols = [f"venta_mes_{i}" for i in range(19, 25)]
    venta_base = df_skus[ventas_cols].mean(axis=1).values[:len(df_ss)]
    np.random.seed(42)
    df_forecast = df_ss[["sku_id","nombre"]].copy()
    df_forecast["Mes 1"] = (venta_base * np.random.uniform(0.95, 1.10, len(df_ss))).astype(int)
    df_forecast["Mes 2"] = (venta_base * np.random.uniform(0.95, 1.10, len(df_ss))).astype(int)
    df_forecast["Mes 3"] = (venta_base * np.random.uniform(0.95, 1.10, len(df_ss))).astype(int)
    df_forecast["Promedio"] = ((df_forecast["Mes 1"] + df_forecast["Mes 2"] + df_forecast["Mes 3"]) / 3).astype(int)
    st.dataframe(df_forecast.rename(columns={"sku_id":"SKU","nombre":"Producto"}), use_container_width=True, hide_index=True)

    st.divider()
    skus_graf = df_forecast.head(6)
    datos_graf = []
    for _, row in skus_graf.iterrows():
        for mes in ["Mes 1","Mes 2","Mes 3"]:
            datos_graf.append({"SKU": row["nombre"][:15], "Mes": mes, "Unidades": row[mes]})
    df_graf = pd.DataFrame(datos_graf)
    fig_fc = px.line(df_graf, x="Mes", y="Unidades", color="SKU",
                    title="Proyeccion de ventas proximos 3 meses", markers=True)
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()
    st.subheader("Reporte de SKUs")
    for _, row in df_ss.iterrows():
        if row["prioridad"] == "URGENTE":
            st.error(f"QUIEBRE | {row['sku_id']} - {row['nombre']} | Stock: {row['stock_actual']} | Cobertura: {row['dias_cobertura']} dias | Orden: {row['orden_sugerida']} unidades")
        elif row["prioridad"] == "SOBRESTOCK":
            st.warning(f"SOBRESTOCK | {row['sku_id']} - {row['nombre']} | Stock: {row['stock_actual']} | Cobertura: {row['dias_cobertura']} dias")
        elif row["prioridad"] == "BLOQUEADO":
            st.error(f"BLOQUEADO | {row['sku_id']} - {row['nombre']} | Sin orden por regulatorio")
        else:
            st.success(f"OK | {row['sku_id']} - {row['nombre']} | Stock: {row['stock_actual']} | Cobertura: {row['dias_cobertura']} dias")

elif modulo == "IA Regulatoria":
    st.title("IA Regulatoria A&B")
    st.markdown("Monitor de cumplimiento normativo en tiempo real")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs", len(df_reg))
    col2.metric("OK", len(df_reg[df_reg["estado_regulatorio"]=="OK"]))
    col3.metric("En Alerta", len(df_reg[df_reg["estado_regulatorio"]=="ALERTA"]))
    col4.metric("Bloqueados", len(df_reg[df_reg["bloqueado"]==True]))
    st.divider()
    for _, row in df_reg.iterrows():
        if row["estado_regulatorio"] == "OK":
            st.success(f"VERDE | {row['sku_id']} - {row['nombre']} | {row['pais_origen']} | Vence en {row['dias_para_vencimiento']} dias")
        elif row["estado_regulatorio"] == "ALERTA":
            st.warning(f"AMARILLO | {row['sku_id']} - {row['nombre']} | Vence en {row['dias_para_vencimiento']} dias")
        else:
            st.error(f"ROJO | {row['sku_id']} - {row['nombre']} | BLOQUEADO")

elif modulo == "Marketing Digital IA":
    st.title("Marketing Digital con IA")
    st.markdown("Segmentacion RFM + K-means + Multi-Armed Bandit")
    st.divider()
    features = df_cli[["recencia_dias","frecuencia_compras","monto_total"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cli["cluster"] = kmeans.fit_predict(X)
    score = silhouette_score(X, df_cli["cluster"])
    medias = df_cli.groupby("cluster")["monto_total"].mean()
    orden = medias.sort_values(ascending=False).index.tolist()
    mapa = {orden[0]:"Alto Valor", orden[1]:"Promocionales", orden[2]:"Inactivos"}
    df_cli["segmento"] = df_cli["cluster"].map(mapa)
    st.metric("Silhouette Score K-means", round(score,3))
    col1, col2 = st.columns(2)
    conteo = df_cli["segmento"].value_counts().reset_index()
    conteo.columns = ["Segmento","Clientes"]
    fig1 = px.pie(conteo, values="Clientes", names="Segmento", title="Segmentos de Clientes")
    col1.plotly_chart(fig1, use_container_width=True)
    fig2 = px.scatter(df_cli.sample(300), x="recencia_dias", y="monto_total",
                     color="segmento", title="Mapa RFM")
    col2.plotly_chart(fig2, use_container_width=True)
    st.divider()
    st.subheader("Multi-Armed Bandit — Asignacion de Presupuesto")
    np.random.seed(42)
    anuncios = {"Vino Premium":{"ctr":0.08,"imp":0,"clk":0},
                "Licores":{"ctr":0.05,"imp":0,"clk":0},
                "Espumantes":{"ctr":0.12,"imp":0,"clk":0}}
    for _ in range(1000):
        if np.random.random() < 0.1:
            el = np.random.choice(list(anuncios.keys()))
        else:
            ctrs = {k:v["clk"]/(v["imp"]+1) for k,v in anuncios.items()}
            el = max(ctrs, key=ctrs.get)
        anuncios[el]["imp"] += 1
        if np.random.random() < anuncios[el]["ctr"]:
            anuncios[el]["clk"] += 1
    df_mab = pd.DataFrame([{"Anuncio":k,"Impresiones":v["imp"],"CTR":round(v["clk"]/(v["imp"]+1),3)} for k,v in anuncios.items()])
    fig3 = px.bar(df_mab, x="Anuncio", y="Impresiones", color="CTR",
                 title="Asignacion dinamica de presupuesto", color_continuous_scale="Greens")
    st.plotly_chart(fig3, use_container_width=True)

elif modulo == "Sommelier Digital IA":
    st.title("Sommelier Digital IA")
    st.markdown("Recomendador hibrido que maximiza margen y rota sobrestock")
    st.divider()
    bloqueados = df_reg[df_reg["bloqueado"]==True]["sku_id"].tolist()
    sobrestock_ids = df_ss[df_ss["prioridad"]=="SOBRESTOCK"]["sku_id"].tolist()
    df_cat = df_skus[~df_skus["sku_id"].isin(bloqueados)].copy()
    df_cat["sobrestock"] = df_cat["sku_id"].isin(sobrestock_ids)
    ventas_cols = [f"venta_mes_{i}" for i in range(19, 25)]
    df_cat["venta_promedio"] = df_cat[ventas_cols].mean(axis=1)
    df_cat["baja_rotacion"] = df_cat["venta_promedio"] < 50
    col1, col2, col3 = st.columns(3)
    cat = col1.selectbox("Categoria:", ["Todos","Vino","Espumante","Licor","Destilado"])
    precio = col2.slider("Precio maximo:", 10, 150, 80)
    seg = col3.selectbox("Segmento:", ["Alto Valor","Promocionales","Inactivos"])
    df_f = df_cat if cat == "Todos" else df_cat[df_cat["categoria"]==cat]
    df_f = df_f[df_f["precio_unitario"] <= precio].copy()
    df_f["score"] = df_f["margen"]*30 + df_f["sobrestock"].astype(int)*25 + df_f["baja_rotacion"].astype(int)*15
    df_f = df_f.sort_values("score", ascending=False).head(5)
    st.subheader("Top 5 Recomendaciones")
    for i, (_, row) in enumerate(df_f.iterrows()):
        c1, c2, c3, c4 = st.columns([3,2,1,2])
        c1.write(f"**{i+1}. {row['nombre']}**")
        c2.write(f"{row['categoria']} - {row['pais_origen']}")
        c3.write(f"${row['precio_unitario']}")
        if row["sobrestock"]:
            c4.warning("SOBRESTOCK - PRIORIDAD")
        elif row["baja_rotacion"]:
            c4.info("BAJA ROTACION - TRIGGER ACTIVO")
        else:
            c4.success("Recomendado")
    st.divider()
    st.subheader("Gestion Baja Rotacion — Trigger Marketing")
    baja_rot = df_cat[df_cat["baja_rotacion"] == True]
    if len(baja_rot) > 0:
        for _, row in baja_rot.iterrows():
            c1, c2, c3 = st.columns(3)
            c1.info(f"BAJA ROTACION: {row['sku_id']} - {row['nombre']}")
            c2.warning("Trigger: Campana marketing activada")
            c3.success("Sommelier: Recomendacion prioritaria activada")
    else:
        st.success("No hay SKUs con baja rotacion actualmente")

elif modulo == "Flujos Integrados":
    st.title("Flujos de Integracion entre Modulos")
    st.divider()
    st.subheader("Flujo 1 — Sobrestock detectado")
    sobrestock_skus = df_ss[df_ss["prioridad"]=="SOBRESTOCK"]
    for _, row in sobrestock_skus.iterrows():
        c1, c2, c3 = st.columns(3)
        c1.error(f"Smart Supply: SOBRESTOCK {row['sku_id']} - {row['nombre']}")
        c2.warning("Marketing: Campana activada")
        c3.success("Sommelier: SKU priorizado")
    st.divider()
    st.subheader("Flujo 2 — Registro sanitario bloqueado")
    bloqueados_df = df_reg[df_reg["bloqueado"]==True]
    for _, row in bloqueados_df.iterrows():
        c1, c2 = st.columns(2)
        c1.error(f"Regulatorio bloquea: {row['sku_id']} - {row['nombre']}")
        c2.success("Smart Supply: Orden de compra bloqueada")
    st.divider()
    st.subheader("Arquitectura del Sistema")
    st.code("""
[Datos CSV]
    |
    v
[Smart Supply IA] -----> [IA Regulatoria]
    |                         |
    | sobrestock               | bloqueo
    v                         v
[Marketing IA]         [Bloquea orden compra]
    |
    v
[Sommelier IA] -----> [Recomendacion al cliente]
    """)

elif modulo == "Valor de Negocio":
    st.title("Valor de Negocio y Escalabilidad")
    st.markdown("Impacto economico estimado del sistema IA")
    st.divider()
    st.subheader("Impacto Economico Estimado")
    col1, col2, col3 = st.columns(3)
    col1.metric("Reduccion de quiebres estimada", "30%", delta="Ahorro $18,000/anio")
    col2.metric("Reduccion de sobrestock", "25%", delta="Liberacion $12,000 capital")
    col3.metric("Incremento ticket promedio", "15%", delta="Via recomendador Sommelier")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROI Estimado del Sistema")
        categorias = ["Reduccion quiebres","Liberacion sobrestock","Incremento ticket","Eficiencia regulatoria"]
        valores = [18000, 12000, 8000, 5000]
        fig = px.bar(x=categorias, y=valores,
                    title="Beneficio economico anual estimado (USD)",
                    color=valores, color_continuous_scale="Greens",
                    labels={"x":"Categoria","y":"USD"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Comparacion Antes vs Despues IA")
        df_comp = pd.DataFrame({
            "Metrica": ["Quiebres/mes","Sobrestock dias","Revision regulatoria","Segmentacion clientes"],
            "Antes IA": [8, 145, "3 dias manual", "No existia"],
            "Con IA": [3, 95, "Tiempo real", "1000 clientes segmentados"]
        })
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Hoja de Ruta — Escalabilidad")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Fase 2 — 3 meses**\n- Forecast con Prophet\n- API WhatsApp real\n- PostgreSQL\n- Dashboard en produccion")
    with col2:
        st.warning("**Fase 3 — 6 meses**\n- Modelo LSTM\n- Integracion ERP\n- API Meta Ads\n- Thompson Sampling")
    with col3:
        st.success("**Fase 4 — 12 meses**\n- Data lake AWS\n- MLOps\n- App movil\n- Expansion paises")
    st.divider()
    st.subheader("Stack Tecnologico Justificado")
    df_stack = pd.DataFrame({
        "Componente": ["Forecast","Clustering","MAB","Recomendador","Dashboard","Datos"],
        "MVP": ["Promedio movil","K-means","Epsilon-Greedy","Content-based","Streamlit","CSV simulados"],
        "Produccion": ["Prophet/SARIMA","K-means+DBSCAN","Thompson Sampling","Filtrado colaborativo","React+FastAPI","PostgreSQL+S3"],
        "Justificacion": [
            "MVP interpretable, produccion preciso",
            "Baseline solido, expandible",
            "Simple, escalable",
            "Explicable para el negocio",
            "Rapido para MVP",
            "Mock para demo, real en produccion"
        ]
    })
    st.dataframe(df_stack, use_container_width=True, hide_index=True)
