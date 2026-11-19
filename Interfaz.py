"""Servipan · Inteligencia de mercado y demanda (Streamlit).
Ejecutar:  streamlit run Interfaz.py
La lógica vive en calendario.py, datos.py y modelo.py; aquí solo hay interfaz."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calendario import GRUPOS_CALENDARIO
from datos import (CATALOGO_OFICIAL_SERVIPAN, LUGARES_DISTRIBUCION_SERVIPAN, SIN_ASIGNAR,
                   enriquecer_dataframe_con_maestros, formatear_canales, generar_datos_ejemplo)
from graficos import construir_grafico_tendencia
from modelo import (POSTURAS, UMBRAL_VIDA_CORTA, auditar, construir_costos, decidir, evaluar_politicas,
                    pronosticar, sensibilidad_costos)
from recursos import SVG_ESPIGA_TRIGO, SVG_PAN_ARTESANAL, cargar_imagen_base64

st.set_page_config(page_title="Servipan · Inteligencia de Mercado y Demanda", page_icon="🍞",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'estilos.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

N_BASE_VARS = 9
N_CAL_VARS = sum(len(v) for v in GRUPOS_CALENDARIO.values())


# ------------------------------------------------------------------------------ caché
@st.cache_data(show_spinner=False)
def datos_ejemplo():
    return generar_datos_ejemplo()


@st.cache_data(show_spinner=False)
def enriquecer(df):
    return enriquecer_dataframe_con_maestros(df)


@st.cache_data(show_spinner="Entrenando modelos cuantílicos…")
def pronosticar_cache(df, refs):
    return pronosticar(df, list(refs))


@st.cache_data(show_spinner=False)
def auditar_cache(df, refs):
    return auditar(df, list(refs))


def fmt_n(v, dec=0):
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------------------ sidebar
if "seccion_activa" not in st.session_state:
    st.session_state.seccion_activa = "Resumen"

OPCIONES_MENU = ["Resumen", "Pronóstico", "Productos", "Sedes", "Escenarios", "Inventario", "Alertas", "Configuración"]

with st.sidebar:
    st.markdown('<div class="servipan-emblem-wrap"><div class="servipan-oval"><span>Servipan</span></div>'
                '<div class="servipan-sub-tag">Planta<br>Valledupar</div></div>', unsafe_allow_html=True)
    for opcion in OPCIONES_MENU:
        if st.button(opcion, key=f"nav_btn_{opcion}", width="stretch",
                     type="primary" if st.session_state.seccion_activa == opcion else "secondary"):
            st.session_state.seccion_activa = opcion
            st.rerun()
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.expander("🎯 Postura de riesgo y costos", expanded=True):
        postura_riesgo = st.selectbox("Política de abastecimiento", list(POSTURAS), index=0)
        costo_var_pct = st.slider("Costo variable (% del precio) — SUPUESTO", 40, 80, 60, 5) / 100.0
        penalizacion_pct = st.slider("Penalización por quiebre (% del precio)", 0, 50, 15, 5) / 100.0
        nivel_larga = st.slider("Nivel de servicio · vida útil > 10 días (%)", 50, 90, 90, 5) / 100.0
        st.caption(f"Newsvendor de un período solo para vida útil ≤ {UMBRAL_VIDA_CORTA} días. "
                   "Con vida larga el sobrante se vende después: se usa un nivel de servicio objetivo.")
    with st.expander("💲 Costos reales por SKU (opcional)", expanded=False):
        archivo_costos = st.file_uploader("Excel con columnas Ref, Costo_Unitario [, Precio_Venta]", type=["xlsx"], key="costos_xlsx")
    with st.expander("📥 Cargar datos Excel", expanded=False):
        archivo = st.file_uploader("Subir historial (.xlsx)", type=["xlsx"],
                                   help="Columnas requeridas: Fecha, Ref, Desc item, Cant inv. Anonimice la razón social antes de subir a la nube.")

seccion_activa = st.session_state.seccion_activa

# ------------------------------------------------------------------------------ datos
es_ejemplo, df_raw = True, None
if archivo is not None:
    try:
        df_temp = pd.read_excel(archivo)
        base = ["Fecha", "Ref", "Desc item", "Cant inv"]
        if all(c in df_temp.columns for c in base):
            df_temp["Fecha"] = pd.to_datetime(df_temp["Fecha"])
            df_temp["Ref"] = df_temp["Ref"].astype(str)
            df_raw, es_ejemplo = df_temp, False
        else:
            st.sidebar.error(f"Faltan columnas requeridas: {base}")
    except Exception as exc:
        st.sidebar.error(f"Error leyendo Excel: {exc}")
if df_raw is None:
    df_raw, es_ejemplo = datos_ejemplo(), True

costos_override = None
if archivo_costos is not None:
    try:
        dfc = pd.read_excel(archivo_costos)
        if {"Ref", "Costo_Unitario"}.issubset(dfc.columns):
            costos_override = {}
            for _, r in dfc.iterrows():
                d = {"costo": float(r["Costo_Unitario"])}
                if "Precio_Venta" in dfc.columns and pd.notna(r["Precio_Venta"]):
                    d["precio"] = float(r["Precio_Venta"])
                costos_override[str(r["Ref"]).strip().upper()] = d
        else:
            st.sidebar.error("El archivo de costos requiere las columnas Ref y Costo_Unitario.")
    except Exception as exc:
        st.sidebar.error(f"Error leyendo costos: {exc}")

df_enriquecido, diag = enriquecer(df_raw)
col_cliente = next((c for c in ["Razon social cliente factura", "Razón social cliente factura"] if c in df_enriquecido.columns), None)

with st.sidebar:
    with st.expander("⚙️ Parámetros por producto", expanded=False):
        parametros = []
        for _, row in df_enriquecido[["Ref", "Desc item", "Linea_Oficial", "Und_Caja_Cat", "Min_Cajas_Cat"]].drop_duplicates("Ref").iterrows():
            ref, linea = str(row["Ref"]), str(row["Linea_Oficial"])
            st.markdown(f"**{row['Desc item']}** (`{linea}`)")
            frec = st.number_input("Frecuencia cargue (días/sem)", 1, 7, 6 if linea in ["Panes", "Tajados"] else 3, key=f"frec_{ref}")
            emb = st.number_input("Unidades por caja", min_value=1, value=int(row["Und_Caja_Cat"]), key=f"emb_{ref}")
            minl = st.number_input("Mínimo por lote (cajas)", min_value=1, value=int(row["Min_Cajas_Cat"]), key=f"min_{ref}")
            parametros.append({"Ref": ref, "Frecuencia de cargue (Días/Semana)": frec, "Unidades por Caja": emb, "Mínimo a Producir (Cajas)": minl})
            st.divider()
    st.markdown(f'<div class="sidebar-doodle">{SVG_ESPIGA_TRIGO}{SVG_ESPIGA_TRIGO}</div>', unsafe_allow_html=True)
df_param = pd.DataFrame(parametros)

# ------------------------------------------------------------------------------ cabecera y filtros
c_izq, c_lin, c_sed, c_hor = st.columns([5.1, 1.5, 2.0, 1.4])
badge = ('<span class="data-badge-demo">Benchmark sintético (reproducible, 180 días)</span>' if es_ejemplo
         else '<span class="data-badge-real">Datos reales cargados</span>')
titulos = {
    "Resumen": ("Resumen de inteligencia de mercado", f"{N_BASE_VARS + N_CAL_VARS} variables (calendario sociocultural + rezagos SKU) {badge}"),
    "Pronóstico": ("Pronóstico detallado por SKU", "Curvas cuantílicas diarias con festivos, quincenas y calendario académico"),
    "Productos": ("Catálogo oficial y líneas Servipan", f"Fichas técnicas de {len(CATALOGO_OFICIAL_SERVIPAN)} referencias, presentaciones y canales"),
    "Sedes": ("Cobertura regional Costa Caribe", f"{len(LUGARES_DISTRIBUCION_SERVIPAN)} corredores y {sum(len(v['municipios']) for v in LUGARES_DISTRIBUCION_SERVIPAN.values())} localidades"),
    "Escenarios": ("Simulador de estrategias y riesgos", "El choque se aplica a toda la distribución y se recalculan cajas y lotes"),
    "Inventario": ("Control de frescura, cajas y lotes", "Conversión del nivel de servicio objetivo a cajas comerciales"),
    "Alertas": ("Centro de alertas tempranas", "Riesgo de quiebre, calendario y calibración cuantílica"),
    "Configuración": ("Auditoría ML, ablación y newsvendor", "Origen móvil, pinball, cobertura, ablación y sensibilidad de costos"),
}
with c_izq:
    t, s = titulos[seccion_activa]
    st.markdown(f'<div><h1 class="main-title">{t}</h1><p class="main-subtitle">{s}</p></div>', unsafe_allow_html=True)
with c_lin:
    st.markdown('<div class="filter-label">Línea</div>', unsafe_allow_html=True)
    linea_sel = st.selectbox("Línea", ["Todas las líneas"] + sorted(df_enriquecido["Linea_Oficial"].dropna().unique().tolist()), label_visibility="collapsed")
with c_sed:
    st.markdown('<div class="filter-label">Corredor regional</div>', unsafe_allow_html=True)
    sede_sel = st.selectbox("Corredor regional", ["Todas las sedes"] + sorted(df_enriquecido["Sede_Regional"].dropna().unique().tolist()), label_visibility="collapsed")
with c_hor:
    st.markdown('<div class="filter-label">Horizonte</div>', unsafe_allow_html=True)
    horizonte_sel = st.selectbox("Horizonte", ["7 días", "14 días", "30 días"], label_visibility="collapsed")
horizonte_dias = int(horizonte_sel.split()[0])

df_filtrado = df_enriquecido.copy()
if linea_sel != "Todas las líneas":
    df_filtrado = df_filtrado[df_filtrado["Linea_Oficial"] == linea_sel]
if sede_sel != "Todas las sedes":
    df_filtrado = df_filtrado[df_filtrado["Sede_Regional"] == sede_sel]
if df_filtrado.empty:
    st.warning("No existen registros para la combinación de línea y corredor seleccionada.")
    st.stop()

avisos = []
if diag["fichas_por_respaldo"]:
    avisos.append(f"{diag['fichas_por_respaldo']} registros no coinciden con el catálogo por Ref ni por nombre exacto y usan una ficha por palabra clave (vida útil y lote aproximados).")
if diag["filas_sin_sede"]:
    avisos.append(f"{diag['filas_sin_sede']} registros quedaron «{SIN_ASIGNAR}» (localidad ausente o ambigua entre corredores).")
if avisos:
    st.info("Calidad de datos: " + " ".join(avisos))

df_diario_ml = df_filtrado.groupby(["Fecha", "Ref", "Desc item", "Linea_Oficial", "Presentacion", "Canales", "Vida_Util_Dias", "Precio_Cat"],
                                   as_index=False)["Cant inv"].sum()
refs_param = tuple(df_param["Ref"].astype(str))
res = pronosticar_cache(df_diario_ml, refs_param)
costos = construir_costos(res["meta_map"], costo_var_pct, penalizacion_pct, costos_override)
df_proy, df_manana = decidir(res, df_param, costos, postura_riesgo, horizonte_dias, nivel_larga)
ultima_fecha, met = res["ultima_fecha"], res["metricas"]
n_costos_reales = sum(1 for c in costos.values() if c["fuente"] == "real")

DIAS = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
ev = res["ml"][res["ml"]["h"] <= 7][["Fecha", "Evento_Calendario"]].drop_duplicates("Fecha")
eventos_proximos = [f"{DIAS[f.day_of_week]} ({e})" for f, e in zip(ev["Fecha"], ev["Evento_Calendario"]) if e != "Día hábil / regular"]

# ------------------------------------------------------------------------------ vistas
if seccion_activa == "Resumen":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    demanda = int(df_manana["Pronóstico_num"].sum()) if not df_manana.empty else 0
    orden = int(df_manana["Producir"].sum()) if not df_manana.empty else 0
    var_prom = float(df_manana["Var %"].mean()) if not df_manana.empty else 0.0
    n_riesgo = int((df_manana["Estado"] != "Estable").sum()) if not df_manana.empty else 0

    if met["wape"] is not None:
        k_wape = f"{fmt_n(met['wape'], 2)} %"
        k_wape_sub = f"Naive estacional: {fmt_n(met['wape_naive'], 2)} % · Media 4 sem: {fmt_n(met['wape_media4'], 2)} %"
        k_cov = f"{fmt_n(met['cobertura'], 1)} %"
        k_cov_sub = f"Nominal 80 % · n = {met['n']} · Pinball q50: {fmt_n(met['pinball50'], 2)}"
    else:
        k_wape, k_wape_sub, k_cov, k_cov_sub = "N/D", "Historial insuficiente para backtest", "N/D", "Requiere más historia"

    tarjetas = [
        ("kpi-top-cacao", SVG_PAN_ARTESANAL, "Demanda esperada del ciclo (P50)", f"{fmt_n(demanda)} und",
         "kpi-sub-green" if var_prom >= 0 else "kpi-sub-amber", f"{var_prom:+.1f} % vs. misma referencia semana previa".replace(".", ",")),
        ("kpi-top-dorado", SVG_ESPIGA_TRIGO, "WAPE recursivo (1–7 días)", k_wape, "kpi-sub-neutral", k_wape_sub),
        ("kpi-top-queso", SVG_PAN_ARTESANAL, "SKUs con alerta", f"{n_riesgo} SKUs", "kpi-sub-amber",
         f"Orden total: {fmt_n(orden)} und ({postura_riesgo.split()[0]})"),
        ("kpi-top-salvia", SVG_ESPIGA_TRIGO, "Cobertura empírica P10–P90", k_cov, "kpi-sub-green", k_cov_sub),
    ]
    for col, (cls, svg, titulo, valor, subcls, sub) in zip(st.columns(4), tarjetas):
        col.markdown(f'<div class="kpi-card {cls}"><div class="kpi-watermark-icon">{svg}</div><div class="kpi-title">{titulo}</div>'
                     f'<div class="kpi-value">{valor}</div><div class="{subcls}">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    col_chart, col_alerts = st.columns([6.8, 3.2])
    with col_chart:
        with st.container(border=True):
            etq_banda = "Banda P10–P90 (calibrada)" if res["banda"] else "Banda aproximada (no calibrada)"
            st.markdown('<div class="card-header-flex"><div class="card-title">Demanda observada y proyección agregada</div><div class="chart-legend">'
                        '<span class="legend-item"><span class="legend-line-solid"></span>Observada</span>'
                        '<span class="legend-item"><span class="legend-line-forecast"></span>Mediana P50</span>'
                        f'<span class="legend-item"><span class="legend-band"></span>{etq_banda}</span></div></div>', unsafe_allow_html=True)
            df_tot = df_filtrado.groupby("Fecha", as_index=False)["Cant inv"].sum()
            st.plotly_chart(construir_grafico_tendencia(df_tot, df_proy, ultima_fecha, horizonte_dias, res["banda"]),
                            width="stretch", config={"displayModeBar": False})
    with col_alerts:
        if not df_manana.empty:
            top = df_manana.iloc[0]
            a1_t = str(top["Producto"])
            a1_d = f"Variación {top['Var %']:+.1f} % · vida útil {int(top['Vida_Util_num'])} días · orden sugerida {int(top['Cajas'])} cajas."
        else:
            a1_t, a1_d = "Sin alertas activas", "No hay referencias pendientes de ajuste."
        txt_cal = (f"Próximos hitos (7 días): {', '.join(eventos_proximos[:2])}." if eventos_proximos
                   else f"Semana regular sin festivos ni quincenas en {sede_sel if sede_sel != 'Todas las sedes' else 'la Costa Caribe'}.")
        txt_cal_motor = (f"WAPE recursivo {k_wape} · cobertura P10–P90 {k_cov} ({res['dias_historia']} días de historia)."
                         if met["wape"] is not None else f"{res['motor']} ({res['dias_historia']} días).")
        st.markdown('<div class="white-card" style="height: 100%;"><div class="card-title" style="margin-bottom: 14px;">Alertas dinámicas</div>'
                    f'<div class="alert-box alert-red"><div class="alert-title">{a1_t}</div><div class="alert-desc">{a1_d}</div></div>'
                    f'<div class="alert-box alert-amber"><div class="alert-title">Sensores de calendario local</div><div class="alert-desc">{txt_cal}</div></div>'
                    f'<div class="alert-box alert-green"><div class="alert-title">Estado del motor cuantílico</div><div class="alert-desc">{txt_cal_motor}</div></div></div>',
                    unsafe_allow_html=True)
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        t1, t2 = st.columns([7.5, 2.5])
        ev_m = df_manana["Evento Mañana"].iloc[0] if not df_manana.empty else "Día regular"
        t1.markdown(f'<div class="card-title" style="padding-top: 6px;">Directrices de abastecimiento '
                    f'<span style="font-size:0.78rem; font-weight:500; color:#7C6350;">({ev_m} · {postura_riesgo})</span></div>', unsafe_allow_html=True)
        cols_exp = ["Producto", "Línea", "Presentación", "Canales", "Evento Mañana", "Ciclo (días)", "Pronóstico", "Rango P10–P90", "Nivel_Servicio_Opt", "Regla", "Producir", "Cajas", "Estado"]
        t2.download_button("Exportar plan de producción", df_manana[cols_exp].to_csv(index=False, sep=";").encode("utf-8-sig"),
                           file_name=f"Plan_Abastecimiento_Servipan_{(ultima_fecha + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.csv",
                           mime="text/csv", width="stretch")
        clases = {"Estable": "badge-stable", "Subir producción": "badge-increase"}
        filas = "".join(
            f'<tr><td><div class="prod-name">{r["Producto"]}</div><div class="prod-sub">{r["Línea"]} · {r["Presentación"]} · nivel {r["Nivel_Servicio_Opt"]} % · ciclo {r["Ciclo (días)"]} d</div></td>'
            f'<td>{r["Pronóstico"]}</td><td class="prod-range">{r["Rango P10–P90"]}</td>'
            f'<td class="prod-qty">{fmt_n(r["Producir"])} <span class="prod-sub">({r["Cajas"]} cajas)</span></td>'
            f'<td><span class="badge {clases.get(r["Estado"], "badge-risk")}">{r["Estado"]}</span></td></tr>'
            for _, r in df_manana.head(10).iterrows())
        st.markdown('<table class="prod-table"><thead><tr><th>Producto</th><th>Mediana P50 (ciclo)</th><th>Rango P10–P90</th><th>Orden objetivo</th><th>Estado</th></tr></thead>'
                    f'<tbody>{filas}</tbody></table>', unsafe_allow_html=True)

elif seccion_activa == "Pronóstico":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    col_m, col_c = st.columns([7.2, 2.8])
    with col_m:
        with st.container(border=True):
            prod_sel = st.selectbox("Seleccionar producto:", df_proy["Producto"].unique())
            d = df_proy[df_proy["Producto"] == prod_sel]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=d["Fecha Proyectada"], y=d["P90"], mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=d["Fecha Proyectada"], y=d["P10"], mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor="rgba(245, 214, 107, 0.35)", name="Rango P10–P90"))
            fig.add_trace(go.Scatter(x=d["Fecha Proyectada"], y=d["Pronóstico P50"], mode="lines+markers",
                                     line=dict(color="#6B2E12", width=2.6), marker=dict(color="#D97724", size=7), name="Pronóstico P50"))
            fig.update_layout(title=f"Curva de demanda proyectada ({horizonte_sel}) · {prod_sel}", height=310,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=20))
            st.plotly_chart(fig, width="stretch")
    with col_c:
        info = df_manana[df_manana["Producto"] == prod_sel]
        if not info.empty:
            r = info.iloc[0]
            img = None
            if "queso" in prod_sel.lower():
                img = cargar_imagen_base64("imagen_2.jpg")
            elif "tajado" in prod_sel.lower() or "artesanal" in prod_sel.lower():
                img = cargar_imagen_base64("imagen_3.jpg")
            img = img or cargar_imagen_base64("imagen.jpg")
            visual = (f'<img src="{img}" style="width:100%; max-height:175px; object-fit:cover; border-radius:10px; margin-bottom:10px; border:1px solid #E5C8A6;" />'
                      if img else f'<div style="width:145px; margin:6px auto 10px auto;">{SVG_PAN_ARTESANAL}</div>')
            st.markdown(f'<div class="spotlight-card"><div class="spotlight-badge">{r["Línea"]}</div>{visual}'
                        f'<div class="spotlight-title">{r["Producto"]}</div><div class="spotlight-meta">'
                        f'<b>Presentación:</b> {r["Presentación"]}<br><b>Canales:</b> {r["Canales"]}<br><b>Vida útil:</b> {r["Vida Útil"]}<br>'
                        f'<b>Regla:</b> {r["Regla"]} ({r["Nivel_Servicio_Opt"]} %)<br><b>Orden del ciclo ({r["Ciclo (días)"]} d):</b> {fmt_n(r["Producir"])} und ({r["Cajas"]} cajas)</div></div>',
                        unsafe_allow_html=True)
    with st.container(border=True):
        st.dataframe(d[["Fecha Proyectada", "Día", "Evento Calendario", "Producto", "Pronóstico P50", "P10", "P90", "Nivel servicio",
                        "Cuantil Objetivo", "Producir (Un)", "Sugerido Producir (Cajas)"]], width="stretch", hide_index=True)

elif seccion_activa == "Productos":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f'<div class="card-title" style="margin-bottom: 10px;">Portafolio oficial Servipan (4 líneas · {len(CATALOGO_OFICIAL_SERVIPAN)} referencias)</div>', unsafe_allow_html=True)
        cat = pd.DataFrame(CATALOGO_OFICIAL_SERVIPAN)
        cat["Canales de Distribución"] = cat.apply(formatear_canales, axis=1)
        f1, f2 = st.columns(2)
        lin = f1.selectbox("Filtrar ficha por línea:", ["Todas"] + sorted(cat["Linea"].unique().tolist()))
        can = f2.selectbox("Filtrar por canal:", ["Todos", "Tiendas", "Supermercados", "Institucional"])
        if lin != "Todas":
            cat = cat[cat["Linea"] == lin]
        if can != "Todos":
            cat = cat[cat[can]]
        st.dataframe(cat[["Ref", "Linea", "Producto", "Presentacion", "Canales de Distribución", "Vida_Util_Dias", "Und_Caja", "Min_Cajas", "Precio"]].rename(
            columns={"Linea": "Línea", "Presentacion": "Presentación", "Vida_Util_Dias": "Vida útil (días)", "Und_Caja": "Und / caja",
                     "Min_Cajas": "Lote mín (cajas)", "Precio": "Precio de lista (COP)"}), width="stretch", hide_index=True)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 10px;">Consulta histórica de registros</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        df_op = pd.merge(df_filtrado, df_param, on="Ref", how="left")
        df_op["Cantidad de cajas"] = np.ceil(df_op["Cant inv"] / df_op["Unidades por Caja"])
        p_sel = c1.selectbox("Filtrar por producto:", ["Todos"] + sorted(df_op["Desc item"].dropna().unique().tolist()))
        if p_sel != "Todos":
            df_op = df_op[df_op["Desc item"] == p_sel]
        if col_cliente:
            cli = c2.selectbox("Filtrar por punto / cliente:", ["Todos"] + sorted(df_op[col_cliente].dropna().unique().tolist()))
            if cli != "Todos":
                df_op = df_op[df_op[col_cliente] == cli]
        st.dataframe(df_op, width="stretch")
        st.download_button("Exportar tabla operativa (CSV)", df_op.to_csv(index=False, sep=";").encode("utf-8-sig"),
                           file_name="Consulta_Operativa_Servipan.csv", mime="text/csv")

elif seccion_activa == "Sedes":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 6px;">Demanda histórica por corredor y asignación por participación histórica</div>', unsafe_allow_html=True)
        st.caption("El modelo no pronostica por corredor: la asignación reparte el pronóstico total según la participación histórica.")
        z = df_enriquecido.groupby("Sede_Regional", as_index=False)["Cant inv"].sum().sort_values("Cant inv")
        tot = max(1.0, float(z["Cant inv"].sum()))
        total_hor = float(df_proy["Pronóstico P50"].sum()) if not df_proy.empty else 0.0
        z["Participación (%)"] = (z["Cant inv"] / tot * 100).round(1)
        z[f"Asignación P50 ({horizonte_sel})"] = (z["Cant inv"] / tot * total_hor).round().astype(int)
        fig_z = px.bar(z, y="Sede_Regional", x="Cant inv", orientation="h", text_auto=".3s", color_discrete_sequence=["#6B2E12"],
                       labels={"Sede_Regional": "Corredor regional", "Cant inv": "Unidades históricas"})
        fig_z.update_layout(height=280, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_z, width="stretch")
        st.dataframe(z.sort_values("Cant inv", ascending=False), width="stretch", hide_index=True)
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 10px;">Directorio de corredores comerciales</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, (nombre, info) in enumerate(LUGARES_DISTRIBUCION_SERVIPAN.items()):
            with cols[i % 2]:
                with st.expander(f"📍 {nombre} ({len(info['municipios'])} localidades)", expanded=(i == 0)):
                    st.write(", ".join(info["municipios"]))

elif seccion_activa == "Escenarios":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Simulador de choques de demanda</div>', unsafe_allow_html=True)
        ajuste = st.slider("Variación esperada de demanda por campaña o choque externo (%)", -30, 50, 18, 2)
        _, df_sc = decidir(res, df_param, costos, postura_riesgo, horizonte_dias, nivel_larga, choque_pct=ajuste)
        base = df_manana[["Ref", "Producto", "Línea", "Ciclo (días)", "Pronóstico_num", "Producir", "Estado"]].rename(
            columns={"Pronóstico_num": "P50 base", "Producir": "Orden base (und)", "Estado": "Estado base"})
        sc = df_sc[["Ref", "Pronóstico_num", "Producir", "Estado"]].rename(
            columns={"Pronóstico_num": "P50 simulado", "Producir": "Orden simulada (und)", "Estado": "Estado simulado"})
        sim = base.merge(sc, on="Ref")
        sim["Δ orden (und)"] = sim["Orden simulada (und)"] - sim["Orden base (und)"]
        st.caption("El choque desplaza toda la distribución (P10–P90) y se recalculan nivel de servicio, cajas y lotes mínimos; no es una proporción lineal.")
        st.dataframe(sim.drop(columns="Ref"), width="stretch", hide_index=True)

elif seccion_activa == "Inventario":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Conversión a cajas comerciales, lotes mínimos y frescura</div>', unsafe_allow_html=True)
        inv = pd.merge(df_manana, df_param, on="Ref", how="left")
        st.dataframe(inv[["Producto", "Línea", "Presentación", "Vida Útil", "Regla", "Ciclo (días)", "Pronóstico", "Nivel_Servicio_Opt",
                          "Unidades por Caja", "Mínimo a Producir (Cajas)", "Producir", "Cajas", "Estado"]].rename(
            columns={"Nivel_Servicio_Opt": "Nivel de servicio (%)"}), width="stretch", hide_index=True)

elif seccion_activa == "Alertas":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    ev_txt = ", ".join(eventos_proximos) if eventos_proximos else "Sin festivos ni quincenas en los próximos 7 días"
    riesgos = df_manana[df_manana["Estado"] != "Estable"] if not df_manana.empty else pd.DataFrame()
    if not riesgos.empty:
        lista = ", ".join(f"{r['Producto']} ({r['Var %']:+.1f} %)" for _, r in riesgos.head(4).iterrows())
        tr, dr = f"{len(riesgos)} referencias con alerta activa", f"SKUs priorizados: {lista}. El estado depende solo de los datos, no de la postura elegida."
    else:
        tr, dr = "Portafolio en rango operativo estable", "Ningún SKU supera el umbral de crecimiento (+8 %) ni presenta picos críticos de perecibilidad."
    w = f"{met['wape']} %" if met["wape"] is not None else "N/D"
    cv = f"{met['cobertura']} %" if met["cobertura"] is not None else "N/D"
    st.markdown('<div class="white-card"><div class="card-title" style="margin-bottom: 14px;">Bitácora de alertas tempranas</div>'
                f'<div class="alert-box alert-red"><div class="alert-title">{tr}</div><div class="alert-desc">{dr}</div></div>'
                f'<div class="alert-box alert-amber"><div class="alert-title">Calendario (próximos 7 días)</div><div class="alert-desc">{ev_txt}.</div></div>'
                f'<div class="alert-box alert-green"><div class="alert-title">Calibración · WAPE recursivo {w} · cobertura P10–P90 {cv}</div>'
                f'<div class="alert-desc">{res["dias_historia"]} días de historia con {res["motor"]}.</div></div></div>', unsafe_allow_html=True)

elif seccion_activa == "Configuración":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Arquitectura del motor y regla de decisión</div>', unsafe_allow_html=True)
        st.write(f"- **Origen de datos:** `{'Benchmark sintético reproducible (180 días): solo valida el pipeline, NO el desempeño real' if es_ejemplo else 'Excel cargado por el usuario'}`.")
        st.write(f"- **Algoritmo:** `{res['motor']}`, global sobre las {len(res['refs'])} referencias activas.")
        st.write(f"- **{N_BASE_VARS + N_CAL_VARS} variables:** {N_CAL_VARS} de calendario ({len(GRUPOS_CALENDARIO['basico'])} básicas, "
                 f"{len(GRUPOS_CALENDARIO['legal_economico'])} legales/económicas, {len(GRUPOS_CALENDARIO['sociocultural'])} socioculturales) "
                 f"y {N_BASE_VARS} de producto y memoria (`Ref_ID`, `Linea_ID`, `Vida_Util_Dias`, `lag_1`, `lag_7`, `lag_14`, `roll_mean_7`, `roll_mean_28`, `roll_std_7`).")
        st.write(f"- **Decisión:** newsvendor de un período (CR* = cu / (cu + co)) para vida útil ≤ {UMBRAL_VIDA_CORTA} días; nivel de servicio objetivo para vida larga. "
                 "No hay buffers fijos adicionales. La orden cubre el ciclo de reposición (7 / frecuencia de cargue) sumando cuantiles diarios.")
        st.write(f"- **Costos:** {n_costos_reales} SKU con costo real cargado; el resto usa el supuesto de {int(costo_var_pct * 100)} % del precio de lista. "
                 "Sin costos reales, el nivel de servicio óptimo no es defendible.")
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Auditoría experimental</div>', unsafe_allow_html=True)
        st.caption("4 orígenes móviles × 30 días, pronóstico recursivo, ablación en 4 niveles y evaluación económica de políticas.")
        if es_ejemplo:
            st.warning("Con datos sintéticos los efectos de calendario están programados en el generador: la ablación y el WAPE son circulares. Úselos solo para probar el pipeline.")
        if st.button("Ejecutar auditoría completa", type="primary"):
            st.session_state["auditoria_activa"] = True
        if st.session_state.get("auditoria_activa"):
            with st.spinner("Entrenando orígenes móviles y variantes de ablación…"):
                aud = auditar_cache(df_diario_ml, refs_param)
            if not aud:
                st.warning("Se requieren al menos 75 días de historia para la auditoría a 30 días.")
            else:
                st.markdown("**1. Desempeño por horizonte (referencias: naive estacional y media de 4 semanas)**")
                st.dataframe(aud["resumen"], width="stretch", hide_index=True)
                st.markdown("**2. Ablación de variables (mismos orígenes y horizontes)**")
                st.dataframe(aud["ablacion"], width="stretch", hide_index=True)
                c1, c2 = st.columns(2)
                c1.markdown("**3. Estabilidad por origen**")
                c1.dataframe(aud["por_origen"], width="stretch", hide_index=True)
                c2.markdown("**4. WAPE por línea**")
                c2.dataframe(aud["por_linea"], width="stretch", hide_index=True)
                pol_corta, pol_larga = evaluar_politicas(aud["bt"], aud["meta_map"], df_param, costos, nivel_larga)
                st.markdown(f"**5. Políticas · vida útil ≤ {UMBRAL_VIDA_CORTA} días (costo = faltante × cu + sobrante × co)**")
                st.dataframe(pol_corta, width="stretch", hide_index=True)
                st.markdown(f"**6. Políticas · vida útil > {UMBRAL_VIDA_CORTA} días (solo servicio y excedente; el sobrante no es merma)**")
                st.dataframe(pol_larga, width="stretch", hide_index=True)
                sens = sensibilidad_costos(aud["bt"], aud["meta_map"], df_param)
                if not sens.empty:
                    st.markdown("**7. Sensibilidad: ahorro del newsvendor frente a P50 puro (%) según costo variable y penalización**")
                    st.dataframe(sens.pivot(index="Costo var. %", columns="Penal. quiebre %", values="Ahorro vs P50 %"), width="stretch")
                    st.caption("Mejor política en la cuadrícula: " + ", ".join(f"{k}: {v} casos" for k, v in sens["Mejor política"].value_counts().items()))
