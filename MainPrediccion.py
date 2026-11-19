import base64
from datetime import date, timedelta
import os
import unicodedata
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    SKLEARN_DISPONIBLE = True
except ImportError:
    SKLEARN_DISPONIBLE = False

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Servipan · Inteligencia de Mercado y Demanda",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# 2. RECURSOS VISUALES VECTORIALES E IMÁGENES DE MARCA
# ==============================================================================
def cargar_imagen_base64(ruta: str) -> str | None:
    """Convierte una imagen local (jpg/png) a base64 si existe en el directorio."""
    if os.path.exists(ruta):
        ext = os.path.splitext(ruta)[1].lower().replace(".", "")
        mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
        with open(ruta, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"
    return None


SVG_PAN_ARTESANAL = '<svg viewBox="0 0 260 170" fill="none" xmlns="http://www.w3.org/2000/svg"><ellipse cx="132" cy="108" rx="108" ry="44" transform="rotate(-24 132 108)" fill="#E6D9C5"/><path d="M35 112C22 92 38 62 84 42C130 22 192 14 218 36C244 58 228 92 182 114C136 136 48 132 35 112Z" fill="#DF7A22" stroke="#6B2E12" stroke-width="5.5" stroke-linejoin="round"/><path d="M46 108C72 122 134 122 178 102C206 89 220 71 222 55C228 73 214 96 180 112C134 133 56 128 46 108Z" fill="#F6B25E"/><path d="M72 54C82 65 89 79 90 93" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M77 56C85 65 90 77 91 89" stroke="#FDE2A8" stroke-width="3" stroke-linecap="round"/><path d="M106 41C117 53 124 68 125 83" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M111 43C120 53 125 66 126 79" stroke="#FDE2A8" stroke-width="3" stroke-linecap="round"/><path d="M142 31C153 43 160 58 161 73" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M147 33C156 43 161 56 162 69" stroke="#FDE2A8" stroke-width="3" stroke-linecap="round"/><path d="M177 27C187 38 193 51 194 64" stroke="#6B2E12" stroke-width="4.5" stroke-linecap="round"/><path d="M182 29C190 38 194 49 195 60" stroke="#FDE2A8" stroke-width="3" stroke-linecap="round"/></svg>'

SVG_ESPIGA_TRIGO = '<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M22 98C44 76 68 52 96 24" stroke="#6B2E12" stroke-width="2.2" stroke-linecap="round"/><ellipse cx="48" cy="60" rx="5" ry="11" transform="rotate(-45 48 60)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="60" cy="72" rx="5" ry="11" transform="rotate(45 60 72)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="62" cy="46" rx="5" ry="11" transform="rotate(-45 62 46)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="74" cy="58" rx="5" ry="11" transform="rotate(45 74 58)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="76" cy="32" rx="5" ry="11" transform="rotate(-45 76 32)" stroke="#6B2E12" stroke-width="2"/><ellipse cx="88" cy="44" rx="5" ry="11" transform="rotate(45 88 44)" stroke="#6B2E12" stroke-width="2"/><path d="M96 24L106 14M88 22L94 12M98 32L108 26" stroke="#6B2E12" stroke-width="1.8" stroke-linecap="round"/></svg>'

# ==============================================================================
# 3. ESTILO VISUAL Y TEMA DE IDENTIDAD SERVIPAN (CUSTOM CSS)
# ==============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #FAF5EB;
    color: #26160C;
    font-variant-numeric: tabular-nums;
}

.block-container {
    padding-top: 1.55rem !important;
    padding-bottom: 2.2rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
    max-width: 1440px;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stHeaderActionElements"] {display: none !important;}
header[data-testid="stHeader"] {
    background-color: rgba(250, 245, 235, 0);
}

section[data-testid="stSidebar"] {
    background-color: #FFFDF9 !important;
    border-right: 1px solid #EADBC8 !important;
    width: 250px !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
    padding-left: 0.9rem;
    padding-right: 0.9rem;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.30rem !important;
}

.servipan-emblem-wrap {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 4px 6px 16px 6px;
    margin-bottom: 6px;
    border-bottom: 1px dashed #E8D8C3;
}
.servipan-oval {
    background: linear-gradient(180deg, #FFFDF5 0%, #F9EBC8 100%);
    border: 2.5px solid #6B2E12;
    box-shadow: 0 0 0 2px #FFFDF5, 0 2px 6px rgba(107, 46, 18, 0.16);
    border-radius: 999px;
    padding: 4px 14px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
}
.servipan-oval span {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 700;
    font-style: italic;
    font-size: 1.05rem;
    color: #6B2E12;
    letter-spacing: -0.02em;
}
.servipan-sub-tag {
    font-size: 0.72rem;
    font-weight: 600;
    color: #9A5B25;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background-color: #F4E3CE !important;
    color: #6B2E12 !important;
    border: 1px solid #E5C8A6 !important;
    border-radius: 10px !important;
    padding: 0.60rem 1rem !important;
    justify-content: flex-start !important;
    text-align: left !important;
    box-shadow: 0 1px 2px rgba(107, 46, 18, 0.06) !important;
    min-height: 41px !important;
}
section[data-testid="stSidebar"] button[kind="primary"] p,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p {
    color: #6B2E12 !important;
    font-weight: 600 !important;
    font-size: 0.91rem !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
    background-color: transparent !important;
    color: #4A3728 !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    padding: 0.60rem 1rem !important;
    justify-content: flex-start !important;
    text-align: left !important;
    box-shadow: none !important;
    min-height: 41px !important;
}
section[data-testid="stSidebar"] button[kind="secondary"] p,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] p {
    color: #4A3728 !important;
    font-weight: 500 !important;
    font-size: 0.91rem !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #F9F1E6 !important;
}

section[data-testid="stSidebar"] details {
    background-color: #FAF3E8 !important;
    border: 1px solid #E6D5BE !important;
    border-radius: 10px !important;
    margin-top: 6px !important;
}
section[data-testid="stSidebar"] details summary {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: #6B2E12 !important;
}

.sidebar-doodle {
    margin-top: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    opacity: 0.22;
    pointer-events: none;
}
.sidebar-doodle svg {
    width: 56px;
    height: 56px;
}

.main-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.72rem;
    font-weight: 700;
    color: #6B2E12;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.18;
}
.main-subtitle {
    font-size: 0.84rem;
    color: #7C6350;
    margin-top: 4px;
    margin-bottom: 0;
}

.data-badge-demo {
    display: inline-block;
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #F59E0B;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
    margin-left: 8px;
    vertical-align: middle;
}
.data-badge-real {
    display: inline-block;
    background-color: #E3F3EE;
    color: #1F514D;
    border: 1px solid #4D8B86;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 999px;
    margin-left: 8px;
    vertical-align: middle;
}

.filter-label {
    font-size: 0.74rem;
    font-weight: 600;
    color: #7C6350;
    margin-bottom: 3px;
}
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E2D2BC !important;
    border-radius: 9px !important;
    font-size: 0.84rem !important;
    color: #26160C !important;
    min-height: 36px !important;
}

.kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #EADBC8;
    border-radius: 12px;
    padding: 15px 18px;
    box-shadow: 0 3px 10px rgba(107, 46, 18, 0.04);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.kpi-top-cacao { border-top: 3.5px solid #6B2E12; }
.kpi-top-dorado { border-top: 3.5px solid #D97724; }
.kpi-top-queso { border-top: 3.5px solid #EAB308; }
.kpi-top-salvia { border-top: 3.5px solid #7FA6A3; }

.kpi-watermark-icon {
    position: absolute;
    right: -10px;
    bottom: -12px;
    width: 78px;
    opacity: 0.07;
    pointer-events: none;
}
.kpi-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #7C6350;
    margin-bottom: 7px;
}
.kpi-value {
    font-size: 1.62rem;
    font-weight: 700;
    color: #26160C;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin-bottom: 8px;
}
.kpi-sub-green {
    font-size: 0.76rem;
    font-weight: 600;
    color: #15803D;
    line-height: 1.35;
}
.kpi-sub-neutral {
    font-size: 0.76rem;
    font-weight: 400;
    color: #7C6350;
    line-height: 1.35;
}
.kpi-sub-amber {
    font-size: 0.76rem;
    font-weight: 600;
    color: #B45309;
    line-height: 1.35;
}

.white-card {
    background-color: #FFFFFF;
    border: 1px solid #EADBC8;
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 3px 10px rgba(107, 46, 18, 0.04);
}
.card-header-flex {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 8px;
}
.card-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #6B2E12;
    margin: 0;
}

.chart-legend {
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.75rem;
    color: #5C4332;
}
.legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.legend-line-solid {
    width: 13px;
    height: 2.5px;
    background-color: #6B2E12;
    display: inline-block;
    border-radius: 2px;
}
.legend-line-forecast {
    width: 13px;
    height: 2.5px;
    background-color: #D97724;
    display: inline-block;
    border-radius: 2px;
}
.legend-band {
    width: 13px;
    height: 10px;
    background-color: rgba(245, 214, 107, 0.45);
    border: 1px solid rgba(217, 119, 36, 0.35);
    border-radius: 2px;
    display: inline-block;
}

.alert-box {
    border-radius: 10px;
    padding: 13px 15px;
    margin-bottom: 11px;
    border-left: 4px solid transparent;
}
.alert-box:last-child {
    margin-bottom: 0;
}
.alert-red {
    background-color: #FDE8E8;
    border-left-color: #B91C1C;
}
.alert-red .alert-title {
    color: #9B1C1C;
    font-weight: 700;
    font-size: 0.83rem;
    margin-bottom: 3px;
}
.alert-red .alert-desc {
    color: #881313;
    font-size: 0.77rem;
    line-height: 1.38;
}

.alert-amber {
    background-color: #FEF3C7;
    border-left-color: #D97724;
}
.alert-amber .alert-title {
    color: #92400E;
    font-weight: 700;
    font-size: 0.83rem;
    margin-bottom: 3px;
}
.alert-amber .alert-desc {
    color: #78350F;
    font-size: 0.77rem;
    line-height: 1.38;
}

.alert-green {
    background-color: #E3F3EE;
    border-left-color: #4D8B86;
}
.alert-green .alert-title {
    color: #1F514D;
    font-weight: 700;
    font-size: 0.83rem;
    margin-bottom: 3px;
}
.alert-green .alert-desc {
    color: #2B625D;
    font-size: 0.77rem;
    line-height: 1.38;
}

div.stDownloadButton > button {
    background-color: #6B2E12 !important;
    color: #FFFDF9 !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 0.48rem 1rem !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 5px rgba(107, 46, 18, 0.25) !important;
}
div.stDownloadButton > button:hover {
    background-color: #52210B !important;
    color: #FFFFFF !important;
}

.prod-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 8px;
    font-size: 0.85rem;
}
.prod-table thead th {
    text-align: left;
    color: #7C6350;
    font-weight: 600;
    font-size: 0.78rem;
    padding: 10px 12px;
    border-bottom: 1.5px solid #EADBC8;
    background-color: #FCF9F4;
}
.prod-table tbody td {
    padding: 12px 12px;
    border-bottom: 1px solid #F3EAE0;
    color: #26160C;
    vertical-align: middle;
}
.prod-table tbody tr:last-child td {
    border-bottom: none;
}
.prod-table tbody tr:hover {
    background-color: #FAF3E8;
}
.prod-name {
    font-weight: 600;
    color: #26160C;
}
.prod-sub {
    font-size: 0.73rem;
    color: #7C6350;
}
.prod-qty {
    font-weight: 700;
    color: #6B2E12;
}
.prod-range {
    color: #7C6350;
}

.badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}
.badge-stable {
    background-color: #E3F3EE;
    color: #1F514D;
}
.badge-increase {
    background-color: #FEF3C7;
    color: #92400E;
}
.badge-risk {
    background-color: #FDE8E8;
    color: #9B1C1C;
}

.spotlight-card {
    background: linear-gradient(135deg, #FFFDF7 0%, #F9EBC8 100%);
    border: 1px solid #E5C8A6;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.spotlight-badge {
    display: inline-block;
    background-color: #6B2E12;
    color: #FFFDF9;
    font-size: 0.70rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.spotlight-title {
    font-family: 'Fraunces', Georgia, serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #6B2E12;
    margin: 6px 0 4px 0;
}
.spotlight-meta {
    font-size: 0.78rem;
    color: #5C4332;
    line-height: 1.45;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EADBC8 !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    box-shadow: 0 3px 10px rgba(107, 46, 18, 0.04) !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 4. MOTOR DE CALENDARIO COLOMBIANO, FESTIVIDADES DEL CARIBE Y VACACIONES
# ==============================================================================
FESTIVAL_VALLENATO_FECHAS = {
    2024: (date(2024, 4, 26), date(2024, 4, 30)),
    2025: (date(2025, 4, 30), date(2025, 5, 3)),
    2026: (date(2026, 4, 29), date(2026, 5, 2)),
    2027: (date(2027, 4, 28), date(2027, 5, 1)),
}


def calcular_pascua(anio: int) -> date:
    """Algoritmo de Meeus/Jones/Butcher para Domingo de Pascua."""
    a = anio % 19
    b = anio // 100
    c = anio % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(anio, mes, dia)


def trasladar_lunes_emiliani(d: date) -> date:
    """Ley 51 de 1983 (Ley Emiliani): traslada al siguiente lunes si no cae lunes."""
    return d if d.weekday() == 0 else d + timedelta(days=(7 - d.weekday()))


def obtener_calendario_colombia(anios: list[int]) -> dict:
    """Genera sets exactos de festivos nacionales (Ley 51/1983), Carnaval (Sáb-Mar antes de Ceniza), Semana Santa y Festival Vallenato."""
    festivos = set()
    carnavales = set()
    semana_santa = set()
    receso_octubre = set()
    fest_vallenato = set()

    for anio in anios:
        pascua = calcular_pascua(anio)
        festivos.update(
            [
                date(anio, 1, 1),   # Año Nuevo
                date(anio, 5, 1),   # Día del Trabajo
                date(anio, 7, 20),  # Independencia de Colombia
                date(anio, 8, 7),   # Batalla de Boyacá
                date(anio, 12, 8),  # Inmaculada Concepción
                date(anio, 12, 25), # Navidad
                pascua - timedelta(days=3),  # Jueves Santo
                pascua - timedelta(days=2),  # Viernes Santo
                trasladar_lunes_emiliani(date(anio, 1, 6)),    # Reyes Magos
                trasladar_lunes_emiliani(date(anio, 3, 19)),   # San José
                trasladar_lunes_emiliani(pascua + timedelta(days=39)),  # Ascensión
                trasladar_lunes_emiliani(pascua + timedelta(days=60)),  # Corpus Christi
                trasladar_lunes_emiliani(pascua + timedelta(days=68)),  # Sagrado Corazón
                trasladar_lunes_emiliani(date(anio, 6, 29)),   # San Pedro y San Pablo
                trasladar_lunes_emiliani(date(anio, 8, 15)),   # Asunción de la Virgen
                trasladar_lunes_emiliani(date(anio, 10, 12)),  # Día de la Raza
                trasladar_lunes_emiliani(date(anio, 11, 1)),   # Todos los Santos
                trasladar_lunes_emiliani(date(anio, 11, 11)),  # Independencia de Cartagena
            ]
        )
        # Carnaval de Barranquilla: sábado (-50) a martes (-47) antes de Miércoles de Ceniza (-46)
        for offset in range(50, 46, -1):
            carnavales.add(pascua - timedelta(days=offset))
        # Semana Santa completa (Domingo de Ramos a Domingo de Pascua)
        for offset in range(7, -1, -1):
            semana_santa.add(pascua - timedelta(days=offset))
        # Semana de receso escolar en octubre (Decreto 1373 de 2007)
        oct12 = date(anio, 10, 12)
        lunes_receso = oct12 - timedelta(days=oct12.weekday())
        for offset in range(7):
            receso_octubre.add(lunes_receso + timedelta(days=offset))
        # Festival de la Leyenda Vallenata (Valledupar)
        ini_fv, fin_fv = FESTIVAL_VALLENATO_FECHAS.get(anio, (date(anio, 4, 27), date(anio, 4, 30)))
        cur = ini_fv
        while cur <= fin_fv:
            fest_vallenato.add(cur)
            cur += timedelta(days=1)

    visperas = {f - timedelta(days=1) for f in festivos}
    return {
        "festivos": festivos,
        "visperas": visperas,
        "carnavales": carnavales,
        "semana_santa": semana_santa,
        "receso_octubre": receso_octubre,
        "festival_vallenato": fest_vallenato,
    }


def extraer_features_fecha(fechas: pd.Series, cal: dict) -> pd.DataFrame:
    """Construye las 15 variables explicativas de calendario sociocultural y económico colombiano."""
    dt = pd.to_datetime(fechas)
    d_py = dt.dt.date
    mes = dt.dt.month
    dia = dt.dt.day
    dow = dt.dt.day_of_week

    es_festivo = d_py.isin(cal["festivos"]).astype(int)
    vispera_festivo = d_py.isin(cal["visperas"]).astype(int)

    # Quincenas exactas: día 15 y cierre de mes (30 o fin de febrero), más sábado previo si cae domingo
    es_dia_pago = (dia == 15) | (dia == 30) | ((mes == 2) & dt.dt.is_month_end)
    es_prev_dom = ((dia == 14) | (dia == 29) | ((mes == 2) & (dt + pd.Timedelta(days=1)).dt.is_month_end)) & (dow == 5)
    es_quincena = (es_dia_pago | es_prev_dom).astype(int)

    # Primas legales Colombia (Art. 306 CST: pago hasta 30 jun y 20 dic)
    es_prima = (((mes == 6) & dia.between(15, 30)) | ((mes == 12) & dia.between(10, 20))).astype(int)

    festival_vallenato = d_py.isin(cal["festival_vallenato"]).astype(int)
    carnaval_caribe = d_py.isin(cal["carnavales"]).astype(int)
    semana_santa = d_py.isin(cal["semana_santa"]).astype(int)
    temporada_navidena = (((mes == 12) & (dia >= 15)) | ((mes == 1) & (dia <= 6))).astype(int)

    # Vacaciones Calendario A (Costa Caribe)
    vac_mitad = (((mes == 6) & (dia >= 15)) | ((mes == 7) & (dia <= 15))).astype(int)
    vac_receso_oct = d_py.isin(cal["receso_octubre"]).astype(int)
    vac_fin_colegios = (((mes == 11) & (dia >= 25)) | (mes == 12) | ((mes == 1) & (dia <= 20))).astype(int)
    vac_fin_universidades = (((mes == 11) & (dia >= 25)) | (mes == 12) | (mes == 1) | ((mes == 2) & (dia <= 5))).astype(int)

    vacaciones_colegios = np.maximum.reduce([vac_mitad, vac_receso_oct, vac_fin_colegios])
    vacaciones_universidades = np.maximum.reduce([vac_mitad, vac_fin_universidades])
    periodo_clases_colegio = (1 - vacaciones_colegios) * (dow < 5).astype(int)

    return pd.DataFrame(
        {
            "dow": dow,
            "day": dia,
            "month": mes,
            "is_weekend": (dow >= 5).astype(int),
            "es_festivo": es_festivo,
            "vispera_festivo": vispera_festivo,
            "es_quincena": es_quincena,
            "es_prima": es_prima,
            "festival_vallenato": festival_vallenato,
            "carnaval_caribe": carnaval_caribe,
            "semana_santa": semana_santa,
            "temporada_navidena": temporada_navidena,
            "vacaciones_colegios": vacaciones_colegios,
            "vacaciones_universidades": vacaciones_universidades,
            "periodo_clases_colegio": periodo_clases_colegio,
        },
        index=fechas.index,
    )


def describir_evento_calendario(row_f: pd.Series) -> str:
    etiquetas = []
    if row_f.get("es_festivo", 0) == 1:
        etiquetas.append("Festivo")
    elif row_f.get("vispera_festivo", 0) == 1:
        etiquetas.append("Víspera festivo")
    if row_f.get("festival_vallenato", 0) == 1:
        etiquetas.append("Festival Vallenato")
    if row_f.get("carnaval_caribe", 0) == 1:
        etiquetas.append("Carnaval Caribe")
    if row_f.get("semana_santa", 0) == 1:
        etiquetas.append("Semana Santa")
    if row_f.get("temporada_navidena", 0) == 1:
        etiquetas.append("Navidad")
    if row_f.get("es_prima", 0) == 1:
        etiquetas.append("Prima")
    elif row_f.get("es_quincena", 0) == 1:
        etiquetas.append("Quincena")
    if row_f.get("vacaciones_colegios", 0) == 1:
        etiquetas.append("Vacaciones escolares")
    elif row_f.get("vacaciones_universidades", 0) == 1:
        etiquetas.append("Vacaciones U.")
    return " · ".join(etiquetas) if etiquetas else "Día hábil / regular"


# ==============================================================================
# 5. BASE MAESTRA OFICIAL SERVIPAN (4 LÍNEAS · 23 REFERENCIAS · 6 CORREDORES / 129 LOCALIDADES)
# ==============================================================================
CATALOGO_OFICIAL_SERVIPAN = [
    {"Ref": "PAN-01", "Producto": "Pan Mogolla Blanca", "Linea": "Panes", "Presentacion": "280 gr", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 6, "Und_Caja": 20, "Min_Cajas": 8, "Base_Diaria": 310, "Precio": 3400, "Perfil_Escolar": "familiar"},
    {"Ref": "PAN-02", "Producto": "Pan Mogolla Integral", "Linea": "Panes", "Presentacion": "280 gr", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 6, "Und_Caja": 20, "Min_Cajas": 5, "Base_Diaria": 145, "Precio": 3600, "Perfil_Escolar": "familiar"},
    {"Ref": "PAN-03", "Producto": "Pan Mogolla con bocadillo", "Linea": "Panes", "Presentacion": "Unidad 35g, 60g y 80g", "Tiendas": True, "Supermercados": False, "Institucional": False, "Vida_Util_Dias": 5, "Und_Caja": 30, "Min_Cajas": 6, "Base_Diaria": 280, "Precio": 1800, "Perfil_Escolar": "lonchera"},
    {"Ref": "PAN-04", "Producto": "Pan Hamburguesa", "Linea": "Panes", "Presentacion": "4 unidades (450g)", "Tiendas": False, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 7, "Und_Caja": 16, "Min_Cajas": 6, "Base_Diaria": 190, "Precio": 4500, "Perfil_Escolar": "festivo"},
    {"Ref": "PAN-05", "Producto": "Pan Perro", "Linea": "Panes", "Presentacion": "Paquete Pan Perro de papa", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 7, "Und_Caja": 16, "Min_Cajas": 5, "Base_Diaria": 175, "Precio": 4300, "Perfil_Escolar": "festivo"},
    {"Ref": "PAN-06", "Producto": "Pan de Queso", "Linea": "Panes", "Presentacion": "Paquete extra queso", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 4, "Und_Caja": 20, "Min_Cajas": 8, "Base_Diaria": 260, "Precio": 3800, "Perfil_Escolar": "lonchera"},
    {"Ref": "TAJ-01", "Producto": "Tajado Multicereal", "Linea": "Tajados", "Presentacion": "Bolsa pan tajado", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 10, "Und_Caja": 12, "Min_Cajas": 5, "Base_Diaria": 110, "Precio": 5800, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-02", "Producto": "Bolsa de Pan Artesanal", "Linea": "Tajados", "Presentacion": "Bolsa pan artesanal", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 8, "Und_Caja": 12, "Min_Cajas": 5, "Base_Diaria": 135, "Precio": 6200, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-03", "Producto": "Pan Tajado Blanco XL", "Linea": "Tajados", "Presentacion": "650 g", "Tiendas": False, "Supermercados": False, "Institucional": True, "Vida_Util_Dias": 10, "Und_Caja": 10, "Min_Cajas": 8, "Base_Diaria": 210, "Precio": 6500, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-04", "Producto": "Pan Tajado Integral", "Linea": "Tajados", "Presentacion": "Bolsa pan tajado", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 10, "Und_Caja": 14, "Min_Cajas": 6, "Base_Diaria": 165, "Precio": 5400, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-05", "Producto": "Pan Tajado de Leche", "Linea": "Tajados", "Presentacion": "450g, 500g y 550g", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 9, "Und_Caja": 14, "Min_Cajas": 8, "Base_Diaria": 240, "Precio": 5200, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-06", "Producto": "Pan Tajado Casero", "Linea": "Tajados", "Presentacion": "420 g", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 8, "Und_Caja": 15, "Min_Cajas": 6, "Base_Diaria": 195, "Precio": 4800, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-07", "Producto": "Pan Tajado de Mantequilla", "Linea": "Tajados", "Presentacion": "550 g", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 9, "Und_Caja": 14, "Min_Cajas": 6, "Base_Diaria": 180, "Precio": 5500, "Perfil_Escolar": "familiar"},
    {"Ref": "TAJ-08", "Producto": "Pan Tajado Blanco", "Linea": "Tajados", "Presentacion": "380 g", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 9, "Und_Caja": 16, "Min_Cajas": 10, "Base_Diaria": 340, "Precio": 4400, "Perfil_Escolar": "familiar"},
    {"Ref": "ROS-01", "Producto": "Rosquitas Integrales de Yuca Grandes", "Linea": "Rosquitas", "Presentacion": "Paquete familiar / 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 12, "Min_Cajas": 5, "Base_Diaria": 95, "Precio": 6800, "Perfil_Escolar": "familiar"},
    {"Ref": "ROS-02", "Producto": "Rosquitas Quinua y Chia Grandes", "Linea": "Rosquitas", "Presentacion": "Paquete familiar / 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 12, "Min_Cajas": 4, "Base_Diaria": 85, "Precio": 7200, "Perfil_Escolar": "familiar"},
    {"Ref": "ROS-03", "Producto": "Rosquitas Yuca y Queso Grandes", "Linea": "Rosquitas", "Presentacion": "Paquete familiar / 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 12, "Min_Cajas": 5, "Base_Diaria": 110, "Precio": 7000, "Perfil_Escolar": "familiar"},
    {"Ref": "ROS-04", "Producto": "Rosquitas integrales de Yuca y queso", "Linea": "Rosquitas", "Presentacion": "Unidad 18g / Paquete x 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 24, "Min_Cajas": 6, "Base_Diaria": 150, "Precio": 1500, "Perfil_Escolar": "lonchera"},
    {"Ref": "ROS-05", "Producto": "Rosquitas integrales de Yuca", "Linea": "Rosquitas", "Presentacion": "Unidad 18g / Paquete x 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 24, "Min_Cajas": 6, "Base_Diaria": 140, "Precio": 1500, "Perfil_Escolar": "lonchera"},
    {"Ref": "ROS-06", "Producto": "Rosquitas de Quinua y Chia", "Linea": "Rosquitas", "Presentacion": "Unidad 18g / Paquete x 12 und", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 24, "Min_Cajas": 5, "Base_Diaria": 130, "Precio": 1600, "Perfil_Escolar": "lonchera"},
    {"Ref": "GAL-01", "Producto": "Punto Rojo", "Linea": "Galletas", "Presentacion": "Unidad individual", "Tiendas": True, "Supermercados": False, "Institucional": False, "Vida_Util_Dias": 30, "Und_Caja": 36, "Min_Cajas": 8, "Base_Diaria": 290, "Precio": 1400, "Perfil_Escolar": "lonchera"},
    {"Ref": "GAL-02", "Producto": "Chelitas de Guayaba", "Linea": "Galletas", "Presentacion": "Unidad / Paquete x 6 und (180g)", "Tiendas": True, "Supermercados": False, "Institucional": False, "Vida_Util_Dias": 30, "Und_Caja": 24, "Min_Cajas": 6, "Base_Diaria": 215, "Precio": 3200, "Perfil_Escolar": "lonchera"},
    {"Ref": "GAL-03", "Producto": "Galleta Cuca", "Linea": "Galletas", "Presentacion": "50g / 60 grs (por unidad)", "Tiendas": True, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 35, "Und_Caja": 36, "Min_Cajas": 10, "Base_Diaria": 360, "Precio": 1300, "Perfil_Escolar": "lonchera"},
]

# 6 corredores regionales comerciales (129 localidades asignadas; 126 nombres únicos por homónimos departamentales)
LUGARES_DISTRIBUCION_SERVIPAN = {
    "Cesar - Valledupar": {
        "peso": 0.34,
        "municipios": [
            "Valledupar", "San Diego", "Manaure", "Media Luna", "Atánquez", "Patillal",
            "Los Corazones", "La Paz", "Urumita", "Villanueva", "El Molino", "Codazzi",
            "La Jagua de Ibirico", "Chiriguaná", "La Loma", "Curumaní", "Tamalameque",
            "Pailitas", "Chimichagua", "Arjona", "Astrea", "El Paso", "Candelaria (Cesar)",
            "Sempegua", "Aguachica", "La Gloria (Cesar)", "Regidor", "Río Viejo", "Pelaya",
            "Gamarra", "Bosconia", "El Copey", "Caracolí", "Los Venados",
        ],
    },
    "Guajira - Riohacha": {
        "peso": 0.16,
        "municipios": [
            "Riohacha", "San Juan del Cesar", "Maicao", "La Peña", "La Junta",
            "Cañaveral", "Pandores", "Fonseca", "Distracción", "Buenavista",
            "Barrancas", "Papayal", "Hatonuevo", "Guayacamal", "Albania", "Cuestecita",
        ],
    },
    "Atlántico - Barranquilla": {
        "peso": 0.18,
        "municipios": [
            "Barranquilla", "Soledad", "Sabanalarga", "Arenal", "Pendales", "Santa Rosa",
            "Molineros", "Repelón", "Luruaco", "Manatí", "Santa Cruz", "Isabel López",
            "Pital de Megua", "Baranoa", "Soplaviento", "Santa Lucía", "San Cristóbal",
            "Sabanagrande", "Santo Tomás", "Palmar de Varela", "Ponedera", "Retirada",
            "Martillo", "Candelaria (Atlántico)", "Cascajal", "Puerto Giraldo", "Bohórquez",
            "Campo de la Cruz", "Suán", "Calamar",
        ],
    },
    "Magdalena - Santa Marta": {
        "peso": 0.14,
        "municipios": [
            "Santa Marta", "Fundación", "Ciénaga", "El Difícil", "Pueblo Nuevo",
            "San Ángel", "Tres Esquinas", "Andes", "San Martín de Loba", "Ricaurte",
            "Guaimaral", "Granada", "El Bajo", "Chibolo", "Tenerife", "Palmar",
            "El Pueblito", "Pijiño", "San Fernando (Magdalena)", "Guamal", "Las Flores",
            "Casa de Tabla", "Plato", "La Gloria (Magdalena)", "Mamoncito", "El Banco",
            "Santa Ana", "Santa Inés",
        ],
    },
    "Bolívar - Cartagena": {
        "peso": 0.09,
        "municipios": [
            "Cartagena", "Mompox", "Talaigua Nuevo", "San Sebastián",
            "Menchiquejo", "Rinconada", "San Fernando (Bolívar)", "Guataca",
        ],
    },
    "Sucre - Sincelejo": {
        "peso": 0.09,
        "municipios": [
            "Sincelejo", "Sampués", "La Unión", "Chinú", "Corozal", "San Marcos",
            "Majagual", "Achí", "Zambrano", "Ovejas", "El Carmen de Bolívar (Ruta Sucre)",
            "San Pedro", "Magangué (Ruta Sucre)",
        ],
    },
}


def normalizar_texto(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def formatear_canales(row) -> str:
    canales = []
    if row.get("Tiendas"):
        canales.append("Tiendas")
    if row.get("Supermercados"):
        canales.append("Supermercados")
    if row.get("Institucional"):
        canales.append("Institucional")
    return " · ".join(canales) if canales else "General"


def enriquecer_dataframe_con_maestros(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cruza registros con el catálogo oficial por Ref o nombre exacto y contabiliza asignaciones por respaldo."""
    df_out = df.copy()
    df_cat = pd.DataFrame(CATALOGO_OFICIAL_SERVIPAN)

    mapa_ref = {str(r["Ref"]).strip().upper(): r.to_dict() for _, r in df_cat.iterrows()}
    mapa_nombre = {normalizar_texto(r["Producto"]): r.to_dict() for _, r in df_cat.iterrows()}
    no_mapeados = 0

    def buscar_ficha(row):
        nonlocal no_mapeados
        ref_code = str(row.get("Ref", "")).strip().upper()
        if ref_code in mapa_ref:
            return mapa_ref[ref_code]
        d_norm = normalizar_texto(row.get("Desc item", ""))
        if d_norm in mapa_nombre:
            return mapa_nombre[d_norm]
        no_mapeados += 1
        if "rosquita" in d_norm:
            return {"Linea": "Rosquitas", "Presentacion": "Empaque estándar", "Tiendas": False, "Supermercados": True, "Institucional": True, "Vida_Util_Dias": 45, "Und_Caja": 24, "Min_Cajas": 5, "Precio": 1600}
        if "tajado" in d_norm or "artesanal" in d_norm:
            return {"Linea": "Tajados", "Presentacion": "Bolsa tajado", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 9, "Und_Caja": 14, "Min_Cajas": 6, "Precio": 5200}
        if "galleta" in d_norm or "cuca" in d_norm or "chelita" in d_norm or "punto rojo" in d_norm:
            return {"Linea": "Galletas", "Presentacion": "Empaque galleta", "Tiendas": True, "Supermercados": False, "Institucional": False, "Vida_Util_Dias": 30, "Und_Caja": 30, "Min_Cajas": 6, "Precio": 1500}
        return {"Linea": "Panes", "Presentacion": "Empaque panadería", "Tiendas": True, "Supermercados": True, "Institucional": False, "Vida_Util_Dias": 6, "Und_Caja": 20, "Min_Cajas": 6, "Precio": 3500}

    fichas = df_out.apply(buscar_ficha, axis=1)
    if "Sub linea" not in df_out.columns:
        df_out["Sub linea"] = [f["Linea"] for f in fichas]
    df_out["Linea_Oficial"] = [f["Linea"] for f in fichas]
    df_out["Presentacion"] = [f["Presentacion"] for f in fichas]
    df_out["Canales"] = [formatear_canales(f) for f in fichas]
    df_out["Vida_Util_Dias"] = [f["Vida_Util_Dias"] for f in fichas]
    df_out["Und_Caja_Cat"] = [f["Und_Caja"] for f in fichas]
    df_out["Min_Cajas_Cat"] = [f["Min_Cajas"] for f in fichas]
    df_out["Precio_Cat"] = [f["Precio"] for f in fichas]

    if "Sede_Regional" not in df_out.columns:
        muni_a_sede = {}
        for sede, info in LUGARES_DISTRIBUCION_SERVIPAN.items():
            muni_a_sede[normalizar_texto(sede)] = sede
            for m in info["municipios"]:
                base_m = normalizar_texto(m.split("(")[0])
                muni_a_sede.setdefault(base_m, sede)

        if "Zona" in df_out.columns:
            def mapear_sede(z):
                z_norm = normalizar_texto(z)
                if z_norm in muni_a_sede:
                    return muni_a_sede[z_norm]
                for m_norm, sede_name in muni_a_sede.items():
                    if len(m_norm) >= 5 and m_norm in z_norm:
                        return sede_name
                return str(z)

            df_out["Sede_Regional"] = df_out["Zona"].apply(mapear_sede)
        else:
            df_out["Sede_Regional"] = "Cesar - Valledupar"

    return df_out, no_mapeados


# ==============================================================================
# 6. GENERADOR DE BENCHMARK SINTÉTICO CALIBRADO (180 DÍAS)
# ==============================================================================
@st.cache_data
def generar_datos_ejemplo(dias: int = 180) -> pd.DataFrame:
    hoy = pd.Timestamp.today().normalize()
    fechas = pd.date_range(end=hoy, periods=dias, freq="D")
    rng = np.random.default_rng(2026)

    anios = sorted(list({f.year for f in fechas} | {hoy.year + 1}))
    cal = obtener_calendario_colombia(anios)
    df_cal = extraer_features_fecha(pd.Series(fechas), cal)

    sedes = list(LUGARES_DISTRIBUCION_SERVIPAN.keys())
    pesos_sedes = [LUGARES_DISTRIBUCION_SERVIPAN[s]["peso"] for s in sedes]

    registros = []
    for idx_f, f in enumerate(fechas):
        dow = f.day_of_week
        row_c = df_cal.iloc[idx_f]
        factor_dia = {0: 0.94, 1: 0.96, 2: 0.99, 3: 1.02, 4: 1.11, 5: 1.18, 6: 0.86}[dow]

        mult_cal = 1.0
        if row_c["es_quincena"] == 1:
            mult_cal *= 1.15
        if row_c["es_prima"] == 1:
            mult_cal *= 1.12
        if row_c["vispera_festivo"] == 1:
            mult_cal *= 1.14
        if row_c["es_festivo"] == 1:
            mult_cal *= 0.92
        if row_c["festival_vallenato"] == 1 or row_c["carnaval_caribe"] == 1:
            mult_cal *= 1.18

        for prod in CATALOGO_OFICIAL_SERVIPAN:
            perfil = prod["Perfil_Escolar"]
            mult_prod = mult_cal
            if row_c["vacaciones_colegios"] == 1:
                if perfil == "lonchera":
                    mult_prod *= 0.86
                elif perfil == "familiar":
                    mult_prod *= 1.09
            if perfil == "festivo" and (row_c["vispera_festivo"] == 1 or row_c["is_weekend"] == 1):
                mult_prod *= 1.12

            total_dia = max(12, int(round(prod["Base_Diaria"] * factor_dia * mult_prod * rng.normal(1.0, 0.045))))
            asignaciones = rng.multinomial(total_dia, pesos_sedes)
            for sede, cant in zip(sedes, asignaciones):
                if cant > 0:
                    muni = rng.choice(LUGARES_DISTRIBUCION_SERVIPAN[sede]["municipios"])
                    registros.append(
                        {
                            "Fecha": f,
                            "Ref": prod["Ref"],
                            "Desc item": prod["Producto"],
                            "Cant inv": int(cant),
                            "Zona": muni,
                            "Sede_Regional": sede,
                            "Sub linea": prod["Linea"],
                            "Valor": int(cant * prod["Precio"]),
                            "Razon social cliente factura": f"Punto {muni}",
                        }
                    )
    return pd.DataFrame(registros)


# ==============================================================================
# 7. MOTOR PREDICTIVO CUANTÍLICO + VALIDACIÓN ROLLING-ORIGIN + DECISIÓN NEWSVENDOR
# ==============================================================================
CUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
HORIZONTES = {"1-7": (1, 7), "8-14": (8, 14), "15-30": (15, 30)}


def _col_q(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def _nuevo_modelo(q: float, max_iter: int = 85) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_iter=max_iter,
        max_depth=4,
        learning_rate=0.09,
        l2_regularization=1.0,
        categorical_features=["Ref_ID", "Linea_ID"],
        random_state=42,
    )


def entrenar_cuantiles(train_df: pd.DataFrame, feature_cols: list[str], cuantiles=CUANTILES) -> dict:
    X, y = train_df[feature_cols], train_df["Cant inv"]
    return {q: _nuevo_modelo(q).fit(X, y) for q in cuantiles}


def _fila_features(serie: list[float], ids: tuple, c_row: pd.Series, cal_cols: list[str]) -> dict:
    ref_id, linea_id, vida = ids
    lag_1 = serie[-1]
    lag_7 = serie[-7] if len(serie) >= 7 else lag_1
    lag_14 = serie[-14] if len(serie) >= 14 else lag_7
    w7, w28 = serie[-7:], serie[-28:]
    fila = {
        "Ref_ID": ref_id,
        "Linea_ID": linea_id,
        "Vida_Util_Dias": vida,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "roll_mean_7": float(np.mean(w7)),
        "roll_mean_28": float(np.mean(w28)),
        "roll_std_7": float(np.std(w7, ddof=1)) if len(w7) >= 2 else 0.0,
    }
    fila.update({c: int(c_row[c]) for c in cal_cols})
    return fila


def pronostico_recursivo(
    modelos: dict,
    hist_series: dict,
    refs: list[str],
    meta_ids: dict,
    fechas,
    cal_lookup: pd.DataFrame,
    cal_cols: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Despliegue recursivo: realimenta P50 continuo (sin redondeo) y ordena cuantiles sin distorsión."""
    hist = {r: list(v) for r, v in hist_series.items()}
    qs = sorted(modelos)
    i50 = qs.index(0.50)
    filas = []
    for h, f in enumerate(fechas, start=1):
        c_row = cal_lookup.loc[f]
        X = pd.DataFrame([_fila_features(hist[r], meta_ids[r], c_row, cal_cols) for r in refs])[feature_cols]
        P = np.column_stack([np.maximum(0.0, modelos[q].predict(X)) for q in qs])
        P = np.sort(P, axis=1)
        for i, r in enumerate(refs):
            hist[r].append(float(P[i, i50]))
            filas.append(
                {
                    "Fecha": f,
                    "Ref": r,
                    "h": h,
                    "Evento_Calendario": str(c_row["Evento_Calendario"]),
                    **{_col_q(q): P[i, j] for j, q in enumerate(qs)},
                }
            )
    return pd.DataFrame(filas)


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum() * 100) if y.sum() > 0 else np.nan


def pinball(y, q_pred, q: float) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def razon_critica(precio: float, costo: float, salvamento: float = 0.0, penalizacion_faltante: float = 0.0) -> tuple[float, float, float]:
    cu = max(1.0, precio - costo + penalizacion_faltante)
    co = max(1.0, costo - salvamento)
    return cu / (cu + co), cu, co


def cuantil_demanda(fila: pd.Series, nivel: float, cuantiles=CUANTILES) -> float:
    xs = np.array(cuantiles)
    ys = np.array([fila[_col_q(q)] for q in cuantiles])
    return float(np.interp(nivel, xs, ys))


def a_cajas(q: float, emb: int, min_cajas: int) -> int:
    return int(max(np.ceil(max(q, 0.0) / emb), min_cajas)) if q > 0 else 0


def construir_panel_features(df_diario_in: pd.DataFrame, refs_activas: list[str], horizonte_dias: int):
    df_diario = df_diario_in.copy()
    df_diario["Fecha"] = pd.to_datetime(df_diario["Fecha"])
    ultima_fecha = df_diario["Fecha"].max()
    primera_fecha = df_diario["Fecha"].min()
    dias_historia = (ultima_fecha - primera_fecha).days + 1

    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=max(horizonte_dias, 30), freq="D")
    anios = sorted(list({d.year for d in [primera_fecha, ultima_fecha, fechas_futuras.max()]}))
    cal = obtener_calendario_colombia(anios)

    rango_completo = pd.date_range(start=primera_fecha, end=fechas_futuras.max(), freq="D")
    df_cal_full = extraer_features_fecha(pd.Series(rango_completo), cal)
    df_cal_full["Fecha"] = rango_completo
    df_cal_full["Evento_Calendario"] = df_cal_full.apply(describir_evento_calendario, axis=1)
    cal_lookup = df_cal_full.set_index("Fecha")

    df_diario = df_diario[df_diario["Ref"].astype(str).isin(refs_activas)].copy()
    meta_map = (
        df_diario[["Ref", "Desc item", "Linea_Oficial", "Presentacion", "Canales", "Vida_Util_Dias", "Precio_Cat"]]
        .drop_duplicates("Ref")
        .set_index("Ref")
        .to_dict("index")
    )

    refs_validas = list(meta_map.keys())
    ref_to_id = {r: idx for idx, r in enumerate(refs_validas)}
    lineas_unicas = sorted(list({m["Linea_Oficial"] for m in meta_map.values()}))
    linea_to_id = {l: idx for idx, l in enumerate(lineas_unicas)}
    meta_ids = {
        r: (ref_to_id[r], linea_to_id[meta_map[r]["Linea_Oficial"]], int(meta_map[r]["Vida_Util_Dias"]))
        for r in refs_validas
    }

    grid = pd.MultiIndex.from_product(
        [refs_validas, pd.date_range(primera_fecha, ultima_fecha, freq="D")],
        names=["Ref", "Fecha"],
    ).to_frame(index=False)
    agg_ventas = df_diario.groupby(["Ref", "Fecha"], as_index=False)["Cant inv"].sum()
    panel = pd.merge(grid, agg_ventas, on=["Ref", "Fecha"], how="left")
    panel["Cant inv"] = panel["Cant inv"].fillna(0.0)
    panel = panel.sort_values(["Ref", "Fecha"]).reset_index(drop=True)

    panel["Ref_ID"] = panel["Ref"].map(ref_to_id)
    panel["Linea_ID"] = panel["Ref"].map(lambda r: linea_to_id[meta_map[r]["Linea_Oficial"]])
    panel["Vida_Util_Dias"] = panel["Ref"].map(lambda r: int(meta_map[r]["Vida_Util_Dias"]))

    cal_cols = [c for c in df_cal_full.columns if c not in ("Fecha", "Evento_Calendario")]
    panel = panel.merge(df_cal_full[["Fecha"] + cal_cols], on="Fecha", how="left")

    panel["lag_1"] = panel.groupby("Ref")["Cant inv"].shift(1)
    panel["lag_7"] = panel.groupby("Ref")["Cant inv"].shift(7)
    panel["lag_14"] = panel.groupby("Ref")["Cant inv"].shift(14)
    panel["roll_mean_7"] = panel.groupby("Ref")["Cant inv"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    panel["roll_mean_28"] = panel.groupby("Ref")["Cant inv"].transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    panel["roll_std_7"] = panel.groupby("Ref")["Cant inv"].transform(lambda s: s.shift(1).rolling(7, min_periods=2).std()).fillna(0.0)

    base_cols = [
        "Ref_ID", "Linea_ID", "Vida_Util_Dias",
        "lag_1", "lag_7", "lag_14", "roll_mean_7", "roll_mean_28", "roll_std_7",
    ]
    feature_cols = base_cols + cal_cols
    return panel, meta_map, meta_ids, refs_validas, cal_lookup, cal_cols, base_cols, feature_cols, ultima_fecha, dias_historia


def backtest_rolling_origin(
    panel: pd.DataFrame,
    feature_cols: list[str],
    cal_cols: list[str],
    refs: list[str],
    meta_ids: dict,
    cal_lookup: pd.DataFrame,
    n_origenes: int = 4,
    paso: int = 7,
    horizonte: int = 30,
    min_train: int = 45,
    cuantiles=CUANTILES,
) -> pd.DataFrame:
    """Backtesting recursivo con origen móvil sin recortes ni valores inventados."""
    panel = panel.sort_values(["Ref", "Fecha"])
    ultima, primera = panel["Fecha"].max(), panel["Fecha"].min()
    lookup = panel.set_index(["Ref", "Fecha"])["Cant inv"].to_dict()

    origenes = [ultima - pd.Timedelta(days=horizonte + k * paso) for k in range(n_origenes)]
    origenes = [o for o in origenes if (o - primera).days + 1 >= min_train]

    resultados = []
    for T in sorted(origenes):
        train = panel[panel["Fecha"] <= T].dropna(subset=["lag_7"])
        if len(train) < 30:
            continue
        modelos = entrenar_cuantiles(train, feature_cols, cuantiles=cuantiles)
        hist = {r: panel[(panel["Ref"] == r) & (panel["Fecha"] <= T)]["Cant inv"].tolist() for r in refs}
        fechas = pd.date_range(T + pd.Timedelta(days=1), periods=horizonte, freq="D")
        pred = pronostico_recursivo(modelos, hist, refs, meta_ids, fechas, cal_lookup, cal_cols, feature_cols)
        pred["origen"] = T
        pred["real"] = [lookup.get((r, f), np.nan) for r, f in zip(pred["Ref"], pred["Fecha"])]
        sem = np.ceil(pred["h"] / 7).astype(int)
        pred["naive_estacional"] = [
            lookup.get((r, f - pd.Timedelta(days=7 * s)), np.nan)
            for r, f, s in zip(pred["Ref"], pred["Fecha"], sem)
        ]
        resultados.append(pred)

    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True).dropna(subset=["real"])


@st.cache_data(show_spinner=False)
def ejecutar_motor_ml_servipan(
    df_diario_in: pd.DataFrame,
    df_param_in: pd.DataFrame,
    horizonte_dias: int,
    postura_riesgo: str,
    costo_var_pct: float,
    penalizacion_quiebre_pct: float,
):
    """Entrena el modelo cuantílico, ejecuta validación recursiva sin recortes y aplica decisión Newsvendor."""
    refs_activas = df_param_in["Ref"].astype(str).tolist()
    (
        panel, meta_map, meta_ids, refs_validas,
        cal_lookup, cal_cols, _, feature_cols, ultima_fecha, dias_historia
    ) = construir_panel_features(df_diario_in, refs_activas, horizonte_dias)

    fechas_futuras = pd.date_range(start=ultima_fecha + pd.Timedelta(days=1), periods=horizonte_dias, freq="D")
    usar_ml = SKLEARN_DISPONIBLE and (dias_historia >= 21) and (len(panel) >= 60)

    wape_7d = None
    wape_naive_7d = None
    cobertura_7d = None
    pinball_50_7d = None
    motor_nombre = "HistGradientBoosting Cuantílico (P10/P25/P50/P75/P90)"

    if usar_ml:
        train_df = panel.dropna(subset=["lag_7"]).copy()

        # 1. Validación fuera de muestra recursiva (últimos 7 días, sin clip ni defaults falsos)
        bt_rapido = backtest_rolling_origin(
            panel=panel,
            feature_cols=feature_cols,
            cal_cols=cal_cols,
            refs=refs_validas,
            meta_ids=meta_ids,
            cal_lookup=cal_lookup,
            n_origenes=2,
            paso=7,
            horizonte=7,
            min_train=21,
            cuantiles=(0.10, 0.50, 0.90),
        )
        if not bt_rapido.empty and bt_rapido["real"].sum() > 0:
            y_bt = bt_rapido["real"].to_numpy()
            wape_7d = round(wape(y_bt, bt_rapido["q50"]), 2)
            wape_naive_7d = round(wape(y_bt, bt_rapido["naive_estacional"].fillna(bt_rapido["q50"])), 2)
            cobertura_7d = round(float(((y_bt >= bt_rapido["q10"]) & (y_bt <= bt_rapido["q90"])).mean() * 100.0), 1)
            pinball_50_7d = round(pinball(y_bt, bt_rapido["q50"], 0.50), 2)

        # 2. Entrenamiento final con 100% del historial para los 5 cuantiles
        modelos = entrenar_cuantiles(train_df, feature_cols, cuantiles=CUANTILES)
        hist_series = {
            r: panel[panel["Ref"] == r].sort_values("Fecha")["Cant inv"].tolist()
            for r in refs_validas
        }
        df_ml_raw = pronostico_recursivo(
            modelos, hist_series, refs_validas, meta_ids,
            fechas_futuras, cal_lookup, cal_cols, feature_cols
        )
    else:
        motor_nombre = "Descomposición Estacional (Respaldo < 21 días)"
        pred_registros = []
        for r in refs_validas:
            sub = panel[panel["Ref"] == r].copy()
            sub["dow"] = sub["Fecha"].dt.day_of_week
            prom_d = sub.groupby("dow")["Cant inv"].mean().to_dict()
            std_d = sub.groupby("dow")["Cant inv"].std().fillna(0).to_dict()
            media_g = max(1.0, float(sub["Cant inv"].mean()))
            for f in fechas_futuras:
                c_row = cal_lookup.loc[f]
                base = prom_d.get(f.day_of_week, media_g)
                mult = 1.12 if (c_row["es_quincena"] == 1 or c_row["vispera_festivo"] == 1) else 1.0
                p50 = max(1.0, base * mult)
                desv = std_d.get(f.day_of_week, p50 * 0.12)
                pred_registros.append(
                    {
                        "Fecha": f,
                        "Ref": r,
                        "Evento_Calendario": str(c_row["Evento_Calendario"]),
                        "q10": max(0.0, p50 - 1.28 * desv),
                        "q25": max(0.0, p50 - 0.67 * desv),
                        "q50": p50,
                        "q75": p50 + 0.67 * desv,
                        "q90": p50 + 1.28 * desv,
                    }
                )
        df_ml_raw = pd.DataFrame(pred_registros)

    # 3. Decisión operativa mediante Cuantil Crítico Newsvendor y Postura de Riesgo
    dias_nombre = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    param_map = df_param_in.set_index("Ref").to_dict("index")

    proyecciones = []
    tabla_manana = []

    for r in refs_validas:
        info_m = meta_map[r]
        p_cfg = param_map.get(r, {"Unidades por Caja": 20, "Mínimo a Producir (Cajas)": 5})
        emb = max(1, int(p_cfg["Unidades por Caja"]))
        min_cajas = max(1, int(p_cfg["Mínimo a Producir (Cajas)"]))
        vida_util = int(info_m["Vida_Util_Dias"])
        precio_u = float(info_m.get("Precio_Cat", 3500))

        costo_u = precio_u * costo_var_pct
        salvamento_u = 0.0 if vida_util <= 10 else precio_u * 0.15
        penal_u = precio_u * penalizacion_quiebre_pct
        cr_opt, cu_val, co_val = razon_critica(precio_u, costo_u, salvamento_u, penal_u)

        sub_ml = df_ml_raw[df_ml_raw["Ref"] == r].sort_values("Fecha")
        sub_hist = panel[panel["Ref"] == r]
        media_hist = max(1.0, float(sub_hist["Cant inv"].mean()))

        for idx_step, (_, row_ml) in enumerate(sub_ml.iterrows()):
            f = row_ml["Fecha"]
            p50 = float(row_ml["q50"])
            p10 = float(row_ml["q10"])
            p90 = float(row_ml["q90"])
            evento_cal = str(row_ml["Evento_Calendario"])

            if postura_riesgo == "Conservador (P50)":
                q_objetivo = p50
                nivel_srv = 0.50
            elif postura_riesgo == "Expansivo / Captura (P90)":
                q_objetivo = p90
                nivel_srv = 0.90
            else:
                q_objetivo = cuantil_demanda(row_ml, cr_opt, cuantiles=CUANTILES)
                nivel_srv = cr_opt

            cajas_sug = a_cajas(q_objetivo, emb, min_cajas)
            und_producir = cajas_sug * emb

            fecha_sem_pasada = f - pd.Timedelta(days=7)
            venta_sem_pasada = float(sub_hist[sub_hist["Fecha"] == fecha_sem_pasada]["Cant inv"].sum())
            if venta_sem_pasada <= 0:
                venta_sem_pasada = media_hist
            var_pct = ((p50 - venta_sem_pasada) / venta_sem_pasada * 100.0) if venta_sem_pasada > 0 else 0.0

            proyecciones.append(
                {
                    "Fecha": f,
                    "Fecha Proyectada": f.strftime("%Y-%m-%d"),
                    "Día": dias_nombre[f.day_of_week],
                    "Evento Calendario": evento_cal,
                    "Ref": r,
                    "Producto": info_m["Desc item"],
                    "Línea": info_m["Linea_Oficial"],
                    "Pronóstico P50": int(round(p50)),
                    "P10": int(round(p10)),
                    "P90": int(round(p90)),
                    "Cuantil Objetivo": round(q_objetivo, 1),
                    "CR_Optimo": round(nivel_srv, 3),
                    "Producir (Un)": und_producir,
                    "Sugerido Producir (Cajas)": cajas_sug,
                    "Var %": round(var_pct, 1),
                }
            )

            if idx_step == 0:
                if var_pct >= 8.0 or (vida_util <= 5 and p50 >= media_hist * 1.05):
                    estado = "Riesgo de quiebre"
                elif var_pct >= 3.5 or nivel_srv >= 0.65:
                    estado = "Subir producción"
                else:
                    estado = "Estable"

                tabla_manana.append(
                    {
                        "Ref": r,
                        "Producto": info_m["Desc item"],
                        "Línea": info_m["Linea_Oficial"],
                        "Presentación": info_m["Presentacion"],
                        "Canales": info_m["Canales"],
                        "Vida Útil": f"{vida_util} días",
                        "Vida_Util_num": vida_util,
                        "Evento Mañana": evento_cal,
                        "Pronóstico_num": int(round(p50)),
                        "Pronóstico": f"{int(round(p50)):,} und".replace(",", "."),
                        "P10": int(round(p10)),
                        "P90": int(round(p90)),
                        "Rango P10–P90": f"{int(round(p10)):,} – {int(round(p90)):,}".replace(",", "."),
                        "Nivel_Servicio_Opt": round(nivel_srv * 100, 1),
                        "Producir": und_producir,
                        "Cajas": cajas_sug,
                        "Estado": estado,
                        "Var %": round(var_pct, 1),
                    }
                )

    df_proy = pd.DataFrame(proyecciones)
    df_manana = pd.DataFrame(tabla_manana)
    if not df_manana.empty:
        prioridad = {"Riesgo de quiebre": 0, "Subir producción": 1, "Estable": 2}
        df_manana["_prio"] = df_manana["Estado"].map(prioridad).fillna(2)
        df_manana = df_manana.sort_values(["_prio", "Pronóstico_num"], ascending=[True, False]).drop(columns=["_prio"])

    metricas_modelo = {
        "wape_7d": wape_7d,
        "wape_naive_7d": wape_naive_7d,
        "cobertura_7d": cobertura_7d,
        "pinball_50_7d": pinball_50_7d,
        "motor_nombre": motor_nombre,
        "dias_historia": dias_historia,
    }
    return df_proy, df_manana, ultima_fecha, metricas_modelo


@st.cache_data(show_spinner=False)
def ejecutar_auditoria_rolling_y_ablacion(
    df_diario_in: pd.DataFrame,
    df_param_in: pd.DataFrame,
    costo_var_pct: float,
    penalizacion_quiebre_pct: float,
):
    """Ejecuta backtesting completo (1-7, 8-14, 15-30 días), ablación sin calendario y evaluación económica de políticas."""
    refs_activas = df_param_in["Ref"].astype(str).tolist()
    (
        panel, meta_map, meta_ids, refs_validas,
        cal_lookup, cal_cols, base_cols, feature_cols, _, _
    ) = construir_panel_features(df_diario_in, refs_activas, 30)

    bt_full = backtest_rolling_origin(panel, feature_cols, cal_cols, refs_validas, meta_ids, cal_lookup, n_origenes=4, paso=7, horizonte=30, min_train=45)
    bt_nocal = backtest_rolling_origin(panel, base_cols, [], refs_validas, meta_ids, cal_lookup, n_origenes=4, paso=7, horizonte=30, min_train=45)

    if bt_full.empty:
        return pd.DataFrame(), pd.DataFrame()

    filas_h = []
    for nombre, (a, b) in HORIZONTES.items():
        s1 = bt_full[bt_full["h"].between(a, b)]
        s0 = bt_nocal[bt_nocal["h"].between(a, b)] if not bt_nocal.empty else pd.DataFrame()
        if s1.empty:
            continue
        y = s1["real"].to_numpy()
        w_mod = wape(y, s1["q50"])
        w_nai = wape(y, s1["naive_estacional"].fillna(s1["q50"]))
        w_nc = wape(y, s0["q50"]) if not s0.empty else np.nan
        filas_h.append(
            {
                "Horizonte (días)": nombre,
                "Observaciones (n)": len(s1),
                "WAPE Modelo Full %": round(w_mod, 2),
                "WAPE Sin Calendario %": round(w_nc, 2),
                "WAPE Naive Estacional %": round(w_nai, 2),
                "Pinball q10": round(pinball(y, s1["q10"], 0.10), 2),
                "Pinball q50": round(pinball(y, s1["q50"], 0.50), 2),
                "Pinball q90": round(pinball(y, s1["q90"], 0.90), 2),
                "Cobertura P10–P90 % (objetivo 80 %)": round(float(((y >= s1["q10"]) & (y <= s1["q90"])).mean() * 100), 2),
            }
        )
    df_resumen_bt = pd.DataFrame(filas_h)

    # Evaluación de políticas sobre el backtest
    param_map = df_param_in.set_index("Ref").to_dict("index")
    params_costos = {}
    empaque = {}
    vida_util = {}
    for r in refs_validas:
        pr = float(meta_map[r].get("Precio_Cat", 3500))
        vu = int(meta_map[r]["Vida_Util_Dias"])
        co_u = pr * costo_var_pct
        salv = 0.0 if vu <= 10 else pr * 0.15
        pen = pr * penalizacion_quiebre_pct
        cr, cu, co = razon_critica(pr, co_u, salv, pen)
        params_costos[r] = {"cr": cr, "cu": cu, "co": co}
        p_cfg = param_map.get(r, {"Unidades por Caja": 20, "Mínimo a Producir (Cajas)": 5})
        empaque[r] = (max(1, int(p_cfg["Unidades por Caja"])), max(1, int(p_cfg["Mínimo a Producir (Cajas)"])))
        vida_util[r] = vu

    def buffer_fijo(v):
        return 1.05 if v <= 6 else (1.08 if v <= 12 else 1.12)

    politicas = {
        "Newsvendor (Cuantil Crítico CR*)": lambda f, r: cuantil_demanda(f, params_costos[r]["cr"]),
        "P50 Puro (Mediana)": lambda f, r: f["q50"],
        "P50 + Buffer Fijo (5 % / 8 % / 12 %)": lambda f, r: f["q50"] * buffer_fijo(vida_util[r]),
        "P90 Expansivo (Captura de Mercado)": lambda f, r: f["q90"],
    }

    filas_pol = []
    for nombre_p, fn in politicas.items():
        costo = falt = sobr = dem = prod = 0.0
        for _, f in bt_full.iterrows():
            r = f["Ref"]
            emb, minc = empaque[r]
            Q = a_cajas(fn(f, r), emb, minc) * emb
            D = float(f["real"])
            cu, co = params_costos[r]["cu"], params_costos[r]["co"]
            costo += cu * max(D - Q, 0.0) + co * max(Q - D, 0.0)
            falt += max(D - Q, 0.0)
            sobr += max(Q - D, 0.0)
            dem += D
            prod += Q
        filas_pol.append(
            {
                "Política de Decisión": nombre_p,
                "Costo Total Desbalance (COP)": int(round(costo)),
                "Nivel de Servicio (%)": round((1.0 - falt / dem) * 100.0, 2) if dem > 0 else 0.0,
                "Excedente / Merma Potencial (%)": round((sobr / prod) * 100.0, 2) if prod > 0 else 0.0,
            }
        )
    df_politicas = pd.DataFrame(filas_pol).sort_values("Costo Total Desbalance (COP)").reset_index(drop=True)
    return df_resumen_bt, df_politicas


def construir_grafico_tendencia(
    df_diario_total: pd.DataFrame,
    df_proy_detalle: pd.DataFrame,
    ultima_fecha: pd.Timestamp,
    horizonte_dias: int,
) -> go.Figure:
    """Construye gráfico agregado con intervalo de confianza calibrado por agregación de varianzas (sin sumar cuantiles)."""
    fecha_inicio_hist = ultima_fecha - pd.Timedelta(days=14)
    hist = df_diario_total[df_diario_total["Fecha"] >= fecha_inicio_hist].sort_values("Fecha").copy()
    ultimo_valor_real = float(hist["Cant inv"].iloc[-1]) if not hist.empty else 0.0

    # Agregación de varianza por fecha: sigma_i = (P90_i - P10_i) / (2 * 1.28155)
    df_agg = df_proy_detalle.copy()
    df_agg["var_i"] = ((df_agg["P90"] - df_agg["P10"]) / (2.0 * 1.28155)) ** 2
    agg_f = df_agg.groupby("Fecha", as_index=False).agg(
        p50_sum=("Pronóstico P50", "sum"),
        var_sum=("var_i", "sum"),
    ).sort_values("Fecha")
    agg_f["sigma_agg"] = np.sqrt(agg_f["var_sum"])
    agg_f["p10_cal"] = np.maximum(0.0, agg_f["p50_sum"] - 1.28155 * agg_f["sigma_agg"])
    agg_f["p90_cal"] = agg_f["p50_sum"] + 1.28155 * agg_f["sigma_agg"]

    # El punto inicial t=Hoy parte exactamente del último valor observado (incertidumbre cero en t=0)
    fechas_fut = [ultima_fecha] + agg_f["Fecha"].tolist()
    p50_fut = [ultimo_valor_real] + agg_f["p50_sum"].tolist()
    p10_fut = [ultimo_valor_real] + agg_f["p10_cal"].tolist()
    p90_fut = [ultimo_valor_real] + agg_f["p90_cal"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fechas_fut,
            y=p90_fut,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fechas_fut,
            y=p10_fut,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(245, 214, 107, 0.34)",
            name="Banda P10–P90 (varianza agregada)",
            hovertemplate="Banda P10–P90: %{y:,.0f} und<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["Fecha"],
            y=hist["Cant inv"],
            mode="lines",
            name="Ventas reales",
            line=dict(color="#6B2E12", width=2.5, shape="spline", smoothing=0.45),
            hovertemplate="%{x|%d %b}: <b>%{y:,.0f} und</b><extra>Ventas reales</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fechas_fut,
            y=p50_fut,
            mode="lines",
            name="Pronóstico P50",
            line=dict(color="#D97724", width=2.5, dash="dash", shape="spline", smoothing=0.45),
            hovertemplate="%{x|%d %b}: <b>%{y:,.0f} und</b><extra>Pronóstico P50</extra>",
        )
    )
    fig.add_vline(
        x=ultima_fecha,
        line_width=1.4,
        line_dash="dot",
        line_color="#9A5B25",
    )

    fecha_fin_fut = agg_f["Fecha"].max() if not agg_f.empty else ultima_fecha
    fig.update_layout(
        height=238,
        margin=dict(l=8, r=12, t=12, b=26),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            tickmode="array",
            tickvals=[fecha_inicio_hist, ultima_fecha, fecha_fin_fut],
            ticktext=["−14 días", "Hoy", f"+{horizonte_dias} días"],
            tickfont=dict(size=11, color="#7C6350"),
            showgrid=False,
            zeroline=False,
            linecolor="#EADBC8",
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=True,
            gridcolor="#F5EDE3",
            zeroline=False,
        ),
    )
    return fig


# ==============================================================================
# 8. BARRA LATERAL (EMBLEMA OVALADO SERVIPAN + NAVEGACIÓN + POSTURA DE RIESGO)
# ==============================================================================
if "seccion_activa" not in st.session_state:
    st.session_state.seccion_activa = "Resumen"

opciones_menu = [
    "Resumen",
    "Pronóstico",
    "Productos",
    "Sedes",
    "Escenarios",
    "Inventario",
    "Alertas",
    "Configuración",
]

with st.sidebar:
    st.markdown(
        '<div class="servipan-emblem-wrap">'
        '<div class="servipan-oval"><span>Servipan</span></div>'
        '<div class="servipan-sub-tag">Planta<br>Valledupar</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    for opcion in opciones_menu:
        es_activa = st.session_state.seccion_activa == opcion
        if st.button(
            opcion,
            key=f"nav_btn_{opcion}",
            type="primary" if es_activa else "secondary",
            width="stretch",
        ):
            st.session_state.seccion_activa = opcion
            st.rerun()

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    with st.expander("🎯 Postura de Riesgo (Newsvendor)", expanded=True):
        postura_riesgo = st.selectbox(
            "Política de abastecimiento",
            [
                "Óptimo Newsvendor (CR*)",
                "Conservador (P50)",
                "Expansivo / Captura (P90)",
            ],
            index=0,
        )
        costo_var_pct = st.slider("Costo variable (% precio)", 40, 80, 60, 5) / 100.0
        penalizacion_quiebre_pct = st.slider("Penalización quiebre (% precio)", 0, 50, 15, 5) / 100.0

    with st.expander("📥 Cargar Datos Excel", expanded=False):
        archivo = st.file_uploader(
            "Subir historial (.xlsx)",
            type=["xlsx"],
            help="Columnas requeridas: Fecha, Ref, Desc item, Cant inv. Nota: anonimizar razón social antes de subir a la nube.",
        )

seccion_activa = st.session_state.seccion_activa

# ==============================================================================
# 9. CARGA Y ENRIQUECIMIENTO DE DATOS
# ==============================================================================
es_ejemplo = True
df_raw = None

if archivo is not None:
    try:
        df_temp = pd.read_excel(archivo)
        columnas_base = ["Fecha", "Ref", "Desc item", "Cant inv"]
        if all(col in df_temp.columns for col in columnas_base):
            df_temp["Fecha"] = pd.to_datetime(df_temp["Fecha"])
            df_temp["Ref"] = df_temp["Ref"].astype(str)
            df_raw = df_temp
            es_ejemplo = False
        else:
            st.sidebar.error(f"Faltan columnas requeridas: {columnas_base}")
    except Exception as exc:
        st.sidebar.error(f"Error leyendo Excel: {exc}")

if df_raw is None:
    df_raw = generar_datos_ejemplo(dias=180)
    es_ejemplo = True

df_enriquecido, n_no_mapeados = enriquecer_dataframe_con_maestros(df_raw)

col_cliente = None
for posible_col in ["Razon social cliente factura", "Razón social cliente factura"]:
    if posible_col in df_enriquecido.columns:
        col_cliente = posible_col
        break

with st.sidebar:
    with st.expander("⚙️ Parámetros por Producto", expanded=False):
        productos_unicos = (
            df_enriquecido[["Ref", "Desc item", "Linea_Oficial", "Und_Caja_Cat", "Min_Cajas_Cat"]]
            .drop_duplicates("Ref")
        )
        parametros_lista = []
        for _, row in productos_unicos.iterrows():
            ref = str(row["Ref"])
            nombre = str(row["Desc item"])
            linea = str(row["Linea_Oficial"])
            st.markdown(f"**{nombre}** (`{linea}`)")
            frec = st.number_input(
                "Frecuencia cargue (días/sem)",
                min_value=1,
                max_value=7,
                value=6 if linea in ["Panes", "Tajados"] else 3,
                key=f"frec_{ref}",
            )
            emb = st.number_input(
                "Unidades por caja",
                min_value=1,
                value=int(row["Und_Caja_Cat"]),
                key=f"emb_{ref}",
            )
            min_lot = st.number_input(
                "Mínimo por lote (cajas)",
                min_value=1,
                value=int(row["Min_Cajas_Cat"]),
                key=f"min_{ref}",
            )
            parametros_lista.append(
                {
                    "Ref": ref,
                    "Frecuencia de cargue (Días/Semana)": frec,
                    "Unidades por Caja": emb,
                    "Mínimo a Producir (Cajas)": min_lot,
                }
            )
            st.divider()

    st.markdown(
        f'<div class="sidebar-doodle">{SVG_ESPIGA_TRIGO}{SVG_ESPIGA_TRIGO}</div>',
        unsafe_allow_html=True,
    )

df_param = pd.DataFrame(parametros_lista)

# ==============================================================================
# 10. CABECERA PRINCIPAL Y FILTROS SUPERIORES
# ==============================================================================
col_head_izq, col_head_linea, col_head_sede, col_head_hor = st.columns([5.1, 1.5, 2.0, 1.4])

badge_fuente = (
    '<span class="data-badge-demo">Benchmark sintético calibrado (180 días)</span>'
    if es_ejemplo
    else '<span class="data-badge-real">Datos reales cargados</span>'
)

subtitulo_fuente = (
    f"Inteligencia de mercado con 24 variables (calendario sociocultural + rezagos SKU) {badge_fuente}"
)

titulos_seccion = {
    "Resumen": ("Resumen de inteligencia de mercado", subtitulo_fuente),
    "Pronóstico": ("Pronóstico detallado por SKU", "Curvas cuantílicas diarias con detección de festivos, quincenas y calendario académico"),
    "Productos": ("Catálogo oficial y líneas Servipan", "Fichas técnicas de 23 referencias, presentaciones y canales de distribución"),
    "Sedes": ("Cobertura regional Costa Caribe", "6 corredores regionales y 129 localidades atendidas"),
    "Escenarios": ("Simulador de estrategias y riesgos", "Evaluación controlada de choques de demanda y posturas Newsvendor"),
    "Inventario": ("Control de frescura, cajas y lotes", "Conversión del cuantil objetivo a cajas comerciales según vida útil"),
    "Alertas": ("Centro de alertas tempranas", "Monitoreo dinámico de riesgo de quiebre, calendario y calibración cuantílica"),
    "Configuración": ("Auditoría ML, ablación y Newsvendor", "Backtesting con origen móvil, Pinball Loss y comparación de políticas"),
}

titulo_actual, subtitulo_actual = titulos_seccion[seccion_activa]

with col_head_izq:
    st.markdown(
        f'<div><h1 class="main-title">{titulo_actual}</h1>'
        f'<p class="main-subtitle">{subtitulo_actual}</p></div>',
        unsafe_allow_html=True,
    )

with col_head_linea:
    st.markdown('<div class="filter-label">Línea</div>', unsafe_allow_html=True)
    lineas_opciones = ["Todas las líneas"] + sorted(df_enriquecido["Linea_Oficial"].dropna().unique().tolist())
    linea_sel = st.selectbox("Línea", lineas_opciones, index=0, label_visibility="collapsed")

with col_head_sede:
    st.markdown('<div class="filter-label">Corredor Regional</div>', unsafe_allow_html=True)
    sedes_opciones = ["Todas las sedes"] + sorted(df_enriquecido["Sede_Regional"].dropna().unique().tolist())
    sede_sel = st.selectbox("Corredor Regional", sedes_opciones, index=0, label_visibility="collapsed")

with col_head_hor:
    st.markdown('<div class="filter-label">Horizonte</div>', unsafe_allow_html=True)
    horizonte_sel = st.selectbox(
        "Horizonte",
        ["7 días", "14 días", "30 días"],
        index=0,
        label_visibility="collapsed",
    )

horizonte_dias = int(horizonte_sel.split()[0])

df_filtrado = df_enriquecido.copy()
if linea_sel != "Todas las líneas":
    df_filtrado = df_filtrado[df_filtrado["Linea_Oficial"] == linea_sel]
if sede_sel != "Todas las sedes":
    df_filtrado = df_filtrado[df_filtrado["Sede_Regional"] == sede_sel]

if df_filtrado.empty:
    st.warning("No existen registros para la combinación de línea y corredor regional seleccionada.")
    st.stop()

if n_no_mapeados > 0:
    st.info(f"Aviso de calidad de datos: {n_no_mapeados} registros se asignaron mediante reglas heurísticas de línea por no coincidir exactamente con el código Ref del catálogo.")

df_diario_ml = df_filtrado.groupby(
    ["Fecha", "Ref", "Desc item", "Linea_Oficial", "Presentacion", "Canales", "Vida_Util_Dias", "Precio_Cat"],
    as_index=False,
)["Cant inv"].sum()

df_proy, df_manana, ultima_fecha, metricas_modelo = ejecutar_motor_ml_servipan(
    df_diario_ml,
    df_param,
    horizonte_dias=horizonte_dias,
    postura_riesgo=postura_riesgo,
    costo_var_pct=costo_var_pct,
    penalizacion_quiebre_pct=penalizacion_quiebre_pct,
)

eventos_proximos = []
if not df_proy.empty:
    proximos_7 = df_proy[df_proy["Fecha"] <= (ultima_fecha + pd.Timedelta(days=7))][["Fecha Proyectada", "Día", "Evento Calendario"]].drop_duplicates()
    for _, ev_r in proximos_7.iterrows():
        if ev_r["Evento Calendario"] != "Día hábil / regular":
            eventos_proximos.append(f"{ev_r['Día']} ({ev_r['Evento Calendario']})")

# ==============================================================================
# 11. VISTA PRINCIPAL: "RESUMEN"
# ==============================================================================
if seccion_activa == "Resumen":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    demanda_manana = int(df_manana["Pronóstico_num"].sum()) if not df_manana.empty else 0
    orden_manana = int(df_manana["Producir"].sum()) if not df_manana.empty else 0
    var_prom = float(df_manana["Var %"].mean()) if not df_manana.empty else 0.0
    signo = "+" if var_prom >= 0 else ""
    kpi_demanda_cls = "kpi-sub-green" if var_prom >= 0 else "kpi-sub-amber"
    n_riesgo = int((df_manana["Estado"] != "Estable").sum()) if not df_manana.empty else 0

    wape_val = metricas_modelo["wape_7d"]
    wape_nai = metricas_modelo["wape_naive_7d"]
    cov_val = metricas_modelo["cobertura_7d"]

    kpi_demanda = f"{demanda_manana:,} und".replace(",", ".")
    kpi_demanda_sub = f"{signo}{var_prom:.1f} % vs. mismo día semana pasada"

    if wape_val is not None:
        kpi_wape = f"{wape_val:.2f} %".replace(".", ",")
        mejora_nai = ((1.0 - wape_val / wape_nai) * 100.0) if (wape_nai and wape_nai > 0) else 0.0
        kpi_wape_sub = f"Naive estacional: {wape_nai:.2f} % ({mejora_nai:+.1f} % mejora)".replace(".", ",")
    else:
        kpi_wape = "N/D"
        kpi_wape_sub = "Historial insuficiente para backtest"

    kpi_riesgo = f"{n_riesgo} SKUs"
    kpi_riesgo_sub = f"Orden total: {orden_manana:,} und ({postura_riesgo.split()[0]})".replace(",", ".")

    if cov_val is not None:
        kpi_cov = f"{cov_val:.1f} %".replace(".", ",")
        kpi_cov_sub = f"Objetivo nominal P10–P90: 80,0 % · Pinball: {metricas_modelo['pinball_50_7d']}"
    else:
        kpi_cov = "N/D"
        kpi_cov_sub = "Requiere ≥ 28 días de historia"

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f'<div class="kpi-card kpi-top-cacao">'
            f'<div class="kpi-watermark-icon">{SVG_PAN_ARTESANAL}</div>'
            f'<div class="kpi-title">Demanda esperada mañana (P50)</div>'
            f'<div class="kpi-value">{kpi_demanda}</div>'
            f'<div class="{kpi_demanda_cls}">{kpi_demanda_sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            f'<div class="kpi-card kpi-top-dorado">'
            f'<div class="kpi-watermark-icon">{SVG_ESPIGA_TRIGO}</div>'
            f'<div class="kpi-title">Error WAPE recursivo (7 días)</div>'
            f'<div class="kpi-value">{kpi_wape}</div>'
            f'<div class="kpi-sub-neutral">{kpi_wape_sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f'<div class="kpi-card kpi-top-queso">'
            f'<div class="kpi-watermark-icon">{SVG_PAN_ARTESANAL}</div>'
            f'<div class="kpi-title">Alertas de tracción / quiebre</div>'
            f'<div class="kpi-value">{kpi_riesgo}</div>'
            f'<div class="kpi-sub-amber">{kpi_riesgo_sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f'<div class="kpi-card kpi-top-salvia">'
            f'<div class="kpi-watermark-icon">{SVG_ESPIGA_TRIGO}</div>'
            f'<div class="kpi-title">Cobertura empírica P10–P90</div>'
            f'<div class="kpi-value">{kpi_cov}</div>'
            f'<div class="kpi-sub-green">{kpi_cov_sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- FILA 2: GRÁFICO PRINCIPAL (70%) + PANEL DE ALERTAS DINÁMICAS (30%) ---
    col_chart, col_alerts = st.columns([6.8, 3.2])

    with col_chart:
        with st.container(border=True):
            st.markdown(
                '<div class="card-header-flex">'
                '<div class="card-title">Demanda observada y proyección cuantílica agregada</div>'
                '<div class="chart-legend">'
                '<span class="legend-item"><span class="legend-line-solid"></span>Observada</span>'
                '<span class="legend-item"><span class="legend-line-forecast"></span>Mediana P50</span>'
                '<span class="legend-item"><span class="legend-band"></span>Banda P10–P90 (varianza)</span>'
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            df_diario_tot = df_filtrado.groupby("Fecha", as_index=False)["Cant inv"].sum()
            fig_main = construir_grafico_tendencia(
                df_diario_tot,
                df_proy,
                ultima_fecha=ultima_fecha,
                horizonte_dias=horizonte_dias,
            )
            st.plotly_chart(fig_main, width="stretch", config={"displayModeBar": False})

    with col_alerts:
        if not df_manana.empty:
            top_row = df_manana.iloc[0]
            prod_alerta_1 = str(top_row["Producto"])
            var_top = float(top_row["Var %"])
            vu_top = int(top_row["Vida_Util_num"])
            desc_alerta_1 = (
                f"Variación proyectada {var_top:+.1f} % vs. semana previa · Vida útil {vu_top} días · "
                f"Orden sugerida: {int(top_row['Cajas'])} cajas."
            )
        else:
            prod_alerta_1 = "Sin alertas activas"
            desc_alerta_1 = "No hay referencias pendientes de ajuste."

        texto_calendario = (
            f"Próximos hitos (7 días): {', '.join(eventos_proximos[:2])}."
            if eventos_proximos
            else f"Semana regular sin festivos ni quincenas en {sede_sel if sede_sel != 'Todas las sedes' else 'Costa Caribe'}."
        )

        texto_calibracion = (
            f"WAPE recursivo: {kpi_wape} · Cobertura P10–P90: {kpi_cov} ({metricas_modelo['dias_historia']} días analizados)."
            if wape_val is not None
            else f"{metricas_modelo['motor_nombre']} ({metricas_modelo['dias_historia']} días disponibles)."
        )

        st.markdown(
            f'<div class="white-card" style="height: 100%;">'
            f'<div class="card-title" style="margin-bottom: 14px;">Alertas Dinámicas</div>'
            f'<div class="alert-box alert-red">'
            f'<div class="alert-title">{prod_alerta_1}</div>'
            f'<div class="alert-desc">{desc_alerta_1}</div>'
            f"</div>"
            f'<div class="alert-box alert-amber">'
            f'<div class="alert-title">Sensores de calendario local</div>'
            f'<div class="alert-desc">{texto_calendario}</div>'
            f"</div>"
            f'<div class="alert-box alert-green">'
            f'<div class="alert-title">Estado del motor cuantílico</div>'
            f'<div class="alert-desc">{texto_calibracion}</div>'
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # --- FILA 3: TABLA DECISIONARIA "QUÉ PRODUCIR Y DESPACHAR MAÑANA" ---
    with st.container(border=True):
        t_col1, t_col2 = st.columns([7.5, 2.5])
        with t_col1:
            evento_manana_str = df_manana["Evento Mañana"].iloc[0] if not df_manana.empty else "Día regular"
            st.markdown(
                f'<div class="card-title" style="padding-top: 6px;">Directrices de abastecimiento mañana '
                f'<span style="font-size:0.78rem; font-weight:500; color:#7C6350;">({evento_manana_str} · {postura_riesgo})</span></div>',
                unsafe_allow_html=True,
            )
        with t_col2:
            df_export = df_manana[["Producto", "Línea", "Presentación", "Canales", "Evento Mañana", "Pronóstico", "Rango P10–P90", "Nivel_Servicio_Opt", "Producir", "Cajas", "Estado"]].copy()
            csv_plan = df_export.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="Exportar plan de producción",
                data=csv_plan,
                file_name=f"Plan_Abastecimiento_Servipan_{(ultima_fecha + pd.Timedelta(days=1)).strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                width="stretch",
            )

        filas_html = []
        for _, r in df_manana.head(10).iterrows():
            estado = str(r["Estado"])
            if estado == "Estable":
                badge_cls = "badge-stable"
            elif estado == "Subir producción":
                badge_cls = "badge-increase"
            else:
                badge_cls = "badge-risk"

            prod_fmt = f"{int(r['Producir']):,}".replace(",", ".")
            filas_html.append(
                f'<tr>'
                f'<td><div class="prod-name">{r["Producto"]}</div><div class="prod-sub">{r["Línea"]} · {r["Presentación"]} · CR*: {r["Nivel_Servicio_Opt"]} %</div></td>'
                f'<td>{r["Pronóstico"]}</td>'
                f'<td class="prod-range">{r["Rango P10–P90"]}</td>'
                f'<td class="prod-qty">{prod_fmt} <span class="prod-sub">({r["Cajas"]} cajas)</span></td>'
                f'<td><span class="badge {badge_cls}">{estado}</span></td>'
                f'</tr>'
            )

        tabla_html = (
            '<table class="prod-table">'
            '<thead><tr>'
            '<th>Producto</th>'
            '<th>Mediana P50</th>'
            '<th>Rango P10–P90</th>'
            '<th>Orden Objetivo</th>'
            '<th>Estado</th>'
            '</tr></thead>'
            f'<tbody>{"".join(filas_html)}</tbody>'
            '</table>'
        )
        st.markdown(tabla_html, unsafe_allow_html=True)

# ==============================================================================
# 12. VISTAS SECUNDARIAS (MÓDULOS 2 A 8)
# ==============================================================================
elif seccion_activa == "Pronóstico":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    col_p_main, col_p_card = st.columns([7.2, 2.8])

    with col_p_main:
        with st.container(border=True):
            prod_sel = st.selectbox("Seleccionar producto:", df_proy["Producto"].unique())
            df_p_sub = df_proy[df_proy["Producto"] == prod_sel]

            fig_p = go.Figure()
            fig_p.add_trace(
                go.Scatter(
                    x=df_p_sub["Fecha Proyectada"],
                    y=df_p_sub["P90"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig_p.add_trace(
                go.Scatter(
                    x=df_p_sub["Fecha Proyectada"],
                    y=df_p_sub["P10"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(245, 214, 107, 0.35)",
                    name="Rango P10–P90",
                )
            )
            fig_p.add_trace(
                go.Scatter(
                    x=df_p_sub["Fecha Proyectada"],
                    y=df_p_sub["Pronóstico P50"],
                    mode="lines+markers",
                    line=dict(color="#6B2E12", width=2.6),
                    marker=dict(color="#D97724", size=7),
                    name="Pronóstico P50 (Unidades)",
                )
            )
            fig_p.update_layout(
                title=f"Curva de demanda proyectada ({horizonte_sel}) · {prod_sel}",
                height=310,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig_p, width="stretch")

    with col_p_card:
        info_prod = df_manana[df_manana["Producto"] == prod_sel]
        if not info_prod.empty:
            r_p = info_prod.iloc[0]
            img_b64 = None
            if "queso" in prod_sel.lower():
                img_b64 = cargar_imagen_base64("imagen_2.jpg")
            elif "tajado" in prod_sel.lower() or "artesanal" in prod_sel.lower():
                img_b64 = cargar_imagen_base64("imagen_3.jpg")
            if img_b64 is None:
                img_b64 = cargar_imagen_base64("imagen.jpg")

            visual_html = (
                f'<img src="{img_b64}" style="width:100%; max-height:175px; object-fit:cover; border-radius:10px; margin-bottom:10px; border:1px solid #E5C8A6;" />'
                if img_b64
                else f'<div style="width:145px; margin:6px auto 10px auto;">{SVG_PAN_ARTESANAL}</div>'
            )

            st.markdown(
                f'<div class="spotlight-card">'
                f'<div class="spotlight-badge">{r_p["Línea"]}</div>'
                f'{visual_html}'
                f'<div class="spotlight-title">{r_p["Producto"]}</div>'
                f'<div class="spotlight-meta">'
                f'<b>Presentación:</b> {r_p["Presentación"]}<br>'
                f'<b>Canales:</b> {r_p["Canales"]}<br>'
                f'<b>Vida útil:</b> {r_p["Vida Útil"]}<br>'
                f'<b>Nivel servicio objetivo:</b> {r_p["Nivel_Servicio_Opt"]} %<br>'
                f'<b>Orden mañana:</b> {int(r_p["Producir"]):,} und ({r_p["Cajas"]} cajas)'
                f"</div>"
                f"</div>".replace(",", "."),
                unsafe_allow_html=True,
            )

    with st.container(border=True):
        st.dataframe(
            df_p_sub[
                [
                    "Fecha Proyectada",
                    "Día",
                    "Evento Calendario",
                    "Producto",
                    "Pronóstico P50",
                    "P10",
                    "P90",
                    "Cuantil Objetivo",
                    "Producir (Un)",
                    "Sugerido Producir (Cajas)",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

elif seccion_activa == "Productos":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 10px;">Portafolio Oficial Servipan (4 Líneas · 23 Referencias)</div>', unsafe_allow_html=True)
        df_cat_vista = pd.DataFrame(CATALOGO_OFICIAL_SERVIPAN)
        df_cat_vista["Canales de Distribución"] = df_cat_vista.apply(formatear_canales, axis=1)

        f_c1, f_c2 = st.columns(2)
        linea_cat = f_c1.selectbox("Filtrar ficha por Línea:", ["Todas"] + sorted(df_cat_vista["Linea"].unique().tolist()))
        canal_cat = f_c2.selectbox("Filtrar por Canal de Distribución:", ["Todos", "Tiendas", "Supermercados", "Institucional"])

        if linea_cat != "Todas":
            df_cat_vista = df_cat_vista[df_cat_vista["Linea"] == linea_cat]
        if canal_cat != "Todos":
            df_cat_vista = df_cat_vista[df_cat_vista[canal_cat] == True]

        st.dataframe(
            df_cat_vista[["Ref", "Linea", "Producto", "Presentacion", "Canales de Distribución", "Vida_Util_Dias", "Und_Caja", "Min_Cajas", "Precio"]].rename(
                columns={
                    "Linea": "Línea",
                    "Presentacion": "Presentación",
                    "Vida_Util_Dias": "Vida Útil (Días)",
                    "Und_Caja": "Und / Caja",
                    "Min_Cajas": "Lote Mín (Cajas)",
                    "Precio": "Precio Base (COP)",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 10px;">Consulta Histórica de Registros</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        df_op = pd.merge(df_filtrado, df_param, on="Ref", how="left")
        df_op["Cantidad de Cajas"] = np.ceil(df_op["Cant inv"] / df_op["Unidades por Caja"])

        prods = ["Todos"] + sorted(df_op["Desc item"].dropna().unique().tolist())
        p_sel = c1.selectbox("Filtrar por Producto:", prods)
        if p_sel != "Todos":
            df_op = df_op[df_op["Desc item"] == p_sel]

        if col_cliente:
            clis = ["Todos"] + sorted(df_op[col_cliente].dropna().unique().tolist())
            cli_sel = c2.selectbox("Filtrar por Punto / Cliente:", clis)
            if cli_sel != "Todos":
                df_op = df_op[df_op[col_cliente] == cli_sel]

        st.dataframe(df_op, width="stretch")
        st.download_button(
            "Exportar tabla operativa (CSV)",
            data=df_op.to_csv(index=False, sep=";").encode("utf-8-sig"),
            file_name="Consulta_Operativa_Servipan.csv",
            mime="text/csv",
        )

elif seccion_activa == "Sedes":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 6px;">Demanda Histórica y Asignación Proyectada por Corredor Regional (Costa Caribe)</div>', unsafe_allow_html=True)
        df_z = df_enriquecido.groupby("Sede_Regional", as_index=False)["Cant inv"].sum().sort_values("Cant inv")
        tot_hist_sedes = max(1.0, float(df_z["Cant inv"].sum()))
        dem_tot_hor = float(df_proy["Pronóstico P50"].sum()) if not df_proy.empty else 0.0
        df_z["Participación (%)"] = (df_z["Cant inv"] / tot_hist_sedes * 100).round(1)
        df_z[f"Demanda Proyectada ({horizonte_sel})"] = (df_z["Cant inv"] / tot_hist_sedes * dem_tot_hor).round().astype(int)

        fig_z = px.bar(
            df_z,
            y="Sede_Regional",
            x="Cant inv",
            orientation="h",
            text_auto=".3s",
            color_discrete_sequence=["#6B2E12"],
            labels={"Sede_Regional": "Corredor Regional", "Cant inv": "Unidades Históricas"},
        )
        fig_z.update_layout(height=280, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_z, width="stretch")
        st.dataframe(df_z.sort_values("Cant inv", ascending=False), width="stretch", hide_index=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="card-title" style="margin-bottom: 10px;">Directorio de Corredores Comerciales (6 Sedes · 129 Localidades Asignadas)</div>', unsafe_allow_html=True)
        cols_sedes = st.columns(2)
        for idx, (sede_nombre, info_sede) in enumerate(LUGARES_DISTRIBUCION_SERVIPAN.items()):
            with cols_sedes[idx % 2]:
                with st.expander(f"📍 {sede_nombre} ({len(info_sede['municipios'])} localidades)", expanded=(idx == 0)):
                    st.write(", ".join(info_sede["municipios"]))

elif seccion_activa == "Escenarios":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Simulador de Estrategias Comerciales y Choques de Demanda</div>', unsafe_allow_html=True)
        ajuste_pct = st.slider("Variación esperada de demanda por campaña o choque externo (%)", min_value=-30, max_value=50, value=18, step=2)
        df_sim = df_manana[["Producto", "Línea", "Canales", "Pronóstico_num", "P90", "Producir", "Estado"]].copy()
        df_sim["Demanda Simulada P50 (Un)"] = (df_sim["Pronóstico_num"] * (1 + ajuste_pct / 100)).round().astype(int)
        df_sim["Orden Simulada (Un)"] = (df_sim["Producir"] * (1 + ajuste_pct / 100)).round().astype(int)
        st.dataframe(df_sim, width="stretch", hide_index=True)

elif seccion_activa == "Inventario":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Conversión de Cuantil Objetivo a Cajas Comerciales, Lotes Mínimos y Frescura</div>', unsafe_allow_html=True)
        df_inv = pd.merge(df_manana, df_param, on="Ref", how="left")
        st.dataframe(
            df_inv[
                [
                    "Producto",
                    "Línea",
                    "Presentación",
                    "Vida Útil",
                    "Pronóstico",
                    "Nivel_Servicio_Opt",
                    "Unidades por Caja",
                    "Mínimo a Producir (Cajas)",
                    "Producir",
                    "Cajas",
                    "Estado",
                ]
            ].rename(columns={"Nivel_Servicio_Opt": "Nivel Servicio CR* (%)"}),
            width="stretch",
            hide_index=True,
        )

elif seccion_activa == "Alertas":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    eventos_str = ", ".join(eventos_proximos) if eventos_proximos else "Sin festivos ni quincenas en los próximos 7 días"

    riesgos_df = df_manana[df_manana["Estado"] != "Estable"] if not df_manana.empty else pd.DataFrame()
    if not riesgos_df.empty:
        lista_skus_riesgo = ", ".join([f"{r['Producto']} ({r['Var %']:+.1f} %)" for _, r in riesgos_df.head(4).iterrows()])
        titulo_r = f"{len(riesgos_df)} referencias con alerta activa de tracción o perecibilidad"
        desc_r = f"SKUs priorizados: {lista_skus_riesgo}. Postura activa: {postura_riesgo}."
    else:
        titulo_r = "Portafolio en rango operativo estable"
        desc_r = "Ningún SKU supera el umbral de crecimiento semanal (+8 %) ni presenta picos críticos de perecibilidad."

    wape_str = f"{metricas_modelo['wape_7d']} %" if metricas_modelo["wape_7d"] is not None else "N/D"
    cov_str = f"{metricas_modelo['cobertura_7d']} %" if metricas_modelo["cobertura_7d"] is not None else "N/D"

    st.markdown(
        f'<div class="white-card">'
        f'<div class="card-title" style="margin-bottom: 14px;">Bitácora Dinámica de Alertas Tempranas</div>'
        f'<div class="alert-box alert-red">'
        f'<div class="alert-title">{titulo_r}</div>'
        f'<div class="alert-desc">{desc_r}</div>'
        f"</div>"
        f'<div class="alert-box alert-amber">'
        f'<div class="alert-title">Calendario sociocultural y económico (Próximos 7 días)</div>'
        f'<div class="alert-desc">{eventos_str}. El modelo cuantílico ajusta automáticamente las referencias según el contexto local.</div>'
        f"</div>"
        f'<div class="alert-box alert-green">'
        f'<div class="alert-title">Auditoría de calibración · WAPE recursivo: {wape_str} · Cobertura P10–P90: {cov_str}</div>'
        f'<div class="alert-desc">Evaluado sobre {metricas_modelo["dias_historia"]} días de historia con {metricas_modelo["motor_nombre"]}.</div>'
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

elif seccion_activa == "Configuración":
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Arquitectura del Motor Predictivo y Regla Newsvendor</div>', unsafe_allow_html=True)
        st.write(f"- **Origen de datos activo:** `{'Benchmark sintético calibrado (180 días)' if es_ejemplo else 'Archivo Excel cargado por el usuario'}`.")
        st.write(f"- **Algoritmo activo:** `{metricas_modelo['motor_nombre']}` entrenado de forma global sobre las 23 referencias (5 cuantiles: `0.10, 0.25, 0.50, 0.75, 0.90`).")
        st.write("- **24 variables explicativas:** `15` variables de calendario sociocultural y económico colombiano (día de semana, día del mes, mes, fin de semana, 18 festivos nacionales bajo Ley 51/1983, víspera de festivo, quincenas, primas legales, Festival de la Leyenda Vallenata, Carnaval de Barranquilla [sábado a martes previos al Miércoles de Ceniza], Semana Santa, temporada navideña, vacaciones escolares Calendario A, recesos universitarios y días efectivos de clase) + `9` variables de producto y memoria autorregresiva (`Ref_ID`, `Linea_ID`, `Vida_Util_Dias`, `lag_1`, `lag_7`, `lag_14`, `roll_mean_7`, `roll_mean_28`, `roll_std_7`).")
        st.write("- **Regla de decisión Newsvendor:** Calcula el cuantil crítico $CR^* = c_u / (c_u + c_o)$ por SKU según margen unitario, penalización por quiebre y valor de salvamento por perecibilidad, permitiendo alternar entre posturas `Conservador (P50)`, `Óptimo Newsvendor (CR*)` y `Expansivo / Captura (P90)`.")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="card-title">Auditoría Experimental: Backtesting Rolling-Origin (30 días), Ablación y Políticas</div>', unsafe_allow_html=True)
        st.caption("Evalúa pronósticos recursivos multi-origen (4 orígenes separados cada 7 días) comparando el modelo completo contra el modelo sin variables de calendario y contra Naive Estacional.")
        if st.button("Ejecutar Auditoría Completa (Rolling-Origin + Ablación + Newsvendor)", type="primary"):
            with st.spinner("Entrenando orígenes móviles y evaluando políticas económicas..."):
                df_bt_res, df_pol_res = ejecutar_auditoria_rolling_y_ablacion(
                    df_diario_ml, df_param, costo_var_pct, penalizacion_quiebre_pct
                )
            if df_bt_res.empty:
                st.warning("Se requieren al menos 75 días de historia para ejecutar la auditoría completa a 30 días.")
            else:
                st.markdown("**1. Desempeño Predictivo y Estudio de Ablación por Horizonte**")
                st.dataframe(df_bt_res, width="stretch", hide_index=True)
                st.markdown("**2. Evaluación Económica de Políticas de Abastecimiento (Backtest Out-of-Sample)**")
                st.dataframe(df_pol_res, width="stretch", hide_index=True)
