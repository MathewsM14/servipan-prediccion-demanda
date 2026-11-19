"""Calendario colombiano, festividades de la Costa Caribe y variables sociocultural/económicas."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# VERIFICAR con la organización del Festival (las de años futuros son proyecciones).
FESTIVAL_VALLENATO_FECHAS = {
    2024: (date(2024, 4, 26), date(2024, 4, 30)),
    2025: (date(2025, 4, 30), date(2025, 5, 3)),
    2026: (date(2026, 4, 29), date(2026, 5, 2)),
    2027: (date(2027, 4, 28), date(2027, 5, 1)),
}

# Grupos para el estudio de ablación (de lo más básico a lo más sociocultural).
GRUPOS_CALENDARIO = {
    "basico": ["dow", "day", "month", "is_weekend"],
    "legal_economico": ["es_festivo", "vispera_festivo", "es_quincena", "es_prima", "dias_desde_quincena"],
    "sociocultural": [
        "festival_vallenato", "carnaval_caribe", "semana_santa", "temporada_navidena",
        "vacaciones_colegios", "vacaciones_universidades", "periodo_clases_colegio",
    ],
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
    """Ley 51 de 1983: traslada al siguiente lunes si no cae lunes."""
    return d if d.weekday() == 0 else d + timedelta(days=(7 - d.weekday()))


def obtener_calendario_colombia(anios: list[int]) -> dict:
    festivos, carnavales, semana_santa, receso_octubre, fest_vallenato = set(), set(), set(), set(), set()
    for anio in anios:
        pascua = calcular_pascua(anio)
        festivos.update([
            date(anio, 1, 1), date(anio, 5, 1), date(anio, 7, 20), date(anio, 8, 7),
            date(anio, 12, 8), date(anio, 12, 25),
            pascua - timedelta(days=3), pascua - timedelta(days=2),
            trasladar_lunes_emiliani(date(anio, 1, 6)),
            trasladar_lunes_emiliani(date(anio, 3, 19)),
            trasladar_lunes_emiliani(pascua + timedelta(days=39)),
            trasladar_lunes_emiliani(pascua + timedelta(days=60)),
            trasladar_lunes_emiliani(pascua + timedelta(days=68)),
            trasladar_lunes_emiliani(date(anio, 6, 29)),
            trasladar_lunes_emiliani(date(anio, 8, 15)),
            trasladar_lunes_emiliani(date(anio, 10, 12)),
            trasladar_lunes_emiliani(date(anio, 11, 1)),
            trasladar_lunes_emiliani(date(anio, 11, 11)),
        ])
        # Carnaval de Barranquilla: sábado (-50) a martes (-47) antes del Miércoles de Ceniza (-46)
        for offset in range(50, 46, -1):
            carnavales.add(pascua - timedelta(days=offset))
        for offset in range(7, -1, -1):
            semana_santa.add(pascua - timedelta(days=offset))
        oct12 = date(anio, 10, 12)
        lunes_receso = oct12 - timedelta(days=oct12.weekday())
        for offset in range(7):
            receso_octubre.add(lunes_receso + timedelta(days=offset))
        ini, fin = FESTIVAL_VALLENATO_FECHAS.get(anio, (date(anio, 4, 27), date(anio, 4, 30)))
        cur = ini
        while cur <= fin:
            fest_vallenato.add(cur)
            cur += timedelta(days=1)

    return {
        "festivos": festivos,
        "visperas": {f - timedelta(days=1) for f in festivos},
        "carnavales": carnavales,
        "semana_santa": semana_santa,
        "receso_octubre": receso_octubre,
        "festival_vallenato": fest_vallenato,
    }


def extraer_features_fecha(fechas: pd.Series, cal: dict) -> pd.DataFrame:
    """16 variables de calendario: 4 básicas, 5 legales/económicas y 7 socioculturales."""
    dt = pd.to_datetime(fechas)
    d_py = dt.dt.date
    mes, dia, dow = dt.dt.month, dt.dt.day, dt.dt.day_of_week

    es_festivo = d_py.isin(cal["festivos"]).astype(int)
    vispera_festivo = d_py.isin(cal["visperas"]).astype(int)

    # Pagos quincenales: día 15 y cierre de mes (30, o fin de febrero); sábado previo si cae domingo
    fin_mes_pago = (dia >= 30) | ((mes == 2) & dt.dt.is_month_end)
    es_dia_pago = (dia == 15) | fin_mes_pago
    es_prev_dom = ((dia == 14) | (dia == 29) | ((mes == 2) & (dt + pd.Timedelta(days=1)).dt.is_month_end)) & (dow == 5)
    es_quincena = (es_dia_pago | es_prev_dom).astype(int)
    # El efecto de liquidez dura varios días: días transcurridos desde el último pago (0 a 14)
    dias_desde_quincena = np.where(fin_mes_pago, 0, np.where(dia >= 15, dia - 15, dia))

    # Prima legal (Art. 306 CST: hasta 30 de junio y 20 de diciembre)
    es_prima = (((mes == 6) & dia.between(15, 30)) | ((mes == 12) & dia.between(10, 20))).astype(int)

    temporada_navidena = (((mes == 12) & (dia >= 15)) | ((mes == 1) & (dia <= 6))).astype(int)
    vac_mitad = (((mes == 6) & (dia >= 15)) | ((mes == 7) & (dia <= 15))).astype(int)
    vac_receso_oct = d_py.isin(cal["receso_octubre"]).astype(int)
    vac_fin_col = (((mes == 11) & (dia >= 25)) | (mes == 12) | ((mes == 1) & (dia <= 20))).astype(int)
    vac_fin_uni = (((mes == 11) & (dia >= 25)) | (mes == 12) | (mes == 1) | ((mes == 2) & (dia <= 5))).astype(int)
    vac_col = np.maximum.reduce([vac_mitad, vac_receso_oct, vac_fin_col])
    vac_uni = np.maximum.reduce([vac_mitad, vac_fin_uni])

    return pd.DataFrame(
        {
            "dow": dow, "day": dia, "month": mes, "is_weekend": (dow >= 5).astype(int),
            "es_festivo": es_festivo, "vispera_festivo": vispera_festivo,
            "es_quincena": es_quincena, "es_prima": es_prima,
            "dias_desde_quincena": dias_desde_quincena,
            "festival_vallenato": d_py.isin(cal["festival_vallenato"]).astype(int),
            "carnaval_caribe": d_py.isin(cal["carnavales"]).astype(int),
            "semana_santa": d_py.isin(cal["semana_santa"]).astype(int),
            "temporada_navidena": temporada_navidena,
            "vacaciones_colegios": vac_col, "vacaciones_universidades": vac_uni,
            "periodo_clases_colegio": (1 - vac_col) * (dow < 5).astype(int),
        },
        index=fechas.index,
    )


def describir_evento_calendario(row_f: pd.Series) -> str:
    et = []
    if row_f.get("es_festivo", 0) == 1:
        et.append("Festivo")
    elif row_f.get("vispera_festivo", 0) == 1:
        et.append("Víspera festivo")
    if row_f.get("festival_vallenato", 0) == 1:
        et.append("Festival Vallenato")
    if row_f.get("carnaval_caribe", 0) == 1:
        et.append("Carnaval Caribe")
    if row_f.get("semana_santa", 0) == 1:
        et.append("Semana Santa")
    if row_f.get("temporada_navidena", 0) == 1:
        et.append("Navidad")
    if row_f.get("es_prima", 0) == 1:
        et.append("Prima")
    elif row_f.get("es_quincena", 0) == 1:
        et.append("Quincena")
    if row_f.get("vacaciones_colegios", 0) == 1:
        et.append("Vacaciones escolares")
    elif row_f.get("vacaciones_universidades", 0) == 1:
        et.append("Vacaciones U.")
    return " · ".join(et) if et else "Día hábil / regular"
