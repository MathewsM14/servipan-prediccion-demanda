"""Catálogo, corredores regionales, enriquecimiento de facturación y generador sintético."""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from calendario import extraer_features_fecha, obtener_calendario_colombia

# Fecha fija: el benchmark sintético es reproducible (no cambia con la fecha de ejecución).
FECHA_FIN_SINTETICO = "2026-11-15"

_C = lambda ref, prod, lin, pres, t, s, i, vu, uc, mc, base, pr, perf: {
    "Ref": ref, "Producto": prod, "Linea": lin, "Presentacion": pres, "Tiendas": t,
    "Supermercados": s, "Institucional": i, "Vida_Util_Dias": vu, "Und_Caja": uc,
    "Min_Cajas": mc, "Base_Diaria": base, "Precio": pr, "Perfil_Escolar": perf,
}

# Precio = precio de lista del catálogo. Para el newsvendor se necesita el PRECIO NETO de venta
# a tiendas y el COSTO VARIABLE real (ver carga de costos en la interfaz).
CATALOGO_OFICIAL_SERVIPAN = [
    _C("PAN-01", "Pan Mogolla Blanca", "Panes", "280 gr", True, True, False, 6, 20, 8, 310, 3400, "familiar"),
    _C("PAN-02", "Pan Mogolla Integral", "Panes", "280 gr", True, True, False, 6, 20, 5, 145, 3600, "familiar"),
    _C("PAN-03", "Pan Mogolla con bocadillo", "Panes", "Unidad 35g, 60g y 80g", True, False, False, 5, 30, 6, 280, 1800, "lonchera"),
    _C("PAN-04", "Pan Hamburguesa", "Panes", "4 unidades (450g)", False, True, False, 7, 16, 6, 190, 4500, "festivo"),
    _C("PAN-05", "Pan Perro", "Panes", "Paquete Pan Perro de papa", True, True, False, 7, 16, 5, 175, 4300, "festivo"),
    _C("PAN-06", "Pan de Queso", "Panes", "Paquete extra queso", True, True, False, 4, 20, 8, 260, 3800, "lonchera"),
    _C("TAJ-01", "Tajado Multicereal", "Tajados", "Bolsa pan tajado", True, True, False, 10, 12, 5, 110, 5800, "familiar"),
    _C("TAJ-02", "Bolsa de Pan Artesanal", "Tajados", "Bolsa pan artesanal", True, True, False, 8, 12, 5, 135, 6200, "familiar"),
    _C("TAJ-03", "Pan Tajado Blanco XL", "Tajados", "650 g", False, False, True, 10, 10, 8, 210, 6500, "familiar"),
    _C("TAJ-04", "Pan Tajado Integral", "Tajados", "Bolsa pan tajado", True, True, False, 10, 14, 6, 165, 5400, "familiar"),
    _C("TAJ-05", "Pan Tajado de Leche", "Tajados", "450g, 500g y 550g", True, True, False, 9, 14, 8, 240, 5200, "familiar"),
    _C("TAJ-06", "Pan Tajado Casero", "Tajados", "420 g", True, True, False, 8, 15, 6, 195, 4800, "familiar"),
    _C("TAJ-07", "Pan Tajado de Mantequilla", "Tajados", "550 g", True, True, False, 9, 14, 6, 180, 5500, "familiar"),
    _C("TAJ-08", "Pan Tajado Blanco", "Tajados", "380 g", True, True, False, 9, 16, 10, 340, 4400, "familiar"),
    _C("ROS-01", "Rosquitas Integrales de Yuca Grandes", "Rosquitas", "Paquete familiar / 12 und", False, True, True, 45, 12, 5, 95, 6800, "familiar"),
    _C("ROS-02", "Rosquitas Quinua y Chia Grandes", "Rosquitas", "Paquete familiar / 12 und", False, True, True, 45, 12, 4, 85, 7200, "familiar"),
    _C("ROS-03", "Rosquitas Yuca y Queso Grandes", "Rosquitas", "Paquete familiar / 12 und", False, True, True, 45, 12, 5, 110, 7000, "familiar"),
    _C("ROS-04", "Rosquitas integrales de Yuca y queso", "Rosquitas", "Unidad 18g / Paquete x 12 und", False, True, True, 45, 24, 6, 150, 1500, "lonchera"),
    _C("ROS-05", "Rosquitas integrales de Yuca", "Rosquitas", "Unidad 18g / Paquete x 12 und", False, True, True, 45, 24, 6, 140, 1500, "lonchera"),
    _C("ROS-06", "Rosquitas de Quinua y Chia", "Rosquitas", "Unidad 18g / Paquete x 12 und", False, True, True, 45, 24, 5, 130, 1600, "lonchera"),
    _C("GAL-01", "Punto Rojo", "Galletas", "Unidad individual", True, False, False, 30, 36, 8, 290, 1400, "lonchera"),
    _C("GAL-02", "Chelitas de Guayaba", "Galletas", "Unidad / Paquete x 6 und (180g)", True, False, False, 30, 24, 6, 215, 3200, "lonchera"),
    _C("GAL-03", "Galleta Cuca", "Galletas", "50g / 60 grs (por unidad)", True, True, True, 35, 36, 10, 360, 1300, "lonchera"),
]

# 6 corredores comerciales; 129 localidades asignadas (homónimos con sufijo departamental).
# Algunas localidades pertenecen administrativamente a otro departamento: son corredores, no departamentos.
LUGARES_DISTRIBUCION_SERVIPAN = {
    "Cesar - Valledupar": {"peso": 0.34, "municipios": [
        "Valledupar", "San Diego", "Manaure", "Media Luna", "Atánquez", "Patillal", "Los Corazones", "La Paz",
        "Urumita", "Villanueva", "El Molino", "Codazzi", "La Jagua de Ibirico", "Chiriguaná", "La Loma",
        "Curumaní", "Tamalameque", "Pailitas", "Chimichagua", "Arjona", "Astrea", "El Paso", "Candelaria (Cesar)",
        "Sempegua", "Aguachica", "La Gloria (Cesar)", "Regidor", "Río Viejo", "Pelaya", "Gamarra", "Bosconia",
        "El Copey", "Caracolí", "Los Venados"]},
    "Guajira - Riohacha": {"peso": 0.16, "municipios": [
        "Riohacha", "San Juan del Cesar", "Maicao", "La Peña", "La Junta", "Cañaveral", "Pandores", "Fonseca",
        "Distracción", "Buenavista", "Barrancas", "Papayal", "Hatonuevo", "Guayacamal", "Albania", "Cuestecita"]},
    "Atlántico - Barranquilla": {"peso": 0.18, "municipios": [
        "Barranquilla", "Soledad", "Sabanalarga", "Arenal", "Pendales", "Santa Rosa", "Molineros", "Repelón",
        "Luruaco", "Manatí", "Santa Cruz", "Isabel López", "Pital de Megua", "Baranoa", "Soplaviento",
        "Santa Lucía", "San Cristóbal", "Sabanagrande", "Santo Tomás", "Palmar de Varela", "Ponedera",
        "Retirada", "Martillo", "Candelaria (Atlántico)", "Cascajal", "Puerto Giraldo", "Bohórquez",
        "Campo de la Cruz", "Suán", "Calamar"]},
    "Magdalena - Santa Marta": {"peso": 0.14, "municipios": [
        "Santa Marta", "Fundación", "Ciénaga", "El Difícil", "Pueblo Nuevo", "San Ángel", "Tres Esquinas",
        "Andes", "San Martín de Loba", "Ricaurte", "Guaimaral", "Granada", "El Bajo", "Chibolo", "Tenerife",
        "Palmar", "El Pueblito", "Pijiño", "San Fernando (Magdalena)", "Guamal", "Las Flores", "Casa de Tabla",
        "Plato", "La Gloria (Magdalena)", "Mamoncito", "El Banco", "Santa Ana", "Santa Inés"]},
    "Bolívar - Cartagena": {"peso": 0.09, "municipios": [
        "Cartagena", "Mompox", "Talaigua Nuevo", "San Sebastián", "Menchiquejo", "Rinconada",
        "San Fernando (Bolívar)", "Guataca"]},
    "Sucre - Sincelejo": {"peso": 0.09, "municipios": [
        "Sincelejo", "Sampués", "La Unión", "Chinú", "Corozal", "San Marcos", "Majagual", "Achí", "Zambrano",
        "Ovejas", "El Carmen de Bolívar (Ruta Sucre)", "San Pedro", "Magangué (Ruta Sucre)"]},
}

SIN_ASIGNAR = "Sin asignar"
_COLS_FICHA = ["Linea", "Presentacion", "Tiendas", "Supermercados", "Institucional",
               "Vida_Util_Dias", "Und_Caja", "Min_Cajas", "Precio"]


def normalizar_texto(texto: str) -> str:
    s = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("utf-8")
    return s.lower().strip()


def formatear_canales(row) -> str:
    canales = [n for n in ("Tiendas", "Supermercados", "Institucional") if row.get(n)]
    return " · ".join(canales) if canales else "General"


def _ficha_por_defecto(nombre_norm: str) -> dict:
    """Respaldo por palabra clave (se contabiliza y se avisa al usuario)."""
    if "rosquita" in nombre_norm:
        return dict(zip(_COLS_FICHA, ["Rosquitas", "Empaque estándar", False, True, True, 45, 24, 5, 1600]))
    if "tajado" in nombre_norm or "artesanal" in nombre_norm:
        return dict(zip(_COLS_FICHA, ["Tajados", "Bolsa tajado", True, True, False, 9, 14, 6, 5200]))
    if any(k in nombre_norm for k in ("galleta", "cuca", "chelita", "punto rojo")):
        return dict(zip(_COLS_FICHA, ["Galletas", "Empaque galleta", True, False, False, 30, 30, 6, 1500]))
    return dict(zip(_COLS_FICHA, ["Panes", "Empaque panadería", True, True, False, 6, 20, 6, 3500]))


def _mapa_localidades() -> dict:
    """nombre normalizado -> corredor, o None si el nombre es ambiguo entre corredores."""
    tmp: dict[str, set] = {}
    for sede, info in LUGARES_DISTRIBUCION_SERVIPAN.items():
        tmp.setdefault(normalizar_texto(sede), set()).add(sede)
        for m in info["municipios"]:
            tmp.setdefault(normalizar_texto(m.split("(")[0]), set()).add(sede)
            tmp.setdefault(normalizar_texto(m), set()).add(sede)
    return {k: (next(iter(v)) if len(v) == 1 else None) for k, v in tmp.items()}


def enriquecer_dataframe_con_maestros(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cruce vectorizado por Ref y por nombre exacto. Devuelve (df, diagnóstico de calidad)."""
    out = df.copy()
    cat = pd.DataFrame(CATALOGO_OFICIAL_SERVIPAN)
    cat["_ref"] = cat["Ref"].str.strip().str.upper()
    cat["_nom"] = cat["Producto"].map(normalizar_texto)
    out["_ref"] = out["Ref"].astype(str).str.strip().str.upper()
    out["_nom"] = out["Desc item"].map(normalizar_texto)

    ficha = out[["_ref"]].merge(cat[["_ref"] + _COLS_FICHA], on="_ref", how="left")
    ficha.index = out.index
    falta = ficha["Linea"].isna()
    if falta.any():
        por_nombre = out.loc[falta, ["_nom"]].merge(cat[["_nom"] + _COLS_FICHA], on="_nom", how="left")
        por_nombre.index = out.index[falta]
        ficha.loc[falta, _COLS_FICHA] = por_nombre[_COLS_FICHA]
    falta = ficha["Linea"].isna()
    n_respaldo = int(falta.sum())
    if n_respaldo:
        for nom in out.loc[falta, "_nom"].unique():
            fd = _ficha_por_defecto(nom)
            idx = out.index[falta & (out["_nom"] == nom)]
            for c in _COLS_FICHA:
                ficha.loc[idx, c] = fd[c]

    if "Sub linea" not in out.columns:
        out["Sub linea"] = ficha["Linea"]
    out["Linea_Oficial"] = ficha["Linea"]
    out["Presentacion"] = ficha["Presentacion"]
    out["Canales"] = ficha.apply(formatear_canales, axis=1)
    out["Vida_Util_Dias"] = ficha["Vida_Util_Dias"].astype(int)
    out["Und_Caja_Cat"] = ficha["Und_Caja"].astype(int)
    out["Min_Cajas_Cat"] = ficha["Min_Cajas"].astype(int)
    out["Precio_Cat"] = ficha["Precio"].astype(float)

    if "Sede_Regional" not in out.columns:
        mapa = _mapa_localidades()
        if "Zona" in out.columns:
            out["Sede_Regional"] = out["Zona"].map(lambda z: mapa.get(normalizar_texto(z))).fillna(SIN_ASIGNAR)
        else:
            out["Sede_Regional"] = SIN_ASIGNAR
    n_sin_sede = int((out["Sede_Regional"] == SIN_ASIGNAR).sum())
    return out.drop(columns=["_ref", "_nom"]), {"fichas_por_respaldo": n_respaldo, "filas_sin_sede": n_sin_sede, "filas": len(out)}


def generar_datos_ejemplo(dias: int = 180, fecha_fin: str = FECHA_FIN_SINTETICO) -> pd.DataFrame:
    """Benchmark SINTÉTICO y reproducible. Los efectos de calendario se programan aquí mismo,
    por lo que NO sirve para validar desempeño real (solo para probar el pipeline)."""
    fechas = pd.date_range(end=pd.Timestamp(fecha_fin), periods=dias, freq="D")
    rng = np.random.default_rng(2026)
    cal = obtener_calendario_colombia(sorted({f.year for f in fechas} | {fechas[-1].year + 1}))
    df_cal = extraer_features_fecha(pd.Series(fechas), cal)
    sedes = list(LUGARES_DISTRIBUCION_SERVIPAN)
    pesos = [LUGARES_DISTRIBUCION_SERVIPAN[s]["peso"] for s in sedes]
    factor_dow = {0: 0.94, 1: 0.96, 2: 0.99, 3: 1.02, 4: 1.11, 5: 1.18, 6: 0.86}

    registros = []
    for idx_f, f in enumerate(fechas):
        c = df_cal.iloc[idx_f]
        mult_cal = 1.0
        if c["es_quincena"]:
            mult_cal *= 1.15
        if c["es_prima"]:
            mult_cal *= 1.12
        if c["vispera_festivo"]:
            mult_cal *= 1.14
        if c["es_festivo"]:
            mult_cal *= 0.92
        if c["festival_vallenato"] or c["carnaval_caribe"]:
            mult_cal *= 1.18
        for prod in CATALOGO_OFICIAL_SERVIPAN:
            perfil, m = prod["Perfil_Escolar"], mult_cal
            if c["vacaciones_colegios"]:
                m *= 0.86 if perfil == "lonchera" else (1.09 if perfil == "familiar" else 1.0)
            if perfil == "festivo" and (c["vispera_festivo"] or c["is_weekend"]):
                m *= 1.12
            total = max(12, int(round(prod["Base_Diaria"] * factor_dow[f.day_of_week] * m * rng.normal(1.0, 0.045))))
            for sede, cant in zip(sedes, rng.multinomial(total, pesos)):
                if cant > 0:
                    muni = rng.choice(LUGARES_DISTRIBUCION_SERVIPAN[sede]["municipios"])
                    registros.append({
                        "Fecha": f, "Ref": prod["Ref"], "Desc item": prod["Producto"], "Cant inv": int(cant),
                        "Zona": muni, "Sede_Regional": sede, "Sub linea": prod["Linea"],
                        "Valor": int(cant * prod["Precio"]),
                        "Razon social cliente factura": f"Punto {muni}",
                    })
    return pd.DataFrame(registros)
