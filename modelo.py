"""Motor predictivo cuantílico, validación con origen móvil, ablación y decisión newsvendor.
No depende de Streamlit (se puede probar y reutilizar en scripts)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    SKLEARN_DISPONIBLE = True
except ImportError:  # pragma: no cover
    SKLEARN_DISPONIBLE = False

from calendario import (GRUPOS_CALENDARIO, describir_evento_calendario,
                        extraer_features_fecha, obtener_calendario_colombia)

CUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QCOLS = [f"q{int(round(q * 100)):02d}" for q in CUANTILES]
HORIZONTES = {"1-7": (1, 7), "8-14": (8, 14), "15-30": (15, 30)}
HORIZONTE_MAX = 30
UMBRAL_VIDA_CORTA = 10  # días: el newsvendor de un período solo aplica a perecibles de vida corta
POSTURAS = ("Óptimo Newsvendor (CR*)", "Conservador (P50)", "Expansivo / Captura (P90)")


def _col_q(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


# ------------------------------------------------------------------------------
# Panel y variables
# ------------------------------------------------------------------------------
def construir_panel_features(df_diario_in: pd.DataFrame, refs_activas: list[str]) -> dict:
    df = df_diario_in.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    ultima, primera = df["Fecha"].max(), df["Fecha"].min()
    dias_historia = (ultima - primera).days + 1
    fechas_fut = pd.date_range(ultima + pd.Timedelta(days=1), periods=HORIZONTE_MAX, freq="D")
    cal = obtener_calendario_colombia(sorted({primera.year, ultima.year, fechas_fut.max().year}))
    rango = pd.date_range(primera, fechas_fut.max(), freq="D")
    df_cal = extraer_features_fecha(pd.Series(rango), cal)
    df_cal["Fecha"] = rango
    df_cal["Evento_Calendario"] = df_cal.apply(describir_evento_calendario, axis=1)
    cal_lookup = df_cal.set_index("Fecha")
    cal_cols = [c for c in df_cal.columns if c not in ("Fecha", "Evento_Calendario")]

    df = df[df["Ref"].astype(str).isin(refs_activas)]
    meta_map = (df[["Ref", "Desc item", "Linea_Oficial", "Presentacion", "Canales", "Vida_Util_Dias", "Precio_Cat"]]
                .drop_duplicates("Ref").set_index("Ref").to_dict("index"))
    refs = list(meta_map)
    ref_to_id = {r: i for i, r in enumerate(refs)}
    linea_to_id = {l: i for i, l in enumerate(sorted({m["Linea_Oficial"] for m in meta_map.values()}))}
    meta_ids = {r: (ref_to_id[r], linea_to_id[meta_map[r]["Linea_Oficial"]], int(meta_map[r]["Vida_Util_Dias"])) for r in refs}

    grid = pd.MultiIndex.from_product([refs, pd.date_range(primera, ultima, freq="D")], names=["Ref", "Fecha"]).to_frame(index=False)
    ventas = df.groupby(["Ref", "Fecha"], as_index=False)["Cant inv"].sum()
    # Aviso: un 0 imputado puede ser un quiebre de stock y no demanda cero (demanda censurada).
    panel = grid.merge(ventas, on=["Ref", "Fecha"], how="left")
    panel["Cant inv"] = panel["Cant inv"].fillna(0.0)
    panel = panel.sort_values(["Ref", "Fecha"]).reset_index(drop=True)
    panel["Ref_ID"] = panel["Ref"].map(ref_to_id)
    panel["Linea_ID"] = panel["Ref"].map(lambda r: meta_ids[r][1])
    panel["Vida_Util_Dias"] = panel["Ref"].map(lambda r: meta_ids[r][2])
    panel = panel.merge(df_cal[["Fecha"] + cal_cols], on="Fecha", how="left")

    g = panel.groupby("Ref")["Cant inv"]
    panel["lag_1"], panel["lag_7"], panel["lag_14"] = g.shift(1), g.shift(7), g.shift(14)
    panel["roll_mean_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    panel["roll_mean_28"] = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    panel["roll_std_7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=2).std()).fillna(0.0)

    base_cols = ["Ref_ID", "Linea_ID", "Vida_Util_Dias", "lag_1", "lag_7", "lag_14",
                 "roll_mean_7", "roll_mean_28", "roll_std_7"]
    return dict(panel=panel, meta_map=meta_map, meta_ids=meta_ids, refs=refs, cal_lookup=cal_lookup,
                cal_cols=cal_cols, base_cols=base_cols, feature_cols=base_cols + cal_cols,
                ultima_fecha=ultima, dias_historia=dias_historia, fechas_futuras=fechas_fut)


# ------------------------------------------------------------------------------
# Modelos y pronóstico recursivo
# ------------------------------------------------------------------------------
def _nuevo_modelo(q: float, max_iter: int = 85):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=max_iter, max_depth=4, learning_rate=0.09,
        l2_regularization=1.0, categorical_features=["Ref_ID", "Linea_ID"], random_state=42)


def entrenar_cuantiles(train_df, feature_cols, cuantiles=CUANTILES) -> dict:
    X, y = train_df[feature_cols], train_df["Cant inv"]
    return {q: _nuevo_modelo(q).fit(X, y) for q in cuantiles}


def _fila_features(serie, ids, c_row, cal_cols) -> dict:
    ref_id, linea_id, vida = ids
    lag_1 = serie[-1]
    lag_7 = serie[-7] if len(serie) >= 7 else lag_1
    lag_14 = serie[-14] if len(serie) >= 14 else lag_7
    w7, w28 = serie[-7:], serie[-28:]
    fila = {"Ref_ID": ref_id, "Linea_ID": linea_id, "Vida_Util_Dias": vida, "lag_1": lag_1,
            "lag_7": lag_7, "lag_14": lag_14, "roll_mean_7": float(np.mean(w7)),
            "roll_mean_28": float(np.mean(w28)),
            "roll_std_7": float(np.std(w7, ddof=1)) if len(w7) >= 2 else 0.0}
    fila.update({c: int(c_row[c]) for c in cal_cols})
    return fila


def pronostico_recursivo(modelos, hist_series, refs, meta_ids, fechas, cal_lookup, cal_cols, feature_cols):
    """Realimenta el P50 continuo (sin redondeo) y ordena cuantiles sin imponer márgenes."""
    hist = {r: list(v) for r, v in hist_series.items()}
    qs = sorted(modelos)
    i50 = qs.index(0.50)
    filas = []
    for h, f in enumerate(fechas, start=1):
        c_row = cal_lookup.loc[f]
        X = pd.DataFrame([_fila_features(hist[r], meta_ids[r], c_row, cal_cols) for r in refs])[feature_cols]
        P = np.sort(np.column_stack([np.maximum(0.0, modelos[q].predict(X)) for q in qs]), axis=1)
        ev = str(c_row["Evento_Calendario"])
        for i, r in enumerate(refs):
            hist[r].append(float(P[i, i50]))
            filas.append({"Fecha": f, "Ref": r, "h": h, "Evento_Calendario": ev,
                          **{_col_q(q): P[i, j] for j, q in enumerate(qs)}})
    return pd.DataFrame(filas)


def backtest_rolling_origin(ctx, feature_cols=None, cal_cols=None, n_origenes=4, paso=7,
                            horizonte=30, min_train=45, cuantiles=CUANTILES) -> pd.DataFrame:
    """Entrena solo con Fecha <= T y pronostica recursivamente T+1..T+horizonte (igual que producción).
    Incluye dos referencias: naive estacional (mismo día, semana anterior disponible) y media de 4 semanas."""
    feature_cols = feature_cols or ctx["feature_cols"]
    cal_cols = ctx["cal_cols"] if cal_cols is None else cal_cols
    panel = ctx["panel"]
    ultima, primera = panel["Fecha"].max(), panel["Fecha"].min()
    lookup = panel.set_index(["Ref", "Fecha"])["Cant inv"].to_dict()
    origenes = [ultima - pd.Timedelta(days=horizonte + k * paso) for k in range(n_origenes)]
    origenes = [o for o in origenes if (o - primera).days + 1 >= min_train]

    resultados = []
    for T in sorted(origenes):
        train = panel[panel["Fecha"] <= T].dropna(subset=["lag_7"])
        if len(train) < 30:
            continue
        modelos = entrenar_cuantiles(train, feature_cols, cuantiles)
        hist = {r: panel[(panel["Ref"] == r) & (panel["Fecha"] <= T)]["Cant inv"].tolist() for r in ctx["refs"]}
        fechas = pd.date_range(T + pd.Timedelta(days=1), periods=horizonte, freq="D")
        pred = pronostico_recursivo(modelos, hist, ctx["refs"], ctx["meta_ids"], fechas,
                                    ctx["cal_lookup"], cal_cols, feature_cols)
        pred["origen"] = T
        pred["real"] = [lookup.get((r, f), np.nan) for r, f in zip(pred["Ref"], pred["Fecha"])]
        sem = np.ceil(pred["h"] / 7).astype(int)
        pred["naive_estacional"] = [lookup.get((r, f - pd.Timedelta(days=7 * s)), np.nan)
                                    for r, f, s in zip(pred["Ref"], pred["Fecha"], sem)]
        pred["media_4sem"] = [np.nanmean([lookup.get((r, f - pd.Timedelta(days=7 * (s + k))), np.nan) for k in range(4)])
                              for r, f, s in zip(pred["Ref"], pred["Fecha"], sem)]
        resultados.append(pred)
    if not resultados:
        return pd.DataFrame()
    return pd.concat(resultados, ignore_index=True).dropna(subset=["real"])


# ------------------------------------------------------------------------------
# Métricas
# ------------------------------------------------------------------------------
def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / y.sum() * 100) if y.sum() > 0 else np.nan


def pinball(y, q_pred, q) -> float:
    d = np.asarray(y, float) - np.asarray(q_pred, float)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _comparables(s):
    """Solo filas donde existen ambas referencias: sin rellenar con el propio modelo."""
    return s.dropna(subset=["naive_estacional", "media_4sem"])


def metricas_rapidas(bt: pd.DataFrame) -> dict:
    """KPI de la portada: horizontes 1-7 días."""
    vacio = dict(wape=None, wape_naive=None, wape_media4=None, cobertura=None, pinball50=None, n=0)
    if bt is None or bt.empty:
        return vacio
    s = bt[bt["h"] <= 7]
    c = _comparables(s)
    if c.empty or c["real"].sum() <= 0:
        return vacio
    y = c["real"]
    return dict(wape=round(wape(y, c["q50"]), 2), wape_naive=round(wape(y, c["naive_estacional"]), 2),
                wape_media4=round(wape(y, c["media_4sem"]), 2),
                cobertura=round(float(((s["real"] >= s["q10"]) & (s["real"] <= s["q90"])).mean() * 100), 1),
                pinball50=round(pinball(s["real"], s["q50"], 0.5), 2), n=int(len(s)))


def resumen_metricas(bt: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for nombre, (a, b) in HORIZONTES.items():
        s = bt[bt["h"].between(a, b)]
        c = _comparables(s)
        if s.empty or c.empty:
            continue
        y, w_mod, w_nai = c["real"], wape(c["real"], c["q50"]), wape(c["real"], c["naive_estacional"])
        filas.append({
            "Horizonte (días)": nombre, "n": len(s),
            "WAPE modelo %": w_mod, "WAPE naive estacional %": w_nai,
            "WAPE media 4 sem %": wape(y, c["media_4sem"]),
            "Mejora vs naive %": (1 - w_mod / w_nai) * 100 if w_nai else np.nan,
            "Pinball q10": pinball(s["real"], s["q10"], 0.10), "Pinball q50": pinball(s["real"], s["q50"], 0.50),
            "Pinball q90": pinball(s["real"], s["q90"], 0.90),
            "Cobertura P10-P90 % (ideal 80)": ((s["real"] >= s["q10"]) & (s["real"] <= s["q90"])).mean() * 100,
            "Bajo P10 % (ideal 10)": (s["real"] < s["q10"]).mean() * 100,
            "Sobre P90 % (ideal 10)": (s["real"] > s["q90"]).mean() * 100,
        })
    return pd.DataFrame(filas).round(2)


def metricas_por_origen(bt):
    def f(s):
        c = _comparables(s)
        return pd.Series({"WAPE modelo %": wape(c["real"], c["q50"]), "WAPE naive %": wape(c["real"], c["naive_estacional"]),
                          "WAPE media 4 sem %": wape(c["real"], c["media_4sem"]),
                          "Cobertura P10-P90 %": ((s["real"] >= s["q10"]) & (s["real"] <= s["q90"])).mean() * 100})
    return bt.groupby("origen").apply(f, include_groups=False).round(2).reset_index()


def metricas_por_linea(bt, meta_map):
    t = bt.assign(Línea=bt["Ref"].map(lambda r: meta_map[r]["Linea_Oficial"]))
    def f(s):
        c = _comparables(s)
        return pd.Series({"WAPE modelo %": wape(c["real"], c["q50"]), "WAPE naive %": wape(c["real"], c["naive_estacional"]),
                          "WAPE media 4 sem %": wape(c["real"], c["media_4sem"])})
    return t.groupby("Línea").apply(f, include_groups=False).round(2).reset_index()


def ejecutar_ablacion(ctx, bt_full) -> pd.DataFrame:
    """Cuatro niveles de información. Todos con los mismos orígenes y horizontes."""
    base, g = ctx["base_cols"], GRUPOS_CALENDARIO
    niveles = {
        "1. Sin calendario (rezagos + producto)": [],
        "2. + calendario básico (dow, día, mes)": g["basico"],
        "3. + legal/económico (festivos, quincena, prima)": g["basico"] + g["legal_economico"],
        "4. Modelo completo (+ sociocultural)": g["basico"] + g["legal_economico"] + g["sociocultural"],
    }
    filas = []
    for nombre, cols in niveles.items():
        completo = nombre.startswith("4.")
        bt = bt_full if completo else backtest_rolling_origin(ctx, base + cols, cols, cuantiles=(0.50,))
        if bt.empty:
            continue
        fila = {"Variante": nombre, "Nº variables": len(base) + len(cols)}
        for h, (a, b) in HORIZONTES.items():
            s = _comparables(bt[bt["h"].between(a, b)])
            fila[f"WAPE {h} d %"] = wape(s["real"], s["q50"]) if not s.empty else np.nan
        filas.append(fila)
    return pd.DataFrame(filas).round(2)


def calibrar_banda_agregada(bt, min_puntos=15):
    """Residuos relativos de la SUMA de SKU por (origen, día): cuantiles empíricos 10/90 por tramo de horizonte."""
    if bt is None or bt.empty or "q50" not in bt:
        return None
    agg = bt.groupby(["origen", "Fecha", "h"], as_index=False).agg(real=("real", "sum"), p50=("q50", "sum"))
    agg = agg[agg["p50"] > 0]
    agg["res"] = (agg["real"] - agg["p50"]) / agg["p50"]
    out = {}
    for nombre, (a, b) in {"1-7": (1, 7), "8-14": (8, 14)}.items():
        r = agg[agg["h"].between(a, b)]["res"]
        if len(r) >= min_puntos:
            out[nombre] = (float(r.quantile(0.10)), float(r.quantile(0.90)))
    return out or None


# ------------------------------------------------------------------------------
# Pronóstico completo (cacheable: depende solo de los datos)
# ------------------------------------------------------------------------------
def pronosticar(df_diario_in: pd.DataFrame, refs_activas: list[str]) -> dict:
    ctx = construir_panel_features(df_diario_in, refs_activas)
    panel, refs = ctx["panel"], ctx["refs"]
    usar_ml = SKLEARN_DISPONIBLE and ctx["dias_historia"] >= 21 and len(panel) >= 60
    bt = pd.DataFrame()
    motor = "HistGradientBoosting Cuantílico (P10/P25/P50/P75/P90)"

    if usar_ml:
        bt = backtest_rolling_origin(ctx, n_origenes=3, paso=7, horizonte=14, min_train=21, cuantiles=(0.10, 0.50, 0.90))
        modelos = entrenar_cuantiles(panel.dropna(subset=["lag_7"]), ctx["feature_cols"])
        hist = {r: panel[panel["Ref"] == r]["Cant inv"].tolist() for r in refs}
        ml = pronostico_recursivo(modelos, hist, refs, ctx["meta_ids"], ctx["fechas_futuras"],
                                  ctx["cal_lookup"], ctx["cal_cols"], ctx["feature_cols"])
    else:
        motor = "Descomposición estacional (respaldo < 21 días)"
        filas = []
        for r in refs:
            sub = panel[panel["Ref"] == r].assign(dow=lambda d: d["Fecha"].dt.day_of_week)
            prom, std = sub.groupby("dow")["Cant inv"].mean().to_dict(), sub.groupby("dow")["Cant inv"].std().fillna(0).to_dict()
            media = max(1.0, float(sub["Cant inv"].mean()))
            for h, f in enumerate(ctx["fechas_futuras"], start=1):
                c = ctx["cal_lookup"].loc[f]
                p50 = max(1.0, prom.get(f.day_of_week, media) * (1.12 if (c["es_quincena"] or c["vispera_festivo"]) else 1.0))
                d = std.get(f.day_of_week, p50 * 0.12)
                filas.append({"Fecha": f, "Ref": r, "h": h, "Evento_Calendario": str(c["Evento_Calendario"]),
                              "q10": max(0.0, p50 - 1.28 * d), "q25": max(0.0, p50 - 0.67 * d), "q50": p50,
                              "q75": p50 + 0.67 * d, "q90": p50 + 1.28 * d})
        ml = pd.DataFrame(filas)

    return dict(ml=ml, bt=bt, meta_map=ctx["meta_map"], refs=refs, ultima_fecha=ctx["ultima_fecha"],
                dias_historia=ctx["dias_historia"], panel_hist=panel[["Ref", "Fecha", "Cant inv"]],
                metricas=metricas_rapidas(bt), motor=motor, banda=calibrar_banda_agregada(bt))


def auditar(df_diario_in: pd.DataFrame, refs_activas: list[str]) -> dict:
    """Backtesting completo (4 orígenes × 30 días), ablación en 4 niveles y métricas por origen/línea."""
    ctx = construir_panel_features(df_diario_in, refs_activas)
    bt = backtest_rolling_origin(ctx, n_origenes=4, paso=7, horizonte=30, min_train=45)
    if bt.empty:
        return {}
    return dict(bt=bt, resumen=resumen_metricas(bt), por_origen=metricas_por_origen(bt),
                por_linea=metricas_por_linea(bt, ctx["meta_map"]), ablacion=ejecutar_ablacion(ctx, bt),
                meta_map=ctx["meta_map"])


# ------------------------------------------------------------------------------
# Decisión: newsvendor (vida corta) / nivel de servicio objetivo (vida larga)
# ------------------------------------------------------------------------------
def razon_critica(precio, costo, salvamento=0.0, penal=0.0):
    cu = max(1.0, precio - costo + penal)   # costo de faltante
    co = max(1.0, costo - salvamento)       # costo de sobrante (vencimiento)
    return cu / (cu + co), cu, co


def construir_costos(meta_map, costo_var_pct, penal_pct, override=None) -> dict:
    """costo_var_pct es un SUPUESTO salvo que se cargue 'override' con costos reales por SKU."""
    out = {}
    for r, m in meta_map.items():
        precio, costo, fuente = float(m.get("Precio_Cat", 3500)), None, "supuesto"
        costo = precio * costo_var_pct
        k = str(r).strip().upper()
        if override and k in override:
            precio = float(override[k].get("precio") or precio)
            costo = float(override[k].get("costo", costo))
            fuente = "real"
        out[r] = {"precio": precio, "costo": costo, "penal": precio * penal_pct, "fuente": fuente}
    return out


def interp_cuantil(Q: np.ndarray, niveles: np.ndarray) -> np.ndarray:
    """Interpolación vectorizada de la función cuantil entre q10..q90 (se acota a [0,10; 0,90])."""
    xs = np.array(CUANTILES)
    lv = np.clip(niveles, xs[0], xs[-1])
    idx = np.clip(np.searchsorted(xs, lv, side="right") - 1, 0, len(xs) - 2)
    w = (lv - xs[idx]) / (xs[idx + 1] - xs[idx])
    ar = np.arange(len(lv))
    return Q[ar, idx] + w * (Q[ar, idx + 1] - Q[ar, idx])


def nivel_objetivo(vida, costos_ref, postura, nivel_larga):
    """Nivel de servicio por SKU según postura y vida útil."""
    if postura == "Conservador (P50)":
        return 0.50
    if postura == "Expansivo / Captura (P90)":
        return 0.90
    if vida > UMBRAL_VIDA_CORTA:
        return nivel_larga
    return razon_critica(costos_ref["precio"], costos_ref["costo"], 0.0, costos_ref["penal"])[0]


def a_cajas(q, emb, min_cajas) -> int:
    return int(max(np.ceil(max(q, 0.0) / emb), min_cajas)) if q > 0 else 0


def decidir(res: dict, df_param: pd.DataFrame, costos: dict, postura: str, horizonte_dias: int,
            nivel_larga: float = 0.90, choque_pct: float = 0.0):
    """Convierte el pronóstico en órdenes. Rápido (sin reentrenar): se puede llamar al mover cualquier control.
    La orden de 'mañana' cubre el CICLO de reposición del SKU (7 / frecuencia de cargue, redondeado),
    sumando cuantiles diarios (supuesto conservador de correlación perfecta entre días)."""
    ml = res["ml"].copy()
    if choque_pct:
        ml[QCOLS] = ml[QCOLS] * (1 + choque_pct / 100.0)
    dias_nombre = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves", 4: "Viernes", 5: "Sábado", 6: "Domingo"}
    param = df_param.set_index("Ref").to_dict("index")
    hist = res["panel_hist"]
    ultima = res["ultima_fecha"]
    proy, manana = [], []

    for r in res["refs"]:
        m = res["meta_map"][r]
        p = param.get(r, {})
        emb = max(1, int(p.get("Unidades por Caja", 20)))
        min_caj = max(1, int(p.get("Mínimo a Producir (Cajas)", 5)))
        frec = max(1, int(p.get("Frecuencia de cargue (Días/Semana)", 6)))
        ciclo = max(1, int(round(7 / frec)))
        vida = int(m["Vida_Util_Dias"])
        niv = nivel_objetivo(vida, costos[r], postura, nivel_larga)

        sub = ml[ml["Ref"] == r].sort_values("Fecha").reset_index(drop=True)
        sub = sub.assign(q_obj=interp_cuantil(sub[QCOLS].to_numpy(), np.full(len(sub), niv)))
        h_ref = hist[hist["Ref"] == r].set_index("Fecha")["Cant inv"]
        media = max(1.0, float(h_ref.mean()))
        p50_fut = sub.set_index("Fecha")["q50"].to_dict()

        def referencia(f):
            g = f - pd.Timedelta(days=7)
            if g <= ultima:
                v = float(h_ref.get(g, 0.0))
                return v if v > 0 else media
            return max(1.0, float(p50_fut.get(g, media)))

        for _, row in sub[sub["h"] <= horizonte_dias].iterrows():
            f = row["Fecha"]
            cajas = a_cajas(row["q_obj"], emb, min_caj)
            ref_v = referencia(f)
            proy.append({"Fecha": f, "Fecha Proyectada": f.strftime("%Y-%m-%d"), "Día": dias_nombre[f.day_of_week],
                         "Evento Calendario": row["Evento_Calendario"], "Ref": r, "Producto": m["Desc item"],
                         "Línea": m["Linea_Oficial"], "Pronóstico P50": int(round(row["q50"])),
                         "P10": int(round(row["q10"])), "P90": int(round(row["q90"])),
                         "Cuantil Objetivo": round(row["q_obj"], 1), "Nivel servicio": round(niv, 3),
                         "Producir (Un)": cajas * emb, "Sugerido Producir (Cajas)": cajas,
                         "Var %": round((row["q50"] - ref_v) / ref_v * 100, 1)})

        ini = sub.iloc[:ciclo]
        q_ciclo, p50c, p10c, p90c = ini["q_obj"].sum(), ini["q50"].sum(), ini["q10"].sum(), ini["q90"].sum()
        ref_c = sum(referencia(f) for f in ini["Fecha"])
        var_pct = (p50c - ref_c) / ref_c * 100 if ref_c > 0 else 0.0
        cajas_c = a_cajas(q_ciclo, emb, min_caj)
        # Estado: depende SOLO de los datos (no de la postura elegida)
        if var_pct >= 8.0 or (vida <= 5 and ini["q50"].iloc[0] >= media * 1.05):
            estado = "Riesgo de quiebre"
        elif var_pct >= 3.5:
            estado = "Subir producción"
        else:
            estado = "Estable"
        fmt = lambda v: f"{int(round(v)):,}".replace(",", ".")
        manana.append({"Ref": r, "Producto": m["Desc item"], "Línea": m["Linea_Oficial"], "Presentación": m["Presentacion"],
                       "Canales": m["Canales"], "Vida Útil": f"{vida} días", "Vida_Util_num": vida,
                       "Evento Mañana": ini["Evento_Calendario"].iloc[0], "Ciclo (días)": ciclo,
                       "Pronóstico_num": int(round(p50c)), "Pronóstico": f"{fmt(p50c)} und",
                       "P10": int(round(p10c)), "P90": int(round(p90c)),
                       "Rango P10–P90": f"{fmt(p10c)} – {fmt(p90c)}",
                       "Nivel_Servicio_Opt": round(niv * 100, 1), "Producir": cajas_c * emb, "Cajas": cajas_c,
                       "Estado": estado, "Var %": round(var_pct, 1),
                       "Regla": "Newsvendor" if (vida <= UMBRAL_VIDA_CORTA and postura.startswith("Óptimo")) else
                                ("Nivel de servicio objetivo" if postura.startswith("Óptimo") else postura)})

    df_proy, df_man = pd.DataFrame(proy), pd.DataFrame(manana)
    if not df_man.empty:
        prio = {"Riesgo de quiebre": 0, "Subir producción": 1, "Estable": 2}
        df_man = (df_man.assign(_p=df_man["Estado"].map(prio)).sort_values(["_p", "Pronóstico_num"], ascending=[True, False])
                  .drop(columns="_p"))
    return df_proy, df_man


# ------------------------------------------------------------------------------
# Evaluación económica de políticas y sensibilidad (sobre el backtest completo)
# ------------------------------------------------------------------------------
def _buffer_fijo(v):
    return np.where(v <= 6, 1.05, np.where(v <= 12, 1.08, 1.12))


def _arrays(bt, meta_map, df_param):
    param = df_param.set_index("Ref").to_dict("index")
    d = bt.dropna(subset=["naive_estacional"]).reset_index(drop=True)
    emb = d["Ref"].map(lambda r: max(1, int(param.get(r, {}).get("Unidades por Caja", 20)))).to_numpy()
    minc = d["Ref"].map(lambda r: max(1, int(param.get(r, {}).get("Mínimo a Producir (Cajas)", 5)))).to_numpy()
    vida = d["Ref"].map(lambda r: int(meta_map[r]["Vida_Util_Dias"])).to_numpy()
    precio = d["Ref"].map(lambda r: float(meta_map[r].get("Precio_Cat", 3500))).to_numpy()
    return d, emb, minc, vida, precio


def _a_unidades(Q, emb, minc):
    return np.where(Q > 0, np.maximum(np.ceil(np.maximum(Q, 0) / emb), minc), 0) * emb


def _politicas_Q(d, vida, cr, nivel_larga):
    Qm = d[QCOLS].to_numpy()
    niv = np.where(vida <= UMBRAL_VIDA_CORTA, cr, nivel_larga)
    return {"Newsvendor / nivel objetivo": interp_cuantil(Qm, niv), "P50 puro": d["q50"].to_numpy(),
            "P50 × buffer fijo (5/8/12 %)": d["q50"].to_numpy() * _buffer_fijo(vida),
            "P90": d["q90"].to_numpy(), "Naive estacional": d["naive_estacional"].to_numpy()}


def evaluar_politicas(bt, meta_map, df_param, costos, nivel_larga=0.90):
    """Devuelve (vida corta: costo/servicio/merma, vida larga: servicio/excedente).
    En vida larga el sobrante no es merma (se vende después), por eso NO se calcula costo allí."""
    d, emb, minc, vida, precio = _arrays(bt, meta_map, df_param)
    cu = np.array([max(1.0, precio[i] - costos[r]["costo"] + costos[r]["penal"]) for i, r in enumerate(d["Ref"])])
    co = np.array([max(1.0, costos[r]["costo"]) for r in d["Ref"]])
    cr = cu / (cu + co)
    D = d["real"].to_numpy()
    corta, larga = [], []
    for nombre, Q in _politicas_Q(d, vida, cr, nivel_larga).items():
        Qu = _a_unidades(Q, emb, minc)
        falt, sobr = np.maximum(D - Qu, 0), np.maximum(Qu - D, 0)
        for mask, dest, con_costo in ((vida <= UMBRAL_VIDA_CORTA, corta, True), (vida > UMBRAL_VIDA_CORTA, larga, False)):
            if mask.sum() == 0:
                continue
            fila = {"Política": nombre, "Nivel de servicio %": round((1 - falt[mask].sum() / D[mask].sum()) * 100, 2),
                    ("Merma %" if con_costo else "Excedente %"): round(sobr[mask].sum() / Qu[mask].sum() * 100, 2)}
            if con_costo:
                fila = {"Política": nombre, "Costo total desbalance (COP)": int(round((cu[mask] * falt[mask] + co[mask] * sobr[mask]).sum())), **{k: v for k, v in fila.items() if k != "Política"}}
            dest.append(fila)
    df_c = pd.DataFrame(corta)
    if not df_c.empty:
        df_c = df_c.sort_values("Costo total desbalance (COP)").reset_index(drop=True)
    return df_c, pd.DataFrame(larga)


def sensibilidad_costos(bt, meta_map, df_param, grid_costo=(0.40, 0.50, 0.60, 0.70, 0.80),
                        grid_penal=(0.0, 0.15, 0.30, 0.50)) -> pd.DataFrame:
    """Ahorro del newsvendor frente a P50 puro y al buffer fijo, para una cuadrícula de supuestos (solo vida corta)."""
    d, emb, minc, vida, precio = _arrays(bt, meta_map, df_param)
    mk = vida <= UMBRAL_VIDA_CORTA
    if mk.sum() == 0:
        return pd.DataFrame()
    d, emb, minc, vida, precio = d[mk].reset_index(drop=True), emb[mk], minc[mk], vida[mk], precio[mk]
    D = d["real"].to_numpy()
    filas = []
    for c in grid_costo:
        for p in grid_penal:
            cu, co = np.maximum(1.0, precio * (1 - c + p)), np.maximum(1.0, precio * c)
            cr = cu / (cu + co)
            costos_pol = {}
            for nombre, Q in _politicas_Q(d, vida, cr, 0.90).items():
                Qu = _a_unidades(Q, emb, minc)
                costos_pol[nombre] = float((cu * np.maximum(D - Qu, 0) + co * np.maximum(Qu - D, 0)).sum())
            nv = costos_pol["Newsvendor / nivel objetivo"]
            filas.append({"Costo var. %": int(c * 100), "Penal. quiebre %": int(p * 100), "CR* medio": round(float(cr.mean()), 3),
                          "Ahorro vs P50 %": round((1 - nv / costos_pol["P50 puro"]) * 100, 2),
                          "Ahorro vs buffer fijo %": round((1 - nv / costos_pol["P50 × buffer fijo (5/8/12 %)"]) * 100, 2),
                          "Mejor política": min(costos_pol, key=costos_pol.get)})
    return pd.DataFrame(filas)
