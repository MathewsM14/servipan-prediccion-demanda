"""Gráfico agregado con banda calibrada empíricamente (no suma de cuantiles)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def banda_agregada(df_proy: pd.DataFrame, banda_cal: dict | None) -> tuple[pd.DataFrame, str]:
    """Banda P10-P90 de la SUMA de SKU.
    - Con backtest suficiente: cuantiles empíricos de los residuos relativos agregados (captura la correlación entre SKU).
    - Sin backtest: aproximación por suma de varianzas (asume independencia -> banda demasiado angosta)."""
    agg = df_proy.groupby("Fecha", as_index=False).agg(p50=("Pronóstico P50", "sum"), p10=("P10", "sum"), p90=("P90", "sum"))
    agg = agg.sort_values("Fecha").reset_index(drop=True)
    if banda_cal:
        def lim(i):
            clave = "1-7" if i < 7 else "8-14"
            lo, hi = banda_cal.get(clave) or next(iter(banda_cal.values()))
            return lo, hi
        r = [lim(i) for i in range(len(agg))]
        agg["lo"] = np.maximum(0.0, agg["p50"] * (1 + np.array([x[0] for x in r])))
        agg["hi"] = agg["p50"] * (1 + np.array([x[1] for x in r]))
        return agg, "Banda P10–P90 (residuos agregados del backtest)"
    sigma = np.sqrt(((df_proy["P90"] - df_proy["P10"]) / (2 * 1.28155)) ** 2)
    var = (df_proy.assign(v=sigma ** 2).groupby("Fecha")["v"].sum().sort_index().to_numpy())
    agg["lo"] = np.maximum(0.0, agg["p50"] - 1.28155 * np.sqrt(var))
    agg["hi"] = agg["p50"] + 1.28155 * np.sqrt(var)
    return agg, "Banda aproximada (no calibrada)"


def construir_grafico_tendencia(df_diario_total, df_proy, ultima_fecha, horizonte_dias, banda_cal=None) -> go.Figure:
    ini = ultima_fecha - pd.Timedelta(days=14)
    hist = df_diario_total[df_diario_total["Fecha"] >= ini].sort_values("Fecha")
    ultimo = float(hist["Cant inv"].iloc[-1]) if not hist.empty else 0.0
    agg, etiqueta = banda_agregada(df_proy, banda_cal)
    fx = [ultima_fecha] + agg["Fecha"].tolist()
    p50, lo, hi = [ultimo] + agg["p50"].tolist(), [ultimo] + agg["lo"].tolist(), [ultimo] + agg["hi"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fx, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(245, 214, 107, 0.34)", name=etiqueta,
                             hovertemplate="Banda: %{y:,.0f} und<extra></extra>"))
    fig.add_trace(go.Scatter(x=hist["Fecha"], y=hist["Cant inv"], mode="lines", name="Observada",
                             line=dict(color="#6B2E12", width=2.5, shape="spline", smoothing=0.45),
                             hovertemplate="%{x|%d %b}: <b>%{y:,.0f} und</b><extra>Observada</extra>"))
    fig.add_trace(go.Scatter(x=fx, y=p50, mode="lines", name="Mediana P50",
                             line=dict(color="#D97724", width=2.5, dash="dash", shape="spline", smoothing=0.45),
                             hovertemplate="%{x|%d %b}: <b>%{y:,.0f} und</b><extra>P50</extra>"))
    fig.add_vline(x=ultima_fecha, line_width=1.4, line_dash="dot", line_color="#9A5B25")
    fin = agg["Fecha"].max() if not agg.empty else ultima_fecha
    fig.update_layout(
        height=238, margin=dict(l=8, r=12, t=12, b=26), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False, hovermode="x unified",
        xaxis=dict(tickmode="array", tickvals=[ini, ultima_fecha, fin], ticktext=["−14 días", "Hoy", f"+{horizonte_dias} días"],
                   tickfont=dict(size=11, color="#7C6350"), showgrid=False, zeroline=False, linecolor="#EADBC8"),
        yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#F5EDE3", zeroline=False))
    return fig
