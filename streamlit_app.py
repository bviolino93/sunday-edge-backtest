"""
Sunday Edge — Backtest

Answers one question: does an opponent-adjusted power rating add anything
to the NFL closing line? If it doesn't, there is no app worth building.

Schedules only — no play-by-play. That keeps memory well under the
Streamlit Community Cloud limit and the whole run to a few seconds.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import Ridge

import nfl_data_py as nfl

st.set_page_config(page_title="Sunday Edge — Backtest", layout="wide")


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_schedules(first, last):
    df = nfl.import_schedules(list(range(first, last + 1)))
    keep = ["game_id", "season", "week", "home_team", "away_team",
            "home_score", "away_score", "spread_line", "total_line"]
    return df[[c for c in keep if c in df.columns]].copy()


# ----------------------------------------------------------------------
# Ratings
# ----------------------------------------------------------------------
def _design(hist, teams, symmetric):
    """Team dummies plus a constant. symmetric=True for totals (both teams
    add points), False for margin (home minus away)."""
    idx = {t: i for i, t in enumerate(teams)}
    X = np.zeros((len(hist), len(teams) + 1))
    h = hist["home_team"].values
    a = hist["away_team"].values
    for r in range(len(hist)):
        if h[r] in idx:
            X[r, idx[h[r]]] = 1.0
        if a[r] in idx:
            X[r, idx[a[r]]] = 1.0 if symmetric else -1.0
        X[r, -1] = 1.0
    return X, idx


def fit_ratings(hist, teams, target, alpha, symmetric=False):
    """
    Ridge, not OLS. With 32 teams and a short window the design matrix is
    near-singular and unshrunk ratings chase early-season noise.
    """
    if len(hist) < 40:
        return None, None
    X, idx = _design(hist, teams, symmetric)
    m = Ridge(alpha=alpha, fit_intercept=False).fit(X, hist[target].values)
    return {t: m.coef_[idx[t]] for t in teams}, float(m.coef_[-1])


# ----------------------------------------------------------------------
# Walk-forward backtest
# ----------------------------------------------------------------------
def backtest(sched, min_week, window_games, carryover, alpha, progress=None):
    g = sched.dropna(subset=["home_score", "away_score", "spread_line"]).copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]
    g = g.sort_values(["season", "week"]).reset_index(drop=True)

    # Verify the line convention instead of assuming it. A flipped sign
    # would silently invert every result below.
    corr = float(np.corrcoef(g["spread_line"], g["home_margin"])[0, 1])
    sign = 1.0 if corr > 0 else -1.0
    g["mkt_margin"] = sign * g["spread_line"]

    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    seasons = sorted(g["season"].unique())
    rows = []

    for si, season in enumerate(seasons):
        for wk in sorted(g[g["season"] == season]["week"].unique()):
            if wk < min_week:
                continue

            # Strictly prior games. This is what makes it walk-forward:
            # nothing here has seen the game it is predicting.
            prior = g[(g["season"] < season)
                      | ((g["season"] == season) & (g["week"] < wk))]
            if len(prior) < 40:
                continue
            recent = prior.tail(window_games)
            in_season = prior[prior["season"] == season]

            r_all, hfa = fit_ratings(recent, teams, "home_margin", alpha)
            if r_all is None:
                continue
            t_all, tbase = fit_ratings(recent, teams, "total_points",
                                       alpha, symmetric=True)

            # Early in a season the in-season sample is thin, so lean on
            # last year's ratings and let the blend shift as games arrive.
            r_cur, _ = fit_ratings(in_season, teams, "home_margin", alpha)
            if r_cur is not None:
                w = min(1.0, len(in_season) / 160.0)
                blend = {t: w * r_cur.get(t, 0.0)
                            + (1 - w) * carryover * r_all.get(t, 0.0)
                         for t in teams}
            else:
                blend = {t: carryover * r_all.get(t, 0.0) for t in teams}

            for _, row in g[(g["season"] == season) & (g["week"] == wk)].iterrows():
                h, a = row["home_team"], row["away_team"]
                rows.append({
                    "season": season, "week": wk,
                    "matchup": f"{a} @ {h}",
                    "actual_margin": row["home_margin"],
                    "actual_total": row["total_points"],
                    "mkt_margin": row["mkt_margin"],
                    "mkt_total": row.get("total_line", np.nan),
                    "pred_margin": blend.get(h, 0.0) - blend.get(a, 0.0) + hfa,
                    "pred_total": (t_all.get(h, 0.0) + t_all.get(a, 0.0) + tbase)
                                   if t_all else np.nan,
                })
        if progress:
            progress((si + 1) / len(seasons), f"Season {season}")

    return pd.DataFrame(rows), corr, sign


# ----------------------------------------------------------------------
# The test
# ----------------------------------------------------------------------
def incremental_test(y, market, model):
    """
    Regress the outcome on BOTH the closing line and the model.
    If the market already prices what the model knows, the model's
    coefficient collapses toward zero. No threshold, no selection bias,
    every game counted.
    """
    X = np.column_stack([np.ones(len(y)), market, model])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (resid @ resid / dof)))
    return beta, beta[2] / se[2]


def ats_table(r):
    r = r.copy()
    r["edge"] = r.pred_margin - r.mkt_margin
    r["cover"] = np.where(r.edge > 0,
                          r.actual_margin > r.mkt_margin,
                          r.actual_margin < r.mkt_margin)
    out = []
    for thr in [0.5, 1, 2, 3, 4, 6, 8]:
        s = r[(r.edge.abs() >= thr) & (r.actual_margin != r.mkt_margin)]
        if len(s) < 30:
            continue
        w = int(s.cover.sum()); l = len(s) - w
        out.append({
            "Edge threshold": f"{thr:g} pts",
            "Bets": len(s),
            "Record": f"{w}-{l}",
            "Win %": f"{w/len(s):.1%}",
            "ROI": f"{(w * (100/110) - l) / len(s):+.1%}",
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Sunday Edge — Backtest")
st.write(
    "Tests whether an opponent-adjusted power rating adds anything to the "
    "NFL closing line. Every prediction uses only games played before it."
)

with st.sidebar:
    st.header("Settings")
    yr = st.slider("Seasons", 2007, 2025, (2007, 2025))
    min_week = st.slider("Start predicting at week", 2, 8, 4,
                         help="Early weeks have almost no in-season data.")
    window = st.slider("Rolling window (games)", 160, 800, 320, step=40,
                       help="How much history the ratings are fit on.")
    carry = st.slider("Prior-season carryover", 0.0, 1.0, 0.35, 0.05)
    alpha = st.slider("Ridge shrinkage", 1.0, 30.0, 8.0, 1.0,
                      help="Higher pulls team ratings toward average.")
    run = st.button("Run backtest", type="primary", use_container_width=True)

if not run:
    st.info("Set the seasons on the left, then run the backtest.")
    st.stop()

bar = st.progress(0.0, "Loading schedules")
try:
    sched = load_schedules(yr[0], yr[1])
except Exception as e:
    bar.empty()
    st.error(f"Could not load schedule data: {e}")
    st.stop()

res, corr, sign = backtest(
    sched, min_week, window, carry, alpha,
    progress=lambda f, m: bar.progress(f, m),
)
bar.empty()

r = res.dropna(subset=["pred_margin", "mkt_margin", "actual_margin"])
if len(r) < 200:
    st.error("Not enough graded games. Widen the season range.")
    st.stop()

beta, t_model = incremental_test(
    r.actual_margin.values, r.mkt_margin.values, r.pred_margin.values
)

# --- verdict ---------------------------------------------------------
st.header("Does the model beat the line?")
if abs(t_model) < 2:
    st.error(
        f"**No.** The model adds nothing to the closing line "
        f"(coefficient {beta[2]:+.3f}, t = {t_model:+.2f}). "
        f"The market already prices what these ratings know."
    )
elif beta[2] > 0:
    st.success(
        f"**Some signal.** The model adds information beyond the closing "
        f"line (coefficient {beta[2]:+.3f}, t = {t_model:+.2f})."
    )
else:
    st.warning(
        f"**Anti-predictive** (coefficient {beta[2]:+.3f}, t = {t_model:+.2f}). "
        f"Check for a sign error before reading anything into it."
    )

st.caption(
    f"Regression of actual margin on the closing line and the model "
    f"prediction, {len(r):,} games, {r.season.min()}–{r.season.max()}. "
    f"Closing-line coefficient {beta[1]:+.3f}, where 1.00 means the market "
    f"is perfectly calibrated. A model coefficient near zero means the "
    f"market has already priced everything the model sees."
)

# --- accuracy --------------------------------------------------------
st.subheader("Prediction accuracy")
mae_model = float((r.pred_margin - r.actual_margin).abs().mean())
mae_mkt = float((r.mkt_margin - r.actual_margin).abs().mean())
c1, c2, c3 = st.columns(3)
c1.metric("Model error", f"{mae_model:.2f} pts")
c2.metric("Market error", f"{mae_mkt:.2f} pts")
c3.metric("Market advantage", f"{mae_model - mae_mkt:+.2f} pts")
st.caption(
    "The market normally wins here — it has injury news and sharp money the "
    "ratings don't. Losing on raw accuracy is not disqualifying. Edge lives "
    "in what's left over, which is what the test above measures."
)

# --- ATS -------------------------------------------------------------
st.subheader("If you had bet the disagreements")
tbl = ats_table(r)
if not tbl.empty:
    st.dataframe(tbl, hide_index=True, use_container_width=True)
st.caption(
    "Breakeven at -110 is 52.4%. Seven thresholds are shown, so one of them "
    "clearing the bar is expected even from noise. Trust the test above, "
    "not this table."
)

# --- totals ----------------------------------------------------------
t = res.dropna(subset=["pred_total", "mkt_total", "actual_total"])
if len(t) > 200:
    bt, t_tot = incremental_test(
        t.actual_total.values, t.mkt_total.values, t.pred_total.values
    )
    st.subheader("Totals")
    st.write(
        f"Model coefficient {bt[2]:+.3f} (t = {t_tot:+.2f}) across "
        f"{len(t):,} games — "
        + ("no edge." if abs(t_tot) < 2 else "some signal.")
    )

# --- data ------------------------------------------------------------
with st.expander("Game-by-game results"):
    st.dataframe(res, hide_index=True, use_container_width=True)
st.download_button(
    "Download results as CSV",
    res.to_csv(index=False).encode(),
    file_name="nfl_backtest_results.csv",
    mime="text/csv",
)
st.caption(
    f"Line convention check: correlation between spread_line and home "
    f"margin is {corr:+.3f}, so the sign was set to {sign:+.0f}."
)
