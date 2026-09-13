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
    keep = ["game_id", "season", "week", "gameday", "home_team", "away_team",
            "home_score", "away_score", "spread_line", "total_line",
            "home_rest", "away_rest", "div_game", "roof", "surface",
            "temp", "wind", "home_qb_name", "away_qb_name"]
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
def weight_fn(scheme, param):
    """How much to trust in-season data given n games played."""
    if scheme == "linear":
        return lambda n: min(1.0, n / param)
    if scheme == "bayes":
        # n/(n+k): approaches 1 but never discards the prior entirely.
        return lambda n: n / (n + param)
    if scheme == "sqrt":
        return lambda n: min(1.0, (n / param) ** 0.5)
    return lambda n: min(1.0, n / 160.0)


def backtest(sched, min_week, window_games, carryover, alpha, progress=None,
             wfn=None, skip_totals=False):
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
            if skip_totals:
                t_all, tbase = None, 0.0
            else:
                t_all, tbase = fit_ratings(recent, teams, "total_points",
                                           alpha, symmetric=True)

            # Early in a season the in-season sample is thin, so lean on
            # last year's ratings and let the blend shift as games arrive.
            r_cur, _ = fit_ratings(in_season, teams, "home_margin", alpha)
            if r_cur is not None:
                w = (wfn or (lambda n: min(1.0, n / 160.0)))(len(in_season))
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
# Signal lab
# ----------------------------------------------------------------------
def ols_t(y, x):
    """Regress y on x plus a constant. Returns slope, t-stat, n."""
    ok = np.isfinite(x) & np.isfinite(y)
    y, x = y[ok], x[ok]
    if len(y) < 100:
        return None
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    try:
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (r @ r / (len(y) - 2))))
    except Exception:
        return None
    return float(beta[1]), float(beta[1] / se[1]), int(len(y))


def qb_change_flags(g):
    """1 where a team's starting QB differs from its previous game."""
    last = {}
    out = []
    for _, row in g.sort_values(["season", "week"]).iterrows():
        f = {}
        for side in ("home", "away"):
            team = row[f"{side}_team"]
            qb = row.get(f"{side}_qb_name")
            prev = last.get(team)
            f[side] = 1.0 if (prev and qb and prev != qb) else 0.0
            if qb:
                last[team] = qb
        out.append(f)
    d = pd.DataFrame(out, index=g.sort_values(["season", "week"]).index)
    return d.reindex(g.index)


def run_signal_lab(sched, sign):
    g = sched.dropna(subset=["home_score", "away_score", "spread_line"]).copy()
    g["mkt_margin"] = sign * g["spread_line"]
    g["home_margin"] = g["home_score"] - g["away_score"]
    g["total_points"] = g["home_score"] + g["away_score"]

    # Residual = what the closing line MISSED. If a feature predicts the
    # residual, the market is underpricing that feature. That is an edge;
    # predicting the outcome itself is not.
    g["resid_margin"] = g["home_margin"] - g["mkt_margin"]
    g["resid_total"] = g["total_points"] - pd.to_numeric(
        g.get("total_line"), errors="coerce")

    for c in ["home_rest", "away_rest", "div_game", "temp", "wind"]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")

    g["rest_diff"] = g.get("home_rest", np.nan) - g.get("away_rest", np.nan)
    g["roof_str"] = g.get("roof", "").astype(str).str.lower()
    g["is_outdoor"] = g["roof_str"].isin(["outdoors", "open"]).astype(float)
    g["fav_size"] = g["mkt_margin"].abs()

    qb = qb_change_flags(g)
    g["qb_change_home"] = qb["home"]
    g["qb_change_away"] = qb["away"]
    g["qb_change_diff"] = g["qb_change_home"] - g["qb_change_away"]
    g["qb_change_any"] = ((g["qb_change_home"] + g["qb_change_away"]) > 0).astype(float)

    outdoor = g[g["is_outdoor"] == 1]

    tests = [
        ("Rest advantage (home minus away days)", "Spread",
         g["resid_margin"].values, g["rest_diff"].values,
         "Does extra rest beat the number?"),
        ("Divisional game", "Spread",
         g["resid_margin"].values, g["div_game"].values,
         "Do division games play closer than priced?"),
        ("Favorite size", "Spread",
         g["resid_margin"].values, g["fav_size"].values,
         "Are big favorites over- or under-priced?"),
        ("QB change (home minus away)", "Spread",
         g["resid_margin"].values, g["qb_change_diff"].values,
         "Does the market misprice a new starting QB?"),
        ("Wind speed, outdoor only", "Total",
         outdoor["resid_total"].values, outdoor["wind"].values,
         "The classic candidate: markets are said to underreact to wind."),
        ("Temperature, outdoor only", "Total",
         outdoor["resid_total"].values, outdoor["temp"].values,
         "Cold-weather unders."),
        ("Indoor game", "Total",
         g["resid_total"].values, (1 - g["is_outdoor"]).values,
         "Domes and scoring."),
        ("Divisional game", "Total",
         g["resid_total"].values, g["div_game"].values, ""),
        ("QB change, either team", "Total",
         g["resid_total"].values, g["qb_change_any"].values, ""),
        ("Total line level", "Total",
         g["resid_total"].values,
         pd.to_numeric(g.get("total_line"), errors="coerce").values,
         "Are high totals systematically too high?"),
    ]

    rows = []
    for name, market, y, x, note in tests:
        r = ols_t(np.asarray(y, dtype=float), np.asarray(x, dtype=float))
        if r is None:
            continue
        slope, t, n = r
        rows.append({"Signal": name, "Market": market, "Games": n,
                     "Effect per unit": round(slope, 3), "t": round(t, 2),
                     "Verdict": "SIGNAL" if abs(t) > 2.5 else "nothing",
                     "_note": note})
    return pd.DataFrame(rows).sort_values("t", key=lambda s: s.abs(),
                                          ascending=False)


# ----------------------------------------------------------------------
# EPA model — the NFL analogue of SP+/PPA
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_team_game_epa(first, last):
    """
    Offensive EPA per play for each team in each game, plus what they allowed.
    Only the columns needed, so 19 seasons of play-by-play stays inside the
    Community Cloud memory limit.
    """
    cols = ["game_id", "season", "week", "posteam", "defteam", "epa",
            "play_type"]
    pbp = nfl.import_pbp_data(list(range(first, last + 1)), columns=cols,
                              downcast=True, cache=False)
    pbp = pbp[pbp["play_type"].isin(["run", "pass"])
              & pbp["epa"].notna() & pbp["posteam"].notna()]
    g = (pbp.groupby(["game_id", "season", "week", "posteam", "defteam"])["epa"]
         .agg(["mean", "count"]).reset_index()
         .rename(columns={"posteam": "team", "defteam": "opp",
                          "mean": "off_epa", "count": "plays"}))
    return g[g["plays"] >= 20].copy()


def fit_epa_ratings(hist, teams, alpha):
    """
    Opponent-adjusted offence and defence. Each team-game contributes its
    offensive EPA, explained by that offence and the defence it faced, so a
    good number against a good defence counts for more. This is what scoring
    margin cannot see: a team can move the ball all day and lose on turnovers.
    """
    if len(hist) < 80:
        return None
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(hist), 2 * n + 1))
    tm = hist["team"].values
    op = hist["opp"].values
    for r in range(len(hist)):
        if tm[r] in idx:
            X[r, idx[tm[r]]] = 1.0            # offence
        if op[r] in idx:
            X[r, n + idx[op[r]]] = 1.0        # defence faced
        X[r, -1] = 1.0
    m = Ridge(alpha=alpha, fit_intercept=False).fit(X, hist["off_epa"].values)
    off = {t: m.coef_[idx[t]] for t in teams}
    dfn = {t: m.coef_[n + idx[t]] for t in teams}
    return {"off": off, "def": dfn, "base": float(m.coef_[-1])}


def backtest_epa(sched, epa, min_week, window_games, alpha, progress=None):
    g = sched.dropna(subset=["home_score", "away_score", "spread_line"]).copy()
    g["home_margin"] = g["home_score"] - g["away_score"]
    g = g.sort_values(["season", "week"]).reset_index(drop=True)
    corr = float(np.corrcoef(g["spread_line"], g["home_margin"])[0, 1])
    sign = 1.0 if corr > 0 else -1.0
    g["mkt_margin"] = sign * g["spread_line"]

    epa = epa.sort_values(["season", "week"]).reset_index(drop=True)
    teams = sorted(set(epa["team"]) | set(epa["opp"]))
    seasons = sorted(g["season"].unique())
    rows = []

    for si, season in enumerate(seasons):
        for wk in sorted(g[(g["season"] == season)]["week"].unique()):
            if wk < min_week:
                continue
            prior_g = g[(g["season"] < season)
                        | ((g["season"] == season) & (g["week"] < wk))]
            prior_e = epa[(epa["season"] < season)
                          | ((epa["season"] == season) & (epa["week"] < wk))]
            if len(prior_g) < 60 or len(prior_e) < 200:
                continue
            rt = fit_epa_ratings(prior_e.tail(window_games * 2), teams, alpha)
            if rt is None:
                continue

            # Convert an EPA edge into points, using only prior games.
            tr = prior_g.tail(window_games)
            feats, targ = [], []
            for _, r in tr.iterrows():
                h, a = r["home_team"], r["away_team"]
                if h not in rt["off"] or a not in rt["off"]:
                    continue
                feats.append([(rt["off"][h] - rt["def"][a])
                              - (rt["off"][a] - rt["def"][h]), 1.0])
                targ.append(r["home_margin"])
            if len(feats) < 60:
                continue
            beta, *_ = np.linalg.lstsq(np.array(feats), np.array(targ),
                                       rcond=None)

            for _, r in g[(g["season"] == season) & (g["week"] == wk)].iterrows():
                h, a = r["home_team"], r["away_team"]
                if h not in rt["off"] or a not in rt["off"]:
                    continue
                d = (rt["off"][h] - rt["def"][a]) - (rt["off"][a] - rt["def"][h])
                rows.append({
                    "season": season, "week": wk,
                    "actual_margin": r["home_margin"],
                    "mkt_margin": r["mkt_margin"],
                    "pred_margin": float(beta[0] * d + beta[1]),
                    "epa_edge": float(d),
                })
        if progress:
            progress((si + 1) / len(seasons), f"season {season}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Robustness sweep
# ----------------------------------------------------------------------
GRID = [
    ("linear", 160.0), ("linear", 100.0), ("linear", 240.0),
    ("bayes", 20.0), ("bayes", 40.0), ("bayes", 80.0), ("bayes", 160.0),
    ("sqrt", 160.0),
]
ALPHAS = [4.0, 8.0, 16.0]
WINDOWS = [240, 320, 480]


def score_config(sched, cfg, min_week):
    scheme, param, alpha, window, carry = cfg
    res, _, _ = backtest(sched, min_week, window, carry, alpha,
                         wfn=weight_fn(scheme, param), skip_totals=True)
    r = res.dropna(subset=["pred_margin", "mkt_margin", "actual_margin"])
    if len(r) < 200:
        return None
    _, t = incremental_test(r.actual_margin.values, r.mkt_margin.values,
                            r.pred_margin.values)
    return t, len(r)


def run_sweep(sched, min_week, progress=None):
    """
    Search on the EARLY seasons only, then test the winner on the late
    seasons it never saw. Picking the best of N configurations on the same
    data it was chosen from guarantees a flattering number; the holdout is
    the only figure that means anything.
    """
    seasons = sorted(sched["season"].unique())
    cut = seasons[int(len(seasons) * 0.6)]
    train = sched[sched["season"] < cut]
    hold = sched[sched["season"] >= cut]

    configs = [(sc, pa, al, wi, 0.35)
               for sc, pa in GRID for al in ALPHAS for wi in WINDOWS]
    rows = []
    for i, cfg in enumerate(configs):
        out = score_config(train, cfg, min_week)
        if out:
            rows.append({"scheme": cfg[0], "param": cfg[1], "alpha": cfg[2],
                         "window": cfg[3], "train_t": round(out[0], 2),
                         "n": out[1]})
        if progress:
            progress((i + 1) / len(configs),
                     f"config {i+1} of {len(configs)}")
    if not rows:
        return None
    tbl = pd.DataFrame(rows).sort_values("train_t", ascending=False)
    best = tbl.iloc[0]
    best_cfg = (best["scheme"], best["param"], best["alpha"],
                int(best["window"]), 0.35)
    base_cfg = ("linear", 160.0, 8.0, 320, 0.35)
    hb = score_config(hold, best_cfg, min_week)
    hd = score_config(hold, base_cfg, min_week)
    return {"table": tbl, "cut": cut, "best": best_cfg,
            "holdout_best": hb, "holdout_base": hd,
            "n_configs": len(configs)}


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

mode = st.sidebar.radio(
    "What to run",
    ["Backtest", "EPA model", "Signal lab", "Robustness sweep"])

if not run:
    st.info("Set the seasons on the left, then run.")
    st.stop()

bar = st.progress(0.0, "Loading schedules")
try:
    sched = load_schedules(yr[0], yr[1])
except Exception as e:
    bar.empty()
    st.error(f"Could not load schedule data: {e}")
    st.stop()

if mode == "EPA model":
    st.header("EPA model")
    st.write(
        "Opponent-adjusted EPA per play instead of scoring margin \u2014 the "
        "NFL analogue of the SP+/PPA inputs the college model uses. Scored by "
        "the same test: does it add anything to the closing line?"
    )
    try:
        bar.progress(0.05, "Loading play-by-play (slow the first time)\u2026")
        _epa = load_team_game_epa(yr[0], yr[1])
    except Exception as e:
        bar.empty()
        st.error(f"Could not load play-by-play: {e}")
        st.stop()
    st.caption(f"{len(_epa):,} team-games of EPA loaded.")
    _res = backtest_epa(sched, _epa, min_week, window, alpha,
                        lambda f, m: bar.progress(f, m))
    bar.empty()
    _r = _res.dropna(subset=["pred_margin", "mkt_margin", "actual_margin"])
    if len(_r) < 200:
        st.error("Not enough graded games. Widen the season range.")
        st.stop()
    _b, _t = incremental_test(_r.actual_margin.values, _r.mkt_margin.values,
                              _r.pred_margin.values)
    c1, c2, c3 = st.columns(3)
    c1.metric("Model coefficient", f"{_b[2]:+.3f}")
    c2.metric("t-stat", f"{_t:+.2f}")
    c3.metric("Games", f"{len(_r):,}")
    st.caption(
        f"Margin-only model scored +0.099 (t = +1.19) on the same test. "
        f"Closing-line coefficient here is {_b[1]:+.3f}."
    )
    if abs(_t) < 2:
        st.error(
            "**No edge.** EPA does not add measurable information beyond the "
            "closing line either. The market prices public play-by-play data "
            "as efficiently as it prices scoring margin."
        )
    elif _b[2] > 0:
        st.success(
            f"**Signal.** EPA adds information the closing line misses "
            f"(t = {_t:+.2f}). This is worth building Sunday Edge around."
        )
    else:
        st.warning("Anti-predictive. Check for a sign error before acting.")
    mae_m = float((_r.pred_margin - _r.actual_margin).abs().mean())
    mae_k = float((_r.mkt_margin - _r.actual_margin).abs().mean())
    st.caption(f"Mean absolute error \u2014 model {mae_m:.2f} pts, "
               f"market {mae_k:.2f} pts.")
    st.download_button("Download results as CSV",
                       _res.to_csv(index=False).encode(),
                       "nfl_epa_backtest.csv", "text/csv")
    st.stop()

if mode == "Robustness sweep":
    st.header("Robustness sweep")
    st.write(
        "Searches 72 weighting configurations on the early seasons, then "
        "tests the winner on later seasons it never saw. The holdout number "
        "is the only one worth reading."
    )
    out = run_sweep(sched, min_week, lambda f, m: bar.progress(f, m))
    bar.empty()
    if not out:
        st.error("Not enough data. Widen the season range.")
        st.stop()

    st.caption(f"Trained on seasons before {out['cut']}, held out "
               f"{out['cut']} onward. {out['n_configs']} configurations.")
    b = out["best"]
    st.subheader("Best configuration found")
    st.code(f"scheme  {b[0]}\nparam   {b[1]}\nalpha   {b[2]}\n"
            f"window  {b[3]}", language=None)

    c1, c2 = st.columns(2)
    if out["holdout_best"]:
        c1.metric("Winner, holdout t", f"{out['holdout_best'][0]:+.2f}")
    if out["holdout_base"]:
        c2.metric("Current settings, holdout t",
                  f"{out['holdout_base'][0]:+.2f}")

    ht = out["holdout_best"][0] if out["holdout_best"] else 0.0
    if abs(ht) < 2:
        st.error(
            f"**No configuration survives.** The best of "
            f"{out['n_configs']} settings reaches t = {ht:+.2f} out of "
            f"sample. Tuning the weighting scheme does not create an edge "
            f"here — the constraint is the information the model uses, not "
            f"how that information is weighted."
        )
    else:
        st.success(
            f"Holdout t = {ht:+.2f}. This survived selection on data it "
            f"never saw, which is a real result. Worth adopting."
        )
    st.subheader("All configurations, ranked on training seasons")
    st.caption("These training numbers are inflated by selection — the best "
               "of 72 always looks good. Do not read them as results.")
    st.dataframe(out["table"], hide_index=True, use_container_width=True)
    st.stop()

if mode == "Signal lab":
    bar.empty()
    st.header("Signal lab")
    st.write(
        "Each row asks whether a feature predicts what the closing line "
        "MISSED. Predicting the outcome is not an edge — the line already "
        "does that. Predicting the line's error is."
    )
    tbl = run_signal_lab(sched, line_sign(sched))
    notes = dict(zip(tbl["Signal"] + "|" + tbl["Market"], tbl["_note"]))
    st.dataframe(tbl.drop(columns=["_note"]), hide_index=True,
                 use_container_width=True)
    hits = tbl[tbl["Verdict"] == "SIGNAL"]
    if hits.empty:
        st.info(
            "Nothing clears |t| > 2.5. With ten tests, a bar of 2.0 would "
            "throw roughly one false positive per run, so the bar is set "
            "higher on purpose."
        )
    else:
        for _, h in hits.iterrows():
            st.success(
                f"**{h['Signal']}** ({h['Market']}): "
                f"{h['Effect per unit']:+.3f} pts per unit, t = {h['t']:+.2f}, "
                f"{h['Games']:,} games. "
                + (notes.get(h["Signal"] + "|" + h["Market"]) or "")
            )
        st.caption(
            "A hit here is a candidate, not a bet. Next step is checking it "
            "holds out of sample — split the seasons in half and see whether "
            "it survives in both."
        )
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
