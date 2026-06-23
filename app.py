import re
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLB Stack Command Center", layout="wide")

TEAM_ALIASES = {
    "WSH": "WAS", "WAS": "WAS", "TB": "TBR", "TBR": "TBR", "SF": "SFG", "SFG": "SFG",
    "CWS": "CHW", "CHW": "CHW", "KC": "KCR", "KCR": "KCR", "SD": "SDP", "SDP": "SDP",
    "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS", "CHC": "CHC", "CIN": "CIN",
    "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU", "LAA": "LAA", "LAD": "LAD",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NYM": "NYM", "NYY": "NYY", "OAK": "ATH",
    "ATH": "ATH", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA", "STL": "STL", "TEX": "TEX",
    "TOR": "TOR",
}

DK_POS_COLS = ["P", "P1", "P2", "C", "1B", "2B", "3B", "SS", "OF", "OF1", "OF2", "OF3", "UTIL"]
HITTER_POS_COLS = ["C", "1B", "2B", "3B", "SS", "OF", "OF1", "OF2", "OF3", "UTIL"]
PITCHER_POS_COLS = ["P", "P1", "P2", "SP", "SP1", "SP2"]


def canon_team(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z]", "", s)
    return TEAM_ALIASES.get(s, s if s else None)


def num_series(s, pct_hint=False):
    out = pd.to_numeric(s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False), errors="coerce")
    if pct_hint and out.max(skipna=True) and out.max(skipna=True) > 1.5:
        out = out / 100.0
    return out.fillna(0.0)


def norm(s):
    s = pd.Series(s).astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def read_csv(uploaded):
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(uploaded, encoding="latin1")


def parse_game_info(game):
    if pd.isna(game):
        return None, None
    token = str(game).split()[0]
    if "@" not in token:
        return None, None
    away, home = token.split("@", 1)
    return canon_team(away), canon_team(home)


def build_game_map(dk):
    if dk is None or "Game Info" not in dk.columns:
        return pd.DataFrame(columns=["Team", "Opponent", "Home", "Away", "ParkTeam", "Game Info"])
    games = []
    for gi in dk["Game Info"].dropna().unique():
        away, home = parse_game_info(gi)
        if away and home:
            games.append({"Team": away, "Opponent": home, "Home": False, "Away": True, "ParkTeam": home, "Game Info": gi})
            games.append({"Team": home, "Opponent": away, "Home": True, "Away": False, "ParkTeam": home, "Game Info": gi})
    return pd.DataFrame(games).drop_duplicates("Team") if games else pd.DataFrame(columns=["Team", "Opponent", "Home", "Away", "ParkTeam", "Game Info"])


def build_player_map(dk):
    if dk is None or "Name" not in dk.columns or "TeamAbbrev" not in dk.columns:
        return {}
    temp = dk.copy()
    temp["Team"] = temp["TeamAbbrev"].map(canon_team)
    return dict(zip(temp["Name"].astype(str).str.strip(), temp["Team"]))


def build_pitcher_grades(dk, stack_rankings):
    if dk is None:
        return pd.DataFrame()
    d = dk.copy()
    if "Roster Position" not in d.columns or "Name" not in d.columns:
        return pd.DataFrame()
    d["Team"] = d.get("TeamAbbrev", pd.Series([None]*len(d))).map(canon_team)
    d["Salary"] = num_series(d.get("Salary", pd.Series([0]*len(d))))
    d["AvgFP"] = num_series(d.get("AvgPointsPerGame", pd.Series([0]*len(d))))
    d["is_pitcher"] = d["Roster Position"].astype(str).str.contains("P", case=False, na=False) & ~d["Roster Position"].astype(str).str.contains("OF|UTIL|C|1B|2B|3B|SS", case=False, na=False)
    p = d[d["is_pitcher"]].copy()
    if p.empty:
        return p
    gmap = build_game_map(dk)
    p = p.merge(gmap[["Team", "Opponent", "ParkTeam"]], on="Team", how="left")
    opp = stack_rankings[["Team", "Final Stack Score", "Boom Score", "Park Score"]].rename(columns={"Team":"Opponent", "Final Stack Score":"Opp Stack Strength", "Boom Score":"Opp Boom Risk", "Park Score":"Park Hitter Boost"}) if stack_rankings is not None and not stack_rankings.empty else pd.DataFrame()
    if not opp.empty:
        p = p.merge(opp, on="Opponent", how="left")
    for c in ["Opp Stack Strength", "Opp Boom Risk", "Park Hitter Boost"]:
        if c not in p.columns:
            p[c] = 50.0
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(50.0)
    p["Value"] = p["AvgFP"] / (p["Salary"].replace(0, np.nan) / 1000.0)
    p["Pitcher Grade"] = (
        0.45 * norm(p["AvgFP"]) +
        0.20 * norm(p["Value"].fillna(0)) +
        0.20 * (100 - norm(p["Opp Stack Strength"])) +
        0.10 * (100 - norm(p["Opp Boom Risk"])) +
        0.05 * (100 - norm(p["Park Hitter Boost"]))
    ).round(2)
    return p[["Name", "Team", "Opponent", "Salary", "AvgFP", "Value", "Opp Stack Strength", "Opp Boom Risk", "Pitcher Grade"]].sort_values("Pitcher Grade", ascending=False)


def build_stack_rankings(scoring, matchups, parks, dk):
    if scoring is None or scoring.empty:
        return pd.DataFrame()
    if "names" not in scoring.columns:
        raise ValueError("Scoring sheet must have a column named 'names'.")

    s = scoring.copy()
    s["Team"] = s["names"].map(canon_team)
    s = s[s["Team"].notna()].drop_duplicates("Team").copy()
    for c in ["avgScore", "eightPlusRuns", "topScore", "teamOwnPct", "winPercentage", "avgFirstInning", "avgFifthInning"]:
        if c not in s.columns:
            s[c] = 0
    s["Avg Score"] = num_series(s["avgScore"])
    s["8+ Runs %"] = num_series(s["eightPlusRuns"], pct_hint=True)
    s["Top Score %"] = num_series(s["topScore"], pct_hint=True)
    s["Team Own %"] = num_series(s["teamOwnPct"], pct_hint=True)
    s["Win %"] = num_series(s["winPercentage"], pct_hint=True)
    s["F1 Avg"] = num_series(s["avgFirstInning"])
    s["F5 Avg"] = num_series(s["avgFifthInning"])

    out = s[["Team", "oppSP", "Avg Score", "8+ Runs %", "Top Score %", "Team Own %", "Win %", "F1 Avg", "F5 Avg"]].copy()

    if matchups is not None and not matchups.empty and "Team" in matchups.columns:
        m = matchups.copy()
        m["Team"] = m["Team"].map(canon_team)
        keep = ["Team"] + [c for c in ["Avg Score L5", "Avg Score L10", "Trending Score", "8+ For", "8+ For L5", "8+ For L10", "Trending 8+ For", "Matchup Avg Trend", "Matchup 8+ Trend", "R2_to_Opp_Trend"] if c in m.columns]
        m = m[keep].drop_duplicates("Team")
        out = out.merge(m, on="Team", how="left")

    gmap = build_game_map(dk)
    if not gmap.empty:
        out = out.merge(gmap[["Team", "Opponent", "Home", "ParkTeam", "Game Info"]], on="Team", how="left")

    if parks is not None and not parks.empty:
        p = parks.copy()
        name_col = "Names" if "Names" in p.columns else ("Team" if "Team" in p.columns else None)
        if name_col:
            p["ParkTeam"] = p[name_col].map(canon_team)
            if "Set" in p.columns:
                overall = p[p["Set"].astype(str).str.lower().eq("overall")]
                if not overall.empty:
                    p = overall
            keep = ["ParkTeam"] + [c for c in ["Runs", "HR", "1B", "2B"] if c in p.columns]
            out = out.merge(p[keep].drop_duplicates("ParkTeam"), on="ParkTeam", how="left")

    for c in ["Avg Score L5", "Avg Score L10", "Trending Score", "8+ For", "8+ For L5", "8+ For L10", "Trending 8+ For", "Matchup Avg Trend", "Matchup 8+ Trend", "R2_to_Opp_Trend", "Runs", "HR", "1B", "2B"]:
        if c not in out.columns:
            out[c] = 0
        out[c] = num_series(out[c])

    scoring_score = 0.35*norm(out["Avg Score"]) + 0.30*norm(out["Top Score %"]) + 0.25*norm(out["8+ Runs %"]) + 0.10*norm(out["F5 Avg"])
    trend_score = 0.35*norm(out["Trending Score"]) + 0.25*norm(out["Trending 8+ For"]) + 0.25*norm(out["Matchup Avg Trend"]) + 0.15*norm(out["Matchup 8+ Trend"])
    park_score = 0.40*norm(out["Runs"]) + 0.40*norm(out["HR"]) + 0.10*norm(out["1B"]) + 0.10*norm(out["2B"])
    own = out["Team Own %"].replace(0, np.nan)
    leverage_raw = (out["Top Score %"] / own).replace([np.inf, -np.inf], np.nan).fillna(0)
    leverage_score = norm(leverage_raw)
    boom_score = 0.55*norm(out["8+ Runs %"]) + 0.35*norm(out["Top Score %"]) + 0.10*park_score
    safe_score = 0.45*norm(out["Avg Score"]) + 0.25*norm(out["Win %"]) + 0.20*norm(out["F5 Avg"]) + 0.10*park_score
    fade_risk = (0.60*norm(out["Team Own %"]) + 0.25*(100 - norm(out["Top Score %"])) + 0.15*(100 - trend_score))

    out["Scoring Score"] = scoring_score.round(2)
    out["Trend Score"] = trend_score.round(2)
    out["Park Score"] = park_score.round(2)
    out["Leverage Score"] = leverage_score.round(2)
    out["Boom Score"] = boom_score.round(2)
    out["Safe Score"] = safe_score.round(2)
    out["Fade Risk"] = fade_risk.round(2)
    out["Final Stack Score"] = (0.40*scoring_score + 0.25*trend_score + 0.20*park_score + 0.15*leverage_score).round(2)

    def label(row):
        if row["Final Stack Score"] >= 82:
            return "Elite Stack"
        if row["Leverage Score"] >= 75 and row["Boom Score"] >= 55:
            return "Leverage Stack"
        if row["Fade Risk"] >= 75 and row["Final Stack Score"] < 65:
            return "Over-Owned Risk"
        if row["Final Stack Score"] >= 65:
            return "Strong Stack"
        if row["Final Stack Score"] >= 45:
            return "MME Only"
        return "Avoid"
    out["Recommendation"] = out.apply(label, axis=1)
    return out.sort_values("Final Stack Score", ascending=False).reset_index(drop=True)


def find_lineup_columns(port):
    return [c for c in port.columns if str(c).strip().upper() in DK_POS_COLS or str(c).strip().upper().startswith("OF")]


def team_counts_for_row(row, cols, player_map):
    teams = []
    for c in cols:
        val = row.get(c)
        if pd.isna(val):
            continue
        name = str(val).strip()
        if name in player_map:
            teams.append(player_map[name])
        else:
            cleaned = re.sub(r"\s*\([^)]*\)$", "", name).strip()
            if cleaned in player_map:
                teams.append(player_map[cleaned])
    return Counter([t for t in teams if t])


def detect_primary_secondary(counter):
    if not counter:
        return None, None, "Unknown"
    common = counter.most_common()
    primary = common[0][0]
    secondary = common[1][0] if len(common) > 1 and common[1][1] >= 2 else None
    sizes = sorted(counter.values(), reverse=True)
    stack_type = "-".join(str(x) for x in sizes)
    return primary, secondary, stack_type


def grade_portfolio(port, dk, stack_rankings, pitcher_grades):
    if port is None or port.empty or dk is None or dk.empty:
        return pd.DataFrame(), pd.DataFrame()
    player_map = build_player_map(dk)
    cols = find_lineup_columns(port)
    if not cols:
        return pd.DataFrame(), pd.DataFrame()
    graded = port.copy()
    hitter_cols = [c for c in cols if str(c).strip().upper() in HITTER_POS_COLS or str(c).strip().upper().startswith("OF")]
    pitcher_cols = [c for c in cols if str(c).strip().upper() in PITCHER_POS_COLS]
    if not pitcher_cols and "P" in cols:
        pitcher_cols = ["P"]

    score_map = dict(zip(stack_rankings["Team"], stack_rankings["Final Stack Score"])) if stack_rankings is not None and not stack_rankings.empty else {}
    boom_map = dict(zip(stack_rankings["Team"], stack_rankings["Boom Score"])) if stack_rankings is not None and not stack_rankings.empty else {}
    pitch_map = dict(zip(pitcher_grades["Name"], pitcher_grades["Pitcher Grade"])) if pitcher_grades is not None and not pitcher_grades.empty else {}

    prim, sec, typ, stscore, pscore = [], [], [], [], []
    for _, row in graded.iterrows():
        cnt = team_counts_for_row(row, hitter_cols, player_map)
        p, s, t = detect_primary_secondary(cnt)
        prim.append(p); sec.append(s); typ.append(t)
        stscore.append(float(score_map.get(p, 0)) + 0.35*float(score_map.get(s, 0)))
        pg = []
        for c in pitcher_cols:
            name = str(row.get(c, "")).strip()
            if name in pitch_map:
                pg.append(float(pitch_map[name]))
        pscore.append(np.mean(pg) if pg else np.nan)
    graded["Primary Stack"] = prim
    graded["Secondary Stack"] = sec
    graded["Stack Type"] = typ
    graded["Stack Score"] = pd.Series(stscore).round(2)
    graded["Pitching Score"] = pd.Series(pscore).round(2)
    graded["Overall Lineup Score"] = (0.70*num_series(graded["Stack Score"]) + 0.30*num_series(graded["Pitching Score"])).round(2)
    graded["Action"] = np.where(graded["Overall Lineup Score"] >= 75, "Keep", np.where(graded["Overall Lineup Score"] >= 55, "Review", "Cut"))

    exposure = graded.groupby("Primary Stack", dropna=False).agg(
        Lineups=("Primary Stack", "size"),
        AvgOverall=("Overall Lineup Score", "mean"),
        Keeps=("Action", lambda x: (x=="Keep").sum()),
        Reviews=("Action", lambda x: (x=="Review").sum()),
        Cuts=("Action", lambda x: (x=="Cut").sum()),
    ).reset_index()
    exposure["Exposure %"] = (exposure["Lineups"] / len(graded) * 100).round(2)
    exposure["AvgOverall"] = exposure["AvgOverall"].round(2)
    exposure = exposure.sort_values(["Lineups", "AvgOverall"], ascending=[False, False])
    return graded, exposure


st.title("⚾ MLB DFS Stack Command Center")
st.caption("Scoring Sheet = slate master. DK Salaries = player/team/game map. Portfolio CSV = detected primary and secondary stacks.")

with st.sidebar:
    st.header("Upload CSVs")
    scoring_file = st.file_uploader("Scoring % Sheet", type="csv")
    matchups_file = st.file_uploader("Matchups Master Sheet", type="csv")
    parks_file = st.file_uploader("Park Factors Sheet", type="csv")
    dk_file = st.file_uploader("DK Salaries Sheet", type="csv")
    portfolio_file = st.file_uploader("Optional: Lineup Portfolio Export", type="csv")
    st.info("V4: Fixes notna crash, keeps portfolio filters from uploaded portfolio, adds pitcher grades.")

scoring = read_csv(scoring_file)
matchups = read_csv(matchups_file)
parks = read_csv(parks_file)
dk = read_csv(dk_file)
portfolio = read_csv(portfolio_file)

if scoring is None or matchups is None or parks is None or dk is None:
    st.warning("Upload the scoring, matchups, park factors, and DK salaries CSVs to build rankings.")
    st.stop()

try:
    rankings = build_stack_rankings(scoring, matchups, parks, dk)
    pitcher_grades = build_pitcher_grades(dk, rankings)
except Exception as e:
    st.error(f"Could not build rankings: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["Stack Rankings", "Pitcher Grades", "Portfolio Mode", "Debug"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Slate Teams", len(rankings))
    c2.metric("Top Stack", rankings.iloc[0]["Team"] if not rankings.empty else "-")
    c3.metric("Top Score", rankings.iloc[0]["Final Stack Score"] if not rankings.empty else "-")
    c4.metric("Elite/Strong", int(rankings["Recommendation"].isin(["Elite Stack", "Strong Stack", "Leverage Stack"]).sum()))
    st.dataframe(rankings, use_container_width=True, hide_index=True)
    st.download_button("Download stack rankings CSV", rankings.to_csv(index=False), "stack_rankings.csv", "text/csv")

with tab2:
    if pitcher_grades.empty:
        st.warning("No pitcher rows detected from DK Salaries.")
    else:
        st.dataframe(pitcher_grades, use_container_width=True, hide_index=True)
        st.download_button("Download pitcher grades CSV", pitcher_grades.to_csv(index=False), "pitcher_grades.csv", "text/csv")

with tab3:
    if portfolio is None:
        st.info("Upload a portfolio CSV to grade lineups.")
    else:
        graded, exposure = grade_portfolio(portfolio, dk, rankings, pitcher_grades)
        if graded.empty:
            st.warning("Could not detect lineup/player columns in the portfolio CSV. Make sure it has DK roster columns like P/P1/P2/C/1B/2B/3B/SS/OF/OF1/OF2/OF3.")
        else:
            teams = sorted([x for x in graded["Primary Stack"].dropna().unique()])
            sec_teams = sorted([x for x in graded["Secondary Stack"].dropna().unique()])
            actions = st.multiselect("Action", sorted(graded["Action"].unique()), default=sorted(graded["Action"].unique()))
            psel = st.multiselect("Primary Stack", teams, default=teams)
            ssel = st.multiselect("Secondary Stack", sec_teams, default=[])
            f = graded[graded["Action"].isin(actions)]
            if psel:
                f = f[f["Primary Stack"].isin(psel)]
            if ssel:
                f = f[f["Secondary Stack"].isin(ssel)]
            a, b, c, d = st.columns(4)
            a.metric("Lineups", len(graded))
            b.metric("Keep", int((graded["Action"]=="Keep").sum()))
            c.metric("Review", int((graded["Action"]=="Review").sum()))
            d.metric("Cut", int((graded["Action"]=="Cut").sum()))
            st.subheader("Graded Lineups — with Players")
            st.dataframe(f, use_container_width=True, hide_index=True)
            st.download_button("Download filtered graded lineups CSV", f.to_csv(index=False), "filtered_graded_lineups.csv", "text/csv")
            st.subheader("Stack Exposure vs Quality")
            st.dataframe(exposure, use_container_width=True, hide_index=True)

with tab4:
    st.write("Scoring columns", list(scoring.columns))
    st.write("Matchups columns", list(matchups.columns))
    st.write("Park columns", list(parks.columns))
    st.write("DK columns", list(dk.columns))
    if portfolio is not None:
        st.write("Portfolio columns", list(portfolio.columns))
