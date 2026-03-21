from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm  # type: ignore
from statsmodels.stats.sandwich_covariance import cov_cluster


DEFAULT_INPUT = "outputs/cleaning/shinagawa_letters_cleaned.csv"
DEFAULT_OUTDIR = "outputs/phase3_11_3"

PERIOD_COL = "活動期"
BIRTH_REGION_COL = "出生地域_主"
SENDER_COL = "発信者"
LETTER_ID_COL = "整理番号"

PERIODS = [
    ("1.維新期", pd.Timestamp("1868-01-01"), pd.Timestamp("1870-07-31")),
    ("2.ドイツ滞在期①", pd.Timestamp("1870-08-01"), pd.Timestamp("1875-10-05")),
    ("3.殖産興業官僚期", pd.Timestamp("1875-10-06"), pd.Timestamp("1886-03-31")),
    ("4.ドイツ滞在期②", pd.Timestamp("1886-04-01"), pd.Timestamp("1887-06-05")),
    ("5.帰朝～宮中顧問官期", pd.Timestamp("1887-06-06"), pd.Timestamp("1889-05-12")),
    ("6.宮内省御料局長期", pd.Timestamp("1889-05-13"), pd.Timestamp("1891-05-31")),
    ("7.内務大臣期", pd.Timestamp("1891-06-01"), pd.Timestamp("1892-03-11")),
    ("8.晩年期", pd.Timestamp("1892-03-12"), pd.Timestamp("1900-02-26")),
]
PERIOD_ORDER = [p[0] for p in PERIODS]

ERA_START_GREG = {
    "M": 1868, "明治": 1868,
    "T": 1912, "大正": 1912,
    "S": 1926, "昭和": 1926,
    "K": 1865, "慶応": 1865,
}

DEFAULT_ATTR_MAP = {
    "①": "毛利家関係者",
    "②": "旧長州藩士・農兵隊",
    "③": "政治家・官員（農商務省関係者）",
    "④": "政治家・官員（宮内省関係者）",
    "⑤": "政治家・官員（内務省本省関係者）",
    "⑥": "政治家・官員（地方庁関係者）",
    "⑦": "政治家・官員（第一次松方内閣関係者）",
    "⑧": "政治家・官員（③～⑦以外）",
    "⑨": "国民協会関係者",
    "⑩": "実業家",
    "⑪": "美術工芸作家",
    "⑫": "僧侶",
    "⑬": "神官",
    "⑭": "維新志士関係者（長州藩以外）",
    "⑮": "軍人",
    "⑯": "ジャーナリスト",
    "⑰": "真宗門徒・関係者",
    "⑱": "品川家関係者",
    "⑲": "老農・篤農家・産業実践者等",
    "⑳": "独逸学関係者",
    "㉑": "医師",
    "ND": "不明",
    "O": "その他",
}
DEFAULT_TARGET_ATTRS = ["③", "④", "⑨", "⑩"]
DEFAULT_SAFE = {"③": "a03", "④": "a04", "⑨": "a09", "⑩": "a10"}
DEFAULT_RESCUE_ATTRS = ["⑨"]

DEFAULT_RUNS = [
    {"run": "A_main_withRegion", "top_attrs": 12, "top_regions": 8, "min_count_keep": 10, "include_region": True},
    {"run": "B_main_noRegion", "top_attrs": 12, "top_regions": 8, "min_count_keep": 10, "include_region": False},
    {"run": "C_wideAttr_withRegion", "top_attrs": 20, "top_regions": 8, "min_count_keep": 5, "include_region": True},
    {"run": "D_strictMin20_withRegion", "top_attrs": 12, "top_regions": 8, "min_count_keep": 20, "include_region": True},
]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\u3000", " ").strip()
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def load_attr_map(path: Optional[str]) -> Dict[str, str]:
    out = DEFAULT_ATTR_MAP.copy()
    if not path:
        return out
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            k, v = line.split("\t", 1)
        elif "," in line:
            k, v = line.split(",", 1)
        elif ":" in line:
            k, v = line.split(":", 1)
        else:
            continue
        k, v = k.strip(), v.strip()
        if k:
            out[k] = v
    return out


def short_label(code: object, attr_map: Dict[str, str]) -> str:
    c = normalize_text(code)
    return attr_map.get(c, c)


def safe_name_map(attr_codes: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used = set()
    for i, a in enumerate(attr_codes, start=1):
        a_norm = normalize_text(a)
        if a_norm in DEFAULT_SAFE:
            cand = DEFAULT_SAFE[a_norm]
        elif a_norm == "ND":
            cand = "a_nd"
        elif a_norm == "O":
            cand = "a_other"
        else:
            cand = f"a{i:02d}"
        base = cand
        n = 1
        while cand in used:
            n += 1
            cand = f"{base}_{n}"
        mapping[a_norm] = cand
        used.add(cand)
    return mapping


def split_attr_tokens(s: object) -> List[str]:
    txt = normalize_text(s)
    if not txt:
        return []
    return [p.strip() for p in re.split(r"[、,/\s]+", txt) if p.strip()]


def explode_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Use only canonical attribute-code columns: 属性, 属性1, 属性2, ...

    Exclude derived/helper columns such as 属性_リスト, 属性_複数フラグ, 属性_数,
    属性_組み合わせ, which would otherwise pollute the token space.
    """
    attr_candidates = [c for c in df.columns if (c == "属性" or re.fullmatch(r"属性\d+", c))]
    attr_candidates = list(dict.fromkeys(attr_candidates))
    if not attr_candidates:
        raise ValueError("属性列が見つかりません。")

    rows: List[dict] = []
    for _, row in df.iterrows():
        codes: List[str] = []
        for c in attr_candidates:
            vals = split_attr_tokens(row.get(c))
            codes.extend(vals)
        codes = sorted(set([x for x in codes if x]))
        if not codes:
            rows.append({LETTER_ID_COL: row[LETTER_ID_COL], "属性_code": "ND"})
        else:
            for code in codes:
                rows.append({LETTER_ID_COL: row[LETTER_ID_COL], "属性_code": code})
    return pd.DataFrame(rows)


def compress_series(
    s: pd.Series,
    topk: int = 8,
    min_count: int = 10,
    other_label: str = "O",
) -> pd.Series:
    s2 = s.fillna("ND").astype(str)
    vc = s2.value_counts()
    keep = vc[(vc >= min_count)].head(topk).index
    return s2.where(s2.isin(keep), other_label)


def era_to_gregorian_year(era: object, y: object) -> float:
    if pd.isna(era) or pd.isna(y):
        return np.nan
    e = normalize_text(era)
    if e not in ERA_START_GREG:
        return np.nan
    try:
        return round(ERA_START_GREG[e] + (float(y) - 1))
    except Exception:
        return np.nan


def is_valid_month_day(m: object, d: object | None = None, y: object | None = None) -> bool:
    try:
        if pd.isna(m):
            return False
        m_int = int(m)
        if d is None or pd.isna(d):
            return 1 <= m_int <= 12
        d_int = int(d)
        if y is not None and not pd.isna(y):
            pd.Timestamp(year=int(y), month=m_int, day=d_int)
            return True
        return (1 <= m_int <= 12) and (1 <= d_int <= 31)
    except Exception:
        return False


def pid_for_ymd(dt: pd.Timestamp) -> Optional[str]:
    for pid, start, end in PERIODS:
        if start <= dt <= end:
            return pid
    return None


def period_for_year_midpoint(y: int) -> Optional[str]:
    return pid_for_ymd(pd.Timestamp(int(y), 7, 1))


def range_single_period(start_y: int, end_y: int) -> Optional[str]:
    hit = []
    for pid, start, end in PERIODS:
        ys, ye = start.year, end.year
        if start_y >= ys and end_y <= ye:
            hit.append(pid)
    if len(hit) == 1:
        return hit[0]
    return None


def build_effective_period_sensitivity(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Sensitivity仕様:
    - 年代幅が単一期に収まるならその期
    - またぐなら代表年で補完
    - 月日ありはYMD厳密
    - 月日欠損は年ベース判定を許容（感度分析の一部として）
    """
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    rule = pd.Series(pd.NA, index=df.index, dtype="object")

    year_greg = pd.to_numeric(df.get("年代_西暦"), errors="coerce")
    rep_candidates = ["年代_代表", "年代_代表値", "年代_代表値_西暦"]
    rep_col = next((c for c in rep_candidates if c in df.columns), None)
    rep_raw = pd.to_numeric(df[rep_col], errors="coerce") if rep_col else pd.Series(np.nan, index=df.index)
    era_col = df["年代"] if "年代" in df.columns else pd.Series(pd.NA, index=df.index)

    rep_greg = pd.Series(np.nan, index=df.index, dtype="float")
    if rep_col:
        if rep_col.endswith("西暦"):
            rep_greg = rep_raw.copy()
        else:
            rep_greg = pd.Series(
                [era_to_gregorian_year(e, y) for e, y in zip(era_col, rep_raw)],
                index=df.index,
                dtype="float",
            )

    year_point = pd.Series(np.floor(year_greg.where(year_greg.notna(), rep_greg) + 0.5), index=df.index)

    start_year = pd.to_numeric(df.get("年代_開始"), errors="coerce")
    end_year = pd.to_numeric(df.get("年代_終了"), errors="coerce")
    month = pd.to_numeric(df.get("月"), errors="coerce")
    day = pd.to_numeric(df.get("日"), errors="coerce")

    for idx in df.index:
        sy, ey = start_year.get(idx), end_year.get(idx)
        if pd.notna(sy) and pd.notna(ey):
            single = range_single_period(int(np.floor(sy + 0.5)), int(np.floor(ey + 0.5)))
            if single is not None:
                out.at[idx] = single
                rule.at[idx] = "range_single_period"
                continue

        yp = year_point.get(idx)
        if pd.isna(yp):
            continue

        m = month.get(idx)
        d = day.get(idx)

        if pd.notna(m) and pd.notna(d) and is_valid_month_day(m, d, yp):
            dt = pd.Timestamp(int(yp), int(m), int(d))
            pid = pid_for_ymd(dt)
            if pid is not None:
                out.at[idx] = pid
                rule.at[idx] = "ymd_strict"
                continue

        pid = period_for_year_midpoint(int(yp))
        if pid is not None:
            out.at[idx] = pid
            rule.at[idx] = "year_midpoint_sensitivity"

    return out, rule


def prune_levels_for_y(df: pd.DataFrame, ycol: str, catcol: str, min_n: int = 30) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    g = df.groupby(catcol, observed=False)[ycol].agg(["count", "sum"]).reset_index()
    g["drop_reason"] = ""
    g.loc[g["count"] < min_n, "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "count<min_n")
    g.loc[g["sum"] == 0, "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "all_zero")
    g.loc[g["sum"] == g["count"], "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "all_one")
    keep = g[g["drop_reason"] == ""][catcol].tolist()
    out = df[df[catcol].isin(keep)].copy()
    return out, keep, g


def prune_cells_for_y(df: pd.DataFrame, ycol: str, rowcat: str, colcat: str, min_n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = df.groupby([rowcat, colcat], observed=False)[ycol].agg(["count", "sum"]).reset_index()
    g["drop_reason"] = ""
    g.loc[g["count"] < min_n, "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "count<min_cell_n")
    g.loc[g["sum"] == 0, "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "all_zero")
    g.loc[g["sum"] == g["count"], "drop_reason"] = g["drop_reason"].mask(g["drop_reason"] == "", "all_one")
    keep_cells = g[g["drop_reason"] == ""][[rowcat, colcat]]
    if keep_cells.empty:
        return df.iloc[0:0].copy(), g
    out = df.merge(keep_cells, on=[rowcat, colcat], how="inner")
    return out, g


def fit_glm_cluster_manual(formula: str, data: pd.DataFrame, group_col: str, maxiter: int = 300):
    model = smf.glm(formula, data=data, family=sm.families.Binomial())
    try:
        res = model.fit(disp=0, maxiter=maxiter, cov_type="cluster", cov_kwds={"groups": data[group_col]})
        converged = bool(getattr(res, "converged", True))
        return res.params, res.bse, res.pvalues, converged, "fit(cov_type=cluster)"
    except Exception:
        pass
    res = model.fit(disp=0, maxiter=maxiter)
    converged = bool(getattr(res, "converged", True))
    try:
        cov = cov_cluster(res, data[group_col])
        se = pd.Series(np.sqrt(np.diag(cov)), index=res.params.index)
        z = res.params / se
        p = pd.Series(2 * (1 - norm.cdf(np.abs(z))), index=res.params.index)
        return res.params, se, p, converged, "manual cov_cluster"
    except Exception as e:
        return res.params, res.bse, res.pvalues, converged, f"no_cluster({type(e).__name__})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3_11_3 public: sensitivity runs for target-attribute GLM/logit.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--attr_map", default=None)
    p.add_argument("--target_attrs", nargs="*", default=DEFAULT_TARGET_ATTRS)
    p.add_argument("--rescue_attrs", nargs="*", default=DEFAULT_RESCUE_ATTRS)
    p.add_argument("--min_pos", type=int, default=20)
    p.add_argument("--min_level_n", type=int, default=30)
    p.add_argument("--min_cell_n", type=int, default=20)
    p.add_argument("--ref_period", default="3.殖産興業官僚期")
    p.add_argument("--ref_region", default=None)
    p.add_argument("--sender", default=SENDER_COL)
    p.add_argument("--period_col", default=PERIOD_COL)
    p.add_argument("--birth_region_col", default=BIRTH_REGION_COL)
    p.add_argument("--letter_id_col", default=LETTER_ID_COL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.outdir)
    ensure_outdir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    attr_map = load_attr_map(args.attr_map)

    df = pd.read_csv(args.input, encoding=args.encoding)
    req = [args.birth_region_col, args.sender, args.letter_id_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"必須列がありません: {miss}")

    df[args.period_col], df["_period_rule"] = build_effective_period_sensitivity(df)

    df2 = df[df[args.period_col].notna()].copy()
    df2 = df2[df2[args.birth_region_col].notna()].copy()
    df2 = df2[df2[args.sender].notna()].copy()
    df2 = df2[df2[args.letter_id_col].notna()].copy()

    dfA = explode_attributes(df2)

    base = df2[[args.letter_id_col, args.period_col, args.birth_region_col, args.sender]].copy()
    safe_map = safe_name_map([normalize_text(a) for a in args.target_attrs])
    safe_rows = [
        {"attr_code": a, "attr_label": short_label(a, attr_map), "safe_name": safe_map[a]}
        for a in [normalize_text(a) for a in args.target_attrs]
    ]
    safe_csv = out_dir / f"phase3_11_3_attr_safe_name_map_{ts}.csv"
    pd.DataFrame(safe_rows).to_csv(safe_csv, index=False, encoding="utf-8-sig")

    run_index_rows: List[dict] = []
    all_outputs: List[Path] = [safe_csv]

    for run_cfg in DEFAULT_RUNS:
        run_name = run_cfg["run"]
        run_dir = out_dir / run_name
        ensure_outdir(run_dir)

        log_lines: List[str] = []

        def log(s: str = "") -> None:
            log_lines.append(s)
            print(f"[{run_name}] {s}")

        log_txt = run_dir / f"phase3_11_3_logit_log_{run_name}_{ts}.txt"
        result_csv = run_dir / f"phase3_11_3_logit_target_attrs_{run_name}_{ts}.csv"

        top_attrs = int(run_cfg["top_attrs"])
        top_regions = int(run_cfg["top_regions"])
        min_count_keep = int(run_cfg["min_count_keep"])
        include_region = bool(run_cfg["include_region"])

        log("📋 実行条件")
        log(f"   input: {args.input}")
        log(f"   encoding: {args.encoding}")
        log(f"   outdir: {run_dir}")
        log(f"   attr_map指定: {args.attr_map if args.attr_map else '(なし: DEFAULTのみ)'}")
        log(f"   target_attrs: {args.target_attrs}")
        log(f"   rescue_attrs: {args.rescue_attrs}")
        log(f"   top_attrs: {top_attrs}")
        log(f"   top_regions: {top_regions}")
        log(f"   min_count_keep: {min_count_keep}")
        log(f"   include_region: {include_region}")
        log(f"   min_pos: {args.min_pos}")
        log(f"   min_level_n: {args.min_level_n}")
        log(f"   min_cell_n: {args.min_cell_n}")
        log(f"   ref_period(希望): {args.ref_period}")
        log(f"   ref_region(希望): {args.ref_region}")
        log("")
        log("📌 sensitivity仕様")
        log("   月日欠損の活動期判定は、年ベース midpoint 判定を許容（感度分析仕様）。")
        log("   他phaseの厳密YMD判定と完全一致しない可能性があるが、仕様として保持。")
        log("")

        # compress per run
        tmpA = dfA.copy()
        tmpA["属性_分析"] = compress_series(tmpA["属性_code"], topk=top_attrs, min_count=min_count_keep, other_label="O")
        attrs_per_letter = (
            tmpA.groupby(args.letter_id_col, observed=False)["属性_分析"]
            .apply(lambda x: sorted(set(x)))
            .rename("attrs")
        )

        tmp_base = base.merge(attrs_per_letter, on=args.letter_id_col, how="left")
        tmp_base["attrs"] = tmp_base["attrs"].apply(lambda x: x if isinstance(x, list) else [])
        tmp_base["region"] = compress_series(tmp_base[args.birth_region_col], topk=top_regions, min_count=min_count_keep, other_label="O")

        b = tmp_base[[args.period_col, "region", args.sender, "attrs"]].copy()
        b = b.rename(columns={args.period_col: "period", args.sender: "sender"})
        b["period"] = pd.Categorical(b["period"], categories=PERIOD_ORDER, ordered=True)
        b["region"] = pd.Categorical(b["region"])

        ref_region = args.ref_region if args.ref_region is not None else (
            b["region"].value_counts().index[0] if len(b) else None
        )

        target_attrs = [normalize_text(a) for a in args.target_attrs]
        for a in target_attrs:
            ycol = f"Y_{safe_map[a]}"
            b[ycol] = b["attrs"].apply(lambda lst: 1 if a in lst else 0)

        pos_info = [(a, int(b[f"Y_{safe_map[a]}"].sum())) for a in target_attrs]
        log(f"🎯 対象属性 pos: {pos_info}")
        log("")

        rescue_notes = []
        results: List[dict] = []

        for a in target_attrs:
            ycol = f"Y_{safe_map[a]}"
            pos = int(b[ycol].sum())
            if pos < args.min_pos:
                log(f"⚠️ スキップ attr={a}: pos<{args.min_pos} (pos={pos})")
                continue

            tmp = b.copy()
            tmp, keep_p, stats_p = prune_levels_for_y(tmp, ycol, "period", min_n=args.min_level_n)
            if include_region:
                tmp, keep_r, stats_r = prune_levels_for_y(tmp, ycol, "region", min_n=args.min_level_n)
                tmp, stats_cells = prune_cells_for_y(tmp, ycol, "period", "region", min_n=args.min_cell_n)
            else:
                keep_r, stats_r = [], pd.DataFrame()
                stats_cells = pd.DataFrame()

            if hasattr(tmp["period"], "cat"):
                tmp["period"] = tmp["period"].cat.remove_unused_categories()
            if include_region and hasattr(tmp["region"], "cat"):
                tmp["region"] = tmp["region"].cat.remove_unused_categories()

            log(f"   attr={a}: N after prune={len(tmp)}")

            if len(tmp) == 0:
                log(f"❌ attr={a}: 間引き後にデータが消滅")
                continue

            ref_p = args.ref_period if args.ref_period in list(tmp["period"].cat.categories) else tmp["period"].value_counts().index[0]
            if include_region:
                ref_r = ref_region if (ref_region is not None and ref_region in list(tmp["region"].cat.categories)) else tmp["region"].value_counts().index[0]
            else:
                ref_r = None

            if include_region:
                f_main = (
                    f"{ycol} ~ "
                    f"C(period, Treatment(reference='{ref_p}')) + "
                    f"C(region, Treatment(reference='{ref_r}'))"
                )
                f_rescue = f"{ycol} ~ C(period, Treatment(reference='{ref_p}'))"
            else:
                f_main = f"{ycol} ~ C(period, Treatment(reference='{ref_p}'))"
                f_rescue = f_main

            model_note = ""
            try:
                params, bse, pvals, converged, note = fit_glm_cluster_manual(f_main, tmp, "sender", maxiter=300)
                model_note = "main(period+region)" if include_region else "main(period only)"
            except Exception as e:
                if a in set(args.rescue_attrs) and include_region:
                    rescue_notes.append(
                        {
                            "attr_code": a,
                            "attr_label": short_label(a, attr_map),
                            "reason": f"main failed: {type(e).__name__}",
                            "policy": "属性⑨は主モデル（period+region）で分離・収束不良が生じやすいため、感度分析においても period のみの rescue model を許可した。",
                        }
                    )
                    log(f"⚠️ attr={a} main失敗({type(e).__name__}) → rescue(periodのみ)へ")
                    try:
                        params, bse, pvals, converged, note = fit_glm_cluster_manual(f_rescue, tmp, "sender", maxiter=300)
                        model_note = "rescue(period only)"
                    except Exception as e2:
                        log(f"❌ attr={a} rescueも失敗: {type(e2).__name__}: {e2}")
                        continue
                else:
                    log(f"❌ attr={a} 推定失敗: {type(e).__name__}: {e}")
                    continue

            if not converged:
                log(f"⚠️ attr={a}: 収束せず（CSVに出さない）")
                continue

            for term in params.index:
                if term == "Intercept":
                    continue
                beta = float(params[term])
                se = float(bse[term])
                or_ = float(np.exp(beta))
                lo = float(np.exp(beta - 1.96 * se))
                hi = float(np.exp(beta + 1.96 * se))
                results.append(
                    {
                        "run": run_name,
                        "include_region": include_region,
                        "attr_code": a,
                        "attr_label": short_label(a, attr_map),
                        "safe_name": safe_map[a],
                        "term": term,
                        "beta": beta,
                        "se_cluster": se,
                        "OR": or_,
                        "CI95_low": lo,
                        "CI95_high": hi,
                        "p_cluster": float(pvals[term]),
                        "pos_n": pos,
                        "N_used": int(len(tmp)),
                        "ref_period": ref_p,
                        "ref_region": ref_r if ref_r is not None else "",
                        "keep_period_levels": "|".join(map(str, keep_p)),
                        "keep_region_levels": "|".join(map(str, keep_r)),
                        "model_note": model_note + (f" / {note}" if note else ""),
                        "top_attrs": top_attrs,
                        "top_regions": top_regions,
                        "min_count_keep": min_count_keep,
                    }
                )

        res_df = pd.DataFrame(results)
        res_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
        all_outputs.extend([log_txt, result_csv])

        log(f"✅ 結果CSV: {result_csv} / rows={len(res_df)}")
        if rescue_notes:
            log("")
            log("📌 rescue model 注記")
            for r in rescue_notes:
                log(f"   attr={r['attr_code']} ({r['attr_label']}): {r['policy']} / reason={r['reason']}")

        log("")
        log("📦 出力一覧")
        log(f"   result_csv: {result_csv}")
        log(f"   safe_map_csv: {safe_csv}")
        log_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

        run_index_rows.append(
            {
                "run": run_name,
                "input": args.input,
                "outdir": str(run_dir),
                "encoding": args.encoding,
                "include_region": include_region,
                "top_attrs": top_attrs,
                "top_regions": top_regions,
                "min_count_keep": min_count_keep,
                "min_pos": args.min_pos,
                "min_level_n": args.min_level_n,
                "min_cell_n": args.min_cell_n,
                "ref_period_requested": args.ref_period,
                "ref_region_requested": args.ref_region if args.ref_region is not None else "",
                "result_csv": str(result_csv),
                "log_txt": str(log_txt),
            }
        )

    run_index_csv = out_dir / f"phase3_11_3_run_index_{ts}.csv"
    pd.DataFrame(run_index_rows).to_csv(run_index_csv, index=False, encoding="utf-8-sig")
    all_outputs.append(run_index_csv)
    print(f"✅ run_index出力: {run_index_csv}")


if __name__ == "__main__":
    main()
