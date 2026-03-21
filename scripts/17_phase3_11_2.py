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
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy.stats import norm  # type: ignore


# =========================
# defaults / constants
# =========================
DEFAULT_INPUT = "outputs/cleaning/shinagawa_letters_cleaned.csv"
DEFAULT_OUTDIR = "outputs/phase3_11_2"

PERIOD_COL = "活動期"
BIRTH_REGION_COL = "出生地域_主"
SENDER_COL = "発信者"
LETTER_ID_COL = "整理番号"

ERA_START_GREG = {
    "M": 1868, "明治": 1868,
    "T": 1912, "大正": 1912,
    "S": 1926, "昭和": 1926,
    "K": 1865, "慶応": 1865,
}

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

# Example:
# python scripts/17_phase3_11_2.py --input outputs/cleaning/shinagawa_letters_cleaned.csv --outdir outputs/phase3_11_2

# =========================
# util
# =========================
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
    parts = [p.strip() for p in re.split(r"[、,/\s]+", txt) if p.strip()]
    return parts


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


def range_single_period(start_y: int, end_y: int) -> Optional[str]:
    hit = []
    for pid, start, end in PERIODS:
        ys, ye = start.year, end.year
        if start_y >= ys and end_y <= ye:
            hit.append(pid)
    if len(hit) == 1:
        return hit[0]
    return None


def build_effective_period(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")

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
            # 代表年が和暦なら era から西暦化
            rep_greg = pd.Series(
                [era_to_gregorian_year(e, y) for e, y in zip(era_col, rep_raw)],
                index=df.index,
                dtype="float",
            )

    year_for_point = year_greg.where(year_greg.notna(), rep_greg)
    year_point = pd.Series(np.floor(year_for_point + 0.5), index=df.index)

    month = pd.to_numeric(df.get("月"), errors="coerce")
    day = pd.to_numeric(df.get("日"), errors="coerce")

    start_year = pd.to_numeric(df.get("年代_開始"), errors="coerce")
    end_year = pd.to_numeric(df.get("年代_終了"), errors="coerce")

    for idx in df.index:
        # 年代幅優先: 単一期ならその期
        sy, ey = start_year.get(idx), end_year.get(idx)
        if pd.notna(sy) and pd.notna(ey):
            single = range_single_period(int(np.floor(sy + 0.5)), int(np.floor(ey + 0.5)))
            if single is not None:
                out.at[idx] = single
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
                continue

        if pd.notna(m) and pd.isna(d) and is_valid_month_day(m):
            # 月日不完備は安全側に倒す: 15日で仮置きしつつ境界月なら保留
            boundary_months = {(p[1].year, p[1].month) for p in PERIODS[1:]} | {(p[2].year, p[2].month) for p in PERIODS[:-1]}
            if (int(yp), int(m)) in boundary_months:
                continue
            dt = pd.Timestamp(int(yp), int(m), 15)
            pid = pid_for_ymd(dt)
            if pid is not None:
                out.at[idx] = pid
                continue

        # 年のみ: 年央で代表
        dt = pd.Timestamp(int(yp), 7, 1)
        pid = pid_for_ymd(dt)
        if pid is not None:
            out.at[idx] = pid

    return out


def prune_levels_for_y(df: pd.DataFrame, ycol: str, catcol: str, min_n: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    g = df.groupby(catcol, observed=False)[ycol].agg(["count", "sum"])
    keep = g[(g["count"] >= min_n) & (g["sum"] > 0) & (g["sum"] < g["count"])].index.tolist()
    out = df[df[catcol].isin(keep)].copy()
    return out, keep


def prune_cells_for_y(df: pd.DataFrame, ycol: str, rowcat: str, colcat: str, min_n: int = 20) -> pd.DataFrame:
    g = df.groupby([rowcat, colcat], observed=False)[ycol].agg(["count", "sum"]).reset_index()
    g["all0_or_all1"] = (g["sum"] == 0) | (g["sum"] == g["count"])
    keep_cells = g[(g["count"] >= min_n) & (~g["all0_or_all1"])][[rowcat, colcat]]
    if keep_cells.empty:
        return df.iloc[0:0].copy()
    return df.merge(keep_cells, on=[rowcat, colcat], how="inner")


def fit_glm_cluster_params(formula: str, data: pd.DataFrame, group_col: str, maxiter: int = 300):
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
    p = argparse.ArgumentParser(description="Phase3_11_2 public: target-attribute GLM/logit with rescue.")
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--attr_map", default=None, help="Optional attribute map text/csv/tsv file.")
    p.add_argument("--top_attrs", type=int, default=12)
    p.add_argument("--top_regions", type=int, default=8)
    p.add_argument("--min_count_keep", type=int, default=10)
    p.add_argument("--min_pos", type=int, default=20, help="Minimum positive cases to fit model.")
    p.add_argument("--min_level_n", type=int, default=30, help="Minimum count per period/region level.")
    p.add_argument("--min_cell_n", type=int, default=20, help="Minimum count per period-region cell.")
    p.add_argument("--ref_period", default="3.殖産興業官僚期")
    p.add_argument("--ref_region", default="中国")
    p.add_argument("--target_attrs", nargs="*", default=DEFAULT_TARGET_ATTRS)
    p.add_argument("--rescue_attrs", nargs="*", default=DEFAULT_RESCUE_ATTRS)
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

    log_lines: List[str] = []

    def log(s: str = "") -> None:
        log_lines.append(s)
        print(s)

    log_txt = out_dir / f"phase3_11_2_logit_log_{ts}.txt"
    result_csv = out_dir / f"phase3_11_2_logit_target_attrs_{ts}.csv"
    safe_csv = out_dir / f"phase3_11_2_attr_safe_name_map_{ts}.csv"

    # attr map
    attr_map = load_attr_map(args.attr_map)
    log("📘 属性対応表")
    log(f"   attr_map指定: {args.attr_map if args.attr_map else '(なし: DEFAULTのみ)'}")
    log(f"   attr_map件数: {len(attr_map)}")
    log("")

    # input
    df = pd.read_csv(args.input, encoding=args.encoding)
    n_raw = len(df)

    req = [args.birth_region_col, args.sender, args.letter_id_col]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise KeyError(f"必須列がありません: {miss}")

    # period
    df[args.period_col] = build_effective_period(df)
    n_period_ok = int(df[args.period_col].notna().sum())

    log("📋 実行条件")
    log(f"   input: {args.input}")
    log(f"   encoding: {args.encoding}")
    log(f"   outdir: {out_dir}")
    log(f"   top_attrs: {args.top_attrs}")
    log(f"   top_regions: {args.top_regions}")
    log(f"   min_count_keep: {args.min_count_keep}")
    log(f"   min_pos: {args.min_pos}")
    log(f"   min_level_n: {args.min_level_n}")
    log(f"   min_cell_n: {args.min_cell_n}")
    log(f"   ref_period(希望): {args.ref_period}")
    log(f"   ref_region(希望): {args.ref_region}")
    log(f"   target_attrs: {args.target_attrs}")
    log(f"   rescue_attrs: {args.rescue_attrs}")
    log("")
    log("📋 活動期付与")
    log(f"   元データ: {n_raw} 件")
    log(f"   活動期あり: {n_period_ok} 件")
    log("")

    # required filters
    df2 = df[df[args.period_col].notna()].copy()
    df2 = df2[df2[args.birth_region_col].notna()].copy()
    df2 = df2[df2[args.sender].notna()].copy()
    df2 = df2[df2[args.letter_id_col].notna()].copy()
    log("📋 有効データ")
    log(f"   活動期 × 地域 × 発信者 × 整理番号まで有効: {len(df2)} 件")
    log("")

    # compress region
    df2["出生地域_分析"] = compress_series(
        df2[args.birth_region_col], topk=args.top_regions, min_count=args.min_count_keep
    )

    # explode attrs (canonical code columns only)
    dfA = explode_attributes(df2)
    dfA["属性_分析"] = dfA["属性_code"].fillna("ND").astype(str)
    dfA["属性_label"] = dfA["属性_分析"].apply(lambda x: short_label(x, attr_map))

    # one letter = one row
    base = df2[[args.letter_id_col, args.period_col, "出生地域_分析", args.sender]].copy()
    attrs_per_letter = (
        dfA.groupby(args.letter_id_col, observed=False)["属性_分析"]
        .apply(lambda x: sorted(set(x)))
        .rename("attrs")
    )
    base = base.merge(attrs_per_letter, on=args.letter_id_col, how="left")
    base["attrs"] = base["attrs"].apply(lambda x: x if isinstance(x, list) else [])

    b = base[[args.period_col, "出生地域_分析", args.sender, "attrs"]].copy()
    b = b.rename(columns={args.period_col: "period", "出生地域_分析": "region", args.sender: "sender"})
    b["period"] = pd.Categorical(b["period"], categories=PERIOD_ORDER, ordered=True)
    b["region"] = pd.Categorical(b["region"])

    ref_region = args.ref_region if args.ref_region is not None else (
        b["region"].value_counts().index[0] if len(b) else None
    )

    target_attrs = [normalize_text(a) for a in args.target_attrs]
    safe_map = safe_name_map(target_attrs)
    safe_rows = [
        {"attr_code": a, "attr_label": short_label(a, attr_map), "safe_name": safe_map[a]}
        for a in target_attrs
    ]
    pd.DataFrame(safe_rows).to_csv(safe_csv, index=False, encoding="utf-8-sig")

    for a in target_attrs:
        ycol = f"Y_{safe_map[a]}"
        b[ycol] = b["attrs"].apply(lambda lst: 1 if a in lst else 0)

    pos_info = [(a, int(b[f"Y_{safe_map[a]}"].sum())) for a in target_attrs]
    log(f"🎯 ロジット対象属性: {pos_info}")
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
        tmp, keep_p = prune_levels_for_y(tmp, ycol, "period", min_n=args.min_level_n)
        tmp, keep_r = prune_levels_for_y(tmp, ycol, "region", min_n=args.min_level_n)
        tmp = prune_cells_for_y(tmp, ycol, "period", "region", min_n=args.min_cell_n)

        if hasattr(tmp["period"], "cat"):
            tmp["period"] = tmp["period"].cat.remove_unused_categories()
        if hasattr(tmp["region"], "cat"):
            tmp["region"] = tmp["region"].cat.remove_unused_categories()

        if len(tmp) == 0:
            log(f"❌ attr={a}: 間引き後にデータが消滅（スカスカ/分離）")
            continue

        ref_p = args.ref_period if args.ref_period in list(tmp["period"].cat.categories) else tmp["period"].value_counts().index[0]
        ref_r = ref_region if (ref_region is not None and ref_region in list(tmp["region"].cat.categories)) else tmp["region"].value_counts().index[0]

        f_main = (
            f"{ycol} ~ "
            f"C(period, Treatment(reference='{ref_p}')) + "
            f"C(region, Treatment(reference='{ref_r}'))"
        )
        f_rescue = (
            f"{ycol} ~ "
            f"C(period, Treatment(reference='{ref_p}'))"
        )

        model_note = ""
        try:
            params, bse, pvals, converged, note = fit_glm_cluster_params(f_main, tmp, "sender", maxiter=300)
            model_note = "main(period+region)"
        except Exception as e:
            if a in set(args.rescue_attrs):
                rescue_notes.append(
                    {
                        "attr_code": a,
                        "attr_label": short_label(a, attr_map),
                        "reason": f"main failed: {type(e).__name__}",
                        "policy": "論文再現方針として、period+region 主モデルで分離・収束不良が生じやすい属性について period のみの rescue model を許可した。",
                    }
                )
                log(f"⚠️ attr={a} main失敗({type(e).__name__}) → rescue(periodのみ)へ")
                try:
                    params, bse, pvals, converged, note = fit_glm_cluster_params(f_rescue, tmp, "sender", maxiter=300)
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
                    "ref_region": ref_r,
                    "keep_period_levels": "|".join(map(str, keep_p)),
                    "keep_region_levels": "|".join(map(str, keep_r)),
                    "model_note": model_note + (f" / {note}" if note else ""),
                }
            )

    res_df = pd.DataFrame(results)
    res_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    log(f"✅ ロジットCSV出力: {result_csv} / rows={len(res_df)}")
    log(f"✅ safe名対応表出力: {safe_csv}")

    if rescue_notes:
        log("")
        log("📌 rescue model 注記")
        for r in rescue_notes:
            log(f"   attr={r['attr_code']} ({r['attr_label']}): {r['policy']} / reason={r['reason']}")

    log("")
    log("📦 出力一覧")
    log(f"   log: {log_txt}")
    log(f"   result_csv: {result_csv}")
    log(f"   safe_map_csv: {safe_csv}")

    log_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"🧾 log保存: {log_txt}")


if __name__ == "__main__":
    main()
