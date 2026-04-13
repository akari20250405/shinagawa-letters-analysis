from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf  # type: ignore[import-untyped]
from matplotlib import font_manager


# =========================
# 固定：活動期（YYYYMMDDの閉区間）
# =========================
PERIODS: list[tuple[str, int, int]] = [
    ("1.維新期",                   18680101, 18700731),
    ("2.ドイツ滞在期①",            18700801, 18751005),
    ("3.殖産興業官僚期",            18751006, 18860331),
    ("4.ドイツ滞在期②",            18860401, 18870228),
    ("5.帰朝～宮中顧問官期",        18870301, 18890512),
    ("6.宮内省御料局長期",          18890513, 18910531),
    ("7.内務大臣期",                18910601, 18920311),
    ("8.晩年期",                    18920312, 19000226),
]
PERIOD_ORDER = [p[0] for p in PERIODS]


# =========================
# 固定：属性ラベル
# =========================
DEFAULT_ATTR_MAP: dict[str, str] = {
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
    "O":  "その他",
    "ND": "不明",
}

DIGIT_TO_CIRCLED = {
    "1":"①","2":"②","3":"③","4":"④","5":"⑤","6":"⑥","7":"⑦","8":"⑧","9":"⑨","10":"⑩",
    "11":"⑪","12":"⑫","13":"⑬","14":"⑭","15":"⑮","16":"⑯","17":"⑰","18":"⑱","19":"⑲","20":"⑳","21":"㉑",
    "０":"0","１":"1","２":"2","３":"3","４":"4","５":"5","６":"6","７":"7","８":"8","９":"9",
}
FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="属性×活動期×出生地域：可視化 + MCA + ロジット")
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv", help="Input CSV")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default: utf-8-sig)")
    p.add_argument("--outdir", default="outputs/phase3_11_1", help="Output directory")
    p.add_argument("--attr_map", default=None, help="Optional attribute map text file")

    # columns
    p.add_argument("--year_greg", default="年代_西暦")
    p.add_argument("--year_start", default="年代_開始")
    p.add_argument("--year_end", default="年代_終了")
    p.add_argument("--rep_candidates", nargs="*", default=["年代_代表", "年代_代表値", "年代_代表値_西暦"])
    p.add_argument("--month", default="月")
    p.add_argument("--day", default="日")
    p.add_argument("--period_col", default="活動期")

    p.add_argument("--birth_region", default="出生地域_主")
    p.add_argument("--sender", default="発信者")
    p.add_argument("--letter_id", default="整理番号")
    p.add_argument("--attr_cols", nargs="*", default=["属性1", "属性2", "属性3", "属性4"])

    # grouping
    p.add_argument("--top_attrs", type=int, default=12)
    p.add_argument("--top_regions", type=int, default=8)
    p.add_argument("--min_count_keep", type=int, default=10)

    # reference categories (logit)
    p.add_argument("--ref_period", default="3.殖産興業官僚期")
    p.add_argument("--ref_region", default=None, help="Reference birth region (default: most frequent)")
    p.add_argument("--min_pos_logit", type=int, default=20)
    p.add_argument("--min_pos_lr", type=int, default=40)

    # plot options
    p.add_argument("--font", default=None, help="Preferred font family (optional)")
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--no_plots", action="store_true", help="Skip png outputs (csv/log only)")
    p.add_argument("--quiet", action="store_true", help="Less console output")

    return p.parse_args()


# =========================
# 基本ユーティリティ
# =========================
def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def meiji_to_gregorian_year(meiji_year: int) -> int:
    return 1867 + int(meiji_year)  # 明治元年=1868


def _to_float_or_nan(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def to_gregorian_year(x) -> float:
    if pd.isna(x):
        return np.nan
    v = _to_float_or_nan(x)
    if np.isnan(v):
        return np.nan
    v_int = int(np.floor(v + 0.5))
    if 1 <= v_int <= 99:
        return float(meiji_to_gregorian_year(v_int))
    if 1700 <= v_int <= 2100:
        return float(v_int)
    return np.nan


def ymd_int(year, month, day):
    if pd.isna(year) or pd.isna(month) or pd.isna(day):
        return pd.NA
    y = int(year)
    m = int(month)
    d = int(day)
    return y * 10000 + m * 100 + d


def assign_period_from_ymd(ymd_val):
    if pd.isna(ymd_val):
        return None
    ymd_val = int(ymd_val)
    for label, start_ymd, end_ymd in reversed(PERIODS):
        if start_ymd <= ymd_val <= end_ymd:
            return label
    return None


def range_within_single_period(y_start_g, y_end_g):
    if pd.isna(y_start_g) or pd.isna(y_end_g):
        return None
    ys = int(y_start_g)
    ye = int(y_end_g)
    if ye < ys:
        ys, ye = ye, ys
    start_ymd = ys * 10000 + 101
    end_ymd = ye * 10000 + 1231
    for label, p_start, p_end in PERIODS:
        if p_start <= start_ymd and end_ymd <= p_end:
            return label
    return None


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([np.nan] * len(df), index=df.index)


def parse_attr_map_text(text: str) -> dict[str, str]:
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "属性ラベル" in ln and "対応" in ln and len(ln) < 50:
            continue
        lines.append(ln)

    blob = " ".join(lines)
    blob = blob.replace(",", "、").replace("，", "、").replace(";", "、").replace("；", "、")
    parts = [p.strip() for p in blob.split("、") if p.strip()]

    mapping: dict[str, str] = {}
    for p in parts:
        m = re.match(r"^(ND|O)\s*[:：]?\s*(.+)$", p)
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()
            continue
        m = re.match(r"^([①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑])\s*[:：]?\s*(.+)$", p)
        if m:
            mapping[m.group(1).strip()] = m.group(2).strip()
            continue
    return mapping


def normalize_attr_code(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "ND"
    s = str(x).strip()
    if s in ("", "nan", "None"):
        return "ND"
    s2 = "".join(DIGIT_TO_CIRCLED.get(ch, ch) for ch in s)
    if re.fullmatch(r"\d+", s2):
        s2 = str(int(s2))
    if s2 in DIGIT_TO_CIRCLED:
        return DIGIT_TO_CIRCLED[s2]
    return s2


def short_label(attr_code, attr_map: dict[str, str]) -> str:
    code = normalize_attr_code(attr_code)
    name = attr_map.get(code, "")
    return f"{code} {name}" if name else code


def compress_series(s: pd.Series, topk: int | None = None, min_count: int | None = None, other_label: str = "その他") -> pd.Series:
    vc = s.value_counts(dropna=True)
    if topk is not None:
        keep = vc.head(topk).index
    else:
        keep = vc[vc >= (min_count if min_count is not None else 1)].index
    return s.where(s.isin(keep), other_label)


def explode_attributes(df: pd.DataFrame, attr_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    for c in attr_cols:
        tmp[c] = tmp[c].astype(str).str.strip()
        tmp.loc[tmp[c].isin(["", "nan", "None"]), c] = np.nan
        tmp.loc[tmp[c].notna(), c] = tmp.loc[tmp[c].notna(), c].apply(normalize_attr_code)

    long = tmp.melt(
        id_vars=[c for c in tmp.columns if c not in attr_cols],
        value_vars=attr_cols,
        var_name="属性_slot",
        value_name="属性_code",
    )
    long = long[long["属性_code"].notna()].copy()
    return long


def choose_font(preferred: str | None = None) -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [preferred] if preferred else []
    candidates.extend(FONT_CANDIDATES)
    for name in candidates:
        if name and name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "default"


def safe_attr_name(attr_code: str) -> str:
    code = normalize_attr_code(attr_code)
    circled_to_num = {v: int(k) for k, v in DIGIT_TO_CIRCLED.items() if k.isdigit()}
    if code in circled_to_num:
        return f"a{circled_to_num[code]:02d}"
    if code == "O":
        return "a_other"
    if code == "ND":
        return "a_nd"
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", code).strip("_")
    return f"a_{cleaned}" if cleaned else "a_unknown"


# =========================
# 活動期付与
# =========================
def build_effective_ymd_and_period(
    df: pd.DataFrame,
    *,
    year_col_greg: str,
    year_start_col: str,
    year_end_col: str,
    rep_col: str | None,
    month_col: str,
    day_col: str,
    period_col: str,
) -> pd.DataFrame:
    out = df.copy()

    out["_y_greg"] = _num_series(out, year_col_greg).apply(to_gregorian_year)
    out["_ys_greg"] = _num_series(out, year_start_col).apply(to_gregorian_year)
    out["_ye_greg"] = _num_series(out, year_end_col).apply(to_gregorian_year)
    if rep_col and rep_col in out.columns:
        out["_yr_greg"] = _num_series(out, rep_col).apply(to_gregorian_year)
    else:
        out["_yr_greg"] = np.nan

    out["_m"] = pd.to_numeric(out.get(month_col), errors="coerce")
    out["_d"] = pd.to_numeric(out.get(day_col), errors="coerce")

    out["_ymd_eff"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["_period_rule"] = ""
    out[period_col] = pd.Series(pd.NA, index=out.index, dtype="object")

    for r in out.itertuples():
        idx = r.Index
        ys = out.at[idx, "_ys_greg"]
        ye = out.at[idx, "_ye_greg"]
        yr = out.at[idx, "_yr_greg"]
        yg = out.at[idx, "_y_greg"]
        m = out.at[idx, "_m"]
        d = out.at[idx, "_d"]

        has_range = pd.notna(ys) and pd.notna(ye) and int(ys) != int(ye)

        if has_range:
            p = range_within_single_period(ys, ye)
            if p is not None:
                out.at[idx, period_col] = p
                out.at[idx, "_period_rule"] = "range_within"
                continue

            if pd.notna(yr):
                rep_m = int(m) if pd.notna(m) else 7
                rep_d = int(d) if pd.notna(d) else 1
                ymd = ymd_int(int(yr), rep_m, rep_d)
                out.at[idx, "_ymd_eff"] = ymd
                out.at[idx, period_col] = assign_period_from_ymd(ymd)
                out.at[idx, "_period_rule"] = "range_boundary_use_rep"
                continue

            out.at[idx, period_col] = None
            out.at[idx, "_period_rule"] = "range_boundary_no_rep_drop"
            continue

        y0 = yg
        if pd.isna(y0) and pd.notna(ys):
            y0 = ys
        if pd.isna(y0) and pd.notna(yr):
            y0 = yr

        if pd.isna(y0):
            out.at[idx, period_col] = None
            out.at[idx, "_period_rule"] = "no_year_drop"
            continue

        if pd.notna(m) and pd.notna(d):
            ymd = ymd_int(int(y0), int(m), int(d))
            out.at[idx, "_ymd_eff"] = ymd
            out.at[idx, period_col] = assign_period_from_ymd(ymd)
            out.at[idx, "_period_rule"] = "single_year_with_md"
            continue

        p = range_within_single_period(y0, y0)
        if p is not None:
            out.at[idx, period_col] = p
            out.at[idx, "_period_rule"] = "single_year_yearonly_within"
            continue

        if pd.notna(yr):
            ymd = ymd_int(int(yr), 7, 1)
            out.at[idx, "_ymd_eff"] = ymd
            out.at[idx, period_col] = assign_period_from_ymd(ymd)
            out.at[idx, "_period_rule"] = "single_year_yearonly_use_rep"
            continue

        out.at[idx, period_col] = None
        out.at[idx, "_period_rule"] = "single_year_yearonly_ambiguous_drop"

    return out


# =========================
# 図
# =========================
def plot_bar_counts(df_counts: pd.DataFrame, out_png: Path, title: str, ylabel: str, dpi: int = 220):
    df_counts = df_counts.copy()
    if "label" not in df_counts.columns or "count" not in df_counts.columns:
        df_counts.columns = ["label", "count"]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.bar(df_counts["label"], df_counts["count"])
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    for i, v in enumerate(df_counts["count"].values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def plot_heatmap(df_vals: pd.DataFrame, out_png: Path, title: str, cmap: str = "YlOrRd", fmt: str = "{:.1f}", skip_zeros: bool = True, dpi: int = 220):
    plt.figure(figsize=(max(10, df_vals.shape[1] * 1.3), max(6, df_vals.shape[0] * 0.55)))
    img = plt.imshow(df_vals.values, aspect="auto", cmap=cmap)
    plt.colorbar(img, fraction=0.03, pad=0.02)
    plt.xticks(range(df_vals.shape[1]), df_vals.columns, rotation=45, ha="right")
    plt.yticks(range(df_vals.shape[0]), df_vals.index)
    plt.title(title)
    for i in range(df_vals.shape[0]):
        for j in range(df_vals.shape[1]):
            v = df_vals.iat[i, j]
            if skip_zeros and v == 0:
                continue
            plt.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def std_residual_table(ct: pd.DataFrame) -> pd.DataFrame:
    obs = ct.values.astype(float)
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    exp = (row_sum @ col_sum) / (total if total else 1)
    std_resid = (obs - exp) / np.sqrt(np.where(exp == 0, np.nan, exp))
    std_resid = np.nan_to_num(std_resid, nan=0.0)
    return pd.DataFrame(std_resid, index=ct.index, columns=ct.columns)


def save_std_resid_heatmap(std_resid_df: pd.DataFrame, out_png: Path, title: str, dpi: int = 220):
    plt.figure(figsize=(max(10, std_resid_df.shape[1] * 1.3), max(6, std_resid_df.shape[0] * 0.55)))
    vmax = np.max(np.abs(std_resid_df.values)) if std_resid_df.size else 1.0
    img = plt.imshow(std_resid_df.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(img, fraction=0.03, pad=0.02)
    plt.xticks(range(std_resid_df.shape[1]), std_resid_df.columns, rotation=45, ha="right")
    plt.yticks(range(std_resid_df.shape[0]), std_resid_df.index)
    plt.title(title)
    for i in range(std_resid_df.shape[0]):
        for j in range(std_resid_df.shape[1]):
            v = std_resid_df.iat[i, j]
            if abs(v) < 2.0:
                continue
            plt.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


# =========================
# CA / MCA
# =========================
def correspondence_analysis(P: np.ndarray):
    r = P.sum(axis=1, keepdims=True)
    c = P.sum(axis=0, keepdims=True)
    E = r @ c
    S = (P - E) / np.sqrt(np.where(E == 0, np.nan, E))
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    U, s, Vt = np.linalg.svd(S, full_matrices=False)

    r_flat = r.flatten()
    c_flat = c.flatten()
    Dr_inv_sqrt = np.diagflat(np.where(r_flat > 0, 1.0 / np.sqrt(r_flat), 0.0))
    Dc_inv_sqrt = np.diagflat(np.where(c_flat > 0, 1.0 / np.sqrt(c_flat), 0.0))

    F = Dr_inv_sqrt @ U @ np.diag(s)
    G = Dc_inv_sqrt @ Vt.T @ np.diag(s)
    return F, G, s


def plot_mca_categories(cat_df: pd.DataFrame, out_png: Path, title: str, label_top: int = 25, dpi: int = 220):
    plt.figure(figsize=(9, 8))
    ax = plt.gca()
    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.scatter(cat_df["dim1"], cat_df["dim2"], s=20)
    ax.set_title(title)
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")

    cat_df = cat_df.copy()
    cat_df["ctr_sum"] = cat_df["ctr1"].fillna(0) + cat_df["ctr2"].fillna(0)
    top = cat_df.sort_values("ctr_sum", ascending=False).head(label_top)
    for _, r in top.iterrows():
        ax.text(r["dim1"], r["dim2"], r["name"], fontsize=8)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


# =========================
# メイン
# =========================
def main() -> None:
    args = parse_args()

    out_dir = Path(args.outdir)
    ensure_outdir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    used_font = choose_font(args.font)

    log_lines: list[str] = []
    outputs_created: list[Path] = []

    def remember(path: Path):
        outputs_created.append(path)
        return path

    def log(s: str = ""):
        log_lines.append(s)
        if not args.quiet:
            print(s)

    # 属性対応表
    attr_map = DEFAULT_ATTR_MAP.copy()
    attr_map_source = "DEFAULTのみ"
    if args.attr_map:
        attr_map_path = Path(args.attr_map)
        if attr_map_path.exists():
            try:
                fm = parse_attr_map_text(attr_map_path.read_text(encoding="utf-8"))
                attr_map.update(fm)
                attr_map_source = f"DEFAULT + {attr_map_path}"
                log(f"🧾 属性対応表: ファイル補助読み込みOK / 合計={len(attr_map)}")
            except Exception as e:
                log(f"⚠️ 属性対応表ファイルの読み込みに失敗（DEFAULTのみで続行）: {e}")
        else:
            log(f"⚠️ 属性対応表ファイルが見つからないため DEFAULT のみで続行: {attr_map_path}")
    else:
        log(f"🧾 属性対応表: DEFAULTのみ / 件数={len(attr_map)}")

    in_path = Path(args.input)
    df = pd.read_csv(in_path, encoding=args.encoding)
    n_raw = len(df)

    rep_col = next((c for c in args.rep_candidates if c in df.columns), None)
    log(f"🧷 代表年列: {rep_col if rep_col else '（なし）'}")

    need_cols = [args.month, args.day, args.birth_region, args.sender, args.letter_id] + list(args.attr_cols)
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"必要列がない: {miss}\ncolumns={list(df.columns)}")

    for col in [args.birth_region, args.sender, args.letter_id]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = np.nan

    df = build_effective_ymd_and_period(
        df,
        year_col_greg=args.year_greg,
        year_start_col=args.year_start,
        year_end_col=args.year_end,
        rep_col=rep_col,
        month_col=args.month,
        day_col=args.day,
        period_col=args.period_col,
    )

    n_period_ok = int(df[args.period_col].notna().sum())
    n_unclassified = int(df[args.period_col].isna().sum())

    log("📋 活動期付与（確定年/推定幅/境目→代表年）")
    log(f"   元データ: {n_raw} 件")
    log(f"   活動期あり: {n_period_ok} 件 ({(n_period_ok / n_raw * 100 if n_raw else 0):.1f}%)")
    log(f"   活動期なし: {n_unclassified} 件")
    log("")

    rule_counts = df["_period_rule"].value_counts(dropna=False)
    log("🔎 _period_rule 内訳")
    for k, v in rule_counts.items():
        log(f"   {k}: {int(v)}")
    log("")

    df2 = df[df[args.period_col].notna()].copy()
    df2 = df2[df2[args.birth_region].notna()].copy()
    df2 = df2[df2[args.sender].notna()].copy()
    df2 = df2[df2[args.letter_id].notna()].copy()
    df2 = df2.drop_duplicates(subset=[args.letter_id]).copy()

    log("📋 有効データ:")
    log(f"   活動期×出身地域×発信者×整理番号まで有効: {df2.shape[0]} 件")
    log("")

    df2["出生地域_分析"] = compress_series(df2[args.birth_region], topk=args.top_regions, min_count=args.min_count_keep)

    dfA = explode_attributes(df2, list(args.attr_cols))
    dfA["属性_分析"] = compress_series(dfA["属性_code"], topk=args.top_attrs, min_count=args.min_count_keep, other_label="O")
    dfA["属性_label"] = dfA["属性_分析"].apply(lambda x: short_label(x, attr_map))
    log(f"🔎 属性（延べ行数）: {len(dfA)} 件")
    log("")

    # 図7-1〜7-3（延べ）
    bar_png = remember(out_dir / f"phase3_11_1_attr_bar_counts_{ts}.png")
    bar_csv = remember(out_dir / f"phase3_11_1_attr_bar_counts_{ts}.csv")

    vc_attr = dfA["属性_label"].value_counts()
    bar_df = vc_attr.reset_index(name="count")
    bar_df.columns = ["label", "count"]
    bar_df.to_csv(bar_csv, encoding="utf-8-sig", index=False)

    ct = pd.crosstab(dfA["属性_label"], dfA[args.period_col]).reindex(columns=PERIOD_ORDER, fill_value=0)
    col_share = ct.div(ct.sum(axis=0), axis=1).fillna(0) * 100

    heat_share_png = remember(out_dir / f"phase3_11_1_attr_x_period_colshare_{ts}.png")
    heat_share_csv = remember(out_dir / f"phase3_11_1_attr_x_period_colshare_{ts}.csv")
    col_share.to_csv(heat_share_csv, encoding="utf-8-sig")

    std_resid_df = std_residual_table(ct)
    heat_resid_png = remember(out_dir / f"phase3_11_1_attr_x_period_stdresid_{ts}.png")
    heat_resid_csv = remember(out_dir / f"phase3_11_1_attr_x_period_stdresid_{ts}.csv")
    std_resid_df.to_csv(heat_resid_csv, encoding="utf-8-sig")

    if not args.no_plots:
        plot_bar_counts(bar_df, bar_png, "図7-1 属性別の書簡通数（延べ件数）", "書簡通数（属性の延べ件数）", dpi=args.dpi)
        plot_heatmap(col_share, heat_share_png, "図7-2 属性×活動期（列百分率：各期の構成比%）", cmap="YlOrRd", fmt="{:.1f}", skip_zeros=True, dpi=args.dpi)
        save_std_resid_heatmap(std_resid_df, heat_resid_png, "図7-3 属性×活動期（標準化残差：期待値との差）", dpi=args.dpi)

    # 参考：整理番号ユニーク（属性もユニーク）
    dfA_u = dfA.drop_duplicates(subset=[args.letter_id, "属性_分析"]).copy()

    bar_u_png = remember(out_dir / f"phase3_11_1_attr_bar_counts_ref_unique_{ts}.png")
    bar_u_csv = remember(out_dir / f"phase3_11_1_attr_bar_counts_ref_unique_{ts}.csv")
    vc_attr_u = dfA_u["属性_label"].value_counts()
    bar_u_df = vc_attr_u.reset_index(name="count")
    bar_u_df.columns = ["label", "count"]
    bar_u_df.to_csv(bar_u_csv, encoding="utf-8-sig", index=False)

    ct_u = pd.crosstab(dfA_u["属性_label"], dfA_u[args.period_col]).reindex(columns=PERIOD_ORDER, fill_value=0)

    letters_per_period = df2.groupby(args.period_col)[args.letter_id].nunique().reindex(PERIOD_ORDER).fillna(0).astype(int)
    share_u = ct_u.astype(float).copy()
    for p in PERIOD_ORDER:
        denom = float(letters_per_period.get(p, 0))
        share_u[p] = (share_u[p] / denom * 100) if denom > 0 else 0.0

    heat_u_png = remember(out_dir / f"phase3_11_1_attr_x_period_share_ref_unique_{ts}.png")
    heat_u_csv = remember(out_dir / f"phase3_11_1_attr_x_period_share_ref_unique_{ts}.csv")
    share_u.to_csv(heat_u_csv, encoding="utf-8-sig")

    std_resid_u_df = std_residual_table(ct_u)
    heat_resid_u_png = remember(out_dir / f"phase3_11_1_attr_x_period_stdresid_ref_unique_{ts}.png")
    heat_resid_u_csv = remember(out_dir / f"phase3_11_1_attr_x_period_stdresid_ref_unique_{ts}.csv")
    std_resid_u_df.to_csv(heat_resid_u_csv, encoding="utf-8-sig")

    if not args.no_plots:
        plot_bar_counts(bar_u_df, bar_u_png, "図7-1（参考） 属性別の書簡通数（整理番号ユニーク）", "書簡通数（整理番号ユニーク）", dpi=args.dpi)
        plot_heatmap(share_u, heat_u_png, "図7-2（参考） 属性×活動期（書簡ベース割合：各期の書簡に占める割合%）", cmap="YlOrRd", fmt="{:.1f}", skip_zeros=True, dpi=args.dpi)
        save_std_resid_heatmap(std_resid_u_df, heat_resid_u_png, "図7-3（参考） 属性×活動期（標準化残差：書簡ベース）", dpi=args.dpi)

    log("📝 参考図について：列合計が100%を超えうる（複数属性が同一書簡に載るため）。正常。")
    log("")

    # MCA
    base = df2[[args.letter_id, args.period_col, "出生地域_分析", args.sender]].copy()
    attrs_per_letter = dfA_u.groupby(args.letter_id)["属性_分析"].apply(lambda x: sorted(set(x))).rename("attrs")
    base = base.merge(attrs_per_letter, on=args.letter_id, how="left")
    base["attrs"] = base["attrs"].apply(lambda x: x if isinstance(x, list) else [])

    period_cats = [f"Period:{p}" for p in PERIOD_ORDER]
    region_cats = [f"Region:{r}" for r in base["出生地域_分析"].value_counts().index.tolist()]
    attr_cats_raw = dfA["属性_分析"].value_counts().index.tolist()
    attr_cats = [f"Attr:{short_label(a, attr_map)}" for a in attr_cats_raw]

    X = pd.DataFrame(0, index=base.index, columns=period_cats + region_cats + attr_cats, dtype=float)

    period_to_col = {p: f"Period:{p}" for p in PERIOD_ORDER}
    for i, p in enumerate(base[args.period_col].values):
        col = period_to_col.get(p)
        if col in X.columns:
            X.iat[i, X.columns.get_loc(col)] = 1.0

    for i, r in enumerate(base["出生地域_分析"].values):
        col = f"Region:{r}"
        if col in X.columns:
            X.iat[i, X.columns.get_loc(col)] = 1.0

    attr_to_col = {a: f"Attr:{short_label(a, attr_map)}" for a in attr_cats_raw}
    for i, attrs in enumerate(base["attrs"].values):
        for a in attrs:
            col = attr_to_col.get(a)
            if col in X.columns:
                X.iat[i, X.columns.get_loc(col)] = 1.0

    col_sum = X.sum(axis=0)
    keep_cols = col_sum[col_sum > 0].index.tolist()
    X2 = X[keep_cols].copy()

    if X2.shape[1] >= 2:
        grand = X2.values.sum()
        P = X2.values / (grand if grand else 1.0)
        F, G, s = correspondence_analysis(P)

        cat_names = X2.columns.tolist()
        cat_type = []
        for nm in cat_names:
            if nm.startswith("Period:"):
                cat_type.append("Period")
            elif nm.startswith("Region:"):
                cat_type.append("Region")
            else:
                cat_type.append("Attr")

        c_mass = P.sum(axis=0)
        eig = s ** 2
        dim1, dim2 = 0, 1

        if len(eig) < 2 or eig[0] == 0 or eig[1] == 0:
            log("⚠️ MCA: 次元が足りない/固有値0でスキップ")
        else:
            ctr1 = c_mass * (G[:, dim1] ** 2) / eig[dim1]
            ctr2 = c_mass * (G[:, dim2] ** 2) / eig[dim2]

            cat_df = pd.DataFrame({
                "name": cat_names,
                "type": cat_type,
                "mass": c_mass,
                "dim1": G[:, dim1],
                "dim2": G[:, dim2],
                "ctr1": ctr1,
                "ctr2": ctr2,
            })

            mca_csv = remember(out_dir / f"phase3_11_1_mca_categories_{ts}.csv")
            mca_png = remember(out_dir / f"phase3_11_1_mca_categories_{ts}.png")
            cat_df.to_csv(mca_csv, encoding="utf-8-sig", index=False)
            if not args.no_plots:
                plot_mca_categories(cat_df, mca_png, "図7-4 MCA（カテゴリ座標：Period/Region/Attr）", label_top=30, dpi=args.dpi)
    else:
        log("⚠️ MCA: 有効カテゴリが少なすぎるのでスキップ")

    # ロジスティック回帰（書簡ベース）
    base[args.period_col] = pd.Categorical(base[args.period_col], categories=PERIOD_ORDER, ordered=True)
    base["出生地域_分析"] = pd.Categorical(base["出生地域_分析"])

    if args.ref_region is None:
        ref_region = base["出生地域_分析"].value_counts().index[0]
    else:
        ref_region = args.ref_region

    safe_map: dict[str, str] = {}
    used_safe = set()
    for a in attr_cats_raw:
        base_name = safe_attr_name(a)
        safe_name = base_name
        k = 2
        while safe_name in used_safe:
            safe_name = f"{base_name}_{k}"
            k += 1
        used_safe.add(safe_name)
        safe_map[a] = safe_name

    results = []
    target_attrs = attr_cats_raw
    logit_success = 0

    for a in target_attrs:
        yname = safe_map[a]
        base[yname] = base["attrs"].apply(lambda lst: 1 if a in lst else 0)

        pos = int(base[yname].sum())
        if pos < args.min_pos_logit:
            continue

        formula = (
            f"{yname} ~ "
            f"C(Q('{args.period_col}'), Treatment(reference='{args.ref_period}')) + "
            f"C(Q('出生地域_分析'), Treatment(reference='{ref_region}'))"
        )

        try:
            model = smf.logit(formula, data=base).fit(disp=0)
            if hasattr(model, "get_robustcov_results"):
                rob = model.get_robustcov_results(cov_type="cluster", groups=base[args.sender])
                params = rob.params
                bse = rob.bse
                pvals = rob.pvalues
            else:
                params = model.params
                bse = model.bse
                pvals = model.pvalues
            logit_success += 1

            for term in params.index:
                if term == "Intercept":
                    continue
                beta = params[term]
                se = bse[term]
                or_ = np.exp(beta)
                lo = np.exp(beta - 1.96 * se)
                hi = np.exp(beta + 1.96 * se)
                results.append({
                    "attr_code": a,
                    "attr_safe": yname,
                    "attr_label": short_label(a, attr_map),
                    "term": term,
                    "beta": beta,
                    "se_cluster": se,
                    "OR": or_,
                    "CI95_low": lo,
                    "CI95_high": hi,
                    "p_cluster": pvals[term],
                    "pos_n": pos,
                    "N": len(base),
                })
        except Exception as e:
            log(f"⚠️ logit失敗 attr={a} safe={yname} pos={pos}: {e}")

    res_df = pd.DataFrame(results)
    logit_csv = remember(out_dir / f"phase3_11_1_logit_attr_period_region_{ts}.csv")
    res_df.to_csv(logit_csv, encoding="utf-8-sig", index=False)

    # safe名対応表
    safe_map_df = pd.DataFrame(
        [{"attr_code": k, "attr_safe": v, "attr_label": short_label(k, attr_map)} for k, v in safe_map.items()]
    )
    safe_map_csv = remember(out_dir / f"phase3_11_1_attr_safe_name_map_{ts}.csv")
    safe_map_df.to_csv(safe_map_csv, encoding="utf-8-sig", index=False)

    # 交互作用LR
    inter_rows = []
    scipy_ok = False
    try:
        from scipy.stats import chi2
        scipy_ok = True
    except Exception:
        log("⚠️ scipy が無いので交互作用LRはスキップ（pip install scipy で有効化）")

    lr_success = 0
    if scipy_ok:
        for a in target_attrs:
            yname = safe_map[a]
            if yname not in base.columns:
                continue
            pos = int(base[yname].sum())
            if pos < args.min_pos_lr:
                continue

            f0 = (
                f"{yname} ~ "
                f"C(Q('{args.period_col}'), Treatment(reference='{args.ref_period}')) + "
                f"C(Q('出生地域_分析'), Treatment(reference='{ref_region}'))"
            )
            f1 = (
                f"{yname} ~ "
                f"C(Q('{args.period_col}'), Treatment(reference='{args.ref_period}')) * "
                f"C(Q('出生地域_分析'), Treatment(reference='{ref_region}'))"
            )
            try:
                m0 = smf.logit(f0, data=base).fit(disp=0)
                m1 = smf.logit(f1, data=base).fit(disp=0)
                lr = 2 * (m1.llf - m0.llf)
                df_diff = int(m1.df_model - m0.df_model)
                p_lr = 1 - chi2.cdf(lr, df_diff) if df_diff > 0 else np.nan
                lr_success += 1
                inter_rows.append({
                    "attr_code": a,
                    "attr_safe": yname,
                    "attr_label": short_label(a, attr_map),
                    "pos_n": pos,
                    "LR": lr,
                    "df": df_diff,
                    "p_LR": p_lr,
                })
            except Exception as e:
                log(f"⚠️ LR失敗 attr={a} safe={yname} pos={pos}: {e}")

    inter_df = pd.DataFrame(inter_rows)
    if not inter_df.empty:
        inter_df = inter_df.sort_values("p_LR")
    else:
        inter_df = pd.DataFrame(columns=["attr_code", "attr_safe", "attr_label", "pos_n", "LR", "df", "p_LR"])

    inter_csv = remember(out_dir / f"phase3_11_1_logit_interaction_lr_{ts}.csv")
    inter_df.to_csv(inter_csv, encoding="utf-8-sig", index=False)

    # ログ保存
    log("")
    log("📌 実行条件")
    log(f"   入力: {in_path}")
    log(f"   encoding: {args.encoding}")
    log(f"   outdir: {out_dir}")
    log(f"   属性対応表: {attr_map_source}")
    log(f"   使用フォント: {used_font}")
    log(f"   dpi: {args.dpi}")
    log(f"   no_plots: {args.no_plots}")
    log(f"   top_attrs: {args.top_attrs}")
    log(f"   top_regions: {args.top_regions}")
    log(f"   min_count_keep: {args.min_count_keep}")
    log(f"   ref_period: {args.ref_period}")
    log(f"   ref_region: {ref_region}")
    log(f"   min_pos_logit: {args.min_pos_logit}")
    log(f"   min_pos_lr: {args.min_pos_lr}")
    log(f"   使用列: year_greg={args.year_greg}, year_start={args.year_start}, year_end={args.year_end}, rep_col={rep_col}, month={args.month}, day={args.day}, period_col={args.period_col}, birth_region={args.birth_region}, sender={args.sender}, letter_id={args.letter_id}, attr_cols={args.attr_cols}")
    log(f"   ロジット成功属性数: {logit_success}")
    log(f"   LR成功属性数: {lr_success}")
    log("")
    log("📂 出力ファイル")
    for p in outputs_created:
        if args.no_plots and p.suffix.lower() == ".png":
            continue
        log(f"   {p}")

    log_txt = out_dir / f"phase3_11_1_attr_models_log_{ts}.txt"
    log_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("✅ 出力完了")
    print(f"  ロジット: {logit_csv}")
    print(f"  交互作用LR: {inter_csv}")
    print(f"  safe名対応: {safe_map_csv}")
    print(f"  log: {log_txt}")


if __name__ == "__main__":
    main()
