from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 活動期（YYYYMMDD 閉区間）: 研究仕様として固定
# =========================
PERIODS: List[Tuple[str, int, int]] = [
    ("1.維新期", 18680101, 18700731),
    ("2.ドイツ滞在期①", 18700801, 18751005),
    ("3.殖産興業官僚期", 18751006, 18860331),
    ("4.ドイツ滞在期②", 18860401, 18870228),
    ("5.帰朝～宮中顧問官期", 18870301, 18890512),
    ("6.宮内省御料局長期", 18890513, 18910531),
    ("7.内務大臣期", 18910601, 18920311),
    ("8.晩年期", 18920312, 19000226),
]
PERIOD_ORDER = [p[0] for p in PERIODS]


# =========================
# ユーティリティ：四捨五入（0.5切り上げ）Int64
# =========================
def round_half_up_to_int(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.floor(x + 0.5).astype("Int64")


# =========================
# 期の長さ（日数→年換算）
# =========================
def duration_years_ymd(start_ymd: int, end_ymd: int) -> float:
    s = pd.to_datetime(str(start_ymd), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(str(end_ymd), format="%Y%m%d", errors="coerce")
    if pd.isna(s) or pd.isna(e) or e < s:
        return 1 / 365.2425
    days = (e - s).days + 1  # 閉区間
    return max(days / 365.2425, 1 / 365.2425)


DURATIONS = {label: duration_years_ymd(s, e) for (label, s, e) in PERIODS}


# =========================
# フォント
# =========================
def configure_matplotlib_font(requested_font: Optional[str]) -> str:
    """公開用に、単一フォント固定ではなく候補順で設定する。"""
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:
        available = set()

    candidates: List[str] = []
    if requested_font:
        candidates.append(requested_font)
    candidates.extend([
        "MS Gothic",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
        "Hiragino Sans",
        "sans-serif",
    ])

    chosen = "sans-serif"
    for font in candidates:
        if not available or font in available or font == "sans-serif":
            chosen = font
            break

    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return chosen


# =========================
# プロット類
# =========================
def plot_heatmap(
    df_vals: pd.DataFrame,
    out_png: Path,
    title: str,
    cmap: str,
    fmt: str = "{v:.1f}",
    skip_zero: bool = True,
    dpi: int = 220,
) -> None:
    plt.figure(figsize=(max(10, df_vals.shape[1] * 1.3), max(6, df_vals.shape[0] * 0.55)))
    img = plt.imshow(df_vals.values, aspect="auto", cmap=cmap)
    plt.colorbar(img, fraction=0.03, pad=0.02)
    plt.xticks(range(df_vals.shape[1]), df_vals.columns, rotation=45, ha="right")
    plt.yticks(range(df_vals.shape[0]), df_vals.index)
    plt.title(title)
    plt.tight_layout()

    for i in range(df_vals.shape[0]):
        for j in range(df_vals.shape[1]):
            v = df_vals.iat[i, j]
            if np.isnan(v):
                continue
            if skip_zero and v == 0:
                continue
            plt.text(j, i, fmt.format(v=v), ha="center", va="center", fontsize=8, color="black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def plot_stacked_share(col_share: pd.DataFrame, out_png: Path, title: str, dpi: int = 220) -> None:
    plt.figure(figsize=(12, 6))
    ax = col_share.T.plot(kind="bar", stacked=True, ax=plt.gca())
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("構成比（%）")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    threshold = 5.0
    for container in ax.containers:
        labels = [(f"{v:.1f}%" if v >= threshold else "") for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def plot_residual_heatmap(df_vals: pd.DataFrame, out_png: Path, title: str, dpi: int = 220) -> None:
    plt.figure(figsize=(max(10, df_vals.shape[1] * 1.3), max(6, df_vals.shape[0] * 0.55)))
    vmax = np.nanmax(np.abs(df_vals.values)) if np.isfinite(df_vals.values).any() else 1.0
    img = plt.imshow(df_vals.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(img, fraction=0.03, pad=0.02)
    plt.xticks(range(df_vals.shape[1]), df_vals.columns, rotation=45, ha="right")
    plt.yticks(range(df_vals.shape[0]), df_vals.index)
    plt.title(title)
    plt.tight_layout()

    for i in range(df_vals.shape[0]):
        for j in range(df_vals.shape[1]):
            v = df_vals.iat[i, j]
            if np.isnan(v):
                continue
            if abs(v) < 2.0:
                continue
            plt.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color="black")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi)
    plt.close()


# =========================
# 集中度（TopK送信者シェア）
# =========================
def topk_letter_share(g: pd.DataFrame, sender_col: str, k: int = 5) -> float:
    vc = g.groupby(sender_col).size().sort_values(ascending=False)
    total_letters = vc.sum()
    if total_letters == 0:
        return np.nan
    return float(vc.head(k).sum() / total_letters)


# =========================
# 活動期付与（点判定 → 年代幅containで上書き）
# =========================
def assign_period(
    df2: pd.DataFrame,
    periods: List[Tuple[str, int, int]],
    *,
    year_col: str,
    month_col: Optional[str],
    day_col: Optional[str],
    year_start_col: str,
    year_end_col: str,
    era_col: str,
    rep_year_candidates: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    labels = [lab for lab, _, _ in periods]

    rep_col = next((c for c in rep_year_candidates if c in df2.columns), None)
    era_offset = {"M": 1867, "K": 1864, "S": 1925}
    offset = (
        df2.get(era_col).map(era_offset)
        if era_col in df2.columns
        else pd.Series(pd.NA, index=df2.index, dtype="Float64")
    )

    year_point_raw = (
        pd.to_numeric(df2.get(year_col), errors="coerce")
        if year_col in df2.columns
        else pd.Series(np.nan, index=df2.index)
    )

    rep_raw = pd.to_numeric(df2.get(rep_col), errors="coerce") if rep_col else pd.Series(np.nan, index=df2.index)
    if rep_col == "年代_代表値_西暦":
        rep_year_greg = rep_raw
    else:
        rep_year_greg = rep_raw + offset

    year_for_point_raw = year_point_raw.where(year_point_raw.notna(), rep_year_greg)
    df2["_year"] = round_half_up_to_int(year_for_point_raw)

    if month_col and month_col in df2.columns:
        df2["_month"] = round_half_up_to_int(df2[month_col])
    else:
        df2["_month"] = pd.Series(pd.NA, index=df2.index, dtype="Int64")

    if day_col and day_col in df2.columns:
        df2["_day"] = round_half_up_to_int(df2[day_col])
    else:
        df2["_day"] = pd.Series(pd.NA, index=df2.index, dtype="Int64")

    start_raw = (
        pd.to_numeric(df2.get(year_start_col), errors="coerce")
        if year_start_col in df2.columns
        else pd.Series(np.nan, index=df2.index)
    )
    end_raw = (
        pd.to_numeric(df2.get(year_end_col), errors="coerce")
        if year_end_col in df2.columns
        else pd.Series(np.nan, index=df2.index)
    )
    is_range = start_raw.notna() & end_raw.notna()

    start_year_g = round_half_up_to_int(start_raw + offset)
    end_year_g = round_half_up_to_int(end_raw + offset)
    has_any_year = df2["_year"].notna() | is_range | rep_year_greg.notna()

    df2["_ymd"] = pd.Series(pd.NA, index=df2.index, dtype="Int64")
    mask_ymd = df2["_year"].notna() & df2["_month"].notna() & df2["_day"].notna()
    df2.loc[mask_ymd, "_ymd"] = (
        df2.loc[mask_ymd, "_year"] * 10000
        + df2.loc[mask_ymd, "_month"] * 100
        + df2.loc[mask_ymd, "_day"]
    ).astype("Int64")

    df2["_ym"] = pd.Series(pd.NA, index=df2.index, dtype="Int64")
    mask_ym = df2["_year"].notna() & df2["_month"].notna()
    df2.loc[mask_ym, "_ym"] = (df2.loc[mask_ym, "_year"] * 100 + df2.loc[mask_ym, "_month"]).astype("Int64")

    boundary_years = {1870, 1875, 1886, 1887, 1889, 1891, 1892, 1900}
    day_sensitive_months = {187510, 188706, 188905, 189203, 190002}

    period = pd.Series(pd.NA, index=df2.index, dtype="object")

    conds_ymd = [df2["_ymd"].between(s, e) for _, s, e in periods]
    sel_ymd = np.select([c.fillna(False) for c in conds_ymd], labels, default=None)
    sel_ymd_s = pd.Series(sel_ymd, index=df2.index)
    mask_assign_ymd = df2["_ymd"].notna() & sel_ymd_s.notna()
    period.loc[mask_assign_ymd] = sel_ymd_s.loc[mask_assign_ymd]

    mask_ym_only = df2["_ym"].notna() & df2["_ymd"].isna() & period.isna()
    mask_hold_m = mask_ym_only & df2["_ym"].isin(list(day_sensitive_months))
    mask_ym_safe = mask_ym_only & (~mask_hold_m)

    periods_ym = [(lab, s // 100, e // 100) for (lab, s, e) in periods]
    conds_ym = [df2["_ym"].between(s_ym, e_ym) for _, s_ym, e_ym in periods_ym]
    sel_ym = np.select([c.fillna(False) for c in conds_ym], labels, default=None)
    sel_ym_s = pd.Series(sel_ym, index=df2.index)
    mask_assign_ym = mask_ym_safe & sel_ym_s.notna()
    period.loc[mask_assign_ym] = sel_ym_s.loc[mask_assign_ym]

    mask_y_only = df2["_year"].notna() & df2["_month"].isna() & period.isna()
    mask_y_safe = mask_y_only & (~df2["_year"].isin(list(boundary_years)))

    yy = df2["_year"]
    sel_y = pd.Series(pd.NA, index=df2.index, dtype="object")
    for lab, s, e in periods:
        sy, ey = (s // 10000), (e // 10000)
        inside = mask_y_safe & (yy > sy) & (yy < ey)
        sel_y.loc[inside] = lab
    period.loc[mask_y_safe] = sel_y.loc[mask_y_safe]

    df2["活動期"] = period

    start_ymd_range = (start_year_g * 10000 + 101).astype("Int64")
    end_ymd_range = (end_year_g * 10000 + 1231).astype("Int64")

    start_lab = pd.Series(pd.NA, index=df2.index, dtype="object")
    end_lab = pd.Series(pd.NA, index=df2.index, dtype="object")

    conds_start = [start_ymd_range.between(s, e) for _, s, e in periods]
    conds_end = [end_ymd_range.between(s, e) for _, s, e in periods]

    start_lab.loc[is_range] = np.select([c.fillna(False) for c in conds_start], labels, default=pd.NA)[is_range]
    end_lab.loc[is_range] = np.select([c.fillna(False) for c in conds_end], labels, default=pd.NA)[is_range]

    range_contained = is_range & start_lab.notna() & (start_lab == end_lab) & start_lab.isin(labels)
    df2.loc[range_contained, "活動期"] = start_lab.loc[range_contained]

    stats = {
        "n_has_any_year": int(has_any_year.sum()),
        "n_out_of_period": int(df2["活動期"].isna().sum()),
        "n_range_total": int(is_range.sum()),
        "n_range_contained": int(range_contained.sum()),
        "n_range_boundary": int((is_range & (~range_contained)).sum()),
    }
    return df2, stats


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase3-9: 出生地域×活動期の可視化（年代幅対応）")

    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv", help="Input CSV path")
    p.add_argument("--encoding", default="utf-8-sig", help="Input CSV encoding")
    p.add_argument("--outdir", default="outputs/phase3_9", help="Output directory")
    p.add_argument("--timestamp", default=None, help="Timestamp string for filenames (default: now)")

    p.add_argument("--birth_region_col", default="出生地域_主")
    p.add_argument("--birth_place_col", default="出生地_主")
    p.add_argument("--sender_col", default="発信者")

    p.add_argument("--year_col", default="年代_西暦")
    p.add_argument("--month_col", default="月")
    p.add_argument("--day_col", default="日")

    p.add_argument("--year_start_col", default="年代_開始")
    p.add_argument("--year_end_col", default="年代_終了")
    p.add_argument("--era_col", default="年代_時代")
    p.add_argument("--rep_year_candidates", nargs="*", default=["年代_代表", "年代_代表値", "年代_代表値_西暦"])

    p.add_argument("--topk_senders", type=int, default=5, help="TopK senders for concentration")
    p.add_argument("--top_regions", type=int, default=None, help="Keep top N regions; rest -> その他 (default: by min_count)")
    p.add_argument("--min_count_for_keep", type=int, default=10, help="Keep regions with count >= this when --top_regions is None")
    p.add_argument("--include_unknown", action="store_true", help="Include unknown birth region (fillna='不明')")

    p.add_argument("--cmap", default="YlOrRd", help="Colormap for main heatmaps")
    p.add_argument("--dpi", type=int, default=220, help="PNG dpi")
    p.add_argument("--font", default=None, help="Preferred font family (falls back automatically)")
    p.add_argument("--no_plots", action="store_true", help="Skip PNG outputs (CSV/log only)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    chosen_font = configure_matplotlib_font(args.font)

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    log_lines: List[str] = []

    def log(s: str = "") -> None:
        log_lines.append(s)
        print(s)

    df = pd.read_csv(in_path, encoding=args.encoding)
    n_raw = len(df)

    required_cols = [args.birth_region_col, args.sender_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"列が見つからん: {col} / columns={list(df.columns)}")

    has_birth_place_col = args.birth_place_col in df.columns

    df2 = df.copy()
    if args.include_unknown:
        df2[args.birth_region_col] = df2[args.birth_region_col].fillna("不明")
    else:
        df2 = df2[df2[args.birth_region_col].notna()]

    df2, stats = assign_period(
        df2,
        PERIODS,
        year_col=args.year_col,
        month_col=args.month_col if args.month_col in df2.columns else None,
        day_col=args.day_col if args.day_col in df2.columns else None,
        year_start_col=args.year_start_col,
        year_end_col=args.year_end_col,
        era_col=args.era_col,
        rep_year_candidates=args.rep_year_candidates,
    )

    log("📋 有効データのフィルタリング（段階別）:")
    log(f"   入力: {in_path.resolve()}")
    log(f"   出力先: {out_dir.resolve()}")
    log(f"   使用フォント: {chosen_font}")
    log(f"   元データ: {n_raw} 件")
    log(f"   出生地域あり: {int(df[args.birth_region_col].notna().sum())} 件")
    log(f"   年代情報あり（点/幅/代表のいずれか）: {stats['n_has_any_year']} 件")
    log(f"   活動期に入らない（範囲外/欠損/保留）: {stats['n_out_of_period']} 件")
    log(f"   年代幅データ: {stats['n_range_total']} 件（単一期に収まる: {stats['n_range_contained']} / 境界またぎ等: {stats['n_range_boundary']}）")
    log("")

    df2 = df2[df2["活動期"].notna()].copy()
    n_after_period = len(df2)
    log(f"   活動期まで有効: {n_after_period} 件")
    log(f"   データ利用率（元データ比）: {n_after_period / n_raw * 100:.1f} %")
    log("")

    df2[args.sender_col] = df2[args.sender_col].astype(str).str.strip()
    df2.loc[df2[args.sender_col].isin(["", "nan", "None"]), args.sender_col] = np.nan
    n_missing_sender = int(df2[args.sender_col].isna().sum())
    df2 = df2[df2[args.sender_col].notna()].copy()
    log(f"   送信者欠損除外: {n_missing_sender} 件")
    log(f"   最終分析対象: {len(df2)} 件")
    log("")

    vc = df2[args.birth_region_col].value_counts()
    log(f"   出生地域の種類数: {df2[args.birth_region_col].nunique()} 種")
    log(f"   主要な出生地域: {vc.head(5).index.tolist()}")
    log("")

    if args.top_regions is not None:
        keep = vc.head(args.top_regions).index.tolist()
        region_rule = f"上位{args.top_regions}地域を保持、それ以外は『その他』"
        df2["出生地域_分析"] = df2[args.birth_region_col].where(df2[args.birth_region_col].isin(keep), "その他")
    else:
        keep = vc[vc >= args.min_count_for_keep].index.tolist()
        region_rule = f"件数{args.min_count_for_keep}以上の地域を保持、それ以外は『その他』"
        df2["出生地域_分析"] = df2[args.birth_region_col].where(df2[args.birth_region_col].isin(keep), "その他")

    log(f"   地域まとめ規則: {region_rule}")
    if args.include_unknown:
        log("   欠損出生地域は『不明』として保持")
    else:
        log("   欠損出生地域は除外")
    log("")

    ct_raw = pd.crosstab(df2["出生地域_分析"], df2["活動期"]).reindex(columns=PERIOD_ORDER, fill_value=0)

    ct_norm = ct_raw.astype(float).copy()
    for col in ct_norm.columns:
        ct_norm[col] = ct_norm[col] / DURATIONS[col]

    order = ct_raw.sum(axis=1).sort_values(ascending=False).index
    ct_raw = ct_raw.loc[order]
    ct_norm = ct_norm.loc[order]

    col_share = ct_raw.div(ct_raw.sum(axis=0), axis=1) * 100
    col_share = col_share.fillna(0.0)

    obs = ct_raw.values.astype(float)
    total = obs.sum()
    if total == 0:
        std_resid_df = pd.DataFrame(np.nan, index=ct_raw.index, columns=ct_raw.columns)
    else:
        row_sum = obs.sum(axis=1, keepdims=True)
        col_sum = obs.sum(axis=0, keepdims=True)
        exp = (row_sum @ col_sum) / total
        std_resid = (obs - exp) / np.sqrt(exp)
        std_resid_df = pd.DataFrame(std_resid, index=ct_raw.index, columns=ct_raw.columns)

    cell_topk = (
        df2.groupby(["出生地域_分析", "活動期"])[args.sender_col]
        .apply(lambda s: (s.value_counts().head(args.topk_senders).sum() / len(s)) if len(s) else np.nan)
        .unstack()
        .reindex(index=ct_raw.index, columns=ct_raw.columns)
    )
    cell_topk_pct = (cell_topk * 100).fillna(0)

    cell_top1 = (
        df2.groupby(["出生地域_分析", "活動期"])[args.sender_col]
        .apply(lambda s: (s.value_counts().head(1).sum() / len(s)) if len(s) else np.nan)
        .unstack()
        .reindex(index=ct_raw.index, columns=ct_raw.columns)
    )
    cell_top1_pct = (cell_top1 * 100).fillna(0)

    cell_letter_count = (
        df2.groupby(["出生地域_分析", "活動期"]).size().unstack()
        .reindex(index=ct_raw.index, columns=ct_raw.columns).fillna(0).astype(int)
    )
    cell_unique_senders = (
        df2.groupby(["出生地域_分析", "活動期"])[args.sender_col].nunique().unstack()
        .reindex(index=ct_raw.index, columns=ct_raw.columns).fillna(0).astype(int)
    )
    cell_letters_per_sender = cell_letter_count.div(cell_unique_senders).replace([np.inf, -np.inf], np.nan).fillna(0)

    nu_long = (
        pd.concat(
            [
                cell_letter_count.stack(future_stack=True).rename("N_書簡数"),
                cell_unique_senders.stack(future_stack=True).rename("U_発信者数"),
                cell_letters_per_sender.stack(future_stack=True).rename("N_div_U_平均書簡数_per_発信者"),
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"level_0": "出生地域_分析", "level_1": "活動期"})
    )

    if has_birth_place_col:
        sender_period_birth = (
            df2[[args.sender_col, "活動期", args.birth_region_col, args.birth_place_col]]
            .dropna(subset=[args.sender_col, "活動期", args.birth_region_col])
            .copy()
        )
        sender_period_birth[args.sender_col] = sender_period_birth[args.sender_col].astype(str).str.strip()
        sender_period_birth = sender_period_birth.drop_duplicates(subset=[args.sender_col, "活動期"])

        mask_chugoku = sender_period_birth[args.birth_region_col].eq("中国")
        total_chugoku_u = (
            sender_period_birth.loc[mask_chugoku]
            .groupby("活動期")[args.sender_col]
            .nunique()
            .reindex(PERIOD_ORDER)
            .fillna(0)
            .astype(int)
        )
        yamaguchi_in_chugoku_u = (
            sender_period_birth.loc[mask_chugoku & sender_period_birth[args.birth_place_col].eq("山口")]
            .groupby("活動期")[args.sender_col]
            .nunique()
            .reindex(PERIOD_ORDER)
            .fillna(0)
            .astype(int)
        )
        yamaguchi_share_in_chugoku = (
            yamaguchi_in_chugoku_u.div(total_chugoku_u.replace(0, np.nan)) * 100
        ).round(1)
        yamaguchi_share_df = pd.DataFrame({
            "活動期": PERIOD_ORDER,
            "中国出身発信者_U": total_chugoku_u.reindex(PERIOD_ORDER).values,
            "うち山口出身発信者_U": yamaguchi_in_chugoku_u.reindex(PERIOD_ORDER).values,
            "割合_pct": yamaguchi_share_in_chugoku.reindex(PERIOD_ORDER).values,
        })
    else:
        yamaguchi_share_df = pd.DataFrame(columns=["活動期", "中国出身発信者_U", "うち山口出身発信者_U", "割合_pct"])

    n_regions, n_periods = ct_raw.shape
    theoretical = n_regions * n_periods
    nonzero = int((ct_raw.values > 0).sum())
    density = (nonzero / theoretical * 100) if theoretical else 0.0

    log("📈 クロス集計結果:")
    log(f"   対象地域数: {n_regions}")
    log(f"   活動期間数: {n_periods}")
    log(f"   理論的組み合わせ数: {theoretical}")
    log(f"   実際のデータがある組み合わせ数: {nonzero}")
    log(f"   データ密度: {density:.1f} %")
    log("")

    log("📅 品川の活動期間別書簡数:")
    counts_by_period = df2["活動期"].value_counts().reindex(PERIOD_ORDER).fillna(0).astype(int)
    for col in PERIOD_ORDER:
        log(f"   {col}: {counts_by_period[col]} 件")
    log("")

    log("🧮 図5-6 の分子・分母（N/U）:")
    log("   N = セル内書簡数, U = セル内ユニーク発信者数, 図5-6 = N / U")
    log(f"   N（書簡数）CSV: {out_dir / f'phase3_9_birth_region_by_period_letter_count_{ts}.csv'}")
    log(f"   U（発信者数）CSV: {out_dir / f'phase3_9_birth_region_by_period_unique_senders_{ts}.csv'}")
    log(f"   N/U 一覧CSV: {out_dir / f'phase3_9_birth_region_by_period_N_U_summary_{ts}.csv'}")
    log("")

    if has_birth_place_col:
        log("🗺️ 中国出身者のうち山口出身者が占める割合（活動期別・人ベース）:")
        for _, row in yamaguchi_share_df.iterrows():
            pct = row["割合_pct"]
            pct_str = "NA" if pd.isna(pct) else f"{pct:.1f}%"
            log(
                f"   {row['活動期']}: {int(row['うち山口出身発信者_U'])}/{int(row['中国出身発信者_U'])} 人 ({pct_str})"
            )
        log("")
    else:
        log(f"⚠️ 出生地列が見つからないため、山口比率は未計算: {args.birth_place_col}")
        log("")

    raw_csv = out_dir / f"phase3_9_birth_region_by_period_counts_raw_{ts}.csv"
    norm_csv = out_dir / f"phase3_9_birth_region_by_period_counts_per_year_{ts}.csv"
    share_csv = out_dir / f"phase3_9_birth_region_by_period_colshare_percent_{ts}.csv"
    resid_csv = out_dir / f"phase3_9_birth_region_by_period_std_residuals_{ts}.csv"
    topk_csv = out_dir / f"phase3_9_birth_region_by_period_top{args.topk_senders}_sender_share_{ts}.csv"
    top1_csv = out_dir / f"phase3_9_birth_region_by_period_top1_sender_share_{ts}.csv"
    letter_count_csv = out_dir / f"phase3_9_birth_region_by_period_letter_count_{ts}.csv"
    unique_senders_csv = out_dir / f"phase3_9_birth_region_by_period_unique_senders_{ts}.csv"
    letters_per_sender_csv = out_dir / f"phase3_9_birth_region_by_period_letters_per_sender_{ts}.csv"
    nu_summary_csv = out_dir / f"phase3_9_birth_region_by_period_N_U_summary_{ts}.csv"
    yamaguchi_share_csv = out_dir / f"phase3_9_chugoku_including_yamaguchi_share_by_period_{ts}.csv"
    log_txt = out_dir / f"phase3_9_birth_region_by_period_log_{ts}.txt"

    ct_raw.to_csv(raw_csv, encoding="utf-8-sig")
    ct_norm.to_csv(norm_csv, encoding="utf-8-sig")
    col_share.to_csv(share_csv, encoding="utf-8-sig")
    std_resid_df.to_csv(resid_csv, encoding="utf-8-sig")
    cell_topk.to_csv(topk_csv, encoding="utf-8-sig")
    cell_top1.to_csv(top1_csv, encoding="utf-8-sig")
    cell_letter_count.to_csv(letter_count_csv, encoding="utf-8-sig")
    cell_unique_senders.to_csv(unique_senders_csv, encoding="utf-8-sig")
    cell_letters_per_sender.to_csv(letters_per_sender_csv, encoding="utf-8-sig")
    nu_long.to_csv(nu_summary_csv, index=False, encoding="utf-8-sig")
    yamaguchi_share_df.to_csv(yamaguchi_share_csv, index=False, encoding="utf-8-sig")
    log_txt.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    if not args.no_plots:
        norm_png = out_dir / f"phase3_9_birth_region_by_period_heatmap_counts_per_year_{ts}.png"
        share_png = out_dir / f"phase3_9_birth_region_by_period_stackedbar_colshare_percent_{ts}.png"
        resid_png = out_dir / f"phase3_9_birth_region_by_period_heatmap_std_residuals_{ts}.png"
        topk_png = out_dir / f"phase3_9_birth_region_by_period_heatmap_top{args.topk_senders}_sender_share_{ts}.png"
        top1_png = out_dir / f"phase3_9_birth_region_by_period_heatmap_top1_sender_share_{ts}.png"
        letters_per_sender_png = out_dir / f"phase3_9_birth_region_by_period_heatmap_letters_per_sender_{ts}.png"

        plot_heatmap(ct_norm, norm_png, "図5-1 出生地域 × 品川活動期（正規化：件/年）", cmap=args.cmap, fmt="{v:.1f}", dpi=args.dpi)
        plot_stacked_share(col_share, share_png, "図5-2 活動期別の出生地域構成（%）", dpi=args.dpi)
        plot_residual_heatmap(std_resid_df, resid_png, "図5-3 出生地域×活動期（標準化残差：期待値との差）", dpi=args.dpi)
        plot_heatmap(cell_topk_pct, topk_png, f"図5-4 セル内の上位{args.topk_senders}送信者占有率（書簡数ベース, %）", cmap=args.cmap, fmt="{v:.1f}", dpi=args.dpi)
        plot_heatmap(cell_top1_pct, top1_png, "図5-5 セル内の上位1送信者占有率（書簡数ベース, %）", cmap=args.cmap, fmt="{v:.1f}", dpi=args.dpi)
        plot_heatmap(cell_letters_per_sender, letters_per_sender_png, "図5-6 セル内の平均書簡数/送信者", cmap=args.cmap, fmt="{v:.2f}", dpi=args.dpi)

        print("✅ 出力完了（PNGあり）:")
        print(f"   正規化ヒートマップ: {norm_png}")
        print(f"   Top{args.topk_senders}: {topk_png}")
        print(f"   Top1: {top1_png}")
        print(f"   ログ: {log_txt}")
    else:
        print("✅ 出力完了（CSV/LOGのみ）:")
        print(f"   ログ: {log_txt}")


if __name__ == "__main__":
    main()
