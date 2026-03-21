from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# =========================
# 活動期定義
# =========================
PERIODS = [
    ("1.維新期", "1868-01-01", "1870-07-31"),
    ("2.ドイツ滞在期①", "1870-08-01", "1875-10-05"),
    ("3.殖産興業官僚期", "1875-10-06", "1886-03-31"),
    ("4.ドイツ滞在期②", "1886-04-01", "1887-06-05"),
    ("5.帰朝～宮中顧問官期", "1887-06-06", "1889-05-12"),
    ("6.宮内省御料局長期", "1889-05-13", "1891-05-31"),
    ("7.内務大臣期", "1891-06-01", "1892-03-11"),
    ("8.晩年期", "1892-03-12", "1900-02-26"),
]

PERIOD_LABELS = [p[0] for p in PERIODS]
PERIOD_STARTS = {p[0]: pd.Timestamp(p[1]) for p in PERIODS}
PERIOD_ENDS = {p[0]: pd.Timestamp(p[2]) for p in PERIODS}
DAY_SENSITIVE_MONTHS = {(ts.month, ts.year) for _, s, e in PERIODS for ts in [pd.Timestamp(s), pd.Timestamp(e)]}

ERA_OFFSET = {"M": 1867, "K": 1864, "S": 1925}

YEAR_CANDIDATES = ["年代_西暦"]
REP_CANDIDATES = ["年代_代表", "年代_代表値", "年代_代表値_西暦"]
ERA_CANDIDATES = ["年代_時代", "年代_元号"]
YEAR_START_CANDIDATES = ["年代_開始", "年代_開始値"]
YEAR_END_CANDIDATES = ["年代_終了", "年代_終了値"]
MONTH_CANDIDATES = ["月", "月_数値", "年代_月", "月日_月"]
DAY_CANDIDATES = ["日", "日_数値", "年代_日", "月日_日"]
# 旧図の趣旨に合わせて「居住地」ベースを優先
RESIDENCE_COL_CANDIDATES = ["居住地", "居住地_主", "居住地_元"]

FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]
TOPN_DEFAULT = 3
KL_EPS_DEFAULT = 1e-12


# =========================
# ユーティリティ
# =========================
def find_first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_year_rules(periods: list[tuple[str, str, str]]) -> dict[int, tuple[str | None, str | None, bool]]:
    rules: dict[int, tuple[str | None, str | None, bool]] = {}
    for i, (_, s, e) in enumerate(periods):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        for year in range(s_ts.year, e_ts.year + 1):
            if year not in rules:
                rules[year] = (None, None, False)
            prev_start, prev_end, _ = rules[year]
            start_label = PERIOD_LABELS[i] if year == s_ts.year else prev_start
            end_label = PERIOD_LABELS[i] if year == e_ts.year else prev_end
            full = (year > s_ts.year) and (year < e_ts.year)
            rules[year] = (start_label, end_label, full)
    return rules


YEAR_RULES = build_year_rules(PERIODS)


def round_half_up_to_int(x: pd.Series) -> pd.Series:
    x_num = pd.to_numeric(x, errors="coerce")
    out = np.where(x_num.notna(), np.floor(x_num + 0.5), np.nan)
    return pd.Series(out, index=x.index).astype("Int64")


def to_gregorian_from_era(rep: pd.Series, era: pd.Series) -> pd.Series:
    rep_num = pd.to_numeric(rep, errors="coerce")
    era_str = era.astype("string")
    out = pd.Series(np.nan, index=rep.index, dtype="float64")
    for k, off in ERA_OFFSET.items():
        m = era_str.eq(k) & rep_num.notna()
        out.loc[m] = off + rep_num.loc[m]
    return out


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


def pid_for_ymd(y: int, m: int, d: int) -> str | None:
    dt = pd.Timestamp(year=int(y), month=int(m), day=int(d))
    for label, s, e in PERIODS:
        if pd.Timestamp(s) <= dt <= pd.Timestamp(e):
            return label
    return None


def assign_activity_period(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    year_col = find_first_existing_col(out, YEAR_CANDIDATES)
    rep_col = find_first_existing_col(out, REP_CANDIDATES)
    era_col = find_first_existing_col(out, ERA_CANDIDATES)
    ys_col = find_first_existing_col(out, YEAR_START_CANDIDATES)
    ye_col = find_first_existing_col(out, YEAR_END_CANDIDATES)
    month_col = find_first_existing_col(out, MONTH_CANDIDATES)
    day_col = find_first_existing_col(out, DAY_CANDIDATES)

    year_direct = pd.to_numeric(out[year_col], errors="coerce") if year_col else pd.Series(np.nan, index=out.index)
    rep_raw = pd.to_numeric(out[rep_col], errors="coerce") if rep_col else pd.Series(np.nan, index=out.index)
    era_raw = out[era_col] if era_col else pd.Series(pd.NA, index=out.index, dtype="string")
    rep_greg = to_gregorian_from_era(rep_raw, era_raw) if rep_col and era_col else pd.Series(np.nan, index=out.index)
    point_year_source = year_direct.where(year_direct.notna(), rep_greg)
    y_ser = round_half_up_to_int(point_year_source)

    month_ser = (
        pd.to_numeric(out[month_col], errors="coerce").astype("Int64")
        if month_col else pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    )
    day_ser = (
        pd.to_numeric(out[day_col], errors="coerce").astype("Int64")
        if day_col else pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    )

    ys = (
        pd.to_numeric(out[ys_col], errors="coerce").astype("Int64")
        if ys_col else pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    )
    ye = (
        pd.to_numeric(out[ye_col], errors="coerce").astype("Int64")
        if ye_col else pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"), index=out.index)
    )

    out["活動期"] = pd.NA
    out["活動期_判定種別"] = pd.NA

    mask_ymd = y_ser.notna() & month_ser.notna() & day_ser.notna()
    mask_ymd_valid = mask_ymd & pd.Series(
        [is_valid_month_day(m, d, y) for y, m, d in zip(y_ser, month_ser, day_ser)],
        index=out.index,
    )
    if mask_ymd_valid.any():
        out.loc[mask_ymd_valid, "活動期"] = [
            pid_for_ymd(int(y), int(m), int(d))
            for y, m, d in zip(y_ser[mask_ymd_valid], month_ser[mask_ymd_valid], day_ser[mask_ymd_valid])
        ]
        out.loc[mask_ymd_valid, "活動期_判定種別"] = "ymd"

    mask_ym = y_ser.notna() & month_ser.notna() & day_ser.isna() & out["活動期"].isna()
    mask_ym = mask_ym & month_ser.apply(is_valid_month_day)
    for idx in out.index[mask_ym]:
        y, m = int(y_ser.loc[idx]), int(month_ser.loc[idx])
        if (m, y) in DAY_SENSITIVE_MONTHS:
            out.at[idx, "活動期"] = "境界月（日欠損）"
            out.at[idx, "活動期_判定種別"] = "ym_boundary"
        else:
            try:
                out.at[idx, "活動期"] = pid_for_ymd(y, m, 15)
                out.at[idx, "活動期_判定種別"] = "ym_midmonth"
            except Exception:
                pass

    mask_y = y_ser.notna() & month_ser.isna() & out["活動期"].isna()
    for idx in out.index[mask_y]:
        y = int(y_ser.loc[idx])
        if y not in YEAR_RULES:
            continue
        start_label, end_label, full = YEAR_RULES[y]
        if full:
            labels = [p for p in PERIOD_LABELS if PERIOD_STARTS[p].year < y < PERIOD_ENDS[p].year]
            if labels:
                out.at[idx, "活動期"] = labels[0]
                out.at[idx, "活動期_判定種別"] = "y_full"
        elif start_label == end_label and start_label is not None:
            out.at[idx, "活動期"] = start_label
            out.at[idx, "活動期_判定種別"] = "y_single_year"
        else:
            out.at[idx, "活動期"] = "境界年（月欠損）"
            out.at[idx, "活動期_判定種別"] = "y_boundary"

    mask_range = ys.notna() & ye.notna()
    for idx in out.index[mask_range]:
        start_y, end_y = int(ys.loc[idx]), int(ye.loc[idx])
        containing = [
            p for p in PERIOD_LABELS
            if PERIOD_STARTS[p].year <= start_y <= end_y <= PERIOD_ENDS[p].year
        ]
        if len(containing) == 1:
            out.at[idx, "活動期"] = containing[0]
            out.at[idx, "活動期_判定種別"] = "range_single_period"
        elif len(containing) == 0 and pd.isna(out.at[idx, "活動期"]):
            out.at[idx, "活動期"] = "複数期またぎ/要確認"
            out.at[idx, "活動期_判定種別"] = "range_cross"

    return out


def shannon_entropy(counts: pd.Series) -> float:
    counts = counts[counts > 0]
    if counts.empty:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def normalized_entropy(counts: pd.Series) -> float:
    k = int((counts > 0).sum())
    if k <= 1:
        return 0.0 if k == 1 else np.nan
    h = shannon_entropy(counts)
    return float(h / np.log2(k)) if pd.notna(h) else np.nan


def effective_number(counts: pd.Series) -> float:
    h = shannon_entropy(counts)
    return float(2 ** h) if pd.notna(h) else np.nan


def kl_divergence_prev_to_now(prev_counts: pd.Series, now_counts: pd.Series, epsilon: float = KL_EPS_DEFAULT) -> float:
    cats = sorted(set(prev_counts.index) | set(now_counts.index))
    prev_vec = prev_counts.reindex(cats, fill_value=0).astype(float) + epsilon
    now_vec = now_counts.reindex(cats, fill_value=0).astype(float) + epsilon
    prev_p = prev_vec / prev_vec.sum()
    now_p = now_vec / now_vec.sum()
    return float((now_p * np.log2(now_p / prev_p)).sum())


def choose_font(preferred: str | None = None) -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([f for f in FONT_CANDIDATES if f not in candidates])
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return None


def annotate_bar_values(ax, bars, fmt: str = "{:.1f}%") -> None:
    for b in bars:
        h = b.get_height()
        if pd.notna(h):
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom", fontsize=9)


def annotate_line_values(ax, xs, ys, fmt: str = "{:.2f}", yoffset: float = 0.0) -> None:
    for x, y in zip(xs, ys):
        if pd.notna(y):
            ax.text(x, y + yoffset, fmt.format(y), ha="center", va="bottom", fontsize=9)


# =========================
# 引数
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="活動期別の居住地分布集中度を集計・可視化する。")
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv", help="Input CSV path")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    p.add_argument("--outdir", default="outputs/phase3_10", help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    p.add_argument("--font", default=None, help="Preferred Japanese font name")
    p.add_argument("--tokyo_regex", default=r"東京", help="Regex for Tokyo concentration")
    p.add_argument("--kl_epsilon", type=float, default=KL_EPS_DEFAULT, help="Additive smoothing for KL(prev→now)")
    p.add_argument("--no_plots", action="store_true", help="Skip plotting PNG files")
    p.add_argument("--topn", type=int, default=TOPN_DEFAULT, help="Number of top residence places to save")
    return p.parse_args()


# =========================
# メイン
# =========================
def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    used_font = choose_font(args.font)

    out_top_long = outdir / f"phase3_10_top{args.topn}_residence_long_{ts}.csv"
    out_top_wide = outdir / f"phase3_10_top{args.topn}_residence_wide_{ts}.csv"
    out_summary = outdir / f"phase3_10_residence_entropy_summary_{ts}.csv"
    out_log = outdir / f"phase3_10_residence_entropy_log_{ts}.txt"
    out_png_tokyo = outdir / f"図6-1_phase3_10_tokyo_share_{ts}.png"
    out_png_entropy = outdir / f"図6-2_phase3_10_entropy_lines_{ts}.png"
    out_png_effn = outdir / f"図6-3_phase3_10_effective_number_{ts}.png"
    out_png_kl = outdir / f"図6-4_phase3_10_kl_prev_{ts}.png"

    df = pd.read_csv(args.input, encoding=args.encoding)
    n_raw = len(df)
    df = assign_activity_period(df)

    res_col = find_first_existing_col(df, RESIDENCE_COL_CANDIDATES)
    if res_col is None:
        raise KeyError(f"Residence column not found. candidates={RESIDENCE_COL_CANDIDATES}")

    year_col = find_first_existing_col(df, YEAR_CANDIDATES)
    rep_col = find_first_existing_col(df, REP_CANDIDATES)
    era_col = find_first_existing_col(df, ERA_CANDIDATES)
    ys_col = find_first_existing_col(df, YEAR_START_CANDIDATES)
    ye_col = find_first_existing_col(df, YEAR_END_CANDIDATES)
    month_col = find_first_existing_col(df, MONTH_CANDIDATES)
    day_col = find_first_existing_col(df, DAY_CANDIDATES)

    df[res_col] = df[res_col].astype("string").str.strip()
    valid = df[df[res_col].notna() & (df[res_col] != "")].copy()
    valid = valid[valid["活動期"].isin(PERIOD_LABELS)].copy()

    summary_rows: list[dict[str, object]] = []
    top_long_rows: list[dict[str, object]] = []
    period_counts: dict[str, pd.Series] = {}

    for p in PERIOD_LABELS:
        sub = valid[valid["活動期"] == p]
        n = len(sub)
        vc = sub[res_col].value_counts(dropna=True)
        period_counts[p] = vc

        H = shannon_entropy(vc)
        Hn = normalized_entropy(vc)
        N_eff = effective_number(vc)
        tokyo_share = float(sub[res_col].astype(str).str.contains(args.tokyo_regex, regex=True, na=False).mean() * 100) if n > 0 else np.nan

        summary_rows.append({
            "活動期": p,
            "n_letters": n,
            "n_unique_residence": int(vc.size),
            "entropy_H": H,
            "normalized_entropy_Hnorm": Hn,
            "effective_number": N_eff,
            "tokyo_share_pct": tokyo_share,
        })

        for rank, (name, cnt) in enumerate(vc.head(args.topn).items(), start=1):
            top_long_rows.append({
                "活動期": p,
                "rank": rank,
                "居住地": name,
                "count": int(cnt),
                "share_pct": float(cnt / n * 100) if n > 0 else np.nan,
            })

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["kl_prev_to_now"] = np.nan
        for i in range(1, len(summary)):
            prev_label = summary.at[i - 1, "活動期"]
            now_label = summary.at[i, "活動期"]
            summary.at[i, "kl_prev_to_now"] = kl_divergence_prev_to_now(
                period_counts[prev_label],
                period_counts[now_label],
                epsilon=args.kl_epsilon,
            )

    top_long = pd.DataFrame(top_long_rows)
    if not top_long.empty:
        top_wide = top_long.pivot(index="活動期", columns="rank", values=["居住地", "count", "share_pct"])
        top_wide.columns = [f"{a}{b}" for a, b in top_wide.columns]
        top_wide = top_wide.reset_index()
    else:
        top_wide = pd.DataFrame(columns=["活動期"])

    summary.to_csv(out_summary, index=False, encoding="utf-8-sig")
    top_long.to_csv(out_top_long, index=False, encoding="utf-8-sig")
    top_wide.to_csv(out_top_wide, index=False, encoding="utf-8-sig")

    log_lines: list[str] = []
    log_lines.append("# Phase3_10 Residence Entropy Summary")
    log_lines.append("")
    log_lines.append(f"- input: {args.input}")
    log_lines.append(f"- encoding: {args.encoding}")
    log_lines.append(f"- outdir: {outdir}")
    log_lines.append(f"- font_preferred: {args.font}")
    log_lines.append(f"- font_used: {used_font}")
    log_lines.append(f"- dpi: {args.dpi}")
    log_lines.append(f"- no_plots: {args.no_plots}")
    log_lines.append(f"- tokyo_regex: {args.tokyo_regex}")
    log_lines.append(f"- kl_epsilon: {args.kl_epsilon}")
    log_lines.append(f"- topn: {args.topn}")
    log_lines.append("")
    log_lines.append("## 入力と判定")
    log_lines.append(f"- 元データ件数: {n_raw}")
    log_lines.append(f"- 居住地列: {res_col}")
    log_lines.append(f"- 年列: {year_col}")
    log_lines.append(f"- 代表年列: {rep_col}")
    log_lines.append(f"- 時代列: {era_col}")
    log_lines.append(f"- 年代幅開始列: {ys_col}")
    log_lines.append(f"- 年代幅終了列: {ye_col}")
    log_lines.append(f"- 月列: {month_col}")
    log_lines.append(f"- 日列: {day_col}")
    log_lines.append("- 代表年列は補助的に使用し、年代_西暦を優先して活動期判定を行う")
    log_lines.append("")
    log_lines.append("## 件数")
    log_lines.append(f"- 居住地あり件数: {int((df[res_col].astype('string').str.strip().fillna('') != '').sum())}")
    log_lines.append(f"- 活動期未分類件数: {int(df['活動期'].isna().sum())}")
    log_lines.append(f"- 集計有効件数: {len(valid)}")
    log_lines.append("")
    log_lines.append("## 期間別サマリ")
    if summary.empty:
        log_lines.append("- 該当データなし")
    else:
        for _, r in summary.iterrows():
            kl_txt = "NA" if pd.isna(r["kl_prev_to_now"]) else f"{r['kl_prev_to_now']:.3f}"
            log_lines.append(
                f"- {r['活動期']}: n={int(r['n_letters'])}, unique={int(r['n_unique_residence'])}, "
                f"H={r['entropy_H']:.3f}, Hnorm={r['normalized_entropy_Hnorm']:.3f}, "
                f"N_eff={r['effective_number']:.3f}, KL(prev→now)={kl_txt}, "
                f"Tokyo={r['tokyo_share_pct']:.1f}%"
            )
    log_lines.append("")
    log_lines.append("## 保存先")
    log_lines.append(f"- summary_csv: {out_summary}")
    log_lines.append(f"- top_long_csv: {out_top_long}")
    log_lines.append(f"- top_wide_csv: {out_top_wide}")
    if not args.no_plots:
        log_lines.append(f"- tokyo_share_png: {out_png_tokyo}")
        log_lines.append(f"- entropy_lines_png: {out_png_entropy}")
        log_lines.append(f"- effective_number_png: {out_png_effn}")
        log_lines.append(f"- kl_prev_png: {out_png_kl}")
    log_lines.append(f"- log_txt: {out_log}")

    if not args.no_plots and not summary.empty:
        x = np.arange(len(summary))
        labels = summary["活動期"].tolist()

        # 図6-1 東京集中度（棒）
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x, summary["tokyo_share_pct"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("東京居住割合（%）")
        ax.set_title("図6-1　居住地ベース：東京集中度（%）")
        annotate_bar_values(ax, bars, fmt="{:.1f}%")
        fig.tight_layout()
        fig.savefig(out_png_tokyo, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # 図6-2 エントロピー + 正規化エントロピー（折れ線）
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ent = pd.to_numeric(summary["entropy_H"], errors="coerce")
        hnorm = pd.to_numeric(summary["normalized_entropy_Hnorm"], errors="coerce")

        ax1.plot(x, ent, marker="o", label="Entropy(H, log2)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")
        ax1.set_ylabel("Entropy(H, log2)")
        ax1.set_title("図6-2　居住地分布の分散度：シャノンエントロピーと正規化エントロピー")

        # 左軸の上限を少し広げる
        if ent.notna().any():
            ent_min = float(ent.min())
            ent_max = float(ent.max())
            ent_span = max(ent_max - ent_min, 0.2)
            ax1.set_ylim(ent_min - ent_span * 0.08, ent_max + ent_span * 0.22)

        # 左軸の数値ラベル
        for i, (xi, yi) in enumerate(zip(x, ent)):
            if pd.notna(yi):
                # 端の上側ラベルだけ少し下へ逃がす
                if i == len(ent) - 1:
                    offset = (0, -10)
                else:
                    offset = (0, 6)
                ax1.annotate(f"{yi:.2f}", (xi, yi), xytext=offset,
                            textcoords="offset points", ha="center")

        # legendは枠内の少し内側
        ax1.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), borderaxespad=0.0)

        ax2 = ax1.twinx()
        ax2.plot(x, hnorm, marker="^", label="NormEntropy(H/log2K)")
        ax2.set_ylabel("NormEntropy (0-1)")

        # 右軸の上限を少し広げる
        if hnorm.notna().any():
            hnorm_min = float(hnorm.min())
            hnorm_max = float(hnorm.max())
            hnorm_span = max(hnorm_max - hnorm_min, 0.05)
            ax2.set_ylim(
                max(0.0, hnorm_min - hnorm_span * 0.10),
                hnorm_max + hnorm_span * 0.18
            )

        # 右軸の数値ラベル
        for i, (xi, yi) in enumerate(zip(x, hnorm)):
            if pd.notna(yi):
                # 左上の 1.00 を legend から少し下へ逃がす
                if i == 0:
                    offset = (0, 6)
                else:
                    offset = (0, 6)
                ax2.annotate(f"{yi:.2f}", (xi, yi), xytext=offset,
                            textcoords="offset points", ha="center")

        # legendは枠内の少し内側
        ax2.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.0)

        # プロット領域は上まで使う
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_png_entropy, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # 図6-3 実効数（折れ線）
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x, summary["effective_number"], marker="o", label="Effective(2^H)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("2^H")
        ax.set_title("図6-3　居住地の“実効数”（Effective number = 2^H）")
        annotate_line_values(ax, x, summary["effective_number"], fmt="{:.2f}", yoffset=0.02)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_png_effn, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # 図6-4 KL(prev→now)（折れ線）
        kl_df = summary.loc[summary["kl_prev_to_now"].notna(), ["活動期", "kl_prev_to_now"]].reset_index(drop=True)
        if not kl_df.empty:
            x_kl = np.arange(len(kl_df))
            labels_kl = kl_df["活動期"].tolist()
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(x_kl, kl_df["kl_prev_to_now"], marker="o", label="KL(prev→now, log2)")
            ax.set_xticks(x_kl)
            ax.set_xticklabels(labels_kl, rotation=45, ha="right")
            ax.set_ylabel("D_KL (log2)")
            ax.set_title("図6-4　居住地分布の変化量：KLダイバージェンス（前期→今期）")
            annotate_line_values(ax, x_kl, kl_df["kl_prev_to_now"], fmt="{:.2f}", yoffset=0.02)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(out_png_kl, dpi=args.dpi, bbox_inches="tight")
            plt.close(fig)

    out_log.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"[SAVED] {out_summary}")
    print(f"[SAVED] {out_top_long}")
    print(f"[SAVED] {out_top_wide}")
    if not args.no_plots and not summary.empty:
        print(f"[SAVED] {out_png_tokyo}")
        print(f"[SAVED] {out_png_entropy}")
        print(f"[SAVED] {out_png_effn}")
        print(f"[SAVED] {out_png_kl}")
    print(f"[SAVED] {out_log}")


if __name__ == "__main__":
    main()
