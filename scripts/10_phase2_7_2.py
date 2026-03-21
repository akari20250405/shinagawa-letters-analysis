from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------
# Activity periods (closed interval YYYYMMDD)
# -------------------------
PERIODS = [
    (1, "維新期",             "1868-1870年 (明治元年-3年7月)",                  18680101, 18700731),
    (2, "ドイツ滞在期①",       "1870-1875年 (明治3年8月-8年10月5日)",            18700801, 18751005),
    (3, "殖産興業官僚期",       "1875-1886年 (明治8年10月6日-19年3月)",           18751006, 18860331),
    (4, "ドイツ滞在期②",       "1886-1887年 (明治19年4月-20年6月5日)",           18860401, 18870605),
    (5, "帰朝～宮中顧問官期",   "1887-1889年 (明治20年6月6日-22年5月12日)",       18870606, 18890512),
    (6, "宮内省御料局長期",     "1889-1891年 (明治22年5月13日-24年5月末)",        18890513, 18910531),
    (7, "内務大臣期",           "1891-1892年 (明治24年6月-25年3月11日)",          18910601, 18920311),
    (8, "晩年期",               "1892-1900年 (明治25年3月12日-33年2月26日)",      18920312, 19000226),
]

ERA_OFFSET = {
    "M": 1867,  # 明治1=1868
    "K": 1864,  # 慶応1=1865
    "S": 1925,  # 昭和1=1926
}

MONTH_CANDIDATES = ["月", "月_数値", "month", "月日_月", "月_num"]
DAY_CANDIDATES = ["日", "日_数値", "day", "月日_日", "日_num"]
YEAR_REP_CANDIDATES = ["年代_代表", "年代_代表値", "年代_代表値_西暦"]
FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]


def ymd_to_date(x: int) -> date:
    y = x // 10000
    m = (x // 100) % 100
    d = x % 100
    return date(int(y), int(m), int(d))


def period_length_days(start_ymd: int, end_ymd: int) -> int:
    return (ymd_to_date(end_ymd) - ymd_to_date(start_ymd)).days + 1


def round_half_up_to_int(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.floor(x + 0.5).astype("Int64")


def detect_first(columns: pd.Index, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in columns:
            return c
    return None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def setup_japanese_font(preferred: str | None = None) -> str | None:
    available = {f.name for f in fm.fontManager.ttflist}
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend([f for f in FONT_CANDIDATES if f not in candidates])
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase2-7-2: 活動期別（件/年で正規化, 年代幅対応）")
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    p.add_argument("--outdir", default="outputs/phase2_7_2", help="Output directory")
    p.add_argument("--log", default=None, help="Log txt path (default: outdir/phase2_7_2_activity_period_rate_per_year_log_<timestamp>.txt)")
    p.add_argument("--png", default=None, help="Plot png path (default: outdir/phase2_7_2_activity_period_rate_per_year_<timestamp>.png)")
    p.add_argument("--table", default=None, help="Summary table csv path (default: outdir/phase2_7_2_activity_period_rate_per_year_table_<timestamp>.csv)")
    p.add_argument("--font", default=None, help="Preferred Matplotlib font family (optional)")
    p.add_argument("--show", action="store_true", help="Show plot window")
    p.add_argument("--col_year", default="年代_西暦", help="Point-year column (Gregorian year)")
    p.add_argument("--col_year_start", default="年代_開始", help="Range start year (era-year)")
    p.add_argument("--col_year_end", default="年代_終了", help="Range end year (era-year)")
    p.add_argument("--col_era", default="年代_時代", help="Era column (M/K/S)")
    p.add_argument(
        "--rep_is_gregorian",
        action="store_true",
        help="Treat representative year column as Gregorian year (no era offset).",
    )
    return p.parse_args()


def pid_for_ymd(ymd: int, periods) -> int | None:
    for pid, *_rest, s, e in periods:
        if s <= ymd <= e:
            return pid
    return None


def build_year_rules(periods, year_min=None, year_max=None):
    start_year = periods[0][3] // 10000
    end_year = periods[-1][4] // 10000
    if year_min is None:
        year_min = start_year
    if year_max is None:
        year_max = end_year

    safe_years_by_pid: dict[int, list[int]] = {}
    boundary_years: set[int] = set()

    for y in range(year_min, year_max + 1):
        pid_jan1 = pid_for_ymd(y * 10000 + 101, periods)
        pid_dec31 = pid_for_ymd(y * 10000 + 1231, periods)
        if pid_jan1 is not None and pid_jan1 == pid_dec31:
            safe_years_by_pid.setdefault(pid_jan1, []).append(y)
        else:
            boundary_years.add(y)

    ranges: list[tuple[int, int, int]] = []
    for pid, years in sorted(safe_years_by_pid.items()):
        years = sorted(years)
        s = prev = years[0]
        for yy in years[1:]:
            if yy == prev + 1:
                prev = yy
            else:
                ranges.append((pid, s, prev))
                s = prev = yy
        ranges.append((pid, s, prev))
    return boundary_years, ranges


def ymd_to_ym(ymd: int) -> int:
    return ymd // 100


def is_month_boundary_sensitive(ymd: int, is_start: bool) -> bool:
    y = ymd // 10000
    m = (ymd // 100) % 100
    d = ymd % 100
    last = calendar.monthrange(y, m)[1]
    if is_start:
        return d != 1
    return d != last


def build_day_sensitive_months(periods) -> set[int]:
    sensitive = set()
    for _pid, *_rest, s, e in periods:
        if is_month_boundary_sensitive(s, is_start=True):
            sensitive.add(ymd_to_ym(s))
        if is_month_boundary_sensitive(e, is_start=False):
            sensitive.add(ymd_to_ym(e))
    return sensitive


DAY_SENSITIVE_MONTHS = build_day_sensitive_months(PERIODS)


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log else (outdir / f"phase2_7_2_activity_period_rate_per_year_log_{ts}.txt")
    png_path = Path(args.png) if args.png else (outdir / f"phase2_7_2_activity_period_rate_per_year_{ts}.png")
    table_path = Path(args.table) if args.table else (outdir / f"phase2_7_2_activity_period_rate_per_year_table_{ts}.csv")

    font_used = setup_japanese_font(args.font)

    cols = pd.read_csv(in_path, nrows=0, encoding=args.encoding).columns
    col_month = detect_first(cols, MONTH_CANDIDATES)
    col_day = detect_first(cols, DAY_CANDIDATES)
    col_rep = detect_first(cols, YEAR_REP_CANDIDATES)

    usecols = [
        c
        for c in [
            args.col_year,
            args.col_year_start,
            args.col_year_end,
            args.col_era,
            col_rep,
            col_month,
            col_day,
        ]
        if c and c in cols
    ]

    df = pd.read_csv(in_path, usecols=usecols, encoding=args.encoding)
    n_total = len(df)

    if args.col_era in df.columns:
        offset = df[args.col_era].map(ERA_OFFSET)
    else:
        offset = pd.Series(pd.NA, index=df.index, dtype="Float64")

    year_point_raw = pd.to_numeric(df.get(args.col_year), errors="coerce")
    rep_raw = pd.to_numeric(df.get(col_rep), errors="coerce") if col_rep else pd.Series(np.nan, index=df.index)

    if args.rep_is_gregorian or (col_rep == "年代_代表値_西暦"):
        rep_year_greg = rep_raw
        rep_mode = "rep_as_gregorian"
    else:
        rep_year_greg = rep_raw + offset
        rep_mode = "rep_plus_era_offset"

    year_for_point_raw = year_point_raw.where(year_point_raw.notna(), rep_year_greg)
    year_point = round_half_up_to_int(year_for_point_raw)

    month = round_half_up_to_int(df.get(col_month)) if col_month else pd.Series(pd.NA, index=df.index, dtype="Int64")
    day = round_half_up_to_int(df.get(col_day)) if col_day else pd.Series(pd.NA, index=df.index, dtype="Int64")

    has_y = year_point.notna()
    has_m = month.notna()
    has_d = day.notna()

    ym = (year_point * 100 + month).astype("Int64")
    ymd = (year_point * 10000 + month * 100 + day).astype("Int64")

    period_id = pd.Series(pd.NA, index=df.index, dtype="object")
    choices = [pid for pid, *_ in PERIODS]
    valid_pids = choices

    mask_ymd = has_y & has_m & has_d
    conds_ymd = [ymd.between(s, e) for _, _, _, s, e in PERIODS]
    sel_ymd = pd.Series(np.select([c.fillna(False) for c in conds_ymd], choices, default=pd.NA), index=df.index)
    period_id.loc[mask_ymd] = sel_ymd.loc[mask_ymd]

    mask_ym_only = has_y & has_m & (~has_d)
    is_day_hold = mask_ym_only & ym.isin(list(DAY_SENSITIVE_MONTHS))
    period_id.loc[is_day_hold] = "境界月（日欠損）"

    mask_ym_safe = mask_ym_only & (~ym.isin(list(DAY_SENSITIVE_MONTHS)))
    periods_ym = [(pid, s // 100, e // 100) for pid, _, _, s, e in PERIODS]
    conds_ym = [ym.between(s, e) for _, s, e in periods_ym]
    sel_ym = pd.Series(np.select([c.fillna(False) for c in conds_ym], choices, default=pd.NA), index=df.index)
    period_id.loc[mask_ym_safe] = sel_ym.loc[mask_ym_safe]

    mask_y_only = has_y & (~has_m)
    boundary_years, safe_ranges = build_year_rules(PERIODS)

    is_year_hold = mask_y_only & year_point.isin(boundary_years)
    period_id.loc[is_year_hold] = "境界年（月欠損）"

    mask_y_safe = mask_y_only & (~year_point.isin(boundary_years))
    yy = year_point
    conds_y = [yy.between(s, e) for (_pid, s, e) in safe_ranges]
    choices_y = [pid for (pid, _s, _e) in safe_ranges]
    sel_y = pd.Series(np.select([c.fillna(False) for c in conds_y], choices_y, default=pd.NA), index=df.index)
    period_id.loc[mask_y_safe] = sel_y.loc[mask_y_safe]

    start_raw = (
        pd.to_numeric(df.get(args.col_year_start), errors="coerce")
        if args.col_year_start in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    end_raw = (
        pd.to_numeric(df.get(args.col_year_end), errors="coerce")
        if args.col_year_end in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    is_range = start_raw.notna() & end_raw.notna()

    start_year_g = round_half_up_to_int(start_raw + offset)
    end_year_g = round_half_up_to_int(end_raw + offset)

    start_ymd_range = (start_year_g * 10000 + 101).astype("Int64")
    end_ymd_range = (end_year_g * 10000 + 1231).astype("Int64")

    start_pid = pd.Series(pd.NA, index=df.index, dtype="object")
    end_pid = pd.Series(pd.NA, index=df.index, dtype="object")
    conds_start = [start_ymd_range.between(s, e) for _, _, _, s, e in PERIODS]
    conds_end = [end_ymd_range.between(s, e) for _, _, _, s, e in PERIODS]

    start_pid.loc[is_range] = np.select([c.fillna(False) for c in conds_start], choices, default=pd.NA)[is_range]
    end_pid.loc[is_range] = np.select([c.fillna(False) for c in conds_end], choices, default=pd.NA)[is_range]

    range_contained = is_range & start_pid.notna() & (start_pid == end_pid) & start_pid.isin(valid_pids)
    period_id.loc[range_contained] = start_pid.loc[range_contained]

    period_df = pd.DataFrame(PERIODS, columns=["pid", "pname", "prange", "start_ymd", "end_ymd"])
    period_df["days"] = period_df.apply(lambda r: period_length_days(r["start_ymd"], r["end_ymd"]), axis=1)
    period_df["years"] = period_df["days"] / 365.2425
    period_df["xlabel"] = period_df["pid"].astype(str) + ". " + period_df["pname"]

    classified_mask = period_id.isin(valid_pids)
    counts = period_id.loc[classified_mask].astype(int).value_counts().sort_index()
    period_df["count"] = period_df["pid"].map(counts).fillna(0).astype(int)
    period_df["rate_per_year"] = period_df["count"] / period_df["years"]

    n_year_valid_point = int(year_point.notna().sum())
    n_range_total = int(is_range.sum())
    n_range_contained = int(range_contained.sum())
    n_range_boundary = int((is_range & (~range_contained)).sum())

    n_classified = int(classified_mask.sum())
    n_unclassified = n_total - n_classified
    unclassified_pct = (n_unclassified / n_total * 100) if n_total else 0.0

    has_any_year = year_point.notna() | is_range | rep_year_greg.notna()
    n_any_year = int(has_any_year.sum())
    any_year_pct = (n_any_year / n_total * 100) if n_total else 0.0

    n_year_missing = int((~year_point.notna() & ~is_range & rep_year_greg.isna()).sum())
    n_day_hold = int((period_id == "境界月（日欠損）").sum())
    n_year_hold = int((period_id == "境界年（月欠損）").sum())

    lines: list[str] = []
    lines.append("品川弥二郎書簡データ分析 Phase2-7-2（活動期別・正規化） 年代幅対応版")
    lines.append(f"分析開始時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("============================================================")
    lines.append(f"入力: {in_path.resolve()}")
    lines.append(f"出力ディレクトリ: {outdir.resolve()}")
    lines.append(f"ログ保存先: {log_path.resolve()}")
    lines.append(f"図保存先: {png_path.resolve()}")
    lines.append(f"表保存先: {table_path.resolve()}")
    lines.append(f"encoding: {args.encoding}")
    lines.append(f"show: {args.show}")
    lines.append(f"使用フォント: {font_used if font_used else 'default'}")
    lines.append(f"全体データ: {n_total}件")
    lines.append(f"年代分析用有効データ（点年 or 年代幅 or 代表年）: {n_any_year}件 ({any_year_pct:.1f}%)")
    lines.append(f"点年（年代_西暦 or 代表年で補完）あり: {n_year_valid_point}件")
    lines.append(f"年代幅データ: {n_range_total}件（単一期に収まる: {n_range_contained} / 境界またぎ等: {n_range_boundary}）")
    lines.append(f"使用カラム: year='{args.col_year}', month='{col_month}', day='{col_day}'")
    lines.append(f"年代幅: start='{args.col_year_start}', end='{args.col_year_end}', era='{args.col_era}', rep='{col_rep}'")
    lines.append(f"代表年の扱い: {rep_mode} (override: --rep_is_gregorian)")
    lines.append("補足: 代表年列は補助的に用い、年代_西暦を優先して点年判定を行う。")
    lines.append("")
    lines.append("=== 活動期分類状況 ===")
    lines.append(f"活動期間分類可能データ: {n_classified}件")
    lines.append(f"活動期間分類不可データ: {n_unclassified}件 ({unclassified_pct:.1f}%)")
    lines.append(f"  - 年代情報なし: {n_year_missing}件")
    if n_year_hold:
        lines.append(f"  - 境界年（月欠損）で保留: {n_year_hold}件")
    if n_day_hold:
        lines.append(f"  - 境界月（日欠損）で保留: {n_day_hold}件")

    lines.append("")
    lines.append("=== 品川活動期間別受信書簡数（件/年で正規化） ===")
    for _, r in period_df.iterrows():
        lines.append(
            f"{int(r['pid'])}. {r['pname']}: {int(r['count'])}件 / {r['years']:.2f}年 = {r['rate_per_year']:.1f}件/年  - {r['prange']}"
        )

    log_text = "\n".join(lines)
    print(log_text)

    log_path.write_text(log_text, encoding="utf-8")
    period_df.to_csv(table_path, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(period_df))
    y = period_df["rate_per_year"].values

    y_max = float(np.nanmax(y)) if len(y) else 0.0
    pad = y_max * 0.12
    ax.set_ylim(0, y_max + pad)

    ax.plot(x, y, marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(period_df["xlabel"], rotation=25, ha="right")
    ax.set_ylabel("受信書簡数（件/年）")
    ax.set_title("図3-2 品川の活動期間別 受信書簡数（期の長さで正規化：件/年, 年代幅対応）")

    for i, r in period_df.iterrows():
        ax.text(i, r["rate_per_year"], f"{r['rate_per_year']:.1f}\n({int(r['count'])}件)", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    print(f"\n[SAVED] {png_path.resolve()}")
    print(f"[SAVED] {log_path.resolve()}")
    print(f"[SAVED] {table_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
