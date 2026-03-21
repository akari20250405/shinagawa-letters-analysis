from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import font_manager

MONTH_CANDIDATES = ["月", "月_数値", "month", "月日_月", "月_num"]
DAY_CANDIDATES = ["日", "日_数値", "day", "月日_日", "日_num"]
FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]

# 活動期定義
PERIODS = [
    (1, "維新期", "1868-1870年 (明治元年-3年7月)", 18680101, 18700731),
    (2, "ドイツ滞在期①", "1870-1875年 (明治3年8月-8年10月5日)", 18700801, 18751005),
    (3, "殖産興業官僚期", "1875-1886年 (明治8年10月6日-19年3月)", 18751006, 18860331),
    (4, "ドイツ滞在期②", "1886-1887年 (明治19年4月-20年6月5日)", 18860401, 18870605),
    (5, "帰朝～宮中顧問官期", "1887-1889年 (明治20年6月6日-22年5月12日)", 18870606, 18890512),
    (6, "宮内省御料局長期", "1889-1891年 (明治22年5月13日-24年5月末)", 18890513, 18910531),
    (7, "内務大臣期", "1891-1892年 (明治24年6月-25年3月11日)", 18910601, 18920311),
    (8, "晩年期", "1892-1900年 (明治25年3月12日-33年2月26日)", 18920312, 19000226),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase2-7 補助：活動期の分類可能率チェック（YYYYMMDD対応）")
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    p.add_argument("--outdir", default="outputs/phase2_7_3", help="Output directory")
    p.add_argument("--col_year", default="年代_西暦", help="Gregorian year column")
    return p.parse_args()


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def pick_existing_font(candidates: list[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in candidates:
        if cand in available:
            return cand
    return "sans-serif"


# 境界ルールヘルプ
# 重要: ロジックは PERIODS を唯一の真実源とする

def pid_for_ymd(ymd: int, periods: list[tuple[int, str, str, int, int]]) -> int | None:
    for pid, *_rest, s, e in periods:
        if s <= ymd <= e:
            return pid
    return None


def build_year_rules(periods: list[tuple[int, str, str, int, int]], year_min: int | None = None, year_max: int | None = None):
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


def build_day_sensitive_months(periods: list[tuple[int, str, str, int, int]]) -> set[int]:
    sensitive: set[int] = set()
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
    outdir.mkdir(parents=True, exist_ok=True)

    ts = timestamp()
    out_csv = outdir / f"phase2_7_3_activity_period_classifiability_table_{ts}.csv"
    out_log = outdir / f"phase2_7_3_activity_period_classifiability_log_{ts}.txt"

    font_name = pick_existing_font(FONT_CANDIDATES)

    df = pd.read_csv(in_path, encoding=args.encoding)

    # 1) 年・月・日を安全に作る
    year_raw = pd.to_numeric(df.get(args.col_year, np.nan), errors="coerce")
    df["_year"] = np.floor(year_raw + 0.5).astype("Int64")

    col_month = next((c for c in MONTH_CANDIDATES if c in df.columns), None)
    col_day = next((c for c in DAY_CANDIDATES if c in df.columns), None)

    month_raw = pd.to_numeric(df.get(col_month, np.nan), errors="coerce") if col_month else pd.Series(np.nan, index=df.index)
    day_raw = pd.to_numeric(df.get(col_day, np.nan), errors="coerce") if col_day else pd.Series(np.nan, index=df.index)

    df["_month"] = month_raw.astype("Int64")
    df["_day"] = day_raw.astype("Int64")

    df["_ym"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    df["_ymd"] = pd.Series(pd.NA, index=df.index, dtype="Int64")

    mask_ym = df["_year"].notna() & df["_month"].notna()
    mask_ymd = df["_year"].notna() & df["_month"].notna() & df["_day"].notna()

    df.loc[mask_ym, "_ym"] = (df.loc[mask_ym, "_year"] * 100 + df.loc[mask_ym, "_month"]).astype("Int64")
    df.loc[mask_ymd, "_ymd"] = (
        df.loc[mask_ymd, "_year"] * 10000
        + df.loc[mask_ymd, "_month"] * 100
        + df.loc[mask_ymd, "_day"]
    ).astype("Int64")

    n_total = len(df)
    n_year_valid = int(df["_year"].notna().sum())
    valid_rate = (n_year_valid / n_total * 100) if n_total else 0.0

    # 2) 活動期定義
    periods = PERIODS
    valid_pids = [pid for pid, *_ in periods]
    periods_ym = [(pid, s // 100, e // 100) for pid, _, _, s, e in periods]

    # 3) 期分類（ベクトル）
    pid = pd.Series(pd.NA, index=df.index, dtype="object")
    status = pd.Series("未分類", index=df.index, dtype="string")

    mask_year_missing = df["_year"].isna()
    status.loc[mask_year_missing] = "未分類（年欠損）"

    # 3-1) 年月日あり
    mask_have_ymd = df["_ymd"].notna()
    conds_ymd = [df["_ymd"].between(s, e) for _, _, _, s, e in periods]
    sel_ymd = pd.Series(np.select([c.fillna(False) for c in conds_ymd], valid_pids, default=pd.NA), index=df.index)

    mask_ymd_assigned = mask_have_ymd & sel_ymd.notna()
    pid.loc[mask_ymd_assigned] = sel_ymd.loc[mask_ymd_assigned]
    status.loc[mask_ymd_assigned] = "分類（年月日あり）"

    mask_ymd_present_but_out = mask_have_ymd & sel_ymd.isna() & (~mask_year_missing)
    status.loc[mask_ymd_present_but_out] = "未分類（年月日ありだが範囲外）"

    # 3-2) 年月のみ
    mask_have_ym_only = df["_ym"].notna() & df["_ymd"].isna()

    mask_day_hold = mask_have_ym_only & df["_ym"].isin(list(DAY_SENSITIVE_MONTHS))
    status.loc[mask_day_hold] = "未分類（境界月で日欠損）"

    mask_ym_safe = mask_have_ym_only & (~df["_ym"].isin(list(DAY_SENSITIVE_MONTHS)))
    conds_ym = [df["_ym"].between(s, e) for _, s, e in periods_ym]
    sel_ym = pd.Series(np.select([c.fillna(False) for c in conds_ym], valid_pids, default=pd.NA), index=df.index)

    mask_ym_assigned = mask_ym_safe & sel_ym.notna() & pid.isna()
    pid.loc[mask_ym_assigned] = sel_ym.loc[mask_ym_assigned]
    status.loc[mask_ym_assigned] = "分類（年月あり）"

    mask_ym_present_but_out = mask_ym_safe & sel_ym.isna() & pid.isna() & (~mask_year_missing)
    status.loc[mask_ym_present_but_out] = "未分類（年月ありだが範囲外）"

    # 3-3) 年のみ（保守的）
    mask_year_only = df["_year"].notna() & df["_ym"].isna()
    boundary_years, _safe_ranges = build_year_rules(PERIODS)

    mask_year_hold = mask_year_only & df["_year"].isin(list(boundary_years)) & pid.isna()
    status.loc[mask_year_hold] = "未分類（境界年で月欠損）"

    mask_year_safe = mask_year_only & (~df["_year"].isin(list(boundary_years))) & pid.isna()
    yy = df["_year"]

    for pid_i, _name, _desc, s_ymd, e_ymd in periods:
        s_y = s_ymd // 10000
        s_m = (s_ymd // 100) % 100
        s_d = s_ymd % 100
        e_y = e_ymd // 10000
        e_m = (e_ymd // 100) % 100
        e_d = e_ymd % 100

        safe_start_year = s_m == 1 and s_d == 1
        safe_end_year = e_m == 12 and e_d == 31

        cond_inside = (yy > s_y) & (yy < e_y)
        cond_start = (yy == s_y) & safe_start_year
        cond_end = (yy == e_y) & safe_end_year

        cond = mask_year_safe & (cond_inside | cond_start | cond_end)
        pid.loc[cond] = pid_i
        status.loc[cond] = "分類（年のみ）"

    df["_period_id"] = pid
    df["_period_status"] = status

    # 4) 期別：候補に対する分類可能率
    rows: list[dict[str, object]] = []
    for pid_i, name, desc, s_ymd, e_ymd in periods:
        s_y = s_ymd // 10000
        e_y = e_ymd // 10000

        cand = df[df["_year"].between(s_y, e_y, inclusive="both") & df["_year"].notna()]
        cand_n = int(len(cand))

        assigned = df[df["_period_id"] == pid_i]
        assigned_n = int(len(assigned))

        boundary_year_drop_n = int((cand["_period_status"] == "未分類（境界年で月欠損）").sum())
        boundary_month_day_drop_n = int((cand["_period_status"] == "未分類（境界月で日欠損）").sum())

        rate = (assigned_n / cand_n * 100) if cand_n else 0.0

        rows.append({
            "期": f"{pid_i}.{name}",
            "期間": desc,
            "候補（年レンジ内）": cand_n,
            "分類できた": assigned_n,
            "分類可能率(%)": round(rate, 1),
            "候補内：境界年で月欠損": boundary_year_drop_n,
            "候補内：境界月で日欠損": boundary_month_day_drop_n,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_csv, index=False, encoding="utf-8-sig")

    lines: list[str] = []
    lines.append("Phase2-7 補助：活動期の『分類可能率』チェック（YYYYMMDD対応）")
    lines.append(f"実行時刻: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"入力: {in_path.resolve()}")
    lines.append(f"encoding: {args.encoding}")
    lines.append(f"全体データ: {n_total}件")
    lines.append(f"年代分析用有効データ（年あり）: {n_year_valid}件")
    lines.append(f"有効率: {valid_rate:.1f}%")
    lines.append(f"使用カラム: '{args.col_year}', month='{col_month}', day='{col_day}'")
    lines.append(f"使用フォント: {font_name}")
    lines.append(f"日センシティブ月: {sorted(DAY_SENSITIVE_MONTHS)}")
    lines.append(f"境界年: {sorted(boundary_years)}")
    lines.append("分類可能率(%) = 分類できた件数 / 候補（年レンジ内） × 100")
    lines.append(f"表保存先: {out_csv.resolve()}")
    lines.append(f"ログ保存先: {out_log.resolve()}")
    lines.append("")
    lines.append("[SUMMARY]")
    lines.append(summary.to_string(index=False))

    out_log.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
