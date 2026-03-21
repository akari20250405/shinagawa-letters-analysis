from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# 活動期（ymd: int）
# =========================
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
VALID_PIDS = [pid for pid, *_ in PERIODS]
CHOICES = VALID_PIDS

FONT_CANDIDATES = [
    "MS Gothic",
    "Yu Gothic",
    "Meiryo",
    "IPAexGothic",
    "IPAGothic",
    "Noto Sans CJK JP",
    "TakaoGothic",
]


def set_japanese_font() -> str:
    """利用可能な日本語フォントを探索して設定する。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in FONT_CANDIDATES:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    plt.rcParams["axes.unicode_minus"] = False
    return "default"


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def round_half_up_to_int(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.floor(x + 0.5).astype("Int64")


def parse_args():
    p = argparse.ArgumentParser(description="Phase2-7-1: 活動期間別受信書簡数（点年＋年代幅対応）")
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv")
    p.add_argument("--outdir", default="outputs/phase2_7_1")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--show", action="store_true", help="図を画面表示する")
    return p.parse_args()


def pid_for_ymd(ymd: int, periods):
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

    safe_years_by_pid = {}
    boundary_years = set()

    for y in range(year_min, year_max + 1):
        pid_jan1 = pid_for_ymd(y * 10000 + 101, periods)
        pid_dec31 = pid_for_ymd(y * 10000 + 1231, periods)

        if pid_jan1 is not None and pid_jan1 == pid_dec31:
            safe_years_by_pid.setdefault(pid_jan1, []).append(y)
        else:
            boundary_years.add(y)

    ranges = []
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



def main():
    args = parse_args()
    selected_font = set_japanese_font()

    csv_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_png = out_dir / f"phase2_7_1_activity_period_line_{ts}.png"
    out_log = out_dir / f"phase2_7_1_activity_period_log_{ts}.txt"

    df = pd.read_csv(csv_path, encoding=args.encoding)
    n_total = len(df)

    # ---- カラム設定 ----
    col_year_point = "年代_西暦"
    col_year_start = "年代_開始"
    col_year_end = "年代_終了"
    rep_candidates = ["年代_代表", "年代_代表値", "年代_代表値_西暦"]
    month_candidates = ["月", "月_数値", "month", "月日_月", "月_num"]
    day_candidates = ["日", "日_数値", "day", "月日_日", "日_num"]

    col_rep = pick_first_existing(df, rep_candidates)
    col_month = pick_first_existing(df, month_candidates)
    col_day = pick_first_existing(df, day_candidates)

    # ---- 数値化 ----
    df["_year_point_raw"] = pd.to_numeric(df.get(col_year_point), errors="coerce")
    if col_rep is not None:
        df["_year_rep_raw"] = pd.to_numeric(df.get(col_rep), errors="coerce")
    else:
        df["_year_rep_raw"] = np.nan

    df["_year_start_meiji"] = pd.to_numeric(df.get(col_year_start), errors="coerce")
    df["_year_end_meiji"] = pd.to_numeric(df.get(col_year_end), errors="coerce")
    df["_month"] = pd.to_numeric(df.get(col_month), errors="coerce") if col_month else np.nan
    df["_day"] = pd.to_numeric(df.get(col_day), errors="coerce") if col_day else np.nan

    # =========================
    # 1) 点データによる period 判定
    # =========================
    year_for_point = df["_year_point_raw"].where(df["_year_point_raw"].notna(), df["_year_rep_raw"])

    y_ser = round_half_up_to_int(year_for_point)
    m_ser = round_half_up_to_int(df["_month"])
    d_ser = round_half_up_to_int(df["_day"])

    has_y = y_ser.notna()
    has_m = m_ser.notna()
    has_d = d_ser.notna()

    ymd = (y_ser * 10000 + m_ser * 100 + d_ser).astype("Int64")
    ym = (y_ser * 100 + m_ser).astype("Int64")

    period_id = pd.Series(pd.NA, index=df.index, dtype="object")

    # ---- 年月日あり ----
    mask_ymd = has_y & has_m & has_d
    conds_ymd = [ymd.between(s, e) for _, _, _, s, e in PERIODS]
    sel_ymd = pd.Series(np.select([c.fillna(False) for c in conds_ymd], CHOICES, default=pd.NA), index=df.index)
    period_id.loc[mask_ymd] = sel_ymd.loc[mask_ymd]

    # ---- 年月あり・日なし（境界月は保留）----
    mask_ym_only = has_y & has_m & (~has_d)
    is_day_hold = mask_ym_only & ym.isin(list(DAY_SENSITIVE_MONTHS))
    period_id.loc[is_day_hold] = "境界月（日欠損）"

    mask_ym_safe = mask_ym_only & (~ym.isin(list(DAY_SENSITIVE_MONTHS)))
    periods_ym = [(pid, s // 100, e // 100) for pid, _, _, s, e in PERIODS]
    conds_ym = [ym.between(s_ym, e_ym) for _, s_ym, e_ym in periods_ym]
    sel_ym = pd.Series(np.select([c.fillna(False) for c in conds_ym], CHOICES, default=pd.NA), index=df.index)
    period_id.loc[mask_ym_safe] = sel_ym.loc[mask_ym_safe]

    # ---- 年のみ（境界年は保留）----
    mask_y_only = has_y & (~has_m)
    boundary_years, safe_ranges = build_year_rules(PERIODS)

    is_year_hold = mask_y_only & y_ser.isin(list(boundary_years))
    period_id.loc[is_year_hold] = "境界年（月欠損）"

    mask_y_safe = mask_y_only & (~y_ser.isin(list(boundary_years)))
    yy = y_ser
    conds_y = [yy.between(s, e) for (_pid, s, e) in safe_ranges]
    choices_y = [pid for (pid, _s, _e) in safe_ranges]
    sel_y = pd.Series(
        np.select([c.fillna(False) for c in conds_y], choices_y, default=pd.NA),
        index=df.index,
    )
    period_id.loc[mask_y_safe] = sel_y.loc[mask_y_safe]

    # =========================
    # 2) 年代幅（開始/終了）対応
    # =========================
    is_range = df["_year_start_meiji"].notna() & df["_year_end_meiji"].notna()

    start_year_g = round_half_up_to_int(df["_year_start_meiji"] + 1867)
    end_year_g = round_half_up_to_int(df["_year_end_meiji"] + 1867)

    start_ymd_range = (start_year_g * 10000 + 101).astype("Int64")
    end_ymd_range = (end_year_g * 10000 + 1231).astype("Int64")

    start_pid = pd.Series(pd.NA, index=df.index, dtype="object")
    end_pid = pd.Series(pd.NA, index=df.index, dtype="object")

    conds_start = [start_ymd_range.between(s, e) for _, _, _, s, e in PERIODS]
    conds_end = [end_ymd_range.between(s, e) for _, _, _, s, e in PERIODS]

    start_sel = pd.Series(np.select([c.fillna(False) for c in conds_start], CHOICES, default=pd.NA), index=df.index)
    end_sel = pd.Series(np.select([c.fillna(False) for c in conds_end], CHOICES, default=pd.NA), index=df.index)

    start_pid.loc[is_range] = start_sel.loc[is_range]
    end_pid.loc[is_range] = end_sel.loc[is_range]

    range_contained = is_range & start_pid.notna() & (start_pid == end_pid) & start_pid.isin(VALID_PIDS)
    period_id.loc[range_contained] = start_pid.loc[range_contained]

    df["活動期_id"] = period_id

    # =========================
    # 集計・ログ
    # =========================
    classifiable_mask = df["活動期_id"].isin(VALID_PIDS)
    n_classifiable = int(classifiable_mask.sum())

    has_any_year = df["_year_point_raw"].notna() | is_range | df["_year_rep_raw"].notna()
    n_valid_year = int(has_any_year.sum())
    valid_rate = (n_valid_year / n_total * 100) if n_total else 0.0

    n_year_missing = int((~has_any_year).sum())
    n_boundary_hold = int((df["活動期_id"] == "境界年（月欠損）").sum())
    n_day_hold = int((df["活動期_id"] == "境界月（日欠損）").sum())

    n_range_total = int(is_range.sum())
    n_range_contained = int(range_contained.sum())
    n_range_boundary = int((is_range & ~range_contained).sum())

    counts = {pid: int((df["活動期_id"] == pid).sum()) for pid, *_ in PERIODS}
    denom = n_classifiable if n_classifiable else 1
    pct = {pid: (counts[pid] / denom * 100) for pid in counts}

    log = []
    log.append("品川弥二郎書簡データ分析 Phase2-7-1: 活動期間別受信書簡数")
    log.append(f"分析開始時刻: {start_ts}")
    log.append(f"入力: {csv_path.resolve()}")
    log.append(f"出力ディレクトリ: {out_dir.resolve()}")
    log.append(f"エンコーディング: {args.encoding}")
    log.append(f"show: {args.show}")
    log.append(f"使用フォント: {selected_font}")
    log.append(f"ログ保存先: {out_log.resolve()}")
    log.append(f"図保存先: {out_png.resolve()}")
    log.append(f"全体データ: {n_total}件")
    log.append(f"年代分析用有効データ（点年 or 年代幅 or 代表年）: {n_valid_year}件")
    log.append(f"有効率: {valid_rate:.1f}%")
    log.append(
        f"使用カラム: '{col_year_point}'"
        + (f", '{col_month}'" if col_month else "（月カラムなし）")
        + (f", '{col_day}'" if col_day else "（日カラムなし）")
    )
    log.append(
        f"年代幅対応: '{col_year_start}', '{col_year_end}'"
        + (f", 代表='{col_rep}'" if col_rep else "（代表年カラムなし）")
    )
    log.append("代表年列は補助的に使用し、西暦列（年代_西暦）を優先して点判定する。")
    log.append("")
    log.append("=== 品川活動期間別受信書簡数 ===")
    log.append(f"活動期間分類可能データ: {n_classifiable}件")
    log.append(f"活動期間分類不可データ: {n_year_missing}件（年代情報なし）")

    if n_range_total:
        log.append(f"年代幅データ: {n_range_total}件")
        log.append(f"  - 単一活動期に収まるため採用: {n_range_contained}件")
        log.append(f"  - 境界またぎ等のため点判定（代表年）に委任: {n_range_boundary}件")

    if n_day_hold:
        log.append(f"（注意）境界月（日欠損）のため分類保留: {n_day_hold}件")
    if n_boundary_hold:
        log.append(f"（注意）境界年（月欠損）のため分類保留: {n_boundary_hold}件")

    for pid, pname, pdesc, *_ in PERIODS:
        log.append(f"{pid}.{pname}: {counts[pid]}件 ({pct[pid]:.1f}%) - {pdesc}")

    out_log.write_text("\n".join(log), encoding="utf-8")
    print("[LOG SAVED]", out_log.resolve())

    # =========================
    # 図（折れ線）
    # =========================
    x_labels = [f"{pid}.{pname}" for pid, pname, *_ in PERIODS]
    y_vals = [counts[pid] for pid, *_ in PERIODS]
    x = np.arange(len(x_labels))

    plt.figure(figsize=(14, 6))
    plt.plot(x, y_vals, marker="o", linewidth=2)
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel("受信書簡数（件）")
    plt.title("図3-1 品川の活動期間別 受信書簡数（点年＋年代幅対応）")
    plt.grid(axis="y", alpha=0.3)

    for xi, yi in zip(x, y_vals):
        plt.text(xi, yi, f"{yi}", ha="center", va="bottom", fontsize=10)

    plt.text(
        0.99,
        0.01,
        f"generated: {start_ts}",
        transform=plt.gca().transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    print("[SAVED]", out_png.resolve())


if __name__ == "__main__":
    main()
