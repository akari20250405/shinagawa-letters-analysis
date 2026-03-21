from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PHASE_NAME = "phase1"
DEFAULT_OUTDIR = Path("outputs/phase1")

COL_SHUROKU = "『品川文書』収録"
COL_HASSHIN = "発信書簡有無"
COL_MENSHIKI = "受信当初面識の有無"
COL_SUITEI = "推定年代"
REQUIRED_COLUMNS = [COL_SHUROKU, COL_HASSHIN, COL_MENSHIKI, COL_SUITEI]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate basic descriptive statistics log for Phase 1."
    )
    parser.add_argument("--input", required=True, help="Path to cleaned CSV file.")
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Input CSV encoding (default: utf-8-sig).",
    )
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"Output directory (default: {DEFAULT_OUTDIR}).",
    )
    return parser.parse_args()


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def num_str(value: object) -> str:
    """Format values for logs while preserving the original appearance as much as possible."""
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.1f}"
    return str(value)


def distribution_lines(
    series: pd.Series,
    label_map: dict | None = None,
    *,
    sort_values: bool = True,
) -> list[str]:
    """Return distribution lines for non-missing values."""
    valid = series.dropna()
    n_valid = len(valid)
    value_counts = valid.value_counts(dropna=False)

    if sort_values:
        try:
            value_counts = value_counts.sort_index()
        except Exception:
            pass

    lines: list[str] = []
    for value, count in value_counts.items():
        pct = (count / n_valid * 100) if n_valid else 0.0
        value_str = num_str(value)
        if label_map is not None and value in label_map:
            lines.append(
                f"  {value_str} ({label_map[value]}): {fmt_int(count)}件 ({pct:.1f}%)"
            )
        else:
            lines.append(
                f"  {value_str} (分類{value_str}): {fmt_int(count)}件 ({pct:.1f}%)"
            )
    return lines


def missing_info(series: pd.Series, n_total: int) -> tuple[int, float, int]:
    missing = int(series.isna().sum())
    valid = int(n_total - missing)
    missing_pct = (missing / n_total * 100) if n_total else 0.0
    return missing, missing_pct, valid


def build_phase1_log(df: pd.DataFrame, *, input_path: Path) -> str:
    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_total = len(df)
    n_cols = df.shape[1]
    missing_list: dict[str, tuple[int, float]] = {}

    log: list[str] = []
    log.append("品川弥二郎書簡データ分析 Phase 1: 基礎統計")
    log.append(f"分析開始時刻: {start_ts}")
    log.append(f"入力ファイル: {input_path}")
    log.append(f"データ読み込み完了: {fmt_int(n_total)}件, {fmt_int(n_cols)}カラム")

    # 1. 『品川文書』収録
    log.append("=" * 60)
    log.append("1. 「『品川文書』収録」の有無分析")
    log.append("=" * 60)

    s1 = df[COL_SHUROKU]
    na1, na1_pct, valid1 = missing_info(s1, n_total)
    missing_list[COL_SHUROKU] = (na1, na1_pct)

    log.append(f"総データ数: {fmt_int(n_total)}件")
    log.append(f"欠損値: {fmt_int(na1)}件 ({na1_pct:.1f}%)")
    log.append(f"有効データ: {fmt_int(valid1)}件 ({100 - na1_pct:.1f}%)")
    log.append("")
    log.append("有効データの分布:")
    log.extend(distribution_lines(s1, label_map={0: "未収録", 1: "収録済み"}))
    log.append("")

    # 2. 発信書簡有無
    log.append("=" * 60)
    log.append("2. 発信書簡有無分析")
    log.append("=" * 60)

    s2 = df[COL_HASSHIN]
    na2, na2_pct, valid2 = missing_info(s2, n_total)
    missing_list[COL_HASSHIN] = (na2, na2_pct)

    log.append(f"総データ数: {fmt_int(n_total)}件")
    log.append(f"欠損値: {fmt_int(na2)}件 ({na2_pct:.1f}%)")
    log.append(f"有効データ: {fmt_int(valid2)}件 ({100 - na2_pct:.1f}%)")
    log.append("")
    log.append("有効データの分布:")
    label_hasshin = {
        0: "品川発信書簡なし",
        1: "品川発信書簡あり",
        2: "品川が発信書簡の有無を明記",
        3: "品川が何らかの反応を示した",
        4: "返書した旨の記載あり",
    }
    log.extend(distribution_lines(s2, label_map=label_hasshin))
    log.append("")

    # 3. 受信当初面識の有無
    log.append("=" * 60)
    log.append("3. 受信当初面識の有無分析")
    log.append("=" * 60)

    s3 = df[COL_MENSHIKI]
    na3, na3_pct, valid3 = missing_info(s3, n_total)
    missing_list[COL_MENSHIKI] = (na3, na3_pct)

    log.append("【数値データ】")
    log.append(f"総データ数: {fmt_int(n_total)}件")
    log.append(f"欠損値: {fmt_int(na3)}件 ({na3_pct:.1f}%)")
    log.append(f"有効データ: {fmt_int(valid3)}件 ({100 - na3_pct:.1f}%)")
    log.append("")
    log.append("数値データの分布:")
    label_menshiki = {0.0: "面識なし", 1.0: "面識あり", 0: "面識なし", 1: "面識あり"}
    log.extend(distribution_lines(s3, label_map=label_menshiki))
    log.append("")

    # 4. 推定年代
    log.append("=" * 60)
    log.append("4. 推定年代分析")
    log.append("=" * 60)

    s4 = df[COL_SUITEI]
    na4, na4_pct, _valid4 = missing_info(s4, n_total)
    missing_list[COL_SUITEI] = (na4, na4_pct)

    log.append(f"総データ数: {fmt_int(n_total)}件")
    log.append("")
    log.append("推定年代の分布:")
    log.extend(distribution_lines(s4, label_map=None))
    log.append("")

    # Summary
    log.append("=" * 80)
    log.append("Phase 1: 基礎統計分析 総合サマリー")
    log.append("=" * 80)
    log.append("欠損率一覧:")
    for column in REQUIRED_COLUMNS:
        na, na_pct = missing_list[column]
        log.append(f"  {column}: {fmt_int(na)}件 ({na_pct:.1f}%)")
    log.append("")
    log.append("品質評価:")
    log.append(f"  総レコード数: {fmt_int(n_total)}件")
    log.append(f"  総カラム数: {fmt_int(n_cols)}列")

    complete = df[REQUIRED_COLUMNS].dropna()
    n_complete = len(complete)
    pct_complete = (n_complete / n_total * 100) if n_total else 0.0
    log.append(
        f"  完全データ（4項目すべて有効）: {fmt_int(n_complete)}件 ({pct_complete:.1f}%)"
    )
    log.append("")
    log.append("Phase 1 基礎統計分析完了")
    log.append("=" * 80)
    log.append("")
    log.append("【実行結果確認】")
    log.append(f"品川文書収録: {type(s1)}")
    log.append(f"発信書簡有無: {type(s2)}")
    log.append(f"受信当初面識: {type(s3)}")
    log.append(f"推定年代: {type(s4)}")

    return "\n".join(log)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding=args.encoding)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    log_text = build_phase1_log(df, input_path=input_path)
    print(log_text)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = outdir / f"{PHASE_NAME}_basic_stats_log_{timestamp}.txt"
    out_log.write_text(log_text, encoding="utf-8")
    print(f"\n[LOG SAVED] {out_log.resolve()}")


if __name__ == "__main__":
    main()
