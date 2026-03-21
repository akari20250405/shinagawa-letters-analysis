from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


LABEL = {
    0: "品川発信書簡確認できず",
    1: "品川発信書簡あり",
    2: "品川が発信書簡の有無を明記",
    3: "品川が何らかの反応を示した",
    4: "返書した旨の記載あり",
}
EXPECTED_VALUES = {0, 1, 2, 3, 4}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="発信書簡有無（0〜4）の分布チェック")
    p.add_argument("--input", required=True, help="Input CSV path (cleaned)")
    p.add_argument("--col", default="発信書簡有無", help="Target column name")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    p.add_argument("--outdir", default="outputs/phase1_2_1", help="Output directory")
    return p.parse_args()


def build_log_lines(df: pd.DataFrame, col: str, csv_path: Path) -> list[str]:
    s = df[col]
    n_total = len(s)
    n_missing = int(s.isna().sum())
    n_nonmissing = n_total - n_missing

    # 数値化（失敗したらNaNへ）
    s_num = pd.to_numeric(s, errors="coerce")
    n_numeric_valid = int(s_num.notna().sum())
    n_unexpected_non_numeric = n_nonmissing - n_numeric_valid

    vc = s_num.dropna().astype(int).value_counts().sort_index()

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("発信書簡有無（0〜4） 分布チェック")
    lines.append("=" * 60)
    lines.append(f"入力: {csv_path.resolve()}")
    lines.append(f"対象カラム: {col}")
    lines.append(f"総データ数: {n_total:,}件")
    lines.append(f"欠損値: {n_missing:,}件 ({(n_missing / n_total * 100 if n_total else 0):.1f}%)")
    lines.append(f"非欠損データ: {n_nonmissing:,}件 ({(n_nonmissing / n_total * 100 if n_total else 0):.1f}%)")
    lines.append(
        f"数値化成功データ: {n_numeric_valid:,}件 ({(n_numeric_valid / n_total * 100 if n_total else 0):.1f}%)"
    )
    lines.append(
        f"数値化失敗（非欠損だが数値でない値）: {n_unexpected_non_numeric:,}件 "
        f"({(n_unexpected_non_numeric / n_total * 100 if n_total else 0):.1f}%)"
    )
    lines.append("")
    lines.append("数値化成功データの分布:")

    for k in sorted(EXPECTED_VALUES):
        cnt = int(vc.get(k, 0))
        pct = (cnt / n_numeric_valid * 100) if n_numeric_valid else 0.0
        lines.append(f"  {k} ({LABEL.get(k, '不明')}): {cnt:,}件 ({pct:.1f}%)")

    observed_values = set(s_num.dropna().astype(int).unique())
    unexpected_numeric = sorted(observed_values - EXPECTED_VALUES)
    if unexpected_numeric:
        lines.append("")
        lines.append(f"⚠ 想定外の数値が混入: {unexpected_numeric}")

    return lines


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding=args.encoding)

    if args.col not in df.columns:
        raise KeyError(f"Missing column: {args.col}. Available columns: {list(df.columns)}")

    log_lines = build_log_lines(df=df, col=args.col, csv_path=csv_path)
    log_text = "\n".join(log_lines)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_log = outdir / f"phase1_2_1_hasshin_distribution_log_{ts}.txt"
    out_log.write_text(log_text, encoding="utf-8")

    print(log_text)
    print("")
    print(f"saved: {out_log.resolve()}")


if __name__ == "__main__":
    main()
