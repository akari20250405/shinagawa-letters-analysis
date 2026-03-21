from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "outputs/cleaning/shinagawa_letters_cleaned.csv"
DEFAULT_OUTDIR = "outputs/phase1_2_2"
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_COL_CODE = "発信書簡有無"
DEFAULT_COL_NAME = "発信者"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="発信書簡有無コード別に発信者一覧（頻度付き）をCSV出力する"
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned CSV path")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV encoding")
    p.add_argument("--col_code", default=DEFAULT_COL_CODE, help="Code column name")
    p.add_argument("--col_name", default=DEFAULT_COL_NAME, help="Sender name column name")
    return p.parse_args()


def build_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_sender_name(series: pd.Series) -> pd.Series:
    """欠損は欠損のまま保ちつつ、文字列値の前後空白のみ除去する。"""
    s = series.copy()
    mask = s.notna()
    s.loc[mask] = s.loc[mask].astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def export_name_list(sub: pd.DataFrame, col_name: str, out_path: Path) -> pd.DataFrame:
    vc = (
        sub[col_name]
        .dropna()
        .value_counts()
        .rename_axis("発信者")
        .reset_index(name="件数")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vc.to_csv(out_path, index=False, encoding="utf-8-sig")
    return vc


def summarize_group(label: str, sub: pd.DataFrame, vc: pd.DataFrame) -> list[str]:
    return [
        f"- {label} レコード件数: {len(sub):,}",
        f"- {label} ユニーク発信者数: {len(vc):,}",
    ]


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, encoding=args.encoding)

    required = [args.col_code, args.col_name]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    sender_series = normalize_sender_name(df[args.col_name])
    code_num = pd.to_numeric(df[args.col_code], errors="coerce")
    n_code_nonmissing = int(df[args.col_code].notna().sum())
    n_code_numeric = int(code_num.notna().sum())
    n_code_non_numeric = n_code_nonmissing - n_code_numeric
    unexpected_values = sorted(v for v in code_num.dropna().unique().tolist() if v not in [3, 4])

    work = df.copy()
    work[args.col_name] = sender_series
    work["_code_num"] = code_num

    df3 = work[work["_code_num"] == 3]
    df4 = work[work["_code_num"] == 4]
    df34 = work[work["_code_num"].isin([3, 4])]

    ts = build_timestamp()
    file3 = outdir / f"phase1_2_2_senders_code3_{ts}.csv"
    file4 = outdir / f"phase1_2_2_senders_code4_{ts}.csv"
    file34 = outdir / f"phase1_2_2_senders_code3_4_combined_{ts}.csv"
    log_path = outdir / f"phase1_2_2_sender_lists_log_{ts}.txt"

    vc3 = export_name_list(df3, args.col_name, file3)
    vc4 = export_name_list(df4, args.col_name, file4)
    vc34 = export_name_list(df34, args.col_name, file34)

    log_lines = [
        "Phase1_2_2 発信書簡有無コード別 発信者一覧 出力ログ",
        "=" * 60,
        f"入力ファイル: {csv_path.resolve()}",
        f"出力先: {outdir.resolve()}",
        f"使用カラム（コード）: {args.col_code}",
        f"使用カラム（発信者）: {args.col_name}",
        f"総レコード件数: {len(df):,}",
        f"コード列 非欠損件数: {n_code_nonmissing:,}",
        f"コード列 数値化成功件数: {n_code_numeric:,}",
        f"コード列 数値化失敗件数: {n_code_non_numeric:,}",
    ]
    if unexpected_values:
        log_lines.append(
            "コード列 想定外値: " + ", ".join(str(int(v)) if float(v).is_integer() else str(v) for v in unexpected_values)
        )
    else:
        log_lines.append("コード列 想定外値: なし")

    log_lines.extend(summarize_group("コード3", df3, vc3))
    log_lines.extend(summarize_group("コード4", df4, vc4))
    log_lines.extend(summarize_group("コード3+4", df34, vc34))
    log_lines.extend([
        f"出力ファイル（コード3）: {file3.name}",
        f"出力ファイル（コード4）: {file4.name}",
        f"出力ファイル（コード3+4）: {file34.name}",
    ])

    log_text = "\n".join(log_lines)
    log_path.write_text(log_text, encoding="utf-8")

    print(log_text)
    print(f"ログ保存: {log_path.resolve()}")


if __name__ == "__main__":
    main()
