from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "outputs/cleaning/shinagawa_letters_cleaned.csv"
DEFAULT_OUTDIR = "outputs/phase1_3"
DEFAULT_ENCODING = "utf-8-sig"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="受信当初面識の有無=0（面識なし）の発信者一覧と該当レコードを出力する"
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input cleaned CSV path")
    p.add_argument("--encoding", default=DEFAULT_ENCODING, help="CSV encoding")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")

    p.add_argument("--col_acq", default="受信当初面識の有無", help="Acquaintance column name")
    p.add_argument("--col_sender", default="発信者", help="Sender column name")

    # 表示用（存在すれば出す）
    p.add_argument("--col_id", default="整理番号", help="Optional id column")
    p.add_argument("--col_year", default="年代_代表値", help="Optional year column")
    p.add_argument("--col_md", default="月日_代表値", help="Optional month-day representative column")

    p.add_argument(
        "--max_rows",
        type=int,
        default=200,
        help="Max rows to print for detail table in console (0 = no detail print)",
    )
    return p.parse_args()



def normalize_acq_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace("０", "0", regex=False),
        errors="coerce",
    )



def build_log_text(
    *,
    csv_path: Path,
    df: pd.DataFrame,
    df_no: pd.DataFrame,
    df_no_disp: pd.DataFrame,
    sender_list: list[str],
    col_acq: str,
    max_rows: int,
    ts_human: str,
) -> tuple[str, str]:
    # console 用
    lines_console: list[str] = []
    lines_console.append("============================================================")
    lines_console.append("面識なし（0）書簡の発信者一覧（Phase1_3ログ）")
    lines_console.append(f"出力時刻: {ts_human}")
    lines_console.append("============================================================")
    lines_console.append(f"入力: {csv_path.resolve()}")
    lines_console.append(f"総データ数: {len(df):,}件")
    lines_console.append(f"面識なし（0）判定件数: {len(df_no):,}件")

    s_num = normalize_acq_series(df[col_acq])
    n_nonmissing_raw = int(df[col_acq].notna().sum())
    n_numeric_valid = int(s_num.notna().sum())
    n_non_numeric = int(n_nonmissing_raw - n_numeric_valid)

    lines_console.append(f"面識列 非欠損件数: {n_nonmissing_raw:,}件")
    lines_console.append(f"面識列 数値化成功件数: {n_numeric_valid:,}件")
    lines_console.append(f"面識列 数値化失敗件数: {n_non_numeric:,}件")
    lines_console.append("")
    lines_console.append("【発信者（ユニーク）一覧】")
    lines_console.append(f"ユニーク人数: {len(sender_list)}人")
    for name in sender_list:
        lines_console.append(f" - {name}")
    lines_console.append("")
    lines_console.append("【該当レコード詳細（確認用）】")

    detail_text = df_no_disp.to_string(index=False)
    if max_rows == 0:
        lines_console.append("(詳細表は非表示: --max_rows 0)")
    elif len(df_no_disp) > max_rows:
        lines_console.append(f"(表示は先頭 {max_rows} 行のみ: --max_rows で調整可)")
        lines_console.append(df_no_disp.head(max_rows).to_string(index=False))
    else:
        lines_console.append(detail_text)
    lines_console.append("============================================================")

    # file 用（常に全件）
    lines_file: list[str] = []
    lines_file.append("============================================================")
    lines_file.append("面識なし（0）書簡の発信者一覧（Phase1_3ログ）")
    lines_file.append(f"出力時刻: {ts_human}")
    lines_file.append("============================================================")
    lines_file.append(f"入力: {csv_path.resolve()}")
    lines_file.append(f"総データ数: {len(df):,}件")
    lines_file.append(f"面識なし（0）判定件数: {len(df_no):,}件")
    lines_file.append(f"面識列 非欠損件数: {n_nonmissing_raw:,}件")
    lines_file.append(f"面識列 数値化成功件数: {n_numeric_valid:,}件")
    lines_file.append(f"面識列 数値化失敗件数: {n_non_numeric:,}件")
    lines_file.append("")
    lines_file.append("【発信者（ユニーク）一覧】")
    lines_file.append(f"ユニーク人数: {len(sender_list)}人")
    for name in sender_list:
        lines_file.append(f" - {name}")
    lines_file.append("")
    lines_file.append("【該当レコード詳細（全件）】")
    lines_file.append(detail_text)
    lines_file.append("============================================================")

    return "\n".join(lines_console), "\n".join(lines_file)



def main() -> None:
    args = parse_args()
    csv_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_human = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_log = outdir / f"phase1_3_no_acquaintance_senders_log_{ts}.txt"

    df = pd.read_csv(csv_path, encoding=args.encoding)

    missing = [c for c in [args.col_acq, args.col_sender] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")

    s_num = normalize_acq_series(df[args.col_acq])
    mask_no = s_num == 0
    df_no = df.loc[mask_no].copy()

    cols: list[str] = []
    for c in [args.col_id, args.col_sender, args.col_year, args.col_md, args.col_acq]:
        if c and c in df_no.columns:
            cols.append(c)
    df_no_disp = df_no[cols].copy() if cols else df_no.copy()

    senders = df_no[args.col_sender].dropna().astype(str).str.strip()
    senders = senders[(senders != "") & (senders.str.lower() != "nan")]
    sender_list = sorted(senders.unique())

    log_console, log_file = build_log_text(
        csv_path=csv_path,
        df=df,
        df_no=df_no,
        df_no_disp=df_no_disp,
        sender_list=sender_list,
        col_acq=args.col_acq,
        max_rows=args.max_rows,
        ts_human=ts_human,
    )

    print(log_console)
    out_log.write_text(log_file, encoding="utf-8")
    print(f"\n[LOG SAVED] {out_log.resolve()}")


if __name__ == "__main__":
    main()
