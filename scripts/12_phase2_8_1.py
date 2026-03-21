from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="発信者ごとの書簡数を集計し、上位棒グラフと分布・パレート図を出力する"
    )
    p.add_argument(
        "--input",
        default="outputs/cleaning/shinagawa_letters_cleaned.csv",
        help="Input CSV path",
    )
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    p.add_argument("--outdir", default="outputs/phase2_8_1", help="Output directory")
    p.add_argument("--col_sender", default="発信者", help="Sender column name")
    p.add_argument("--top_n", type=int, default=10, help="Top N senders for bar chart")
    p.add_argument(
        "--important_th", type=int, default=10, help="Threshold for 'important' senders"
    )
    p.add_argument("--pareto_n", type=int, default=50, help="Top N for Pareto chart")
    p.add_argument("--show", action="store_true", help="Show plots interactively")
    return p.parse_args()


def choose_japanese_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for cand in FONT_CANDIDATES:
        if cand in available:
            plt.rcParams["font.family"] = cand
            return cand
    return "default"


def is_missing_sender(x: object) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    if s == "":
        return True
    if s.upper() in {"ND", "NAN", "NONE"}:
        return True
    return False


def normalize_sender(x: object) -> object:
    if is_missing_sender(x):
        return np.nan
    return str(x).strip()


def add_count_labels(ax, bars, is_horizontal: bool = False, fontsize: int = 10) -> None:
    for b in bars:
        if is_horizontal:
            w = b.get_width()
            y = b.get_y() + b.get_height() / 2
            ax.text(w + max(1, w * 0.01), y, f"{int(w):,}", va="center", fontsize=fontsize)
        else:
            h = b.get_height()
            x = b.get_x() + b.get_width() / 2
            ax.text(x, h + max(1, h * 0.01), f"{int(h):,}", ha="center", va="bottom", fontsize=fontsize)


def write_log(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_bar = out_dir / f"phase2_8_1_top{args.top_n}_senders_bar_{ts}.png"
    out_pareto = out_dir / f"phase2_8_1_sender_distribution_pareto_{ts}.png"
    out_top_csv = out_dir / f"phase2_8_1_top{args.top_n}_senders_table_{ts}.csv"
    out_dist_csv = out_dir / f"phase2_8_1_sender_distribution_table_{ts}.csv"
    out_log = out_dir / f"phase2_8_1_sender_distribution_log_{ts}.txt"

    plt.rcParams["axes.unicode_minus"] = False
    font_used = choose_japanese_font()

    df = pd.read_csv(in_path, encoding=args.encoding)
    if args.col_sender not in df.columns:
        raise KeyError(f"Missing column: {args.col_sender}\nAvailable: {list(df.columns)}")

    df["_sender"] = df[args.col_sender].apply(normalize_sender)

    n_total = len(df)
    n_unknown = int(df["_sender"].isna().sum())
    n_valid = n_total - n_unknown
    unknown_pct = (n_unknown / n_total * 100) if n_total else 0.0
    valid_pct = (n_valid / n_total * 100) if n_total else 0.0

    sender_counts = df.loc[df["_sender"].notna(), "_sender"].value_counts()
    n_senders = int(sender_counts.shape[0])

    if n_senders == 0:
        lines = [
            "# phase2_8_1 sender distribution log",
            f"input: {in_path}",
            f"encoding: {args.encoding}",
            f"outdir: {out_dir}",
            f"col_sender: {args.col_sender}",
            f"font_used: {font_used}",
            f"show: {args.show}",
            "",
            f"総書簡数: {n_total:,}件",
            f"発信者不明: {n_unknown:,}件 ({unknown_pct:.1f}%)",
            f"有効データ: {n_valid:,}件 ({valid_pct:.1f}%)",
            "総発信者数: 0名",
            "発信者が全員不明のため、図表は生成しなかった。",
        ]
        write_log(out_log, lines)
        print("発信者が全員不明やから、図は作らず終了。")
        print(f"[SAVED] {out_log.resolve()}")
        return

    avg_letters = (n_valid / n_senders) if n_senders else 0.0
    top_sender = str(sender_counts.index[0])
    top_sender_n = int(sender_counts.iloc[0])

    important = sender_counts[sender_counts >= args.important_th]
    n_important = int(important.shape[0])
    important_total = int(important.sum()) if n_important else 0
    important_pct_total = (important_total / n_valid * 100) if n_valid else 0.0

    # 上位発信者表
    topn_desc = sender_counts.head(args.top_n)
    topn_desc.rename_axis("発信者").reset_index(name="書簡数").to_csv(
        out_top_csv, index=False, encoding="utf-8-sig"
    )

    # 分布表
    count_freq = sender_counts.value_counts().sort_index()
    dist_df = count_freq.rename_axis("書簡数").reset_index(name="発信者数")
    dist_df.to_csv(out_dist_csv, index=False, encoding="utf-8-sig")

    # ① 上位棒グラフ
    topn = topn_desc.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(topn.index, topn.values)
    ax.set_title(f"図4-1 発信数 上位{args.top_n}名（人物ベース）")
    ax.set_xlabel("書簡数（件）")
    ax.set_ylabel("発信者")
    add_count_labels(ax, bars, is_horizontal=True, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_bar, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    # ② 分布 + パレート
    pareto = sender_counts.head(args.pareto_n)
    pareto_total = int(sender_counts.sum())
    cum_pct = (pareto.cumsum() / pareto_total) * 100 if pareto_total else pd.Series([], dtype=float)
    cum_all = (sender_counts.cumsum() / pareto_total) * 100 if pareto_total else pd.Series([], dtype=float)
    n80 = int(np.argmax(cum_all.values >= 80) + 1) if len(cum_all) else 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.bar(count_freq.index.astype(int), count_freq.values, width=0.9)
    ax.set_title("図4-2-1 発信者の書簡数分布")
    ax.set_xlabel("書簡数（件）")
    ax.set_ylabel("発信者数（人）")

    ax = axes[1]
    x = np.arange(1, len(pareto) + 1)
    ax.bar(x, pareto.values, alpha=0.6)
    ax.set_title("図4-2-2 上位発信者の寄与率")
    ax.set_xlabel("発信者順位（上位）")
    ax.set_ylabel("書簡数（件）")

    ax2 = ax.twinx()
    if len(pareto):
        ax2.plot(x, cum_pct.values, marker="o")
    ax2.set_ylabel("累積寄与率（%）")
    ax2.set_ylim(0, 100)
    ax2.axhline(80, linestyle="--")
    if n80:
        ax2.text(max(1, len(pareto) * 0.55), 82, f"上位{n80}名で80%", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_pareto, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    lines = [
        "# phase2_8_1 sender distribution log",
        f"input: {in_path}",
        f"encoding: {args.encoding}",
        f"outdir: {out_dir}",
        f"col_sender: {args.col_sender}",
        f"top_n: {args.top_n}",
        f"important_th: {args.important_th}",
        f"pareto_n: {args.pareto_n}",
        f"show: {args.show}",
        f"font_used: {font_used}",
        "",
        "発信者欠損は pd.NA に加えて、空文字・ND・nan・None 文字列も除外して集計した。",
        "",
        f"総書簡数: {n_total:,}件",
        f"発信者不明: {n_unknown:,}件 ({unknown_pct:.1f}%)",
        f"有効データ: {n_valid:,}件 ({valid_pct:.1f}%)",
        f"総発信者数: {n_senders:,}名",
        f"平均書簡数: {avg_letters:.2f}件/人",
        f"最多発信者: {top_sender} ({top_sender_n:,}件)",
        "",
        f"重要人物（{args.important_th}件以上）: {n_important:,}名",
        f"重要人物の書簡総数: {important_total:,}件 ({important_pct_total:.1f}%)",
        "",
        "--- パレート補助ログ ---",
        f"総書簡数（有効分）: {pareto_total:,}件",
        f"80%到達に必要な上位発信者数: {n80:,}名" if n80 else "80%到達に必要な上位発信者数: 該当なし",
        f"上位{args.pareto_n}名の累積寄与率: {cum_pct.values[-1]:.1f}%" if len(cum_pct) else f"上位{args.pareto_n}名の累積寄与率: 0.0%",
        "",
        f"top_senders_table: {out_top_csv}",
        f"distribution_table: {out_dist_csv}",
        f"top_bar_figure: {out_bar}",
        f"pareto_figure: {out_pareto}",
        f"log_file: {out_log}",
    ]
    write_log(out_log, lines)

    print("=================")
    print(f"総書簡数: {n_total:,}件")
    print(f"発信者不明: {n_unknown:,}件 ({unknown_pct:.1f}%)")
    print(f"有効データ: {n_valid:,}件 ({valid_pct:.1f}%)")
    print()
    print(f"総発信者数: {n_senders:,}名")
    print(f"平均書簡数: {avg_letters:.2f}件/人")
    print(f"最多発信者: {top_sender} ({top_sender_n:,}件)")
    print()
    print(f"重要人物（{args.important_th}件以上）: {n_important:,}名")
    print(f"重要人物の書簡総数: {important_total:,}件 ({important_pct_total:.1f}%)")
    print("\n--- パレート補助ログ ---")
    print(f"総書簡数（有効分）: {pareto_total:,}件")
    if n80:
        print(f"80%到達に必要な上位発信者数: {n80:,}名")
    if len(cum_pct):
        print(f"上位{args.pareto_n}名の累積寄与率: {cum_pct.values[-1]:.1f}%")
    print(f"\n[SAVED] {out_top_csv.resolve()}")
    print(f"[SAVED] {out_dist_csv.resolve()}")
    print(f"[SAVED] {out_bar.resolve()}")
    print(f"[SAVED] {out_pareto.resolve()}")
    print(f"[SAVED] {out_log.resolve()}")


if __name__ == "__main__":
    main()
