from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ========= Utilities =========
def save_png(path: Path, dpi: int = 300) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[PNG SAVED] {path.resolve()}")


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def norm_text(s: pd.Series) -> pd.Series:
    s = s.fillna("不明").astype(str).str.replace("\u3000", " ", regex=False)
    s = s.str.strip()
    s = s.replace({"": "不明", "nan": "不明", "None": "不明"})
    return s


def set_japanese_font() -> str | None:
    candidates = [
        "MS Gothic",
        "Yu Gothic",
        "Meiryo",
        "IPAexGothic",
        "IPAGothic",
        "Noto Sans CJK JP",
        "TakaoGothic",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/cleaning/shinagawa_letters_cleaned.csv")
    p.add_argument("--outdir", default="outputs/phase2_5")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--top_bar", type=int, default=20)
    p.add_argument("--top_barh", type=int, default=30)
    p.add_argument("--top_pie", type=int, default=5)
    p.add_argument("--include_unknown_and_foreign", action="store_true")
    p.add_argument("--show", action="store_true", help="Show figures interactively")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    used_font = set_japanese_font()

    csv_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()
    log_path = out_dir / f"phase2_5_birthplace_ranking_log_{ts}.txt"

    df = pd.read_csv(csv_path, encoding=args.encoding)

    # 出生地カラムを自動選択（正規化済み優先）
    col_birth = pick_col(df, ["出生地_主", "出生地", "出生地_元"])
    if col_birth is None:
        raise KeyError("出生地の列が見つからん。csvの列名を確認してくれ。")

    s_birth = norm_text(df[col_birth])
    counts_all = s_birth.value_counts(dropna=False)

    if args.include_unknown_and_foreign:
        counts = counts_all.copy()
    else:
        counts = counts_all.drop(index=[i for i in ["不明", "外国"] if i in counts_all.index])

    n_total = len(df)
    n_unknown = int((s_birth == "不明").sum())
    n_foreign = int((s_birth == "外国").sum())
    n_analysis = int(counts.sum())

    # ========= Log =========
    log = []
    log.append("品川弥二郎書簡データ分析 Phase 2-5: 出生地ランキング")
    log.append(f"分析開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.append(f"入力ファイル: {csv_path}")
    log.append(f"出力先: {out_dir}")
    log.append(f"使用フォント: {used_font if used_font else '未検出（matplotlib既定値）'}")
    log.append(f"データ読み込み完了: {n_total:,}件, {df.shape[1]}カラム")
    log.append("=" * 60)
    log.append("5. 発信者の出生地ランキング（全順位）")
    log.append("=" * 60)
    log.append(f"使用カラム: {col_birth}")
    log.append(f"top_bar: {args.top_bar}")
    log.append(f"top_barh: {args.top_barh}")
    log.append(f"top_pie: {args.top_pie}")
    log.append(f"include_unknown_and_foreign: {args.include_unknown_and_foreign}")
    log.append("割合は総レコード数を分母に計算")
    log.append(f"総レコード数: {n_total:,}件")
    log.append(f"分析対象件数: {n_analysis:,}件")
    log.append(f"欠損（不明）: {n_unknown:,}件 ({n_unknown / n_total * 100:.1f}%)")
    log.append(f"外国: {n_foreign:,}件 ({n_foreign / n_total * 100:.1f}%)")
    log.append("")
    log.append("分布（全順位）: 地名: 件数 (割合)")
    for name, c in counts.items():
        log.append(f"  {name}: {int(c):,}件 ({c / n_total * 100:.1f}%)")
    log.append("")
    log.append("=" * 60)
    log.append("Phase 2-5 出生地ランキング分析完了")
    log.append("=" * 60)

    log_path.write_text("\n".join(log), encoding="utf-8")
    print(f"[LOG SAVED] {log_path.resolve()}")

    # ========= 1) Vertical bar =========
    top_bar = counts.head(args.top_bar)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(top_bar.index, top_bar.values)
    plt.title(f"出生地ランキング（上位{args.top_bar}）")
    plt.ylabel("件数")
    plt.xticks(rotation=45, ha="right")

    ymax = int(top_bar.max()) if len(top_bar) else 0
    offset = max(1, int(ymax * 0.01)) if ymax > 0 else 1
    for b, v in zip(bars, top_bar.values):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + offset,
            f"{int(v):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    save_png(out_dir / f"phase2_5_birthplace_bar_top{args.top_bar}_labeled_{ts}.png")
    if args.show:
        plt.show()
    plt.close()

    # ========= 2) Horizontal bar =========
    top_barh = counts.head(args.top_barh).sort_values(ascending=True)
    plt.figure(figsize=(12, 9))
    bars = plt.barh(top_barh.index, top_barh.values)
    plt.title(f"出生地ランキング（上位{args.top_barh}）")
    plt.xlabel("件数")

    xmax = int(top_barh.max()) if len(top_barh) else 0
    offset = max(1, int(xmax * 0.01)) if xmax > 0 else 1
    for b, v in zip(bars, top_barh.values):
        plt.text(
            b.get_width() + offset,
            b.get_y() + b.get_height() / 2,
            f"{int(v):,}",
            va="center",
            ha="left",
        )

    plt.tight_layout()
    save_png(out_dir / f"phase2_5_birthplace_barh_top{args.top_barh}_labeled_{ts}.png")
    if args.show:
        plt.show()
    plt.close()

    # ========= 3) Pie =========
    top_pie = counts.head(args.top_pie)
    others = counts.iloc[args.top_pie:].sum()
    pie_labels = list(top_pie.index)
    pie_values = list(top_pie.values)
    if int(others) > 0:
        pie_labels.append("その他")
        pie_values.append(others)

    plt.figure(figsize=(8, 8))
    plt.pie(pie_values, labels=pie_labels, autopct="%1.1f%%")
    plt.title(f"出生地ランキング（上位{args.top_pie} + その他）")
    plt.tight_layout()
    save_png(out_dir / f"phase2_5_birthplace_pie_top{args.top_pie}_others_{ts}.png")
    if args.show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    main()
