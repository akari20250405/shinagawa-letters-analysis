from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


def setup_jp_font() -> str:
    candidates = ["MS Gothic", "Yu Gothic", "IPAexGothic", "Noto Sans CJK JP"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = "sans-serif"
    for f in candidates:
        if f in available:
            chosen = f
            break
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1_2_2_1: make figures from intermediate tables.")
    p.add_argument("--indir", default="outputs/phase1_2_2_1", help="Directory containing intermediate CSVs")
    p.add_argument("--figdir", default=None, help="Figure output directory; default is <indir>/figures")
    p.add_argument("--show", action="store_true", help="Display figures interactively")
    return p.parse_args()


def latest_file(indir: Path, pattern: str) -> Path:
    matches = sorted(indir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing file matching: {indir / pattern}")
    return matches[-1]


def savefig(out_png: Path, show: bool, dpi: int = 300) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    print(f"[PNG SAVED] {out_png.resolve()}")
    if show:
        plt.show()
    plt.close()


def bar_rate_with_labels(df: pd.DataFrame, label_col: str, rate_col: str, numer_col: str | None = None,
                         denom_col: str | None = None, title: str = "", xlabel: str = "反応率（%）",
                         figsize: tuple[int, int] = (12, 7), top_n: int = 20, min_denom: int = 20,
                         out_png: Path | None = None, show: bool = False) -> None:
    d = df.copy()
    if denom_col and denom_col in d.columns:
        d = d[d[denom_col] >= min_denom]
    d = d.sort_values(rate_col, ascending=True).tail(top_n)

    labels = d[label_col].tolist()
    rates = d[rate_col].tolist()
    plt.figure(figsize=figsize)
    bars = plt.barh(labels, rates)
    plt.title(title)
    plt.xlabel(xlabel)
    xmax = max(rates) if rates else 0
    offset = xmax * 0.01 if xmax else 0.5

    for i, (b, r) in enumerate(zip(bars, rates)):
        txt = f"{r:.1f}%"
        if numer_col and denom_col and numer_col in d.columns and denom_col in d.columns:
            txt += f"  ({int(d.iloc[i][numer_col])}/{int(d.iloc[i][denom_col])})"
        plt.text(b.get_width() + offset, b.get_y() + b.get_height() / 2, txt, va="center", ha="left")

    plt.tight_layout()
    if out_png is not None:
        savefig(out_png, show=show)


def bar2_rate(tokyo_rate: float, other_rate: float, title: str = "", out_png: Path | None = None,
              show: bool = False) -> None:
    plt.figure(figsize=(7, 5))
    plt.bar(["東京", "それ以外"], [tokyo_rate, other_rate])
    plt.title(title)
    plt.ylabel("反応率（%）")
    for x, v in zip(["東京", "それ以外"], [tokyo_rate, other_rate]):
        plt.text(x, v, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    if out_png is not None:
        savefig(out_png, show=show)


def main() -> None:
    args = parse_args()
    indir = Path(args.indir)
    figdir = Path(args.figdir) if args.figdir else indir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    font_used = setup_jp_font()

    a_path = latest_file(indir, "phase1_2_2_1_A_attr_*.csv")
    c_path = latest_file(indir, "phase1_2_2_1_C_attr_*.csv")
    r_path = latest_file(indir, "phase1_2_2_1_rates_summary_*.csv")

    A_attr = pd.read_csv(a_path, encoding="utf-8-sig", index_col=0)
    C_attr = pd.read_csv(c_path, encoding="utf-8-sig", index_col=0)
    rates = pd.read_csv(r_path, encoding="utf-8-sig")
    r = rates.iloc[0]

    A_plot = A_attr.reset_index().rename(columns={
        "属性_label": "label", "反応率*(%)": "rate", "反応(3+4)": "numer", "観測可能母集団(1〜4)": "denom"
    })
    A_plot_noND = A_plot[~A_plot["label"].astype(str).str.startswith("ND")].copy()

    C_plot = C_attr.reset_index().rename(columns={
        "属性_label": "label", "人物ベース反応率*(%)": "rate", "反応あり発信者数": "numer", "発信者数（ユニーク）": "denom"
    })
    C_plot_noND = C_plot[~C_plot["label"].astype(str).str.startswith("ND")].copy()

    f1 = figdir / f"phase1_2_2_1_A1_tokyo_vs_other_letters_rate_{ts}.png"
    f2 = figdir / f"phase1_2_2_1_A2_attr_letters_rate_top20_noND_{ts}.png"
    f3 = figdir / f"phase1_2_2_1_C1_tokyo_vs_other_senders_rate_{ts}.png"
    f4 = figdir / f"phase1_2_2_1_C2_attr_senders_rate_top20_noND_{ts}.png"
    log_path = figdir / f"phase1_2_2_1_make_figures_log_{ts}.txt"

    bar2_rate(float(r["A_tokyo_rate"]), float(r["A_other_rate"]),
              title="図1-1 東京 vs それ以外（lettersベース, 反応率*）", out_png=f1, show=args.show)
    bar_rate_with_labels(A_plot_noND, label_col="label", rate_col="rate", numer_col="numer", denom_col="denom",
                         title="図2-1 属性別 反応率*（lettersベース, ND除外）※分母>=20",
                         figsize=(14, 8), top_n=20, min_denom=20, out_png=f2, show=args.show)
    bar2_rate(float(r["C_tokyo_rate"]), float(r["C_other_rate"]),
              title="図1-2 東京 vs それ以外（人物ベース, 反応率*）", out_png=f3, show=args.show)
    bar_rate_with_labels(C_plot_noND, label_col="label", rate_col="rate", numer_col="numer", denom_col="denom",
                         title="図2-2 属性別 人物ベース反応率*（ND除外）※分母>=5",
                         figsize=(14, 8), top_n=20, min_denom=5, out_png=f4, show=args.show)

    lines = [
        "Phase1_2_2_1 make figures log",
        f"indir: {indir}",
        f"figdir: {figdir}",
        f"font_used: {font_used}",
        f"show: {args.show}",
        f"input_A_attr: {a_path.name}",
        f"input_C_attr: {c_path.name}",
        f"input_rates_summary: {r_path.name}",
        f"saved_figure: {f1.name}",
        f"saved_figure: {f2.name}",
        f"saved_figure: {f3.name}",
        f"saved_figure: {f4.name}",
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {log_path.resolve()}")


if __name__ == "__main__":
    main()
