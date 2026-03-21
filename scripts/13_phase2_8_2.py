from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


UNKNOWN_TOKENS_DEFAULT = {"", "ND", "ＮＤ", "不明", "nan", "None"}
TARGET_ATTR_CODES_DEFAULT = ["⑩", "⑲"]
FONT_CANDIDATES = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic"]
ATTR_LABEL_MAP = {
    "①": "毛利家関係者",
    "②": "旧長州藩士・農兵隊",
    "③": "政治家・官員（農商務省関係者）",
    "④": "政治家・官員（宮内省関係者）",
    "⑤": "政治家・官員（内務省本省関係者）",
    "⑥": "政治家・官員（地方庁関係者）",
    "⑦": "政治家・官員（第一次松方内閣関係者）",
    "⑧": "政治家・官員（③～⑦以外）",
    "⑨": "国民協会関係者",
    "⑩": "実業家",
    "⑪": "美術工芸作家",
    "⑫": "僧侶",
    "⑬": "神官",
    "⑭": "維新志士関係者（長州藩以外）",
    "⑮": "軍人",
    "⑯": "ジャーナリスト",
    "⑰": "真宗門徒・関係者",
    "⑱": "品川家関係者",
    "⑲": "老農・篤農家・産業実践者等",
    "⑳": "独逸学関係者",
    "㉑": "医師",
    "O": "その他",
    "ND": "不明",
}
ATTR_CODE_PATTERN = re.compile(r"ND|O|㉑|[①-⑳]")
ATTR_N_COL_PATTERN = re.compile(r"^属性\d+$")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase2-8-2: 発信者分布の数値化ログを出力する")
    p.add_argument(
        "--input",
        default="outputs/cleaning/shinagawa_letters_cleaned.csv",
        help="Input CSV path",
    )
    p.add_argument("--encoding", default="utf-8-sig", help="Input CSV encoding")
    p.add_argument("--outdir", default="outputs/phase2_8_2", help="Output directory")

    p.add_argument("--col_sender", default="発信者", help="Sender column name")
    p.add_argument(
        "--col_attr",
        default="属性",
        help="Primary attribute column name (fallback: first available among 属性 / 属性1..)",
    )
    p.add_argument("--important_th", type=int, default=10, help="Important threshold (letters >= this)")
    p.add_argument("--top_attr_n", type=int, default=20, help="Top N senders to show dominant attribute")
    p.add_argument("--top_hypo_n", type=int, default=50, help="Top N senders for hypothesis check")
    p.add_argument(
        "--target_attr_codes",
        nargs="*",
        default=TARGET_ATTR_CODES_DEFAULT,
        help="Target attribute codes for hypothesis check (e.g. ⑩ ⑲)",
    )
    p.add_argument(
        "--unknown_tokens",
        nargs="*",
        default=sorted(UNKNOWN_TOKENS_DEFAULT),
        help="Tokens treated as unknown",
    )
    return p.parse_args()


def clean_text_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.replace("\u3000", " ", regex=False).str.strip()


def clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\u3000", " ").strip()


def is_unknown_series(s: pd.Series, unknown_tokens: set[str]) -> pd.Series:
    s2 = clean_text_series(s)
    return s.isna() | (s2 == "") | s2.isin(list(unknown_tokens))


def pick_font_name() -> str:
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in FONT_CANDIDATES:
            if name in available:
                return name
    except Exception:
        pass
    return FONT_CANDIDATES[0]


def find_attr_column(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    if "属性" in df.columns:
        return "属性"
    attr_n_cols = sorted([c for c in df.columns if ATTR_N_COL_PATTERN.match(c)], key=lambda x: int(x[2:]))
    return attr_n_cols[0] if attr_n_cols else None


def normalize_attr_value(x: object, unknown_tokens: set[str]) -> str:
    s = clean_text(x)
    if not s or s in unknown_tokens:
        return ""
    codes = ATTR_CODE_PATTERN.findall(s)
    return "".join(codes) if codes else s


def format_attr_display(attr: str) -> str:
    if not attr:
        return "（属性不明）"
    codes = ATTR_CODE_PATTERN.findall(attr)
    if not codes:
        return attr
    labels = [f"{c}{ATTR_LABEL_MAP.get(c, '')}" for c in codes]
    return " / ".join(labels)


def dominant_attr_for_senders(
    df_valid: pd.DataFrame,
    col_sender: str,
    col_attr: str | None,
    unknown_tokens: set[str],
    senders: list[str],
) -> dict[str, str]:
    if col_attr is None:
        return {s: "（属性列なし）" for s in senders}

    sub = df_valid.loc[df_valid[col_sender].isin(senders), [col_sender, col_attr]].copy()
    sub["_attr_norm"] = sub[col_attr].map(lambda x: normalize_attr_value(x, unknown_tokens))
    sub = sub.loc[sub["_attr_norm"] != ""]

    if len(sub) == 0:
        return {s: "（属性不明）" for s in senders}

    freq = sub.groupby([col_sender, "_attr_norm"]).size().reset_index(name="n")
    freq = freq.sort_values([col_sender, "n", "_attr_norm"], ascending=[True, False, True])
    best = freq.drop_duplicates(subset=[col_sender], keep="first")
    mapping = {str(k): format_attr_display(str(v)) for k, v in zip(best[col_sender], best["_attr_norm"])}

    for s in senders:
        mapping.setdefault(s, "（属性不明）")
    return mapping


def is_target_attr(attr: str, target_codes: set[str]) -> bool:
    codes = set(ATTR_CODE_PATTERN.findall(attr))
    return bool(codes & target_codes)


def main() -> None:
    args = parse_args()
    ts = timestamp_now()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_log = out_dir / f"phase2_8_2_sender_distribution_log_{ts}.txt"
    out_top_attr_csv = out_dir / f"phase2_8_2_top_senders_dominant_attr_{ts}.csv"
    out_hypo_csv = out_dir / f"phase2_8_2_hypothesis_hits_{ts}.csv"

    unknown_tokens = set(args.unknown_tokens)
    target_codes = set(args.target_attr_codes)
    target_labels = [f"{c}{ATTR_LABEL_MAP.get(c, '')}" for c in args.target_attr_codes]
    font_name = pick_font_name()

    df = pd.read_csv(in_path, encoding=args.encoding)
    if args.col_sender not in df.columns:
        raise KeyError(f"'{args.col_sender}' が見つからん。columns={list(df.columns)}")

    col_attr = find_attr_column(df, args.col_attr)

    sender_raw = df[args.col_sender]
    sender_unknown = is_unknown_series(sender_raw, unknown_tokens)

    n_total = len(df)
    n_sender_unknown = int(sender_unknown.sum())
    n_valid = n_total - n_sender_unknown

    df_valid = df.loc[~sender_unknown].copy()
    df_valid[args.col_sender] = clean_text_series(df_valid[args.col_sender])

    counts = df_valid[args.col_sender].value_counts()
    n_senders = int(counts.size)

    total_letters_valid = int(counts.sum())
    mean_letters = (total_letters_valid / n_senders) if n_senders else 0.0
    median_letters = float(counts.median()) if n_senders else 0.0

    top_sender = str(counts.index[0]) if n_senders else ""
    top_sender_n = int(counts.iloc[0]) if n_senders else 0

    singletons = int((counts == 1).sum())
    singletons_pct = (singletons / n_senders * 100) if n_senders else 0.0

    one_to_nine = int(((counts >= 1) & (counts <= 9)).sum())
    one_to_nine_pct = (one_to_nine / n_senders * 100) if n_senders else 0.0

    def top_share(p: float) -> tuple[int, float]:
        k = int(math.ceil(n_senders * p))
        k = max(k, 1) if n_senders else 0
        share = (counts.iloc[:k].sum() / total_letters_valid * 100) if total_letters_valid else 0.0
        return k, float(share)

    k1, s1 = top_share(0.01)
    k5, s5 = top_share(0.05)
    k10, s10 = top_share(0.10)

    if n_senders:
        threshold = total_letters_valid * 0.80
        cum = counts.cumsum().values
        k80 = int(np.argmax(cum >= threshold) + 1)
    else:
        k80 = 0
    k80_pct = (k80 / n_senders * 100) if n_senders else 0.0

    important_mask = counts >= args.important_th
    n_important = int(important_mask.sum())
    important_letters = int(counts.loc[important_mask].sum()) if n_important else 0
    important_share = (important_letters / total_letters_valid * 100) if total_letters_valid else 0.0

    top_attr_n = min(args.top_attr_n, n_senders)
    topN = counts.head(top_attr_n)
    senders_for_attr = [str(s) for s in topN.index.tolist()]
    dom_attr_map = dominant_attr_for_senders(
        df_valid=df_valid,
        col_sender=args.col_sender,
        col_attr=col_attr,
        unknown_tokens=unknown_tokens,
        senders=senders_for_attr,
    )

    top_attr_rows = []
    for i, (sender, c) in enumerate(topN.items(), start=1):
        top_attr_rows.append(
            {
                "rank": i,
                "sender": str(sender),
                "letters": int(c),
                "dominant_attr": dom_attr_map.get(str(sender), "（属性不明）"),
            }
        )
    pd.DataFrame(top_attr_rows).to_csv(out_top_attr_csv, index=False, encoding="utf-8-sig")

    top_hypo_n = min(args.top_hypo_n, n_senders)
    top_senders = [str(s) for s in counts.head(top_hypo_n).index.tolist()]
    dom_attr_top_map = dominant_attr_for_senders(
        df_valid=df_valid,
        col_sender=args.col_sender,
        col_attr=col_attr,
        unknown_tokens=unknown_tokens,
        senders=top_senders,
    )

    hypo_hits = []
    for sender in top_senders:
        attr = dom_attr_top_map.get(sender, "（属性不明）")
        if is_target_attr(attr, target_codes):
            hypo_hits.append(
                {
                    "sender": sender,
                    "letters": int(counts.loc[sender]),
                    "dominant_attr": attr,
                }
            )
    pd.DataFrame(hypo_hits, columns=["sender", "letters", "dominant_attr"]).to_csv(
        out_hypo_csv, index=False, encoding="utf-8-sig"
    )

    lines = []
    lines.append("===================================")
    lines.append("Phase2-8-2: 発信者分布の数値化ログ")
    lines.append("===================================")
    lines.append(f"入力: {in_path.resolve()}")
    lines.append(f"encoding: {args.encoding}")
    lines.append(f"出力先ディレクトリ: {out_dir.resolve()}")
    lines.append(f"ログ保存先: {out_log.resolve()}")
    lines.append(f"上位属性表CSV: {out_top_attr_csv.resolve()}")
    lines.append(f"仮説ヒットCSV: {out_hypo_csv.resolve()}")
    lines.append(f"発信者列: {args.col_sender}")
    lines.append(f"属性列（指定）: {args.col_attr}")
    lines.append(f"属性列（実使用）: {col_attr if col_attr is not None else '（属性列なし）'}")
    lines.append(f"important_th: {args.important_th}")
    lines.append(f"top_attr_n: {args.top_attr_n}")
    lines.append(f"top_hypo_n: {args.top_hypo_n}")
    lines.append(f"target_attr_codes: {args.target_attr_codes}")
    lines.append(f"target_attr_labels: {target_labels}")
    lines.append(f"unknown_tokens: {sorted(unknown_tokens)}")
    lines.append(f"使用フォント候補: {font_name}")
    lines.append("")
    lines.append(f"総書簡数: {n_total}件")
    lines.append(f"発信者不明: {n_sender_unknown}件 ({(n_sender_unknown / n_total * 100 if n_total else 0):.1f}%)")
    lines.append(f"有効データ: {n_valid}件 ({(n_valid / n_total * 100 if n_total else 0):.1f}%)")
    lines.append("")
    lines.append(f"総発信者数: {n_senders}名")
    lines.append(f"平均書簡数: {mean_letters:.2f}件/人")
    lines.append(f"中央値: {int(median_letters) if n_senders else 0}件")
    lines.append(f"最多発信者: {top_sender} ({top_sender_n}件)")
    lines.append("")
    lines.append(f"1件のみの発信者: {singletons}人 ({singletons_pct:.1f}%)")
    lines.append(f"発信書簡1-9件の発信者: {one_to_nine}人 ({one_to_nine_pct:.1f}%)")
    lines.append(f"上位1%（約{k1}人）で全体の {s1:.1f}%")
    lines.append(f"上位5%（約{k5}人）で全体の {s5:.1f}%")
    lines.append(f"上位10%（約{k10}人）で全体の {s10:.1f}%")
    lines.append(f"全体80%到達に必要な発信者: 上位{k80}人 ({k80_pct:.1f}%)")
    lines.append("")
    lines.append(f"重要人物（{args.important_th}件以上）: {n_important}名")
    lines.append(f"重要人物の書簡総数: {important_letters}件 ({important_share:.1f}%)")
    lines.append("")
    lines.append("===================================")
    lines.append(f"発信数 上位{top_attr_n}名: 支配的属性（ざっくり確認）")
    lines.append("===================================")
    for row in top_attr_rows:
        lines.append(f"{row['rank']:2d}. {row['sender']}: {row['letters']}件 / 主属性: {row['dominant_attr']}")
    lines.append("")
    lines.append("===================================")
    lines.append(f"仮説チェック: 上位{top_hypo_n}に入るターゲット属性（{', '.join(target_labels)}）")
    lines.append("===================================")
    if hypo_hits:
        for row in hypo_hits:
            lines.append(f"- {row['sender']}: {row['letters']}件 / 主属性: {row['dominant_attr']}")
    else:
        lines.append("該当なし")

    text = "\n".join(lines)
    print(text)
    out_log.write_text(text, encoding="utf-8")
    print(f"\n[SAVED] {out_log.resolve()}")


if __name__ == "__main__":
    main()
