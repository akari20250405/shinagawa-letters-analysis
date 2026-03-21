from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd

COL_CODE = "発信書簡有無"
COL_SENDER = "発信者"
COL_RES = "居住地_主"

TOKYO_LABELS_DEFAULT = {"東京", "東京府"}
OBS_CODES = {1, 2, 3, 4}
REACT_CODES = {3, 4}
FALLBACK_ATTR_COLS = ["属性", "属性_組み合わせ", "属性_リスト"]

ATTR_LABEL = {
    "1": "①毛利家関係者",
    "2": "②旧長州藩士・農兵隊",
    "3": "③政治家・官員（農商務省関係者）",
    "4": "④政治家・官員（宮内省関係者）",
    "5": "⑤政治家・官員（内務省本省関係者）",
    "6": "⑥政治家・官員（地方庁関係者）",
    "7": "⑦政治家・官員（第一次松方内閣関係者）",
    "8": "⑧政治家・官員（③～⑦以外）",
    "9": "⑨国民協会関係者",
    "10": "⑩実業家",
    "11": "⑪美術工芸作家",
    "12": "⑫僧侶",
    "13": "⑬神官",
    "14": "⑭維新志士関係者（長州藩以外）",
    "15": "⑮軍人",
    "16": "⑯ジャーナリスト",
    "17": "⑰真宗門徒・関係者",
    "18": "⑱品川家関係者",
    "19": "⑲老農・篤農家・産業実践者等",
    "20": "⑳独逸学関係者",
    "21": "㉑医師",
    "O": "O：その他",
    "ND": "ND：不明",
}

circled_map = {chr(0x2460 + i): str(i + 1) for i in range(20)}


def pct(n: float, d: float) -> float:
    return (n / d * 100) if d else 0.0


def extract_attrs_from_any(v) -> list[str]:
    if pd.isna(v):
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            out.extend(extract_attrs_from_any(x))
        return out

    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return []

    found = [circled_map[ch] for ch in s if ch in circled_map]
    if found:
        return found

    found2: list[str] = []
    if "ND" in s or "不明" in s:
        found2.append("ND")
    if re.search(r"\bO\b", s) or ("Ｏ" in s) or ("その他" in s) or ("其外" in s):
        found2.append("O")
    if found2:
        return list(dict.fromkeys(found2))

    return re.findall(r"\d+", s)


def detect_attr_cols(df: pd.DataFrame, attr_cols: list[str] | None) -> list[str]:
    if attr_cols:
        return attr_cols
    detected = [c for c in df.columns if re.fullmatch(r"属性\d+", c)]
    if detected:
        return sorted(detected, key=lambda x: int(x.replace("属性", "")))
    fallback = [c for c in FALLBACK_ATTR_COLS if c in df.columns]
    if fallback:
        return fallback
    raise KeyError("属性列が見つからん。属性1.. または fallback 候補列を確認して。")


def make_attr_long(df_obs: pd.DataFrame, attr_cols: list[str]) -> pd.DataFrame:
    pieces = []
    for c in attr_cols:
        tmp = df_obs[["_row_id", COL_SENDER, COL_RES, COL_CODE, "_is_react", "_is_tokyo", "_is_other_valid", c]].copy()
        tmp["属性_code"] = tmp[c].apply(extract_attrs_from_any)
        tmp = tmp.drop(columns=[c])
        pieces.append(tmp)

    attr_long = pd.concat(pieces, ignore_index=True)
    attr_long = attr_long.explode("属性_code")
    attr_long["属性_code"] = attr_long["属性_code"].fillna("ND").astype(str).str.strip()
    attr_long = attr_long[attr_long["属性_code"] != ""]
    attr_long["属性_label"] = attr_long["属性_code"].map(lambda x: ATTR_LABEL.get(str(x), f"(未定義){x}"))
    return attr_long


def prepare_observed(df: pd.DataFrame, tokyo_labels: set[str], exclude_foreign_and_unknown_for_other: bool) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d[COL_CODE] = pd.to_numeric(d[COL_CODE], errors="coerce")
    d["_row_id"] = np.arange(len(d))
    df_obs = d[d[COL_CODE].isin(list(OBS_CODES))].copy()
    df_obs["_is_react"] = df_obs[COL_CODE].isin(list(REACT_CODES))
    res = df_obs[COL_RES].fillna("不明").astype(str).str.strip()
    df_obs["_is_tokyo"] = res.isin(tokyo_labels)
    if exclude_foreign_and_unknown_for_other:
        df_obs["_is_other_valid"] = (~df_obs["_is_tokyo"]) & (~res.isin(["外国", "不明"]))
    else:
        df_obs["_is_other_valid"] = ~df_obs["_is_tokyo"]
    return df_obs, res.tolist()


def build_A_attr(df: pd.DataFrame, tokyo_labels: set[str] | None = None, attr_cols: list[str] | None = None,
                 exclude_foreign_and_unknown_for_other: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame, list[str]]:
    tokyo_labels = tokyo_labels or TOKYO_LABELS_DEFAULT
    attr_cols = detect_attr_cols(df, attr_cols)
    df_obs, _ = prepare_observed(df, tokyo_labels, exclude_foreign_and_unknown_for_other)
    attr_long = make_attr_long(df_obs, attr_cols)
    g = attr_long.groupby("属性_label", dropna=False)
    out = pd.DataFrame({
        "観測可能母集団(1〜4)": g.size(),
        "反応(3+4)": g["_is_react"].sum().astype(int),
    })
    out["反応率*(%)"] = out.apply(lambda r: pct(r["反応(3+4)"], r["観測可能母集団(1〜4)"]), axis=1)
    out = out.sort_values(["反応率*(%)", "反応(3+4)"], ascending=[False, False])

    tokyo_sub = df_obs[df_obs["_is_tokyo"]]
    other_sub = df_obs[df_obs["_is_other_valid"]]
    summary = {
        "A_all_denom": len(df_obs),
        "A_all_numer": int(df_obs["_is_react"].sum()),
        "A_all_rate": pct(int(df_obs["_is_react"].sum()), len(df_obs)),
        "A_tokyo_denom": len(tokyo_sub),
        "A_tokyo_numer": int(tokyo_sub["_is_react"].sum()),
        "A_tokyo_rate": pct(int(tokyo_sub["_is_react"].sum()), len(tokyo_sub)),
        "A_other_denom": len(other_sub),
        "A_other_numer": int(other_sub["_is_react"].sum()),
        "A_other_rate": pct(int(other_sub["_is_react"].sum()), len(other_sub)),
    }
    return out, summary, attr_long, attr_cols


def build_C_attr(df: pd.DataFrame, tokyo_labels: set[str] | None = None, attr_cols: list[str] | None = None,
                 exclude_foreign_and_unknown_for_other: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    tokyo_labels = tokyo_labels or TOKYO_LABELS_DEFAULT
    attr_cols = detect_attr_cols(df, attr_cols)
    df_obs, _ = prepare_observed(df, tokyo_labels, exclude_foreign_and_unknown_for_other)
    attr_long = make_attr_long(df_obs, attr_cols)

    sender_tbl = df_obs.dropna(subset=[COL_SENDER]).copy()
    sender_tbl[COL_SENDER] = sender_tbl[COL_SENDER].astype(str).str.strip()
    sender_tbl = sender_tbl[sender_tbl[COL_SENDER] != ""]

    sender_flag = sender_tbl.groupby(COL_SENDER, dropna=False)["_is_react"].any().rename("反応あり").reset_index()

    sl = attr_long.dropna(subset=[COL_SENDER]).copy()
    sl[COL_SENDER] = sl[COL_SENDER].astype(str).str.strip()
    sl = sl[sl[COL_SENDER] != ""]
    sender_attr = sl.groupby([COL_SENDER, "属性_label"], dropna=False)["_is_react"].any().reset_index()

    g = sender_attr.groupby("属性_label", dropna=False)
    out = pd.DataFrame({
        "発信者数（ユニーク）": g.size(),
        "反応あり発信者数": g["_is_react"].sum().astype(int),
    })
    out["人物ベース反応率*(%)"] = out.apply(lambda r: pct(r["反応あり発信者数"], r["発信者数（ユニーク）"]), axis=1)
    out = out.sort_values(["人物ベース反応率*(%)", "反応あり発信者数"], ascending=[False, False])

    sender_tokyo = sender_tbl.groupby(COL_SENDER)["_is_tokyo"].any().rename("東京扱い").reset_index()
    sender_other_valid = sender_tbl.groupby(COL_SENDER)["_is_other_valid"].any().rename("それ以外有効扱い").reset_index()
    sender_merge = sender_flag.merge(sender_tokyo, on=COL_SENDER, how="left").merge(sender_other_valid, on=COL_SENDER, how="left")

    tokyo_people = sender_merge[sender_merge["東京扱い"] == True]
    other_people = sender_merge[sender_merge["それ以外有効扱い"] == True]

    summary = {
        "C_all_denom": len(sender_flag),
        "C_all_numer": int(sender_flag["反応あり"].sum()),
        "C_all_rate": pct(int(sender_flag["反応あり"].sum()), len(sender_flag)),
        "C_tokyo_denom": len(tokyo_people),
        "C_tokyo_numer": int(tokyo_people["反応あり"].sum()),
        "C_tokyo_rate": pct(int(tokyo_people["反応あり"].sum()), len(tokyo_people)),
        "C_other_denom": len(other_people),
        "C_other_numer": int(other_people["反応あり"].sum()),
        "C_other_rate": pct(int(other_people["反応あり"].sum()), len(other_people)),
    }
    return out, summary, sender_merge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase1_2_2_1: build intermediate tables for rate figures.")
    p.add_argument("--input", required=True, help="Input dataset path (.csv or .parquet)")
    p.add_argument("--outdir", default="outputs/phase1_2_2_1", help="Output directory")
    p.add_argument("--encoding", default="utf-8-sig", help="CSV encoding if input is csv")
    return p.parse_args()


def load_data(path: Path, encoding: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding=encoding)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def save_table(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, encoding="utf-8-sig")


def save_summary(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = load_data(in_path, args.encoding)
    A_attr, A_sum, attr_long, attr_cols = build_A_attr(df)
    C_attr, C_sum, sender_merge = build_C_attr(df)

    a_path = outdir / f"phase1_2_2_1_A_attr_{ts}.csv"
    c_path = outdir / f"phase1_2_2_1_C_attr_{ts}.csv"
    s_path = outdir / f"phase1_2_2_1_rates_summary_{ts}.csv"
    log_path = outdir / f"phase1_2_2_1_build_tables_log_{ts}.txt"

    save_table(A_attr, a_path)
    save_table(C_attr, c_path)
    save_summary({**A_sum, **C_sum}, s_path)

    undef_mask = attr_long["属性_label"].astype(str).str.startswith("(未定義)")
    undef_counts = attr_long.loc[undef_mask, "属性_label"].value_counts().sort_values(ascending=False)

    lines = [
        "Phase1_2_2_1 build tables log",
        f"input: {in_path}",
        f"outdir: {outdir}",
        f"attr_cols: {', '.join(attr_cols)}",
        f"tokyo_labels: {', '.join(sorted(TOKYO_LABELS_DEFAULT))}",
        "other_definition: 東京以外 かつ 外国・不明を除外",
        "sender_tokyo_rule: 発信者単位で _is_tokyo が1件でも True なら 東京扱い",
        "sender_other_rule: 発信者単位で _is_other_valid が1件でも True なら それ以外有効扱い",
        f"rows_total: {len(df)}",
        f"A_all: {A_sum['A_all_numer']}/{A_sum['A_all_denom']} ({A_sum['A_all_rate']:.2f}%)",
        f"A_tokyo: {A_sum['A_tokyo_numer']}/{A_sum['A_tokyo_denom']} ({A_sum['A_tokyo_rate']:.2f}%)",
        f"A_other: {A_sum['A_other_numer']}/{A_sum['A_other_denom']} ({A_sum['A_other_rate']:.2f}%)",
        f"C_all: {C_sum['C_all_numer']}/{C_sum['C_all_denom']} ({C_sum['C_all_rate']:.2f}%)",
        f"C_tokyo: {C_sum['C_tokyo_numer']}/{C_sum['C_tokyo_denom']} ({C_sum['C_tokyo_rate']:.2f}%)",
        f"C_other: {C_sum['C_other_numer']}/{C_sum['C_other_denom']} ({C_sum['C_other_rate']:.2f}%)",
        f"sender_unique_total: {sender_merge[COL_SENDER].nunique()}",
        f"sender_tokyo_true: {int((sender_merge['東京扱い'] == True).sum())}",
        f"sender_other_valid_true: {int((sender_merge['それ以外有効扱い'] == True).sum())}",
        f"undefined_attr_rows: {int(undef_mask.sum())}",
        f"saved_A_attr: {a_path.name}",
        f"saved_C_attr: {c_path.name}",
        f"saved_rates_summary: {s_path.name}",
    ]
    if not undef_counts.empty:
        lines.append("undefined_attr_breakdown:")
        lines.extend([f"  - {k}: {int(v)}" for k, v in undef_counts.items()])

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[SAVED] {a_path.resolve()}")
    print(f"[SAVED] {c_path.resolve()}")
    print(f"[SAVED] {s_path.resolve()}")
    print(f"[SAVED] {log_path.resolve()}")


if __name__ == "__main__":
    main()
