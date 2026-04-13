from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

ERA_START_GREG = {"M": 1868, "T": 1912, "S": 1926, "H": 1989, "R": 2019, "K": 1865}
ATTR_TOKEN_RE = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑O]")

PREFECTURES = set(
    """北海道 青森 岩手 宮城 秋田 山形 福島 茨城 栃木 群馬 埼玉 千葉 東京 神奈川 新潟 富山 石川 福井 山梨 長野 岐阜 静岡 愛知 三重 滋賀 京都 大阪 兵庫 奈良 和歌山 鳥取 島根 岡山 広島 山口 徳島 香川 愛媛 高知 福岡 佐賀 長崎 熊本 大分 宮崎 鹿児島 沖縄""".split()
)
CITY_TO_PREF = {"神戸": "兵庫"}
FOREIGN_MARKERS = {
    "朝鮮", "清国", "台湾", "韓国", "アメリカ", "イギリス", "フランス", "ドイツ",
    "オランダ", "ベルギー", "オーストリア", "タイ", "ニューヨーク"
}
ALLOWED_REGIONS = {"北海道", "東北", "関東", "甲信越", "北陸", "東海", "近畿", "中国", "四国", "九州", "沖縄", "外国"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean shinagawa letters metadata table and export normalized outputs.")
    p.add_argument("--input", required=True, help="Input Excel file (.xlsx)")
    p.add_argument("--sheet", default="Table1", help="Sheet name to read")
    p.add_argument("--outdir", default="outputs/cleaning", help="Output directory")
    return p.parse_args()


# ===== helpers =====
def is_datetime_like(x: object) -> bool:
    return isinstance(x, (pd.Timestamp, dt.datetime))


def strip_all(x: object) -> object:
    if pd.isna(x):
        return x
    if is_datetime_like(x):
        return x
    s = str(x).replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_missing_text(x: object) -> object:
    if pd.isna(x):
        return np.nan
    if is_datetime_like(x):
        return x
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    if s in {"不明", "ND", "ＮＤ", "na", "NA"}:
        return np.nan
    return s

def parse_era_year(s: object) -> tuple[object, float, float, float]:
    if pd.isna(s):
        return (np.nan, np.nan, np.nan, np.nan)

    s = str(s).strip()
    s_std = (
        s.replace("〜", "～").replace("~", "～")
        .replace("－", "～").replace("-", "～")
        .replace("—", "～").replace("–", "～").replace("−", "～")
    )

    m = re.match(r"^([A-Za-z]+)\s*(.*)$", s_std)
    if not m:
        return (np.nan, np.nan, np.nan, np.nan)

    era, rest = m.group(1), m.group(2).strip()
    if rest == "":
        return (era, np.nan, np.nan, np.nan)

    if "～" in rest:
        a, b = rest.split("～", 1)
        a, b = a.strip(), b.strip()
        y1 = float(a) if a.isdigit() else np.nan
        y2 = float(b) if b.isdigit() else np.nan

        if not np.isnan(y1) and not np.isnan(y2):
            rep = (y1 + y2) / 2
        else:
            rep = y1 if not np.isnan(y1) else np.nan

        return (era, y1, y2, rep)

    if rest.isdigit():
        y = float(rest)
        return (era, y, y, y)

    return (era, np.nan, np.nan, np.nan)

def era_year_to_greg(era: object, y: object) -> float:
    if pd.isna(era) or pd.isna(y):
        return np.nan
    era = str(era)
    if era not in ERA_START_GREG:
        return np.nan
    return round(ERA_START_GREG[era] + (float(y) - 1))

def excel_serial_to_timestamp(x: object) -> pd.Timestamp | None:
    """
    Excel serial date を Timestamp に変換する。
    1899-12-30 起点（pandas / Excel互換）を使う。
    """
    try:
        v = float(x)
    except Exception:
        return None

    if np.isnan(v):
        return None

    # 月日列として入ってくる serial はだいたいこの範囲
    # あまり広く取りすぎると普通の数値を誤認するので軽く制限
    if not (30000 <= v <= 60000):
        return None

    try:
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(v, unit="D")
    except Exception:
        return None

def parse_month_day(x: object) -> tuple[float, float, bool, object]:
    if pd.isna(x):
        return (np.nan, np.nan, False, None)

    # 1) datetime / Timestamp
    if is_datetime_like(x):
        ts = pd.Timestamp(x)
        return (float(ts.month), float(ts.day), True, "datetime")

    # 2) Excel serial date（例: 46334）
    ts_from_serial = excel_serial_to_timestamp(x)
    if ts_from_serial is not None:
        return (float(ts_from_serial.month), float(ts_from_serial.day), True, "excel_serial")

    # 3) 文字列として解釈
    s = str(x).strip()
    if s == "":
        return (np.nan, np.nan, False, "blank")

    # ISO形式: 2026-11-08
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m_iso:
        return (float(int(m_iso.group(2))), float(int(m_iso.group(3))), True, "iso")

    # 文字列が数値だけなら、Excel serial の可能性をもう一度見る
    if re.fullmatch(r"\d{4,6}", s):
        ts_from_digits = excel_serial_to_timestamp(s)
        if ts_from_digits is not None:
            return (float(ts_from_digits.month), float(ts_from_digits.day), True, "excel_serial")

    # 3/12, 3月12日, /12 など
    s2 = re.sub(r"[〔\[\(].*?[〕\]\)]", "", s)
    s2 = s2.replace("月", "/").replace("日", "").replace("　", "").strip()

    m = re.match(r"^(\d{1,2})\s*/\s*(\d{1,2})?$", s2)
    if m:
        mo = float(m.group(1)) if m.group(1) else np.nan
        da = float(m.group(2)) if m.group(2) else np.nan
        return (mo, da, True, "slash")

    m = re.match(r"^/(\d{1,2})$", s2)
    if m:
        return (np.nan, float(m.group(1)), True, "slash")

    return (np.nan, np.nan, False, "unparsed")

def validate_month_day(month: float, day: float) -> bool:
    if pd.isna(month) and pd.isna(day):
        return False
    if not pd.isna(month) and not (1 <= month <= 12):
        return False
    if not pd.isna(day) and not (1 <= day <= 31):
        return False
    if not pd.isna(month) and not pd.isna(day):
        mo_i, da_i = int(month), int(day)
        if mo_i in {4, 6, 9, 11} and da_i > 30:
            return False
        if mo_i == 2 and da_i > 29:
            return False
    return True


def split_attrs(x: object) -> list[object]:
    if pd.isna(x):
        return []
    s = str(x).replace("\u3000", " ").strip()
    if s == "" or s == "不明":
        return []
    s = re.sub(r"[\s,、・/;|＋+]+", "", s)
    tokens = ATTR_TOKEN_RE.findall(s)
    return tokens if tokens else [s]


def normalize_place_cell(x: object) -> tuple[object, object, bool, list[str], list[str]]:
    if pd.isna(x):
        return (np.nan, np.nan, False, [], [])
    s = str(x).replace("\u3000", " ").strip()
    if s == "" or s == "不明":
        return (np.nan, np.nan, False, [], [])
    s = s.replace("、", "・").replace("/", "・").replace(" ", "")
    s_base = re.sub(r"（.*?）", "", s)
    parts = [p for p in re.split(r"[・]+", s_base) if p]
    mapped: list[str] = []
    unknown: list[str] = []
    for p in parts:
        if p in CITY_TO_PREF:
            mapped.append(CITY_TO_PREF[p])
        elif p in PREFECTURES:
            mapped.append(p)
        elif p in FOREIGN_MARKERS:
            mapped.append("外国")
        else:
            unknown.append(p)
    if unknown:
        cls = np.nan
    else:
        if mapped and len(set(mapped)) == 1 and mapped[0] == "外国":
            cls = "外国"
        elif all(m in PREFECTURES for m in mapped):
            cls = mapped[0] if mapped else np.nan
        else:
            cls = np.nan
    multi = len(set(m for m in mapped if m != "外国")) > 1
    return (s, cls, multi, mapped, unknown)


def normalize_region_cell(x: object) -> tuple[object, object, bool, list[str], list[str]]:
    if pd.isna(x):
        return (np.nan, np.nan, False, [], [])
    s = str(x).replace("\u3000", " ").strip()
    if s == "" or s == "不明":
        return (np.nan, np.nan, False, [], [])
    s = s.replace("、", "・").replace("/", "・").replace(" ", "")
    parts = [p for p in re.split(r"[・]+", s) if p]
    mapped: list[str] = []
    unknown: list[str] = []
    for p in parts:
        if p in ALLOWED_REGIONS:
            mapped.append(p)
        else:
            unknown.append(p)
    return (s, mapped[0] if mapped else np.nan, len(mapped) > 1, mapped, unknown)


def md_escape(s: object) -> str:
    return str(s).replace("|", r"\|").replace("\n", " ")


def column_profile_table(df: pd.DataFrame, top_k: int = 5) -> str:
    n = len(df)
    lines = [
        "【カラム別プロファイル】",
        "",
        f"| カラム | dtype | 有効件数 | 欠損件数 | 欠損率 | ユニーク数 | 上位値（最大{top_k}） |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    def _to_key(x: object) -> object:
        if isinstance(x, list):
            return "|".join(map(str, x))
        return x

    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        na = int(s.isna().sum())
        non_na = int(n - na)
        na_pct = (na / n * 100) if n else 0.0
        s_non = s.dropna()
        has_list = s_non.apply(lambda x: isinstance(x, list)).any() if len(s_non) else False
        if has_list:
            s_key = s_non.apply(_to_key).astype(str)
            nunique = int(s_key.nunique(dropna=True))
        else:
            nunique = int(s.nunique(dropna=True))

        top_str = ""
        if dtype == "object" or nunique <= 50:
            s2 = s_non.apply(_to_key).astype(str) if has_list else s.dropna().astype(str)
            vc = s2.value_counts().head(top_k)
            if len(vc) > 0:
                top_str = "; ".join(f"{md_escape(k)} ({int(v)})" for k, v in vc.items())

        lines.append(
            f"| {md_escape(col)} | {md_escape(dtype)} | {non_na} | {na} | {na_pct:.1f}% | {nunique} | {top_str} |"
        )
    return "\n".join(lines)


def missingness_rank(df: pd.DataFrame, top_n: int = 20) -> str:
    n = len(df)
    na_counts = df.isna().sum().sort_values(ascending=False).head(top_n)
    lines = [f"【欠損値状況（上位{top_n}）】"]
    for col, cnt in na_counts.items():
        pct = (cnt / n * 100) if n else 0.0
        lines.append(f"  {col}: {int(cnt)}件 ({pct:.1f}%)")
    return "\n".join(lines)


def suspicious_text_values(df: pd.DataFrame, cols: list[str], top_n: int = 30) -> str:
    lines = ["【表記ゆれ・要確認候補】"]
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        has_zen = s[s.str.contains("\u3000", regex=False)]
        has_ws = s[s.str.match(r"^\s+|\s+$", na=False)]
        norm = (
            s.str.replace("\u3000", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.replace("・", "", regex=False)
        )
        groups: dict[str, set[str]] = {}
        for raw, key in zip(s.tolist(), norm.tolist()):
            groups.setdefault(key, set()).add(raw)
        variants = {k: v for k, v in groups.items() if len(v) >= 2}

        lines.append(f"- {col}:")
        if len(has_zen) > 0:
            lines.append(f"  ⚠ 全角スペース混入: {len(has_zen)}件（例: {has_zen.iloc[0]}）")
        if len(has_ws) > 0:
            lines.append(f"  ⚠ 前後空白あり: {len(has_ws)}件（例: '{has_ws.iloc[0]}'）")
        if len(variants) > 0:
            lines.append("  ⚠ ゆる正規化で同一扱いになりそうな揺れ（上位例）:")
            for _, vset in list(variants.items())[: min(top_n, 10)]:
                lines.append("    - " + " / ".join(sorted(vset)))
        if len(has_zen) == 0 and len(has_ws) == 0 and len(variants) == 0:
            lines.append("  特記事項なし")
    return "\n".join(lines)


def clean_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    for col in df.columns:
        df[col] = df[col].apply(strip_all)

    obj_cols = [col for col in df.columns if df[col].dtype == "object"]
    for col in obj_cols:
        df[col] = df[col].apply(normalize_missing_text)

    if "年代" in df.columns:
        era_parsed = df["年代"].apply(parse_era_year)
        df["年代_時代"] = [t[0] for t in era_parsed]
        df["年代_開始"] = [t[1] for t in era_parsed]
        df["年代_終了"] = [t[2] for t in era_parsed]
        df["年代_代表値"] = [t[3] for t in era_parsed]
        df["年代_西暦"] = [era_year_to_greg(e, y) for e, y in zip(df["年代_時代"], df["年代_代表値"])]

    if "月日" in df.columns:
        md_parsed = df["月日"].apply(parse_month_day)
        df["月"] = [t[0] for t in md_parsed]
        df["日"] = [t[1] for t in md_parsed]
        df["月日_成功"] = [t[2] for t in md_parsed]
        df["月日_形式"] = [t[3] for t in md_parsed]
        df["月日_妥当"] = [validate_month_day(month, day) for month, day in zip(df["月"], df["日"])]

    if "属性" in df.columns:
        df["属性_リスト"] = df["属性"].apply(split_attrs)
        df["属性_複数フラグ"] = df["属性_リスト"].apply(lambda xs: len(xs) > 1)
        df["属性_数"] = df["属性_リスト"].apply(len)
        max_attr = int(df["属性_数"].max()) if len(df) else 0
        for i in range(1, max_attr + 1):
            df[f"属性{i}"] = df["属性_リスト"].apply(lambda xs: xs[i - 1] if len(xs) >= i else np.nan)
        df["属性_組み合わせ"] = df["属性_リスト"].apply(lambda xs: "".join(xs) if xs else np.nan)

    for col in ["居住地", "出生地"]:
        if col in df.columns:
            norm = df[col].apply(normalize_place_cell)
            df[f"{col}_元"] = [t[0] for t in norm]
            df[f"{col}_主"] = [t[1] for t in norm]
            df[f"{col}_複数フラグ"] = [t[2] for t in norm]
            df[f"{col}_リスト"] = [t[3] for t in norm]
            df[f"{col}_未知"] = [t[4] for t in norm]

    if "出生地域" in df.columns:
        reg_norm = df["出生地域"].apply(normalize_region_cell)
        df["出生地域_元"] = [t[0] for t in reg_norm]
        df["出生地域_主"] = [t[1] for t in reg_norm]
        df["出生地域_複数フラグ"] = [t[2] for t in reg_norm]
        df["出生地域_リスト"] = [t[3] for t in reg_norm]
        df["出生地域_未知"] = [t[4] for t in reg_norm]

    return df


def build_display_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_disp = df.copy()
    for col in ["発信者", "居住地_主", "出生地_主", "出生地域_主"]:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].fillna("不明")
    return df_disp


def build_log(df_raw: pd.DataFrame, df: pd.DataFrame, input_path: Path, sheet_name: str, outdir: Path) -> str:
    n_total = len(df)
    valid_md = int(df["月日_妥当"].sum()) if "月日_妥当" in df.columns else 0
    era_counts = df["年代_時代"].value_counts(dropna=False).to_dict() if "年代_時代" in df.columns else {}
    n_unknown_year = int(df["年代_代表値"].isna().sum()) if "年代_代表値" in df.columns else 0
    if {"年代_開始", "年代_終了"}.issubset(df.columns):
        n_ranges = int(((~df["年代_開始"].isna()) & ((df["年代_終了"].isna()) | (df["年代_開始"] != df["年代_終了"]))).sum())
    else:
        n_ranges = 0

    log: list[str] = []
    log.append("品川弥二郎書簡データクリーニングログ")
    log.append("=" * 60)
    log.append(f"入力ファイル: {input_path}")
    log.append(f"入力シート: {sheet_name}")
    log.append(f"出力先: {outdir}")
    log.append(f"総レコード数: {n_total}")
    log.append(f"総カラム数: {df.shape[1]}")
    log.append("")
    log.append(f"原データのカラム: {list(df_raw.columns)}")
    log.append("")

    if era_counts:
        log.append("【年代データ】")
        for k, v in era_counts.items():
            log.append(f"  {k}: {int(v)}件")
        log.append(f"  年代不明（年が欠損）: {n_unknown_year}件")
        log.append(f"  年代レンジ表記（開始≠終了または終端欠損）: {n_ranges}件")
        log.append("")

    if "月日_妥当" in df.columns:
        pct = (valid_md / n_total * 100) if n_total else 0.0
        log.append("【月日データ】")
        log.append(f"  有効な月日データ: {valid_md}件")
        log.append(f"  処理成功率: {pct:.1f}% ({valid_md}/{n_total})")
        log.append("")

    log.append(column_profile_table(df, top_k=5))
    log.append("")
    log.append(missingness_rank(df, top_n=20))
    log.append("")
    log.append(suspicious_text_values(df, cols=["発信者", "居住地", "出生地", "出生地域", "年代", "月日", "属性"]))
    log.append("")
    return "\n".join(log)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv_bom = outdir / "shinagawa_letters_cleaned.csv"
    out_csv_utf8 = outdir / "shinagawa_letters_cleaned_utf8.csv"
    out_xlsx = outdir / "shinagawa_letters_cleaned.xlsx"
    out_log = outdir / "shinagawa_letters_cleaning_log.md"

    df_raw = pd.read_excel(input_path, sheet_name=args.sheet)
    df = clean_dataframe(df_raw)
    df_disp = build_display_dataframe(df)

    df.to_csv(out_csv_bom, index=False, encoding="utf-8-sig")
    df.to_csv(out_csv_utf8, index=False, encoding="utf-8")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned")
        df_disp.to_excel(writer, index=False, sheet_name="cleaned_disp")

    log_text = build_log(df_raw=df_raw, df=df, input_path=input_path, sheet_name=args.sheet, outdir=outdir)
    out_log.write_text(log_text, encoding="utf-8")

    print("done:")
    print(f"  {out_csv_bom}")
    print(f"  {out_csv_utf8}")
    print(f"  {out_xlsx}")
    print(f"  {out_log}")


if __name__ == "__main__":
    main()
