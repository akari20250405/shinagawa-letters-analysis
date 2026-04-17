<<<<<<< HEAD
# やじてんデータ分析

品川弥二郎受信書簡データの前処理・基礎集計・可視化を段階的に実行するためのスクリプト群です。  
各スクリプトは、前段階で生成された出力を入力として利用することを想定しています。

---

## 実行環境

- OS: Windows
- Python: Anaconda 環境を想定
- VS Code（Python拡張）からの実行にも対応

---

## 使用ライブラリ

- pandas
- numpy
- matplotlib
- statsmodels
- scipy
- openpyxl

---

## ディレクトリ構成（想定）

```text
yajiten/
├─ scripts/
│  ├─ 01_cleaning.py
│  ├─ 02_phase1.py
│  ├─ 03_phase1_2_1.py
│  ├─ 04_phase1_2_2.py
│  ├─ 05_phase1_2_2_1_build_tables.py
│  ├─ 06_phase1_2_2_1_make_figures.py
│  ├─ 07_phase1_3.py
│  ├─ 08_phase2_5.py
│  ├─ 09_phase2_7_1.py
│  ├─ 10_phase2_7_2.py
│  ├─ 11_phase2_7_3.py
│  ├─ 12_phase2_8_1.py
│  ├─ 13_phase2_8_2.py
│  ├─ 14_phase3_9.py
│  ├─ 15_phase3_10.py
│  └─ 16_phase3_11_1.py
├─ data/
│  └─ <raw_input_file>   # 元データ（非公開、各自で配置）
├─ outputs/
│  ├─ cleaning/
│  │  ├─ shinagawa_letters_cleaned.csv
│  │  ├─ shinagawa_letters_cleaned_utf8.csv
│  │  ├─ shinagawa_letters_cleaned.xlsx
│  │  └─ shinagawa_letters_cleaning_log.md
│  ├─ phase1/
│  ├─ phase1_2_1/
│  ├─ phase1_2_2/
│  ├─ phase1_2_2_1/
│  │  └─ figures/
│  ├─ phase1_3/
│  ├─ phase2_5/
│  ├─ phase2_7_1/
│  ├─ phase2_7_2/
│  ├─ phase2_7_3/
│  ├─ phase2_8_1/
│  ├─ phase2_8_2/
│  ├─ phase3_9/
│  ├─ phase3_10/
│  └─ phase3_11_1/
└─ README.md
```

補足:
- `data/` 配下の元データは公開していません。上記はローカル実行時の配置例です。
- `outputs/` 配下には、前処理後CSVを含む再生成可能な生成物を保存します。
- 後続スクリプトの標準入力は、原則として `outputs/cleaning/shinagawa_letters_cleaned.csv` を想定しています。
---

## 出力ファイルの扱い

本リポジトリでは、**前処理後CSVを含む再生成可能な生成物を outputs/ 配下に保存**します。  
したがって、`shinagawa_letters_cleaned.csv` は入力資産ではなく、`01_cleaning.py` の出力ファイルとして扱います。

- 元データ（手元で管理するファイル）: `data/`
- 再生成可能な中間生成物・集計表・図表・ログ: `outputs/`

---

## 基本的な実行手順

### 1. プロジェクト直下へ移動

```powershell
cd C:\path\to\yajiten
```

補足:
- `C:\path\to\yajiten` は実際の保存先に置き換えてください。
- 以後の相対パスは、すべてプロジェクト直下基準です。

---

### 2. 前処理（cleaning）

元データから、後続スクリプトで利用する前処理済みCSVを作成します。

```powershell
python scripts/01_cleaning.py --input data/<raw_input_file> 
```

実行後、以下のファイルが `outputs/cleaning/` に生成されます。

- `shinagawa_letters_cleaned.csv`
- `shinagawa_letters_cleaned_utf8.csv`
- `shinagawa_letters_cleaned.xlsx`
- `shinagawa_letters_cleaning_log.md`

---

### 3. 後続分析の基本形

後続スクリプトは、原則として `outputs/cleaning/shinagawa_letters_cleaned.csv` を入力として使用します。

例:

```powershell
python scripts/02_phase1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

補足:
- `02_phase1.py` は実行したいスクリプト名に置き換えてください。
- スクリプトは、ファイル名冒頭の番号（01, 02, …）の昇順で実行することを想定しています。
- 各スクリプトの `--input` には、前段階で生成されたファイルを指定してください。
- 06_phase1_2_2_1_make_figures.py のみは、前段で生成した中間表ディレクトリを入力とするため --input ではなく --indir を使用します。

---

## スクリプト別の実行例と出力

### Phase1

#### 02_phase1.py
基礎統計ログを出力します。

```powershell
python scripts/02_phase1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase1/` に基礎統計ログファイルが生成されます。

---

#### 03_phase1_2_1.py
「発信書簡有無」列の分布確認を行います。

```powershell
python scripts/03_phase1_2_1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase1_2_1/` に分布確認ログファイルが生成されます。

---

#### 04_phase1_2_2.py
コード別の発信者一覧を出力します。

```powershell
python scripts/04_phase1_2_2.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase1_2_2/` に以下のファイルが生成されます。

- コード3の発信者一覧CSV
- コード4の発信者一覧CSV
- コード3+4合算の発信者一覧CSV
- ログファイル

---

#### 05_phase1_2_2_1_build_tables.py
図作成用の中間集計表を作成します。

```powershell
python scripts/05_phase1_2_2_1_build_tables.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase1_2_2_1/` に以下のファイルが生成されます。

- 属性別集計表CSV
- 発信者ベース集計表CSV
- 比率サマリCSV
- ログファイル

補足:
- 発信者単位の東京判定は、同一発信者に東京該当レコードが1件でもあれば東京扱いとします。
- 詳細はログファイルに記録されます。

---

#### 06_phase1_2_2_1_make_figures.py
05で生成した中間表を読み込み、図を作成します。

```powershell
python scripts/06_phase1_2_2_1_make_figures.py --indir outputs/phase1_2_2_1
```

実行後、`outputs/phase1_2_2_1/figures/` に図ファイルが生成されます。  
あわせて、`outputs/phase1_2_2_1/` にログファイルが生成されます。

---

#### 07_phase1_3.py
「受信当初面識なし」の送信者一覧と確認ログを出力します。

```powershell
python scripts/07_phase1_3.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase1_3/` に面識なし送信者ログファイルが生成されます。

---

### Phase2

#### 08_phase2_5.py
出生地ランキングと関連図を出力します。

```powershell
python scripts/08_phase2_5.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_5/` に以下のファイルが生成されます。

- 出生地ランキングの縦棒グラフ
- 出生地ランキングの横棒グラフ
- 出生地ランキングの円グラフ
- ランキングログファイル

---

#### 09_phase2_7_1.py
活動期ごとの件数推移を確認する図とログを出力します。

```powershell
python scripts/09_phase2_7_1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_7_1/` に以下のファイルが生成されます。

- 活動期推移図
- 活動期付与ログ

---

#### 10_phase2_7_2.py
活動期ごとの年正規化比率（rate per year）を表と図で出力します。

```powershell
python scripts/10_phase2_7_2.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_7_2/` に以下のファイルが生成されます。

- 活動期ごとの年正規化比率表CSV
- 活動期ごとの年正規化比率図
- ログファイル

---

#### 11_phase2_7_3.py
活動期ごとの分類可能率を表とログで出力します。

```powershell
python scripts/11_phase2_7_3.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_7_3/` に以下のファイルが生成されます。

- 活動期ごとの分類可能率表CSV
- ログファイル

---

#### 12_phase2_8_1.py
発信者分布の可視化とパレート分析を行います。

```powershell
python scripts/12_phase2_8_1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_8_1/` に以下のファイルが生成されます。

- 上位発信者棒グラフ
- 発信者分布パレート図
- 上位発信者表CSV
- 分布表CSV
- ログファイル

---

#### 13_phase2_8_2.py
上位発信者の支配的属性と仮説ヒット一覧を出力します。

```powershell
python scripts/13_phase2_8_2.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase2_8_2/` に以下のファイルが生成されます。

- 上位発信者の支配的属性表CSV
- 仮説ヒット一覧CSV
- ログファイル

---

### Phase3

#### 14_phase3_9.py
出生地域 × 活動期クロス集計と関連図を出力します。

```powershell
python scripts/14_phase3_9.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase3_9/` に以下のファイルが生成されます。

- raw クロス集計グラフ・CSV
- 年正規化グラフ・CSV
- 構成比グラフ・CSV
- 標準化残差図・CSV
- Top送信者シェア・表CSV
- ヒートマップ図
- ログファイル

---

#### 15_phase3_10.py
活動期ごとの居住地分布の entropy 指標と関連図を出力します。

```powershell
python scripts/15_phase3_10.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase3_10/` に以下のファイルが生成されます。

- 居住地 entropy・サマリCSV
- Top居住地一覧CSV
- 東京集中度図
- entropy関連図
- ログファイル

---

#### 16_phase3_11_1.py
属性 × 活動期 × 出生地域に関する集計・MCA結果を出力します。

```powershell
python scripts/16_phase3_11_1.py --input outputs/cleaning/shinagawa_letters_cleaned.csv
```

実行後、`outputs/phase3_11_1/` に以下のファイルが生成されます。

- 属性別棒グラフ
- 属性 × 活動期クロス集計図・CSV
- 構成比グラフ・CSV
- 標準化残差表・CSV
- MCAカテゴリ座標・CSV
- ログファイル

---

## 注意事項

- 入力列名は、各スクリプトで想定している列名と一致している必要があります。
- 多くのスクリプトは `utf-8-sig` を前提にCSVを読み書きします。
- 日本語フォントは環境依存です。図の文字化けが出る場合は、使用可能な日本語フォントを確認してください。
- `outputs/` 配下の生成物は、再実行により再作成できることを前提としています。

---

## データの公開について

本リポジトリには、分析対象となる元データは含まれていません。  
これらのデータは、著作権・利用条件等の都合により公開していません。

README 内の `data/` 配下のファイル配置は、あくまでローカル環境での実行例を示したものです。  
本リポジトリを利用する場合は、各自で適切な入力データを用意し、対応するパスに配置してください。

公開しているのは、再現可能な範囲のスクリプト、出力仕様、および再生成可能な生成物の管理方法です。

---

## ライセンス/License
このデータセットは、クリエイティブ・コモンズ表示4.0国際ライセンスの下でライセンスされています。

This dataset is licensed under a Creative Commons Attribution 4.0 International License.

---

## 引用/Citation
このデータセットを使用する場合は、以下のように引用してください。
池田さなえ(2026) 「品川弥二郎受信書簡メタデータの統計的分析」(Version v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.19362502

If you use this dataset, please cite it as below:
Sanae IKEDA. (2026).  shinagawa-letters-analysis(Version v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.19362502