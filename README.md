# 判読性を考慮したフォント生成

文字識別モデル FontOCR を使ってデータセットのフォントの判読性を算出し、Attr2Font のモデルをもとにした生成モデルを用いて属性値 legible に判読性を反映することを目指した。

# 実行環境

主に FontOCR/Attr2Font モデル自体を扱う際は学科サーバを用い、モデルの出力を分析・加工する処理は Google Colaboratory を用いた。

## 学科サーバ

Attr2Font ディレクトリ・FontOCR ディレクトリ以下のコードは学科サーバにて実行した。処理内容は以下。

- FontOCR の学習
- FontOCR を用いた Attr2Font のデータセット用の legibility の算出
- Attr2Font の学習
- 評価用のフォント画像生成

Attr2Font ディレクトリ・FontOCR ディレクトリそれぞれのディレクトリ直下に singularity の def ファイルを用意している。

## Google Colaboratory

NoteBooks ディレクトリ以下のコードは Google Colaboratory にて実行した。処理内容は以下。

- FontOCR の分析・Legibility ファイルの作成

# 手順

## 使用するデータセット

使用するフォントのデータセットは 2 種類。

- AttrFont-ENG
  - Attr2Font で用いているデータセット。Attr2Font の GitHub の[Datasets のセクション](https://github.com/hologerry/Attr2Font?tab=readme-ov-file#datasets)でダウンロード方法が記載されている。
  - `./Attr2Font/data/explor_all`の下にデータセットの中身を配置する
  - `FontOCR/attr2font/images`に`explor_all/image`下の画像を配置する。
    - `./Attr2Font/data/explor_all/image`の画像へのシンボリックリンクでも動作するはず
- FOCR-datset
  - [Chen らの研究](https://arxiv.org/abs/1909.02072)で用いられたデータセット(LST-dataset)から文字以外のフォントを除外したもの。LST-dataset は[このページ](https://www.cs.rochester.edu/u/tchen45/font/font.html)の dataset.tar.gz のリンクからダウンロード可能。
  - 本研究で除外したフォント一覧は `./FontOCR/large_scale_tag/eliminate_fonts.txt` に記載。
  - `./FontOCR/large_scale_tag/images`にデータセットのディレクトリごと配置
  - `./FontOCR/large_scale_tag/images/dataset/データセット内のファイル・ディレクトリ`のようになれば OK

大まかには以下のようにして進める

1. AttrFont-ENG を用いて Attr2Font の学習
1. FOCR-dataset を用いて FontOCR の学習
1. FontOCR に AttrFont-ENG の画像を入力して出力の csv を得る
1. csv をもとに判読性の値を算出
1. source styles 選択用のファイル生成
1. AttrFont-ENG の属性値 legible の値を算出した判読性で置き換え
1. Attr2Font を置き換えたデータを用いてファインチューニング
1. legible 置き換え前/置き換え後それぞれのモデルで評価を行う

## FontOCR

### FOCR-datset のデータ分割

以下でデータセットから 10000 フォントを抽出して train/valid/test に分割、

```sh
python main.py --phase split_datasets --dataset_root data/large_scale_tag --image_dir_name images/dataset/fontimage
```

### 学習

```sh
python main.py --phase train --dataset_root data/large_scale_tag --image_dir_name images/dataset/fontimage --max_fonts_num -1
```

`--dataset_root`にデータセットのディレクトリ名を設定することで読み込むデータセットを変更する。

### 評価

```sh
python main.py --phase test --dataset_root data/large_scale_tag --image_dir_name images/dataset/fontimage --max_fonts_num -1 --load_weight_path '評価するFontOCRモデルの重みファイル（.pth）'
```

### AttrFont-ENG の画像を入力して softmax の値を出力

```sh
python main.py --phase forward --dataset_root data/attr2font --image_dir_name images/explor_all/image --is_attr2font_dataset --load_weight_path 'FontOCRモデルの重みファイル（.pth）' --max_fonts_num -1 --output_loss_csv_dir '出力先のパス'
```

## Attr2Font

### 学習

```sh
python main.py \
       --phase train \
       --batch_size 8 \
       --img_size 64 \
       --n_style 52 \
       --multi_gpu 'True' \
       --experiment_name '出力用ディレクトリのパス' \
       --attributes_file '属性値ファイルのパス' \
       --n_epochs 500 \
       --style_char_file 'source styles選択用ファイルのパス'
```

### ファインチューニング

```sh
python main.py \
       --phase train \
       --batch_size 8 \
       --img_size 64 \
       --multi_gpu 'True' \
       --experiment_name '出力用ディレクトリのパス' \
       --attributes_file '属性値ファイルのパス' \
       --style_char_file 'source styles選択用ファイルのパス' \
       --fine_tune_dir 'ファインチューニング元にする出力用ディレクトリ'/checkpoint \
       --fine_tune_init_epoch 'ファインチューンニング元にするepoch数' \
       --n_epochs 100
```

### 評価

```sh
experiment_name='評価するモデルの出力用ディレクトリのパス'
test_output_name='評価データの出力先'
n_epoch='評価するモデルのepoch数'
attributes_file='属性値ファイルのパス'
style_char_file='source styles選択用ファイルのパス' # Noneを設定すると先行研究と同じくランダムに選択
load_weight_path='評価にしようするFontOCRの重みファイル（.pth）'
test_num=1000

python test_legibility.py \
       --n_epoch ${n_epoch} \
       --experiment_name ${experiment_name} \
       --test_output_name ${test_output_name}\
       --attributes_file ${attributes_file} \
       --style_char_file ${style_char_file} \
       --load_weight_path ${load_weight_path} \
       --test_num ${test_num} \
       --fix_index 25532 # 入力するベースフォントの番号
```

## NoteBooks

| ファイル名              | 処理内容                                         |
| ----------------------- | ------------------------------------------------ |
| analyze_FOCR_csv.ipynb  | FontOCR の出力を分析、判読性の値の算出           |
| gen_sim_char_file.ipynb | 判読性の値を元に source styles 選択用の csv 生成 |
