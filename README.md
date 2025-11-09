# 東大 深層学習講座 コンペティション

## コンペティション結果

- **最終順位**: **3位** / 1439人中
- **LBスコア**: **0.9485**

## 概要

今Lessonで学んだことを元に，MNISTのファッション版 (Fashion MNIST，クラス数10) を多層パーセプトロンによって分類してみましょう．

Fashion MNISTの詳細については以下のリンクを参考にしてください．

Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## アプローチと工夫した点

ベースラインモデル（例：ResNet50）に対し、以下の点を工夫することで精度向上を図りました。

- **1. [TBD: 工夫した点 1]**
  - 例：データ拡張（Data Augmentation）の強化
  - 例：Mixup, CutMix, RandAugment などを試し、最も効果的だった〇〇を採用しました。

- **2. [TBD: 工夫した点 2]**
  - 例：モデルアーキテクチャの変更
  - 例：EfficientNetB3 をベースに、最終層に〇〇を追加しました。

- **3. [TBD: 工夫した点 3]**
  - 例：ハイパーパラメータチューニング
  - 例：学習率のスケジューリング（Cosine Annealing）や、最適なOptimizer（AdamW）の選定を行いました。

- **4. [TBD: 工夫した点 4]**
  - 例：アンサンブル手法
  - 例：最終的に、特性の異なる3つのモデル（例：ResNet, EfficientNet）の予測結果を加重平均しました。

## 🛠️ 使用技術 (Tech Stack)

- Python [TBD: 3.x]
- PyTorch [TBD: 1.x]
- Pandas / NumPy
- Matplotlib / Seaborn
- Scikit-learn
- [TBD: もし使っていれば: Optuna, Albumentations など]

## 実行方法

**1. リポジトリのクローン**
```bash
git clone [https://github.com/](https://github.com/)[あなたのGitHubユーザー名]/[リポジトリ名].git
cd [リポジトリ名]