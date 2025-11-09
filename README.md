# 東大 深層学習講座 コンペティション

## コンペティション結果

- **最終順位**: **3位** / 1439人中
- **LBスコア**: **0.9485**

## 概要
MNISTのファッション版 (Fashion MNIST，クラス数10) を多層パーセプトロンによって分類．

Fashion MNISTの詳細については以下のリンクを参考にしてください．
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## ルール

- 訓練データは`x_train`，`t_train`，テストデータは`x_test`で与えられます．
- 予測ラベルは one_hot表現ではなく0~9のクラスラベルで表してください．
- 下のセルで指定されている`x_train`，`t_train`以外の学習データは使わないでください．
- PyTorchを利用して構いません．
- ただし，`torch.nn.Conv2d`のような高レベルのAPIは使用しないで下さい．具体的には，`nn.Parameter`, `nn.Module`, `nn.Sequential`以外のnn系のAPIです．使用した場合エラーになります．
- `torchvision`等で既に実装されているモデルも使用しないで下さい．

## アプローチ

- データ前処理/分割
  - `x_train.npy`, `t_train.npy`/`y_train.npy`, `x_test.npy` を読み込み、28×28の `float32`・[0,1] に正規化
  - 訓練データ末尾10%を検証に分割（固定シード）
  - `DataLoader` は学習時シャッフル、検証/テストは順序固定

- 画像拡張（スケジュール付き）
  - 回転(±8°)、平行移動(±1.5px)、拡大縮小(±6%)、せん断(±6°)、水平反転(確率0.25)、Cutout(確率0.5)
  - エポック10→26で強度を線形フェード、エポック32以降は拡張OFF

- 特徴抽出（HOG + 生画素、8296次元）
  - Sobel勾配からHOGを3スケールで集約: cell=4,bins=8(1152), cell=3,bins=9(2304), cell=2,bins=6(4056)
  - 生画素784を連結し、学習データで推定した平均/分散で標準化

- モデル（自作MLP）
  - 全結合のみで 8296→3072→1536→768→10
  - 活性化: GELU（自前実装）、正規化: 1D LayerNorm（自前実装）
  - Dropoutを訓練進行で増加: 初期(0.10,0.12,0.15) → 終盤(0.32,0.35,0.38)
  - `nn.*`制約に対応（高レベル層は未使用、検査関数で強制）

- 学習と正則化
  - Optimizer: AdamW（weight_decay=6e-4）
  - 学習率: 6エポックのウォームアップ後、Cosine decay（基準LR=2.5e-3）
  - 損失: ラベルスムージング付きCE（0.12 → 0.04に線形減少）
  - Mixup: α=0.18（エポック28まで有効）
  - 勾配クリップ: グローバルノルム5.0、AMP + GradScaler使用

- EMA/SWA と検証選択
  - EMA: 0.9992→0.9996 に段階的強化
  - SWA: 学習後半（拡張停止以降）で更新
  - 各エポックで last/EMA/SWA を評価し最良重みを保持

- TTA と温度スケーリング・重み学習
  - 変換候補: 恒等、±5°/±7°、(±1,±1) 平行移動、水平反転
  - 本設定では恒等と平行移動2種を採用（keep=[0,5,6]）
  - 恒等ロジットで温度TをLBFGSで学習し、クラス条件の重み行列Wを最適化してTTAロジットを加重合成
  - 予測信頼度に応じて恒等をブーストするIdentityアンカーを適用

- プロトタイプ補正（任意）
  - 学習特徴のクラス中心/分散を用い、プロト距離＋ガウス尤度由来のボーナスをロジットへ加算（低信頼時のみ）

- MC Dropout（自動）
  - 検証で複数スケールを探索し、精度向上が閾値超なら有効化。最終予測は base と MC の平均

- 推論/保存
  - 上記を組み合わせたロジットからクラスを決定し、`data/output/submission.csv` に `label` 列・`id` インデックスで保存

## 使用技術 (Tech Stack)

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