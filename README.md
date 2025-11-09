## Fashion-MNIST Scratch MLP

A from-scratch Multi-Layer Perceptron (MLP) classifier for Fashion-MNIST, implemented with low-level PyTorch primitives and a custom HOG-based feature extractor. The code is adapted for local execution with this repository’s directory structure.

### Repository Layout

- `src/lecture04_homework.py`: Main training/inference script
- `data/input/`: Place provided `.npy` files here
  - `x_train.npy`, `y_train.npy` (or `t_train.npy`), `x_test.npy`
- `data/output/`: Outputs (e.g., `submission.csv`) are saved here

### Requirements

- Python 3.9+
- PyTorch (CUDA optional)
- NumPy, Pandas, Pillow

Install (example):

```bash
pip install torch numpy pandas pillow
```

### How to Run

1) Put the dataset files in `data/input/`:

```
data/input/
  ├─ x_train.npy
  ├─ y_train.npy  # or t_train.npy
  └─ x_test.npy
```

2) Run the script from the repository root:

```bash
python3 src/lecture04_homework.py
```

3) Outputs:

- Predictions for `x_test` are saved to `data/output/submission.csv` with header `label` and index `id`.

### Training/Model Notes

- Custom HOG features (3 scales) + raw pixels → concatenated feature vector
- MLP classifier built using low-level components (`nn.Module`, `nn.Parameter`), custom LayerNorm/Dropout/GELU
- Training with AdamW, cosine LR schedule, warmup, EMA/SWA selection, label smoothing, mixup, augmentation schedule
- TTA-based validation blending with temperature scaling and optional prototype-based correction
- Optional MC-Dropout blending during inference

### Assignment Constraints (Included as provided)

目標値: Accuracy 85%

ルール:
- 訓練データは `x_train`, `t_train`，テストデータは `x_test` で与えられます。
- 予測ラベルは one_hot 表現ではなく 0~9 のクラスラベルで表してください。
- 下のセルで指定されている `x_train`, `t_train` 以外の学習データは使わないでください。
- PyTorch を利用して構いません。
- ただし，`torch.nn.Conv2d` のような高レベルの API は使用しないで下さい。具体的には，`nn.Parameter`, `nn.Module`, `nn.Sequential` 以外の nn 系の API です。使用した場合エラーになります。
- `torchvision` 等で既に実装されているモデルも使用しないで下さい。

提出方法:
1. テストデータ (`x_test`) に対する予測ラベルを csv 形式で保存し，Omnicampus の宿題タブから「第4回 ニューラルネットワークの最適化・正則化」を選択して提出してください。
2. それに対応する python のコードを「ファイル＞ダウンロード＞.py をダウンロード」から保存し，Omnicampus の宿題タブから「第4回 ニューラルネットワークの最適化・正則化 (code)」を選択して提出してください。python ファイル自体の提出ではなく，「提出内容」の部分にコード全体をコピー&ペーストしてください。
3. なお，採点は 1 で行い，2 はコードの確認用として利用します（成績優秀者はコード内容を公開させていただくかもしれません）。コードの内容を変更した場合は，1 と 2 の両方を提出し直してください。

参考: Fashion MNIST `https://github.com/zalandoresearch/fashion-mnist`

### Reproducibility Tips

- Seeds are set for Python, NumPy, and PyTorch in the script.
- On CPU, training may take longer; CUDA automatically enabled if available.

### Quick Validation

For a quicker dry run (to verify environment), reduce epochs by editing `n_epochs` near the top of the training configuration in `src/lecture04_homework.py` and re-run. Restore to default to aim for the target accuracy.


