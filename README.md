# QViT Benchmark for Tyre Defect Detection

Comparative study of four vision architectures for Apollo-Tyres-style
industrial quality control on the **TyreNet** dataset (Mendeley, ~1,700
images of good vs. defective tyres):

| Model | Description |
|-------|-------------|
| `ClassicalCNN` | 3-layer convolutional baseline |
| `ViT-Tiny`     | Patch-16, depth-12, dim-192 with classical Multi-Head Attention |
| `QCNN`         | Hybrid "Quanvolutional" front-end (Henderson et al., 2019) + small CNN classifier |
| `QViT`         | ViT where each Multi-Head Attention block is replaced by a **Hybrid Quantum Self-Attention** head (Cherrat et al., 2024; Boucher et al., 2025) |

The **hypothesis** under test: QSA offers $O(n)$ parameter scaling and a
stronger inductive bias for global feature extraction, so the QViT should
hit competitive accuracy with **far fewer trainable parameters** than the
classical ViT-Tiny.

## Repo layout

```
src/
  data/tyrenet_dataset.py   Dual-resolution Dataset (224x224 + per-patch quantum features)
  models/quantum_attention  HybridQuantumMultiHeadAttention (VQC-based QSA)
  models/qvit.py            QViT (ViT w/ QSA blocks)
  models/qcnn.py            Quanvolutional CNN
  models/classical_vit.py   ViT-Tiny reference
  models/classical_cnn.py   3-layer CNN baseline
  training/trainer.py       Supervised + Knowledge-Distillation loops
  utils/flops.py            Classical FLOPs counter
  utils/plots.py            Acc-vs-params / Acc-vs-epochs / FLOPs-vs-gates plots
scripts/benchmark.py        End-to-end benchmark entrypoint
configs/default.yaml        Reference hyper-parameter file
```

## Quantum design

* State preparation: `qml.AngleEmbedding(rotation="Y")` on reduced features
  (tanh-squashed to keep them in the AngleEmbedding-friendly range).
* Variational ansatz: `qml.BasicEntanglerLayers` (RX layers + ring of CNOTs).
* Output: per-wire `PauliZ` expectations.
* Device: `lightning.qubit` by default; switch to `lightning.gpu` on Colab.

## Quickstart (Google Colab)

```python
!pip install -q pennylane "pennylane-lightning[gpu]" torch torchvision scikit-learn matplotlib
!git clone <this repo> && cd QViT_Experiment
# Layout your dataset as: ./data/tyrenet/good/*.jpg and ./data/tyrenet/defective/*.jpg
!python scripts/benchmark.py \
    --data_root ./data/tyrenet \
    --epochs 10 --batch_size 16 \
    --n_qubits 4 --n_layers 2 --n_heads 2 \
    --qdevice lightning.gpu \
    --use_kd
```

Outputs:
* `results/plots/acc_vs_params.png`
* `results/plots/acc_vs_epochs.png`
* `results/plots/flops_vs_gates.png`
* `results/logs/benchmark.json` (full per-epoch history)

## Knowledge distillation

When `--use_kd` is set, the QViT student is trained with the standard
Hinton-style soft-label loss against the classical ViT teacher:

$$L = \alpha \cdot \mathrm{CE}(s, y) + (1-\alpha) \cdot T^2 \cdot \mathrm{KL}(\sigma(s/T) \Vert \sigma(t/T))$$

with $T=4$, $\alpha=0.5$ by default.

## References

* Cherrat, El Amine et al. *Quantum Vision Transformers.* (2024).
* Boucher, P. et al. *Inductive bias of quantum attention.* (2025).
* Henderson, M. et al. *Quanvolutional Neural Networks.* arXiv:1904.04767 (2019).
* Hinton, G. et al. *Distilling the Knowledge in a Neural Network.* (2015).
