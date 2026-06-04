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
  models/quantum_attention  HybridQuantumMultiHeadAttention (VQC-based QSA, attn capture)
  models/qvit.py            QViT (ViT w/ QSA blocks)
  models/qcnn.py            Quanvolutional CNN (exposes per-qubit feature maps)
  models/classical_vit.py   ViT-Tiny w/ attention-weight capture
  models/classical_cnn.py   3-layer CNN baseline (Grad-CAM hooks)
  training/trainer.py       Supervised + KD loops, quantum warm-up, best-ckpt
  explain/gradcam.py        Grad-CAM for CNN / QCNN
  explain/attention_rollout Attention rollout for ViT / QViT
  explain/overlay.py        Heatmap overlays + 4-model comparison + defect-focus
  explain/embeddings.py     t-SNE / PCA of learned feature space
  explain/quantum_viz.py    Circuit drawing, Bloch spheres, expressivity curve
  utils/metrics.py          Confusion / ROC-PR / Pareto frontier
  utils/flops.py            Classical FLOPs counter
  utils/plots.py            Acc-vs-params / Acc-vs-epochs / FLOPs-vs-gates plots
scripts/benchmark.py        End-to-end benchmark + full evidence suite
notebooks/QViT_TyreNet_Benchmark.ipynb   Narrated Colab walkthrough
configs/default.yaml        Reference hyper-parameter file
```

## Quantum design

* State preparation: `qml.AngleEmbedding(rotation="Y")` on reduced features,
  tanh-scaled to the maximally-expressive `[-pi, pi]` range.
* Variational ansatz: `BasicEntanglerLayers` (default) or `StronglyEntanglingLayers`,
  with optional **data re-uploading** (Perez-Salinas 2020) for higher expressivity.
* Output: per-wire `PauliZ` expectations.
* Device: `lightning.qubit` by default; switch to `lightning.gpu` on Colab.

## Evidence the benchmark produces

* **`pareto_frontier.png`** — accuracy vs. #params with the efficiency frontier
  (the parameter-efficiency proof).
* **`saliency_compare_*.png`** — original | CNN | ViT | QCNN | QViT heatmap overlays
  on real tyres, with a **defect-focus score** (does the model look at the flaw?).
* **`embeddings_*.png`** — t-SNE of each model's learned features.
* **`confusion_*.png`, `roc_pr.png`** — standard diagnostics.
* **`acc_vs_*.png`, `flops_vs_gates.png`** — convergence + compute footprint.

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
