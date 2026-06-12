from .attention_rollout import attention_rollout, gradient_attention_rollout
from .embeddings import extract_embeddings, plot_embedding, project_2d
from .gradcam import GradCAM
from .overlay import (
    compare_models_figure,
    defect_focus_score,
    denormalize,
    overlay_heatmap,
)
from .quantum_viz import (
    bloch_vectors,
    draw_circuit,
    expressivity_curve,
    plot_bloch,
)
from .saliency import model_saliency

__all__ = [
    "attention_rollout",
    "gradient_attention_rollout",
    "GradCAM",
    "model_saliency",
    "overlay_heatmap",
    "compare_models_figure",
    "defect_focus_score",
    "denormalize",
    "extract_embeddings",
    "project_2d",
    "plot_embedding",
    "draw_circuit",
    "bloch_vectors",
    "plot_bloch",
    "expressivity_curve",
]
