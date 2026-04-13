"""
clip_latent_reading — utility functions
FabLight / CLIP Latent Reading project
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from pathlib import Path


# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = ROOT / "data"
IMAGES_DIR   = DATA_DIR / "images"
RESULTS_DIR  = ROOT / "results"
FIGURES_DIR  = RESULTS_DIR / "figures"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"

for d in [FIGURES_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# CATEGORY COLORS  (art_historical / technical / poetic)
# ──────────────────────────────────────────────

CATEGORY_COLORS = {
    "art_historical": "#C8842A",   # warm amber — painting tradition
    "technical":      "#3A6EA5",   # cool blue — scientific/technical
    "poetic":         "#7B5EA7",   # soft purple — poetic / phenomenological
}

CATEGORY_MARKERS = {
    "art_historical": "o",
    "technical":      "s",
    "poetic":         "^",
}

IMAGE_COLOR = "#2E7D52"   # dark green for image points


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────

def load_prompts(csv_path=None):
    """Load text prompts CSV. Returns a DataFrame."""
    if csv_path is None:
        csv_path = DATA_DIR / "prompts.csv"
    return pd.read_csv(csv_path)


def load_image_metadata(csv_path=None):
    """Load image metadata CSV. Returns a DataFrame."""
    if csv_path is None:
        csv_path = DATA_DIR / "image_metadata.csv"
    return pd.read_csv(csv_path)


def load_images_pil(images_dir=None, metadata_df=None):
    """
    Load images as PIL objects.
    Returns list of (image_id, PIL.Image) tuples.
    Only loads images whose filename exists in images_dir.
    """
    if images_dir is None:
        images_dir = IMAGES_DIR
    images_dir = Path(images_dir)

    if metadata_df is None:
        metadata_df = load_image_metadata()

    loaded = []
    missing = []
    for _, row in metadata_df.iterrows():
        img_path = images_dir / row["filename"]
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            loaded.append((row["image_id"], img))
        else:
            missing.append(row["filename"])

    if missing:
        print(f"[warn] Missing image files: {missing}")
    print(f"[info] Loaded {len(loaded)} images from {images_dir}")
    return loaded


# ──────────────────────────────────────────────
# CLIP ENCODING
# ──────────────────────────────────────────────

def encode_images(clip_model, clip_processor, pil_images, device="cpu", batch_size=4):
    """
    Encode a list of PIL images with CLIP.

    Parameters
    ----------
    pil_images : list of PIL.Image
    Returns : np.ndarray of shape (N, d), L2-normalized
    """
    import torch
    all_embeddings = []

    for i in range(0, len(pil_images), batch_size):
        batch = pil_images[i:i+batch_size]
        inputs = clip_processor(images=batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)   # L2 normalize
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)


def encode_texts(clip_model, clip_processor, texts, device="cpu", batch_size=8):
    """
    Encode a list of text strings with CLIP.

    Parameters
    ----------
    texts : list of str
    Returns : np.ndarray of shape (N, d), L2-normalized
    """
    import torch
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            feats = clip_model.get_text_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)


# ──────────────────────────────────────────────
# SIMILARITY
# ──────────────────────────────────────────────

def cosine_similarity_matrix(a, b):
    """
    Compute cosine similarity between two sets of L2-normalized embeddings.
    a: (N, d), b: (M, d)  → returns (N, M)
    If embeddings are already L2-normalized, this is just the dot product.
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def top_k_matches(sim_matrix, query_labels, target_labels, k=3):
    """
    For each row (query), return the top-k target labels by similarity.
    Returns a DataFrame with columns: query, rank, target, similarity.
    """
    rows = []
    for i, q_label in enumerate(query_labels):
        scores = sim_matrix[i]
        top_idx = np.argsort(scores)[::-1][:k]
        for rank, j in enumerate(top_idx):
            rows.append({
                "query":      q_label,
                "rank":       rank + 1,
                "target":     target_labels[j],
                "similarity": float(scores[j]),
            })
    return pd.DataFrame(rows)


def category_mean_embeddings(text_embeddings, categories):
    """
    Compute per-category mean embedding.
    categories: list of str, len == len(text_embeddings)
    Returns dict {category: mean_embedding (d,)}
    """
    categories = np.array(categories)
    unique_cats = np.unique(categories)
    means = {}
    for cat in unique_cats:
        mask = categories == cat
        means[cat] = text_embeddings[mask].mean(axis=0)
    return means


# ──────────────────────────────────────────────
# SEMANTIC DIRECTION  (art_historical − technical axis)
# ──────────────────────────────────────────────

def compute_semantic_axis(cat_means, cat_a="art_historical", cat_b="technical"):
    """
    Returns the unit vector from cat_b mean to cat_a mean.
    This direction encodes the 'art_historical vs technical' axis.
    """
    direction = cat_means[cat_a] - cat_means[cat_b]
    return direction / np.linalg.norm(direction)


def project_onto_axis(embeddings, axis):
    """
    Project embeddings onto a unit direction vector.
    Returns scalar scores of shape (N,).
    """
    return embeddings @ axis


# ──────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ──────────────────────────────────────────────

def run_pca(embeddings, n_components=2):
    """PCA on embeddings. Returns (coords, explained_variance_ratio)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(embeddings)
    return coords, pca.explained_variance_ratio_


def run_umap(embeddings, n_components=2, n_neighbors=5, min_dist=0.3, metric="cosine"):
    """UMAP on embeddings. Returns 2D coords."""
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ──────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────

def plot_text_projection(
    coords_2d,
    prompts_df,
    method_name="PCA",
    explained_var=None,
    save_path=None,
    title=None,
):
    """
    Scatter plot of text embeddings in 2D, colored by discourse category.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    categories = prompts_df["category"].values
    texts_short = [t[:55] + "…" if len(t) > 55 else t for t in prompts_df["text"].values]

    for i, (x, y) in enumerate(coords_2d):
        cat = categories[i]
        color = CATEGORY_COLORS.get(cat, "#888888")
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.scatter(x, y, color=color, marker=marker, s=110, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(
            prompts_df["id"].iloc[i],
            (x, y),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color="#444444",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=cat.replace("_", " ").title())
        for cat, c in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.85,
              fontsize=9, edgecolor="#cccccc")

    # Axis labels
    if explained_var is not None and method_name == "PCA":
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)", fontsize=10)
    else:
        ax.set_xlabel(f"{method_name} dim 1", fontsize=10)
        ax.set_ylabel(f"{method_name} dim 2", fontsize=10)

    default_title = f"Text embeddings — {method_name} projection\nCLIP (openai/clip-vit-base-patch32)"
    ax.set_title(title or default_title, fontsize=12, pad=14, color="#222222")
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig, ax


def plot_combined_projection(
    text_coords,
    image_coords,
    prompts_df,
    image_ids,
    method_name="PCA",
    explained_var=None,
    save_path=None,
):
    """
    Combined scatter of both text and image embeddings in the same 2D space.
    Requires joint PCA/UMAP run on concatenated embeddings.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    n_text = len(text_coords)
    categories = prompts_df["category"].values

    # Text points
    for i, (x, y) in enumerate(text_coords):
        cat = categories[i]
        color = CATEGORY_COLORS.get(cat, "#888")
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.scatter(x, y, color=color, marker=marker, s=90, zorder=3,
                   edgecolors="white", linewidths=0.8, alpha=0.85)
        ax.annotate(prompts_df["id"].iloc[i], (x, y),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color="#555555")

    # Image points (stars)
    for i, (x, y) in enumerate(image_coords):
        ax.scatter(x, y, color=IMAGE_COLOR, marker="*", s=220, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(image_ids[i], (x, y),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=IMAGE_COLOR, fontweight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=cat.replace("_", " ").title())
        for cat, c in CATEGORY_COLORS.items()
    ]
    legend_patches.append(
        mpatches.Patch(color=IMAGE_COLOR, label="Image (painting)")
    )
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.85,
              fontsize=9, edgecolor="#cccccc")

    if explained_var is not None and method_name == "PCA":
        ax.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}% var)", fontsize=10)
        ax.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}% var)", fontsize=10)
    else:
        ax.set_xlabel(f"{method_name} dim 1", fontsize=10)
        ax.set_ylabel(f"{method_name} dim 2", fontsize=10)

    ax.set_title(
        f"Joint image + text embeddings — {method_name}\nCLIP (openai/clip-vit-base-patch32)",
        fontsize=12, pad=14, color="#222222",
    )
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", alpha=0.35, color="#aaaaaa")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig, ax


def plot_similarity_heatmap(sim_matrix, row_labels, col_labels, save_path=None, title=None):
    """
    Heatmap of a similarity matrix (e.g. image × text).
    """
    import matplotlib.colors as mcolors
    fig, ax = plt.subplots(figsize=(max(8, len(col_labels)*0.7), max(5, len(row_labels)*0.6)))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    im = ax.imshow(sim_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine similarity")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = sim_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.6 else "white")

    ax.set_title(title or "Image × Text cosine similarity (CLIP)", fontsize=12, pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig, ax


def plot_semantic_axis(
    image_scores,
    image_ids,
    text_scores=None,
    prompts_df=None,
    save_path=None,
):
    """
    1D projection onto the art_historical − technical semantic axis.
    Images as colored dots, texts as small labeled ticks.
    """
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor("#F7F5F0")
    ax.set_facecolor("#F7F5F0")

    # Draw axis
    ax.axhline(0, color="#cccccc", linewidth=1.5)
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.5)

    # Images
    for i, score in enumerate(image_scores):
        ax.scatter(score, 0, color=IMAGE_COLOR, marker="*", s=260, zorder=4,
                   edgecolors="white", linewidths=0.8)
        ax.text(score, 0.08, image_ids[i], ha="center", fontsize=8,
                color=IMAGE_COLOR, fontweight="bold")

    # Text prompts
    if text_scores is not None and prompts_df is not None:
        categories = prompts_df["category"].values
        for i, score in enumerate(text_scores):
            cat = categories[i]
            color = CATEGORY_COLORS.get(cat, "#888")
            marker = CATEGORY_MARKERS.get(cat, "o")
            ax.scatter(score, -0.08, color=color, marker=marker, s=70, zorder=3,
                       edgecolors="white", linewidths=0.6)
            ax.text(score, -0.18, prompts_df["id"].iloc[i], ha="center",
                    fontsize=6.5, color=color)

    # Labels at extremes
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < -0.1 else -0.2, 0,
            "← Technical", ha="left", va="center", fontsize=9, color=CATEGORY_COLORS["technical"])
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0.1 else 0.2, 0,
            "Art-historical →", ha="right", va="center", fontsize=9, color=CATEGORY_COLORS["art_historical"])

    legend_patches = [
        mpatches.Patch(color=c, label=cat.replace("_", " ").title())
        for cat, c in CATEGORY_COLORS.items()
    ]
    legend_patches.append(mpatches.Patch(color=IMAGE_COLOR, label="Image"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.85)

    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("Projection score on semantic axis (art_historical − technical)", fontsize=10)
    ax.set_title("Semantic axis projection — where do images fall?", fontsize=12, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    return fig, ax


# ──────────────────────────────────────────────
# SAVING
# ──────────────────────────────────────────────

def save_embeddings(embeddings_dict, embeddings_dir=None):
    """
    Save dict of {name: np.ndarray} to .npy files.
    embeddings_dict example: {"image": img_emb, "text": txt_emb}
    """
    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in embeddings_dict.items():
        path = embeddings_dir / f"{name}_embeddings.npy"
        np.save(path, arr)
        print(f"[saved] {path}  shape={arr.shape}")


def load_embeddings(names, embeddings_dir=None):
    """Load .npy embeddings by name list. Returns dict."""
    if embeddings_dir is None:
        embeddings_dir = EMBEDDINGS_DIR
    out = {}
    for name in names:
        path = Path(embeddings_dir) / f"{name}_embeddings.npy"
        out[name] = np.load(path)
        print(f"[loaded] {path}  shape={out[name].shape}")
    return out
