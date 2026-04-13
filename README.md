
Main infos about the project  : 
The project name is: "clip-latent-reading"
The package name is: "multimodal-interpretability-pilot"
You are logged in as: shannon

CLIP Latent Reading: Light Across Discursive Descriptions

This repository builds on the FabLight research project, which investigates the representation of light in 18th-century painting at the intersection of art history, history of science, and technology.

The project explores how technical developments in lighting—particularly the evolution of lamps—expanded the possibilities for painters to represent light, both as a visual device (e.g., enhancing nudes, structuring composition, staging scenes) and as an object of knowledge in its own right. In certain artistic contexts, lamps themselves appear as subjects of attention, reflecting a broader epistemic shift in how light is observed, studied, and represented.

Light thus operates as a multidisciplinary and multilingual concept, circulating across artistic, technical, and scientific discourses. This raises a central question:
How do multimodal AI models handle such conceptual complexity?

Using CLIP (Contrastive Language–Image Pretraining), this project investigates how images and textual descriptions related to light are organized in a shared embedding space. It compares different discursive registers—art-historical, technical, and poetic—to examine how the model aligns visual and linguistic representations.

Rather than treating the model as a neutral retrieval tool, this repository approaches CLIP as a cultural structure, whose latent space can be read and interpreted. The project therefore proposes a small-scale experiment in “latent reading”, asking how disciplinary distinctions are preserved, transformed, or flattened within a multimodal AI system.

Packages  : 
torch → runs the model
transformers → loads CLIP
pillow → loads images
pandas → handles prompts.csv
scikit-learn → PCA + cosine similarity
matplotlib → plots

## Corpus

### Images (4 paintings — prototype corpus)

| ID | Title | Artist | Date | Light source |
|----|-------|--------|------|-------------|
| IMG1 | A young woman holding a lamp | Philip Dawe (after Foldsone) | 1769–1784 | Candle |
| IMG2 | Old woman examining a coin by a lantern | Gerrit van Honthorst | c. 1623 | Lantern |
| IMG3 | Boy Reading at Artificial Light | Jens Juel | 1763–1764 | Candle |
| IMG4 | The Elegant Reader | Georg Friedrich Kersting | 1812 | Argand lamp |

All images show figures by artificial light in interior settings — a controlled scene type that allows isolation of the variable of light source.

### Text prompts (12 prompts, 3 categories × 4 prompts)

| Category | IDs | Description |
|----------|-----|-------------|
| `art_historical` | t01–t04 | Art-historical descriptions of candlelit interiors, chiaroscuro, Enlightenment iconography |
| `technical` | t05–t08 | Technical descriptions of lamp mechanisms, photometry, combustion |
| `poetic` | t09–t12 | Phenomenological and metaphorical descriptions of light as atmosphere and threshold |

---

## Methods

- **Model**: `openai/clip-vit-base-patch32` (Hugging Face `transformers`)
- **Embeddings**: L2-normalized image and text embeddings (512-dim)
- **Similarity**: Cosine similarity matrix (image × text, text × text)
- **Dimensionality reduction**: PCA (joint image+text projection); optional UMAP
- **Semantic axis**: mean(art_historical) − mean(technical) direction, projected onto image embeddings

---

## Repository structure

```
clip-latent-reading/
├── data/
│   ├── images/                   # Image files (rename to match metadata)
│   ├── prompts.csv               # 12 text prompts with category labels
│   └── image_metadata.csv        # Image metadata (artist, date, medium, etc.)
├── notebooks/
│   └── 01_clip_latent_reading.ipynb   # Main analysis notebook
├── results/
│   ├── figures/                  # Saved plots (PCA, heatmaps, semantic axis)
│   └── embeddings/               # Saved .npy embeddings and summary CSV
├── src/
│   └── utils.py                  # Reusable functions (encoding, similarity, visualization)
├── README.md
└── requirements.txt
```

---

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add images
# Place image files in data/images/
# Rename to match filenames in data/image_metadata.csv

# Launch notebook
jupyter notebook notebooks/01_clip_latent_reading.ipynb
```

---

## First observations

*(To be filled in after running the notebook — Section 8)*

---

## Theoretical positioning

This experiment treats CLIP not as a benchmark tool but as an object of cultural analysis. The model was trained on large-scale internet image-text pairs, making its embedding space a kind of sediment of the way visual and linguistic representations have been co-organized online.

Reading the latent space is therefore also reading a cultural history of how knowledge domains are linked — or severed — in contemporary representational practice. If art-historical and technical descriptions of light cluster separately, this tells us something about how CLIP has learned to distinguish (or not) between modes of attention to visual phenomena.

---

## Next steps

- [ ] Expand image corpus (add scientific illustrations, lamp diagrams)
- [ ] Add French-language prompts (multilingual CLIP comparison)
- [ ] Controlled prompt variation study (candle / lamp / lantern / torch)
- [ ] Three-way semantic ternary plot (art / technical / poetic)
- [ ] Attention map probing (GradCAM) per discourse category

---

## License

Research prototype — FabLight project. Not for public release.