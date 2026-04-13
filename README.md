
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
