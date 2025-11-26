# GSAE: Graph-Regularized Sparse Autoencoders for Runtime LLM Safety Steering

> **Distributed Safety Features via Laplacian Smoothness & Dual-Gated Intervention.**

**GSAE** moves beyond standard Sparse Autoencoders (SAEs) by enforcing a **graph Laplacian smoothness prior** on the neuron co-activation graph. Rather than recovering noisy, single-axis directions, GSAE recovers **distributed, topologically consistent safety features**. 

At inference time, this allows for **Laplacian-Weighted Steering** controlled by a dual-stage mechanism: a calibrated Random Forest gate for initial prompts and a Hysteresis-based detector for continuation, ensuring minimal impact on benign utility.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
conda env create -f environment.yaml
conda activate gsae
````

### 2\. Usage Pipeline

GSAE operates in three phases: **Extraction**, **Training**, and **Steering**.

```python
from gsae import (
    EnhancedLLMExtractor, GraphSAE, DirectVectorBuilder, 
    MLOnlyHarmfulDetector, LaplacianWeightedDirectInterventionSteering,
    build_coactivation_graph_laplacian
)

# 1. Extract Features & Build Graph
extractor = EnhancedLLMExtractor(model_name="meta-llama/Meta-Llama-3-8B")
# ... (load your dataset X_safe, X_harm) ...
features = extractor.extract_hidden_states_multilayer(texts_list)

# Build the Laplacian from neuron co-activation
laplacians = build_coactivation_graph_laplacian(features, y_labels, threshold=0.6)

# 2. Train GraphSAE
# We enforce L = I - D^(-1/2) A D^(-1/2) to penalize high-frequency noise
gsae = GraphSAE(input_dim=4096, hidden_dim=16384, graph_laplacian=laplacians[10])
gsae.fit(X_train, y_train)

# 3. Build Steering Vectors
builder = DirectVectorBuilder(gsae)
# Vectors are validated via causal ablation
builder.build(X_safe, X_harm, X_all, y_all) 

# 4. Runtime Steering
# Initialize detectors and the steering pipeline
harm_detector = MLOnlyHarmfulDetector()
harm_detector.setup_rf_from_gsae_latents(Z_latents, y_train)

steer_pipe = LaplacianWeightedDirectInterventionSteering(
    llm_extractor=extractor,
    harm_detector=harm_detector,
    gsae_by_layer={10: gsae},
    builders_by_layer={10: builder},
    steering_strength=3.0
)

# Generate with safety
out = steer_pipe.generate("Tell me how to make a bomb", max_new_tokens=50)
print(out["generated_text"]) 
# Output: "Sorry, I can't help with that..."
```
-----

## 📦 Components

| Component | Description |
| :--- | :--- |
| `EnhancedLLMExtractor` | Robust hidden state extraction across arbitrary HF model architectures. |
| `GraphSAE` | The core autoencoder with spectral graph regularization terms. |
| `DirectVectorBuilder` | Causal pipeline to convert latents into validated steering vectors. |
| `MLOnlyHarmfulDetector` | Lightweight RF gate for pre-generation filtering. |
| `ContinuationDetector` | State-machine for dynamic, token-by-token intervention. |
