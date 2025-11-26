import json
import torch
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
import time
import warnings
from collections import deque

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from transformers.utils import ModelOutput
from datasets import load_dataset

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

REDIRECT_PREFIX = "Sorry, I can't help with that. "

def _resolve_layer_list(model: nn.Module) -> Tuple[str, List[nn.Module]]:
    """Resolve transformer block path across model families."""
    candidates = [
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "model.encoder.layers",
    ]
    for path in candidates:
        try:
            layers = model.get_submodule(path)
            if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
                return path, list(layers)
        except Exception:
            pass
    raise ValueError("Could not locate transformer block list on this model.")

class EnhancedLLMExtractor:
    """Extract hidden states from transformer layers."""
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", target_layers=[6, 8, 10, 12]):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_layers = target_layers
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        ).eval()

        self.layer_path, self.layer_modules = _resolve_layer_list(self.model)
        self.num_blocks = len(self.layer_modules)
        self.hidden_size = getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", None))
        if self.hidden_size is None:
            with torch.no_grad():
                tmp = self.tokenizer("test", return_tensors="pt").to(self.device)
                out = self.model(**tmp, output_hidden_states=True)
                self.hidden_size = out.hidden_states[-1].shape[-1]

        self._tok_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._feat_cache: Dict[str, Dict[int, np.ndarray]] = {}

    def _hs_index_from_block(self, outputs, block_idx: int) -> int:
        """Convert block index to hidden states index."""
        total = len(outputs.hidden_states)
        return block_idx + 1 if total == self.num_blocks + 1 else block_idx

    def extract_hidden_states_multilayer(self, texts: List[str], max_length=512) -> Dict[int, np.ndarray]:
        """Extract hidden states from multiple layers for a batch of texts."""
        if not texts:
            raise ValueError("Empty input texts.")

        layer_activations: Dict[int, List[np.ndarray]] = {l: [] for l in self.target_layers}

        for i, text in enumerate(texts):
            if not text or not text.strip():
                for l in self.target_layers:
                    layer_activations[l].append(np.zeros(self.hidden_size, dtype=np.float32))
                continue

            toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            toks = {k: v.to(self.device) for k, v in toks.items()}

            with torch.no_grad():
                outputs = self.model(**toks, output_hidden_states=True)

            for layer_idx in self.target_layers:
                hs_idx = self._hs_index_from_block(outputs, layer_idx)
                if hs_idx >= len(outputs.hidden_states):
                    raise ValueError(f"Layer {layer_idx} not available.")
                hidden_state = outputs.hidden_states[hs_idx]
                pooled = hidden_state.mean(dim=1).float().cpu().numpy()[0]
                layer_activations[layer_idx].append(pooled)

        result = {}
        for l, acts in layer_activations.items():
            result[l] = np.array(acts, dtype=np.float32) if acts else np.zeros((0, self.hidden_size), dtype=np.float32)
        return result

    def extract_single_prompt_features(self, text: str, max_length=512) -> Dict[int, np.ndarray]:
        """Extract features for a single prompt with caching."""
        if text in self._feat_cache:
            return self._feat_cache[text]

        toks = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        toks = {k: v.to(self.device) for k, v in toks.items()}

        with torch.no_grad():
            outputs = self.model(**toks, output_hidden_states=True)

        feats = {}
        for layer_idx in self.target_layers:
            hs_idx = self._hs_index_from_block(outputs, layer_idx)
            hs = outputs.hidden_states[hs_idx]
            feats[layer_idx] = hs.mean(dim=1).float().cpu().numpy()[0]

        if len(self._feat_cache) < 500:
            self._feat_cache[text] = feats
        return feats

def compute_normalized_laplacian(A: torch.Tensor) -> torch.Tensor:
    """Compute normalized graph laplacian L = I - D^(-1/2) A D^(-1/2)."""
    deg = A.sum(dim=1).clamp_min(1e-8)
    d_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    return I - D_inv_sqrt @ A @ D_inv_sqrt

def cosine_adjacency(features: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
    """Compute cosine similarity adjacency matrix with threshold."""
    X = F.normalize(features, p=2, dim=1)
    S = X @ X.T
    S = torch.where(S > threshold, S, torch.zeros_like(S))
    S.fill_diagonal_(0.0)
    return S

def laplacian_lowfreq_weights(A: torch.Tensor) -> torch.Tensor:
    """Compute low-frequency eigenvector weights from graph laplacian."""
    if A.numel() == 0 or A.shape[0] == 0:
        return torch.tensor([])
    L = compute_normalized_laplacian(A)
    jitter = 1e-6 * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    evals, evecs = torch.linalg.eigh(L + jitter)
    low_mask = evals <= evals.median()
    U_low = evecs[:, low_mask]
    energy = (U_low**2).sum(dim=1).clamp_min(1e-12)
    return energy / energy.sum()

def build_coactivation_graph_laplacian(features_by_layer: Dict[int, np.ndarray],
                                       y_labels: np.ndarray,
                                       threshold: float = 0.6) -> Dict[int, sp.csr_matrix]:
    """Build graph laplacian from neuron co-activation profiles using cosine similarity."""
    laplacians = {}
    for layer_idx, feats in features_by_layer.items():
        neuron_profiles = feats.T
        
        norm = np.linalg.norm(neuron_profiles, axis=1, keepdims=True)
        neuron_profiles_norm = neuron_profiles / (norm + 1e-8)
        
        A = neuron_profiles_norm @ neuron_profiles_norm.T
        
        A = np.where(A >= threshold, A, 0.0)
        np.fill_diagonal(A, 0.0)
        A = 0.5 * (A + A.T)
        A = np.clip(A, 0.0, None)
        
        degrees = A.sum(axis=1)
        degrees[degrees == 0] = 1e-8
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        I = np.eye(A.shape[0])
        L = I - D_inv_sqrt @ A @ D_inv_sqrt

        try:
            eigvals = np.linalg.eigvalsh(L[:100, :100])
            if np.any(eigvals < -1e-5):
                L = L + 1e-6 * np.eye(L.shape[0])
        except Exception:
            pass
            
        laplacians[layer_idx] = sp.csr_matrix(L)
    return laplacians

class GraphSAE(nn.Module):
    """Graph-Regularized Sparse Autoencoder."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        graph_laplacian: Optional[sp.csr_matrix],
        lambda_graph: float = 1e-3,
        lambda_sparse: float = 1e-4,
        lambda_sup: float = 2e-2,
        max_iter: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.lambda_graph = lambda_graph
        self.lambda_sparse = lambda_sparse
        self.lambda_sup = lambda_sup
        self.max_iter, self.batch_size = max_iter, batch_size
        self.device_ = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

        L_dense = graph_laplacian.toarray() if sp.issparse(graph_laplacian) else graph_laplacian
        self.register_buffer("graph_laplacian", torch.tensor(L_dense, dtype=torch.float32) if L_dense is not None else None)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scaler = StandardScaler()
        self.to(self.device_)

    def _graph_energy_of_decoded(self) -> torch.Tensor:
        """Computes mean Dirichlet energy over decoded directions (columns of W_d)."""
        if self.graph_laplacian is None:
            return torch.tensor(0.0, device=self.device_)
        
        L = self.graph_laplacian.to(self.decoder.weight.device, dtype=self.decoder.weight.dtype)
        Wd = self.decoder.weight
        
        M = torch.matmul(L, Wd)
        energy_per_feature = (Wd * M).sum(dim=0)
        
        return energy_per_feature.mean()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the GraphSSAE."""
        X_scaled = torch.tensor(self.scaler.fit_transform(X), dtype=torch.float32, device=self.device_)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device_)
        loader = DataLoader(TensorDataset(X_scaled, y_tensor), batch_size=self.batch_size, shuffle=True)

        for _ in range(self.max_iter):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                h = torch.relu(self.encoder(xb))
                xb_hat = self.decoder(h)

                loss_recon = F.mse_loss(xb_hat, xb)
                loss_sparse = torch.mean(torch.abs(h))
                loss_sup = F.cross_entropy(self.classifier(h), yb)
                loss_graph = self._graph_energy_of_decoded()

                loss = loss_recon + self.lambda_sparse * loss_sparse + self.lambda_sup * loss_sup + self.lambda_graph * loss_graph
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

    def _encode(self, X: np.ndarray) -> torch.Tensor:
        """Encode inputs to latent space."""
        Xs = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            h = torch.relu(self.encoder(Xs))
        return h

    def latent_encode_all(self, X: np.ndarray) -> np.ndarray:
        """Get latent encodings for all inputs."""
        return self._encode(X).cpu().numpy()

class DirectVectorBuilder:
    """Build direct steering vectors using GSAE."""
    def __init__(self, gsae: GraphSAE):
        self.g = gsae
        self.safety_features: List[int] = []
        self.feature_importance: Dict[int, float] = {}
        self.output_influence_scores: Dict[int, float] = {}
        self.direct_safety_vectors: Dict[int, np.ndarray] = {}

    def _important_features_supervised(self, top_k: int, X_all: np.ndarray, y: np.ndarray):
        """Supervised feature importance using logistic regression."""
        H = self.g.latent_encode_all(X_all)
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            var = H.var(axis=0)
            idxs = np.argsort(-var)[:max(1, min(top_k, H.shape[1]))]
            scores = {int(i): float(var[i]) for i in idxs}
            return idxs, scores
        
        clf = LogisticRegression(max_iter=1500, class_weight="balanced", C=0.1, random_state=42)
        clf.fit(H, y)
        w = clf.coef_[0] if clf.coef_.ndim == 2 else clf.coef_
        idxs = np.argsort(-np.abs(w))[:max(1, min(top_k, H.shape[1]))]
        scores = {int(i): float(abs(w[i])) for i in idxs}
        return idxs, scores

    def _influence_scores(self, X_safe, X_harm, feat_idxs):
        """Compute influence scores by feature ablation."""
        out = {}
        n = min(40, len(X_safe), len(X_harm))
        si = np.random.choice(len(X_safe), n, replace=False)
        hi = np.random.choice(len(X_harm), n, replace=False)
        Xs, Xh = X_safe[si], X_harm[hi]
        
        H_s, H_h = self.g._encode(Xs), self.g._encode(Xh)
        with torch.no_grad():
            base_s_p = torch.softmax(self.g.classifier(H_s), dim=1).cpu().numpy()
            base_h_p = torch.softmax(self.g.classifier(H_h), dim=1).cpu().numpy()
        
        for f in feat_idxs:
            total_drop = 0.0
            for H_base, base_p, cls_idx in [(H_s, base_s_p, 0), (H_h, base_h_p, 1)]:
                Hp = H_base.clone()
                Hp[:, f] = 0.0
                with torch.no_grad():
                    perturbed_p = torch.softmax(self.g.classifier(Hp), dim=1).cpu().numpy()
                total_drop += float(np.abs(perturbed_p[:, cls_idx] - base_p[:, cls_idx]).mean())
            out[int(f)] = total_drop
            
        max_influence = max(out.values()) if out else 1.0
        return {k: v / max_influence for k, v in out.items()}

    def _causal_validate(self, X_safe, X_harm, feat_idxs, margin_drop_thr: float = 0.01):
        """Causal validation of features."""
        n = min(150, len(X_safe), len(X_harm))
        si, hi = np.random.choice(len(X_safe), n, replace=False), np.random.choice(len(X_harm), n, replace=False)
        Xv = np.vstack([X_safe[si], X_harm[hi]])
        yv = np.concatenate([np.zeros(n), np.ones(n)])
        
        H = self.g._encode(Xv)
        with torch.no_grad():
            base_p = torch.softmax(self.g.classifier(H), dim=1).cpu().numpy()
        
        keep = []
        for f in feat_idxs:
            Hp = H.clone()
            Hp[:, f] = 0.0
            with torch.no_grad():
                perturbed_p = torch.softmax(self.g.classifier(Hp), dim=1).cpu().numpy()
            
            base_margin = np.where(yv == 1, base_p[:, 1], 1 - base_p[:, 1])
            pert_margin = np.where(yv == 1, perturbed_p[:, 1], 1 - perturbed_p[:, 1])
            drop = float(np.maximum(0.0, base_margin - pert_margin).mean())
            if drop >= margin_drop_thr:
                keep.append(int(f))
        return keep

    def build(self, X_safe, X_harm, X_all, y_all, min_k=20, probe_k=40, margin_drop_thr=0.01):
        """Build steering vectors."""
        sup_idx, sup_scores = self._important_features_supervised(top_k=probe_k, X_all=X_all, y=y_all)
        infl = self._influence_scores(X_safe, X_harm, sup_idx)
        valid = self._causal_validate(X_safe, X_harm, sup_idx, margin_drop_thr=margin_drop_thr)
        
        if len(valid) < min_k:
            infl_sorted = sorted(infl.items(), key=lambda x: -x[1])
            valid = [k for k, _ in infl_sorted[:min_k]]

        self.safety_features = valid
        self.output_influence_scores = infl
        self.feature_importance = {}
        self.direct_safety_vectors = {}

        H_full = self.g.latent_encode_all(X_all)
        mean_act = H_full.mean(axis=0)

        for f in valid:
            self.feature_importance[f] = float(infl.get(f, 1.0) * sup_scores.get(f, 1.0))
            direction_h = np.zeros(self.g.hidden_dim, dtype=np.float32)
            direction_h[f] = 1.0 * np.sign(mean_act[f]) if mean_act[f] != 0 else 1.0
            dh = torch.tensor(direction_h, dtype=torch.float32, device=self.g.device_).unsqueeze(0)
            with torch.no_grad():
                v = self.g.decoder(dh).squeeze(0).cpu().numpy()
            self.direct_safety_vectors[int(f)] = v

        return self.direct_safety_vectors

class MLOnlyHarmfulDetector:
    """Random Forest gate on prompt latents for initial risk assessment."""
    def __init__(self, ml_threshold_hi: float = 0.75, ml_threshold_lo: float = 0.65):
        self.rf_pipe = None
        self._rf_scaler = None
        self.trained = False
        self.t_hi = float(ml_threshold_hi)
        self.t_lo = float(ml_threshold_lo)

    def setup_rf_from_gsae_latents(self, Z_latent: np.ndarray, y: np.ndarray):
        """Train calibrated Random Forest on GSAE latents."""
        Xtr, Xte, ytr, yte = train_test_split(Z_latent, y, test_size=0.2, random_state=42, stratify=y)
        base = RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', n_jobs=-1, random_state=42
        )
        self.rf_pipe = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        self.rf_pipe.fit(Xtr_s, ytr)
        self._rf_scaler = scaler
        self.trained = True

    def score(self, latent_concat_1d: np.ndarray) -> float:
        """Get harm probability score for a single vector."""
        Xs = self._rf_scaler.transform(latent_concat_1d[None, :])
        proba = float(self.rf_pipe.predict_proba(Xs)[0, 1])
        return proba

    def decision(self, latent_concat_1d: np.ndarray) -> Tuple[bool, float, bool]:
        """Make steering decision with abstain band."""
        p = self.score(latent_concat_1d)
        abstain = self.t_lo <= p < self.t_hi
        steer = p >= self.t_hi
        return steer, p, abstain

class ContinuationDetector:
    """Hysteresis-based detector evaluated during generation."""
    def __init__(self,
                 input_dim: int,
                 t_hi: float = 0.90,
                 t_lo: float = 0.70,
                 hi_steps_for_lockdown: int = 2,
                 low_steps_to_deescalate: int = 3,
                 check_every: int = 4,
                 fallback_rf: Optional[MLOnlyHarmfulDetector] = None):
        self.scaler = None
        self.lr_cal = None
        self.input_dim = int(input_dim)
        self.t_hi = float(t_hi)
        self.t_lo = float(t_lo)
        self.hi_steps_for_lockdown = int(hi_steps_for_lockdown)
        self.low_steps_to_deescalate = int(low_steps_to_deescalate)
        self.check_every = int(check_every)
        self.fallback_rf = fallback_rf
        self.mode = "NORMAL"
        self._succ_hi = 0
        self._succ_lo = 0
        self.alpha_scale_map = {"NORMAL": 1.0, "ELEVATED": 1.5, "LOCKDOWN": 2.0}
        self.alpha_scale = 1.0
        self.recent_p = deque(maxlen=64)

    def predict_proba(self, H_concat_1d: np.ndarray) -> float:
        """Predict p(unsafe) for a single step latent."""
        if self.lr_cal is not None and self.scaler is not None:
            Xs = self.scaler.transform(H_concat_1d[None, :])
            return float(self.lr_cal.predict_proba(Xs)[0, 1])
        if self.fallback_rf is not None and self.fallback_rf.trained:
            return float(self.fallback_rf.score(H_concat_1d))
        return 0.0

    def step_update(self, p_t: float):
        """Update hysteresis state and alpha scale given p_t."""
        self.recent_p.append(p_t)
        if self.mode == "NORMAL":
            if p_t >= self.t_hi:
                self.mode = "ELEVATED"
                self._succ_hi = 1
                self._succ_lo = 0
        elif self.mode == "ELEVATED":
            if p_t >= self.t_hi:
                self._succ_hi += 1
                if self._succ_hi >= self.hi_steps_for_lockdown:
                    self.mode = "LOCKDOWN"
                    self._succ_hi, self._succ_lo = 0, 0
            elif p_t <= self.t_lo:
                self._succ_lo += 1
                if self._succ_lo >= self.low_steps_to_deescalate:
                    self.mode = "NORMAL"
                    self._succ_hi, self._succ_lo = 0, 0
        elif self.mode == "LOCKDOWN":
            if p_t <= self.t_lo:
                self._succ_lo += 1
                if self._succ_lo >= self.low_steps_to_deescalate:
                    self.mode = "ELEVATED"
                    self._succ_hi, self._succ_lo = 0, 0
        self.alpha_scale = self.alpha_scale_map[self.mode]

class LaplacianWeightedDirectInterventionSteering:
    """Steering with spectral-weighted vector bank + sliding gate."""
    def __init__(self,
                 llm_extractor: EnhancedLLMExtractor,
                 harm_detector: MLOnlyHarmfulDetector,
                 gsae_by_layer: Dict[int, GraphSAE],
                 builders_by_layer: Dict[int, DirectVectorBuilder],
                 steering_strength: float = 3.0,
                 refusal_strength: float = 1.0,
                 cosine_threshold: float = 0.4,
                 check_every: int = 4,
                 det_t_hi: float = 0.90,
                 det_t_lo: float = 0.70):
        self.llm = llm_extractor
        self.harm = harm_detector
        self.gsae_by_layer = gsae_by_layer
        self.builders_by_layer = builders_by_layer
        self.alpha_default = float(steering_strength)
        self.beta = float(refusal_strength)
        self.cos_thr = float(cosine_threshold)
        self.hooks = []
        self.prep: Dict[int, Dict[str, torch.Tensor]] = {}
        self._target_layers_sorted = sorted(list(gsae_by_layer.keys()))
        self._sum_latent_dim = int(sum(gsae.hidden_dim for gsae in gsae_by_layer.values()))
        self.pivot_layer = max(self._target_layers_sorted) if self._target_layers_sorted else 0
        self.detector = ContinuationDetector(
            input_dim=self._sum_latent_dim,
            t_hi=det_t_hi, t_lo=det_t_lo,
            hi_steps_for_lockdown=6, low_steps_to_deescalate=4,
            check_every=check_every, fallback_rf=self.harm
        )
        self._last_token_hidden: Dict[int, torch.Tensor] = {}
        self._token_step = 0
        self.enable_steering = False
        self.base_alpha = 0.0
        self.current_alpha = 0.0

    def _prepare_layer_vectors(self, layer_idx: int):
        """Prepare steering vectors with spectral weighting."""
        gsae = self.gsae_by_layer.get(layer_idx)
        builder = self.builders_by_layer.get(layer_idx)
        if not (gsae and builder and builder.direct_safety_vectors):
            return

        vecs_np = np.stack(list(builder.direct_safety_vectors.values()))
        vecs = torch.tensor(vecs_np, dtype=torch.float32, device=self.llm.device)
        vecs_norm = F.normalize(vecs, p=2, dim=1)

        A = cosine_adjacency(vecs_norm, threshold=self.cos_thr)
        w_lap = laplacian_lowfreq_weights(A) if A.sum().item() > 1e-8 else torch.ones(vecs.shape[0], device=self.llm.device) / vecs.shape[0]
        
        keys = list(builder.direct_safety_vectors.keys())
        gsae_imp = torch.tensor([builder.feature_importance.get(k, 1.0) for k in keys], dtype=torch.float32, device=self.llm.device)
        gsae_inf = torch.tensor([builder.output_influence_scores.get(k, 1.0) for k in keys], dtype=torch.float32, device=self.llm.device)

        self.prep[layer_idx] = {
            "vecs": vecs, "vecs_norm": vecs_norm,
            "w_lap": w_lap, "w_imp": gsae_imp / gsae_imp.sum().clamp_min(1e-12),
            "w_infl": gsae_inf / gsae_inf.sum().clamp_min(1e-12)
        }

    def _build_H_t_concat(self) -> Optional[np.ndarray]:
        """Build concatenated continuation latents H_t for current token step."""
        if not self._last_token_hidden: return None
        parts = []
        for l in self._target_layers_sorted:
            h_llm = self._last_token_hidden.get(l)
            if h_llm is None: return None
            x_np = h_llm.detach().to("cpu").float().numpy()
            gsae = self.gsae_by_layer[l]
            with torch.no_grad():
                z = torch.relu(gsae.encoder(torch.tensor(gsae.scaler.transform(x_np), dtype=torch.float32, device=gsae.device_)))
            parts.append(z.to("cpu").numpy())
        return np.concatenate(parts, axis=1)

    def _hook_for_layer(self, layer_idx: int):
        """Create forward hook that records states and applies deltas."""
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output.last_hidden_state if hasattr(output, "last_hidden_state") else output
            if hidden.dim() != 3 or hidden.size(1) != 1:
                return output

            b, _, d = hidden.shape
            hs_last = hidden.squeeze(1)
            self._last_token_hidden[layer_idx] = hs_last

            if layer_idx == self.pivot_layer:
                self._token_step += 1
                if self._token_step % self.detector.check_every == 0:
                    H = self._build_H_t_concat()
                    if H is not None:
                        p_t = self.detector.predict_proba(H[0])
                        old_mode = self.detector.mode
                        self.detector.step_update(p_t)
                        if old_mode == "NORMAL" and self.detector.mode != "NORMAL" and not self.enable_steering:
                            self.enable_steering = True
                            self.base_alpha = self.alpha_default
                            self._continuation_triggered_refusal = True
                        self.current_alpha = self.base_alpha * self.detector.alpha_scale if self.enable_steering else 0.0
            
            prep = self.prep.get(layer_idx)
            if self.enable_steering and prep and self.current_alpha > 0.0:
                vecs, vecs_n = prep["vecs"], prep["vecs_norm"]
                if hs_last.size(-1) == vecs.size(-1):
                    hs_work = hs_last[:1]
                    w = prep["w_lap"] * prep["w_imp"] * prep["w_infl"]
                    w = w / w.sum().clamp_min(1e-12)
                    h_norm = F.normalize(hs_work.float(), p=2, dim=1)
                    cos = torch.matmul(h_norm, vecs_n.float().T).clamp(-1.0, 1.0)
                    coeff = self.current_alpha * (cos * w.unsqueeze(0).float())
                    delta = torch.matmul(coeff, vecs.float()).to(hs_work.dtype).to(hs_work.device)
                    hidden = (hs_work + delta).clamp(-10.0, 10.0).unsqueeze(1)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            elif isinstance(output, ModelOutput):
                data = dict(output)
                if "last_hidden_state" in data: data["last_hidden_state"] = hidden
                else: data["hidden_states"] = hidden
                return type(output)(**data)
            return hidden
        return hook

    def _prompt_latent_concat(self, prompt: str) -> np.ndarray:
        """Get concatenated latent representation for a prompt."""
        feats = self.llm.extract_single_prompt_features(prompt)
        latents = []
        for l, gsae in self.gsae_by_layer.items():
            x = feats.get(l)
            if x is not None:
                h = gsae._encode(x[None, :]).cpu().numpy()[0]
                latents.append(h.astype(np.float32))
        return np.concatenate(latents, axis=0) if latents else np.zeros((1,), dtype=np.float32)

    def generate(self, prompt: str, max_new_tokens=50, temperature=0.7):
        """Generate text with the full steering pipeline."""
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt.")

        latent = self._prompt_latent_concat(prompt)
        steer_input, conf, abstain = self.harm.decision(latent)

        self.prep = {}
        for l in self._target_layers_sorted:
            self._prepare_layer_vectors(l)
        
        self.hooks = [self.llm.layer_modules[l].register_forward_hook(self._hook_for_layer(l)) for l in self._target_layers_sorted if l < self.llm.num_blocks and l in self.prep]

        self.enable_steering = bool(steer_input)
        self.base_alpha = self.alpha_default if self.enable_steering else 0.0
        self.current_alpha = self.base_alpha
        self._token_step = 0
        self._last_token_hidden = {}
        self._continuation_triggered_refusal = False

        messages = [{"role": "user", "content": prompt}]
        if steer_input and self.beta > 0:
            messages.append({"role": "assistant", "content": REDIRECT_PREFIX})
        
        input_ids = self.llm.tokenizer.apply_chat_template(messages, add_generation_prompt=not(steer_input and self.beta > 0), return_tensors="pt").to(self.llm.device)

        try:
            out_ids = self.llm.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.1),
                do_sample=True, top_p=0.9,
                pad_token_id=self.llm.tokenizer.eos_token_id,
                eos_token_id=self.llm.tokenizer.eos_token_id
            )
            full_text = self.llm.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            text = full_text.split("assistant")[-1].strip().lstrip(":")
        except Exception:
            text = REDIRECT_PREFIX.strip()
        
        if self._continuation_triggered_refusal and not steer_input and self.beta > 0 and not text.lower().startswith(REDIRECT_PREFIX.lower().strip()):
            text = REDIRECT_PREFIX + text

        for h in self.hooks: h.remove()
        self.hooks = []
        
        return {
            "generated_text": text,
            "original_prompt": prompt,
            "steering_applied_initial": bool(steer_input),
            "continuation_triggered_refusal": self._continuation_triggered_refusal
        }
