"""
model_hybrid.py
===============
EdgePhish-5G: Hybrid Model Architecture

PURPOSE:
    Defines the complete EdgePhish-5G neural network architecture:
        - DistilBERT encoder branch (semantic features)
        - Compressed TF-IDF branch (statistical features)
        - Learnable scalar fusion gate (alpha)
        - Classification head

    Also implements:
        - Teacher model wrapper (source manuscript BERT+TF-IDF)
        - Knowledge distillation loss function
        - Baseline models (CNN, RNN-GRU, Logistic Regression, etc.)
        - Model complexity analysis (parameter counting)

INPUTS:
    - BERT tokens: input_ids [batch × 128], attention_mask [batch × 128]
    - TF-IDF features: [batch × 2048] float32
    - Labels: [batch] binary (for loss computation)
    - Soft teacher logits: [batch × 1] (for KD loss)

OUTPUTS:
    - Phishing probability: [batch × 1] sigmoid output
    - Classification decision: BLOCK (>0.5) or ALLOW
    - Learned alpha value (fusion gate state)

WHY IT EXISTS:
    The architecture directly implements Contributions C2 (distillation),
    C3 (learnable fusion gate), and partially C1 (edge-deployable model).
    Keeping it separate from training.py enables:
        - Architecture ablation without touching training logic
        - ONNX export independent of training code
        - Fusion strategy swapping (C3 ablation E5)
"""

import os
import math
import json
import logging
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger('EdgePhish-5G.Model')

# ── PyTorch availability check ────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch {torch.__version__} available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. Model classes defined but not executable. "
        "Install PyTorch to run training. Sklearn simulation active."
    )

# ── DistilBERT availability check ─────────────────────────────────────────────
try:
    from transformers import DistilBertModel, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ── Model configuration ────────────────────────────────────────────────────────
@dataclass
class EdgePhishConfig:
    """
    Complete configuration for EdgePhish-5G model.
    All hyperparameters from Stage 3 methodology design.
    """
    # BERT branch
    bert_model_name: str = 'distilbert-base-uncased'
    bert_hidden_dim: int = 768
    bert_pca_components: int = 256
    bert_dropout: float = 0.10          # Lower: preserves semantic structure

    # TF-IDF branch
    tfidf_input_dim: int = 2048         # After SVD compression
    tfidf_projection_dim: int = 256     # Dimensional parity with BERT
    tfidf_dropout: float = 0.30         # Higher: prevents n-gram overfitting

    # Fusion gate
    fusion_strategy: str = 'alpha_gate' # 'concat'|'equal'|'late'|'alpha_only'|'alpha_gate'
    alpha_init: float = 0.0             # sigmoid(0.0) = 0.5 initial balance

    # Classification head
    # Input: concat(z_weighted[256], z_bert[256], z_tfidf[256]) = 768
    head_hidden_dims: List[int] = None
    head_dropout: float = 0.20

    # Distillation
    temperature: float = 4.0
    lambda_kd: float = 0.70

    # Output
    num_classes: int = 1                # Binary classification (sigmoid)
    decision_threshold: float = 0.50

    def __post_init__(self):
        if self.head_hidden_dims is None:
            self.head_hidden_dims = [256, 64]


if TORCH_AVAILABLE:

    # ── TF-IDF Branch ─────────────────────────────────────────────────────────
    class TFIDFBranch(nn.Module):
        """
        Processes compressed TF-IDF features through dense layers.

        Architecture:
            Linear(tfidf_input_dim → projection_dim) + ReLU
            Dropout(0.30)
            → [batch × 256]

        WHY ASYMMETRIC DROPOUT (0.30 vs BERT's 0.10):
            TF-IDF features are corpus-specific statistical signals.
            Higher dropout prevents the model from memorizing
            dataset-specific n-gram patterns that won't generalize
            to new phishing campaigns — critical for zero-day detection.
        """

        def __init__(self, config: EdgePhishConfig):
            super().__init__()
            self.projection = nn.Linear(
                config.tfidf_input_dim,
                config.tfidf_projection_dim,
                bias=True
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(config.tfidf_dropout)
            self.layer_norm = nn.LayerNorm(config.tfidf_projection_dim)

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """
            Args:
                x: TF-IDF features [batch × tfidf_input_dim]

            Returns:
                z_tfidf: [batch × tfidf_projection_dim (256)]
            """
            x = self.projection(x)      # [batch × 256]
            x = self.activation(x)
            x = self.layer_norm(x)
            x = self.dropout(x)
            return x

    # ── BERT Branch ───────────────────────────────────────────────────────────
    class BERTBranch(nn.Module):
        """
        DistilBERT encoder with PCA-equivalent learned projection.

        Architecture:
            DistilBERT (6 layers, 768-dim, 8 pruned heads)
            → CLS token [batch × 768]
            → Linear(768 → pca_components) + ReLU  [PCA approximation]
            → Dropout(0.10)
            → [batch × 256]

        NOTE ON PCA APPROXIMATION:
            In the full pipeline, offline PCA is fitted on training CLS
            representations and applied as a fixed transform. Here we use
            a trainable Linear layer as an in-model approximation, which
            allows end-to-end gradient flow through the projection.
            This is a design improvement over the source manuscript's
            offline PCA — it allows the projection to be jointly
            optimized with the classification objective.

        NOTE ON SIMULATION MODE:
            When DistilBERT is not available, uses a learned embedding
            layer over character tokens for architectural consistency.
            Replace with actual DistilBertModel for real training.
        """

        def __init__(self, config: EdgePhishConfig, simulation: bool = False):
            super().__init__()
            self.simulation = simulation
            self.hidden_dim = config.bert_hidden_dim
            self.pca_dim = config.bert_pca_components

            if not simulation and TRANSFORMERS_AVAILABLE:
                self.bert = DistilBertModel.from_pretrained(
                    config.bert_model_name
                )
                logger.info(
                    f"DistilBERT loaded: {config.bert_model_name}"
                )
            else:
                # Simulation: learned character embeddings + transformer
                logger.info(
                    "BERT simulation mode: using character embedding + "
                    "2-layer transformer (NOT equivalent to DistilBERT)"
                )
                self.char_embedding = nn.Embedding(
                    num_embeddings=30522,   # BERT vocab size
                    embedding_dim=128,
                    padding_idx=0
                )
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=4,
                    dim_feedforward=256,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=2
                )
                self.hidden_dim = 128   # Override for simulation

            # Projection layer (PCA approximation, jointly trained)
            self.projection = nn.Linear(
                self.hidden_dim,
                config.bert_pca_components
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(config.bert_dropout)
            self.layer_norm = nn.LayerNorm(config.bert_pca_components)

        def forward(
            self,
            input_ids: 'torch.Tensor',
            attention_mask: 'torch.Tensor'
        ) -> 'torch.Tensor':
            """
            Args:
                input_ids:      [batch × seq_len]
                attention_mask: [batch × seq_len]

            Returns:
                z_bert: [batch × pca_components (256)]
            """
            if not self.simulation and TRANSFORMERS_AVAILABLE:
                # Full DistilBERT forward pass
                outputs = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # CLS token: first position of last hidden state
                cls_output = outputs.last_hidden_state[:, 0, :]  # [batch × 768]
            else:
                # Simulation: character embedding + transformer
                x = self.char_embedding(input_ids)   # [batch × seq × 128]
                # Create padding mask for transformer
                # (True = position to ignore)
                pad_mask = (attention_mask == 0)     # [batch × seq]
                x = self.transformer(
                    x,
                    src_key_padding_mask=pad_mask
                )
                # CLS equivalent: mean pooling over non-padding positions
                mask_expanded = attention_mask.unsqueeze(-1).float()
                cls_output = (x * mask_expanded).sum(dim=1) / \
                             mask_expanded.sum(dim=1).clamp(min=1e-9)
                # [batch × 128]

            # Projection + normalization
            z = self.projection(cls_output)  # [batch × 256]
            z = self.activation(z)
            z = self.layer_norm(z)
            z = self.dropout(z)
            return z

    # ── Learnable Fusion Gate ─────────────────────────────────────────────────
    class FusionGate(nn.Module):
        """
        Implements the learnable scalar fusion gate (Contribution C3).

        Architecture:
            w_alpha: scalar trainable parameter (initialized = alpha_init)
            alpha = sigmoid(w_alpha) ∈ (0, 1)
            z_weighted = alpha * z_bert + (1 - alpha) * z_tfidf
            z_fused = concat([z_weighted, z_bert, z_tfidf])  → [batch × 768]

        The concatenation of all three (z_weighted, z_bert, z_tfidf)
        gives the classification head residual access to both raw
        feature streams, allowing it to learn corrections beyond the
        alpha weighting. This is the 'alpha_gate' strategy.

        Alternative fusion strategies (for ablation E5):
            'concat':      Fixed concatenation only [z_bert, z_tfidf] → [512]
            'equal':       Fixed alpha=0.5, no concat → [256]
            'late':        No shared fusion, each branch gets own head
            'alpha_only':  Weighted sum only, no concat → [256]
            'alpha_gate':  Full proposed (default) → [768]
        """

        def __init__(self, config: EdgePhishConfig):
            super().__init__()
            self.strategy = config.fusion_strategy
            self.branch_dim = config.bert_pca_components  # 256

            if self.strategy == 'alpha_gate':
                # Single scalar trainable weight
                self.w_alpha = nn.Parameter(
                    torch.tensor(config.alpha_init)
                )
                self.output_dim = self.branch_dim * 3  # 768

            elif self.strategy == 'alpha_only':
                self.w_alpha = nn.Parameter(
                    torch.tensor(config.alpha_init)
                )
                self.output_dim = self.branch_dim  # 256

            elif self.strategy == 'equal':
                # Fixed alpha = 0.5, no parameter
                self.output_dim = self.branch_dim  # 256

            elif self.strategy == 'concat':
                # No weighting, just concatenate
                self.output_dim = self.branch_dim * 2  # 512

            elif self.strategy == 'late':
                # Late fusion: handled externally; gate is identity
                self.output_dim = self.branch_dim * 2  # 512

            else:
                raise ValueError(
                    f"Unknown fusion strategy: {self.strategy}. "
                    "Choose from: alpha_gate|alpha_only|equal|concat|late"
                )

        def forward(
            self,
            z_bert: 'torch.Tensor',
            z_tfidf: 'torch.Tensor'
        ) -> Tuple['torch.Tensor', float]:
            """
            Args:
                z_bert:   BERT branch output [batch × 256]
                z_tfidf:  TF-IDF branch output [batch × 256]

            Returns:
                z_fused:  Fused representation [batch × output_dim]
                alpha:    Current gate value (scalar, for logging)
            """
            if self.strategy == 'alpha_gate':
                alpha = torch.sigmoid(self.w_alpha)
                z_weighted = alpha * z_bert + (1 - alpha) * z_tfidf
                z_fused = torch.cat([z_weighted, z_bert, z_tfidf], dim=-1)
                return z_fused, alpha.item()

            elif self.strategy == 'alpha_only':
                alpha = torch.sigmoid(self.w_alpha)
                z_fused = alpha * z_bert + (1 - alpha) * z_tfidf
                return z_fused, alpha.item()

            elif self.strategy == 'equal':
                z_fused = 0.5 * z_bert + 0.5 * z_tfidf
                return z_fused, 0.5

            elif self.strategy in ('concat', 'late'):
                z_fused = torch.cat([z_bert, z_tfidf], dim=-1)
                return z_fused, 0.5

        def get_alpha(self) -> float:
            """Get current learned alpha value (for analysis/logging)."""
            if hasattr(self, 'w_alpha'):
                return torch.sigmoid(self.w_alpha).item()
            return 0.5

    # ── Classification Head ───────────────────────────────────────────────────
    class ClassificationHead(nn.Module):
        """
        Multi-layer classification head for binary phishing prediction.

        Architecture (default for alpha_gate, input_dim=768):
            FC(768 → 256) → BatchNorm → ReLU → Dropout(0.2)
            FC(256 → 64)  → BatchNorm → ReLU → Dropout(0.2)
            FC(64  → 1)   → Sigmoid

        BatchNorm after each FC:
            - Stabilizes training with large hybrid input
            - Reduces sensitivity to feature scale differences
              between BERT (dense semantic) and TF-IDF (statistical)

        WHY SIGMOID NOT SOFTMAX:
            Binary classification → sigmoid on single logit is
            numerically equivalent to softmax on [logit, 0] but
            faster and produces well-calibrated probabilities
            for threshold-tuning in 5G deployment contexts.
        """

        def __init__(self, input_dim: int, config: EdgePhishConfig):
            super().__init__()
            layers = []
            in_dim = input_dim

            for hidden_dim in config.head_hidden_dims:
                layers.extend([
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.head_dropout)
                ])
                in_dim = hidden_dim

            # Final output layer
            layers.append(nn.Linear(in_dim, config.num_classes))
            self.network = nn.Sequential(*layers)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """
            Args:
                x: Fused representation [batch × input_dim]

            Returns:
                prob: Phishing probability [batch × 1]
            """
            logits = self.network(x)
            return self.sigmoid(logits)

    # ── Main EdgePhish-5G Model ───────────────────────────────────────────────
    class EdgePhish5G(nn.Module):
        """
        Full EdgePhish-5G hybrid model.

        Combines:
            - BERTBranch (semantic URL understanding)
            - TFIDFBranch (statistical pattern recognition)
            - FusionGate (learnable alpha weighting)
            - ClassificationHead (binary phishing prediction)

        Forward pass:
            1. BERT: [input_ids, attention_mask] → z_bert [batch × 256]
            2. TF-IDF: [tfidf_features] → z_tfidf [batch × 256]
            3. Fusion: [z_bert, z_tfidf] → z_fused [batch × 768]
            4. Head: [z_fused] → p(phishing) [batch × 1]

        The model is designed for:
            - Knowledge distillation training (with teacher soft labels)
            - Standard binary cross-entropy training
            - INT8 quantization-aware training
            - ONNX export for edge deployment
        """

        def __init__(
            self,
            config: EdgePhishConfig = None,
            simulation: bool = False
        ):
            """
            Args:
                config: EdgePhishConfig (uses defaults if None)
                simulation: Use simulation BERT if True
                            (when transformers not available)
            """
            super().__init__()
            self.config = config or EdgePhishConfig()
            self.threshold = self.config.decision_threshold

            # Branch modules
            self.bert_branch = BERTBranch(
                self.config,
                simulation=simulation or not TRANSFORMERS_AVAILABLE
            )
            self.tfidf_branch = TFIDFBranch(self.config)

            # Fusion gate
            self.fusion_gate = FusionGate(self.config)

            # Classification head (input dim from fusion gate)
            self.head = ClassificationHead(
                input_dim=self.fusion_gate.output_dim,
                config=self.config
            )

            # Track learned alpha across training
            self._alpha_history = []

            logger.info(
                f"EdgePhish-5G initialized | "
                f"strategy={self.config.fusion_strategy} | "
                f"fusion_dim={self.fusion_gate.output_dim} | "
                f"params={self.count_parameters():,}"
            )

        def forward(
            self,
            input_ids: 'torch.Tensor',
            attention_mask: 'torch.Tensor',
            tfidf_features: 'torch.Tensor'
        ) -> Tuple['torch.Tensor', float]:
            """
            Full forward pass.

            Args:
                input_ids:      [batch × 128] int64
                attention_mask: [batch × 128] int64
                tfidf_features: [batch × tfidf_input_dim] float32

            Returns:
                probs:  Phishing probabilities [batch × 1]
                alpha:  Current fusion gate value (scalar)
            """
            # Branch 1: BERT (semantic)
            z_bert = self.bert_branch(input_ids, attention_mask)

            # Branch 2: TF-IDF (statistical)
            z_tfidf = self.tfidf_branch(tfidf_features)

            # Fusion gate
            z_fused, alpha = self.fusion_gate(z_bert, z_tfidf)

            # Classification
            probs = self.head(z_fused)

            return probs, alpha

        def predict(
            self,
            input_ids: 'torch.Tensor',
            attention_mask: 'torch.Tensor',
            tfidf_features: 'torch.Tensor'
        ) -> Tuple['torch.Tensor', 'torch.Tensor', float]:
            """
            Inference-mode prediction with threshold application.

            Returns:
                probs:      Phishing probabilities [batch × 1]
                decisions:  Binary decisions [batch × 1] (1=BLOCK)
                alpha:      Fusion gate value
            """
            with torch.no_grad():
                probs, alpha = self.forward(
                    input_ids, attention_mask, tfidf_features
                )
                decisions = (probs >= self.threshold).float()
            return probs, decisions, alpha

        def count_parameters(self) -> int:
            """Count trainable parameters."""
            return sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

        def count_parameters_by_component(self) -> Dict[str, int]:
            """Count parameters per component for Table I."""
            return {
                'bert_branch': sum(
                    p.numel() for p in self.bert_branch.parameters()
                    if p.requires_grad
                ),
                'tfidf_branch': sum(
                    p.numel() for p in self.tfidf_branch.parameters()
                    if p.requires_grad
                ),
                'fusion_gate': sum(
                    p.numel() for p in self.fusion_gate.parameters()
                    if p.requires_grad
                ),
                'classification_head': sum(
                    p.numel() for p in self.head.parameters()
                    if p.requires_grad
                ),
                'total': self.count_parameters()
            }

        def get_model_size_mb(self) -> Dict[str, float]:
            """Compute model size in MB for different precisions."""
            total_params = self.count_parameters()
            return {
                'FP32_MB': round(total_params * 4 / (1024 ** 2), 1),
                'FP16_MB': round(total_params * 2 / (1024 ** 2), 1),
                'INT8_MB': round(total_params * 1 / (1024 ** 2), 1),
                'total_params_M': round(total_params / 1e6, 2)
            }

        def save_checkpoint(
            self,
            path: str,
            epoch: int,
            val_f1: float,
            optimizer_state: Optional[Dict] = None
        ):
            """Save model checkpoint."""
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'val_f1': val_f1,
                'model_state_dict': self.state_dict(),
                'config': self.config.__dict__,
                'alpha': self.fusion_gate.get_alpha()
            }
            if optimizer_state:
                checkpoint['optimizer_state_dict'] = optimizer_state
            torch.save(checkpoint, path)
            logger.info(
                f"Checkpoint saved: epoch={epoch} | "
                f"val_f1={val_f1:.4f} | alpha={checkpoint['alpha']:.3f}"
            )

        @classmethod
        def load_checkpoint(
            cls, path: str, simulation: bool = False
        ) -> 'EdgePhish5G':
            """Load model from checkpoint."""
            checkpoint = torch.load(path, map_location='cpu')
            config_dict = checkpoint.get('config', {})
            config = EdgePhishConfig(**config_dict)
            model = cls(config=config, simulation=simulation)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(
                f"Checkpoint loaded: epoch={checkpoint.get('epoch')} | "
                f"val_f1={checkpoint.get('val_f1', 0):.4f}"
            )
            return model

    # ── Distillation Loss ─────────────────────────────────────────────────────
    class DistillationLoss(nn.Module):
        """
        Knowledge distillation loss combining:
            - Hard label cross-entropy (ground truth)
            - Soft label KL divergence (teacher knowledge)

        Formula:
            L_total = (1 - lambda) * CE(y_hard, p_student)
                    + lambda * T^2 * KL(p_teacher(T) || p_student(T))

        The T^2 scaling compensates for the reduction in gradient
        magnitude caused by soft labels at high temperature.

        For binary classification:
            KL(p_T || p_S) = p_T * log(p_T/p_S) + (1-p_T) * log((1-p_T)/(1-p_S))

        Args:
            temperature: Distillation temperature T ∈ {1,2,4,6,8,10}
            lambda_kd:   Weight of KD term (0=pure CE, 1=pure KD)
        """

        def __init__(
            self,
            temperature: float = 4.0,
            lambda_kd: float = 0.70,
            eps: float = 1e-8
        ):
            super().__init__()
            self.T = temperature
            self.lambda_kd = lambda_kd
            self.eps = eps

        def forward(
            self,
            student_probs: 'torch.Tensor',
            teacher_logits: 'torch.Tensor',
            hard_labels: 'torch.Tensor'
        ) -> Tuple['torch.Tensor', Dict[str, float]]:
            """
            Args:
                student_probs:  Student phishing probabilities [batch × 1]
                teacher_logits: Teacher raw logits (pre-sigmoid) [batch × 1]
                hard_labels:    Ground truth binary labels [batch]

            Returns:
                loss:      Scalar total loss
                loss_dict: Component losses for logging
            """
            student_probs = student_probs.squeeze(-1)       # [batch]
            teacher_logits = teacher_logits.squeeze(-1)     # [batch]
            hard_labels = hard_labels.float()               # [batch]

            # Hard label CE loss
            ce_loss = F.binary_cross_entropy(
                student_probs.clamp(self.eps, 1 - self.eps),
                hard_labels
            )

            # Soft teacher probabilities at temperature T
            # For binary: apply temperature to logit, then sigmoid
            p_teacher_soft = torch.sigmoid(teacher_logits / self.T)
            p_student_soft = torch.sigmoid(
                torch.log(
                    student_probs.clamp(self.eps, 1 - self.eps) /
                    (1 - student_probs).clamp(self.eps, 1 - self.eps)
                ) / self.T
            )

            # KL divergence for binary classification
            kl_loss = (
                p_teacher_soft *
                torch.log(
                    p_teacher_soft.clamp(self.eps) /
                    p_student_soft.clamp(self.eps)
                ) +
                (1 - p_teacher_soft) *
                torch.log(
                    (1 - p_teacher_soft).clamp(self.eps) /
                    (1 - p_student_soft).clamp(self.eps)
                )
            ).mean()

            # Total loss with T^2 scaling
            total_loss = (
                (1 - self.lambda_kd) * ce_loss +
                self.lambda_kd * (self.T ** 2) * kl_loss
            )

            return total_loss, {
                'ce_loss': ce_loss.item(),
                'kl_loss': kl_loss.item(),
                'total_loss': total_loss.item()
            }

    # ── Baseline Models (PyTorch) ─────────────────────────────────────────────
    class CharCNN(nn.Module):
        """
        Character-level CNN baseline for phishing URL detection.
        Based on Tang & Mahmoud (2021) architecture.
        """

        def __init__(
            self,
            vocab_size: int = 128,
            embed_dim: int = 64,
            num_filters: int = 128,
            kernel_sizes: List[int] = None,
            dropout: float = 0.5,
            max_len: int = 128
        ):
            super().__init__()
            if kernel_sizes is None:
                kernel_sizes = [3, 4, 5]

            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, num_filters, k)
                for k in kernel_sizes
            ])
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            """x: [batch × seq_len] character indices"""
            x = self.embedding(x)               # [batch × seq × embed_dim]
            x = x.permute(0, 2, 1)              # [batch × embed_dim × seq]

            pooled = []
            for conv in self.convs:
                c = F.relu(conv(x))             # [batch × filters × (seq-k+1)]
                c = F.max_pool1d(c, c.size(2))  # [batch × filters × 1]
                pooled.append(c.squeeze(2))

            x = torch.cat(pooled, dim=1)        # [batch × (filters*n_kernels)]
            x = self.dropout(x)
            return self.sigmoid(self.fc(x))

    class BiGRU(nn.Module):
        """
        Bidirectional GRU baseline.
        """

        def __init__(
            self,
            vocab_size: int = 128,
            embed_dim: int = 64,
            hidden_size: int = 256,
            num_layers: int = 2,
            dropout: float = 0.3
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.gru = nn.GRU(
                embed_dim, hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size * 2, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
            x = self.embedding(x)
            output, hidden = self.gru(x)
            # Use last forward + last backward hidden states
            h = torch.cat([hidden[-2], hidden[-1]], dim=1)
            h = self.dropout(h)
            return self.sigmoid(self.fc(h))


# ── Sklearn-based Baseline Models ─────────────────────────────────────────────
class SklearnBaselineFactory:
    """
    Factory for sklearn baseline models (Group A baselines from Stage 4).

    These run without PyTorch, enabling full baseline comparison
    even in environments without GPU.
    """

    @staticmethod
    def get_logistic_regression():
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            random_state=42, class_weight='balanced'
        )

    @staticmethod
    def get_random_forest():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, max_depth=None,
            random_state=42, class_weight='balanced',
            n_jobs=-1
        )

    @staticmethod
    def get_svm():
        from sklearn.svm import SVC
        return SVC(
            kernel='rbf', C=10.0, gamma='scale',
            probability=True, random_state=42,
            class_weight='balanced'
        )

    @staticmethod
    def get_xgboost():
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                random_state=42, eval_metric='logloss',
                use_label_encoder=False
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("XGBoost not available, using GradientBoosting")
            return GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=6,
                random_state=42
            )

    @staticmethod
    def get_all() -> Dict:
        """Return dict of all sklearn baselines."""
        factory = SklearnBaselineFactory
        return {
            'Logistic Regression + TF-IDF': factory.get_logistic_regression(),
            'Random Forest + TF-IDF': factory.get_random_forest(),
            'SVM + TF-IDF': factory.get_svm(),
        }


# ── Model Analysis Utilities ──────────────────────────────────────────────────
class ModelAnalyzer:
    """
    Compute model complexity metrics for Table I of the paper.
    Works with both PyTorch models and sklearn models.
    """

    @staticmethod
    def analyze_pytorch(model: 'nn.Module', name: str = "Model") -> Dict:
        """Analyze a PyTorch model."""
        if not TORCH_AVAILABLE:
            return {}
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        return {
            'name': name,
            'total_params': total,
            'trainable_params': trainable,
            'frozen_params': total - trainable,
            'size_fp32_mb': round(total * 4 / (1024 ** 2), 1),
            'size_int8_mb': round(total * 1 / (1024 ** 2), 1),
        }

    @staticmethod
    def print_model_table(models_info: List[Dict]):
        """Print formatted model comparison table."""
        print(f"\n{'Model':<35} {'Params (M)':>12} {'FP32 (MB)':>10} {'INT8 (MB)':>10}")
        print("-" * 70)
        for info in models_info:
            name = info.get('name', 'Unknown')[:34]
            params_m = info.get('total_params', 0) / 1e6
            fp32 = info.get('size_fp32_mb', 0)
            int8 = info.get('size_int8_mb', 0)
            print(f"  {name:<33} {params_m:>10.2f}M {fp32:>9.1f} {int8:>9.1f}")
        print()


# ── Simulation model for testing without PyTorch ─────────────────────────────
class EdgePhish5GSimulation:
    """
    NumPy/sklearn simulation of EdgePhish-5G for environments
    without PyTorch. Used for pipeline testing and Stage 5 validation.

    Implements the same interface as EdgePhish5G but uses:
        - Logistic Regression on TF-IDF features (TF-IDF branch proxy)
        - Averaged token attention as BERT proxy
        - Weighted voting for fusion gate simulation
    """

    def __init__(self, config: EdgePhishConfig = None, seed: int = 42):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.config = config or EdgePhishConfig()
        self.seed = seed
        self.threshold = self.config.decision_threshold

        # TF-IDF branch: Logistic Regression
        self.tfidf_clf = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            random_state=seed, class_weight='balanced'
        )

        # BERT branch proxy: Logistic Regression on token density features
        self.bert_clf = LogisticRegression(
            C=1.0, max_iter=1000, solver='lbfgs',
            random_state=seed, class_weight='balanced'
        )

        # Learned alpha (initialized to 0.5)
        self.alpha = 0.5
        self.is_fitted = False

        logger.info(
            "EdgePhish-5G SIMULATION initialized (no PyTorch). "
            "Results are approximations — use PyTorch for paper experiments."
        )

    def fit(
        self,
        tfidf_features: np.ndarray,
        bert_tokens: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> 'EdgePhish5GSimulation':
        """
        Fit simulation model on training data.

        Args:
            tfidf_features: [n × 2048] float32
            bert_tokens:    Dict with 'input_ids' [n × 128]
            labels:         [n] binary
        """
        # TF-IDF branch
        self.tfidf_clf.fit(tfidf_features, labels)

        # BERT proxy: use token density features
        bert_features = self._extract_bert_proxy_features(bert_tokens)
        self.bert_clf.fit(bert_features, labels)

        # Optimize alpha on training data via simple grid search
        best_alpha, best_f1 = 0.5, 0.0
        from sklearn.metrics import f1_score

        for alpha_test in np.arange(0.1, 1.0, 0.1):
            p_tfidf = self.tfidf_clf.predict_proba(tfidf_features)[:, 1]
            p_bert = self.bert_clf.predict_proba(bert_features)[:, 1]
            p_fused = alpha_test * p_bert + (1 - alpha_test) * p_tfidf
            preds = (p_fused >= self.threshold).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_alpha = alpha_test

        self.alpha = best_alpha
        self.is_fitted = True
        logger.info(
            f"Simulation model fitted | learned alpha={self.alpha:.2f} | "
            f"train F1≈{best_f1:.4f}"
        )
        return self

    def predict_proba(
        self,
        tfidf_features: np.ndarray,
        bert_tokens: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Returns phishing probabilities [n_samples].
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() first")

        p_tfidf = self.tfidf_clf.predict_proba(tfidf_features)[:, 1]
        bert_features = self._extract_bert_proxy_features(bert_tokens)
        p_bert = self.bert_clf.predict_proba(bert_features)[:, 1]

        # Fusion gate simulation
        p_fused = self.alpha * p_bert + (1 - self.alpha) * p_tfidf
        return p_fused

    def predict(
        self,
        tfidf_features: np.ndarray,
        bert_tokens: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Returns (probabilities, binary_decisions, alpha).
        """
        probs = self.predict_proba(tfidf_features, bert_tokens)
        decisions = (probs >= self.threshold).astype(int)
        return probs, decisions, self.alpha

    def _extract_bert_proxy_features(
        self, bert_tokens: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Extract simple features from tokenized URLs as BERT proxy.

        Features:
            - Sequence length (effective tokens)
            - Token diversity (unique token ratio)
            - Mean token value (proxy for character distribution)
            - Special token density
        """
        input_ids = bert_tokens['input_ids']    # [n × 128]
        attention_mask = bert_tokens['attention_mask']  # [n × 128]

        seq_lengths = attention_mask.sum(axis=1)         # [n]
        mean_token = (input_ids * attention_mask).sum(axis=1) / \
                     np.maximum(seq_lengths, 1)           # [n]
        token_std = np.array([
            input_ids[i, :seq_lengths[i]].std()
            for i in range(len(input_ids))
        ])

        # Count tokens in suspicious ASCII ranges (phishing character proxies)
        # ASCII 48-57: digits; 45: hyphen; 95: underscore; 64: @
        suspicious_mask = np.isin(input_ids, [45, 64, 95, 48, 49, 50])
        suspicious_ratio = (suspicious_mask * attention_mask).sum(axis=1) / \
                           np.maximum(seq_lengths, 1)

        return np.column_stack([
            seq_lengths,
            mean_token,
            token_std,
            suspicious_ratio
        ]).astype(np.float32)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import EdgePhishDataset
    from feature_extraction import FeatureExtractor

    print("=" * 60)
    print("  EdgePhish-5G — Model Architecture Validation")
    print("=" * 60)

    # ── Test with PyTorch (if available) ──────────────────────────────
    if TORCH_AVAILABLE:
        print("\n[PyTorch Mode]")
        config = EdgePhishConfig(
            tfidf_input_dim=256,        # Use 256 for speed in test
            fusion_strategy='alpha_gate'
        )
        model = EdgePhish5G(config=config, simulation=True)

        # Print parameter breakdown
        param_breakdown = model.count_parameters_by_component()
        size_info = model.get_model_size_mb()
        print(f"\nParameter breakdown:")
        for component, count in param_breakdown.items():
            print(f"  {component:<25}: {count:>10,}")
        print(f"\nModel size:")
        for precision, size in size_info.items():
            print(f"  {precision:<15}: {size}")

        # Test forward pass
        batch_size = 4
        input_ids = torch.randint(0, 30522, (batch_size, 128))
        attention_mask = torch.ones(batch_size, 128, dtype=torch.long)
        tfidf_features = torch.randn(batch_size, 256)

        probs, alpha = model(input_ids, attention_mask, tfidf_features)
        print(f"\nForward pass test:")
        print(f"  Input: batch={batch_size}, tfidf_dim=256, seq_len=128")
        print(f"  Output probs shape: {probs.shape}")
        print(f"  Output probs: {probs.squeeze().detach().numpy().round(4)}")
        print(f"  Fusion alpha: {alpha:.4f}")

        # Test distillation loss
        teacher_logits = torch.randn(batch_size, 1)
        labels = torch.randint(0, 2, (batch_size,))
        loss_fn = DistillationLoss(temperature=4.0, lambda_kd=0.7)
        loss, loss_dict = loss_fn(probs, teacher_logits, labels)
        print(f"\nDistillation loss test:")
        for k, v in loss_dict.items():
            print(f"  {k}: {v:.4f}")

        # Test fusion ablation variants
        print(f"\nFusion strategy output dims:")
        for strategy in ['concat', 'equal', 'alpha_only', 'alpha_gate']:
            cfg = EdgePhishConfig(tfidf_input_dim=256, fusion_strategy=strategy)
            m = EdgePhish5G(config=cfg, simulation=True)
            p, a = m(input_ids, attention_mask, tfidf_features)
            print(f"  {strategy:<15}: fusion_dim={m.fusion_gate.output_dim}, "
                  f"alpha={a:.3f}")

    # ── Simulation Model Validation ──────────────────────────────────
    print("\n[Simulation Mode — sklearn]")
    dataset = EdgePhishDataset(data_path='data/urls_dataset.csv', seed=42)
    dataset.load_and_preprocess()
    splits = dataset.get_splits()

    train_urls = splits['train']['url'].tolist()[:2000]
    train_labels = splits['train']['label'].values[:2000]
    val_urls = splits['val']['url'].tolist()[:500]
    val_labels = splits['val']['label'].values[:500]

    extractor = FeatureExtractor(
        tfidf_config={'ngram_range': (2,4), 'max_features': 10000,
                      'chi2_k': 1000, 'svd_components': 64},
        seed=42
    )
    extractor.fit_tfidf(train_urls, train_labels)
    X_train = extractor.extract_tfidf(train_urls)
    X_val = extractor.extract_tfidf(val_urls)
    tok_train = extractor.extract_bert_tokens(train_urls)
    tok_val = extractor.extract_bert_tokens(val_urls)

    sim_config = EdgePhishConfig(tfidf_input_dim=64)
    sim_model = EdgePhish5GSimulation(config=sim_config, seed=42)
    sim_model.fit(X_train, tok_train, train_labels)
    probs, decisions, alpha = sim_model.predict(X_val, tok_val)

    from sklearn.metrics import f1_score, accuracy_score
    f1 = f1_score(val_labels, decisions, zero_division=0)
    acc = accuracy_score(val_labels, decisions)
    print(f"\nSimulation model results (val set, n={len(val_labels):,}):")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Alpha (α):   {alpha:.4f}")
    print(f"  Block rate:  {decisions.mean():.4f}")

    print("\n✓ Model architecture module validated")
