"""
model.py - EfficientNet-B0 based Diabetic Retinopathy severity classifier.

Architecture:
    - Backbone: EfficientNet-B0 (pretrained on ImageNet via timm)
    - Dropout layer (p=0.3) with MC-Dropout support
    - Classification head: Linear(1280 → 5) for DR grades 0-4

Features:
    - MC-Dropout toggle for uncertainty estimation at inference
    - Backbone freezing/unfreezing for fine-tuning strategies
"""

import torch
import torch.nn as nn
import timm


class DRClassifier(nn.Module):
    """
    Diabetic Retinopathy severity classifier using EfficientNet-B0.
    
    Outputs 5-class logits corresponding to DR grades:
        0 - No DR
        1 - Mild NPDR
        2 - Moderate NPDR
        3 - Severe NPDR
        4 - Proliferative DR
    
    Args:
        pretrained (bool): Whether to load ImageNet pretrained weights.
                           Default: True.
    """

    def __init__(self, pretrained: bool = True):
        super(DRClassifier, self).__init__()

        # -----------------------------------------------------------------
        # Backbone: EfficientNet-B0 with classifier head removed (num_classes=0)
        # This returns a 1280-dim feature vector per image.
        # -----------------------------------------------------------------
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0  # removes the default classifier, keeps pooling
        )

        # -----------------------------------------------------------------
        # Dropout: p=0.3, also used for Monte-Carlo Dropout at inference
        # -----------------------------------------------------------------
        self.dropout = nn.Dropout(p=0.3)

        # -----------------------------------------------------------------
        # Classification head: maps 1280 backbone features → 5 DR grades
        # -----------------------------------------------------------------
        self.grade_head = nn.Linear(1280, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            grade_logits: Tensor of shape (B, 5) with raw class logits.
        """
        features = self.backbone(x)           # (B, 1280)
        features = self.dropout(features)      # (B, 1280) with dropout
        grade_logits = self.grade_head(features)  # (B, 5)
        return grade_logits

    # =====================================================================
    # MC-Dropout utilities
    # =====================================================================

    def enable_mc_dropout(self) -> None:
        """
        Enable Monte-Carlo Dropout for uncertainty estimation.
        
        Keeps the dropout layer in training mode even when the rest of the
        model is in eval mode.  Run multiple forward passes and aggregate
        predictions to estimate epistemic uncertainty.
        """
        self.dropout.train()

    def disable_mc_dropout(self) -> None:
        """
        Disable Monte-Carlo Dropout, restoring normal eval behaviour
        (dropout is deactivated during inference).
        """
        self.dropout.eval()

    # =====================================================================
    # Backbone freezing / unfreezing for transfer-learning schedules
    # =====================================================================

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters so only the classification head
        is updated during training.  Useful for the initial fine-tuning
        phase on a small dataset.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_layers: int = 30) -> None:
        """
        Unfreeze the last `num_layers` parameter tensors of the backbone.
        
        This enables gradual unfreezing: start by training only the head,
        then progressively unfreeze deeper backbone layers.

        Args:
            num_layers: Number of parameter tensors (from the end) to
                        unfreeze.  Default: 30.
        """
        params = list(self.backbone.parameters())
        # First freeze everything …
        for param in params:
            param.requires_grad = False
        # … then unfreeze the last `num_layers` parameter tensors
        for param in params[-num_layers:]:
            param.requires_grad = True


# =========================================================================
# Quick sanity check
# =========================================================================
if __name__ == "__main__":
    model = DRClassifier(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {logits.shape}")  # Expected: (2, 5)

    # Test MC-Dropout toggle
    model.eval()
    model.enable_mc_dropout()
    print(f"Dropout training mode (MC on): {model.dropout.training}")  # True
    model.disable_mc_dropout()
    print(f"Dropout training mode (MC off): {model.dropout.training}")  # False

    # Test freeze / unfreeze
    model.freeze_backbone()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after freeze: {trainable}")

    model.unfreeze_backbone(num_layers=30)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params after unfreeze(30): {trainable}")
