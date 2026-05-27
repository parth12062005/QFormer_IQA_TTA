"""
TTA Engine — Orchestrates the test-time adaptation loop.

Handles:
  - Model state reset (back to checkpoint) per batch
  - Conditional computation (augmentations / VGG) based on selected losses
  - Adaptive distortion selection (blur/noise/compression) for rank losses
  - Optimizer setup for unfrozen parameters
  - Multi-step TTA optimization
  - Final clean prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .param_strategy import get_tta_params, freeze_all_except
from .augmentations import create_all_augmentations


class TTAEngine:
    """
    Config-driven TTA engine.

    Args:
        qformer:          QformerWrapper (with .forward_qformer, .extract_image_embeds)
        regressor:        Regressor module
        proj_head:        ProjectionHead module
        losses:           List of instantiated TTALoss objects
        unfreeze_strategy: "none" | "layernorm" | "query" | "both"
        tta_steps:        Number of optimization steps per batch
        tta_lr:           Learning rate for TTA optimizer
        freeze_proj_head: If True, projection head is NOT updated during TTA
        vgg_extractor:    VGGFeatureExtractor (loaded only if any loss requires VGG)
        device:           torch.device
    """

    def __init__(
        self,
        qformer,
        regressor,
        proj_head,
        losses,
        unfreeze_strategy: str,
        tta_steps: int = 3,
        tta_lr: float = 1e-3,
        freeze_proj_head: bool = True,
        vgg_extractor=None,
        device=None,
    ):
        self.qformer = qformer
        self.regressor = regressor
        self.proj_head = proj_head
        self.losses = losses
        self.unfreeze_strategy = unfreeze_strategy
        self.tta_steps = tta_steps
        self.tta_lr = tta_lr
        self.freeze_proj_head = freeze_proj_head
        self.vgg_extractor = vgg_extractor
        self.device = device or torch.device("cuda:0")

        # Determine what the selected losses need
        self.needs_augmentations = any(l.requires_augmentations for l in losses)
        self.needs_vgg = any(l.requires_vgg for l in losses)

        if self.needs_vgg and self.vgg_extractor is None:
            raise ValueError(
                "One or more selected losses require VGG features, "
                "but no VGGFeatureExtractor was provided."
            )

        # Snapshot the original model state for reset
        self._snapshot_state()

    def _snapshot_state(self):
        """Save the original checkpoint state so we can reset per batch."""
        self._orig_query_tokens = self.qformer.model.query_tokens.detach().clone()

        self._qf_layernorms = [
            m for m in self.qformer.model.Qformer.modules()
            if isinstance(m, nn.LayerNorm)
        ]
        self._orig_ln_states = [
            {
                "weight": m.weight.detach().clone() if m.weight is not None else None,
                "bias": m.bias.detach().clone() if m.bias is not None else None,
            }
            for m in self._qf_layernorms
        ]

        # Also snapshot proj_head if we're updating it
        if not self.freeze_proj_head:
            self._orig_proj_head = {
                k: v.detach().clone() for k, v in self.proj_head.state_dict().items()
            }

    def _reset_state(self):
        """Reset Q-Former params back to the checkpoint snapshot."""
        with torch.no_grad():
            self.qformer.model.query_tokens.copy_(self._orig_query_tokens)
            for m, state in zip(self._qf_layernorms, self._orig_ln_states):
                if m.weight is not None and state["weight"] is not None:
                    m.weight.copy_(state["weight"])
                if m.bias is not None and state["bias"] is not None:
                    m.bias.copy_(state["bias"])

            # Reset proj_head too if we're updating it
            if not self.freeze_proj_head:
                self.proj_head.load_state_dict(self._orig_proj_head)

    def _adaptive_distortion_selection(self, clip_images, old_preds):
        """
        Per-sample adaptive distortion selection (from original TTA-IQA).

        Creates 3 distortion types (blur, noise, compression) at 2 severities.
        For each sample, picks the distortion type where the model shows
        the LARGEST prediction difference |pred_strong - pred_weak|.

        Returns:
            f_low_embeds:  (B, 257, 1408) — ViT embeddings of weak distortion
            f_high_embeds: (B, 257, 1408) — ViT embeddings of strong distortion
        """
        with torch.no_grad():
            # Create all 6 augmented versions
            augs = create_all_augmentations(clip_images)

            # Extract ViT embeddings for all 6
            aug_embeds = {}
            for name, aug_imgs in augs.items():
                aug_embeds[name] = self.qformer.extract_image_embeds(aug_imgs)

            # Get predictions for all 6 via QFormer + regressor
            aug_preds = {}
            for name, embeds in aug_embeds.items():
                # Use dummy prompts since we only need relative predictions
                B = embeds.size(0)
                dummy_prompts = [""] * B
                mm = self.qformer.forward_qformer(embeds, dummy_prompts, dummy_prompts)
                aug_preds[name] = self.regressor(mm).squeeze(-1)

            # For each sample, pick distortion type with largest |pred_strong - pred_weak|
            diff_blur = torch.abs(aug_preds["blur_strong"] - aug_preds["blur_weak"])
            diff_noise = torch.abs(aug_preds["noise_strong"] - aug_preds["noise_weak"])
            diff_comp = torch.abs(aug_preds["comp_strong"] - aug_preds["comp_weak"])

            all_diff = torch.stack([diff_blur, diff_noise, diff_comp], dim=1)  # (B, 3)
            best_type = all_diff.argmax(dim=1)  # (B,) — 0=blur, 1=noise, 2=comp

            # Map type indices to augmentation names
            type_map = {
                0: ("blur_weak", "blur_strong"),
                1: ("noise_weak", "noise_strong"),
                2: ("comp_weak", "comp_strong"),
            }

            # Select per-sample: build f_low and f_high tensors
            B = clip_images.size(0)
            embed_dim = aug_embeds["blur_weak"].shape[1:]  # (257, 1408)
            f_low = torch.zeros(B, *embed_dim, device=self.device)
            f_high = torch.zeros(B, *embed_dim, device=self.device)

            for i in range(B):
                t = best_type[i].item()
                weak_name, strong_name = type_map[t]
                f_low[i] = aug_embeds[weak_name][i]
                f_high[i] = aug_embeds[strong_name][i]

        return f_low, f_high

    def adapt_and_predict(self, batch):
        """
        Run TTA on a single batch and return predictions.

        Args:
            batch: dict with keys:
                - "image_embeds": (B, 257, 1408) precomputed ViT embeddings
                - "prompts": list[str]
                - "descs": list[str]
                - "gt_scores": (B,) tensor
                - "image_names": list[str]
                - (optional) "clip_images": (B, 3, 224, 224) for augmentations
                - (optional) "vgg_images": (B, 3, 224, 224) for VGG features

        Returns:
            preds: np.ndarray (B,)
            gt_scores: np.ndarray (B,)
            metadata: list[dict] with per-image details
        """
        image_embeds = batch["image_embeds"].to(self.device, non_blocking=True)
        prompts = batch["prompts"]
        descs = batch["descs"]
        gt_scores = batch["gt_scores"]
        B = image_embeds.size(0)

        # 1. Reset model to checkpoint state
        self._reset_state()

        # 2. Only do TTA if we have enough samples and losses are specified
        if len(self.losses) > 0 and B > 1:
            self._run_tta(batch, image_embeds, prompts, descs)

        # 3. Final clean prediction
        self.qformer.eval()
        self.regressor.eval()
        with torch.no_grad():
            mm_embeds = self.qformer.forward_qformer(image_embeds, prompts, descs)
            preds = self.regressor(mm_embeds).squeeze(-1)

        preds_np = preds.float().cpu().numpy()
        gt_np = gt_scores.float().cpu().numpy() if torch.is_tensor(gt_scores) else gt_scores

        # Build per-image metadata
        metadata = []
        for i in range(B):
            metadata.append({
                "image_name": batch["image_names"][i],
                "prompt": prompts[i],
                "gen_answer": descs[i],
                "gt_score": float(gt_np[i]),
                "pred_score": float(preds_np[i]),
            })

        return preds_np, gt_np, metadata

    def _run_tta(self, batch, image_embeds, prompts, descs):
        """Execute TTA optimization steps."""

        # --- Get initial predictions (detached) for GC/FAGC clustering ---
        with torch.no_grad():
            init_embeds = self.qformer.forward_qformer(image_embeds, prompts, descs)
            init_preds = self.regressor(init_embeds).squeeze(-1).detach()

        # --- Precompute augmented embeddings if needed ---
        embeds_weak = embeds_strong = None
        if self.needs_augmentations:
            if "clip_images" not in batch:
                raise RuntimeError(
                    "Rank-based losses need raw images ('clip_images' key in batch) "
                    "to create augmented versions. Make sure the dataset provides them."
                )
            clip_images = batch["clip_images"]
            # Adaptive distortion selection: pick best distortion per sample
            embeds_weak, embeds_strong = self._adaptive_distortion_selection(
                clip_images, init_preds
            )

        # --- Precompute VGG features if needed ---
        vgg_feats = None
        if self.needs_vgg:
            if "vgg_images" not in batch:
                raise RuntimeError(
                    "FAGC loss needs VGG-preprocessed images ('vgg_images' key in batch)."
                )
            vgg_feats = self.vgg_extractor(batch["vgg_images"])

        # --- Setup optimizer ---
        params_to_update = get_tta_params(self.qformer, self.unfreeze_strategy)
        freeze_all_except(self.qformer, params_to_update)

        # Optionally include proj_head params
        if not self.freeze_proj_head:
            all_params = params_to_update + list(self.proj_head.parameters())
            for p in self.proj_head.parameters():
                p.requires_grad = True
        else:
            all_params = params_to_update
            for p in self.proj_head.parameters():
                p.requires_grad = False

        # If no params to update (strategy=none and proj_head frozen), skip
        if len(all_params) == 0:
            return

        optimizer = optim.Adam(all_params, lr=self.tta_lr)

        # Reset adaptive rank counters if present
        for loss_fn in self.losses:
            if hasattr(loss_fn, "reset_counters"):
                loss_fn.reset_counters()

        # --- TTA optimization loop ---
        self.qformer.train()
        if not self.freeze_proj_head:
            self.proj_head.train()

        for step in range(self.tta_steps):
            optimizer.zero_grad()

            # Forward pass
            mm_embeds = self.qformer.forward_qformer(image_embeds, prompts, descs)
            proj_feats = self.proj_head(mm_embeds)

            # Build context
            ctx = {
                "proj_feats": proj_feats,
                "predictions": init_preds,
                "device": self.device,
            }

            # Add augmented features if needed
            if embeds_weak is not None:
                mm_weak = self.qformer.forward_qformer(embeds_weak, prompts, descs)
                mm_strong = self.qformer.forward_qformer(embeds_strong, prompts, descs)
                ctx["proj_feats_weak"] = self.proj_head(mm_weak)
                ctx["proj_feats_strong"] = self.proj_head(mm_strong)

            if vgg_feats is not None:
                ctx["vgg_feats"] = vgg_feats

            # Compute total loss
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            for loss_fn in self.losses:
                l = loss_fn(ctx)
                total_loss = total_loss + l

            if total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()

        self.qformer.eval()
        self.proj_head.eval()
