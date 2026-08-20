
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''

import argparse
import csv
import yaml
import shutil
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

from helpers_cb import (
    CoastalBenchNCDataset,
    DEFAULT_CB_VARS,
    DEFAULT_CB_NAMES,
    build_sparse_condition,
    build_sparse_condition_cb,
    collate_cb,
)
from visualize_cb import visualize_reconstruction_cb_batch
from Model import (
    ConditionalPointFFM, 
    ConditionalPointMLPRBF, 
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
    FNO,
    FNOFFM,
    )
from direct_coherence_loss import (
    DirectCoherenceConfig,
    DirectGlobalCoherenceLoss,
    apply_two_objective_update,
    differentiable_rf_rollout,
    sample_coherence_points,
)
from model_finetune import find_source_run_dir, load_source_config
from coherence.cross_spectral import compute_physical_coherence_loss, CrossSpectralConfig
from cross_spectral_coherence import compute_graph_basis

def _bands_from_basis(basis) -> Dict[str, list]:
    """
    Convert a ``GraphSpectralBasis`` into the band-index dict expected by
    ``compute_physical_coherence_loss``.

    ``basis.band_of_mode`` encodes:
        -1 = DC mode (excluded from all bands)
         0 = large-scale band
         1 = medium-scale band
         2 = small-scale band
    """
    band_labels = {0: "large", 1: "medium", 2: "small"}
    bands: Dict[str, list] = {}
    for band_id, band_name in band_labels.items():
        indices = [int(i) for i, b in enumerate(basis.band_of_mode) if b == band_id]
        if indices:
            bands[band_name] = indices
    return bands


def _pcgrad_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    L_data: torch.Tensor,
    L_coh: torch.Tensor,
    coh_weight: float,
    max_norm: float = 1.0,
) -> None:
    """
    PCGrad gradient surgery (two backward passes).

    For each parameter where the coherence gradient conflicts with the data
    gradient (negative dot product), the coherence gradient is projected onto
    the normal plane of the data gradient before combining.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)
    L_data.backward(retain_graph=True)
    grads_data = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                  for p in params]

    optimizer.zero_grad(set_to_none=True)
    (coh_weight * L_coh).backward()
    for p, g_d in zip(params, grads_data):
        g_c = p.grad if p.grad is not None else torch.zeros_like(p)
        dot = (g_d * g_c).sum()
        if dot < 0:
            g_c = g_c - dot / (g_d.norm() ** 2 + 1e-12) * g_d
        p.grad = g_d + g_c

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()


def _gradnorm_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    L_data: torch.Tensor,
    L_coh: torch.Tensor,
    coh_weight: float,
    max_norm: float = 1.0,
) -> None:
    """
    Gradient-norm balancing (Frank-Wolfe / GradNorm style).

    Each loss is scaled by the inverse of its gradient L2-norm so that neither
    loss dominates purely by magnitude.  The combined gradient is:

        g = g_data / ||g_data|| + coh_weight * g_coh / ||g_coh||
    """
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad(set_to_none=True)
    L_data.backward(retain_graph=True)
    grads_data = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                  for p in params]
    norm_data = sum((g ** 2).sum() for g in grads_data).sqrt().clamp_min(1e-12)

    optimizer.zero_grad(set_to_none=True)
    L_coh.backward()
    grads_coh = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                 for p in params]
    norm_coh = sum((g ** 2).sum() for g in grads_coh).sqrt().clamp_min(1e-12)

    for p, g_d, g_c in zip(params, grads_data, grads_coh):
        p.grad = g_d / norm_data + coh_weight * g_c / norm_coh

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()

def parse_args():

    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str,
                   default=None)
    p.add_argument("--save-dir", type=str,
                   default=f"Save_TrainedModel/ffm_cb_pointcloud")
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")
    p.add_argument("--training-mode", dest="training_mode", type=str, default="standard",
                   choices=["standard", "direct_coherence"])
    p.add_argument("--initialization", type=str, default="scratch", choices=["scratch", "pretrained"])
    p.add_argument("--pretrained-run-dir", dest="pretrained_run_dir", type=str, default=None)
    p.add_argument("--pretrained-source-Demo-Num", dest="pretrained_source_Demo_Num", type=int, default=None)
    p.add_argument("--pretrained-checkpoint", dest="pretrained_checkpoint", type=str, default="best")
    p.add_argument("--pretrained-load-optimizer", dest="pretrained_load_optimizer", action="store_true")
    p.add_argument("--no-pretrained-load-optimizer", dest="pretrained_load_optimizer", action="store_false")
    p.set_defaults(pretrained_load_optimizer=False)
    p.add_argument("--pretrained-strict", dest="pretrained_strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained-use-source-base-config", dest="pretrained_use_source_base_config",
                   action=argparse.BooleanOptionalAction, default=True)
    
    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "GL_rbf", "GL_rbf_ENH"], 
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--train-ratio-downsample", dest="train_ratio_downsample", type=float, default=1.0)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ------------------------------
    # These are hyperparameters for mlp_rbf backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--USE-FOURIER-PE", "--USE_FOURIER_PE", dest="USE_FOURIER_PE", action="store_true",
                   help="If set, feed Fourier positional coordinate features to point_encoder.")
    p.add_argument("--fourier-pe-num-bands", type=int, default=32,
                   help="Number of frequency bands for Fourier positional coordinate encoding.")
    p.add_argument("--fourier-pe-max-freq", type=float, default=64.0,
                   help="Maximum frequency scale for Fourier positional coordinate encoding.")

    # ------------------------------
    # These are hyperparameters for Perceiver backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--latent-dim", type=int, default=256, 
                   help="Token / latent width for the Perceiver backbone.",)
    p.add_argument("--num-latents", type=int, default=128, 
                   help="Number of learned latent slots in the Perceiver.",)
    p.add_argument("--num-heads", type=int, default=8, 
                   help="Number of attention heads for Perceiver attention blocks.",)
    p.add_argument("--num-latent-blocks", type=int, default=4, 
                   help="Number of latent self-attention blocks.",)
    p.add_argument("--ff-mult", type=int, default=4, 
                   help="Expansion factor for Transformer feed-forward layers.",)
    p.add_argument("--attn-dropout", type=float, default=0.0, 
                   help="Dropout used inside attention layers.",)
    p.add_argument("--mlp-dropout", type=float, default=0.0, 
                   help="Dropout used inside token projection / FFN layers.",)
    p.add_argument("--decode-chunk-size", type=int, default=4096,
                   help="Chunk size for Perceiver output decoding. Useful for full-resolution reconstruction.",)
    p.add_argument("--share-query-proj", action="store_true",
        help="If set, use the same projection for Perceiver encoder query tokens and decoder query tokens.",)

    p.add_argument("--summary-type", type=str, default='cls',
        help="Only for GL_rbf; select either cls or mean",)

    # ----------------------------------------------------------
    # Hybrid local-global gather options
    # ----------------------------------------------------------
    p.add_argument(
        "--gather-mode", type=str, default="rbf", choices=["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"],
        help="Gather mode used by ConditionalPointHybridLocalGlobalRBF. 'rbf' preserves the current full gather as default.",
    )
    p.add_argument(
        "--gather-topk", type=int, default=32, 
        help="Number of nearest refined sensor tokens used in top-k gather modes.",
    )
    p.add_argument(
        "--gather-query-chunk-size", type=int, default=None,
        help="Optional query chunk size for memory-friendly gathering. Applies to all gather modes.",
    )
    p.add_argument(
        "--learnable-rbf-sigma", action="store_true",
        help="If set, make the RBF sigma in the hybrid gather learnable.",
    )
    p.add_argument(
        "--neighbor-backend", type=str, default="torch", choices=["auto", "torch", "keops"],
        help="Neighbor / kernel backend for the hybrid gather. "
            "'auto' uses KeOps if available, otherwise falls back to pure PyTorch.",)
    p.add_argument(
        "--sensor-local-topk", type=int, default=8,
        help="Number of local sensor neighbors used by the sensor-side Point-Transformer refinement in gather_mode='topk_rbf_ptlocal'.",)
    p.add_argument(
        "--sensor-local-dropout", type=float, default=0.0,
        help="Dropout used inside the sensor-side local refinement block for gather_mode='topk_rbf_ptlocal'.",
    )
    p.add_argument("--sensor-coord-encoding", type=str, default=None,
                   choices=["raw", "fourier"],
                   help="Sensor coordinate encoding for GL_rbf/GL_rbf_ENH. "
                        "Use 'fourier' to give sensors the same coordinate features as queries.")
    p.add_argument("--latent-sensor-reinject", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, latents periodically re-attend to sparse sensor tokens.")
    p.add_argument("--latent-reinject-every", type=int, default=1,
                   help="Re-inject sensor information every N latent blocks when latent_sensor_reinject is enabled.")
    p.add_argument("--query-latent-readout", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, each query reads global context from latent memory before the final head.")
    p.add_argument("--query-readout-type", type=str, default=None,
                   choices=["point", "coord"],
                   help="'coord' uses Senseiver-style coordinate decoder tokens; "
                        "'point' uses the current flow-state point features.")
    p.add_argument("--query-readout-scale-init", type=float, default=None,
                   help="Initial scale for query-to-latent readout. "
                        "Use small positive values such as 1e-2 for GL_rbf_ENH.")
    p.add_argument("--enhanced-head-norm", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, apply LayerNorm to the fused [query, global, local] head input.")
    p.add_argument("--glres-scale-init", type=float, default=None,
                   help="Initial scale for topk_rbf_glres residual terms: sensor importance and coarse scaffold.")

    # ----------------------------------------------------------
    # These are hyperparameters for fno backbone
    # Num_x / Num_y must be supplied for the FNO baseline.
    # ----------------------------------------------------------
    p.add_argument( "--Num-x", dest="Num_x", type=int, default=None,
        help="Number of grid points along x for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=None,
        help="Number of grid points along y for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument( "--fno-modes-x", type=int, default=32,
        help="Number of retained Fourier modes along x for the FNO baseline.",)
    p.add_argument( "--fno-modes-y", type=int, default=8,
        help="Number of retained Fourier modes along y for the FNO baseline.",)
    p.add_argument( "--fno-hidden-channels", type=int, default=64,
        help="Hidden channel width of the neuraloperator FNO baseline.",)
    p.add_argument( "--fno-n-layers", type=int, default=4,
        help="Number of Fourier layers in the FNO baseline.",)
    p.add_argument(
        "--condition-blur",
        action="store_true",
        help="If set, Gaussian-splat sparse FNO conditioning maps before concatenation.",
    )
    p.add_argument(
        "--condition-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used to splat sparse FNO conditioning maps.",
    )
    p.add_argument(
        "--condition-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to splat sparse FNO conditioning maps.",
    )

    # ------------------------------
    # These are hyperparameters for training process
    # ------------------------------
    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--query-sampling", type=str, default="uniform", choices=["uniform", "obs_mix"])
    p.add_argument("--query-sample-near-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-far-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-sigma-ratio", type=float, default=0.05)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--sigma-min", type=float, default=1e-4) # backward-compatible old args

    p.add_argument("--cond-field", type=int, default=2, help="Legacy single conditioned field.")
    p.add_argument("--n-obs-min", type=int, default=64, help="Legacy single-field minimum sensors.")
    p.add_argument("--n-obs-max", type=int, default=256, help="Legacy single-field maximum sensors.")

    # generalized args
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Conditioned field ids, e.g. --cond-fields 0 2")
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None,
                   help="Per-field minimum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None,
                   help="Per-field maximum sensors. Length 1 broadcasts to all cond_fields.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Visualization conditioned fields. Defaults to cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Visualization exact sensors per field. Defaults to n_obs_max_list.")
    
    # ODE solver used at generation time. For 1-RF, Euler is the main benchmark because the method is designed for coarse-step sampling.
    p.add_argument(
        "--ode-solver", type=str, default="euler",
            choices=["euler", "heun"], help="ODE solver for generation. Use Euler for the main 1-RF benchmark; Heun is optional.")
    # Reconstruction benchmark step counts. These are the NFEs to compare after moving to 1-RF.
    p.add_argument(
        "--benchmark-n-steps", type=int, nargs="+", default=[2, 4, 8, 16],
            help="Sampling step counts used for reconstruction benchmarking.")

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=32)

    # ----------------------------------------------------------
    # Direct coherence training / post-training
    # ----------------------------------------------------------
    p.add_argument("--direct-coherence-enabled", dest="direct_coherence_enabled",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--data-loss-weight", dest="data_loss_weight", type=float, default=1.0)
    p.add_argument("--coherence-loss-weight", dest="coherence_loss_weight", type=float, default=0.1)
    p.add_argument("--coherence-start-epoch", dest="coherence_start_epoch", type=int, default=1)
    p.add_argument("--coherence-every-n-steps", dest="coherence_every_n_steps", type=int, default=1)
    p.add_argument("--coherence-interval-rescale", dest="coherence_interval_rescale",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--coherence-batch-size", dest="coherence_batch_size", type=int, default=4)
    p.add_argument("--coherence-downsample", dest="coherence_downsample",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-n-points", dest="coherence_n_points", type=int, default=4096)
    p.add_argument("--coherence-rollout-steps", dest="coherence_rollout_steps", type=int, default=2)
    p.add_argument("--coherence-rollout-solver", dest="coherence_rollout_solver", type=str, default="euler",
                   choices=["euler", "heun"])
    p.add_argument("--coherence-obs-consistency-mode", dest="coherence_obs_consistency_mode",
                   type=str, default="endpoint_smooth",
                   choices=["none", "default_hard", "endpoint", "endpoint_smooth"])
    p.add_argument("--coherence-obs-consistency-strength", dest="coherence_obs_consistency_strength",
                   type=float, default=1.0)
    p.add_argument("--coherence-obs-consistency-sigma", dest="coherence_obs_consistency_sigma",
                   type=float, default=0.05)
    p.add_argument("--coherence-obs-consistency-schedule-power",
                   dest="coherence_obs_consistency_schedule_power", type=float, default=2.0)
    p.add_argument("--coherence-obs-consistency-final-clamp",
                   dest="coherence_obs_consistency_final_clamp",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-self-weight", dest="coherence_self_weight", type=float, default=1.0)
    p.add_argument("--coherence-mutual-weight", dest="coherence_mutual_weight", type=float, default=0.10)
    p.add_argument("--coherence-cross-weight", dest="coherence_cross_weight", type=float, default=1.0)
    p.add_argument("--coherence-channel-weights", dest="coherence_channel_weights",
                   type=float, nargs="+", default=None)
    p.add_argument("--coherence-cross-num-directions", dest="coherence_cross_num_directions",
                   type=int, default=32)
    p.add_argument("--coherence-cross-top-frac", dest="coherence_cross_top_frac", type=float, default=0.10)
    p.add_argument("--coherence-cross-seed", dest="coherence_cross_seed", type=int, default=1234)
    p.add_argument("--coherence-cross-include-axes", dest="coherence_cross_include_axes",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-cross-qmc", dest="coherence_cross_qmc",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-mutual-num-directions", dest="coherence_mutual_num_directions",
                   type=int, default=16)
    p.add_argument("--coherence-mutual-seed", dest="coherence_mutual_seed", type=int, default=1234)
    p.add_argument("--coherence-use-denorm", dest="coherence_use_denorm",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-balance-mode", dest="gradient_balance_mode", type=str,
                   default="weighted_sum", choices=["weighted_sum", "config"])
    p.add_argument("--config-missing-behavior", dest="config_missing_behavior", type=str,
                   default="error", choices=["error", "weighted_sum"])
    p.add_argument("--config-data-grad-scale", dest="config_data_grad_scale", type=float, default=1.0)
    p.add_argument("--config-coherence-grad-scale", dest="config_coherence_grad_scale",
                   type=float, default=1.0)
    p.add_argument("--coherence-weight-warmup-epochs", dest="coherence_weight_warmup_epochs",
                   type=int, default=0)
    p.add_argument("--xspectral-weight", dest="xspectral_weight", type=float, default=0.0,
                   help="Weight of the cross-spectral GFT coherence term in direct post-training. "
                        "0.0 disables it (default). Try 0.1-0.5 for initial experiments.")
    p.add_argument("--xspectral-n-modes", dest="xspectral_n_modes", type=int, default=256,
                   help="Number of graph-Laplacian eigenmodes for the post-training GFT basis. "
                        "Must be << coherence_n_points.")
    p.add_argument("--xspectral-k-neighbors", dest="xspectral_k_neighbors", type=int, default=16,
                   help="k-NN graph connectivity for the post-training GFT graph Laplacian.")
    p.add_argument("--xspectral-eta-crossfreq", dest="xspectral_eta_crossfreq", type=float, default=1.0,
                   help="Weight of the cross-frequency coupling term inside the cross-spectral loss.")

    # ----------------------------------------------------------
    # Cross-spectral coherence auxiliary loss
    # ----------------------------------------------------------
    p.add_argument(
        "--coh-loss", action="store_true",
        help="Enable cross-spectral coherence auxiliary loss during training.",
    )
    p.add_argument(
        "--coh-n-steps", type=int, default=4,
        help="Number of Euler steps in the differentiable rollout used to compute coherence loss.",
    )
    p.add_argument(
        "--coh-truncate-bptt", type=int, default=None,
        help="Only backprop through the last N rollout steps (truncated BPTT). "
             "Reduces activation memory. Default: None means full BPTT.",
    )
    p.add_argument(
        "--coh-n-modes", type=int, default=256,
        help="Number of graph-Laplacian eigenmodes used for the GFT basis.",
    )
    p.add_argument(
        "--coh-k-neighbors", type=int, default=16,
        help="k-NN graph connectivity for building the graph Laplacian.",
    )
    p.add_argument(
        "--coh-lambda", type=float, default=1.0,
        help="Maximum (final) coherence loss weight. Combined loss = L_data + w * L_coh.",
    )
    p.add_argument(
        "--coh-ramp", action="store_true",
        help="Linearly ramp the coherence weight from 0 at epoch 0 to --coh-lambda "
             "at --coh-warmup-epochs.",
    )
    p.add_argument(
        "--coh-warmup-epochs", type=int, default=None,
        help="Number of epochs over which the coherence weight ramps to --coh-lambda. "
             "Defaults to --epochs (full-training ramp).",
    )
    p.add_argument(
        "--coh-every-n-batches", type=int, default=1,
        help="Compute coherence loss only every N batches. "
             "Set > 1 to reduce the per-epoch overhead (coherence is skipped on other batches).",
    )
    p.add_argument(
        "--coh-eta-crossfreq", type=float, default=1.0,
        help="Weight of the cross-frequency coupling term inside the coherence loss (eta in CrossSpectralConfig).",
    )
    p.add_argument(
        "--coh-n-query-points", type=int, default=None,
        help="Number of spatial points to randomly subsample from the full field before running the "
             "differentiable coherence rollout.  Reduces GPU memory proportionally.  "
             "None (default) uses all N points.  Recommended: 8192 for large point clouds.",
    )
    p.add_argument(
        "--grad-proj", type=str, default="none",
        choices=["none", "sequential", "pcgrad", "gradnorm"],
        help="Gradient projection / backward strategy for combining data and coherence losses. "
             "'none': single combined backward (both graphs in VRAM simultaneously — use only for debugging). "
             "'sequential': data backward first (frees data graph), then coherence rollout + backward "
             "(recommended when --coh-loss is enabled — peak VRAM = max of the two graphs, not their sum). "
             "'pcgrad': project conflicting coherence gradients onto normal plane of data gradient (two backward passes). "
             "'gradnorm': scale each loss by inverse gradient L2-norm before combining.",
    )

    # --- CoastalBench-specific arguments ---
    p.add_argument("--static-nc", type=str, default=None,
                   help="Path to static.nc for CoastalBench (provides lon/lat/mask_rho).")
    p.add_argument("--cb-var-names", type=str, nargs="+", default=None,
                   help="NC variable names to load. Defaults to DEFAULT_CB_VARS.")
    p.add_argument("--cb-field-names", type=str, nargs="+", default=None,
                   help="Human-readable labels matching --cb-var-names.")
    p.add_argument("--cb-var-level", type=int, default=11,
                   help="Vertical s-index for 3D vars. ROMS: 0=bottom, 11=surface.")
    p.add_argument("--cb-use-wetdry-mask", action="store_true",
                   help="Exclude points that are ever dry across training days.")

    return p.parse_args(), p

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_conditioning_args(args):
    # training
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    # visualization
    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)

    return args

class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def collate_snapshots(batch):
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }


def sample_query_subset(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_query: Optional[int],
    mode: str = "uniform",
    obs_coords: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.Tensor] = None,
    near_ratio: float = 0.25,
    far_ratio: float = 0.25,
    sigma_ratio: float = 0.05,
):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None

    bsz, n_pts, coord_dim = coords.shape
    n_query = int(n_query)

    def take_weighted(weights: torch.Tensor, count: int, selected: torch.Tensor) -> torch.Tensor:
        count = min(int(count), int((~selected).sum().item()))
        if count <= 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)

        weights = weights.to(dtype=coords.dtype).clamp_min(0.0)
        weights = weights.masked_fill(selected, 0.0)
        pieces = []

        positive = weights > 0
        if positive.any():
            n_weighted = min(count, int(positive.sum().item()))
            sampled = torch.multinomial(weights, num_samples=n_weighted, replacement=False)
            pieces.append(sampled)
            selected[sampled] = True
            count -= n_weighted

        if count > 0:
            available = (~selected).nonzero(as_tuple=False).squeeze(-1)
            fill = available[torch.randperm(available.numel(), device=coords.device)[:count]]
            pieces.append(fill)
            selected[fill] = True

        return torch.cat(pieces, dim=0) if pieces else torch.empty(0, device=coords.device, dtype=torch.long)

    all_idx = []
    for b in range(bsz):
        if mode == "obs_mix" and obs_coords is not None and obs_mask is not None:
            valid = obs_mask[b].bool()
        else:
            valid = None

        if mode != "obs_mix" or valid is None or not valid.any():
            idx = torch.randperm(n_pts, device=coords.device)[:n_query].sort().values
            all_idx.append(idx)
            continue

        d_min = torch.cdist(coords[b:b + 1], obs_coords[b, valid].unsqueeze(0), p=2.0).squeeze(0).amin(dim=-1)
        bbox_diag = (coords[b].amax(dim=0) - coords[b].amin(dim=0)).norm().clamp_min(1e-6)
        sigma = (sigma_ratio * bbox_diag).clamp_min(1e-6)

        near_count = min(n_query, max(0, int(round(n_query * near_ratio))))
        far_count = min(n_query - near_count, max(0, int(round(n_query * far_ratio))))
        uniform_count = n_query - near_count - far_count

        selected = torch.zeros(n_pts, device=coords.device, dtype=torch.bool)
        near_weights = torch.exp(-(d_min ** 2) / (2 * sigma ** 2 + 1e-12))
        far_weights = d_min.clamp_min(0.0)

        pieces = [
            take_weighted(near_weights, near_count, selected),
            take_weighted(far_weights, far_count, selected),
            take_weighted(torch.ones(n_pts, device=coords.device, dtype=coords.dtype), uniform_count, selected),
        ]
        if int(selected.sum().item()) < n_query:
            pieces.append(
                take_weighted(
                    torch.ones(n_pts, device=coords.device, dtype=coords.dtype),
                    n_query - int(selected.sum().item()),
                    selected,
                )
            )

        idx = torch.cat([p for p in pieces if p.numel() > 0], dim=0).sort().values
        all_idx.append(idx)

    idx = torch.stack(all_idx, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, dim=1, index=coord_idx), torch.gather(fields, dim=1, index=field_idx), idx

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str = "uniform",
    query_sample_near_ratio: float = 0.25,
    query_sample_far_ratio: float = 0.25,
    query_sample_sigma_ratio: float = 0.05,
    epoch: int = 0,
    # --- cross-spectral coherence ---
    U_gft: Optional[torch.Tensor] = None,
    coh_bands: Optional[Dict[str, list]] = None,
    coh_weight: float = 0.0,
    coh_n_steps: int = 4,
    coh_truncate_bptt: Optional[int] = None,
    coh_every_n_batches: int = 1,
    coh_cfg: Optional[CrossSpectralConfig] = None,
    coh_n_query_points: Optional[int] = None,
    grad_proj: str = "none",
) -> Tuple[float, Dict[str, float]]:
    """
    Run one epoch of training or evaluation.

    Returns:
        mean_data_loss: average per-batch data (velocity MSE) loss.
        metrics: dict with optional 'coh_loss', 'total_loss' averages.
    """
    training = optimizer is not None
    model.train(training)

    use_coherence = (
        training
        and coh_weight > 0.0
        and U_gft is not None
        and coh_bands is not None
        and hasattr(model, "differentiable_rollout")
    )

    total_data        = 0.0
    total_coh         = 0.0
    total_loss        = 0.0
    total_L_same      = 0.0
    total_L_crossfreq = 0.0
    count             = 0

    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch_idx, batch in enumerate(pbar):
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)

        # Build generalized sparse observations.
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )

        # for models that must operate on the full regular grid like FNO,
        # point subsampling will be disabled.
        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        sampling_mode = query_sampling if training else "uniform"
        coords_q, fields_q, _ = sample_query_subset(
            coords=coords_full,
            fields=fields_full,
            n_query=effective_n_query,
            mode=sampling_mode,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )

        L_data, _ = model.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        # --- cross-spectral coherence loss (non-sequential modes) ---
        # In "sequential" mode the rollout is deferred until after the data
        # backward so the two computation graphs never coexist in VRAM.
        compute_coh_this_batch = (
            use_coherence and (batch_idx % coh_every_n_batches == 0)
        )
        L_coh           = None
        L_same_val      = 0.0
        L_crossfreq_val = 0.0

        if compute_coh_this_batch and grad_proj != "sequential":
            # Optionally subsample spatial points to reduce GPU memory.
            # The GFT is a linear projection so spectral statistics estimated
            # from a large random subset are essentially identical to using all N
            # points, as long as the subset is much larger than K (n_modes).
            N_full = coords_full.shape[1]
            if coh_n_query_points is not None and coh_n_query_points < N_full:
                coh_idx    = torch.randperm(N_full, device=device)[:coh_n_query_points]
                coords_coh = coords_full[:, coh_idx, :]
                fields_coh = fields_full[:, coh_idx, :]
                U_coh      = U_gft[coh_idx, :]          # [coh_n_query_points, K]
            else:
                coords_coh = coords_full
                fields_coh = fields_full
                U_coh      = U_gft

            x_rollout = model.differentiable_rollout(
                coords=coords_coh,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_steps=coh_n_steps,
                truncate_bptt=coh_truncate_bptt,
            )
            coh_out = compute_physical_coherence_loss(
                fields_pred=x_rollout,
                fields_target=fields_coh,
                U=U_coh,
                bands=coh_bands,
                cfg=coh_cfg,
            )
            L_coh           = coh_out["loss"]
            L_same_val      = float(coh_out["L_same"].detach().cpu())
            L_crossfreq_val = float(coh_out["L_crossfreq"].detach().cpu())

        # --- backward / optimizer step ---
        if training:
            if compute_coh_this_batch and grad_proj == "sequential":
                # Sequential backward: free the data graph first, giving full
                # VRAM headroom for the coherence rollout graph.
                # Calling backward() twice without zero_grad() in between
                # additively accumulates gradients — mathematically equivalent
                # to a single combined backward at a fraction of the peak VRAM.
                optimizer.zero_grad(set_to_none=True)
                L_data.backward()                       # data graph freed here

                # Now build and backward the coherence graph in the freed space.
                N_full = coords_full.shape[1]
                if coh_n_query_points is not None and coh_n_query_points < N_full:
                    coh_idx    = torch.randperm(N_full, device=device)[:coh_n_query_points]
                    coords_coh = coords_full[:, coh_idx, :]
                    fields_coh = fields_full[:, coh_idx, :]
                    U_coh      = U_gft[coh_idx, :]
                else:
                    coords_coh = coords_full
                    fields_coh = fields_full
                    U_coh      = U_gft

                x_rollout = model.differentiable_rollout(
                    coords=coords_coh,
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    n_steps=coh_n_steps,
                    truncate_bptt=coh_truncate_bptt,
                )
                coh_out = compute_physical_coherence_loss(
                    fields_pred=x_rollout,
                    fields_target=fields_coh,
                    U=U_coh,
                    bands=coh_bands,
                    cfg=coh_cfg,
                )
                L_coh           = coh_out["loss"]
                L_same_val      = float(coh_out["L_same"].detach().cpu())
                L_crossfreq_val = float(coh_out["L_crossfreq"].detach().cpu())

                (coh_weight * L_coh).backward()         # coherence graph freed here
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            elif L_coh is not None and grad_proj == "pcgrad":
                _pcgrad_step(model, optimizer, L_data, L_coh, coh_weight)
            elif L_coh is not None and grad_proj == "gradnorm":
                _gradnorm_step(model, optimizer, L_data, L_coh, coh_weight)
            else:
                L_total = L_data + (coh_weight * L_coh if L_coh is not None else 0.0)
                optimizer.zero_grad(set_to_none=True)
                L_total.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # --- bookkeeping ---
        data_val = float(L_data.detach().cpu())
        coh_val  = float(L_coh.detach().cpu()) if L_coh is not None else 0.0
        tot_val  = data_val + coh_weight * coh_val
        total_data += data_val
        total_coh  += coh_val
        total_loss += tot_val
        total_L_same      += L_same_val
        total_L_crossfreq += L_crossfreq_val
        count += 1

        postfix = f"loss={data_val:.4e}"
        if L_coh is not None:
            postfix += f" coh={coh_val:.4e} w={coh_weight:.3f}"
        pbar.set_postfix_str(postfix)

    n = max(count, 1)
    metrics = {
        "data_loss":   total_data        / n,
        "coh_loss":    total_coh         / n,
        "total_loss":  total_loss        / n,
        "L_same":      total_L_same      / n,
        "L_crossfreq": total_L_crossfreq / n,
    }
    return total_data / n, metrics


def build_direct_coherence_config(args) -> DirectCoherenceConfig:
    return DirectCoherenceConfig(
        enabled=bool(args.direct_coherence_enabled),
        self_weight=float(args.coherence_self_weight),
        mutual_weight=float(args.coherence_mutual_weight),
        cross_weight=float(args.coherence_cross_weight),
        channel_weights=args.coherence_channel_weights,
        cross_num_directions=int(args.coherence_cross_num_directions),
        cross_top_frac=float(args.coherence_cross_top_frac),
        cross_seed=int(args.coherence_cross_seed),
        cross_include_axes=bool(args.coherence_cross_include_axes),
        cross_qmc=bool(args.coherence_cross_qmc),
        mutual_num_directions=int(args.coherence_mutual_num_directions),
        mutual_seed=int(args.coherence_mutual_seed),
        use_denorm=bool(args.coherence_use_denorm),
        xspectral_weight=float(getattr(args, "xspectral_weight", 0.0)),
        xspectral_n_modes=int(getattr(args, "xspectral_n_modes", 256)),
        xspectral_k_neighbors=int(getattr(args, "xspectral_k_neighbors", 16)),
        xspectral_eta_crossfreq=float(getattr(args, "xspectral_eta_crossfreq", 1.0)),
    )


def current_coherence_loss_weight(args, epoch: int) -> float:
    base = float(args.coherence_loss_weight)
    warmup = int(getattr(args, "coherence_weight_warmup_epochs", 0) or 0)
    if warmup <= 0 or epoch < int(args.coherence_start_epoch):
        return base
    progress = (epoch - int(args.coherence_start_epoch) + 1) / float(warmup)
    return base * min(max(progress, 0.0), 1.0)


def build_epoch_train_loader(train_set, args, epoch: int) -> DataLoader:
    """
    Build a fresh random subset loader for one epoch.

    `train_ratio` still defines the train/validation split.  This helper only
    down-samples the already-built training split for the current epoch.
    """
    ratio = min(max(float(getattr(args, "train_ratio_downsample", 1.0)), 0.0), 1.0)
    n_total = len(train_set)
    n_epoch = n_total if ratio >= 1.0 else max(1, int(math.ceil(n_total * ratio)))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + int(epoch) * 1009)
    if n_epoch < n_total:
        indices = torch.randperm(n_total, generator=generator)[:n_epoch].tolist()
        epoch_set = Subset(train_set, indices)
    else:
        epoch_set = train_set
    return DataLoader(
        epoch_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cb,
    )


def _grad_norm(model: nn.Module) -> float:
    grads = [
        p.grad.detach().reshape(-1)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return float("nan")
    vec = torch.cat(grads)
    return float(torch.linalg.vector_norm(vec).detach().cpu())


def _mean_metric(rows: list[dict], key: str) -> float:
    vals = [row[key] for row in rows if key in row and row[key] is not None and math.isfinite(float(row[key]))]
    if not vals:
        return float("nan")
    return float(sum(float(v) for v in vals) / len(vals))


def run_epoch_direct_coherence(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str,
    query_sample_near_ratio: float,
    query_sample_far_ratio: float,
    query_sample_sigma_ratio: float,
    direct_cfg: DirectCoherenceConfig,
    args,
    global_step: int,
    epoch: int,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    U_gft: Optional[torch.Tensor] = None,
    coh_bands: Optional[Dict[str, list]] = None,
) -> tuple[dict, int]:
    model.train(True)
    loss_module = DirectGlobalCoherenceLoss(direct_cfg)
    rows = []
    applied = 0
    conflict_count = 0
    mode_str = "DirectCoherence"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )

        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        coords_q, fields_q, _ = sample_query_subset(
            coords=coords_full,
            fields=fields_full,
            n_query=effective_n_query,
            mode=query_sampling,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )
        data_loss, _ = model.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        global_step += 1
        every = max(1, int(args.coherence_every_n_steps))
        coherence_weight = current_coherence_loss_weight(args, epoch)
        coherence_active = (
            bool(direct_cfg.enabled)
            and int(epoch) >= int(args.coherence_start_epoch)
            and global_step % every == 0
            and coherence_weight > 0.0
        )

        row = {
            "data_loss": float(data_loss.detach().cpu()),
            "coherence_loss": float("nan"),
            "coherence_self": float("nan"),
            "coherence_mutual": float("nan"),
            "coherence_cross": float("nan"),
            "xspectral_loss": float("nan"),
            "xspectral_L_same": float("nan"),
            "xspectral_L_crossfreq": float("nan"),
            "coherence_applied": 0.0,
            "data_grad_norm": float("nan"),
            "coherence_grad_norm": float("nan"),
            "gradient_cosine": float("nan"),
            "gradient_conflict": 0.0,
            "global_step": float(global_step),
        }

        if not coherence_active:
            optimizer.zero_grad(set_to_none=True)
            total_loss = float(args.data_loss_weight) * data_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            row["data_grad_norm"] = _grad_norm(model)
            optimizer.step()
            row["total_loss"] = float(total_loss.detach().cpu())
        else:
            bsz = coords_full.shape[0]
            cbsz = min(max(1, int(args.coherence_batch_size)), bsz)
            sel = torch.randperm(bsz, device=device)[:cbsz].sort().values

            coords_c = coords_full.index_select(0, sel)
            fields_c = fields_full.index_select(0, sel)
            obs_coords_c = obs_coords.index_select(0, sel)
            obs_values_c = obs_values.index_select(0, sel)
            obs_mask_c = obs_mask.index_select(0, sel)
            obs_field_ids_c = obs_field_ids.index_select(0, sel)
            obs_indices_c = obs_indices.index_select(0, sel) if obs_indices is not None else None

            if getattr(model, "requires_full_grid", False) or not bool(args.coherence_downsample):
                coords_coh, fields_ref, coh_idx = coords_c, fields_c, None
            else:
                coords_coh, fields_ref, coh_idx = sample_coherence_points(
                    coords=coords_c,
                    fields=fields_c,
                    n_points=args.coherence_n_points,
                )
            # Subset GFT basis to match sampled spatial points
            U_coh = U_gft[coh_idx, :] if (U_gft is not None and coh_idx is not None) else U_gft

            x_gen = differentiable_rf_rollout(
                ffm_model=model,
                coords=coords_coh,
                obs_coords=obs_coords_c,
                obs_values=obs_values_c,
                obs_mask=obs_mask_c,
                obs_field_ids=obs_field_ids_c,
                obs_indices=obs_indices_c,
                n_steps=args.coherence_rollout_steps,
                ode_solver=args.coherence_rollout_solver,
                obs_consistency_mode=args.coherence_obs_consistency_mode,
                obs_consistency_strength=args.coherence_obs_consistency_strength,
                obs_consistency_sigma=args.coherence_obs_consistency_sigma,
                obs_consistency_schedule_power=args.coherence_obs_consistency_schedule_power,
                obs_consistency_final_clamp=args.coherence_obs_consistency_final_clamp,
            )
            coherence_loss_raw, components = loss_module(
                x_gen=x_gen,
                x_ref=fields_ref,
                mean=mean,
                std=std,
                U_gft=U_coh,
                coh_bands=coh_bands,
            )
            coherence_loss_for_update = coherence_loss_raw
            if bool(args.coherence_interval_rescale):
                coherence_loss_for_update = coherence_loss_for_update * every

            grad_info = apply_two_objective_update(
                model=model,
                optimizer=optimizer,
                data_loss=data_loss,
                coherence_loss=coherence_loss_for_update,
                mode=args.gradient_balance_mode,
                data_weight=float(args.data_loss_weight) if args.gradient_balance_mode == "weighted_sum" else float(args.config_data_grad_scale),
                coherence_weight=coherence_weight if args.gradient_balance_mode == "weighted_sum" else float(args.config_coherence_grad_scale),
                grad_clip_norm=1.0,
                config_missing_behavior=args.config_missing_behavior,
            )
            applied += 1
            if bool(grad_info.get("gradient_conflict", False)):
                conflict_count += 1
            total_for_log = float(args.data_loss_weight) * data_loss.detach() + coherence_weight * coherence_loss_for_update.detach()
            row.update({
                "total_loss": float(total_for_log.cpu()),
                "coherence_loss": float(coherence_loss_raw.detach().cpu()),
                "coherence_self": float(components["self_loss"].detach().cpu()),
                "coherence_mutual": float(components["mutual_loss"].detach().cpu()),
                "coherence_cross": float(components["cross_loss"].detach().cpu()),
                "xspectral_loss": float(components["xspectral_loss"].detach().cpu()),
                "xspectral_L_same": float(components["xspectral_L_same"].detach().cpu()),
                "xspectral_L_crossfreq": float(components["xspectral_L_crossfreq"].detach().cpu()),
                "coherence_applied": 1.0,
                "data_grad_norm": float(grad_info.get("data_grad_norm", float("nan"))),
                "coherence_grad_norm": float(grad_info.get("coherence_grad_norm", float("nan"))),
                "gradient_cosine": float(grad_info.get("gradient_cosine", float("nan"))),
                "gradient_conflict": 1.0 if bool(grad_info.get("gradient_conflict", False)) else 0.0,
            })

        rows.append(row)
        pbar.set_postfix_str(
            f"data={row['data_loss']:.3e} coh={row['coherence_loss']:.3e} applied={int(row['coherence_applied'])}"
        )

    count = max(len(rows), 1)
    metrics = {
        "total_loss": _mean_metric(rows, "total_loss"),
        "data_loss": _mean_metric(rows, "data_loss"),
        "coherence_loss": _mean_metric(rows, "coherence_loss"),
        "coherence_self": _mean_metric(rows, "coherence_self"),
        "coherence_mutual": _mean_metric(rows, "coherence_mutual"),
        "coherence_cross": _mean_metric(rows, "coherence_cross"),
        "xspectral_loss": _mean_metric(rows, "xspectral_loss"),
        "L_same": _mean_metric(rows, "xspectral_L_same"),
        "L_crossfreq": _mean_metric(rows, "xspectral_L_crossfreq"),
        "coherence_application_fraction": float(applied / count),
        "data_grad_norm": _mean_metric(rows, "data_grad_norm"),
        "coherence_grad_norm": _mean_metric(rows, "coherence_grad_norm"),
        "gradient_cosine": _mean_metric(rows, "gradient_cosine"),
        "gradient_conflict_fraction": float(conflict_count / max(applied, 1)),
        "global_step": float(global_step),
    }
    return metrics, global_step


def find_latest_run_dir(demo_dir: str, save_dir: str, demo_num: int) -> Optional[Path]:
    save_root = Path(demo_dir) / Path(save_dir).parent
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    if not save_root.exists():
        return None

    candidates = [
        path for path in save_root.glob(f"{run_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, save_dir: str, demo_num: int) -> str:
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    run_name = run_dir.name
    if run_name.startswith(run_prefix):
        return run_name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    if not path.exists():
        return

    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


class TrainingHistoryLogger:
    def __init__(self, run_dir: Path) -> None:
        self.csv_path  = run_dir / "loss_history.csv"
        self.json_path = run_dir / "loss_history.json"
        self.plot_path = run_dir / "loss_history.png"
        self.rows = []
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "train_coh_loss", "L_same", "L_crossfreq",
                "train_total_loss", "coh_weight",
            ])

    def log_and_plot(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_coh_loss: float = 0.0,
        L_same: float = 0.0,
        L_crossfreq: float = 0.0,
        train_total_loss: Optional[float] = None,
        coh_weight: float = 0.0,
    ) -> None:
        if train_total_loss is None:
            train_total_loss = train_loss

        row = {
            "epoch":            int(epoch),
            "train_loss":       float(train_loss),
            "val_loss":         None if val_loss is None else float(val_loss),
            "train_coh_loss":   float(train_coh_loss),
            "L_same":           float(L_same),
            "L_crossfreq":      float(L_crossfreq),
            "train_total_loss": float(train_total_loss),
            "coh_weight":       float(coh_weight),
        }
        self.rows.append(row)

        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                "" if row["val_loss"] is None else row["val_loss"],
                row["train_coh_loss"],
                row["L_same"],
                row["L_crossfreq"],
                row["train_total_loss"],
                row["coh_weight"],
            ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)

        # ------------------------------------------------------------------ #
        # Determine whether coherence was ever active in the logged history.
        # ------------------------------------------------------------------ #
        any_coh = any(r["coh_weight"] > 0.0 for r in self.rows)

        # Epoch at which coherence first became nonzero (for the kick-in line).
        coh_kickin_epoch: Optional[int] = None
        for r in self.rows:
            if r["coh_weight"] > 0.0:
                coh_kickin_epoch = r["epoch"]
                break

        # ------------------------------------------------------------------ #
        # Data series helpers
        # ------------------------------------------------------------------ #
        def _series(key: str) -> list:
            return [
                (r["epoch"], r[key])
                for r in self.rows
                if r.get(key) is not None and r[key] > 0.0
            ]

        train_points = _series("train_loss")
        val_points   = _series("val_loss")
        total_points = _series("train_total_loss") if any_coh else []
        coh_points   = _series("train_coh_loss")
        same_points  = _series("L_same")
        xfreq_points = _series("L_crossfreq")
        weight_points = [
            (r["epoch"], r["coh_weight"])
            for r in self.rows
            if r.get("coh_weight") is not None
        ]

        # ------------------------------------------------------------------ #
        # Figure layout: one panel if coherence is off, two panels if on.
        # ------------------------------------------------------------------ #
        if any_coh:
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(10, 9), sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )
        else:
            fig, ax_top = plt.subplots(figsize=(10, 6))
            ax_bot = None

        # ------------------------------------------------------------------ #
        # Top panel — data + total loss (log scale)
        # ------------------------------------------------------------------ #
        if train_points:
            ax_top.plot(
                [p[0] for p in train_points],
                [p[1] for p in train_points],
                label="Train Data Loss",
                color="steelblue",
                marker="o",
                markersize=3,
                linewidth=1.2,
            )
        if val_points:
            ax_top.plot(
                [p[0] for p in val_points],
                [p[1] for p in val_points],
                label="Validation Loss",
                color="darkorange",
                marker="s",
                markersize=4,
                linewidth=1.4,
            )
        if total_points:
            ax_top.plot(
                [p[0] for p in total_points],
                [p[1] for p in total_points],
                label="Train Total Loss",
                color="navy",
                linestyle="--",
                linewidth=1.0,
            )

        # Vertical kick-in line on top panel
        if coh_kickin_epoch is not None:
            ax_top.axvline(
                coh_kickin_epoch, color="green", linestyle=":", linewidth=1.2,
                label=f"coherence on (ep {coh_kickin_epoch})",
            )

        ax_top.set_ylabel("Loss (log scale)")
        ax_top.set_title("Conditional Point-Cloud FFM Training Progress")
        if train_points or val_points:
            ax_top.set_yscale("log")
        ax_top.legend(fontsize=8)
        ax_top.grid(True, which="both", ls="--", alpha=0.4)

        # ------------------------------------------------------------------ #
        # Bottom panel — coherence components + weight ramp
        # ------------------------------------------------------------------ #
        if ax_bot is not None:
            if coh_points:
                ax_bot.plot(
                    [p[0] for p in coh_points],
                    [p[1] for p in coh_points],
                    label="Coh Loss (total)",
                    color="green",
                    linewidth=1.2,
                )
            if same_points:
                ax_bot.plot(
                    [p[0] for p in same_points],
                    [p[1] for p in same_points],
                    label=r"$L_\mathrm{same}$",
                    color="mediumseagreen",
                    linestyle="-.",
                    linewidth=1.0,
                )
            if xfreq_points:
                ax_bot.plot(
                    [p[0] for p in xfreq_points],
                    [p[1] for p in xfreq_points],
                    label=r"$L_\mathrm{crossfreq}$",
                    color="teal",
                    linestyle=":",
                    linewidth=1.0,
                )

            ax_bot.set_ylabel("Coherence Loss")
            ax_bot.set_xlabel("Epoch")
            ax_bot.grid(True, which="both", ls="--", alpha=0.4)

            # Coherence weight on a secondary right y-axis
            if weight_points:
                ax_w = ax_bot.twinx()
                ax_w.plot(
                    [p[0] for p in weight_points],
                    [p[1] for p in weight_points],
                    label="Coh Weight",
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                )
                ax_w.set_ylabel("Coherence Weight", color="gray")
                ax_w.tick_params(axis="y", labelcolor="gray")
                ax_w.set_ylim(bottom=0.0)
                # Combine legends from both axes
                lines_l, labels_l = ax_bot.get_legend_handles_labels()
                lines_r, labels_r = ax_w.get_legend_handles_labels()
                ax_bot.legend(lines_l + lines_r, labels_l + labels_r, fontsize=8)
            else:
                ax_bot.legend(fontsize=8)

            # Vertical kick-in line on bottom panel too
            if coh_kickin_epoch is not None:
                ax_bot.axvline(
                    coh_kickin_epoch, color="green", linestyle=":", linewidth=1.2,
                )
        else:
            ax_top.set_xlabel("Epoch")

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


class DirectCoherenceHistoryLogger:
    def __init__(self, run_dir: Path) -> None:
        self.csv_path = run_dir / "direct_coherence_history.csv"
        self.json_path = run_dir / "direct_coherence_history.json"
        self.plot_path = run_dir / "direct_coherence_history.png"
        self.rows = []
        self.fieldnames = [
            "epoch",
            "train_total_loss",
            "train_data_loss",
            "train_coherence_loss",
            "coherence_self",
            "coherence_mutual",
            "coherence_cross",
            "xspectral_loss",
            "xspectral_L_same",
            "xspectral_L_crossfreq",
            "coherence_application_fraction",
            "data_grad_norm",
            "coherence_grad_norm",
            "gradient_cosine",
            "gradient_conflict_fraction",
            "lr",
            "global_step",
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, epoch: int, metrics: dict, lr: float, global_step: int) -> None:
        row = {
            "epoch": int(epoch),
            "train_total_loss": metrics.get("total_loss", float("nan")),
            "train_data_loss": metrics.get("data_loss", float("nan")),
            "train_coherence_loss": metrics.get("coherence_loss", float("nan")),
            "coherence_self": metrics.get("coherence_self", float("nan")),
            "coherence_mutual": metrics.get("coherence_mutual", float("nan")),
            "coherence_cross": metrics.get("coherence_cross", float("nan")),
            "xspectral_loss": metrics.get("xspectral_loss", float("nan")),
            "xspectral_L_same": metrics.get("L_same", float("nan")),
            "xspectral_L_crossfreq": metrics.get("L_crossfreq", float("nan")),
            "coherence_application_fraction": metrics.get("coherence_application_fraction", float("nan")),
            "data_grad_norm": metrics.get("data_grad_norm", float("nan")),
            "coherence_grad_norm": metrics.get("coherence_grad_norm", float("nan")),
            "gradient_cosine": metrics.get("gradient_cosine", float("nan")),
            "gradient_conflict_fraction": metrics.get("gradient_conflict_fraction", float("nan")),
            "lr": float(lr),
            "global_step": int(global_step),
        }
        self.rows.append(row)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        self._plot()

    @staticmethod
    def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
        xs = []
        ys = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                continue
            xs.append(int(row["epoch"]))
            ys.append(value)
        return xs, ys

    def _plot(self) -> None:
        any_xspec = any(
            math.isfinite(float(r.get("xspectral_loss", float("nan"))))
            and float(r.get("xspectral_loss", 0.0)) > 0.0
            for r in self.rows
        )

        if any_xspec:
            fig, axes = plt.subplots(1, 3, figsize=(21, 5))
        else:
            fig, axes_2 = plt.subplots(1, 2, figsize=(14, 5))
            axes = list(axes_2) + [None]

        left_specs = [
            ("train_total_loss", "Total", "#1f77b4"),
            ("train_data_loss", "Data", "#2ca02c"),
            ("train_coherence_loss", "Physical coherence", "#d62728"),
        ]
        for key, label, color in left_specs:
            xs, ys = self._series(self.rows, key)
            if xs:
                axes[0].plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        axes[0].set_title("Direct Post-Training Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)
        if axes[0].lines:
            axes[0].set_yscale("log")
            axes[0].legend()

        mid_specs = [
            ("coherence_self", "Self", "#9467bd"),
            ("coherence_mutual", "Mutual", "#ff7f0e"),
            ("coherence_cross", "Cross", "#17becf"),
        ]
        for key, label, color in mid_specs:
            xs, ys = self._series(self.rows, key)
            if xs:
                axes[1].plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        axes[1].set_title("Wasserstein Coherence Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, which="both", ls="--", alpha=0.4)
        if axes[1].lines:
            axes[1].set_yscale("log")
            axes[1].legend()

        if axes[2] is not None:
            xspec_specs = [
                ("xspectral_loss", r"$L_\mathrm{xspectral}$ (total)", "#e377c2"),
                ("xspectral_L_same", r"$L_\mathrm{same}$", "#bcbd22"),
                ("xspectral_L_crossfreq", r"$L_\mathrm{crossfreq}$", "#8c564b"),
            ]
            for key, label, color in xspec_specs:
                xs, ys = self._series(self.rows, key)
                if xs:
                    axes[2].plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
            axes[2].set_title("Cross-Spectral Components")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("Loss")
            axes[2].grid(True, which="both", ls="--", alpha=0.4)
            if axes[2].lines:
                axes[2].set_yscale("log")
                axes[2].legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


def resolve_pretrained_checkpoint(demo_dir: str, args) -> tuple[Optional[Path], Optional[Path]]:
    if str(args.initialization) != "pretrained" or bool(args.RELOAD):
        return None, None
    source_run = find_source_run_dir(
        demo_dir=demo_dir,
        source_run_dir=args.pretrained_run_dir,
        source_Demo_Num=args.pretrained_source_Demo_Num,
        save_dir_hint=args.save_dir,
    )
    ckpt_name = str(args.pretrained_checkpoint)
    ckpt_path = Path(ckpt_name)
    if ckpt_path.suffix != ".pt":
        ckpt_path = ckpt_path.with_suffix(".pt")
    if not ckpt_path.is_absolute():
        ckpt_path = source_run / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    return source_run, ckpt_path


SOURCE_BASE_CONFIG_KEYS = (
    # Dataset/split identity. Keep these aligned with the pretrained model's
    # normalization statistics and split semantics.
    "data",
    "train_ratio",
    "time_stride",

    # Backbone and architecture shape.
    "backbone",
    "hidden_dim",
    "cond_dim",
    "field_embed_dim",
    "rbf_sigma",
    "sigma_min",
    "USE_FOURIER_PE",
    "fourier_pe_num_bands",
    "fourier_pe_max_freq",
    "latent_dim",
    "num_latents",
    "num_heads",
    "num_latent_blocks",
    "ff_mult",
    "attn_dropout",
    "mlp_dropout",
    "decode_chunk_size",
    "share_query_proj",
    "summary_type",
    "gather_mode",
    "gather_topk",
    "gather_query_chunk_size",
    "learnable_rbf_sigma",
    "neighbor_backend",
    "sensor_local_topk",
    "sensor_local_dropout",
    "sensor_coord_encoding",
    "latent_sensor_reinject",
    "latent_reinject_every",
    "query_latent_readout",
    "query_readout_type",
    "query_readout_scale_init",
    "enhanced_head_norm",
    "glres_scale_init",

    # FNO architecture keys.
    "Num_x",
    "Num_y",
    "fno_modes_x",
    "fno_modes_y",
    "fno_hidden_channels",
    "fno_n_layers",
    "condition_blur",
    "condition_blur_kernel",
    "condition_blur_sigma",

    # Base RF query/prior choices.
    "n_query_points",
    "query_sampling",
    "query_sample_near_ratio",
    "query_sample_far_ratio",
    "query_sample_sigma_ratio",
    "prior",
    "rff_features",
    "rff_lengthscale",

    # Sparse conditioning and generation/evaluation defaults.
    "cond_field",
    "n_obs_min",
    "n_obs_max",
    "cond_fields",
    "n_obs_min_list",
    "n_obs_max_list",
    "vis_cond_fields",
    "vis_n_obs_list",
    "ode_solver",
    "benchmark_n_steps",
    "n_steps_generation",
)


def apply_pretrained_source_base_config(args, source_run_dir: Optional[Path]) -> list[str]:
    """
    In pretrained post-training, prefer the source run's base model/data/
    conditioning config for checkpoint compatibility.

    Post-training controls such as Demo_Num, device, optimizer, epochs,
    coherence schedules, and loss weights remain from the direct-posttrain
    config/CLI.
    """
    if source_run_dir is None or not bool(getattr(args, "pretrained_use_source_base_config", True)):
        return []

    source_cfg = load_source_config(source_run_dir)
    inherited = []
    for key in SOURCE_BASE_CONFIG_KEYS:
        if key not in source_cfg or not hasattr(args, key):
            continue
        setattr(args, key, source_cfg[key])
        inherited.append(key)
    return inherited


def checkpoint_metadata(args, direct_cfg: Optional[DirectCoherenceConfig], source_run_dir, source_checkpoint) -> dict:
    if str(args.training_mode) == "direct_coherence":
        method = "direct_coherence_rectified_flow"
    else:
        method = "1_rectified_flow"
    return {
        "method": method,
        "training_mode": args.training_mode,
        "initialization": args.initialization,
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "data_loss_weight": float(args.data_loss_weight),
        "coherence_loss_weight": float(args.coherence_loss_weight),
        "gradient_balance_mode": args.gradient_balance_mode,
        "coherence_config": None if direct_cfg is None else direct_cfg.to_dict(),
        "pretrained_use_source_base_config": bool(getattr(args, "pretrained_use_source_base_config", True)),
        "pretrained_inherited_base_config_keys": list(getattr(args, "pretrained_inherited_base_config_keys", [])),
    }


def architecture_compatibility_hint(args, source_run_dir: Optional[Path]) -> str:
    if source_run_dir is None:
        return ""
    try:
        source_cfg = load_source_config(source_run_dir)
    except Exception as exc:
        return f"\nCould not read source config from {source_run_dir}: {exc}"

    keys = [
        "backbone",
        "hidden_dim",
        "cond_dim",
        "field_embed_dim",
        "USE_FOURIER_PE",
        "fourier_pe_num_bands",
        "fourier_pe_max_freq",
        "latent_dim",
        "num_latents",
        "num_heads",
        "num_latent_blocks",
        "summary_type",
        "gather_mode",
        "gather_topk",
        "gather_query_chunk_size",
        "learnable_rbf_sigma",
        "sensor_coord_encoding",
        "latent_sensor_reinject",
        "query_latent_readout",
        "query_readout_type",
        "query_readout_scale_init",
        "enhanced_head_norm",
        "glres_scale_init",
    ]
    current = vars(args)
    lines = ["\nArchitecture keys that commonly affect checkpoint compatibility:"]
    for key in keys:
        src = source_cfg.get(key, "<missing>")
        cur = current.get(key, "<missing>")
        if src != cur:
            lines.append(f"  {key}: source={src!r}, current={cur!r}")
    if len(lines) == 1:
        lines.append("  No obvious architecture-key differences found in saved config.")
    return "\n".join(lines)


def main():

    args, p = parse_args()
    # Capture CLI-only state (before YAML overrides) so that RELOAD merge
    # can correctly distinguish "user set this on CLI" vs "came from YAML".
    _cli_only_args = vars(p.parse_args())
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir) # Go up one level to \demo
    
    # YAML Loading and Backup
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(config_path):
        print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Apply YAML values only for args that were not explicitly set on the CLI.
        # CLI flags always win over YAML; YAML fills in everything else.
        if yaml_config is not None:
            _cli_defaults = vars(p.parse_args([]))
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    if vars(args)[key] == _cli_defaults.get(key):
                        setattr(args, key, value)
                else:
                    print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)
                    
        # Backup the YAML file
        backup_dir = os.path.join(demo_dir, "Save_config", "pointcloud_ffm")
        os.makedirs(backup_dir, exist_ok=True)
        backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
        shutil.copy(config_path, os.path.join(backup_dir, backup_filename))
        print(f"[*] Config backed up to: {os.path.join(backup_dir, backup_filename)}\n")
    else:
        print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0  # Force Demo_Num to 0 as default
        args = normalize_conditioning_args(args)

    # Setup the Dynamic Directories with Demo_Num
    set_seed(args.seed)

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    global_step = 0
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num)
        if latest_run_dir is not None:
            ckpt_to_load = None
            if (latest_run_dir / "last.pt").exists():
                ckpt_to_load = latest_run_dir / "last.pt"
            elif (latest_run_dir / "best.pt").exists():
                ckpt_to_load = latest_run_dir / "best.pt"

            if ckpt_to_load is not None:
                save_dir = latest_run_dir
                run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
                reload_ckpt = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
                start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
                best_val = float(reload_ckpt.get("val_loss", float("inf")))
                global_step = int(reload_ckpt.get("global_step", 0))
                backup_existing_artifact(ckpt_to_load)
                print(f"[*] RELOAD=True, resuming from: {ckpt_to_load}")
                print(f"[*] Resume will start from epoch {start_epoch}\n")

                # Merge config from the saved run's args.json so the model is
                # rebuilt with the same conditioning and architecture as the
                # original run. CLI flags always win; saved args override both
                # YAML template and argparse defaults.
                saved_args_path = latest_run_dir / "args.json"
                if saved_args_path.exists():
                    with open(saved_args_path, "r") as _f:
                        _saved_args = json.load(_f)
                    _cli_defaults = vars(p.parse_args([]))
                    # Use _cli_only_args (pre-YAML CLI state) for comparison instead of
                    # _cli_defaults (argparse-only defaults). This ensures that values set
                    # by the YAML template — but not by the user on the CLI — are still
                    # overridable by the saved args.json from the original run.
                    for _key, _value in _saved_args.items():
                        if hasattr(args, _key) and _cli_only_args.get(_key) == _cli_defaults.get(_key):
                            setattr(args, _key, _value)
                    # Preserve the inherited-keys record from the original run.
                    if "pretrained_inherited_base_config_keys" in _saved_args:
                        args.pretrained_inherited_base_config_keys = (
                            _saved_args["pretrained_inherited_base_config_keys"]
                        )
                    args = normalize_conditioning_args(args)
                    print(f"[*] RELOAD: merged config from saved args.json in {latest_run_dir}\n")
                else:
                    print("[*] RELOAD: no args.json found in run dir; using YAML/CLI config only.\n")
            else:
                print("[*] RELOAD=True, but no matching checkpoint was found. Training will start from scratch.\n")

    source_run_dir, source_checkpoint = resolve_pretrained_checkpoint(demo_dir, args)
    if source_checkpoint is not None:
        args.source_run_dir = str(source_run_dir)
        args.source_checkpoint = str(source_checkpoint)
        inherited_keys = apply_pretrained_source_base_config(args, source_run_dir)
        args.pretrained_inherited_base_config_keys = inherited_keys
        args = normalize_conditioning_args(args)
        print(f"[*] initialization=pretrained, source run: {source_run_dir}")
        print(f"[*] initialization=pretrained, checkpoint: {source_checkpoint}")
        if inherited_keys:
            print(
                "[*] Inherited base model/data/conditioning keys from source run: "
                + ", ".join(inherited_keys)
            )
        else:
            print("[*] No base config keys inherited from source run.")
    else:
        # Only reset if RELOAD did not already restore the keys from the saved run.
        if not getattr(args, "pretrained_inherited_base_config_keys", None):
            args.pretrained_inherited_base_config_keys = []

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the final parsed args to a JSON in the model folder just to be safe
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if os.path.exists(config_path):
        shutil.copy(config_path, save_dir / "run_config.yaml")

    # Keep all run artifacts under the model directory, matching the unified
    # baseline trainers. The old Save_loss_csv/ and Save_reconstruction_files/
    # roots are no longer used by this trainer.
    recon_dir = save_dir / "Evaluation"

    if args.RELOAD and reload_ckpt is not None:
        for loss_artifact in (
            "loss_history.csv",
            "loss_history.json",
            "loss_history.png",
            "direct_coherence_history.csv",
            "direct_coherence_history.json",
            "direct_coherence_history.png",
        ):
            backup_existing_artifact(save_dir / loss_artifact)
        backup_existing_artifact(recon_dir)

    # Initialize helpers
    logger = TrainingHistoryLogger(save_dir)
    direct_cfg = build_direct_coherence_config(args)
    direct_logger = DirectCoherenceHistoryLogger(save_dir) if args.training_mode == "direct_coherence" else None
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Model checkpoints will save to: {save_dir}")
    print(f"[*] Logging losses to: {save_dir}")
    print(f"[*] Saving recon plots to: {recon_dir}\n")

    device_ids = args.device_ids
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    _var_names   = tuple(args.cb_var_names   or DEFAULT_CB_VARS)
    _field_names = tuple(args.cb_field_names or DEFAULT_CB_NAMES)
    _shared_ds_kwargs = dict(
        data_dir        = args.data,
        static_nc_path  = args.static_nc,
        train_ratio     = args.train_ratio,
        seed            = args.seed,
        var_names       = _var_names,
        var_level       = args.cb_var_level,
        field_names     = _field_names,
        time_stride     = args.time_stride,
        use_wetdry_mask = args.cb_use_wetdry_mask,
        stats_path      = str(save_dir / "dataset_stats.pt"),
    )
    train_set = CoastalBenchNCDataset(split="train", **_shared_ds_kwargs)
    val_set   = CoastalBenchNCDataset(split="val",   **_shared_ds_kwargs)

    coord_dim = train_set.coords.shape[-1]   # 2 for CoastalBench (lon/lat)
    args.coord_dim = coord_dim               # persisted in args.json for downstream scripts

    # Re-save args.json now that coord_dim is known (first save at line ~1886 predates dataset init)
    with open(save_dir / "args.json", "w") as _f:
        import json as _json
        _json.dump(vars(args), _f, indent=2)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cb,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_cb,
    )

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=coord_dim, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=coord_dim,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            rbf_sigma=args.rbf_sigma,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=train_set.num_fields,
            coord_dim=coord_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            field_embed_dim=args.field_embed_dim,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            decode_chunk_size=args.decode_chunk_size,
            share_query_proj=args.share_query_proj,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = args.backbone == "GL_rbf_ENH"

        # Enhanced defaults are resolved here so legacy GL_rbf configs/checkpoints
        # keep their original point-readout and zero-scale behavior.
        sensor_coord_encoding = args.sensor_coord_encoding
        if sensor_coord_encoding is None:
            sensor_coord_encoding = "fourier" if enhanced else "raw"

        latent_sensor_reinject = args.latent_sensor_reinject
        if latent_sensor_reinject is None:
            latent_sensor_reinject = enhanced

        query_latent_readout = args.query_latent_readout
        if query_latent_readout is None:
            query_latent_readout = enhanced

        enhanced_head_norm = args.enhanced_head_norm
        if enhanced_head_norm is None:
            enhanced_head_norm = enhanced

        query_readout_type = args.query_readout_type
        if query_readout_type is None:
            query_readout_type = "coord" if enhanced else "point"

        query_readout_scale_init = args.query_readout_scale_init
        if query_readout_scale_init is None:
            query_readout_scale_init = 1.0e-2 if enhanced else 0.0

        glres_scale_init = args.glres_scale_init
        if glres_scale_init is None:
            glres_scale_init = 1.0e-2 if enhanced else 0.0

        print(
            "[*] GL_rbf settings: "
            f"enhanced={enhanced}, "
            f"sensor_coord_encoding={sensor_coord_encoding}, "
            f"latent_sensor_reinject={latent_sensor_reinject}, "
            f"latent_reinject_every={args.latent_reinject_every}, "
            f"query_latent_readout={query_latent_readout}, "
            f"query_readout_type={query_readout_type}, "
            f"query_readout_scale_init={query_readout_scale_init}, "
            f"enhanced_head_norm={enhanced_head_norm}, "
            f"glres_scale_init={glres_scale_init}"
        )

        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=train_set.num_fields,
            coord_dim=coord_dim,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            rbf_sigma=args.rbf_sigma,
            summary_type=args.summary_type,

            gather_mode=args.gather_mode,
            gather_topk=args.gather_topk,
            gather_query_chunk_size=args.gather_query_chunk_size,
            learnable_rbf_sigma=args.learnable_rbf_sigma,
            neighbor_backend=args.neighbor_backend,

            sensor_local_topk=args.sensor_local_topk,
            sensor_local_dropout=args.sensor_local_dropout,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=args.latent_reinject_every,
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "fno":
        # FNO requires an explicit regular-grid interpretation of the dataset.
        # validate_regular_grid_compatibility is not imported for CoastalBench;
        # FNO backbone is unsupported on the ROMS curvilinear grid.
        raise NotImplementedError(
            "FNO backbone requires a regular Cartesian grid and is not supported "
            "for CoastalBench (ROMS curvilinear grid).  Use --backbone GL_rbf_ENH."
        )

        print(
            "[*] FNO grid detected: "
            f"{grid_info['unique_x']} unique x values x {grid_info['unique_y']} unique y values "
            f"= {grid_info['num_points']} points."
        )
        print(
            "[*] FNO grid spacing in normalized coords: "
            f"x min/med/max={grid_info['x_spacing_min']:.3e}/"
            f"{grid_info['x_spacing_median']:.3e}/{grid_info['x_spacing_max']:.3e}, "
            f"y min/med/max={grid_info['y_spacing_min']:.3e}/"
            f"{grid_info['y_spacing_median']:.3e}/{grid_info['y_spacing_max']:.3e}."
        )
        if grid_info["requires_permutation"]:
            print(
                "[*] FNO grid order: dataset is not row-major; the FNO backbone will "
                "internally permute point order -> row-major grid and invert the "
                "permutation on output."
            )
            print(
                "[*] FNO grid permutation sample: first row-major cells come from "
                f"original indices {grid_info['first_row_original_indices']}; "
                "first original points map to grid cells "
                f"{grid_info['first_original_to_grid_indices']}."
            )
        if not grid_info["spacing_regular"]:
            print(
                "[*] FNO grid note: physical x/y spacing is nonuniform. FNO will run "
                "on the topological index grid; point-cloud baselines still use "
                "the physical coordinates directly."
            )

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
        )
        model = FNOFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

        print(f"[*] Using grid-based FNO baseline with Num_x={args.Num_x}, Num_y={args.Num_y}")
        print("[*] Note: n_query_points is ignored for FNO because it requires the full grid.\n")
    else:
        raise ValueError(
            f'Error!!! Your backbone is not supported: {args.backbone}.'
            'Please select in ["mlp_rbf", "perceiver", "fno"]'
            )
    print(f'\nSelected Backbone: {args.backbone}\n')

    # ---- GFT basis precomputation for coherence loss ----
    U_gft    = None
    coh_bands = None
    coh_cfg  = None
    if args.coh_loss:
        print("[*] Precomputing graph-Fourier basis for coherence loss ...")
        coords_np = train_set.coords.numpy()  # [N, 3], fixed for this dataset
        basis = compute_graph_basis(
            coords_np,
            n_modes=args.coh_n_modes,
            k_neighbors=args.coh_k_neighbors,
            cache_path=str(save_dir / "gft_basis.npz"),
        )
        U_gft = torch.tensor(basis.eigvecs, dtype=torch.float32).to(device)  # [N, K]
        coh_bands = _bands_from_basis(basis)
        coh_cfg = CrossSpectralConfig(eta_crossfreq=args.coh_eta_crossfreq)
        if not coh_bands:
            print("[Warning] No non-DC bands found in GFT basis — coherence loss will be skipped.")
            U_gft = None
        else:
            band_summary = {k: len(v) for k, v in coh_bands.items()}
            print(f"[*] GFT basis ready: {U_gft.shape[1]} modes, bands={band_summary}")
            print(f"[*] Coherence loss enabled: lambda={args.coh_lambda}, "
                  f"ramp={args.coh_ramp}, rollout_steps={args.coh_n_steps}, "
                  f"grad_proj={args.grad_proj}\n")

    # ---- GFT basis for direct_coherence post-training cross-spectral term ----
    if args.training_mode == "direct_coherence" and direct_cfg.xspectral_weight > 0.0:
        print("[*] Precomputing GFT basis for cross-spectral post-training term ...")
        coords_np = train_set.coords.numpy()
        basis = compute_graph_basis(
            coords_np,
            n_modes=direct_cfg.xspectral_n_modes,
            k_neighbors=direct_cfg.xspectral_k_neighbors,
            cache_path=str(save_dir / "gft_basis_posttrain.npz"),
        )
        U_gft = torch.tensor(basis.eigvecs, dtype=torch.float32).to(device)
        coh_bands = _bands_from_basis(basis)
        if not coh_bands:
            print("[Warning] No non-DC bands found in post-training GFT basis — "
                  "xspectral_weight will have no effect.")
            U_gft = None
        else:
            band_summary = {k: len(v) for k, v in coh_bands.items()}
            n_coh = args.coherence_n_points if bool(args.coherence_downsample) else train_set.coords.shape[0]
            if n_coh < direct_cfg.xspectral_n_modes:
                print(f"[Warning] coherence_n_points ({n_coh}) < xspectral_n_modes "
                      f"({direct_cfg.xspectral_n_modes}). GFT will be rank-deficient. "
                      "Reduce xspectral_n_modes or increase coherence_n_points.")
            print(f"[*] Post-training GFT basis ready: {U_gft.shape[1]} modes, bands={band_summary}")
            print(f"[*] Cross-spectral post-training term enabled: "
                  f"xspectral_weight={direct_cfg.xspectral_weight}, "
                  f"eta_crossfreq={direct_cfg.xspectral_eta_crossfreq}\n")

    coh_warmup_epochs = args.coh_warmup_epochs if args.coh_warmup_epochs is not None else args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if reload_ckpt is not None:
        model.load_state_dict(reload_ckpt["model"])
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        print("[*] Reloaded model state from last.pt (or best.pt fallback)")
    elif source_checkpoint is not None:
        pretrained_ckpt = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        pretrained_state = (
            pretrained_ckpt["model"]
            if isinstance(pretrained_ckpt, dict) and "model" in pretrained_ckpt
            else pretrained_ckpt
        )
        try:
            load_result = model.load_state_dict(pretrained_state, strict=bool(args.pretrained_strict))
        except RuntimeError as exc:
            raise RuntimeError(
                "Pretrained checkpoint is architecture-incompatible with the current model. "
                "Check backbone/hidden dimensions/conditioning architecture or set pretrained_strict=false "
                "only if the mismatch is intentional."
                f"{architecture_compatibility_hint(args, source_run_dir)}"
            ) from exc
        if not bool(args.pretrained_strict):
            print(f"[*] Pretrained load missing keys: {load_result.missing_keys}")
            print(f"[*] Pretrained load unexpected keys: {load_result.unexpected_keys}")
        if bool(args.pretrained_load_optimizer) and isinstance(pretrained_ckpt, dict) and "optimizer" in pretrained_ckpt:
            optimizer.load_state_dict(pretrained_ckpt["optimizer"])
            print("[*] Loaded optimizer state from pretrained checkpoint")
        print("[*] Loaded pretrained model state; new run starts at epoch 1")

    for epoch in range(start_epoch, args.epochs + 1):
        if args.training_mode == "direct_coherence":
            # --- Professor's Wasserstein post-training path ---
            epoch_train_loader = build_epoch_train_loader(train_set, args, epoch)
            direct_metrics, global_step = run_epoch_direct_coherence(
                model=model,
                loader=epoch_train_loader,
                optimizer=optimizer,
                device=device,
                cond_fields=args.cond_fields,
                n_obs_min_list=args.n_obs_min_list,
                n_obs_max_list=args.n_obs_max_list,
                n_query_points=args.n_query_points,
                query_sampling=args.query_sampling,
                query_sample_near_ratio=args.query_sample_near_ratio,
                query_sample_far_ratio=args.query_sample_far_ratio,
                query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                direct_cfg=direct_cfg,
                args=args,
                global_step=global_step,
                epoch=epoch,
                mean=train_set.mean.to(device),
                std=train_set.std.to(device),
                U_gft=U_gft,
                coh_bands=coh_bands,
            )
            tr_loss = float(direct_metrics["total_loss"])
            coh_weight = 0.0
            # Bridge direct_metrics into the tr_metrics shape expected by logger.log_and_plot.
            tr_metrics = {
                "coh_loss": direct_metrics.get("coherence_loss", 0.0),
                "total_loss": direct_metrics.get("total_loss", tr_loss),
                "L_same": direct_metrics.get("L_same", 0.0),
                "L_crossfreq": direct_metrics.get("L_crossfreq", 0.0),
            }
        else:
            # --- Standard training with optional cross-spectral coherence loss ---
            direct_metrics = None
            # ---- coherence weight schedule ----
            if args.coh_loss and U_gft is not None:
                if args.coh_ramp:
                    coh_weight = args.coh_lambda * min(1.0, epoch / max(1, coh_warmup_epochs))
                else:
                    coh_weight = args.coh_lambda
            else:
                coh_weight = 0.0
            tr_loss, tr_metrics = run_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                cond_fields=args.cond_fields,
                n_obs_min_list=args.n_obs_min_list,
                n_obs_max_list=args.n_obs_max_list,
                n_query_points=args.n_query_points,
                query_sampling=args.query_sampling,
                query_sample_near_ratio=args.query_sample_near_ratio,
                query_sample_far_ratio=args.query_sample_far_ratio,
                query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                epoch=epoch,
                U_gft=U_gft,
                coh_bands=coh_bands,
                coh_weight=coh_weight,
                coh_n_steps=args.coh_n_steps,
                coh_truncate_bptt=args.coh_truncate_bptt,
                coh_every_n_batches=args.coh_every_n_batches,
                coh_cfg=coh_cfg,
                coh_n_query_points=args.coh_n_query_points,
                grad_proj=args.grad_proj,
            )
            global_step += len(train_loader)
        scheduler.step()

        if direct_metrics is not None:
            print(
                f"[train] epoch={epoch:04d} total={direct_metrics['total_loss']:.6e} "
                f"data={direct_metrics['data_loss']:.6e} coherence={direct_metrics['coherence_loss']:.6e} "
                f"self={direct_metrics['coherence_self']:.3e} mutual={direct_metrics['coherence_mutual']:.3e} "
                f"cross={direct_metrics['coherence_cross']:.3e} "
                f"applied={direct_metrics['coherence_application_fraction']:.2%}"
            )
            if direct_logger is not None:
                direct_logger.log(
                    epoch=epoch,
                    metrics=direct_metrics,
                    lr=optimizer.param_groups[0]["lr"],
                    global_step=global_step,
                )
        elif coh_weight > 0.0:
            print(f"[train] epoch={epoch:04d} data_loss={tr_loss:.6e} "
                  f"coh_loss={tr_metrics['coh_loss']:.6e} "
                  f"total_loss={tr_metrics['total_loss']:.6e} "
                  f"coh_w={coh_weight:.4f}")
        else:
            print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}")
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad():
                val_loss, val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    cond_fields=args.cond_fields,
                    n_obs_min_list=args.n_obs_min_list,
                    n_obs_max_list=args.n_obs_max_list,
                    n_query_points=args.n_query_points,
                    query_sampling=args.query_sampling,
                    query_sample_near_ratio=args.query_sample_near_ratio,
                    query_sample_far_ratio=args.query_sample_far_ratio,
                    query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                    epoch=epoch,
                    # Coherence is evaluated on training only; validation runs
                    # without the coherence rollout to keep eval cost predictable.
                )
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
            }
            ckpt.update(checkpoint_metadata(args, direct_cfg, source_run_dir, source_checkpoint))
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print('Saving the best model...')
        
        if epoch % args.save_every == 0:
            # Benchmark the same validation snapshot at several NFEs.

            recon_dir_epoch = recon_dir / f"epoch_{epoch:04d}"
            recon_dir_epoch.mkdir(parents=True, exist_ok=True)
            
            step_list = args.benchmark_n_steps if args.benchmark_n_steps else [args.n_steps_generation]
            for nfe in step_list:
                # recon_metrics = visualize_reconstruction(
                #     model=model,
                #     dataset=val_set,
                #     epoch=epoch,
                #     device=device,
                #     save_dir=recon_dir_epoch,

                #     cond_fields=args.vis_cond_fields,
                #     n_obs=args.vis_n_obs_list,

                #     n_steps=nfe,
                #     ode_solver=args.ode_solver,
                #     snapshot_index=0,
                #     file_tag=f"{args.ode_solver}_nfe{nfe}",
                # )

                recon_metrics = visualize_reconstruction_cb_batch(
                    model=model,
                    dataset=val_set,
                    epoch=epoch,
                    device=device,
                    save_dir=str(recon_dir_epoch),

                    cond_fields=args.vis_cond_fields,
                    n_obs=args.vis_n_obs_list,
                    n_steps=nfe,
                    ode_solver=args.ode_solver,
                    snapshot_index=0,
                    file_tag=f"{args.ode_solver}_nfe{nfe}",
                    save_metrics_json = True,
                )

                metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
                print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")

        logger.log_and_plot(
            epoch=epoch,
            train_loss=tr_loss,
            val_loss=val_loss,
            train_coh_loss=tr_metrics.get("coh_loss", 0.0),
            L_same=tr_metrics.get("L_same", 0.0),
            L_crossfreq=tr_metrics.get("L_crossfreq", 0.0),
            train_total_loss=tr_metrics.get("total_loss", tr_loss),
            coh_weight=coh_weight,
        )

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()
