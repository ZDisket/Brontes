#!/usr/bin/env python3
"""
Brontes training script with GAN support.

This is a training script for the Brontes model with optional GAN components.
It focuses on reconstruction loss, STFT component losses, and adversarial training.
The script supports both standard audio-to-audio transformation and GAN-based training with
Multi-Period and Multi-Scale discriminators. It also includes optional linear 
learning rate warmup for stable training initialization.
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import yaml
import torch
import numpy as np
torch.backends.cudnn.benchmark = True
import math
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    from .brontes import Brontes
    from .audio_pair_dataset import AudioPairDataset, collate_fn
    from .utils import scan_checkpoint, load_checkpoint, save_checkpoint
    from .audio_loss import AudioLoss
    from .discriminators import Discriminator, DiscriminatorLoss, feature_loss
except ImportError:
    from brontes import Brontes
    from audio_pair_dataset import AudioPairDataset, collate_fn
    from utils import scan_checkpoint, load_checkpoint, save_checkpoint
    from audio_loss import AudioLoss
    from discriminators import Discriminator, DiscriminatorLoss, feature_loss


def process_audio_in_chunks(model, full_waveform, chunk_size, batch_size=8, hop_size=None,
                            window_type='hann', device=None, autocast=False, pad_end=True):
    """
    Process a mono audio tensor of shape (1, 1, L) with chunked inference and overlap-add.

    Args:
        model: callable; returns (recon, latent) when given (B, 1, T) float tensor in [-1, 1] or similar
        full_waveform: float tensor shaped (1, 1, L), mono only
        chunk_size: int, number of samples per chunk
        batch_size: int, max batch size for inference
        hop_size: int or None; if None, defaults to chunk_size // 2
        window_type: 'hann', 'hamming', or None; window applied to chunks before OLA
        device: torch.device or str; if None, infer from model params or tensor
        autocast: bool; if True, use autocast during forward
        pad_end: bool; if True, zero-pad tail so the last chunk is full length

    Returns:
        reconstructed_full: tensor (1, 1, L_out) ≈ original length L (trimmed if padded)
        all_latents: list of latent tensors (concatenated per batch)
    """
    assert full_waveform.dim() == 3 and full_waveform.size(0) == 1 and full_waveform.size(1) == 1, \
        "full_waveform must be shape (1, 1, L)"
    L = full_waveform.size(-1)
    if hop_size is None:
        hop_size = chunk_size // 2
    hop_size = int(hop_size)
    assert chunk_size > 0 and hop_size > 0 and hop_size <= chunk_size

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = full_waveform.device
    full_waveform = full_waveform.to(device)

    # Build analysis/synthesis window
    if window_type is None:
        win = torch.ones(chunk_size, device=device)
    elif window_type.lower() == 'hann':
        win = torch.hann_window(chunk_size, periodic=True, device=device)
    elif window_type.lower() == 'hamming':
        win = torch.hamming_window(chunk_size, periodic=True, device=device)
    else:
        raise ValueError("window_type must be 'hann', 'hamming', or None")
    win = win.view(1, 1, -1)

    # Compute frame indices
    if pad_end:
        n_frames = 1 + max(0, math.ceil((L - chunk_size) / hop_size))
        total_len = (n_frames - 1) * hop_size + chunk_size
        pad_needed = max(0, total_len - L)
        if pad_needed > 0:
            full_waveform = F.pad(full_waveform, (0, pad_needed))
        L_proc = full_waveform.size(-1)
    else:
        if L < chunk_size:
            # Force one frame with padding to chunk_size
            pad_needed = chunk_size - L
            full_waveform = F.pad(full_waveform, (0, pad_needed))
            L_proc = chunk_size
            n_frames = 1
        else:
            n_frames = 1 + (L - chunk_size) // hop_size
            L_proc = (n_frames - 1) * hop_size + chunk_size  # last covered sample

    # Prepare output buffers
    out_len = L_proc
    acc = torch.zeros(1, 1, out_len, device=device)
    weight = torch.zeros(1, 1, out_len, device=device)

    # Batch through frames
    starts = [i * hop_size for i in range(n_frames)]
    all_latents = []

    model_was_training = False
    model.eval()

    with torch.no_grad():
        i = 0
        while i < n_frames:
            j = min(i + batch_size, n_frames)
            batch_starts = starts[i:j]

            # Build the batch of input chunks
            batch_chunks = []
            for s in batch_starts:
                chunk = full_waveform[:, :, s:s + chunk_size]
                if chunk.size(-1) < chunk_size:
                    # tail safety (shouldn’t happen with pad_end=True, but keep robust)
                    chunk = F.pad(chunk, (0, chunk_size - chunk.size(-1)))
                batch_chunks.append(chunk)
            batch_in = torch.cat(batch_chunks, dim=0)  # (B, 1, chunk_size)

            # Optional input windowing to smooth boundaries (analysis window)
            batch_in_win = batch_in * win

            # Inference
            if autocast:
                from torch.cuda.amp import autocast as amp_autocast
                with amp_autocast():
                    # Brontes returns (recon, latent, original_components, processed_components)
                    recon, latent, _, _ = model(batch_in_win)
            else:
                # Brontes returns (recon, latent, original_components, processed_components)
                recon, latent, _, _ = model(batch_in_win)

            # Ensure recon length matches chunk_size
            if recon.size(-1) != chunk_size:
                if recon.size(-1) > chunk_size:
                    recon = recon[..., :chunk_size]
                else:
                    recon = F.pad(recon, (0, chunk_size - recon.size(-1)))

            # Synthesis window
            recon = recon * win

            # Overlap-add into accumulator
            for b, s in enumerate(batch_starts):
                acc[:, :, s:s + chunk_size] += recon[b:b+1]
                weight[:, :, s:s + chunk_size] += win

            # Collect latents if available (Brontes returns None for latent by default)
            try:
                if isinstance(latent, torch.Tensor):
                    all_latents.append(latent.detach().cpu())
            except:
                pass

            i = j

    # Window compensation to avoid gain changes
    eps = 1e-8
    reconstructed = acc / torch.clamp(weight, min=eps)

    # Trim any padding to original L if pad_end=True
    if pad_end and reconstructed.size(-1) > L:
        reconstructed = reconstructed[..., :L]

    if model_was_training:
        model.train()

    # Latent concatenation best-effort
    try:
        if all_latents:
            latents = torch.cat([x for x in all_latents if isinstance(x, torch.Tensor)], dim=0)
        else:
            latents = None
    except:
        latents = None

    return reconstructed, latents


def get_param_num(model):
    """Get the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters())


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config, device):
    """Create Brontes model from configuration."""
    model_config = config['model']
    unet_config = model_config.get('unet_config', {})
    
    model = Brontes(
        unet_config=unet_config
    ).to(device)
    
    return model


def create_discriminators(config, device):
    """Create unified discriminator from configuration."""
    training_config = config['training']
    dataset_config = config['dataset']
    use_se_blocks = training_config.get('use_se_blocks', False)
    sample_rate = dataset_config.get('sample_rate', 44100)
    
    # Get MultiBandSpecDiscriminator config options with defaults
    mbsd_window_lengths = training_config.get('mbsd_window_lengths', [2048, 1024, 512])
    mbsd_hop_factor = training_config.get('mbsd_hop_factor', 0.25)
    
    # Get enable flags for individual discriminators (all enabled by default)
    enable_mpd = training_config.get('enable_mpd', True)
    enable_msd = training_config.get('enable_msd', True)
    enable_mbsd = training_config.get('enable_mbsd', True)
    
    # Instance noise for stabilizing G/D balance
    instance_noise_std = training_config.get('disc_instance_noise_std', 0.0)
    
    # Per-discriminator channel scaling factors (1.0 = default size)
    mpd_ch_scale = training_config.get('mpd_ch_scale', 1.0)
    msd_ch_scale = training_config.get('msd_ch_scale', 1.0)
    mbsd_ch_scale = training_config.get('mbsd_ch_scale', 1.0)
    
    discriminator = Discriminator(
        use_se_blocks=use_se_blocks,
        mbsd_window_lengths=mbsd_window_lengths,
        mbsd_hop_factor=mbsd_hop_factor,
        sample_rate=sample_rate,
        bands=None,  # Use default bands
        audio_channels=1,  # Mono
        enable_mpd=enable_mpd,
        enable_msd=enable_msd,
        enable_mbsd=enable_mbsd,
        instance_noise_std=instance_noise_std,
        mpd_ch_scale=mpd_ch_scale,
        msd_ch_scale=msd_ch_scale,
        mbsd_ch_scale=mbsd_ch_scale,
    ).to(device)
    
    return discriminator


def create_generator_optimizer(model, training_config):
    """Create generator optimizer with optional refiner parameter group support.
    
    If the model has a refiner, creates separate param groups with different learning rates.
    """
    if hasattr(model, 'has_refiner') and model.has_refiner:
        main_params = []
        refiner_params = []
        
        for name, param in model.named_parameters():
            if name.startswith('refiner.'):
                refiner_params.append(param)
            else:
                main_params.append(param)
        
        param_groups = [
            {'params': main_params, 'lr': training_config['learning_rate']},
            {'params': refiner_params, 'lr': training_config['learning_rate'] / 1.0}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=training_config['learning_rate'],
            betas=(training_config['adam_b1'], training_config['adam_b2'])
        )
        return optimizer, True  # has_refiner=True
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            betas=(training_config['adam_b1'], training_config['adam_b2'])
        )
        return optimizer, False  # has_refiner=False


def create_checkpoint_dict(model, optimizer_g, global_step, epoch, best_val_loss, 
                           has_reset_optimizer, use_adversarial=False, 
                           discriminator=None, optimizer_d=None):
    """Create checkpoint dictionary with optional discriminator state."""
    checkpoint_dict = {
        'model': model.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'step': global_step,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'has_reset_optimizer': has_reset_optimizer
    }
    
    if use_adversarial and discriminator is not None:
        checkpoint_dict.update({
            'discriminator': discriminator.state_dict(),
            'optimizer_d': optimizer_d.state_dict() if optimizer_d else None
        })
    
    return checkpoint_dict


def compute_adversarial_step(discriminator, y, y_g_hat, pretraining, training_config, device, disc_loss_fn):
    """Compute discriminator and generator adversarial losses.
    
    Args:
        discriminator: The unified discriminator model
        y: Real audio
        y_g_hat: Generated audio
        pretraining: Whether in pretraining mode
        training_config: Training configuration dict
        device: Torch device
        disc_loss_fn: DiscriminatorLoss instance for loss computation
    
    Returns:
        loss_disc_all: Total discriminator loss
        loss_gen_adv: Generator adversarial loss  
        loss_feat_match: Feature matching loss
        loss_details: Dict with individual losses for logging
    """
    loss_details = {}
    
    if not pretraining:
        # Discriminator loss (with detached generator output)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y, y_g_hat.detach())
        loss_disc_all, d_real_losses, d_fake_losses = disc_loss_fn.discriminator_loss(y_d_rs, y_d_gs)
        
        # Generator losses (non-detached for generator training)
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(y, y_g_hat)
        feature_matching_weight = training_config.get('feature_matching_weight', 2.0)
        loss_feat_match = feature_loss(fmap_rs, fmap_gs) * feature_matching_weight
        loss_gen_adv, gen_losses = disc_loss_fn.generator_loss(y_d_gs)
        
        # Store individual losses for logging
        loss_details = {
            'd_real_losses': d_real_losses,  # List of per-discriminator real losses
            'd_fake_losses': d_fake_losses,  # List of per-discriminator fake losses
            'gen_losses': [l.item() if hasattr(l, 'item') else l for l in gen_losses],  # List of per-discriminator gen losses
        }
    else:
        loss_disc_all = torch.tensor(0.0, device=device)
        loss_gen_adv = torch.tensor(0.0, device=device)
        loss_feat_match = torch.tensor(0.0, device=device)
    
    return loss_disc_all, loss_gen_adv, loss_feat_match, loss_details


def backward_with_clipping(loss, optimizer, parameters, gradient_clip, writer, 
                           global_step, metric_name, scaler=None, use_fp16=False):
    """Perform backward pass with optional gradient scaling and clipping.
    
    Args:
        loss: The loss tensor to backpropagate
        optimizer: The optimizer to step
        parameters: Model parameters for gradient clipping
        gradient_clip: Maximum gradient norm for clipping (0.0 to disable)
        writer: TensorBoard writer for logging
        global_step: Current training step
        metric_name: Name for the gradient norm metric (e.g., 'grad_norm', 'disc_grad_norm')
        scaler: GradScaler for FP16 training (optional)
        use_fp16: Whether to use FP16 gradient scaling
    """
    # Convert parameters to list so it can be iterated multiple times
    parameters = list(parameters)
    
    optimizer.zero_grad()
    
    if use_fp16 and scaler is not None:
        scaler.scale(loss).backward()
        if gradient_clip > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf')).item()
            writer.add_scalar(f'train/{metric_name}_pre_clip', grad_norm, global_step)
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if gradient_clip > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf')).item()
            writer.add_scalar(f'train/{metric_name}_pre_clip', grad_norm, global_step)
            torch.nn.utils.clip_grad_norm_(parameters, gradient_clip)
        optimizer.step()


def create_datasets(config, args):
    """Create training and validation datasets."""
    dataset_config = config['dataset']
    
    # Get quality filter path if provided (handle argparse hyphen to underscore conversion)
    quality_filter_path = getattr(args, 'quality_filter', None)
    
    # First, get the list of file pairs to split them (with quality filter applied)
    temp_dataset = AudioPairDataset(
        input_dir=args.train_input_dir,
        output_dir=args.train_output_dir,
        chunk_size=dataset_config['chunk_size'],  # This is just to find file pairs
        sample_rate=dataset_config['sample_rate'],
        file_extensions=dataset_config.get('file_extensions', None),
        normalize=dataset_config.get('normalize', True),
        mono=dataset_config.get('mono', True),
        min_samples=dataset_config.get('min_samples', 12000),
        quality_filter_path=quality_filter_path
    )
    
    # Get the file pairs list
    file_pairs = temp_dataset.file_pairs
    total_size = len(file_pairs)
    
    # Calculate split sizes
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size
    
    # Shuffle the pairs with a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    shuffled_pairs = file_pairs.copy()
    rng.shuffle(shuffled_pairs)
    
    # Split the file pairs
    train_pairs = shuffled_pairs[:train_size]
    val_pairs = shuffled_pairs[train_size:]
    
    # Create training dataset with chunking
    train_dataset = AudioPairDataset(
        input_dir=args.train_input_dir,
        output_dir=args.train_output_dir,
        chunk_size=dataset_config['chunk_size'],  # Chunked for training
        sample_rate=dataset_config['sample_rate'],
        file_extensions=dataset_config.get('file_extensions', None),
        normalize=dataset_config.get('normalize', True),
        mono=dataset_config.get('mono', True),
        min_samples=dataset_config.get('min_samples', 12000)
    )
    # Override the file_pairs with the split pairs
    train_dataset.file_pairs = train_pairs
    
    # Create validation dataset with full files
    val_dataset = AudioPairDataset(
        input_dir=args.train_input_dir,
        output_dir=args.train_output_dir,
        chunk_size=None,  # Full files for validation
        sample_rate=dataset_config['sample_rate'],
        file_extensions=dataset_config.get('file_extensions', None),
        normalize=dataset_config.get('normalize', True),
        mono=dataset_config.get('mono', True),
        min_samples=dataset_config.get('min_samples', 12000)
    )
    # Override the file_pairs with the validation split pairs
    val_dataset.file_pairs = val_pairs
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    print(f"Training uses chunked audio (chunk_size={dataset_config['chunk_size']})")
    print(f"Validation uses full audio files (chunk_size=None)")
    
    return train_dataset, val_dataset


def is_silent_batch(x, threshold=1e-6):
    """Detect silent samples in a batch based on energy threshold.
    
    Args:
        x (torch.Tensor): Audio batch of shape (batch_size, channels, time)
        threshold (float): Energy threshold below which samples are considered silent
        
    Returns:
        torch.Tensor: Boolean tensor of shape (batch_size,) indicating silent samples
    """
    # Compute energy for each sample in the batch
    energy = torch.mean(x**2, dim=(1, 2))  # Mean over channels and time
    return energy < threshold


def compute_losses(x, y_g_hat, loss_config, silent_mask=None):
    """Compute all losses for the Brontes model.
    
    Args:
        x (torch.Tensor): Original audio
        y_g_hat (torch.Tensor): Reconstructed audio from Brontes model
        loss_config (dict): Configuration for loss weights
        silent_mask (torch.Tensor, optional): Boolean mask for silent samples. Defaults to None.
    """
    # Ensure shapes match
    if y_g_hat.shape[-1] != x.shape[-1]:
        if y_g_hat.shape[-1] < x.shape[-1]:
            padding = x.shape[-1] - y_g_hat.shape[-1]
            y_g_hat = F.pad(y_g_hat, (0, padding))
        else:
            y_g_hat = y_g_hat[..., :x.shape[-1]]
    
    # Get waveform loss config options
    use_waveform_loss = loss_config.get('use_waveform_loss', False)
    waveform_loss_type = loss_config.get('waveform_loss_type', 'mse')
    waveform_loss_weight = loss_config.get('waveform_loss_weight', 1.0)
    waveform_charbonnier_eps = loss_config.get('waveform_loss_charbonnier_eps', 1e-6)
    
    # Compute waveform-domain reconstruction loss based on loss type
    if waveform_loss_type == 'mse':
        recon_loss = F.mse_loss(y_g_hat, x, reduction='none')
    elif waveform_loss_type == 'mae':
        recon_loss = F.l1_loss(y_g_hat, x, reduction='none')
    elif waveform_loss_type == 'charbonnier':
        diff = y_g_hat - x
        recon_loss = torch.sqrt(diff * diff + waveform_charbonnier_eps)
    else:
        raise ValueError(f"Unknown waveform_loss_type: {waveform_loss_type}. Use 'mse', 'mae', or 'charbonnier'.")
    
    # Average over channel and time dimensions
    recon_loss = recon_loss.mean(dim=(1, 2))
    
    # Apply silent mask if provided
    if silent_mask is not None:
        w = (~silent_mask).float()  # [B], 1 for non-silent, 0 for silent
        denom = w.sum().clamp_min(1.0)  # keep it tensor-y; avoids div-by-zero

        recon_loss = (recon_loss * w).sum() / denom
    else:
        recon_loss = recon_loss.mean()
    
    # Apply waveform loss weight (or disable if use_waveform_loss is False)
    if use_waveform_loss:
        total_loss = waveform_loss_weight * recon_loss
    else:
        total_loss = 0.0 * recon_loss
    
    return {
        'total': total_loss,
        'recon': recon_loss
    }


def validate(model, val_loader, device, writer, step, training_config, dataset_config, audio_loss_fn=None, log_static_audio=True):
    """Run validation and log results.
    
    Args:
        log_static_audio: If True, logs input and target audios (first validation only).
                         If False, only logs the inferenced audio which changes each validation.
    """
    model.eval()
    val_losses = {'total': 0, 'recon': 0}
    val_audio_losses = {}
    
    # Determine precision settings
    use_bf16 = training_config.get('bf16_run', False)
    use_fp16 = training_config.get('fp16_run', False) and not use_bf16
    use_mixed_precision = use_bf16 or use_fp16
    
    # Get the training chunk size for efficiency with non-logged samples
    chunk_size = dataset_config['chunk_size']
    num_log = 20
    max_eval = num_log

    with torch.no_grad():
        # Create progress bar for validation
        val_pbar = tqdm(enumerate(val_loader), 
                       desc="Validation", 
                       total=min(len(val_loader), max_eval),
                       leave=False)
        
        for i, batch in val_pbar:

            # For AudioPairDataset, we have input and output keys
            x = batch['input'].to(device, non_blocking=True)  # Input audio
            y = batch['output'].to(device, non_blocking=True)  # Target/output audio
            
            # Use the input audio for validation (model processes input to try to match output)
            validation_audio = x  # FIXED: Use input x instead of target y
            
            # Detect silent samples at the chunked size for loss calculation
            if validation_audio.shape[-1] > chunk_size:
                # Take the first chunk_size samples if audio is too long
                x_for_silence_check = validation_audio[:, :chunk_size]
            elif validation_audio.shape[-1] < chunk_size:
                # Pad with zeros if audio is too short
                padding = chunk_size - validation_audio.shape[-1]
                x_for_silence_check = torch.nn.functional.pad(validation_audio, (0, padding))
            else:
                # Already the right size
                x_for_silence_check = validation_audio
            
            # Check if sample is silent
            silent_mask = is_silent_batch(x_for_silence_check)
            if silent_mask.any():
                continue  # Skip this sample if it's silent
            
            # For the first 10 samples, just process them for logging purposes (no loss calculation)
            if i < num_log:

                # Process the full audio for logging samples (for meaningful comparison)
                x_processed = validation_audio  # Now x_processed is the input audio
                
                # Forward pass based on precision settings for full audio (for logging only)
                y_g_hat, latent = process_audio_in_chunks(model, x_processed, chunk_size, 16, device=device)
                
                # For logging, we use the full audio - compare model output from input to target
                # Only log input and target on first validation (they don't change between validations)
                if log_static_audio:
                    writer.add_audio(f'val_input/sample_{i}', x[0, 0], step, dataset_config['sample_rate'])
                    writer.add_audio(f'val_target/sample_{i}', y[0, 0], step, dataset_config['sample_rate'])
                # Always log the inferenced/reconstructed audio (changes each validation)
                writer.add_audio(f'val_recon/sample_{i}', y_g_hat[0, 0], step, dataset_config['sample_rate'])

            
            # For all samples (including logged ones), process in chunked format for loss calculation
            # This ensures we get consistent loss metrics even for the samples we log
            if validation_audio.shape[-1] > chunk_size:
                # Take the first chunk_size samples if audio is too long
                x_for_loss = validation_audio[:, :chunk_size]  # x_for_loss is now based on input x
            elif validation_audio.shape[-1] < chunk_size:
                # Pad with zeros if audio is too short
                padding = chunk_size - validation_audio.shape[-1]
                x_for_loss = torch.nn.functional.pad(validation_audio, (0, padding))
            else:
                # Already the right size
                x_for_loss = validation_audio  # x_for_loss is now based on input x

            audio_loss_dict = None
            # Forward pass for loss calculation using chunked audio
            if use_mixed_precision:
                with autocast(enabled=True, dtype=torch.bfloat16 if use_bf16 else None):
                    y_g_hat_for_loss, latent, _, _ = model(x_for_loss)
                    losses = compute_losses(
                        y, y_g_hat_for_loss,  # FIXED: Compare target y with model output from x
                        training_config, is_silent_batch(x_for_loss)
                    )
                    if audio_loss_fn is not None:
                        try:
                            audio_loss, audio_loss_dict = audio_loss_fn(y, y_g_hat_for_loss)  # FIXED: Compare target y with model output from x
                        except RuntimeError:
                            audio_loss_dict = None
            else:
                y_g_hat_for_loss, latent, _, _ = model(x_for_loss)
                losses = compute_losses(
                    y, y_g_hat_for_loss,  # FIXED: Compare target y with model output from x
                    training_config, is_silent_batch(x_for_loss)
                )
                if audio_loss_fn is not None:
                    try:
                        audio_loss, audio_loss_dict = audio_loss_fn(y, y_g_hat_for_loss)  # FIXED: Compare target y with model output from x
                    except RuntimeError:
                        audio_loss_dict = None


            # Accumulate losses (from chunked processing)
            for key in val_losses:
                val_losses[key] += losses[key].item()
            
            # Accumulate audio losses
            if audio_loss_dict is not None:
                for key, value in audio_loss_dict.items():
                    try:
                        if key not in val_audio_losses:
                            val_audio_losses[key] = 0.0
                        val_audio_losses[key] += value
                    except TypeError:
                        val_audio_losses = None
            else:
                val_audio_losses = None

            if i >= max_eval:
                break
        # Close validation progress bar
        val_pbar.close()

    # Average the losses (only average over the actual evaluated samples)
    actual_eval_samples = min(len(val_loader), max_eval)
    for key in val_losses:
        val_losses[key] /= max(actual_eval_samples, 1)  # Avoid division by zero
        
    if val_audio_losses is not None:
        for key in val_audio_losses:
            val_audio_losses[key] /= max(actual_eval_samples, 1)
        val_losses.update({f'audio_{k}': v for k, v in val_audio_losses.items()})


    return val_losses


def load_checkpoint_with_options(checkpoint_dir, prefix, device, model, optimizer_g=None, 
                                  discriminator=None, optimizer_d=None, load_optimizers=True,
                                  use_adversarial=False):
    """Load checkpoint from directory with optional optimizer state loading.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Checkpoint filename prefix (e.g., 'brontes_')
        device: Torch device
        model: Model to load state into
        optimizer_g: Generator optimizer (optional, only loaded if load_optimizers=True)
        discriminator: Discriminator model (optional)
        optimizer_d: Discriminator optimizer (optional, only loaded if load_optimizers=True)
        load_optimizers: Whether to load optimizer states
        use_adversarial: Whether adversarial training is enabled
    
    Returns:
        dict with keys: 'loaded', 'checkpoint_path', 'step', 'epoch', 'best_val_loss', 'has_reset_optimizer'
    """
    result = {
        'loaded': False,
        'checkpoint_path': None,
        'step': 0,
        'epoch': 0,
        'best_val_loss': float('inf'),
        'has_reset_optimizer': False
    }
    
    cp = scan_checkpoint(checkpoint_dir, prefix)
    if not cp:
        return result
    
    state_dict = load_checkpoint(cp, device)
    model.load_state_dict(state_dict['model'])
    
    # Load optimizer states if requested
    if load_optimizers:
        if optimizer_g is not None and 'optimizer_g' in state_dict:
            optimizer_g.load_state_dict(state_dict['optimizer_g'])
        if use_adversarial and optimizer_d is not None and 'optimizer_d' in state_dict:
            optimizer_d.load_state_dict(state_dict['optimizer_d'])
    
    # Load discriminator if adversarial training is enabled
    if use_adversarial and discriminator is not None:
        if 'discriminator' in state_dict:
            discriminator.load_state_dict(state_dict['discriminator'])
        elif 'mpd' in state_dict and 'msd' in state_dict:
            # Load from legacy format into unified discriminator
            discriminator.mpd.load_state_dict(state_dict['mpd'])
            discriminator.msd.load_state_dict(state_dict['msd'])
    
    # Extract metadata
    result['loaded'] = True
    result['checkpoint_path'] = cp
    result['step'] = state_dict.get('step', 0)
    result['epoch'] = state_dict.get('epoch', 0)
    result['best_val_loss'] = state_dict.get('best_val_loss', float('inf'))
    result['has_reset_optimizer'] = state_dict.get('has_reset_optimizer', False)
    
    return result


def get_lr_scale(step, warmup_steps, learning_rate):
    """Calculate learning rate scale for linear warmup."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return 1.0


def train(args):
    """Main training function."""
    # Load configuration
    config = load_config(args.config)
    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']
    paths_config = config['paths']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check BF16 support
    if training_config.get('bf16_run', False):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            print("Warning: BF16 is not supported on this device. Falling back to FP32.")
            training_config['bf16_run'] = False
        else:
            print("Using BF16 training")
    elif training_config['fp16_run']:
        print("Using FP16 training")
    else:
        print("Using FP32 training")
    
    # Create model
    model = create_model(config, device)
    param_count = get_param_num(model)
    print(f"Number of parameters: {param_count / 1e6:.2f}M")
    
    # Initialize AudioLoss
    audio_loss_fn = AudioLoss(training_config, device).to(device)
    
    # Setup generator optimizer (handles refiner parameter groups automatically)
    optimizer_g, has_refiner = create_generator_optimizer(model, training_config)
    if has_refiner:
        print(f"Using separate learning rates: Main model = {training_config['learning_rate']}, Refiner = {training_config['learning_rate'] / 1.0}")
    else:
        print(f"Using single learning rate for all parameters: {training_config['learning_rate']}")
    
    # Create discriminators if adversarial training is enabled
    use_adversarial = training_config.get('use_adversarial', False)
    discriminator, optimizer_d, disc_loss_fn = None, None, None
    if use_adversarial:
        discriminator = create_discriminators(config, device)
        # Print discriminator parameter breakdown
        disc_total = sum(p.numel() for p in discriminator.parameters())
        print(f"Discriminator total parameters: {disc_total / 1e6:.2f}M")
        if discriminator.mpd is not None:
            mpd_params = sum(p.numel() for p in discriminator.mpd.parameters())
            print(f"  MPD: {mpd_params / 1e6:.2f}M")
        if discriminator.msd is not None:
            msd_params = sum(p.numel() for p in discriminator.msd.parameters())
            print(f"  MSD: {msd_params / 1e6:.2f}M")
        if discriminator.mbsds is not None:
            mbsd_params = sum(p.numel() for p in discriminator.mbsds.parameters())
            print(f"  MBSD: {mbsd_params / 1e6:.2f}M")
        # Create discriminator loss function (hinge or lsgan, default hinge)
        disc_loss_type = training_config.get('disc_loss_type', 'hinge')
        disc_loss_fn = DiscriminatorLoss(loss_type=disc_loss_type)
        print(f"Using {disc_loss_type.upper()} discriminator loss")
        # Get discriminator learning rate multiplier from config, default to 4.0
        disc_lr_multiplier = training_config.get('discriminator_lr_multiplier', 4.0)
        optimizer_d = torch.optim.AdamW(
            discriminator.parameters(),
            lr=training_config['learning_rate'] * disc_lr_multiplier,
            betas=(training_config['adam_b1'], training_config['adam_b2'])
        )
    
    # Setup schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, 
        gamma=training_config['lr_decay']
    )
    
    scheduler_d = None
    if use_adversarial:
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_d,
            gamma=training_config['lr_decay']
        )
    
    # Setup mixed precision
    scaler = GradScaler(enabled=training_config['fp16_run'])
    
    # Setup checkpoint directory - use command line overrides if provided
    # This must be done before checkpoint loading logic
    log_dir = args.log_dir if args.log_dir else paths_config['log_dir']
    # If log_dir is overridden, default checkpoint_dir to log_dir/checkpoints
    # Otherwise use the config checkpoint_dir
    if args.log_dir:
        checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    else:
        checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else paths_config['checkpoint_dir']
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Handle checkpoint loading with priority: explicit checkpoint_path > pretrained > auto-resume
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    has_reset_optimizer = False
    
    # Priority 1: Explicit checkpoint path (full resume with optimizer state)
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        result = load_checkpoint_with_options(
            args.checkpoint_path, 'brontes_', device, model,
            optimizer_g=optimizer_g, discriminator=discriminator, optimizer_d=optimizer_d,
            load_optimizers=True, use_adversarial=use_adversarial
        )
        if result['loaded']:
            start_step = result['step']
            start_epoch = result['epoch']
            best_val_loss = result['best_val_loss']
            has_reset_optimizer = result['has_reset_optimizer']
            print(f"Resuming from checkpoint: {result['checkpoint_path']}")
            if has_reset_optimizer:
                print("Generator optimizer was previously reset after pretraining")
    
    # Priority 2: Pretrained model path (model weights only, for fine-tuning)
    elif args.pretrained and os.path.exists(args.pretrained):
        result = load_checkpoint_with_options(
            args.pretrained, 'brontes_', device, model,
            optimizer_g=None, discriminator=discriminator, optimizer_d=None,
            load_optimizers=False, use_adversarial=use_adversarial
        )
        if result['loaded']:
            print(f"Loaded pretrained model from: {result['checkpoint_path']}")
            print("Starting fine-tuning with fresh optimizer state")
    
    # Priority 3: Auto-resume from checkpoint_dir if it contains checkpoints
    elif checkpoint_dir and os.path.exists(checkpoint_dir):
        result = load_checkpoint_with_options(
            checkpoint_dir, 'brontes_', device, model,
            optimizer_g=optimizer_g, discriminator=discriminator, optimizer_d=optimizer_d,
            load_optimizers=True, use_adversarial=use_adversarial
        )
        if result['loaded']:
            start_step = result['step']
            start_epoch = result['epoch']
            best_val_loss = result['best_val_loss']
            has_reset_optimizer = result['has_reset_optimizer']
            print(f"Auto-resuming from checkpoint: {result['checkpoint_path']}")
            if has_reset_optimizer:
                print("Generator optimizer was previously reset after pretraining")
    else:
        print("Starting training from scratch")
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_datasets(config, args)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=training_config.get('shuffle', True),
        num_workers=training_config['num_workers'],
        pin_memory=training_config.get('pin_memory', True),
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Use batch_size=1 for validation
        shuffle=False,
        num_workers=training_config['num_workers'],  # Use same num_workers as training
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn  # Use same collate function as training since it's also AudioPairDataset
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup TensorBoard writer (log_dir and checkpoint_dir already defined earlier)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    model.train()
    global_step = start_step
    
    # Get adversarial training parameters
    pretrain_steps = training_config.get('pretrain_steps', 10000)
    pretrain_reset = training_config.get('pretrain_reset', False)
    
    # Track if we've already reset the optimizer
    has_reset_optimizer = False
    
    # Track if this is the first validation (to log static audios only once)
    first_validation = True
    
    print("Starting training...")
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, training_config['num_epochs']), 
                      desc="Training Progress", 
                      initial=start_epoch, 
                      total=training_config['num_epochs'])
    
    for epoch in epoch_pbar:
        epoch_start = time.time()
        epoch_losses = {'total': 0, 'recon': 0}
        if use_adversarial:
            epoch_losses.update({
                'gen_adv': 0, 'feat_match': 0, 'disc_adv': 0
            })
        
        # Create progress bar for batches within this epoch
        batch_pbar = tqdm(enumerate(train_loader), 
                         desc=f"Epoch {epoch+1}", 
                         total=len(train_loader),
                         leave=False)
        
        for batch_idx, batch in batch_pbar:
            batch_start = time.time()
            
            # Get data - AudioPairDataset returns 'input' and 'output' keys
            x = batch['input'].to(device, non_blocking=True)  # Input audio
            y = batch['output'].to(device, non_blocking=True)  # Target/output audio (for reference, but we'll compute loss against model output)
            
            # Detect silent samples
            silent_mask = is_silent_batch(x)
            
            # Determine precision settings
            use_bf16 = training_config.get('bf16_run', False)
            use_fp16 = training_config.get('fp16_run', False) and not use_bf16
            use_mixed_precision = use_bf16 or use_fp16
            
            # Check if we're in pretraining mode
            pretraining = global_step < pretrain_steps
            
            # Check if we just transitioned from pretraining to adversarial training
            if use_adversarial and pretrain_reset and not pretraining and not has_reset_optimizer:
                print(f"Step {global_step}: Resetting generator optimizer after pretraining")
                optimizer_g, has_refiner = create_generator_optimizer(model, training_config)
                if has_refiner:
                    print(f"After pretraining reset: Main model = {training_config['learning_rate']}, Refiner = {training_config['learning_rate'] / 1.0}")
                else:
                    print(f"After pretraining reset: Single learning rate for all parameters: {training_config['learning_rate']}")
                has_reset_optimizer = True
            
            # Apply learning rate warmup if needed
            lr_warmup_steps = training_config.get('lr_warmup_steps', 0)
            if lr_warmup_steps > 0:
                lr_scale = get_lr_scale(global_step, lr_warmup_steps, training_config['learning_rate'])
                for param_group in optimizer_g.param_groups:
                    param_group['lr'] = training_config['learning_rate'] * lr_scale
                if use_adversarial and optimizer_d:
                    # Get discriminator learning rate multiplier from config, default to 4.0
                    disc_lr_multiplier = training_config.get('discriminator_lr_multiplier', 4.0)
                    for param_group in optimizer_d.param_groups:
                        param_group['lr'] = training_config['learning_rate'] * disc_lr_multiplier * lr_scale
            
            # Forward pass (autocast handles both mixed precision and FP32)
            autocast_ctx = autocast(enabled=use_mixed_precision, dtype=torch.bfloat16 if use_bf16 else None)
            with autocast_ctx:
                y_g_hat, latent, _, _ = model(x)
                
                # Compute reconstruction losses
                losses = compute_losses(y, y_g_hat, training_config, silent_mask)
                
                # Compute audio losses
                audio_loss, audio_loss_dict = audio_loss_fn(y, y_g_hat)
                
                # Adversarial training step (unified)
                if use_adversarial:
                    loss_disc_all, loss_gen_adv, loss_feat_match, adv_loss_details = compute_adversarial_step(
                        discriminator, y, y_g_hat, pretraining, training_config, device, disc_loss_fn
                    )
            
            # Combine losses
            total_loss = losses['total'] + audio_loss
            if use_adversarial:
                gen_loss_weight = training_config.get('gen_loss_weight', 1.0)
                total_loss = total_loss + loss_gen_adv * gen_loss_weight + loss_feat_match
            
            # Generator backward pass with clipping
            gen_gradient_clip = training_config.get('generator_gradient_clip', training_config.get('gradient_clip', 0.0))
            backward_with_clipping(
                total_loss, optimizer_g, model.parameters(), gen_gradient_clip,
                writer, global_step, 'grad_norm', scaler, use_fp16
            )
            
            # Discriminator backward pass (only if adversarial and not pretraining)
            if use_adversarial and not pretraining:
                disc_gradient_clip = training_config.get('discriminator_gradient_clip', training_config.get('gradient_clip', 0.0))
                backward_with_clipping(
                    loss_disc_all, optimizer_d, discriminator.parameters(), disc_gradient_clip,
                    writer, global_step, 'disc_grad_norm', scaler, use_fp16
                )
            
            # Accumulate losses for epoch average
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # Accumulate adversarial losses if enabled
            if use_adversarial:
                epoch_losses['gen_adv'] += loss_gen_adv.item()
                epoch_losses['feat_match'] += loss_feat_match.item()
                epoch_losses['disc_adv'] += loss_disc_all.item()
            
            # Update batch progress bar
            postfix_dict = {
                'Loss': f"{total_loss.item():.4f}",
                'Recon': f"{losses['recon'].item():.4f}",
                'Audio': f"{audio_loss.item():.4f}",
                'Step': global_step
            }
            if use_adversarial and not pretraining:
                postfix_dict['Gen'] = f"{loss_gen_adv.item():.4f}"
                postfix_dict['Disc'] = f"{loss_disc_all.item():.4f}"
            elif use_adversarial and pretraining:
                postfix_dict['Pretrain'] = 'Y'
                
            batch_pbar.set_postfix(postfix_dict)
            
            # Logging
            if global_step % training_config['log_interval'] == 0:
                # Update epoch progress bar description with current loss
                desc = f"Epoch {epoch+1} | Loss: {total_loss.item():.4f}"
                if use_adversarial and not pretraining:
                    desc += f" | Gen: {loss_gen_adv.item():.4f} | Disc: {loss_disc_all.item():.4f}"
                elif use_adversarial and pretraining:
                    desc += " | Pretraining"
                epoch_pbar.set_description(desc)
                epoch_pbar.refresh()  # Force refresh to show updated description
                
                print(f"Step {global_step}: "
                      f"Loss={total_loss.item():.4f}, "
                      f"Recon={losses['recon'].item():.4f}, "
                      f"Audio={audio_loss.item():.4f}, "
                      f"Time={time.time() - batch_start:.2f}s")
                if use_adversarial and not pretraining:
                    print(f"  Adversarial - Gen: {loss_gen_adv.item():.4f}, "
                          f"Feat: {loss_feat_match.item():.4f}, "
                          f"Disc: {loss_disc_all.item():.4f}")
                elif use_adversarial and pretraining:
                    print(f"  Pretraining generator ({global_step}/{pretrain_steps})")
                
                # Log to TensorBoard
                for key, value in losses.items():
                    writer.add_scalar(f'train/{key}_loss', value.item(), global_step)
                
                # Log audio losses
                for key, value in audio_loss_dict.items():
                    writer.add_scalar(f'train/audio_{key}', value, global_step)
                
                # Log combined total loss
                writer.add_scalar('train/combined_total_loss', total_loss.item(), global_step)
                
                # Log adversarial losses if enabled
                if use_adversarial:
                    if not pretraining:
                        writer.add_scalar('train/gen_adv_loss', loss_gen_adv.item(), global_step)
                        writer.add_scalar('train/feat_match_loss', loss_feat_match.item(), global_step)
                        writer.add_scalar('train/disc_adv_loss', loss_disc_all.item(), global_step)
                        
                        # Log individual discriminator losses
                        if adv_loss_details:
                            d_real = adv_loss_details.get('d_real_losses', [])
                            d_fake = adv_loss_details.get('d_fake_losses', [])
                            g_losses = adv_loss_details.get('gen_losses', [])
                            
                            # Log each discriminator's total loss (real + fake)
                            for i, (dr, df) in enumerate(zip(d_real, d_fake)):
                                writer.add_scalar(f'train/disc_{i}_loss', dr + df, global_step)
                            for i, gl in enumerate(g_losses):
                                writer.add_scalar(f'train/gen_{i}_loss', gl, global_step)
                            
                            # Log mean losses
                            if d_real and d_fake:
                                d_totals = [dr + df for dr, df in zip(d_real, d_fake)]
                                writer.add_scalar('train/disc_loss_mean', sum(d_totals) / len(d_totals), global_step)
                            if g_losses:
                                writer.add_scalar('train/gen_loss_mean', sum(g_losses) / len(g_losses), global_step)
                    writer.add_scalar('train/pretraining', 1.0 if pretraining else 0.0, global_step)
                
                # Log learning rates
                writer.add_scalar('train/generator_lr', optimizer_g.param_groups[0]['lr'], global_step)
                if use_adversarial:
                    writer.add_scalar('train/discriminator_lr', optimizer_d.param_groups[0]['lr'], global_step)
            
            # Validation
            if global_step % training_config['validation_interval'] == 0 and global_step > 0:
                val_losses = validate(
                    model, val_loader, device, writer, global_step, 
                    training_config, dataset_config, audio_loss_fn,
                    log_static_audio=first_validation
                )
                first_validation = False  # Only log input/target audios on first validation
                print(f"Validation - Total: {val_losses['total']:.4f}, "
                      f"Recon: {val_losses['recon']:.4f}")

                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    if training_config.get('save_best_only', True):
                        checkpoint_path = os.path.join(checkpoint_dir, "brontes_best")
                        checkpoint_dict = create_checkpoint_dict(
                            model, optimizer_g, global_step, epoch, best_val_loss,
                            has_reset_optimizer, use_adversarial, discriminator, optimizer_d
                        )
                        save_checkpoint(checkpoint_path, checkpoint_dict)
                        print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            
            # Regular checkpointing
            if global_step % training_config['checkpoint_interval'] == 0 and global_step > 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"brontes_{global_step:08d}")
                checkpoint_dict = create_checkpoint_dict(
                    model, optimizer_g, global_step, epoch, best_val_loss,
                    has_reset_optimizer, use_adversarial, discriminator, optimizer_d
                )
                save_checkpoint(checkpoint_path, checkpoint_dict)
                print(f"Saved checkpoint at step {global_step}")
            
            global_step += 1
        
        # Close batch progress bar
        batch_pbar.close()
        
        # End of epoch
        scheduler_g.step()
        if use_adversarial and scheduler_d:
            scheduler_d.step()
        
        # Average epoch losses
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s - "
              f"Avg Loss: {epoch_losses['total']:.4f}")
        
        # Update epoch progress bar with final epoch information
        epoch_pbar.set_postfix({
            'Avg Loss': f"{epoch_losses['total']:.4f}",
            'Time': f"{epoch_time:.1f}s"
        })
        
        # Log epoch losses
        for key, value in epoch_losses.items():
            writer.add_scalar(f'epoch/{key}_loss', value, epoch)
    
    # Close epoch progress bar
    epoch_pbar.close()
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, "brontes_final")
    checkpoint_dict = create_checkpoint_dict(
        model, optimizer_g, global_step, training_config['num_epochs'], best_val_loss,
        has_reset_optimizer, use_adversarial, discriminator, optimizer_d
    )
    save_checkpoint(final_checkpoint_path, checkpoint_dict)
    print("Training completed!")
    print(f"Final model saved to: {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Brontes Model')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--train_input_dir', type=str, required=True,
                        help='Path to directory with training input audio files')
    parser.add_argument('--train_output_dir', type=str, required=True,
                        help='Path to directory with training output audio files (target)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Ratio of training data to use for validation (default: 0.1)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to checkpoint to resume training from (loads model + optimizer)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained checkpoint for fine-tuning (loads model only, not optimizer)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Override TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Override checkpoint save directory (defaults to log_dir/checkpoints if log_dir is specified). If contains checkpoints, auto-resumes from latest.')
    parser.add_argument('--quality-filter', type=str, default=None,
                        help='Path to hi-fi quality filter file (basenames, one per line)')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
