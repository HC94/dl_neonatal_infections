"""
Model evaluation and metrics computation for survival analysis.
"""

import numpy as np
import torch
from tqdm import tqdm
from torchsurv.loss import weibull
from torchsurv.metrics.cindex import ConcordanceIndex
from torchsurv.metrics.auc import Auc
from torchsurv.metrics.brier_score import BrierScore


def evaluate_model(model, val_loader, criterion, num_bins, num_calibration_bins, progressive_evaluation, n_bootstrap,
                   confidence, seed, epsilon, device, logger):
    """
    Evaluate model and compute metrics: loss, C-index, AUC, Brier Score, and Expected Calibration Error.

    Args:
        model: The trained model
        val_loader: DataLoader for test/validation data
        criterion: Loss function
        device: torch device
        logger: Logger instance
        num_bins: Number of time bins (recommended value: 24)
        epsilon: Epsilon value for numerical stability
        progressive_evaluation: If True, evaluate using all possible prefix lengths (more comprehensive)
                                If False, use only full sequences (faster, original behavior)

    Returns:
        avg_val_loss: Average validation loss
        val_c_index: Concordance index
        val_auc: Area under the ROC curve (averaged across time points for time-dependent AUC)
        val_brier: Brier score (averaged across time points)
    """
    model.eval()

    # Initialize collection variables
    total_val_loss = 0.0
    total_val_samples = 0  # Track total number of samples for proper averaging
    all_estimates = []
    all_survival_probs = []
    all_rel_times = []
    all_rel_indicators = []
    all_batch_data = []  # Store batch data for survival probability computation (standard evaluation)

    # Progressive evaluation: iterate through samples and all prefix lengths
    if progressive_evaluation:
        logger.my_print("Using progressive evaluation mode (evaluating all prefix lengths)...")
        val_dataset = val_loader.dataset
        num_samples = len(val_dataset)
        num_prefix_predictions = 0

        with torch.no_grad():
            for sample_idx in range(num_samples):
                # Get full patient data
                X_dynamic_idx = val_dataset.X_dynamic[sample_idx]
                mask_dynamic_idx = val_dataset.mask_dynamic[sample_idx]
                dynamic_times_idx = val_dataset.dynamic_times[sample_idx]
                X_static_idx = val_dataset.X_static[sample_idx]
                mask_static_idx = val_dataset.mask_static[sample_idx]
                original_result_time_idx = val_dataset.result_times[sample_idx]
                original_result_indicator_idx = val_dataset.result_indicators[sample_idx]

                # Find all valid (non-padded) timesteps
                valid_indices = np.where(np.isfinite(dynamic_times_idx))[0]
                num_valid_steps = len(valid_indices)

                if num_valid_steps == 0:
                    continue

                # Iterate through all possible prefix lengths
                for prefix_len in range(1, num_valid_steps + 1):
                    # Get indices for this prefix in the full (padded) array
                    prefix_valid_indices = valid_indices[:prefix_len]
                    start_idx = prefix_valid_indices[0]
                    end_idx = prefix_valid_indices[-1]

                    # Extract prefix data
                    prefix_dynamic = X_dynamic_idx[start_idx:end_idx + 1, :]
                    prefix_mask = mask_dynamic_idx[start_idx:end_idx + 1, :]
                    prefix_times = dynamic_times_idx[start_idx:end_idx + 1]

                    # Make times relative to the start of this prefix
                    prefix_times_rel = prefix_times - prefix_times[0]
                    last_obs_time_abs = prefix_times[-1]

                    # Calculate relative time and indicator for this prefix
                    rel_time = original_result_time_idx - last_obs_time_abs

                    # AFT: preserve original indicator
                    rel_indicator = original_result_indicator_idx

                    # Convert to tensors and add batch dimension
                    xd = torch.from_numpy(prefix_dynamic).unsqueeze(0).to(device)
                    md = torch.from_numpy(prefix_mask).unsqueeze(0).to(device)
                    dt = torch.from_numpy(prefix_times_rel).unsqueeze(0).to(device)
                    xs = torch.from_numpy(X_static_idx).unsqueeze(0).to(device)
                    ms = torch.from_numpy(mask_static_idx).unsqueeze(0).to(device)
                    rt = torch.tensor([rel_time], dtype=torch.float32).to(device)
                    ri = torch.tensor([rel_indicator], dtype=torch.long).to(device)

                    # Forward pass and collect predictions
                    outputs = model(
                        x_dynamic=xd,
                        x_dynamic_mask=md,
                        dynamic_times=dt,
                        x_static=xs,
                        x_static_mask=ms,
                    )

                    # Compute loss and collect predictions for AFT
                    loss = criterion(outputs=outputs, result_times=rt, result_indicators=ri)
                    total_val_loss += float(loss.detach().cpu().item())

                    # Store AFT parameters as estimates
                    all_estimates.append(outputs)

                    # Compute survival probabilities
                    time_points = np.arange(1, num_bins + 1)
                    surv = compute_aft_survival_probabilities(
                        model=model,
                        x_dynamic=xd, x_dynamic_mask=md, dynamic_times=dt,
                        x_static=xs, x_static_mask=ms,
                        time_points=time_points,
                        device=device
                    )
                    all_survival_probs.append(surv)

                    # Collect relative times and indicators
                    all_rel_times.append(rt)
                    all_rel_indicators.append(ri)
                    num_prefix_predictions += 1

        # Compute average loss
        avg_val_loss = total_val_loss / max(1, num_prefix_predictions)
        logger.my_print(f"Progressive evaluation: {num_prefix_predictions} prefix predictions from {num_samples} patients")

    # Standard evaluation: iterate through batches
    else:
        with torch.no_grad():
            for i, batch_data in enumerate(val_loader):
                # Unpack common elements
                (pids, eids, Xd, Md, Td, Xs, Ms, rt, ri) = batch_data

                Xd, Md, Td, Xs, Ms, rt, ri = (Xd.to(device), Md.to(device), Td.to(device), Xs.to(device), Ms.to(device),
                                              rt.to(device), ri.to(device))
                outputs = model(
                    x_dynamic=Xd,
                    x_dynamic_mask=Md,
                    dynamic_times=Td,
                    x_static=Xs,
                    x_static_mask=Ms,
                )

                # AFT loss computation
                loss = criterion(outputs=outputs, result_times=rt, result_indicators=ri)

                # Weight loss by batch size for proper averaging across different batch sizes
                batch_size = outputs.size(0)
                total_val_loss += float(loss.detach().cpu().item()) * batch_size
                total_val_samples += batch_size
                # outputs is (B, 2) log_scale and log_shape
                all_estimates.append(outputs)

                # Store batch data for later survival probability computation
                all_batch_data.append((Xd, Md, Td, Xs, Ms))

                # Collect relative times and indicators for C-index
                all_rel_times.append(rt)
                all_rel_indicators.append(ri)

        # Compute average loss for standard evaluation (weighted by batch size)
        avg_val_loss = total_val_loss / max(1, total_val_samples)

    # Compute metrics (shared logic for both evaluation modes)
    val_c_index = None
    val_auc = None
    val_brier = None
    risk_at_horizon = None

    if len(all_estimates) > 0:
        # Concatenate all batches
        all_rel_times = torch.cat(all_rel_times)
        all_rel_indicators = torch.cat(all_rel_indicators).bool()

        # Compute C-index and survival_probabilities
        cindex_metric = ConcordanceIndex()

        # Compute risk scores at 24-hour horizon for AFT
        all_estimates = torch.cat(all_estimates)
        log_scale = all_estimates[:, 0]
        log_shape = all_estimates[:, 1]
        scale = torch.exp(log_scale)
        shape = torch.exp(log_shape)

        # Compute risk at fixed horizon (24 hours)
        t_horizon = torch.tensor(num_bins, dtype=torch.float32, device=device)  # 24 hours
        # Weibull survival: S(t) = exp(-(t/scale)^shape)
        # Risk = 1 - S(t) = 1 - exp(-(t/scale)^shape)
        survival_at_horizon = torch.exp(-torch.pow(t_horizon / scale, shape))
        risk_at_horizon = 1.0 - survival_at_horizon

        # Compute survival probabilities for AFT if needed (standard evaluation only)
        if not progressive_evaluation and len(all_batch_data) > 0:
            time_points = np.arange(1, num_bins + 1)  # Time points 1-24 hours
            surv_probs_list = []
            for (Xd, Md, Td, Xs, Ms) in all_batch_data:
                surv = compute_aft_survival_probabilities(
                    model=model,
                    x_dynamic=Xd, x_dynamic_mask=Md, dynamic_times=Td,
                    x_static=Xs, x_static_mask=Ms,
                    time_points=time_points,
                    device=device
                )
                surv_probs_list.append(surv)
            all_survival_probs = torch.cat(surv_probs_list)  # (N, num_bins)
        elif progressive_evaluation and len(all_survival_probs) > 0:
            # Progressive evaluation: survival probabilities already computed
            all_survival_probs = torch.cat(all_survival_probs)
        else:
            all_survival_probs = None

        # --- C-index: Use 24-hour risk ---
        val_c_index = cindex_metric(risk_at_horizon, all_rel_indicators, all_rel_times)
        val_c_index = float(val_c_index.cpu().item())

        # Compute AUC using risk scores (time-independent)
        try:
            auc_metric = Auc()
            auc_value = auc_metric(
                estimate=risk_at_horizon.cpu(),  # 1 - S(24)
                event=all_rel_indicators.cpu(),
                time=all_rel_times.cpu(),
            )

            # AUC may return multiple values (one per time point), so we average them
            if auc_value.numel() > 1:
                auc_value = auc_value.mean()
            val_auc = float(auc_value.cpu().item())
        except Exception as e:
            logger.my_print(f"Warning: Could not compute AUC: {e}")
            val_auc = None

        # Compute Brier Score using survival probabilities (all time points (1-24h))
        if all_survival_probs is not None and (isinstance(all_survival_probs, torch.Tensor) and all_survival_probs.numel() > 0):
            try:
                brier_metric = BrierScore()

                # Time points for Brier score evaluation
                time_points = torch.arange(1, num_bins + 1, dtype=torch.float32, device=device)

                # Compute Brier score at each time point and average
                brier_scores = brier_metric(
                    estimate=all_survival_probs.cpu(),
                    event=all_rel_indicators.cpu(),
                    time=all_rel_times.cpu(),
                    new_time=time_points.cpu()
                )
                # Average across time points
                val_brier = float(brier_scores.mean().cpu().item())
            except Exception as e:
                logger.my_print(f"Warning: Could not compute Brier Score: {e}")
                val_brier = None

    boot_results = None
    if n_bootstrap and n_bootstrap > 0:
        logger.my_print(f"Performing bootstrap (n={n_bootstrap}) for {confidence * 100:.1f}% confidence intervals...")
        boot_results = bootstrap_metrics(
            risk_at_horizon=risk_at_horizon,
            all_rel_indicators=all_rel_indicators,
            all_rel_times=all_rel_times,
            all_survival_probs=all_survival_probs,
            num_bins=num_bins,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )

        for metric, mean_std_lower_upper_dict in boot_results.items():
            mean = mean_std_lower_upper_dict['mean']
            std = mean_std_lower_upper_dict['std']
            lower = mean_std_lower_upper_dict['lower']
            upper = mean_std_lower_upper_dict['upper']
            if mean is not None:
                logger.my_print(f">>> {metric}: {mean:.4f} ± {std:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")

    return avg_val_loss, val_c_index, val_auc, val_brier, risk_at_horizon, all_rel_indicators, all_rel_times, all_survival_probs, boot_results


@torch.no_grad()
def compute_aft_survival_probabilities(model, x_dynamic, x_dynamic_mask, dynamic_times, x_static, x_static_mask,
                                       time_points, device):
    """
    Compute survival probabilities for AFT (Weibull) model at specified time points.

    Uses the Weibull survival function: S(t|x) = exp(-(t/scale)^shape)
    where scale and shape are predicted by the model.

    Args:
        model: Trained AFT model
        x_dynamic, x_dynamic_mask, dynamic_times, x_static, x_static_mask: Input features
        time_points: Time points at which to compute survival probabilities (e.g., [0, 1, 2, ..., 24])
        device: Device to use for computation

    Returns:
        survival_probs: (batch_size, len(time_points)) survival probabilities
    """
    model.eval()

    # Get log_scale and log_shape from model
    log_params = model(
        x_dynamic=x_dynamic.to(device),
        x_dynamic_mask=x_dynamic_mask.to(device),
        dynamic_times=dynamic_times.to(device),
        x_static=x_static.to(device),
        x_static_mask=x_static_mask.to(device)
    )  # (B, 2)

    if torch.isnan(log_params).any():
        return torch.zeros((x_dynamic.size(0), len(time_points)), dtype=torch.float32, device=device)  # (B, T)

    # Compute survival probabilities for each time point
    # torchsurv expects time to be a single scalar value
    survival_probs_list = []
    for t in time_points:
        # Evaluate survival at time t for all samples
        surv_at_t = weibull.survival_function(
            log_params=log_params,
            time=torch.tensor(float(t), dtype=torch.float32, device=device)
        )  # (B,)
        survival_probs_list.append(surv_at_t)

    # Stack to get (B, T)
    survival_probs = torch.stack(survival_probs_list, dim=1)  # (B, T)
    return survival_probs


def bootstrap_metrics(risk_at_horizon, all_rel_indicators, all_rel_times, all_survival_probs,
                      num_bins, n_bootstrap, confidence, seed):
    """
    Compute bootstrapped confidence intervals for survival metrics.

    Returns dict with mean, std, and CI for each metric.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(risk_at_horizon)

    # Storage for bootstrap results
    boot_c_index = []
    boot_auc = []
    boot_brier = []

    cindex_metric = ConcordanceIndex()
    auc_metric = Auc()
    brier_metric = BrierScore()

    for _ in tqdm(range(n_bootstrap)):
        # Sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        # Resample tensors
        boot_risk = risk_at_horizon[indices]
        boot_event = all_rel_indicators[indices]
        boot_time = all_rel_times[indices]

        # Skip if no events in bootstrap sample
        if boot_event.sum() == 0:
            continue

        try:
            ci = cindex_metric(boot_risk, boot_event, boot_time)
            boot_c_index.append(float(ci.cpu().item()))
        except:
            pass

        try:
            auc_val = auc_metric(
                estimate=boot_risk.cpu(),
                event=boot_event.cpu(),
                time=boot_time.cpu(),
            )
            # AUC may return multiple values (one per time point), so we average them
            if auc_val.numel() > 1:
                auc_val = auc_val.mean()

            boot_auc.append(float(auc_val.cpu().item()))
        except:
            pass

        if all_survival_probs is not None and len(all_survival_probs) > 0:
            try:
                boot_surv = all_survival_probs[indices]

                # Time points for Brier score evaluation
                time_points = torch.arange(1, num_bins + 1, dtype=torch.float32)

                # Compute Brier score at each time point and average
                brier = brier_metric(
                    estimate=boot_surv.cpu(),
                    event=boot_event.cpu(),
                    time=boot_time.cpu(),
                    new_time=time_points.cpu()
                )
                # Average across time points
                boot_brier.append(float(brier.mean().cpu().item()))
            except:
                pass

    def compute_stats(values):
        if len(values) == 0:
            return None, None, None, None
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        alpha = 1 - confidence
        lower = np.percentile(arr, alpha / 2 * 100)
        upper = np.percentile(arr, (1 - alpha / 2) * 100)
        return {'mean': mean, 'std': std, 'lower': lower, 'upper': upper}

    return {
        'c_index': compute_stats(boot_c_index),
        'auc': compute_stats(boot_auc),
        'brier': compute_stats(boot_brier),
    }
