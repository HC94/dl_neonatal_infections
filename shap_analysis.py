"""
SHAP (SHapley Additive exPlanations) analysis functions for model interpretability.
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import shap


def prepare_shap_data(data_loader, n_samples, target_seq_len, logger):
    """
    Extract a subset of data from the loader and flatten it for SHAP.

    This function collects dynamic and static features along with their masks (Md and Ms)
    to properly handle missing values. Missing values are marked as NaN in the output
    so that SHAP aggregation functions (np.nansum) correctly ignore them.

    Args:
        data_loader: DataLoader to sample from
        n_samples: Number of samples to extract
        target_seq_len: Fixed sequence length to pad/truncate to (for consistency). If None, use max length found.
        logger: Logger instance

    Returns:
        X_flat: (n_samples, total_features) flattened feature array with NaN for missing values
        sample_info: Dictionary with metadata about the samples
    """
    # Initialize variables
    patient_id_list = []
    encounter_id_list = []
    X_dynamic_list = []
    mask_dynamic_list = []  # Collect dynamic masks
    X_static_list = []
    mask_static_list = []  # Collect static masks
    dynamic_times_list = []
    rel_times_list = []
    rel_indicators_list = []

    samples_collected = 0
    for batch_data in data_loader:
        if samples_collected >= n_samples:
            break

        # Unpack batch - now using Md (mask_dynamic) and Ms (mask_static)
        (pids, eids, Xd, Md, Td, Xs, Ms, rel_times, rel_indicators, *_) = batch_data

        batch_size = Xd.shape[0]
        remaining = n_samples - samples_collected
        take = min(batch_size, remaining)

        patient_id_list.extend(pids[:take].tolist())
        encounter_id_list.extend(eids[:take].tolist())
        X_dynamic_list.append(Xd[:take].cpu())
        mask_dynamic_list.append(Md[:take].cpu())  # Collect dynamic masks
        dynamic_times_list.append(Td[:take].cpu())
        X_static_list.append(Xs[:take].cpu())
        mask_static_list.append(Ms[:take].cpu())  # Collect static masks
        rel_times_list.append(rel_times[:take].cpu())
        rel_indicators_list.append(rel_indicators[:take].cpu())

        samples_collected += take

    # Determine target sequence length
    if target_seq_len is None:
        # Find max sequence length across all collected batches
        target_seq_len = max(x.shape[1] for x in X_dynamic_list)

    logger.my_print(f">>> Target sequence length for SHAP data: {target_seq_len}")

    # Pad/truncate all tensors to the target sequence length
    X_dynamic_padded_list = []
    mask_dynamic_padded_list = []
    dynamic_times_padded_list = []

    for x_dyn, m_dyn, dyn_times in zip(X_dynamic_list, mask_dynamic_list, dynamic_times_list):
        curr_len = x_dyn.shape[1]

        if curr_len < target_seq_len:
            # Pad with NaN for features
            pad_size = target_seq_len - curr_len
            x_dyn_padded = torch.cat([
                torch.full((x_dyn.shape[0], pad_size, x_dyn.shape[2]), float('nan')),
                x_dyn
            ], dim=1)
            # Pad masks with True (= missing) for padded positions
            m_dyn_padded = torch.cat([
                torch.ones((m_dyn.shape[0], pad_size, m_dyn.shape[2]), dtype=torch.bool),
                m_dyn
            ], dim=1)
            dyn_times_padded = torch.cat([
                torch.full((dyn_times.shape[0], pad_size), float('nan')),
                dyn_times
            ], dim=1)
        elif curr_len > target_seq_len:
            # Truncate (keep the most recent timesteps - right side)
            x_dyn_padded = x_dyn[:, -target_seq_len:, :]
            m_dyn_padded = m_dyn[:, -target_seq_len:, :]
            dyn_times_padded = dyn_times[:, -target_seq_len:]
        else:
            x_dyn_padded = x_dyn
            m_dyn_padded = m_dyn
            dyn_times_padded = dyn_times

        X_dynamic_padded_list.append(x_dyn_padded)
        mask_dynamic_padded_list.append(m_dyn_padded)
        dynamic_times_padded_list.append(dyn_times_padded)

    # Concatenate
    X_dynamic = torch.cat(X_dynamic_padded_list, dim=0)  # (n_samples, seq_len, n_dynamic)
    mask_dynamic = torch.cat(mask_dynamic_padded_list, dim=0)  # (n_samples, seq_len, n_dynamic)
    dynamic_times = torch.cat(dynamic_times_padded_list, dim=0)  # (n_samples, seq_len)
    X_static = torch.cat(X_static_list, dim=0)  # (n_samples, n_static)
    mask_static = torch.cat(mask_static_list, dim=0)  # (n_samples, n_static)

    logger.my_print(f">>> SHAP data preparation: X_dynamic.shape={X_dynamic.shape}, X_static.shape={X_static.shape}")

    # Apply masks: set missing values to NaN so that np.nansum correctly ignores them
    X_dynamic_masked = X_dynamic.clone()
    X_dynamic_masked[mask_dynamic] = float('nan')
    
    X_static_masked = X_static.clone()
    X_static_masked[mask_static] = float('nan')

    logger.my_print(f">>> Applied feature masks: dynamic missing={mask_dynamic.sum().item()}, static missing={mask_static.sum().item()}")

    # Dynamic features
    # Flatten: (n_samples, seq_len, n_dynamic) -> (n_samples, seq_len * n_dynamic)
    n_samples = X_dynamic_masked.shape[0]
    seq_len = X_dynamic_masked.shape[1]
    n_dynamic = X_dynamic_masked.shape[2]
    X_dynamic_flat = X_dynamic_masked.reshape(n_samples, seq_len * n_dynamic)

    # Dynamic times
    assert dynamic_times.shape == (n_samples, seq_len)

    # Static features
    n_static = X_static_masked.shape[1]

    # Concatenate: [dynamic_features | dynamic_times | static_features]
    X_flat = torch.cat([X_dynamic_flat, dynamic_times, X_static_masked], dim=1)  # (n_samples, seq_len*n_dynamic + seq_len + n_static)

    sample_info = {
        'patient_ids': patient_id_list,
        'encounter_ids': encounter_id_list,
        'n_samples': n_samples,
        'seq_len': seq_len,
        'n_dynamic': n_dynamic,
        'n_static': n_static,
        'rel_times': rel_times_list,
        'rel_indicators': rel_indicators_list,
    }

    logger.my_print(f">>> SHAP data flattened: X_flat.shape={X_flat.shape}")
    return X_flat.numpy(), sample_info


def create_shap_model_wrapper(model, num_bins, device):
    """
    Create a wrapper function that converts flattened input to model-compatible format.

    This wrapper is needed because SHAP expects a function that takes a 2D array (samples x features),
    but our model requires structured inputs with dynamic/static features, masks, and times.

    Args:
        model: The trained TransformerSurv model
        num_bins: Number of time bins for prediction horizon
        device: torch device

    Returns:
        wrapper_fn: Function that takes flattened features and returns predictions
    """

    def wrapper_fn(X_flat):
        """
        X_flat: (n_samples, total_features) where total_features = seq_len*n_dynamic + n_static
        """
        # Convert to tensor if needed
        if not isinstance(X_flat, torch.Tensor):
            X_flat = torch.tensor(X_flat, dtype=torch.float32, device=device)
        else:
            X_flat = X_flat.to(device)

        # Extract dimensions from model
        n_samples = X_flat.shape[0]
        seq_len = wrapper_fn.seq_len
        n_dynamic = wrapper_fn.n_dynamic
        n_static = wrapper_fn.n_static

        # Reconstruct structured inputs
        dynamic_flat_size = seq_len * n_dynamic
        x_dynamic_flat = X_flat[:, :dynamic_flat_size]
        x_dynamic = x_dynamic_flat.reshape(n_samples, seq_len, n_dynamic)
        dynamic_times = X_flat[:, dynamic_flat_size:(dynamic_flat_size + seq_len)]
        x_static = X_flat[:, (dynamic_flat_size + seq_len):]

        model.eval()

        # Check if we are in SHAP explanation mode (original_valid_mask is set)
        if hasattr(wrapper_fn, 'use_single_mode') and wrapper_fn.use_single_mode:
            # Run model for each sample individually to handle variable-length sequences
            all_outputs = []
            with torch.no_grad():
                for i in range(n_samples):
                    sample_times = dynamic_times[i]
                    valid_mask = ~torch.isnan(sample_times)
                    valid_indices = torch.where(valid_mask)[0]
                    n_valid = len(valid_indices)
                    assert n_valid > 0

                    x_dyn_i = x_dynamic[i:i + 1, valid_indices, :]
                    times_i = dynamic_times[i:i + 1, valid_indices]
                    x_stat_i = x_static[i:i + 1]

                    # Create masks from NaN positions (True = missing) BEFORE replacing NaN with 0
                    x_dyn_mask_i = torch.isnan(x_dyn_i)
                    x_stat_mask_i = torch.isnan(x_stat_i)
                    # Now replace NaN with 0 for model computation (model uses masks to handle missing values)
                    x_dyn_i = torch.nan_to_num(x_dyn_i, nan=0.0)
                    x_stat_i = torch.nan_to_num(x_stat_i, nan=0.0)

                    # Run model for single patient
                    out_i = model(
                        x_dynamic=x_dyn_i,
                        x_dynamic_mask=x_dyn_mask_i,
                        dynamic_times=times_i,
                        x_static=x_stat_i,
                        x_static_mask=x_stat_mask_i
                    )
                    all_outputs.append(out_i)

            # Stack outputs
            outputs = torch.cat(all_outputs, dim=0)  # (aft) shape: (n_samples, 2)
        else:
            # Use the original sample's valid mask (stored in wrapper)
            original_valid_mask = wrapper_fn.original_valid_mask  # (seq_len,) boolean
            valid_indices = torch.where(original_valid_mask)[0]
            n_valid = len(valid_indices)
            assert n_valid > 0

            # Extract only the valid timesteps for all samples
            x_dynamic_valid = x_dynamic[:, valid_indices, :]  # (n_samples, n_valid, n_dynamic)
            dynamic_times_valid = dynamic_times[:, valid_indices]  # (n_samples, n_valid)

            # Create masks from NaN positions (True = missing) before replacing NaN with 0
            x_dynamic_mask_valid = torch.isnan(x_dynamic_valid)
            x_static_mask = torch.isnan(x_static)
            # Now replace NaN with 0 for model computation (model uses masks to handle missing values)
            x_dynamic_valid = torch.nan_to_num(x_dynamic_valid, nan=0.0)
            x_static = torch.nan_to_num(x_static, nan=0.0)

            # Single batched forward pass: all samples have same seq length
            with torch.no_grad():
                outputs = model(
                    x_dynamic=x_dynamic_valid,
                    x_dynamic_mask=x_dynamic_mask_valid,
                    dynamic_times=dynamic_times_valid,
                    x_static=x_static,
                    x_static_mask=x_static_mask
                )

        # Process outputs for AFT
        # outputs: (n_samples, 2) log_scale and log_shape
        log_scale = outputs[:, 0]
        log_shape = outputs[:, 1]
        scale = torch.exp(log_scale)
        shape = torch.exp(log_shape)
        
        # Compute risk at fixed horizon (24 hours)
        t_horizon = torch.tensor(num_bins, dtype=torch.float32, device=device)  # 24 hours
        # Weibull survival: S(t) = exp(-(t/scale)^shape)
        # Risk = 1 - S(t) = 1 - exp(-(t/scale)^shape)
        survival_at_horizon = torch.exp(-torch.pow(t_horizon / scale, shape))
        risk_at_horizon = 1.0 - survival_at_horizon
        
        return risk_at_horizon.detach().cpu().numpy()  # (Prob. of event by 24h) Higher = higher risk

    return wrapper_fn



def compute_and_plot_shap(model, train_loader, val_loader, num_bins, top_n_features,
                          shap_max_background_samples, n_samples_per_explanation, l1_reg,
                          df_dynamic_filename, df_static_filename, exp_shap_dir, device, logger):
    """
    Compute SHAP values and create visualizations.

    Args:
        model: Trained model
        train_loader: Training data loader (for background data)
        val_loader: Validation data loader (for explanation data)
        num_bins: Number of time bins for prediction horizon
        device: torch device
        logger: Logger instance
        df_dynamic_filename: Filename for dynamic features dataframe
        df_static_filename: Filename for static features dataframe
        exp_shap_dir: Directory to save SHAP files and plots
        top_n_features: Number of top features to display
        n_samples_per_explanation: nsamples for Monte Carlo samples to estimate SHAP values
        shap_max_background_samples: Maximum number of background samples to use
    """
    # Clear GPU memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.my_print("=" * 80)
    logger.my_print(f"Preparing calculating SHAP values...")
    logger.my_print("=" * 80)

    try:
        # Prepare background data (limited training set for memory efficiency)
        n_bg_samples = len(train_loader.dataset)
        if shap_max_background_samples is not None:
            n_bg_samples = min(shap_max_background_samples, n_bg_samples)
        
        logger.my_print(f"Preparing background data ({n_bg_samples} samples from training set)...")
        X_bg_full, bg_full_info = prepare_shap_data(
            data_loader=train_loader,
            n_samples=n_bg_samples,
            target_seq_len=None,
            logger=logger,
        )

        # Prepare data to explain (full validation set)
        logger.my_print("Preparing data to explain (full validation set)...")
        X_explain_full, _ = prepare_shap_data(
            data_loader=val_loader,
            n_samples=len(val_loader.dataset),
            target_seq_len=bg_full_info['seq_len'],
            logger=logger,
        )

        # Verify shapes match
        assert X_bg_full.shape[1] == X_explain_full.shape[1], \
            f"Feature dimension mismatch: X_bg_full has {X_bg_full.shape[1]} features, X_explain_full has {X_explain_full.shape[1]}"

        logger.my_print(f"Shape verification: X_bg_full={X_bg_full.shape}, X_explain_full={X_explain_full.shape}")

        # Extract dimensions
        seq_len = bg_full_info['seq_len']
        n_dynamic = bg_full_info['n_dynamic']
        n_static = bg_full_info['n_static']

        X_bg = X_bg_full  # NumPy array, required for shap.KernelExplainer()
        X_explain = X_explain_full  # NumPy array, required for shap.KernelExplainer()

        logger.my_print(f"SHAP input shapes: X_bg={X_bg.shape}, X_explain={X_explain.shape}")
        logger.my_print(f"SHAP computation device: {device}")

        # Create wrapper function
        logger.my_print("Creating model wrapper for SHAP...")
        wrapper_fn = create_shap_model_wrapper(model=model, num_bins=num_bins, device=device)

        # Store necessary info in wrapper
        wrapper_fn.seq_len = seq_len
        wrapper_fn.n_dynamic = n_dynamic
        wrapper_fn.n_static = n_static
        wrapper_fn.use_single_mode = True  # Individual mode for initialization

        # Create SHAP explainer
        logger.my_print("Creating SHAP explainer...")
        explainer = shap.KernelExplainer(wrapper_fn, X_bg)

        # Compute SHAP values with memory-efficient batching
        shap_batch_size = 1
        logger.my_print(f"Computing SHAP values (batch_size={shap_batch_size}, {len(X_explain)} samples)...")
        dynamic_flat_size = seq_len * n_dynamic
        shap_values_list = []
        
        for i in range(0, len(X_explain), shap_batch_size):
            batch = X_explain[i:i + shap_batch_size]

            # Store the valid mask from the ORIGINAL sample being explained
            original_dynamic_times = batch[0, dynamic_flat_size:(dynamic_flat_size + seq_len)]
            wrapper_fn.original_valid_mask = ~np.isnan(original_dynamic_times)
            wrapper_fn.original_valid_mask = torch.tensor(wrapper_fn.original_valid_mask, device=device)
            wrapper_fn.use_single_mode = False  # Batched mode for SHAP values

            if l1_reg is None:
                shap_batch = explainer.shap_values(batch, nsamples=n_samples_per_explanation)
            else:
                shap_batch = explainer.shap_values(batch, nsamples=n_samples_per_explanation, l1_reg=l1_reg)
            shap_values_list.append(shap_batch)

            # Clear cache after each batch for memory efficiency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        shap_values = np.vstack(shap_values_list)
        logger.my_print(f"SHAP values computed: shape={shap_values.shape}")
        
        assert shap_values.shape == X_explain.shape == (len(X_explain), dynamic_flat_size + seq_len + n_static), \
            (f"SHAP values shape mismatch: expected shap_values.shape = X_explain.shape = ({len(X_explain)}, {dynamic_flat_size} + {seq_len} + {n_static}), "
             f"got shap_values.shape = {shap_values.shape} and X_explain.shape = {X_explain.shape}")

        # --- Aggregate SHAP values across time dimension ---
        X_dynamic_shap_flat = shap_values[:, :dynamic_flat_size]  # (n_samples, seq_len * n_dynamic)
        X_static_shap = shap_values[:, (dynamic_flat_size + seq_len):]  # (n_samples, n_static)

        # Reshape dynamic SHAP values: (n_samples, seq_len * n_dynamic) -> (n_samples, seq_len, n_dynamic)
        X_dynamic_shap_flat = X_dynamic_shap_flat.reshape(-1, seq_len, n_dynamic)

        # Aggregate across time dimension using absolute sum (total contribution of each feature)
        dynamic_shap_aggregated = np.nansum(np.abs(X_dynamic_shap_flat), axis=1)  # (n_samples, n_dynamic)

        # Combine aggregated dynamic SHAP with static SHAP
        shap_values_aggregated = np.concatenate([dynamic_shap_aggregated, X_static_shap], axis=1)  # (n_samples, n_dynamic + n_static)
        logger.my_print(f"Aggregated SHAP values: shape={shap_values_aggregated.shape}")
        assert shap_values_aggregated.shape[1] == (n_dynamic + n_static), \
            f"Aggregated SHAP shape mismatch: expected second dim {n_dynamic + n_static}, got {shap_values_aggregated.shape[1]}"

        # Get aggregated feature names
        try:
            df_dynamic = pd.read_csv(df_dynamic_filename, nrows=0)
            df_static = pd.read_csv(df_static_filename, nrows=0)
            del df_dynamic['Final_Result']  # Remove outcome column if present
            dynamic_feature_names = df_dynamic.columns.tolist()[-n_dynamic:]
            static_feature_names = df_static.columns.tolist()[-n_static:]
            feature_names_aggregated = dynamic_feature_names + static_feature_names
        except Exception as e:
            logger.my_print(f"Warning: Could not load feature names: {e}")
            feature_names_aggregated = [f'dynamic_{i}' for i in range(n_dynamic)] + [f'static_{i}' for i in range(n_static)]

        num_features = len(feature_names_aggregated)
        logger.my_print(f"Feature names aggregated: {num_features} features")

        # Compute and save variable importances
        logger.my_print(f"Computing variable importances...")
        mean_abs_shap = np.abs(shap_values_aggregated).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[::-1]
        top_features = [(feature_names_aggregated[i], mean_abs_shap[i]) for i in top_indices]

        # Save to CSV
        importance_df = pd.DataFrame(top_features, columns=['Variable', 'Mean absolute SHAP'])
        importance_csv_filename = f'top_features.csv'
        importance_df.to_csv(os.path.join(exp_shap_dir, importance_csv_filename), index=False)
        logger.my_print(f"  Saved: {importance_csv_filename}")

        # Print top features
        top_n_features = min(top_n_features, num_features)
        logger.my_print(f"\nTop {top_n_features} most important features:")
        top_features = top_features[:top_n_features]
        for i, (fname, importance) in enumerate(top_features, 1):
            logger.my_print(f"  {i}. {fname}: {importance:.6f}")

        # Create variable importance plot
        logger.my_print("Creating variable importance plot...")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            y_pos = np.arange(len(top_features))
            importances = [imp for _, imp in top_features]
            labels = [fname for fname, _ in top_features]
            ax.barh(y_pos, importances, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('Mean absolute SHAP value')
            ax.set_title(f'Variable importance - top {top_n_features}')
            ax.grid(alpha=0.4, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(exp_shap_dir, f'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.my_print(f"  Saved: feature_importance.png")
        except Exception as e:
            logger.my_print(f"  Error creating variable importance plot: {e}")

        logger.my_print("=" * 80)
        logger.my_print(f"SHAP analysis complete!")
        logger.my_print("=" * 80)

    except Exception as e:
        logger.my_print(f"ERROR in SHAP computation: {e}")
        logger.my_print("Continuing without SHAP analysis...")
        import traceback
        logger.my_print(traceback.format_exc())
        return None
