"""
Training loop and related functions for survival analysis.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from evaluation import evaluate_model


def train_model(model, num_epochs, train_loader, val_loader, criterion,
                optimizer, scheduler, max_grad_norm, early_stopping_patience, exp_checkpoints_dir,
                num_bins, num_calibration_bins, epsilon, seed, device, logger, optuna_metric):
    """
    Train the survival analysis model.
    """
    # Initialize variables
    rng = np.random.RandomState(seed)
    best_model_filename = os.path.join(exp_checkpoints_dir, f'best_model.pth')
    best_val_loss = float('inf')
    best_val_c_index = 0.0
    best_val_auc = 0.0
    best_val_brier = float('inf')
    early_stopping_patience_counter = 0
    train_losses = []

    # --- Initialize statistics tracking ---
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        # For gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(train_loader):
            # Unpack common elements (first 9 are always present)
            (patient_id_batch, encounter_id_batch, X_dynamic_batch, X_dynamic_mask_batch, dynamic_times_batch,
             X_static_batch, X_static_mask_batch, rel_times_batch, rel_indicators_batch) = batch_data

            rel_times_batch = rel_times_batch.to(device)
            rel_indicators_batch = rel_indicators_batch.to(device)

            X_dynamic_batch = X_dynamic_batch.to(device)
            X_dynamic_mask_batch = X_dynamic_mask_batch.to(device)
            dynamic_times_batch = dynamic_times_batch.to(device)
            X_static_batch = X_static_batch.to(device)
            X_static_mask_batch = X_static_mask_batch.to(device)

            outputs = model(
                x_dynamic=X_dynamic_batch,
                x_dynamic_mask=X_dynamic_mask_batch,
                dynamic_times=dynamic_times_batch,
                x_static=X_static_batch,
                x_static_mask=X_static_mask_batch
            )

            # AFT loss computation
            assert (outputs.shape[-1] == 2) and (len(rel_indicators_batch.shape) == 1), f"Outputs.shape = {outputs.shape} or indicators.shape = {rel_indicators_batch.shape} are invalid"
            loss = criterion(outputs=outputs, result_times=rel_times_batch, result_indicators=rel_indicators_batch)

            # Skip batch if loss is NaN
            if torch.isnan(loss):
                logger.my_print(f"Warning: NaN loss detected at batch {batch_idx}, skipping...")
                optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            batch_size_current = X_dynamic_batch.size(0)
            total_train_loss += float(loss.detach().cpu().item()) * batch_size_current
            total_train_samples += batch_size_current

        avg_train_loss = total_train_loss / max(1, total_train_samples)
        train_losses.append(avg_train_loss)

        # Validation
        eval_model = model
        # Progressive evaluation is NOT used here (defaults to False) for computational efficiency during training
        avg_val_loss, val_c_index, val_auc, val_brier, _, _, _, _, _ = evaluate_model(
            model=eval_model,
            val_loader=val_loader,
            criterion=criterion,
            num_bins=num_bins,
            num_calibration_bins=num_calibration_bins,
            progressive_evaluation=False,
            n_bootstrap=None,
            confidence=None,
            seed=seed,
            epsilon=epsilon,
            device=device,
            logger=logger
        )

        val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
        val_brier_str = f"{val_brier:.4f}" if val_brier is not None else "N/A"
        logger.my_print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, "
            f"C-Index: {val_c_index:.4f}, AUC: {val_auc_str}, Brier: {val_brier_str}")

        if scheduler is not None:
            # Handle different scheduler types
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()  # Cosine annealing does not need metric

        # Determine if current epoch is better based on optuna_metric
        is_better = False
        if optuna_metric == 'loss' and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            is_better = True
        elif optuna_metric == 'c_index' and val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            is_better = True
        elif optuna_metric == 'auc' and val_auc is not None and val_auc > best_val_auc:
            best_val_auc = val_auc
            is_better = True
        elif optuna_metric == 'brier' and val_brier is not None and val_brier < best_val_brier:
            best_val_brier = val_brier
            is_better = True

        if is_better:
            best_epoch = epoch
            early_stopping_patience_counter = 0
            # Save the model that was actually used for evaluation (eval_model)
            # Handle DataParallel: save the underlying module's state_dict
            model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model.state_dict(), best_model_filename)
            logger.my_print(f"  -> New best model saved to {best_model_filename} (based on {optuna_metric})")
        else:
            early_stopping_patience_counter += 1
            if early_stopping_patience_counter >= early_stopping_patience:
                logger.my_print(f"Early stopping at epoch {epoch + 1}")
                break

    # After training, load the best model for plotting
    if os.path.exists(best_model_filename):
        logger.my_print(f"Loading best model from {best_model_filename} for plotting.")
        # Handle DataParallel: load into the underlying module
        model = model.module if isinstance(model, nn.DataParallel) else model
        model.load_state_dict(torch.load(best_model_filename, weights_only=True))
    else:
        # E.g., when the model does not train at all from the very beginning (evaluation metric value = None)
        logger.my_print("Warning: No best model found. Using the last model state for plotting.")

    return model
