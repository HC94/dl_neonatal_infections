"""
Main entry point for neonatal infection prediction using Transformer-based survival analysis.
"""

import os
import time
import pickle
import random
import joblib
from datetime import datetime
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

from utils import Logger, create_folder_if_not_exists, get_model_summary, cleanup_objects
from data import (
    prepare_survival_data, patient_grouped_stratified_kfold,
    compute_sample_weights, SurvivalDataset, survival_collate_fn
)
from models import TransformerSurv
from losses import AFTLoss, MultiObjectiveLoss
from training import train_model
from evaluation import evaluate_model


# --- Main Execution ---
if __name__ == "__main__":
    # --- CONFIG ---
    # DIRECTORIES AND FILES
    CWD = os.getcwd()
    DATA_DIR = "data"
    DATA_DICT_FILENAME = 'data_dict.pickle'

    # VARIABLES
    PERFORM_TEST = True
    # PERFORM_TEST = False
    SEED = 0
    USE_MULTI_GPU = False  # Set to False to disable multi-GPU even if available
    NUM_FOLDS = 5
    NORM_TIMES = 60  # Normalize times to hours (60 minutes)
    EPSILON = 1e-8  # Small constant, mainly to avoid division by zero

    # Data and loss functions
    NUM_BINS = 24  # 1-hour intervals
    B_MIN = 0
    B_MAX = 1
    NUM_CALIBRATION_BINS = 5  # Evaluation of ExpectedCalibrationError

    # Model hyperparameters
    MAX_SEQ_LEN = 940

    # Training hyperparameters
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 5  # Early stopping patience

    # Optuna hyperparameter tuning settings
    OPTUNA_PROGRESSIVE_EVALUATION = False
    OPTUNA_N_TRIALS = 250
    OPTUNA_METRIC = 'c_index'  # Metric to optimize; minimize: 'loss', 'brier'; maximize: 'c_index', 'auc'
    OPTUNA_SAMPLER_N_STARTUP_TRIALS = 10
    OPTUNA_N_EI_CANDIDATES = 24
    OPTUNA_SAMPLER_MULTIVARIATE = True
    OPTUNA_SAMPLER_GROUP = True
    OPTUNA_STUDY_NAME = 'optuna_study.pickle'
    OPTUNA_SAMPLER_NAME = 'optuna_sampler.pickle'

    EXPERIMENTS_DIR = os.path.join("experiments", f"{OPTUNA_METRIC}")

    # Set test mode parameters
    if PERFORM_TEST:
        NUM_FOLDS = 2
        NUM_EPOCHS = 1
        OPTUNA_N_TRIALS = 3
        EXPERIMENTS_DIR += '_TEST'

    # Create directories if they do not exist
    OPTUNA_PATH_PICKLES = os.path.join(EXPERIMENTS_DIR, 'optuna_pickles')
    for folder in [EXPERIMENTS_DIR, OPTUNA_PATH_PICKLES]:
        create_folder_if_not_exists(folder)

    # Set seed for reproducibility
    np.random.seed(seed=SEED)
    random.seed(a=SEED)
    torch.manual_seed(seed=SEED)
    torch.backends.cudnn.benchmark = False

    # Determine device and multi-GPU configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    USE_DATA_PARALLEL = USE_MULTI_GPU and NUM_GPUS > 1
    
    print(f"Using device: {DEVICE}")
    print(f"Number of GPUs available: {NUM_GPUS}")
    if USE_DATA_PARALLEL:
        print(f"Multi-GPU mode enabled: Using DataParallel across {NUM_GPUS} GPUs")
        print(f"GPU devices: {[torch.cuda.get_device_name(i) for i in range(NUM_GPUS)]}")
    elif NUM_GPUS == 1:
        print(f"Single GPU mode: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode: No CUDA devices available")

    assert OPTUNA_METRIC in ['loss', 'c_index', 'auc', 'brier'], "Only 'loss', 'c_index', 'auc', and 'brier' are supported."

    # Validate OPTUNA_METRIC and OPTUNA_DIRECTION consistency
    if OPTUNA_METRIC in ['loss', 'brier']:
        OPTUNA_DIRECTION = 'minimize'
    elif OPTUNA_METRIC in ['c_index', 'auc']:
        OPTUNA_DIRECTION = 'maximize'
    else:
        raise ValueError(f"Unknown OPTUNA_METRIC: {OPTUNA_METRIC}")

    # 1. Data preparation
    print("Loading and preparing data...")
    with open(os.path.join(CWD, DATA_DIR, DATA_DICT_FILENAME), 'rb') as f:
        data_dict = pickle.load(f)

    (patient_id_array, encounter_id_array, X_dynamic_raw, X_dynamic_raw_mask, dynamic_times_raw,
     X_static_raw, X_static_raw_mask, result_times_raw, result_indicators_raw) = prepare_survival_data(
        data_dict=data_dict, norm_times=NORM_TIMES
    )

    # Extract number of features
    NUM_DYNAMIC_FEATURES = X_dynamic_raw.shape[-1]
    NUM_STATIC_FEATURES = X_static_raw.shape[-1]
    NUM_FEATURES = NUM_DYNAMIC_FEATURES + NUM_STATIC_FEATURES

    # Hyperparameter tuning using Optuna
    ##### OPTUNA #####
    # (Optuna) 1. Define an objective function to be maximized or minimized.
    def optuna_objective(trial):
        # (Optuna) 2. Suggest values of the hyperparameters using a trial object.
        global optuna_study, optuna_study_run_nr, optuna_sampler_run_nr

        # Training settings
        batch_size_exp = trial.suggest_int('batch_size_exp', 1, 4)
        batch_size = 2 ** batch_size_exp  # 2^1=2 to 2^4=16

        # Data preprocessing and normalization hyperparameters
        sampler_event_boost = trial.suggest_float('sampler_event_boost', 1.0, 20.0)

        # Data augmentation hyperparameters
        min_seq_len_perc = trial.suggest_float('min_seq_len_perc', 0.25, 0.75)
        max_seq_len_perc = trial.suggest_float('max_seq_len_perc', min_seq_len_perc + 0.1, 1.0)
        augmentation_noise_std = trial.suggest_float('augmentation_noise_std', 0.0, 0.1)

        # Multi-objective calibration loss weights
        w_discrimination = 1.0  # Always 1.0 to maintain base loss contribution

        # Calibration loss weights
        w_calibration_regression = trial.suggest_float('w_calibration_regression', 0.1, 1.0)

        # Model hyperparameters
        embed_dim = trial.suggest_categorical('embed_dim', [64, 96, 128, 192])
        possible_heads = [h for h in [2, 4, 8, 16] if embed_dim % h == 0]
        num_heads = trial.suggest_categorical('num_heads', possible_heads)
        num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 2, 8)
        ff_dim_multiplier = trial.suggest_int('ff_dim_multiplier', 2, 4)
        ff_dim = ff_dim_multiplier * embed_dim
        dropout = trial.suggest_float('dropout', 0.0, 0.25)
        drop_path_rate = trial.suggest_float('drop_path_rate', 0.0, 0.25)
        pos_enc_base_exp = trial.suggest_int('pos_enc_base_exp', 1, 3)
        pos_enc_base = 10 ** (pos_enc_base_exp + 1)  # 10^2=100 to 10^4=10000

        # Optimizer
        optimizer_type = trial.suggest_categorical('optimizer_type', ['adam', 'adamw', 'radam'])
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

        # Scheduler
        scheduler_type = trial.suggest_categorical('scheduler_type', ['plateau', 'cosine_warmup'])
        if scheduler_type in ['plateau']:
            lr_factor = trial.suggest_float('lr_factor', 0.25, 0.75)
            lr_patience = 5  # Learning rate patience (default: 5)
        elif scheduler_type in ['cosine_warmup']:
            t_0 = trial.suggest_int('t_0', 5, 25)
            t_mult = trial.suggest_int('t_mult', 1, 2)
        else:
            raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        # Maximum gradient norm enabled (default: True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0)

        ##### START EXPERIMENT #####
        print(f'Trial {globals()["optuna_study_trial_number"]}')
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S") + '_trial_{trial}_fold_{fold}'
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_name)

        # Initialize variables to track metrics across folds
        fold_val_losses = []
        fold_c_indices = []
        fold_aucs = []
        fold_briers = []

        ##### PERFORM CROSS-VALIDATION #####
        for fold_idx, (train_indices, val_indices) in enumerate(
                patient_grouped_stratified_kfold(
                    patient_ids=patient_id_array,
                    result_indicators=result_indicators_raw,
                    num_folds=NUM_FOLDS,
                    seed=SEED
                )
        ):
            # Create folder for this trial and fold
            exp_dir = exp_path.format(trial=globals()['optuna_study_trial_number'], fold=fold_idx)
            exp_checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
            logging_filename = os.path.join(exp_dir, 'logging.txt')

            for p in [exp_dir, exp_checkpoints_dir]:
                create_folder_if_not_exists(p)

            # Initialize logger (only for first fold to avoid clutter)
            if fold_idx == 0:
                try:
                    logger.close()
                    del logger
                except:
                    pass
                logger = Logger(logger_filename=logging_filename)

            logger.my_print(f"Fold {fold_idx}:")

            # Create the training dataset. Normalization parameters will be calculated internally.
            train_dataset_kwargs = {
                'indices': train_indices,
                'patient_ids': patient_id_array,
                'encounter_ids': encounter_id_array,
                'X_dynamic': X_dynamic_raw,
                'mask_dynamic': X_dynamic_raw_mask,
                'dynamic_times': dynamic_times_raw,
                'X_static': X_static_raw,
                'mask_static': X_static_raw_mask,
                'result_times': result_times_raw,
                'result_indicators': result_indicators_raw,
                'min_seq_len_perc': min_seq_len_perc,
                'max_seq_len_perc': max_seq_len_perc,
                'num_bins': NUM_BINS,
                'is_train': True,
                'norm_params': None,
                'b_min': B_MIN,
                'b_max': B_MAX,
                'seed': SEED,
                'augmentation_noise_std': augmentation_noise_std,
                'logger': logger,
            }
            train_dataset = SurvivalDataset(**train_dataset_kwargs)

            # Retrieve the normalization parameters from the training dataset
            norm_params = train_dataset.get_norm_params()

            # Create the val dataset using the parameters from the training set
            val_dataset_kwargs = {
                'indices': val_indices,
                'patient_ids': patient_id_array,
                'encounter_ids': encounter_id_array,
                'X_dynamic': X_dynamic_raw,
                'mask_dynamic': X_dynamic_raw_mask,
                'dynamic_times': dynamic_times_raw,
                'X_static': X_static_raw,
                'mask_static': X_static_raw_mask,
                'result_times': result_times_raw,
                'result_indicators': result_indicators_raw,
                'min_seq_len_perc': 1.0,
                'max_seq_len_perc': 1.0,
                'num_bins': NUM_BINS,
                'is_train': False,
                'norm_params': norm_params,
                'b_min': B_MIN,
                'b_max': B_MAX,
                'seed': SEED,
                'augmentation_noise_std': 0.0,  # No augmentation for validation/test set
                'logger': logger,
            }
            val_dataset = SurvivalDataset(**val_dataset_kwargs)

            # Calculate label distribution for this fold
            train_events = result_indicators_raw[train_indices].sum()
            train_total = len(train_indices)
            val_events = result_indicators_raw[val_indices].sum()
            val_total = len(val_indices)

            patient_id_train_tensor = torch.tensor(patient_id_array[train_indices], dtype=torch.float32)
            patient_id_val_tensor = torch.tensor(patient_id_array[val_indices], dtype=torch.float32)

            # Print
            logger.my_print(
                f"\tTrain: {train_total} encounters from {len(torch.unique(patient_id_train_tensor))} unique patients "
                f"(Events: {train_events}/{train_total} = {train_events / train_total * 100:.1f}%)")
            logger.my_print(
                f"\tVal:  {val_total} encounters from {len(torch.unique(patient_id_val_tensor))} unique patients "
                f"(Events: {val_events}/{val_total} = {val_events / val_total * 100:.1f}%)")

            # Verify no patient appears in both train and val
            overlap = set(patient_id_train_tensor) & set(patient_id_val_tensor)
            logger.my_print(f"\tPatient overlap between train and val set: {len(overlap)} (should be 0)")
            assert len(overlap) == 0

            # Sample weights to deal with imbalance in result_indicators
            sample_weights = compute_sample_weights(
                event_indicators=result_indicators_raw[train_indices],
                sampler_event_boost=sampler_event_boost,
                epsilon=EPSILON,
            )
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                collate_fn=lambda batch: survival_collate_fn(
                    batch=batch,
                    max_global_len=None
                )
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda batch: survival_collate_fn(
                    batch=batch,
                    max_global_len=None
                )
            )
            
            # Base AFT loss
            base_criterion = AFTLoss(epsilon=EPSILON, logger=logger)

            # Wrap in multi-objective loss if calibration weight is != 0.0
            if w_calibration_regression != 0.0:
                logger.my_print(f"Using multi-objective loss with calibration components:")
                logger.my_print(f"  w_discrimination: {w_discrimination}")
                logger.my_print(f"  w_calibration_regression: {w_calibration_regression}")
                criterion = MultiObjectiveLoss(
                    discrimination_criterion=base_criterion,
                    num_bins=NUM_BINS,
                    epsilon=EPSILON,
                    w_discrimination=w_discrimination,
                    w_calibration_regression=w_calibration_regression,
                )
            else:
                criterion = base_criterion
                
            # 3. Initialize model, optimizer, and scheduler
            logger.my_print("Initializing model, loss, and optimizer...")
            model_kwargs = {
                'num_bins': NUM_BINS,
                'num_dynamic_features': NUM_DYNAMIC_FEATURES,
                'num_static_features': NUM_STATIC_FEATURES,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_transformer_blocks': num_transformer_blocks,
                'ff_dim': ff_dim,
                'dropout': dropout,
                'drop_path_rate': drop_path_rate,
                'pos_enc_base': pos_enc_base,
                'max_seq_len': MAX_SEQ_LEN,
                'logger': logger
            }
            model = TransformerSurv(**model_kwargs).to(DEVICE)
            
            # Wrap model with DataParallel for multi-GPU support
            if USE_DATA_PARALLEL:
                logger.my_print(f"Wrapping model with DataParallel across {NUM_GPUS} GPUs")
                model = nn.DataParallel(model)
                logger.my_print(f"Model wrapped successfully. Primary device: {DEVICE}")

            # Get model summary (only for first fold to avoid clutter)
            if fold_idx == 0:
                T_tmp = 3  # Dummy sequence length
                X_dynamic_batch_tmp = torch.randn(batch_size, T_tmp, NUM_DYNAMIC_FEATURES).to(DEVICE)  # (B, T, D)
                X_dynamic_mask_batch_tmp = torch.rand(batch_size, T_tmp, NUM_DYNAMIC_FEATURES).to(DEVICE) > 0.5  # (B, T, D)
                dynamic_times_batch_tmp = torch.randint(-1, 2, (batch_size, T_tmp)).float().to(DEVICE)  # (B, T)
                X_static_batch_tmp = torch.randn(batch_size, NUM_STATIC_FEATURES).to(DEVICE)  # (B, S)
                X_static_mask_batch_tmp = torch.rand(batch_size, NUM_STATIC_FEATURES).to(DEVICE) > 0.5  # (B, S)

                model_summary_filename = os.path.join(exp_dir, 'model_summary.txt')
                total_params = get_model_summary(model=model, input_data=[
                    X_dynamic_batch_tmp,
                    X_dynamic_mask_batch_tmp,
                    dynamic_times_batch_tmp,
                    X_static_batch_tmp,
                    X_static_mask_batch_tmp
                ], filename=model_summary_filename, device=DEVICE, logger=logger)
                logger.my_print(f'Number of model parameters: {total_params}')

            # Initialize optimizer based on hyperparameter
            if optimizer_type == 'adam':
                optimizer = optim.Adam(
                    params=model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'adamw':
                optimizer = optim.AdamW(
                    params=model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'radam':
                optimizer = optim.RAdam(
                    params=model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            else:
                raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

            # Initialize scheduler based on hyperparameter
            if scheduler_type == 'plateau':
                scheduler = ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience
                )
            elif scheduler_type == 'cosine_warmup':
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    T_0=t_0,
                    T_mult=t_mult
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

            # 4. Train model
            logger.my_print("Starting training...")
            model = train_model(
                model=model,
                num_epochs=NUM_EPOCHS,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=max_grad_norm,
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                exp_checkpoints_dir=exp_checkpoints_dir,
                num_bins=NUM_BINS,
                num_calibration_bins=NUM_CALIBRATION_BINS,
                epsilon=EPSILON,
                seed=SEED,
                device=DEVICE,
                logger=logger,
                optuna_metric=OPTUNA_METRIC,
            )

            # After training, load the best model for evaluation
            best_model_filename = os.path.join(exp_checkpoints_dir, f'best_model.pth')
            if os.path.exists(best_model_filename):
                logger.my_print(f"Loading best model from {best_model_filename} for evaluation.")
                # Handle DataParallel: load into the underlying module
                best_model = model.module if isinstance(model, nn.DataParallel) else model
                best_model.load_state_dict(torch.load(best_model_filename, weights_only=True))
            else:
                logger.my_print("Warning: No best model found. Using the last model state for evaluation.")
                best_model = model

            avg_val_loss, val_c_index, val_auc, val_brier, _, _, _, _, _ = evaluate_model(
                model=best_model,
                val_loader=val_loader,
                criterion=criterion,
                num_bins=NUM_BINS,
                num_calibration_bins=NUM_CALIBRATION_BINS,
                progressive_evaluation=OPTUNA_PROGRESSIVE_EVALUATION,
                n_bootstrap=None,
                confidence=None,
                seed=SEED,
                epsilon=EPSILON,
                device=DEVICE,
                logger=logger,
            )

            # Track validation metrics for this fold
            fold_val_losses.append(avg_val_loss)
            fold_c_indices.append(val_c_index)
            fold_aucs.append(val_auc)
            fold_briers.append(val_brier)

            val_auc_str = f"{val_auc:.4f}" if val_auc is not None else "N/A"
            val_brier_str = f"{val_brier:.4f}" if val_brier is not None else "N/A"
            logger.my_print(f"Fold {fold_idx} | "
                            f"Validation loss: {avg_val_loss:.4f} | "
                            f"C-index: {val_c_index:.4f} | "
                            f"AUC: {val_auc_str} | "
                            f"Brier: {val_brier_str}")

            logger.my_print("="*80 + "\n")

            # Cleanup objects to avoid GPU memory leaks and leftover references
            cleanup_objects(best_model, model, optimizer, scheduler)

        # Save optuna objects after each trial
        optuna_out_file_study = os.path.join(OPTUNA_PATH_PICKLES, '{}_'.format(optuna_study_run_nr) + OPTUNA_STUDY_NAME)
        optuna_out_file_sampler = os.path.join(OPTUNA_PATH_PICKLES,
                                               '{}_'.format(optuna_sampler_run_nr) + OPTUNA_SAMPLER_NAME)
        joblib.dump(optuna_study, optuna_out_file_study)
        # Save the sampler for reproducibility after resuming study
        joblib.dump(optuna_study.sampler, optuna_out_file_sampler)

        # Increment trial number
        globals()['optuna_study_trial_number'] += 1

        # Optuna objective: return mean of selected metric across folds
        assert len(fold_val_losses) == len(fold_c_indices) == len(fold_aucs) == len(fold_briers) == NUM_FOLDS
        logger.my_print(f"--- Trial {globals()['optuna_study_trial_number'] - 1} Cross-validation results ---")
        mean_val_loss = mean(fold_val_losses)
        mean_c_index = mean(fold_c_indices)

        # Filter None values for C-index, AUC and Brier before computing mean
        mean_auc = mean([x for x in fold_aucs if x is not None]) if any(
            x is not None for x in fold_aucs) else None
        mean_brier = mean([x for x in fold_briers if x is not None]) if any(
            x is not None for x in fold_briers) else None

        mean_auc_str = f"{mean_auc:.4f}" if mean_auc is not None else "N/A"
        mean_brier_str = f"{mean_brier:.4f}" if mean_brier is not None else "N/A"
        logger.my_print(f"Trial {globals()['optuna_study_trial_number'] - 1} | "
                        f"mean loss: {mean_val_loss:.4f} | "
                        f"mean C-index: {mean_c_index:.4f} | "
                        f"mean AUC: {mean_auc_str} | "
                        f"mean Brier: {mean_brier_str}")

        # Return metric based on OPTUNA_METRIC
        if OPTUNA_METRIC == 'loss':
            logger.my_print(f"Objective: Minimize mean validation loss.")
            logger.close()
            del logger
            return mean_val_loss
        elif OPTUNA_METRIC == 'c_index':
            logger.my_print(f"Objective: Maximize mean C-index.")
            logger.close()
            del logger
            return mean_c_index
        elif OPTUNA_METRIC == 'auc':
            logger.my_print(f"Objective: Maximize mean AUC.")
            logger.close()
            del logger
            return mean_auc
        elif OPTUNA_METRIC == 'brier':
            logger.my_print(f"Objective: Minimize mean Brier Score.")
            logger.close()
            del logger
            return mean_brier
        else:
            raise ValueError(f"Unknown OPTUNA_METRIC: {OPTUNA_METRIC}")



    # (Optuna) 3. Create a study object and optimize the objective function.
    # Resume study if study and sampler files exist
    optuna_file_study_list = [x for x in os.listdir(OPTUNA_PATH_PICKLES) if OPTUNA_STUDY_NAME in x]
    optuna_file_sampler_list = [x for x in os.listdir(OPTUNA_PATH_PICKLES) if OPTUNA_SAMPLER_NAME in x]
    if len(optuna_file_study_list) > 0 and len(optuna_file_sampler_list) > 0:
        # Find last study, and add 1 for the next study run
        optuna_study_run_nr = max([int(x.split('_')[0]) for x in optuna_file_study_list]) + 1
        optuna_sampler_run_nr = max([int(x.split('_')[0]) for x in optuna_file_sampler_list]) + 1
        assert optuna_study_run_nr == optuna_sampler_run_nr
        # Determine the last trial number
        globals()['optuna_study_trial_number'] = [x for x in os.listdir(EXPERIMENTS_DIR) if 'trial_' in x]
        globals()['optuna_study_trial_number'] = max([int(x.split('trial_')[1].split('_')[0]) for x in
                                                      globals()['optuna_study_trial_number']]) + 1
        print(f'Resuming Optuna study from trial number {globals()["optuna_study_trial_number"]}...')
        # Load last sampler and study
        optuna_in_file_sampler = os.path.join(OPTUNA_PATH_PICKLES,
                                              '{}_'.format(optuna_sampler_run_nr - 1) + OPTUNA_SAMPLER_NAME)
        optuna_in_file_study = os.path.join(OPTUNA_PATH_PICKLES, '{}_'.format(optuna_study_run_nr - 1) + OPTUNA_STUDY_NAME)
        print('Resuming previous sampler and study: {} and {}'.format(optuna_in_file_sampler, optuna_in_file_study))
        optuna_sampler = joblib.load(optuna_in_file_sampler)
        optuna_study = joblib.load(optuna_in_file_study)
    else:
        print('Starting new Optuna study...')
        optuna_sampler_run_nr = 0
        optuna_study_run_nr = 0
        globals()['optuna_study_trial_number'] = 0
        # Create new sampler and study
        optuna_sampler = optuna.samplers.TPESampler(
            n_startup_trials=OPTUNA_SAMPLER_N_STARTUP_TRIALS,
            n_ei_candidates=OPTUNA_N_EI_CANDIDATES,
            multivariate=OPTUNA_SAMPLER_MULTIVARIATE,
            group=OPTUNA_SAMPLER_GROUP,
            seed=SEED
        )
        optuna_study = optuna.create_study(
            sampler=optuna_sampler,
            direction=OPTUNA_DIRECTION,
        )

    # Run hyperparameter tuning using Optuna
    optuna_start = time.time()
    optuna_study.optimize(optuna_objective, n_trials=(OPTUNA_N_TRIALS-optuna_study_run_nr))

    # Save study
    optuna_out_file_sampler = os.path.join(OPTUNA_PATH_PICKLES, '{}_'.format(optuna_sampler_run_nr) + OPTUNA_SAMPLER_NAME)
    optuna_out_file_study = os.path.join(OPTUNA_PATH_PICKLES, '{}_'.format(optuna_study_run_nr) + OPTUNA_STUDY_NAME)
    joblib.dump(optuna_study.sampler, optuna_out_file_sampler)
    joblib.dump(optuna_study, optuna_out_file_study)

    optuna_end = time.time()
    print(f'Elapsed time: {optuna_end - optuna_start} seconds')

