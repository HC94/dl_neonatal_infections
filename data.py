"""
Data preparation and dataset classes for survival analysis.
"""

import math
import torch
import numpy as np
from torch.utils.data import Dataset


def prepare_survival_data(data_dict, norm_times):
    """
    Population-wide padding: This function pads all sequences to the absolute maximum length found in
    the entire dataset. This is a one-time data preparation step to create uniformly shaped NumPy arrays
    (X_dynamic_raw, dynamic_times_raw, etc.) for easy indexing and storage.

    This function transforms a nested dictionary of patient data (`data_dict`) into a set of padded,
    right-aligned NumPy arrays suitable for machine learning. "Right-alignment" means that for a time series shorter
    than the maximum length, padding values are added to the beginning of the sequence, not the end.
    This pushes the actual data points to the right side of the array. For example, if the maximum sequence length is 5
    and we have a sequence [A, B, C] of length 3, right-aligning it would result in [pad, pad, A, B, C].
    Moreover, this function prepares full sequences per encounter (no unrolling).
    Relative times are computed dynamically in SurvivalDataset.

    Step-by-step breakdown:
    1.  **Find maximum sequence length**: First iterates through all patient encounters in `data_dict` to
        determine the maximum number of timesteps (`max_different_time_steps`) present in any single encounter.
        This value is used for padding all other sequences.
    2.  **Process and pad data**: Iterate through the data again. For each encounter:
        *   Extracts dynamic features, static features, observation times, and survival outcome data.
        *   Converts time values from minutes to hours by dividing by `norm_times`.
        *   Pads the dynamic features and observation times arrays with `np.nan` at the beginning.
            This makes all sequences equal in length to `max_different_time_steps` and aligns them to the right.
        *   Appends the processed data for the encounter to corresponding lists.
    3.  **Create NumPy arrays**: Convert the lists of data into NumPy arrays.
    4.  **Generate masks and fill NaNs**:
        *   Create boolean masks (`X_dynamic_raw_mask`, `X_static_raw_mask`) where `True` indicates a padded or
            missing value (`NaN`) in the feature arrays.
        *   Then replace all `NaN` values in the feature arrays (`X_dynamic_raw`, `X_static_raw`) with `0.0`;
            the missing values in the feature arrays are dealt by boolean masks.
            Note: the `dynamic_times_raw` array is intentionally left with `NaN`s to identify the padded steps later.
    5.  **Print statistics**: Print summary statistics about the shape and properties of the generated arrays.
    6.  **Return arrays**: Return a tuple containing the patient/encounter IDs, the padded dynamic and
        static feature arrays, their corresponding masks, the padded time array, and the survival outcome arrays.

    Returns:
        patient_id_array, encounter_id_array, X_dynamic_raw, X_dynamic_raw_mask, X_static_raw, X_static_raw_mask,
        dynamic_times_raw, result_times_raw, result_indicators_raw
    """
    print("Loading prepared data from data_dict for time-series modelling...")
    # Initialize lists to hold the processed data for each encounter.
    X_dynamic_raw_list = []
    X_static_raw_list = []
    dynamic_times_list = []
    result_times_raw_list = []
    result_indicators_raw_list = []
    patient_id_list = []
    encounter_id_list = []
    # Initialize a variable to track the maximum sequence length found across all encounters.
    max_different_time_steps = 0
    # A list to store the original sequence length of each encounter for statistics.
    len_dynamic_features = []

    # First pass: Iterate through all encounters to find the maximum number of timesteps.
    # This is necessary to determine the padding length for all sequences.
    for patient_id, encounters in data_dict.items():
        for encounter_id, data in encounters.items():
            # Get the list of observation dynamic_times (in minutes).
            dynamic_times_min = data['dynamic_times']
            # Update the maximum sequence length if the current encounter's sequence is longer.
            max_different_time_steps = max(max_different_time_steps, len(dynamic_times_min))
    print(f"Maximum number of different timesteps (i.e., rows): {max_different_time_steps}")

    # Second pass: Process each encounter, pad the data, and append to the lists.
    for patient_id, encounters in data_dict.items():
        for encounter_id, data in encounters.items():
            # Extract and convert data from the dictionary to NumPy arrays with a specific data type.
            dynamic_times_min = np.asarray(data['dynamic_times'],
                                           dtype=np.float32)  # Observation dynamic_times in minutes.
            dynamic_features = np.asarray(data['dynamic_features'],
                                          dtype=np.float32)  # Dynamic features, may contain NaNs.
            static_features = np.asarray(data['static_features'][0], dtype=np.float32)  # Static features.
            result_time_min = float(data['result_time'][0])  # Time to event/censoring in minutes.
            result_label = int(data['result_label'][0])  # Event indicator (1 for event, 0 for censored).

            # Convert observation dynamic_times from minutes to hours.
            dynamic_times_hours = dynamic_times_min / float(norm_times)
            # Get the original sequence length for this encounter.
            T = len(dynamic_times_hours)
            # Store the original length for later statistics.
            len_dynamic_features.append(T)

            # Ensure dynamic features have 2 dimensions (timesteps, features).
            if dynamic_features.ndim == 1:
                dynamic_features = np.expand_dims(dynamic_features, axis=0)
            # Ensure static features have 1 dimension (features).
            if static_features.ndim == 0:
                static_features = np.expand_dims(static_features, axis=0)

            # Calculate the number of padding steps needed for the current sequence.
            pad_T = max_different_time_steps - T
            if pad_T > 0:
                # Create padding arrays filled with NaN.
                dyn_padding = np.full((pad_T, dynamic_features.shape[1]), np.nan, dtype=np.float32)
                dynamic_times_padding = np.full(pad_T, np.nan, dtype=np.float32)
                # Prepend the padding to the start of the arrays to achieve right-alignment.
                dynamic_features = np.vstack([dyn_padding, dynamic_features])
                dynamic_times_hours = np.concatenate([dynamic_times_padding, dynamic_times_hours])

            # Append the processed (and possibly padded) data to their respective lists.
            X_dynamic_raw_list.append(dynamic_features.astype(np.float32))
            X_static_raw_list.append(static_features.astype(np.float32))
            dynamic_times_list.append(dynamic_times_hours.astype(np.float32))
            # Convert result time to hours and append.
            result_times_raw_list.append(result_time_min / float(norm_times))
            result_indicators_raw_list.append(result_label)
            patient_id_list.append(int(patient_id))
            encounter_id_list.append(int(encounter_id))

    # Convert the lists of arrays into single large NumPy arrays.
    X_dynamic_raw = np.array(X_dynamic_raw_list, dtype=np.float32)
    X_static_raw = np.array(X_static_raw_list, dtype=np.float32)
    dynamic_times_raw = np.array(dynamic_times_list,
                                 dtype=np.float32)  # This array will retain NaNs for padding identification.
    result_times_raw = np.array(result_times_raw_list, dtype=np.float32)
    result_indicators_raw = np.array(result_indicators_raw_list, dtype=np.int32)

    # Create boolean masks to identify NaN values (both original NaNs and padding NaNs).
    # True indicates a missing or padded value.
    X_dynamic_raw_mask = np.isnan(X_dynamic_raw)
    # Replace all NaN values in the feature arrays with 0.0. The masks retain the location of these values.
    X_dynamic_raw = np.nan_to_num(X_dynamic_raw, nan=0.0)
    X_static_raw_mask = np.isnan(X_static_raw)
    X_static_raw = np.nan_to_num(X_static_raw, nan=0.0)

    # Get dimensions and other statistics from the final arrays.
    NUM_SAMPLES = len(X_dynamic_raw)
    NUM_TIME_STEPS = X_dynamic_raw.shape[1]
    NUM_DYNAMIC_FEATURES = X_dynamic_raw.shape[2]
    NUM_STATIC_FEATURES = X_static_raw.shape[1]
    NUM_FEATURES = NUM_DYNAMIC_FEATURES + NUM_STATIC_FEATURES

    # Calculate and print statistics about the padding.
    # Count the number of non-NaN (valid) timesteps for each sample.
    valid_time_steps_per_sample = np.isfinite(dynamic_times_raw).sum(axis=1)
    print(f"Valid timesteps per sample - Mean: {np.mean(valid_time_steps_per_sample):.1f}, "
          f"Min: {np.min(valid_time_steps_per_sample)}, Max: {np.max(valid_time_steps_per_sample)}")

    # Calculate the number of padded timesteps for each sample.
    padding_time_steps_per_sample = NUM_TIME_STEPS - valid_time_steps_per_sample
    print(f"Padding timesteps per sample - Mean: {np.mean(padding_time_steps_per_sample):.1f}, "
          f"Min: {np.min(padding_time_steps_per_sample)}, Max: {np.max(padding_time_steps_per_sample)}")

    # Return all the generated NumPy arrays as a tuple.
    return (np.array(patient_id_list),
            np.array(encounter_id_list),
            X_dynamic_raw,
            X_dynamic_raw_mask,
            dynamic_times_raw,  # keep NaNs so PositionalEncoding can ignore padding
            X_static_raw,
            X_static_raw_mask,
            result_times_raw,
            result_indicators_raw)


def patient_grouped_stratified_kfold(patient_ids, result_indicators, num_folds, seed):
    """
    Perform grouped K-fold cross-validation, because some patient_ids may have multiple encounter_ids;
    Make sure that the same patient_ids belong to the same fold.

    Create stratified K-fold splits ensuring:
    1. All encounters from same patient stay together
    2. Folds have similar proportions of event/censored patients

    Parameters:
    -----------
    patient_ids : array-like
        Array of patient IDs (one per encounter)
    result_indicators : array-like
        Array of event indicators (0=censored, 1=event)
    num_folds : int
        Number of folds
    seed : int
        Random seed for reproducibility

    Yields:
    -------
    train_idx : array
        Indices for training set
    val_idx : array
        Indices for test set
    """
    # Initialize variables
    rng = np.random.RandomState(seed)

    # Get unique patients and their event status
    unique_patients = np.unique(patient_ids)

    # For each patient, determine if they had ANY event across all encounters
    patient_has_event = {}
    for patient_id in unique_patients:
        patient_mask = (patient_ids == patient_id)
        # Patient is "event" if ANY of their encounters had an event
        patient_has_event[patient_id] = int(np.any(result_indicators[patient_mask] == 1))

    # Separate patients by event status
    event_patients = [pid for pid in unique_patients if patient_has_event[pid] == 1]
    censored_patients = [pid for pid in unique_patients if patient_has_event[pid] == 0]

    # Shuffle each group
    rng.shuffle(event_patients)
    rng.shuffle(censored_patients)

    # --- Generate and yield K-fold splits ---
    event_folds = np.array_split(event_patients, num_folds)
    censored_folds = np.array_split(censored_patients, num_folds)

    # Generate train/val indices for each fold
    for fold_idx in range(num_folds):
        # Combine event and censored patients for val fold
        val_patients = np.concatenate([event_folds[fold_idx], censored_folds[fold_idx]])

        # Combine remaining folds for training
        train_event = np.concatenate([event_folds[i] for i in range(num_folds) if i != fold_idx])
        train_censored = np.concatenate([censored_folds[i] for i in range(num_folds) if i != fold_idx])
        train_patients = np.concatenate([train_event, train_censored])

        # Get indices for all encounters of train/val patients
        val_idx = np.where(np.isin(patient_ids, val_patients))[0]
        train_idx = np.where(np.isin(patient_ids, train_patients))[0]

        yield train_idx, val_idx


def compute_sample_weights(event_indicators, sampler_event_boost, epsilon):
    """
    Compute normalized sample weights for (over)sampling encounters with events.
    Ensure the model sees more rare results during training.

    Args:
        event_indicators: 1D array-like of shape (num_samples,) with 0 (censored) or 1 (event) per encounter.
        sampler_event_boost: Factor to boost event sample weights.
        epsilon: Small constant for numerical stability to prevent division by zero.

    Returns:
        weights: torch.Tensor of shape (num_samples,) with normalized weights.
    """
    if isinstance(event_indicators, np.ndarray):
        event_indicators = torch.tensor(event_indicators, dtype=torch.float32)
    has_event = (event_indicators == 1).float()  # (num_samples,)
    weights = torch.where(has_event > 0, sampler_event_boost, 1.0)  # (Over)sample events by sampler_event_boost
    weights = weights / (weights.sum() + epsilon) * len(weights)  # Normalize to sum to num_samples
    return weights


class SurvivalDataset(Dataset):
    """
    Custom Dataset for survival analysis. It handles feature normalization internally
    and generates samples for time-series modeling. For training, it fits the normalization
    parameters. For testing/validation, it uses pre-computed parameters.
    It also supports generating random prefixes (unrolling) for data augmentation (a.k.a. dynamic sample and target
    generation through random prefix unrolling).

    ### Key features:
    1.  **Internal feature normalization**:
        *   Takes raw, unscaled feature data (`X_dynamic` and `X_static`) upon initialization.
        *   Uses an `is_train` boolean flag to control its behavior.
        *   When `is_train=True`, the `_normalize_features` method is called to calculate min-max scaling parameters
            from the provided training data and applies the transformation. These computed parameters are stored internally.
        *   When `is_train=False` (for a validation or test set), it requires a `norm_params` dictionary containing
            the parameters from the training set. It then applies this exact transformation to the new data, ensuring
            consistent scaling and preventing data leakage.
        *   The `get_norm_params()` method allows retrieving the computed parameters from a training dataset instance.

    2.  **Data augmentation via random prefix unrolling**:
        *   The `__getitem__` method implements a data augmentation strategy. Instead of returning a fixed, full sequence,
            it randomly selects a prefix of a valid length from the full encounter data for each sample request.
        *   This means a single patient encounter can generate multiple different training samples of varying lengths,
            which helps the model generalize better.

    3.  **Dynamic target generation**:
        *   After selecting a random prefix, the survival target is calculated dynamically. The time-to-event (`rel_time`)
            is computed relative to the *last observation time* of the chosen prefix.
        *   This class returns the relative time and event indicator for AFT models.

    ### How it works:
    *   **`__init__(...)`**: Initializes the dataset by storing patient/encounter data.
        Immediately perform feature normalization based on the `is_train` flag, storing the scaled features
        (`self.X_dynamic`, `self.X_static`) for later use. Also keeps the original unscaled `dynamic_times` array.
    *   **`_normalize_features(...)`**: A detailed internal method that performs min-max scaling. Correctly handle
        masked (padded or invalid) values by computing statistics only on valid data points. Either compute
        new scaling parameters or apply existing ones.
    *   **`__getitem__(i)`**: This is the core method for generating a single training sample. Retrieve the full,
        pre-normalized data for an encounter, select a random prefix length, truncate the dynamic data,
        calculate the relative survival target, and returns all components as tensors ready for the `DataLoader`.
    """

    def __init__(self, indices, patient_ids, encounter_ids, X_dynamic, mask_dynamic, dynamic_times,
                 X_static, mask_static, result_times, result_indicators,
                 min_seq_len_perc, max_seq_len_perc, num_bins,
                 is_train, norm_params, b_min, b_max, seed,
                 augmentation_noise_std, logger):
        """
        Args:
            ...
            min_seq_len_perc (float): The minimum length of a sub-sequence as a percentage of the original
                                          sequence length (e.g., 0.2 for 20%).
            max_seq_len_perc (float): The maximum length of a sub-sequence as a percentage of the original
                                          sequence length (e.g., 0.8 for 80%).
            is_train (bool): If True, computes and stores normalization parameters.
            norm_params: If not is_train, this dict must contain the parameters ('dynamic' and 'static') for normalization.
            b_min, b_max: Target range for min-max scaling.
        """
        # Initialize variables
        self.patient_ids = patient_ids[indices]
        self.encounter_ids = encounter_ids[indices]
        # self.X_dynamic = X_dynamic[indices]  # Will be created by _normalize_features()
        self.mask_dynamic = mask_dynamic[indices]
        self.dynamic_times = dynamic_times[indices]
        # self.X_static = X_static[indices]  # Will be created by _normalize_features()
        self.mask_static = mask_static[indices]
        self.result_times = result_times[indices]
        self.result_indicators = result_indicators[indices]
        self.min_seq_len_perc = min_seq_len_perc
        self.max_seq_len_perc = max_seq_len_perc
        self.num_bins = num_bins
        self.is_train = is_train
        self.norm_params = {'dynamic': None, 'static': None}
        self.b_min = b_min
        self.b_max = b_max
        self.seed = seed
        self.augmentation_noise_std = augmentation_noise_std
        self.logger = logger

        self.rng = np.random.RandomState(self.seed)
        self.time_bins = np.linspace(0, num_bins, num_bins + 1)

        # --- Internal feature normalization (minmax only) ---
        if self.is_train:
            # Fit and transform training data
            self.X_dynamic, self.norm_params['dynamic'] = self._normalize_features(
                X=X_dynamic[indices], X_mask=self.mask_dynamic, feature_type='Dynamic', params=None,
                b_min=self.b_min, b_max=self.b_max
            )
            self.X_static, self.norm_params['static'] = self._normalize_features(
                X=X_static[indices], X_mask=self.mask_static, feature_type='Static', params=None,
                b_min=self.b_min, b_max=self.b_max
            )
        else:
            # Transform test/validation data using provided parameters
            if norm_params is None or 'dynamic' not in norm_params or 'static' not in norm_params:
                raise ValueError("norm_params must be provided for non-training dataset.")
            self.norm_params = norm_params
            self.X_dynamic, _ = self._normalize_features(
                X=X_dynamic[indices], X_mask=self.mask_dynamic, feature_type='Dynamic', params=self.norm_params['dynamic'],
                b_min=self.b_min, b_max=self.b_max
            )
            self.X_static, _ = self._normalize_features(
                X=X_static[indices], X_mask=self.mask_static, feature_type='Static', params=self.norm_params['static'],
                b_min=self.b_min, b_max=self.b_max
            )

    def _normalize_features(self, X, X_mask, feature_type, params, b_min, b_max, print_n=5):
        """
        Performs min-max normalization on a feature set (dynamic or static) within the dataset.

        This method is called internally during the initialization of the SurvivalDataset.
        - If `params` are not provided (i.e., for a training dataset), it calculates the
          scaling parameters for each feature based on the valid (unmasked) data and applies the scaling.
        - If `params` are provided (i.e., for a validation/test dataset), it applies the
          given scaling parameters to `X`, ensuring no data leakage.

        Args:
            X (np.ndarray): The feature data to normalize (2D for static, 3D for dynamic).
            X_mask (np.ndarray): The boolean mask corresponding to X (True for invalid/padded).
            feature_type (str): A string identifier ('Dynamic' or 'Static') for logging purposes.
            b_min (int): The target minimum value for the scaled data.
            b_max (int): The target maximum value for the scaled data.
            params (list, optional): A list of parameter tuples. If provided, these are used
                                     for scaling instead of being computed from X.
            print_n (int): The number of feature statistics to print for debugging.

        Returns:
            tuple: A tuple containing:
                - X_scaled (np.ndarray): The normalized feature data.
                - return_params (list): The list of parameter tuples used for scaling.
        """
        # Create a copy to avoid modifying the original array passed to the constructor.
        X_scaled = np.copy(X)
        num_features = X.shape[-1]

        # This list will store the normalization parameters for each feature.
        computed_params = []

        # Check if the input array is 3D (dynamic) or 2D (static).
        is_3d = X.ndim == 3

        # Iterate over each feature column to normalize it independently.
        for feature_idx in range(num_features):
            # Define the slice to extract the current feature's data across all samples and timesteps.
            slicer = (slice(None), slice(None), feature_idx) if is_3d else (slice(None), feature_idx)
            feature_data = X[slicer]
            feature_mask = X_mask[slicer]

            # Determine if we need to compute new scaling parameters or use existing ones.
            if params is not None:
                # Use pre-computed parameters (for validation/test set).
                param_tuple = params[feature_idx]
            else:
                # Compute parameters from the data (for training set).
                valid_data = feature_data[~feature_mask]
                if len(valid_data) > 0:
                    a_min, a_max = np.min(valid_data), np.max(valid_data)
                    param_tuple = (a_min, a_max)
                else:
                    # Handle case where a feature is all NaN
                    param_tuple = (0, 0)
                computed_params.append(param_tuple)

            # Apply min-max normalization
            norm_data = np.copy(feature_data)

            a_min, a_max = param_tuple
            if a_max == a_min:
                norm_data[~feature_mask] = b_min
            else:
                # Min-max scaling
                norm_data[~feature_mask] = b_min + (feature_data[~feature_mask] - a_min) * (b_max - b_min) / (
                            a_max - a_min)

            # After scaling, re-apply the mask by setting invalid/padded entries back to 0.0.
            norm_data[feature_mask] = 0.0

            # Place the normalized feature back into the output array.
            X_scaled[slicer] = norm_data

            # Print statistics for the first few features for verification.
            if feature_idx < print_n and params is None:
                self.logger.my_print(
                    f"  - {feature_type} Feature {feature_idx}: Original Range ({param_tuple[0]:.2f}, {param_tuple[1]:.2f}) -> Scaled Range ({b_min}, {b_max})")

        # If parameters were computed, return them. Otherwise, return the parameters that were passed in.
        return_params = computed_params if params is None else params

        assert return_params is not None
        return X_scaled, return_params

    def get_norm_params(self):
        """Returns the normalization parameters computed from the training data."""
        if not self.is_train:
            raise RuntimeError("Normalization parameters can only be retrieved from a training dataset.")
        return self.norm_params

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, i):
        """
        Variable-length samples: This function takes a full, population-padded sequence and generates a random,
        variable-length sub-interval from it. This is the data augmentation step. The output of __getitem__ is a sample
        with a sequence length that is almost always shorter than the population-wide maximum.

        This function is responsible for generating a single training sample by selecting a random
        contiguous sub-interval from the patient's history. The survival_collate_fn function is then responsible for
        padding these samples into a single batch tensor.

        This function's behavior differs for training and evaluation.
        - During training (`is_train=True`), this function performs data augmentation by generating a random,
            variable-length sub-sequence from the patient's full history. This helps the model generalize by exposing it
            to sequences of different lengths.
        - During evaluation (`is_train=False`), it returns the complete, un-truncated sequence for the patient to ensure
            consistent and full-history predictions.
        """
        # --- 1. Retrieve full sequence data for the patient ---
        full_X_dynamic = self.X_dynamic[i]
        full_mask_dynamic = self.mask_dynamic[i]
        full_dynamic_times = self.dynamic_times[i]
        original_result_time = self.result_times[i]
        original_result_indicator = self.result_indicators[i]

        # --- 2. Select a random sub-interval from the valid timesteps ---
        valid_indices = np.where(np.isfinite(full_dynamic_times))[0]
        num_valid_steps = len(valid_indices)

        if num_valid_steps == 0:  # Handle edge case of empty sequences
            last_idx = len(full_dynamic_times) - 1
            valid_indices = np.array([last_idx])
            num_valid_steps = 1

        # If not training, use the full sequence. Otherwise, perform data augmentation.
        if not self.is_train:
            start_idx_in_valid = 0
            end_idx_in_valid = num_valid_steps - 1
            min_len = 1
            max_len = num_valid_steps
            seq_len = num_valid_steps
        else:
            # Calculate minimum required length for the sub-sequence.
            min_len = math.ceil(self.min_seq_len_perc * num_valid_steps)
            min_len = max(1, min_len)

            # Calculate maximum required length for the sub-sequence.
            max_len = math.floor(self.max_seq_len_perc * num_valid_steps)
            max_len = max(1, max_len)

            # Ensure max_len is at least min_len
            if max_len < min_len:
                max_len = min_len

            # If the sequence is shorter than or equal to the required min_len, use the whole valid sequence.
            if num_valid_steps <= min_len:
                start_idx_in_valid = 0
                end_idx_in_valid = num_valid_steps - 1
                seq_len = num_valid_steps
            else:
                # Randomly choose a subsequence length between min_len and max_len (inclusive)
                seq_len = int(self.rng.randint(min_len, max_len + 1))
                # Determine possible start positions (in valid-index space)
                max_start_idx = num_valid_steps - seq_len
                start_idx_in_valid = int(self.rng.randint(0, max_start_idx + 1))
                end_idx_in_valid = start_idx_in_valid + seq_len - 1

        # Get the corresponding indices in the full (padded) array
        start_idx_in_full = valid_indices[start_idx_in_valid]
        end_idx_in_full = valid_indices[end_idx_in_valid]

        # --- 3. Truncate data and make dynamic_times relative to the sub-interval start ---
        sub_interval_slice = slice(start_idx_in_full, end_idx_in_full + 1)

        X_dyn_tensor = torch.from_numpy(full_X_dynamic[sub_interval_slice, :])
        mask_dyn_tensor = torch.from_numpy(full_mask_dynamic[sub_interval_slice, :])

        # --- Apply data augmentation (only during training) ---
        if self.is_train:
            # 1. Add Gaussian noise to features
            if self.augmentation_noise_std > 0:
                noise = torch.randn_like(X_dyn_tensor) * self.augmentation_noise_std
                # Only add noise to valid (non-masked) features
                X_dyn_tensor = X_dyn_tensor + noise * (~mask_dyn_tensor).float()

        # Get absolute dynamic_times for the interval
        absolute_dynamic_times_in_interval = full_dynamic_times[sub_interval_slice]
        # Make dynamic_times relative to the start of the interval
        relative_dynamic_times_in_interval = absolute_dynamic_times_in_interval - absolute_dynamic_times_in_interval[0]
        dynamic_times_tensor = torch.from_numpy(relative_dynamic_times_in_interval)

        # --- 4. Calculate new relative time and indicator ---
        last_absolute_time_in_interval = full_dynamic_times[end_idx_in_full]
        rel_time = original_result_time - last_absolute_time_in_interval

        # AFT models: preserve original indicator without horizon censoring
        rel_indicator = original_result_indicator

        # --- 5. Prepare static data and IDs ---
        X_static_tensor = torch.from_numpy(self.X_static[i])
        mask_static_tensor = torch.from_numpy(self.mask_static[i])
        patient_id_tensor = torch.tensor(self.patient_ids[i], dtype=torch.long)
        encounter_id_tensor = torch.tensor(self.encounter_ids[i], dtype=torch.long)
        rel_time_tensor = torch.tensor(rel_time, dtype=torch.float32)
        rel_indicator_tensor = torch.tensor(rel_indicator, dtype=torch.long)

        # --- 6. Return data for AFT ---

        return (patient_id_tensor, encounter_id_tensor, X_dyn_tensor, mask_dyn_tensor, dynamic_times_tensor,
                X_static_tensor, mask_static_tensor, rel_time_tensor, rel_indicator_tensor)


def survival_collate_fn(batch, max_global_len):
    """
    Batch-wise padding: This function receives a list of these variable-length samples (a batch). To create a single
    tensor for the model, all sequences in that specific batch must have the same length. It achieves this by finding
    the maximum length within the current batch and padding all shorter sequences in that batch up to that length.

    Custom collate for SurvivalDataset: Pads to batch max (or global cap).
    Right-align sequences: keep last pad_len steps; left-pad when needed.
    Handles dynamic and static features separately.
    """
    # Ensure the batch is not empty to avoid errors in subsequent operations.
    if len(batch) == 0:
        raise ValueError("Empty batch")

    # --- 1. Unpack and stack fixed-size data ---
    # Patient and encounter IDs are single values per sample, so they can be stacked directly.
    patient_ids = torch.stack([item[0] for item in batch])
    encounter_ids = torch.stack([item[1] for item in batch])

    # --- 2. Separate variable-length and static data into lists ---
    # Dynamic data components have variable sequence lengths and need padding.
    X_dynamic_list = [item[2] for item in batch]
    mask_dynamic_list = [item[3] for item in batch]
    dynamic_times_list = [item[4] for item in batch]

    # Static data has a fixed size per sample, but we'll stack it after handling dynamic data.
    X_static_list = [item[5] for item in batch]
    mask_static_list = [item[6] for item in batch]

    # --- 3. Pad variable-length dynamic data ---
    # Determine the target sequence length for padding. This is the max length in the current batch.
    batch_max_len = max(x.shape[0] for x in X_dynamic_list)
    # Optionally cap the padding length to a global maximum if provided.
    pad_len = batch_max_len if max_global_len is None else min(batch_max_len, max_global_len)

    # Initialize lists to hold the padded sequences.
    X_dynamic_padded, mask_dynamic_padded, dynamic_times_padded = [], [], []
    # Iterate through each sample's dynamic data to pad or truncate it.
    for x_i, m_i, t_i in zip(X_dynamic_list, mask_dynamic_list, dynamic_times_list):
        curr_len = x_i.shape[0]
        feat_dim = x_i.shape[1]
        # If the current sequence is shorter than the target padding length...
        if curr_len < pad_len:
            # Calculate the number of rows to pad.
            pad_rows = pad_len - curr_len
            # Create a padding tensor of zeros for features.
            X_pad = torch.zeros((pad_rows, feat_dim), dtype=x_i.dtype, device=x_i.device)
            # Create a boolean mask of `True` for the padded area, indicating these steps are invalid.
            M_pad = torch.ones((pad_rows, feat_dim), dtype=torch.bool, device=m_i.device)
            # Create a time padding tensor of `NaN`s. This tells the PositionalEncoding to ignore these steps.
            T_pad = torch.full((pad_rows,), float('nan'), dtype=t_i.dtype, device=t_i.device)

            # Prepend the padding to the original sequence to achieve right-alignment.
            X_i = torch.cat([X_pad, x_i], dim=0)
            M_i = torch.cat([M_pad, m_i], dim=0)
            T_i = torch.cat([T_pad, t_i], dim=0)
        else:
            # If the sequence is longer than or equal to pad_len, truncate it to keep only the last `pad_len` steps.
            X_i = x_i[-pad_len:, :]
            M_i = m_i[-pad_len:, :]
            T_i = t_i[-pad_len:]

        # Append the processed (padded or truncated) sequence to the lists.
        X_dynamic_padded.append(X_i)
        mask_dynamic_padded.append(M_i)
        dynamic_times_padded.append(T_i)

    # --- 4. Assemble the final batch tensors ---
    # Stack the lists of padded sequences into final batch tensors.
    X_dynamic_collated = torch.stack(X_dynamic_padded)  # Shape: (Batch, Pad_Len, Num_Dynamic_Features)
    mask_dynamic_collated = torch.stack(mask_dynamic_padded)  # Shape: (Batch, Pad_Len, Num_Dynamic_Features)
    dynamic_times_collated = torch.stack(dynamic_times_padded)  # Shape: (Batch, Pad_Len)

    # Stack the static features, which do not require padding.
    X_static_collated = torch.stack(X_static_list)  # Shape: (Batch, Num_Static_Features)
    mask_static_collated = torch.stack(mask_static_list)  # Shape: (Batch, Num_Static_Features)

    # --- 5. Return the collated batch for AFT ---
    rel_times = torch.stack([item[7] for item in batch])
    rel_indicators = torch.stack([item[8] for item in batch])
    
    return (patient_ids, encounter_ids, X_dynamic_collated, mask_dynamic_collated,
            dynamic_times_collated, X_static_collated, mask_static_collated, rel_times, rel_indicators)
