"""
Loss functions for survival analysis.
"""

import torch
import torch.nn as nn
from torchsurv.loss import weibull


class AFTLoss(nn.Module):
    """
    Loss for Weibull accelerated failure time (AFT) model using torchsurv library.
    """
    def __init__(self, epsilon, logger):
        super().__init__()
        self.epsilon = epsilon
        self.logger = logger

    def forward(self, log_params, result_times, result_indicators):
        """
        Args:
            log_params: (B, 2) log scale and log shape parameters from model
            result_times: (B,) time-to-event or censoring
            result_indicators: (B,) event indicator (1 for event, 0 for censored)

        Returns:
            loss: negative partial log-likelihood
        """
        # Ensure correct shapes and types
        assert log_params.dim() == 2 and log_params.shape[-1] == 2, \
            f"AFT model log_params (log_scale, log_shape) must have shape (batch_size, 2), got {log_params.shape}"

        # Add small epsilon to prevent log(0)
        log_params = log_params.contiguous()
        result_times = (result_times + self.epsilon).contiguous()
        result_indicators = result_indicators.contiguous().bool()

        # It appears the torchsurv.loss.weibull.neg_log_likelihood function has an internal issue when processing a
        # batch with a single sample, as it may try to squeeze a dimension that does not exist, leading to an error
        # Workaround for torchsurv issue with batch_size=1: by providing two identical samples, the resulting mean loss
        # will be equivalent to the loss of the single sample
        if log_params.shape[0] == 1:
            # Duplicate the single sample to create a batch of size 2
            log_params = log_params.repeat(2, 1)
            result_times = result_times.repeat(2)
            result_indicators = result_indicators.repeat(2)

        # torchsurv expects: (log_params, event, time)
        loss = weibull.neg_log_likelihood(
            log_params=log_params,
            event=result_indicators,
            time=result_times
        )

        # Check for NaN and return large penalty if detected
        if torch.isnan(loss):
            self.logger.my_print(f"NaN loss detected in AFTLoss. Returning large penalty.", level='warning')
            return torch.tensor(1e6, device=log_params.device, requires_grad=True)

        return loss


class CalibrationRegressionLoss(nn.Module):
    """
    Calibration Loss inspired by logistic calibration (Platt scaling).
    Measures whether the model's predicted probabilities are well-calibrated.
    
    A well-calibrated model should have a calibration slope close to 1 and intercept close to 0
    when regressing observed outcomes on predicted probabilities.
    
    This loss encourages the model to produce calibrated predictions by penalizing deviations
    from the ideal calibration line.
    """
    def __init__(self, num_bins, epsilon):
        super().__init__()
        self.num_bins = num_bins
        self.epsilon = epsilon
        
    def forward(self, survival_probs, result_times, result_indicators):
        """
        Args:
            survival_probs: (B, num_bins) predicted survival probabilities
            result_times: (B,) relative time to event/censoring
            result_indicators: (B,) 1 for event, 0 for censored
            
        Returns:
            calibration_regression_loss: penalty based on calibration slope/intercept deviation from ideal
        """
        device = survival_probs.device
        
        # We'll compute calibration at a fixed evaluation time (e.g., median time bin)
        # to avoid computational complexity of evaluating at all time points
        eval_time_idx = self.num_bins // 2
        eval_time = float(eval_time_idx + 1)
        
        # Determine ground truth at evaluation time
        survived_to_t = result_times >= eval_time
        censored_before_t = (result_indicators == 0) & (result_times < eval_time)
        
        # Only use samples where outcome is known at eval_time
        valid_mask = ~censored_before_t

        # Linear regression requires minimum 2 points to fit a line (estimate slope and intercept).
        # Also, mathematical constraint: cannot compute covariance/variance with < 2 samples
        if valid_mask.sum() < 2:
            # Not enough samples for calibration, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Get predictions and outcomes for valid samples
        y_pred = survival_probs[valid_mask, eval_time_idx]  # Predicted survival prob
        y_true = survived_to_t[valid_mask].float()  # True survival status
        
        # Convert to logit space for calibration regression
        # S(t) -> log(S(t) / (1 - S(t)))
        y_pred_clamped = torch.clamp(y_pred, min=self.epsilon, max=1.0 - self.epsilon)
        pred_logits = torch.log(y_pred_clamped / (1.0 - y_pred_clamped))
        
        # Compute calibration by fitting: y_true ~ intercept + slope * pred_logits
        # We want slope ≈ 1 and intercept ≈ 0 for perfect calibration
        
        # Center predictions for numerical stability
        pred_mean = pred_logits.mean()
        pred_centered = pred_logits - pred_mean
        
        # Compute optimal slope using least squares
        # slope = cov(y, pred) / var(pred)
        cov = (pred_centered * (y_true - y_true.mean())).mean()
        var = (pred_centered ** 2).mean() + self.epsilon
        slope = cov / var
        
        # Compute intercept
        intercept = y_true.mean() - slope * pred_logits.mean()
        
        # Loss: penalize deviation from ideal slope=1, intercept=0
        slope_penalty = (slope - 1.0) ** 2
        intercept_penalty = intercept ** 2
        
        calibration_regression_loss = slope_penalty + intercept_penalty

        return calibration_regression_loss


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss combining discrimination-based and calibration-based losses.
    
    For AFT:
        L = w_discrimination * discrimination_loss + w_calibration_regression * calibration_regression_loss
    """
    def __init__(self, discrimination_criterion, num_bins, epsilon,
                 w_discrimination, w_calibration_regression):
        super().__init__()
        self.discrimination_criterion = discrimination_criterion
        self.num_bins = num_bins
        self.epsilon = epsilon
        
        # Loss weights
        self.w_discrimination = w_discrimination
        self.w_calibration_regression = w_calibration_regression

        # Calibration loss component
        if w_calibration_regression != 0.0:
            self.calibration_regression_loss = CalibrationRegressionLoss(num_bins=num_bins, epsilon=epsilon)

    def forward(self, outputs, result_times, result_indicators):
        """
        Args:
            outputs: Model outputs (log_params for AFT)
            targets: Target labels (same as result_indicators for AFT)
            result_times: (B,) relative time to event/censoring
            result_indicators: (B,) 1 for event, 0 for censored
        
        Returns:
            total_loss: Weighted combination of discrimination and calibration losses
            loss_dict: Dictionary containing individual loss components for logging
        """
        # Initialize variables
        device = outputs.device
        
        # 1. Compute discrimination loss (AFT)
        discrimination_loss = self.discrimination_criterion(
            log_params=outputs, result_times=result_times, result_indicators=result_indicators
        )
        
        total_loss = self.w_discrimination * discrimination_loss
        
        # 2. Compute calibration loss (only if weight != 0.0)
        if self.w_calibration_regression != 0.0:
            # For AFT model, compute survival probabilities from Weibull parameters
            log_scale = outputs[:, 0]
            log_shape = outputs[:, 1]
            scale = torch.exp(log_scale)
            shape = torch.exp(log_shape)
            
            # Compute survival probabilities at each time bin
            time_points = torch.arange(1, self.num_bins + 1, dtype=torch.float32, device=device)
            time_points = time_points.unsqueeze(0).expand(outputs.shape[0], -1)  # (B, num_bins)
            
            # Weibull survival function: S(t) = exp(-(t/scale)^shape)
            survival_probs = torch.exp(-((time_points / scale.unsqueeze(1)) ** shape.unsqueeze(1)))
            
            # Compute calibration loss
            cal_slope_loss = self.calibration_regression_loss(survival_probs, result_times, result_indicators)
            total_loss = total_loss + self.w_calibration_regression * cal_slope_loss

        return total_loss
