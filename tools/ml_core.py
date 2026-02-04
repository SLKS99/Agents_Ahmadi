"""
ML core logic shared between ml_automation (headless) and ml_models page.
Extracted to avoid importing the Streamlit page (which creates duplicate widgets).
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd

# sklearn for single GP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler


class GaussianProcessModel:
    """Gaussian Process model for predicting properties from composition and curve fitting features."""

    def __init__(self, kernel_type: str = "RBF", alpha: float = 1e-6):
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.gp = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.target_name = ""

    def _create_kernel(self, n_features: int):
        if self.kernel_type == "RBF":
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        elif self.kernel_type == "Matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=1.5)
        elif self.kernel_type == "RBF+Matern":
            kernel = ConstantKernel(1.0) * (
                RBF(length_scale=np.ones(n_features))
                + Matern(length_scale=np.ones(n_features), nu=1.5)
            )
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        kernel += WhiteKernel(noise_level=self.alpha)
        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str):
        self.feature_names = feature_names
        self.target_name = target_name
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        kernel = self._create_kernel(X.shape[1])
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=10,
            normalize_y=False,
        )
        self.gp.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        X_scaled = self.scaler_X.transform(X)
        if return_std:
            y_pred_scaled, std_scaled = self.gp.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            std = std_scaled * self.scaler_y.scale_[0]
            return y_pred, std
        else:
            y_pred_scaled = self.gp.predict(X_scaled, return_std=False)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            return y_pred, None

    def acquisition_function(self, X: np.ndarray, method: str = "UCB", beta: float = 2.0) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing acquisition")
        y_pred, std = self.predict(X, return_std=True)
        if method == "UCB":
            return y_pred + beta * std
        elif method == "EI":
            best_y = np.max(
                self.scaler_y.inverse_transform(self.gp.y_train_.reshape(-1, 1))
            )
            z = (y_pred - best_y) / (std + 1e-9)
            return std * (
                z * 0.5 * (1 + np.sign(z))
                + (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2)
            )
        elif method == "PI":
            from scipy.stats import norm
            best_y = np.max(
                self.scaler_y.inverse_transform(self.gp.y_train_.reshape(-1, 1))
            )
            z = (y_pred - best_y) / (std + 1e-9)
            return norm.cdf(z)
        else:
            return y_pred + beta * std


def extract_features_from_results(
    results: Dict[str, Any], composition_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract features from curve fitting results and composition data."""
    feature_rows = []
    target_values = []

    for well_name, well_data in results.get("wells", {}).items():
        fitting_results = well_data.get("fitting_results", {})
        quality_metrics = fitting_results.get("quality_metrics", {})
        peaks = fitting_results.get("quality_peaks", [])

        row = {"well": well_name}
        if well_name in composition_df.columns:
            for material in composition_df.index:
                row[f"composition_{material}"] = composition_df.loc[material, well_name]
        else:
            for material in composition_df.index:
                row[f"composition_{material}"] = 0.0

        for i in range(3):
            if i < len(peaks):
                peak = peaks[i]
                row[f"peak_{i+1}_wavelength"] = peak.get("center_nm", 0)
                row[f"peak_{i+1}_intensity"] = peak.get("height", 0)
                row[f"peak_{i+1}_fwhm"] = peak.get("FWHM_nm", 0)
                row[f"peak_{i+1}_area"] = peak.get("area", 0)
            else:
                row[f"peak_{i+1}_wavelength"] = 0
                row[f"peak_{i+1}_intensity"] = 0
                row[f"peak_{i+1}_fwhm"] = 0
                row[f"peak_{i+1}_area"] = 0

        row["r_squared"] = quality_metrics.get("r_squared", 0)
        row["rmse"] = quality_metrics.get("rmse", 0)
        row["num_peaks"] = len(peaks)

        feature_rows.append(row)
        target_values.append(quality_metrics.get("r_squared", 0))

    X_df = pd.DataFrame(feature_rows)
    y_series = pd.Series(target_values, name="target")
    return X_df, y_series


def generate_exploration_candidates(
    composition_df: pd.DataFrame, n_candidates: int = 20
) -> pd.DataFrame:
    """Generate candidate compositions for exploration."""
    materials = composition_df.index.tolist()
    n_materials = len(materials)
    candidates = []
    np.random.seed(42)

    for _ in range(n_candidates):
        composition = np.random.dirichlet(np.ones(n_materials))
        row = {"well": f"candidate_{len(candidates)+1}"}
        for i, material in enumerate(materials):
            row[f"composition_{material}"] = composition[i]
        for i in range(3):
            row[f"peak_{i+1}_wavelength"] = 0
            row[f"peak_{i+1}_intensity"] = 0
            row[f"peak_{i+1}_fwhm"] = 0
            row[f"peak_{i+1}_area"] = 0
        row["r_squared"] = 0
        row["rmse"] = 0
        row["num_peaks"] = 0
        candidates.append(row)

    return pd.DataFrame(candidates)


def instability_score(
    df: pd.DataFrame,
    compositions_with_multiple_peaks: Optional[List[Tuple]] = None,
    target_wavelength: float = 700,
    multiple_peak_penalty: float = 0.5,
    wavelength_tolerance: float = 10,
    degradation_weight: float = 0.4,
    position_weight: float = 0.6,
) -> np.ndarray:
    """Compute instability score for dual GP."""
    max_score = 3
    stb_scores = []
    if compositions_with_multiple_peaks is None:
        compositions_with_multiple_peaks = []
    multiple_peaks_set = set(tuple(row) for row in compositions_with_multiple_peaks)

    initial_pos_col = final_pos_col = initial_int_col = final_int_col = None
    for col in df.columns:
        if "initial_peak_position" in col.lower() or (
            "peak_1_wavelength" in col.lower() and "initial" in col.lower()
        ):
            initial_pos_col = col
        if "final_peak_position" in col.lower() or (
            "peak_1_wavelength" in col.lower() and "final" in col.lower()
        ):
            final_pos_col = col
        if "initial_peak_intensity" in col.lower() or (
            "peak_1_intensity" in col.lower() and "initial" in col.lower()
        ):
            initial_int_col = col
        if "final_peak_intensity" in col.lower() or (
            "peak_1_intensity" in col.lower() and "final" in col.lower()
        ):
            final_int_col = col

    if initial_pos_col is None and "Peak_1_Wavelength" in df.columns:
        initial_pos_col = "Peak_1_Wavelength"
    if final_pos_col is None and "Peak_1_Wavelength" in df.columns:
        final_pos_col = "Peak_1_Wavelength"
    if initial_int_col is None and "Peak_1_Intensity" in df.columns:
        initial_int_col = "Peak_1_Intensity"
    if final_int_col is None and "Peak_1_Intensity" in df.columns:
        final_int_col = "Peak_1_Intensity"

    has_comp_iter = "composition_number" in df.columns and "iteration" in df.columns

    for index, row in df.iterrows():
        current_comp = (
            (row["composition_number"], row["iteration"]) if has_comp_iter else None
        )
        if initial_pos_col and final_pos_col:
            peak_positions_int = row[initial_pos_col] if pd.notna(row[initial_pos_col]) else 0
            peak_positions_fin = row[final_pos_col] if pd.notna(row[final_pos_col]) else 0
        else:
            peak_positions_int = peak_positions_fin = 0
        if initial_int_col and final_int_col:
            peak_intensities_int = row[initial_int_col] if pd.notna(row[initial_int_col]) else 0
            peak_intensities_fin = row[final_int_col] if pd.notna(row[final_int_col]) else 0
        else:
            peak_intensities_int = peak_intensities_fin = 0

        if peak_intensities_int == 0 and peak_intensities_fin == 0:
            stb_scores.append(max_score)
            continue

        intensity_change = np.abs(peak_intensities_int - peak_intensities_fin) / max(
            peak_intensities_int + peak_intensities_fin, 1e-10
        )
        intensity_score = min(intensity_change, 1) * degradation_weight

        initial_position_deviation = max(
            abs(peak_positions_int - target_wavelength) - wavelength_tolerance, 0
        )
        final_position_deviation = max(
            abs(peak_positions_fin - target_wavelength) - wavelength_tolerance, 0
        )
        position_score = (
            min(initial_position_deviation, final_position_deviation) / target_wavelength
        ) * position_weight

        multiple_peaks_score = (
            multiple_peak_penalty if (current_comp and current_comp in multiple_peaks_set) else 0
        )
        total_score = min(intensity_score + position_score + multiple_peaks_score, max_score)
        stb_scores.append(total_score)

    return np.array(stb_scores)


# TorchGaussianProcess for dual GP
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class TorchGaussianProcess:
    """Simple GP in PyTorch for dual-objective optimization."""

    def __init__(
        self,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-4,
        device: Optional[str] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch required for TorchGaussianProcess")
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = self.y_train = self.L = self.alpha = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def _kernel(self, X1: "torch.Tensor", X2: "torch.Tensor") -> "torch.Tensor":
        lengthscale = torch.tensor(self.lengthscale, device=self.device, dtype=torch.float32)
        variance = torch.tensor(self.variance, device=self.device, dtype=torch.float32)
        diff = (X1[:, None, :] - X2[None, :, :]) / lengthscale
        sqdist = torch.sum(diff**2, dim=-1)
        return variance * torch.exp(-0.5 * sqdist)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        self.x_mean = X_t.mean(dim=0, keepdim=True)
        self.x_std = X_t.std(dim=0, keepdim=True) + 1e-8
        Xs = (X_t - self.x_mean) / self.x_std
        self.y_mean = float(y_t.mean().item())
        self.y_std = float(y_t.std().item() + 1e-8)
        ys = (y_t - self.y_mean) / self.y_std
        self.X_train = Xs
        self.y_train = ys
        K = self._kernel(Xs, Xs) + self.noise * torch.eye(Xs.shape[0], device=self.device)
        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(ys.unsqueeze(-1), self.L).squeeze(-1)

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.X_train is None or self.alpha is None or self.L is None:
            raise ValueError("Model must be fitted before prediction")
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Xs = (X_t - self.x_mean) / self.x_std
        K_s = self._kernel(Xs, self.X_train)
        mean_s = K_s @ self.alpha
        mean = (mean_s * self.y_std + self.y_mean).detach().cpu().numpy()
        if not return_std:
            return mean, None
        v = torch.cholesky_solve(K_s.T, self.L)
        K_ss_diag = torch.ones(Xs.shape[0], device=self.device) * self.variance
        var = K_ss_diag - torch.sum(K_s * v.T, dim=1)
        var = torch.clamp(var, min=1e-9)
        std = (torch.sqrt(var) * self.y_std).detach().cpu().numpy()
        return mean, std
