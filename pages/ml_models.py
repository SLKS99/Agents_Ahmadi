import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import torch
import subprocess
from pathlib import Path
from tools.memory import MemoryManager

memory = MemoryManager()
memory.init_session()

st.set_page_config(layout="wide")
st.title("ML Models")
st.markdown(
    "Train and compare different ML models on curve fitting results and composition data, "
    "then use them for optimization cycles (exploration/exploitation)."
)

# -------------------------------------------------------------------------
# Global ML method selector (single hub for all ML methods)
# -------------------------------------------------------------------------

MODEL_SINGLE_GP = "Single-objective GP (scikit-learn)"
MODEL_DUAL_TORCH_GP = "Dual-objective GP (PyTorch)"
MODEL_MONTE_CARLO_TREE = "Monte Carlo Decision Tree (external)"

available_models = [MODEL_SINGLE_GP, MODEL_DUAL_TORCH_GP, MODEL_MONTE_CARLO_TREE]

# If a workflow-specific choice exists, use it as default
default_model = st.session_state.get("workflow_ml_model_choice")
if default_model not in available_models:
    default_model = st.session_state.get("optimization_model_choice", MODEL_SINGLE_GP)
if default_model not in available_models:
    default_model = MODEL_SINGLE_GP

default_index = available_models.index(default_model)

model_choice = st.selectbox(
    "Select ML method for optimization",
    available_models,
    index=default_index,
    help=(
        "Choose which ML model to use for optimization cycles. "
        "You can add more methods here in the future."
    ),
)

# Persist choice for use by other pages/agents in the workflow and automation
st.session_state.optimization_model_choice = model_choice

# Basic per-model config object used by automated runs (curve_fitting.py → ml_automation.py)
if "ml_model_config" not in st.session_state or not isinstance(st.session_state.ml_model_config, dict):
    st.session_state.ml_model_config = {}

# Show automation status
if st.session_state.get("auto_ml_after_curve_fitting", False):
    st.info("Automation Enabled: this model will run automatically after curve fitting completes.")

class GaussianProcessModel:
    """Gaussian Process model for predicting properties from composition and curve fitting features."""
    
    def __init__(self, kernel_type: str = "RBF", alpha: float = 1e-6):
        """
        Initialize GP model.
        
        Args:
            kernel_type: Type of kernel ("RBF", "Matern", or "RBF+Matern")
            alpha: Noise level for observations
        """
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.gp = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.target_name = ""
        
    def _create_kernel(self, n_features: int):
        """Create kernel based on type."""
        if self.kernel_type == "RBF":
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        elif self.kernel_type == "Matern":
            kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=1.5)
        elif self.kernel_type == "RBF+Matern":
            kernel = ConstantKernel(1.0) * (RBF(length_scale=np.ones(n_features)) + 
                                           Matern(length_scale=np.ones(n_features), nu=1.5))
        else:
            kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(n_features))
        
        # Add white noise kernel
        kernel += WhiteKernel(noise_level=self.alpha)
        return kernel
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str):
        """
        Fit GP model to data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Names of features
            target_name: Name of target variable
        """
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Scale features and target
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Create kernel
        kernel = self._create_kernel(X.shape[1])
        
        # Create and fit GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=10,
            normalize_y=False  # We're scaling manually
        )
        
        self.gp.fit(X_scaled, y_scaled)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict target values and uncertainty.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            return_std: Whether to return standard deviation
            
        Returns:
            Mean predictions and optionally standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler_X.transform(X)
        
        if return_std:
            y_pred_scaled, std_scaled = self.gp.predict(X_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            std = std_scaled * self.scaler_y.scale_[0]  # Scale uncertainty
            return y_pred, std
        else:
            y_pred_scaled = self.gp.predict(X_scaled, return_std=False)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            return y_pred, None
    
    def acquisition_function(self, X: np.ndarray, method: str = "UCB", beta: float = 2.0) -> np.ndarray:
        """
        Compute acquisition function for exploration.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            method: Acquisition method ("UCB", "EI", or "PI")
            beta: Exploration-exploitation trade-off (for UCB)
            
        Returns:
            Acquisition function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing acquisition")
        
        y_pred, std = self.predict(X, return_std=True)
        
        if method == "UCB":
            # Upper Confidence Bound
            return y_pred + beta * std
        elif method == "EI":
            # Expected Improvement (simplified)
            # Need best observed value - approximate from training data
            best_y = np.max(self.scaler_y.inverse_transform(
                self.gp.y_train_.reshape(-1, 1)
            ))
            z = (y_pred - best_y) / (std + 1e-9)
            return std * (z * 0.5 * (1 + np.sign(z)) + 
                         (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2))
        elif method == "PI":
            # Probability of Improvement
            best_y = np.max(self.scaler_y.inverse_transform(
                self.gp.y_train_.reshape(-1, 1)
            ))
            z = (y_pred - best_y) / (std + 1e-9)
            from scipy.stats import norm
            return norm.cdf(z)
        else:
            return y_pred + beta * std


# -------------------------------------------------------------------------
# Monte Carlo Decision Tree integration (external project)
# -------------------------------------------------------------------------

def render_monte_carlo_tree_ui():
    """
    Simple integration of the external Monte Carlo Decision Tree project as an ML method.
    
    This does NOT re-implement the model. Instead, it allows:
      - configuring the external repo path
      - running `python main.py` from that repo
      - capturing and displaying console output
    """
    st.subheader("Monte Carlo Decision Tree (external project)")
    st.markdown(
        "This option calls the external Monte Carlo Decision Tree project in "
        "`C:\\Users\\shery\\monte carlo decision tree` to generate optimization "
        "candidates based on your existing experiment history."
    )
    
    # Configuration in session_state so automation can reuse it
    cfg = st.session_state.get("ml_model_config", {})
    mc_cfg = cfg.get("monte_carlo_tree", {})
    
    default_repo = mc_cfg.get(
        "repo_path",
        str(Path(__file__).resolve().parents[2] / "monte carlo decision tree")
    )
    
    repo_path = st.text_input(
        "Monte Carlo Decision Tree repo path",
        value=default_repo,
        help="Folder that contains `main.py` for the Monte Carlo Decision Tree project.",
        key="mc_repo_path_input",
    )
    
    n_attempts = st.number_input(
        "Number of Monte Carlo attempts per cycle (n_attempts)",
        min_value=10,
        max_value=5000,
        value=int(mc_cfg.get("n_attempts", 500)),
        step=10,
        key="mc_n_attempts_input",
    )
    
    use_agent = st.checkbox(
        "Run with LLM agent (`--with-agent --auto-apply`)",
        value=bool(mc_cfg.get("with_agent", False)),
        key="mc_with_agent_input",
    )
    
    # Save config back to session so automation can use it
    st.session_state.ml_model_config["monte_carlo_tree"] = {
        "repo_path": repo_path,
        "n_attempts": int(n_attempts),
        "with_agent": use_agent,
    }
    
    st.markdown("### Run Monte Carlo Decision Tree manually")
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("Run Monte Carlo Decision Tree now", key="run_mc_tree_now"):
            run_cmd = [os.environ.get("PYTHON_EXECUTABLE", "python"), "main.py"]
            # We pass n_attempts via environment to avoid changing the external CLI
            env = os.environ.copy()
            env["MC_N_ATTEMPTS"] = str(int(n_attempts))
            if use_agent:
                run_cmd.append("--with-agent")
                run_cmd.append("--auto-apply")
            
            repo = Path(repo_path)
            if not repo.exists():
                st.error(f"Repository path does not exist: {repo}")
                return
            
            with st.spinner("Running Monte Carlo Decision Tree..."):
                try:
                    result = subprocess.run(
                        run_cmd,
                        cwd=str(repo),
                        capture_output=True,
                        text=True,
                        timeout=None,
                    )
                    st.subheader("Monte Carlo Decision Tree output")
                    st.code(result.stdout or "(no stdout)", language=None)
                    if result.stderr:
                        st.subheader("stderr")
                        st.code(result.stderr, language=None)
                    if result.returncode == 0:
                        st.success("Monte Carlo Decision Tree finished successfully.")
                    else:
                        st.error(f"Process exited with code {result.returncode}.")
                except Exception as e:
                    st.error(f"Error running Monte Carlo Decision Tree: {e}")



class TorchGaussianProcess:
    """
    Simple Gaussian Process regression implemented in PyTorch.
    Uses an RBF kernel and exact inference with Cholesky factorisation.
    """

    def __init__(
        self,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise = noise
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.X_train: Optional[torch.Tensor] = None
        self.y_train: Optional[torch.Tensor] = None
        self.L: Optional[torch.Tensor] = None
        self.alpha: Optional[torch.Tensor] = None

        # For simple standardisation
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[float] = None
        self.y_std: Optional[float] = None

    def _kernel(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        """RBF kernel."""
        # X1: [n1, d], X2: [n2, d]
        lengthscale = torch.tensor(self.lengthscale, device=self.device, dtype=torch.float32)
        variance = torch.tensor(self.variance, device=self.device, dtype=torch.float32)
        diff = (X1[:, None, :] - X2[None, :, :]) / lengthscale
        sqdist = torch.sum(diff**2, dim=-1)
        return variance * torch.exp(-0.5 * sqdist)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to data."""
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)

        # Standardise
        self.x_mean = X_t.mean(dim=0, keepdim=True)
        self.x_std = X_t.std(dim=0, keepdim=True) + 1e-8
        Xs = (X_t - self.x_mean) / self.x_std

        self.y_mean = float(y_t.mean().item())
        self.y_std = float(y_t.std().item() + 1e-8)
        ys = (y_t - self.y_mean) / self.y_std

        self.X_train = Xs
        self.y_train = ys

        K = self._kernel(Xs, Xs)
        n = K.shape[0]
        K = K + (self.noise * torch.eye(n, device=self.device))

        self.L = torch.linalg.cholesky(K)
        self.alpha = torch.cholesky_solve(ys.unsqueeze(-1), self.L).squeeze(-1)

    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict mean and (optionally) standard deviation at new points."""
        if self.X_train is None or self.alpha is None or self.L is None:
            raise ValueError("Model must be fitted before prediction")

        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Xs = (X_t - self.x_mean) / self.x_std

        K_s = self._kernel(Xs, self.X_train)  # [n*, n]
        mean_s = K_s @ self.alpha  # [n*]

        # Unstandardise
        mean = mean_s * self.y_std + self.y_mean

        if not return_std:
            return mean.detach().cpu().numpy(), None

        # Compute variance
        v = torch.cholesky_solve(K_s.T, self.L)  # [n, n*]
        K_ss_diag = torch.ones(Xs.shape[0], device=self.device) * self.variance
        var = K_ss_diag - torch.sum(K_s * v.T, dim=1)
        var = torch.clamp(var, min=1e-9)
        std = torch.sqrt(var) * self.y_std

        return mean.detach().cpu().numpy(), std.detach().cpu().numpy()


def load_curve_fitting_results(json_path: str) -> Dict[str, Any]:
    """Load curve fitting results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_composition_data(csv_path: str) -> pd.DataFrame:
    """Load composition CSV file."""
    return pd.read_csv(csv_path, index_col=0)


def extract_features_from_results(results: Dict[str, Any], composition_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features from curve fitting results and composition data.
    
    Returns:
        X: Feature matrix (DataFrame)
        y: Target values (Series) - using R² as default target
    """
    feature_rows = []
    target_values = []
    
    for well_name, well_data in results.get('wells', {}).items():
        fitting_results = well_data.get('fitting_results', {})
        quality_metrics = fitting_results.get('quality_metrics', {})
        peaks = fitting_results.get('quality_peaks', [])
        
        # Extract features
        row = {'well': well_name}
        
        # Composition features
        if well_name in composition_df.columns:
            for material in composition_df.index:
                row[f'composition_{material}'] = composition_df.loc[material, well_name]
        else:
            # If well not in composition, use zeros
            for material in composition_df.index:
                row[f'composition_{material}'] = 0.0
        
        # Peak features (up to 3 peaks)
        for i in range(3):
            if i < len(peaks):
                peak = peaks[i]
                row[f'peak_{i+1}_wavelength'] = peak.get('center_nm', 0)
                row[f'peak_{i+1}_intensity'] = peak.get('height', 0)
                row[f'peak_{i+1}_fwhm'] = peak.get('FWHM_nm', 0)
                row[f'peak_{i+1}_area'] = peak.get('area', 0)
            else:
                row[f'peak_{i+1}_wavelength'] = 0
                row[f'peak_{i+1}_intensity'] = 0
                row[f'peak_{i+1}_fwhm'] = 0
                row[f'peak_{i+1}_area'] = 0
        
        # Quality metrics as features
        row['r_squared'] = quality_metrics.get('r_squared', 0)
        row['rmse'] = quality_metrics.get('rmse', 0)
        row['num_peaks'] = len(peaks)
        
        feature_rows.append(row)
        
        # Target: Use R² as default (can be changed)
        target_values.append(quality_metrics.get('r_squared', 0))
    
    X_df = pd.DataFrame(feature_rows)
    y_series = pd.Series(target_values, name='target')
    
    return X_df, y_series


def instability_score(
    df: pd.DataFrame,
    compositions_with_multiple_peaks: Optional[List[Tuple]] = None,
    target_wavelength: float = 700,
    multiple_peak_penalty: float = 0.5,
    wavelength_tolerance: float = 10,
    degradation_weight: float = 0.4,
    position_weight: float = 0.6,
) -> np.ndarray:
    """
    Compute instability score based on peak degradation and position deviation.
    
    This matches the implementation from Dual_GP_Film_RT_Cs_PEA_BDA_04M_DMF9DMSO1_V2_ZIP.ipynb
    
    Args:
        df: DataFrame with columns:
            - 'initial_peak_positions' or 'Peak_1_Wavelength' (initial)
            - 'final_peak_positions' or 'Peak_1_Wavelength' (final, if multiple reads)
            - 'initial_peak_intensities' or 'Peak_1_Intensity' (initial)
            - 'final_peak_intensities' or 'Peak_1_Intensity' (final)
            - 'composition_number' and 'iteration' (optional, for multiple peaks detection)
        compositions_with_multiple_peaks: List of tuples (composition_number, iteration) that have multiple peaks
        target_wavelength: Target wavelength for peak position
        multiple_peak_penalty: Penalty score for compositions with multiple peaks
        wavelength_tolerance: Tolerance around target wavelength
        degradation_weight: Weight for intensity degradation component
        position_weight: Weight for position deviation component
    
    Returns:
        Array of instability scores (higher = more unstable, max = 3)
    """
    max_score = 3
    stb_scores = []
    
    # Handle multiple peaks set
    if compositions_with_multiple_peaks is None:
        compositions_with_multiple_peaks = []
    multiple_peaks_set = set(tuple(row) for row in compositions_with_multiple_peaks)
    
    # Detect column names (handle different naming conventions)
    initial_pos_col = None
    final_pos_col = None
    initial_int_col = None
    final_int_col = None
    
    # Try to find initial/final columns
    for col in df.columns:
        if 'initial_peak_position' in col.lower() or ('peak_1_wavelength' in col.lower() and 'initial' in col.lower()):
            initial_pos_col = col
        if 'final_peak_position' in col.lower() or ('peak_1_wavelength' in col.lower() and 'final' in col.lower()):
            final_pos_col = col
        if 'initial_peak_intensity' in col.lower() or ('peak_1_intensity' in col.lower() and 'initial' in col.lower()):
            initial_int_col = col
        if 'final_peak_intensity' in col.lower() or ('peak_1_intensity' in col.lower() and 'final' in col.lower()):
            final_int_col = col
    
    # Fallback: use Peak_1 columns if initial/final not found
    if initial_pos_col is None and 'Peak_1_Wavelength' in df.columns:
        initial_pos_col = 'Peak_1_Wavelength'
    if final_pos_col is None and 'Peak_1_Wavelength' in df.columns:
        final_pos_col = 'Peak_1_Wavelength'  # Will use same if no time series
    
    if initial_int_col is None and 'Peak_1_Intensity' in df.columns:
        initial_int_col = 'Peak_1_Intensity'
    if final_int_col is None and 'Peak_1_Intensity' in df.columns:
        final_int_col = 'Peak_1_Intensity'
    
    # Check if we have composition_number and iteration for multiple peaks detection
    has_comp_iter = 'composition_number' in df.columns and 'iteration' in df.columns
    
    for index, row in df.iterrows():
        # Get composition identifier for multiple peaks check
        if has_comp_iter:
            current_comp = (row['composition_number'], row['iteration'])
        else:
            current_comp = None
        
        # Extract peak positions and intensities
        if initial_pos_col and final_pos_col:
            peak_positions_int = row[initial_pos_col] if pd.notna(row[initial_pos_col]) else 0
            peak_positions_fin = row[final_pos_col] if pd.notna(row[final_pos_col]) else 0
        else:
            peak_positions_int = 0
            peak_positions_fin = 0
        
        if initial_int_col and final_int_col:
            peak_intensities_int = row[initial_int_col] if pd.notna(row[initial_int_col]) else 0
            peak_intensities_fin = row[final_int_col] if pd.notna(row[final_int_col]) else 0
        else:
            peak_intensities_int = 0
            peak_intensities_fin = 0
        
        # If both intensities are zero, assign max score (most unstable)
        if peak_intensities_int == 0 and peak_intensities_fin == 0:
            stb_scores.append(max_score)
            continue
        
        # Intensity degradation component
        intensity_change = np.abs(peak_intensities_int - peak_intensities_fin) / max(peak_intensities_int + peak_intensities_fin, 1e-10)
        intensity_score = min(intensity_change, 1) * degradation_weight
        
        # Position deviation component
        initial_position_deviation = max(abs(peak_positions_int - target_wavelength) - wavelength_tolerance, 0)
        final_position_deviation = max(abs(peak_positions_fin - target_wavelength) - wavelength_tolerance, 0)
        position_score = (min(initial_position_deviation, final_position_deviation) / target_wavelength) * position_weight
        
        # Multiple peaks penalty
        multiple_peaks_score = multiple_peak_penalty if (current_comp and current_comp in multiple_peaks_set) else 0
        
        # Total instability score
        total_score = intensity_score + position_score + multiple_peaks_score
        total_score = min(total_score, max_score)
        stb_scores.append(total_score)
    
    return np.array(stb_scores)


def generate_exploration_candidates(composition_df: pd.DataFrame, n_candidates: int = 20) -> pd.DataFrame:
    """Generate candidate compositions for exploration."""
    materials = composition_df.index.tolist()
    n_materials = len(materials)
    
    candidates = []
    np.random.seed(42)  # For reproducibility
    
    for _ in range(n_candidates):
        # Generate random composition (normalized to sum to 1)
        composition = np.random.dirichlet(np.ones(n_materials))
        
        row = {'well': f'candidate_{len(candidates)+1}'}
        for i, material in enumerate(materials):
            row[f'composition_{material}'] = composition[i]
        
        # Set peak features to zero (unknown for candidates)
        for i in range(3):
            row[f'peak_{i+1}_wavelength'] = 0
            row[f'peak_{i+1}_intensity'] = 0
            row[f'peak_{i+1}_fwhm'] = 0
            row[f'peak_{i+1}_area'] = 0
        
        # Set quality metrics to zero (unknown)
        row['r_squared'] = 0
        row['rmse'] = 0
        row['num_peaks'] = 0
        
        candidates.append(row)
    
    return pd.DataFrame(candidates)


if model_choice == MODEL_SINGLE_GP:
    # ---------------------------------------------------------------------
    # Single-objective GP (scikit-learn) using JSON + composition CSV
    # ---------------------------------------------------------------------
    st.header("Data Input")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Curve Fitting Results")
        curve_fitting_file = st.file_uploader(
            "Upload curve fitting results JSON file:",
            type=["json"],
            help="Upload the JSON file from curve fitting analysis",
            key="single_gp_results_json",
        )

        # Also allow selecting from results directory
        if not curve_fitting_file:
            results_dir = "results"
            if os.path.exists(results_dir):
                json_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_results.json")]
                if json_files:
                    selected_file = st.selectbox(
                        "Or select from results directory:",
                        [None] + json_files,
                        format_func=lambda x: x if x else "Select file...",
                        key="single_gp_results_select",
                    )
                    if selected_file:
                        curve_fitting_file = os.path.join(results_dir, selected_file)

    with col2:
        st.subheader("Composition Data")
        composition_file = st.file_uploader(
            "Upload composition CSV file:",
            type=["csv"],
            help="CSV file with materials as rows and wells as columns",
            key="single_gp_comp_csv",
        )

    # Load data
    results_data = None
    composition_data = None

    if curve_fitting_file:
        try:
            if isinstance(curve_fitting_file, str):
                results_data = load_curve_fitting_results(curve_fitting_file)
            else:
                results_data = json.load(curve_fitting_file)
            st.success(f"✅ Loaded curve fitting results: {len(results_data.get('wells', {}))} wells")
        except Exception as e:
            st.error(f"Error loading curve fitting results: {e}")

    if composition_file:
        try:
            if isinstance(composition_file, str):
                composition_data = load_composition_data(composition_file)
            else:
                composition_data = load_composition_data(composition_file)
            st.success(
                f"✅ Loaded composition data: {len(composition_data.columns)} wells, "
                f"{len(composition_data.index)} materials"
            )
        except Exception as e:
            st.error(f"Error loading composition data: {e}")

    # Extract features
    if results_data and composition_data is not None:
        st.header("Feature Extraction")

        try:
            X_df, y_series = extract_features_from_results(results_data, composition_data)

            st.subheader("Extracted Features")
            st.dataframe(X_df.head(), use_container_width=True)

            st.subheader("Target Variable")
            st.write(f"Target: {y_series.name}")
            st.write(f"Mean: {y_series.mean():.4f}, Std: {y_series.std():.4f}")

            # Model configuration
            st.header("Model Configuration")

            col1, col2, col3 = st.columns(3)
            with col1:
                kernel_type = st.selectbox("Kernel Type", ["RBF", "Matern", "RBF+Matern"], index=0)
            with col2:
                alpha = st.number_input(
                    "Alpha (noise level)", min_value=1e-10, max_value=1.0, value=1e-6, format="%.2e"
                )
            with col3:
                target_col = st.selectbox("Target Variable", ["r_squared", "rmse", "num_peaks"], index=0)

            # Update target if changed
            if target_col == "r_squared":
                y_series = X_df["r_squared"]
            elif target_col == "rmse":
                y_series = X_df["rmse"]
            elif target_col == "num_peaks":
                y_series = X_df["num_peaks"]

            # Prepare feature matrix (exclude well and target columns)
            feature_cols = [col for col in X_df.columns if col not in ["well", "r_squared", "rmse", "num_peaks"]]
            X = X_df[feature_cols].values
            y = y_series.values

            if st.button("🚀 Train GP Model", type="primary", use_container_width=True, key="single_gp_train"):
                with st.spinner("Training Gaussian Process model..."):
                    # Train model
                    gp_model = GaussianProcessModel(kernel_type=kernel_type, alpha=alpha)
                    gp_model.fit(X, y, feature_cols, target_col)

                    # Cross-validation
                    X_scaled = gp_model.scaler_X.transform(X)
                    y_scaled = gp_model.scaler_y.transform(y.reshape(-1, 1)).ravel()
                    cv_scores = cross_val_score(gp_model.gp, X_scaled, y_scaled, cv=5, scoring="r2")

                    st.success("✅ Model trained successfully!")

                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Training Samples", len(X))
                    with col2:
                        st.metric("Features", len(feature_cols))
                    with col3:
                        st.metric("CV R² Score", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

                    # Predictions on training data
                    y_pred, y_std = gp_model.predict(X, return_std=True)

                    # Visualization
                    st.header("Model Performance")

                    fig = make_subplots(
                        rows=1,
                        cols=2,
                        subplot_titles=("Predictions vs Actual", "Residuals"),
                        horizontal_spacing=0.15,
                    )

                    # Predictions plot
                    fig.add_trace(
                        go.Scatter(
                            x=y,
                            y=y_pred,
                            mode="markers",
                            name="Predictions",
                            error_y=dict(type="data", array=y_std, visible=True),
                            marker=dict(color="blue", size=8),
                        ),
                        row=1,
                        col=1,
                    )

                    # Perfect prediction line
                    min_val = min(y.min(), y_pred.min())
                    max_val = max(y.max(), y_pred.max())
                    fig.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode="lines",
                            name="Perfect",
                            line=dict(color="red", dash="dash"),
                        ),
                        row=1,
                        col=1,
                    )

                    # Residuals plot
                    residuals = y - y_pred
                    fig.add_trace(
                        go.Scatter(
                            x=y_pred,
                            y=residuals,
                            mode="markers",
                            name="Residuals",
                            marker=dict(color="green", size=8),
                        ),
                        row=1,
                        col=2,
                    )

                    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

                    fig.update_xaxes(title_text="Actual", row=1, col=1)
                    fig.update_yaxes(title_text="Predicted", row=1, col=1)
                    fig.update_xaxes(title_text="Predicted", row=1, col=2)
                    fig.update_yaxes(title_text="Residuals", row=1, col=2)

                    fig.update_layout(height=500, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)

                    # Exploration
                    st.header("Exploration & Next Experiments")

                    col1, col2 = st.columns(2)
                    with col1:
                        n_candidates = st.number_input(
                            "Number of candidates", min_value=10, max_value=100, value=20, key="single_gp_n_cand"
                        )
                    with col2:
                        acquisition_method = st.selectbox(
                            "Acquisition Function", ["UCB", "EI", "PI"], index=0, key="single_gp_acq"
                        )

                    beta = st.slider(
                        "Exploration-Exploitation Trade-off (β)",
                        min_value=0.1,
                        max_value=5.0,
                        value=2.0,
                        step=0.1,
                        key="single_gp_beta",
                    )

                    if st.button(
                        "🔍 Generate Exploration Candidates",
                        use_container_width=True,
                        key="single_gp_generate_candidates",
                    ):
                        with st.spinner("Generating exploration candidates..."):
                            # Generate candidates
                            candidates_df = generate_exploration_candidates(composition_data, n_candidates)

                            # Extract features for candidates
                            X_candidates = candidates_df[feature_cols].values

                            # Predict and compute acquisition
                            y_pred_candidates, y_std_candidates = gp_model.predict(
                                X_candidates, return_std=True
                            )
                            acquisition_values = gp_model.acquisition_function(
                                X_candidates, method=acquisition_method, beta=beta
                            )

                            # Sort by acquisition value
                            top_indices = np.argsort(acquisition_values)[::-1][:10]  # Top 10

                            st.subheader("Top Exploration Candidates")

                            results_list = []
                            for idx in top_indices:
                                candidate = candidates_df.iloc[idx]
                                results_list.append(
                                    {
                                        "Rank": len(results_list) + 1,
                                        "Candidate": candidate["well"],
                                        "Predicted Value": f"{y_pred_candidates[idx]:.4f}",
                                        "Uncertainty": f"{y_std_candidates[idx]:.4f}",
                                        "Acquisition Score": f"{acquisition_values[idx]:.4f}",
                                        **{
                                            col: f"{candidate[col]:.4f}"
                                            for col in feature_cols
                                            if col.startswith("composition_")
                                        },
                                    }
                                )

                            results_df = pd.DataFrame(results_list)
                            st.dataframe(results_df, use_container_width=True)

                            # Visualization of exploration space
                            comp_cols_all = [col for col in feature_cols if col.startswith("composition_")]
                            if len(comp_cols_all) >= 2:
                                comp_cols = comp_cols_all[:2]

                                fig_explore = go.Figure()

                                # Training data
                                fig_explore.add_trace(
                                    go.Scatter(
                                        x=X_df[comp_cols[0]],
                                        y=X_df[comp_cols[1]],
                                        mode="markers",
                                        name="Training Data",
                                        marker=dict(color="blue", size=10, symbol="circle"),
                                    )
                                )

                                # Top candidates
                                top_candidates_data = candidates_df.iloc[top_indices]
                                fig_explore.add_trace(
                                    go.Scatter(
                                        x=top_candidates_data[comp_cols[0]],
                                        y=top_candidates_data[comp_cols[1]],
                                        mode="markers",
                                        name="Top Candidates",
                                        marker=dict(color="red", size=15, symbol="star"),
                                        text=[f"Candidate {i+1}" for i in range(len(top_indices))],
                                        textposition="top center",
                                    )
                                )

                                fig_explore.update_layout(
                                    title="Exploration Space",
                                    xaxis_title=comp_cols[0],
                                    yaxis_title=comp_cols[1],
                                    height=500,
                                )
                                st.plotly_chart(fig_explore, use_container_width=True)

                            # Save results for analysis agent
                            gp_results = {
                                "model_type": "GaussianProcess",
                                "kernel": kernel_type,
                                "target": target_col,
                                "cv_score": float(cv_scores.mean()),
                                "cv_std": float(cv_scores.std()),
                                "top_candidates": results_list[:5],  # Top 5 for analysis
                                "uncertainty_stats": {
                                    "mean": float(y_std_candidates.mean()),
                                    "std": float(y_std_candidates.std()),
                                    "max": float(y_std_candidates.max()),
                                },
                                "acquisition_method": acquisition_method,
                                "beta": float(beta),
                            }

                            st.session_state.gp_results = gp_results
                            st.session_state.analysis_ready = True

                            st.success("✅ Exploration candidates generated! Results saved for Analysis Agent.")
                            
                            # Check if Analysis Agent is next in workflow and marked as automatic
                            workflow_auto_flags = st.session_state.get("workflow_auto_flags", {})
                            manual_workflow = st.session_state.get("manual_workflow", [])
                            workflow_index = st.session_state.get("workflow_index", 0)
                            
                            analysis_auto_from_workflow = (
                                workflow_index < len(manual_workflow)
                                and manual_workflow[workflow_index] == "Analysis Agent"
                                and workflow_auto_flags.get("Analysis Agent", False)
                            )
                            
                            # Auto-route if workflow says so
                            if analysis_auto_from_workflow:
                                st.info("🔄 Auto-routing to Analysis Agent (workflow automation)...")
                                st.session_state.next_agent = "analysis"
                                st.session_state.gp_results_available = True
                                st.rerun()

                            # Button to send to analysis agent
                            if st.button(
                                "📊 Send to Analysis Agent",
                                use_container_width=True,
                                key="single_gp_send_analysis",
                            ):
                                st.session_state.next_agent = "analysis"
                                st.session_state.gp_results_available = True
                                st.rerun()

                    # Save model to session state
                    st.session_state.gp_model = gp_model
                    st.session_state.gp_training_data = {
                        "X": X_df,
                        "y": y_series,
                        "feature_cols": feature_cols,
                    }

        except Exception as e:
            st.error(f"Error processing data: {e}")
            import traceback

            st.exception(e)

    else:
        st.info("👆 Please upload both curve fitting results and composition data to begin.")


# -------------------------------------------------------------------------
# Other ML methods (render only the selected one)
# -------------------------------------------------------------------------

if model_choice == MODEL_MONTE_CARLO_TREE:
    render_monte_carlo_tree_ui()
    st.stop()

# From here down is the Dual PyTorch GP UI; do not show it unless selected.
if model_choice != MODEL_DUAL_TORCH_GP:
    st.stop()


# -------------------------------------------------------------------------
# PyTorch Dual GP from CSV (performance + stability)
# -------------------------------------------------------------------------

st.markdown("---")
st.header("PyTorch Dual Gaussian Process (from Peak CSV)")
st.markdown(
    "This section trains **two PyTorch Gaussian Processes**: one for a performance "
    "metric (e.g., R² or PL intensity) and one for a **stability score**. "
    "An acquisition score combines both to recommend promising, stable conditions."
)

dual_gp_csv_file = st.file_uploader(
    "Upload peak data CSV for dual GP (e.g., exported from curve fitting):",
    type=["csv"],
    help="Use the CSV that contains peak features, quality metrics, and (optionally) a stability column.",
    key="dual_gp_csv",
)

if dual_gp_csv_file is not None:
    try:
        df_dual = pd.read_csv(dual_gp_csv_file)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        df_dual = None

    if df_dual is not None and not df_dual.empty:
        st.subheader("Preview of Dual GP Data")
        st.dataframe(df_dual.head(), use_container_width=True)

        numeric_cols = df_dual.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns (features + 2 targets) for dual GP.")
        else:
            # Check if we can compute instability score from the data
            has_initial_final = (
                ('initial_peak_positions' in df_dual.columns or 'Peak_1_Wavelength' in df_dual.columns) and
                ('initial_peak_intensities' in df_dual.columns or 'Peak_1_Intensity' in df_dual.columns)
            )
            
            # Check for time-series data (multiple reads) that could be used for initial/final
            has_multiple_reads = 'Read' in df_dual.columns and df_dual['Read'].nunique() > 1
            
            # Heuristics for defaults
            default_perf = "R_squared" if "R_squared" in numeric_cols else numeric_cols[0]
            # Try to guess stability column
            stability_candidates = [c for c in numeric_cols if "stability" in c.lower() or "instability" in c.lower()]
            default_stab = stability_candidates[0] if stability_candidates else (
                numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
            )

            st.subheader("Select Targets and Features")
            
            # Option to compute instability score
            compute_instability = False
            if has_initial_final or has_multiple_reads:
                compute_instability = st.checkbox(
                    "Compute instability score from peak data (using your notebook's formula)",
                    value=False,
                    help="If checked, will compute instability score based on initial/final peak positions and intensities, "
                          "matching your Dual_GP_Film_RT_Cs_PEA_BDA_04M_DMF9DMSO1_V2_ZIP.ipynb implementation"
                )
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                perf_col = st.selectbox(
                    "Performance target column",
                    options=numeric_cols,
                    index=numeric_cols.index(default_perf) if default_perf in numeric_cols else 0,
                )
            with col_t2:
                if compute_instability:
                    st.info("📊 Instability score will be computed automatically")
                    stab_col = None  # Will be computed
                else:
                    stab_col = st.selectbox(
                        "Stability target column",
                        options=numeric_cols,
                    index=numeric_cols.index(default_stab) if default_stab in numeric_cols else 0,
                )

            feature_default = [c for c in numeric_cols if c not in [perf_col, stab_col]]
            feature_cols_dual = st.multiselect(
                "Feature columns for GP (used as inputs X)",
                options=numeric_cols,
                default=feature_default,
            )

            if len(feature_cols_dual) == 0:
                st.warning("Please select at least one feature column.")
            else:
                # Configuration for instability score (if computing it)
                instability_params = {}
                if compute_instability:
                    st.subheader("Instability Score Configuration")
                    col_inst1, col_inst2 = st.columns(2)
                    with col_inst1:
                        instability_params['target_wavelength'] = st.number_input(
                            "Target wavelength (nm)",
                            min_value=300.0,
                            max_value=1000.0,
                            value=700.0,
                            step=10.0,
                            help="Target wavelength for peak position evaluation",
                            key="inst_target_wl"
                        )
                        instability_params['wavelength_tolerance'] = st.number_input(
                            "Wavelength tolerance (nm)",
                            min_value=0.0,
                            max_value=100.0,
                            value=10.0,
                            step=1.0,
                            key="inst_wl_tol"
                        )
                    with col_inst2:
                        instability_params['degradation_weight'] = st.number_input(
                            "Degradation weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.4,
                            step=0.1,
                            key="inst_degrad"
                        )
                        instability_params['position_weight'] = st.number_input(
                            "Position weight",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            step=0.1,
                            key="inst_pos"
                        )
                        instability_params['multiple_peak_penalty'] = st.number_input(
                            "Multiple peak penalty",
                            min_value=0.0,
                            max_value=3.0,
                            value=0.5,
                            step=0.1,
                            key="inst_multi"
                        )
                
                X_dual = df_dual[feature_cols_dual].values.astype(np.float32)
                y_perf = df_dual[perf_col].values.astype(np.float32)
                
                # Compute instability score if requested
                if compute_instability:
                    # Detect compositions with multiple peaks
                    compositions_with_multiple_peaks = []
                    if 'Total_Quality_Peaks' in df_dual.columns:
                        # Find compositions with more than 1 peak
                        multi_peak_mask = df_dual['Total_Quality_Peaks'] > 1
                        if 'composition_number' in df_dual.columns and 'iteration' in df_dual.columns:
                            multi_peak_comps = df_dual[multi_peak_mask][['composition_number', 'iteration']].values
                            compositions_with_multiple_peaks = [tuple(row) for row in multi_peak_comps]
                    
                    # Compute instability score
                    y_stab_raw = instability_score(
                        df_dual,
                        compositions_with_multiple_peaks=compositions_with_multiple_peaks if compositions_with_multiple_peaks else None,
                        target_wavelength=instability_params['target_wavelength'],
                        multiple_peak_penalty=instability_params['multiple_peak_penalty'],
                        wavelength_tolerance=instability_params['wavelength_tolerance'],
                        degradation_weight=instability_params['degradation_weight'],
                        position_weight=instability_params['position_weight'],
                    )
                    
                    # Normalize instability score to [0, 1] (matching notebook)
                    y_stab_min = np.min(y_stab_raw)
                    y_stab_max = np.max(y_stab_raw)
                    if y_stab_max > y_stab_min:
                        y_stab = (y_stab_raw - y_stab_min) / (y_stab_max - y_stab_min)
                    else:
                        y_stab = y_stab_raw
                    
                    # Add computed score to dataframe
                    df_dual['computed_instability_score'] = y_stab
                    df_dual['computed_instability_score_raw'] = y_stab_raw
                    stab_col = 'computed_instability_score'
                    st.info(f"📊 Instability scores computed and normalized (raw: [{y_stab_raw.min():.4f}, {y_stab_raw.max():.4f}], normalized: [{y_stab.min():.4f}, {y_stab.max():.4f}])")
                else:
                    y_stab = df_dual[stab_col].values.astype(np.float32)

                st.subheader("Dual GP Configuration")
                col_cfg1, col_cfg2 = st.columns(2)
                with col_cfg1:
                    noise_level = st.number_input(
                        "GP noise level",
                        min_value=1e-8,
                        max_value=1e-1,
                        value=1e-4,
                        format="%.1e",
                    )
                with col_cfg2:
                    lengthscale = st.number_input(
                        "GP lengthscale (RBF)",
                        min_value=1e-3,
                        max_value=10.0,
                        value=1.0,
                        step=0.1,
                    )

                col_acq1, col_acq2, col_acq3 = st.columns(3)
                with col_acq1:
                    beta_dual = st.slider(
                        "Exploration factor β (on performance GP)",
                        min_value=0.0,
                        max_value=5.0,
                        value=2.0,
                        step=0.1,
                    )
                with col_acq2:
                    instability_threshold_percentile = st.slider(
                        "Instability threshold percentile",
                        min_value=0.5,
                        max_value=0.95,
                        value=0.7,
                        step=0.05,
                        help="Regions with predicted instability above this percentile will be zeroed out in acquisition.",
                    )
                with col_acq3:
                    use_multiplicative_adjustment = st.checkbox(
                        "Use multiplicative adjustment (notebook method)",
                        value=True,
                        help="If checked, uses acq * adjust_tune_score (matching notebook). Otherwise uses additive adjustment.",
                    )

                if st.button("🚀 Train Dual PyTorch GP and Score Acquisition", use_container_width=True):
                    with st.spinner("Training dual Gaussian Processes with PyTorch..."):
                        try:
                            # Train performance GP
                            gp_perf = TorchGaussianProcess(
                                lengthscale=lengthscale, variance=1.0, noise=noise_level
                            )
                            gp_perf.fit(X_dual, y_perf)
                            
                            # Train stability GP (for tune_GP-style adjustment)
                            gp_stab = TorchGaussianProcess(
                                lengthscale=lengthscale, variance=1.0, noise=noise_level
                            )
                            gp_stab.fit(X_dual, y_stab)

                            # Predictions on training data
                            mu_perf, std_perf = gp_perf.predict(X_dual, return_std=True)
                            mu_stab, std_stab = gp_stab.predict(X_dual, return_std=True)
                            
                            # Base acquisition (UCB on performance)
                            acq_base = mu_perf + beta_dual * std_perf
                            
                            # tune_GP-style adjustment: predict stability for all points
                            # and create adjust_tune_score where high instability regions are zeroed
                            init_tune_score = mu_stab  # Predicted instability scores
                            threshold_value = np.quantile(init_tune_score, instability_threshold_percentile)
                            
                            # adjust_tune_score: zero out high instability regions
                            adjust_tune_score = np.where(init_tune_score > threshold_value, 0, init_tune_score)
                            
                            # Normalize adjust_tune_score to [0, 1] for multiplicative scaling
                            if np.max(adjust_tune_score) > 0:
                                adjust_tune_score = adjust_tune_score / np.max(adjust_tune_score)
                            
                            # Apply adjustment to acquisition (matching notebook method)
                            if use_multiplicative_adjustment:
                                # Multiplicative: acq_tune = acq * adjust_tune_score
                                acquisition_score = acq_base * adjust_tune_score
                                # Also zero out high instability regions directly
                                acquisition_score = np.where(init_tune_score > threshold_value, 0, acquisition_score)
                            else:
                                # Additive fallback (original method)
                                stability_weight = st.session_state.get('stability_weight_dual', 1.0)
                                acquisition_score = acq_base - stability_weight * mu_stab

                            df_results = df_dual.copy()
                            df_results["mu_perf"] = mu_perf
                            df_results["std_perf"] = std_perf
                            df_results["mu_stab"] = mu_stab
                            df_results["std_stab"] = std_stab
                            df_results["init_tune_score"] = init_tune_score
                            df_results["adjust_tune_score"] = adjust_tune_score
                            df_results["acquisition_score_base"] = acq_base
                            df_results["acquisition_score"] = acquisition_score

                            # Rank experiments
                            df_ranked = df_results.sort_values(
                                by="acquisition_score", ascending=False
                            ).reset_index(drop=True)

                            st.success("✅ Dual GP trained and acquisition scores computed.")

                            st.subheader("Top Recommended Conditions (by acquisition score)")
                            top_n = min(10, len(df_ranked))
                            
                            # Display key columns for top candidates
                            display_cols = ['acquisition_score', 'acquisition_score_base', 'mu_perf', 'std_perf', 
                                          'init_tune_score', 'adjust_tune_score'] + feature_cols_dual[:3]
                            display_cols = [c for c in display_cols if c in df_ranked.columns]
                            st.dataframe(df_ranked[display_cols].head(top_n), use_container_width=True)
                            
                            # Show full dataframe in expander
                            with st.expander("View Full Results DataFrame"):
                                st.dataframe(df_ranked, use_container_width=True)

                            # Simple summary metrics
                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                st.metric("Samples", len(df_ranked))
                            with col_m2:
                                st.metric("Perf target mean", f"{np.mean(y_perf):.4f}")
                            with col_m3:
                                st.metric("Stability target mean", f"{np.mean(y_stab):.4f}")
                            with col_m4:
                                st.metric("Instability threshold", f"{threshold_value:.4f}")

                            # Save compact results for Analysis Agent
                            gp_dual_results = {
                                "model_type": "DualTorchGP",
                                "backend": "torch",
                                "targets": {
                                    "performance": perf_col,
                                    "stability": stab_col if stab_col else "computed_instability_score",
                                },
                                "feature_columns": feature_cols_dual,
                                "hyperparameters": {
                                    "lengthscale": float(lengthscale),
                                    "noise": float(noise_level),
                                    "beta": float(beta_dual),
                                    "instability_threshold_percentile": float(instability_threshold_percentile),
                                    "use_multiplicative_adjustment": use_multiplicative_adjustment,
                                },
                                "summary": {
                                    "performance_mean": float(np.mean(y_perf)),
                                    "stability_mean": float(np.mean(y_stab)),
                                },
                                "top_candidates": df_ranked.head(5).to_dict(orient="records"),
                            }

                            st.session_state.gp_results = gp_dual_results
                            st.session_state.analysis_ready = True
                            
                            # Check if Analysis Agent is next in workflow and marked as automatic
                            workflow_auto_flags = st.session_state.get("workflow_auto_flags", {})
                            manual_workflow = st.session_state.get("manual_workflow", [])
                            workflow_index = st.session_state.get("workflow_index", 0)
                            
                            analysis_auto_from_workflow = (
                                workflow_index < len(manual_workflow)
                                and manual_workflow[workflow_index] == "Analysis Agent"
                                and workflow_auto_flags.get("Analysis Agent", False)
                            )
                            
                            # Auto-route if workflow says so
                            if analysis_auto_from_workflow:
                                st.info("🔄 Auto-routing to Analysis Agent (workflow automation)...")
                                st.session_state.next_agent = "analysis"
                                st.session_state.gp_results_available = True
                                st.rerun()

                            st.info(
                                "GP results saved to session state as `gp_results` for the Analysis Agent."
                            )

                            if st.button(
                                "📊 Send Dual GP Results to Analysis Agent",
                                use_container_width=True,
                            ):
                                st.session_state.next_agent = "analysis"
                                st.session_state.gp_results_available = True
                                st.rerun()

                        except Exception as e:
                            st.error(f"Error training dual GP or computing acquisition: {e}")
                            import traceback

                            st.exception(e)
