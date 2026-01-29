"""
Automated ML model execution after curve fitting completes.
This module handles automatic ML model training based on workflow configuration.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Constants matching ml_models.py
MODEL_SINGLE_GP = "Single-objective GP (scikit-learn)"
MODEL_DUAL_TORCH_GP = "Dual-objective GP (PyTorch)"
MODEL_MONTE_CARLO_TREE = "Monte Carlo Decision Tree (external)"


def find_latest_curve_fitting_results(results_dir: str = "results") -> Optional[Tuple[str, str]]:
    """
    Find the most recent curve fitting results files.
    
    Returns:
        Tuple of (json_path, csv_path) or None if not found
    """
    if not os.path.exists(results_dir):
        return None
    
    # Find most recent JSON results file
    json_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_results.json")]
    if not json_files:
        return None
    
    latest_json = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
    json_path = os.path.join(results_dir, latest_json)
    
    # Find corresponding CSV export
    base_name = latest_json.replace("_peak_fit_results.json", "")
    csv_path = os.path.join(results_dir, f"{base_name}_peak_fit_export.csv")
    
    if not os.path.exists(csv_path):
        # Try alternative naming
        csv_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_export.csv")]
        if csv_files:
            # Find CSV with matching base name
            matching_csv = [f for f in csv_files if base_name in f]
            if matching_csv:
                csv_path = os.path.join(results_dir, matching_csv[0])
            else:
                csv_path = os.path.join(results_dir, csv_files[-1])  # Use most recent
        else:
            return (json_path, None)
    
    return (json_path, csv_path)


def find_composition_csv(results_dir: str = "results", data_dir: str = "data") -> Optional[str]:
    """
    Find composition CSV file.
    Checks results directory first, then data directory.
    """
    # Check results directory
    if os.path.exists(results_dir):
        csv_files = [f for f in os.listdir(results_dir) if "composition" in f.lower() and f.endswith(".csv")]
        if csv_files:
            return os.path.join(results_dir, csv_files[0])
    
    # Check data directory
    if os.path.exists(data_dir):
        csv_files = [f for f in os.listdir(data_dir) if "composition" in f.lower() and f.endswith(".csv")]
        if csv_files:
            return os.path.join(data_dir, csv_files[0])
    
    return None


def run_automated_ml_model(
    model_choice: str,
    json_path: Optional[str] = None,
    csv_path: Optional[str] = None,
    composition_csv: Optional[str] = None,
    auto_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Automatically run the selected ML model on curve fitting results.
    
    Args:
        model_choice: Selected ML model (MODEL_SINGLE_GP or MODEL_DUAL_TORCH_GP)
        json_path: Path to curve fitting JSON results (if None, finds latest)
        csv_path: Path to curve fitting CSV export (if None, finds latest)
        composition_csv: Path to composition CSV (if None, tries to find)
        auto_config: Optional configuration dict for model hyperparameters
    
    Returns:
        Dictionary with model results and recommendations
    """
    # Find files if not provided
    if json_path is None or csv_path is None:
        found_files = find_latest_curve_fitting_results()
        if found_files:
            json_path, csv_path = found_files
        else:
            return {
                "success": False,
                "error": "No curve fitting results found. Please run curve fitting first.",
            }
    
    if composition_csv is None:
        composition_csv = find_composition_csv()
    
    auto_config = auto_config or {}
    
    if model_choice == MODEL_SINGLE_GP:
        return _run_single_gp_automated(json_path, composition_csv, auto_config)
    elif model_choice == MODEL_DUAL_TORCH_GP:
        return _run_dual_gp_automated(csv_path, auto_config)
    elif model_choice == MODEL_MONTE_CARLO_TREE:
        return _run_monte_carlo_tree_automated(auto_config or {})
    else:
        return {
            "success": False,
            "error": f"Unknown model choice: {model_choice}",
        }


def _run_single_gp_automated(
    json_path: str,
    composition_csv: Optional[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run single GP model automatically."""
    try:
        # Import here to avoid circular imports
        import sys
        import importlib.util
        
        # Load ml_models module dynamically
        ml_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pages", "ml_models.py")
        if not os.path.exists(ml_models_path):
            return {
                "success": False,
                "error": "ML models page not found. Please ensure ml_models.py exists.",
            }
        
        spec = importlib.util.spec_from_file_location("ml_models", ml_models_path)
        ml_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_models)
        
        # Load data
        with open(json_path, 'r') as f:
            results_data = json.load(f)
        
        if composition_csv is None:
            return {
                "success": False,
                "error": "Composition CSV required for single GP model. Please provide composition data.",
            }
        
        composition_data = pd.read_csv(composition_csv, index_col=0)
        
        # Extract features
        X_df, y_series = ml_models.extract_features_from_results(results_data, composition_data)
        
        # Use R² as target by default
        target_col = config.get("target", "r_squared")
        if target_col == "r_squared":
            y_series = X_df["r_squared"]
        elif target_col == "rmse":
            y_series = X_df["rmse"]
        elif target_col == "num_peaks":
            y_series = X_df["num_peaks"]
        
        feature_cols = [col for col in X_df.columns if col not in ["well", "r_squared", "rmse", "num_peaks"]]
        X = X_df[feature_cols].values
        y = y_series.values
        
        # Train model
        kernel_type = config.get("kernel_type", "RBF")
        alpha = config.get("alpha", 1e-6)
        
        gp_model = ml_models.GaussianProcessModel(kernel_type=kernel_type, alpha=alpha)
        gp_model.fit(X, y, feature_cols, target_col)
        
        # Generate predictions
        y_pred, y_std = gp_model.predict(X, return_std=True)
        
        # Generate exploration candidates
        n_candidates = config.get("n_candidates", 20)
        candidates_df = ml_models.generate_exploration_candidates(composition_data, n_candidates)
        X_candidates = candidates_df[feature_cols].values
        
        y_pred_candidates, y_std_candidates = gp_model.predict(X_candidates, return_std=True)
        
        beta = config.get("beta", 2.0)
        acquisition_values = gp_model.acquisition_function(X_candidates, method="UCB", beta=beta)
        
        top_indices = np.argsort(acquisition_values)[::-1][:10]
        
        top_candidates = []
        for idx in top_indices:
            candidate = candidates_df.iloc[idx]
            top_candidates.append({
                "rank": len(top_candidates) + 1,
                "candidate": candidate["well"],
                "predicted_value": float(y_pred_candidates[idx]),
                "uncertainty": float(y_std_candidates[idx]),
                "acquisition_score": float(acquisition_values[idx]),
            })
        
        return {
            "success": True,
            "model_type": "SingleGP",
            "top_candidates": top_candidates,
            "predictions": {
                "mean": float(np.mean(y_pred)),
                "std": float(np.std(y_pred)),
            },
            "uncertainty_stats": {
                "mean": float(y_std_candidates.mean()),
                "max": float(y_std_candidates.max()),
            },
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_dual_gp_automated(
    csv_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run dual GP model automatically."""
    try:
        # Import here to avoid circular imports
        import sys
        import importlib.util
        
        # Load ml_models module dynamically
        ml_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pages", "ml_models.py")
        if not os.path.exists(ml_models_path):
            return {
                "success": False,
                "error": "ML models page not found. Please ensure ml_models.py exists.",
            }
        
        spec = importlib.util.spec_from_file_location("ml_models", ml_models_path)
        ml_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_models)
        
        # Load CSV
        df_dual = pd.read_csv(csv_path)
        
        if df_dual.empty:
            return {
                "success": False,
                "error": "CSV file is empty.",
            }
        
        numeric_cols = df_dual.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 3:
            return {
                "success": False,
                "error": "Need at least 3 numeric columns for dual GP.",
            }
        
        # Select targets and features (use defaults or config)
        perf_col = config.get("performance_target", "R_squared" if "R_squared" in numeric_cols else numeric_cols[0])
        compute_instability = config.get("compute_instability", False)
        
        if compute_instability:
            # Compute instability score
            instability_params = config.get("instability_params", {})
            compositions_with_multiple_peaks = []
            
            if 'Total_Quality_Peaks' in df_dual.columns:
                multi_peak_mask = df_dual['Total_Quality_Peaks'] > 1
                if 'composition_number' in df_dual.columns and 'iteration' in df_dual.columns:
                    multi_peak_comps = df_dual[multi_peak_mask][['composition_number', 'iteration']].values
                    compositions_with_multiple_peaks = [tuple(row) for row in multi_peak_comps]
            
            y_stab_raw = ml_models.instability_score(
                df_dual,
                compositions_with_multiple_peaks=compositions_with_multiple_peaks if compositions_with_multiple_peaks else None,
                target_wavelength=instability_params.get('target_wavelength', 700),
                multiple_peak_penalty=instability_params.get('multiple_peak_penalty', 0.5),
                wavelength_tolerance=instability_params.get('wavelength_tolerance', 10),
                degradation_weight=instability_params.get('degradation_weight', 0.4),
                position_weight=instability_params.get('position_weight', 0.6),
            )
            
            # Normalize
            y_stab_min = np.min(y_stab_raw)
            y_stab_max = np.max(y_stab_raw)
            if y_stab_max > y_stab_min:
                y_stab = (y_stab_raw - y_stab_min) / (y_stab_max - y_stab_min)
            else:
                y_stab = y_stab_raw
            
            df_dual['computed_instability_score'] = y_stab
            stab_col = 'computed_instability_score'
        else:
            stab_col = config.get("stability_target", numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
            y_stab = df_dual[stab_col].values.astype(np.float32)
        
        # Select features
        feature_cols = config.get("feature_columns")
        if feature_cols is None:
            feature_cols = [c for c in numeric_cols if c not in [perf_col, stab_col]]
        
        X_dual = df_dual[feature_cols].values.astype(np.float32)
        y_perf = df_dual[perf_col].values.astype(np.float32)
        
        # Model hyperparameters
        lengthscale = config.get("lengthscale", 1.0)
        noise_level = config.get("noise_level", 1e-4)
        beta_dual = config.get("beta", 2.0)
        instability_threshold_percentile = config.get("instability_threshold_percentile", 0.7)
        use_multiplicative = config.get("use_multiplicative_adjustment", True)
        
        # Train GPs
        gp_perf = ml_models.TorchGaussianProcess(lengthscale=lengthscale, variance=1.0, noise=noise_level)
        gp_stab = ml_models.TorchGaussianProcess(lengthscale=lengthscale, variance=1.0, noise=noise_level)
        
        gp_perf.fit(X_dual, y_perf)
        gp_stab.fit(X_dual, y_stab)
        
        # Predictions
        mu_perf, std_perf = gp_perf.predict(X_dual, return_std=True)
        mu_stab, std_stab = gp_stab.predict(X_dual, return_std=True)
        
        # Acquisition
        acq_base = mu_perf + beta_dual * std_perf
        init_tune_score = mu_stab
        threshold_value = np.quantile(init_tune_score, instability_threshold_percentile)
        
        adjust_tune_score = np.where(init_tune_score > threshold_value, 0, init_tune_score)
        if np.max(adjust_tune_score) > 0:
            adjust_tune_score = adjust_tune_score / np.max(adjust_tune_score)
        
        if use_multiplicative:
            acquisition_score = acq_base * adjust_tune_score
            acquisition_score = np.where(init_tune_score > threshold_value, 0, acquisition_score)
        else:
            stability_weight = config.get("stability_weight", 1.0)
            acquisition_score = acq_base - stability_weight * mu_stab
        
        # Rank results
        df_results = df_dual.copy()
        df_results["acquisition_score"] = acquisition_score
        df_ranked = df_results.sort_values(by="acquisition_score", ascending=False).reset_index(drop=True)
        
        top_candidates = df_ranked.head(10).to_dict(orient="records")
        
        return {
            "success": True,
            "model_type": "DualTorchGP",
            "top_candidates": top_candidates[:5],  # Top 5 for analysis
            "predictions": {
                "performance_mean": float(np.mean(mu_perf)),
                "stability_mean": float(np.mean(mu_stab)),
            },
            "uncertainty_stats": {
                "performance_std_mean": float(np.mean(std_perf)),
                "stability_std_mean": float(np.mean(std_stab)),
            },
            "acquisition_stats": {
                "mean": float(np.mean(acquisition_score)),
                "max": float(np.max(acquisition_score)),
            },
        }
    
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def _run_monte_carlo_tree_automated(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the external Monte Carlo Decision Tree project in headless mode.

    This assumes:
      - The repo path points to the folder containing `main.py`
      - The project manages its own data/history under its `data/` folder
    """
    import subprocess
    from pathlib import Path

    repo_path = config.get("repo_path")
    if not repo_path:
        return {
            "success": False,
            "error": "Monte Carlo repo path not provided in auto_config['repo_path']. "
                     "Configure it in the ML Models page.",
        }

    repo = Path(repo_path)
    if not repo.exists():
        return {
            "success": False,
            "error": f"Monte Carlo repo path does not exist: {repo}",
        }

    n_attempts = int(config.get("n_attempts", 500))
    with_agent = bool(config.get("with_agent", False))

    # Build command
    python_exe = os.environ.get("PYTHON_EXECUTABLE", "python")
    cmd = [python_exe, "main.py"]
    env = os.environ.copy()
    env["MC_N_ATTEMPTS"] = str(n_attempts)
    if with_agent:
        cmd.extend(["--with-agent", "--auto-apply"])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=None,
        )

        success = result.returncode == 0
        
        # Parse results from stdout and output files
        monte_carlo_results = {
            "success": success,
            "model_type": "MonteCarloDecisionTree",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
        
        # Try to parse optimization stats from stdout
        import re
        stdout_lines = result.stdout.split('\n')
        stats = {}
        for line in stdout_lines:
            if "Total Cycles:" in line:
                match = re.search(r'Total Cycles:\s*(\d+)', line)
                if match:
                    stats["total_cycles"] = int(match.group(1))
            elif "Best Quality Achieved:" in line:
                match = re.search(r'Best Quality Achieved:\s*([\d.]+)', line)
                if match:
                    stats["best_quality"] = float(match.group(1))
            elif "Total Improvement:" in line:
                match = re.search(r'Total Improvement:\s*([+-]?[\d.]+)%', line)
                if match:
                    stats["total_improvement_pct"] = float(match.group(1))
            elif "Average Improvement/Cycle:" in line:
                match = re.search(r'Average Improvement/Cycle:\s*([+-]?[\d.]+)%', line)
                if match:
                    stats["avg_improvement_per_cycle_pct"] = float(match.group(1))
        
        if stats:
            monte_carlo_results["optimization_stats"] = stats
        
        # Try to parse LLM agent restriction/constraint information from stdout
        # Look for LLM Agent sections in stdout
        llm_agent_results = {}
        in_llm_section = False
        llm_section_lines = []
        
        for i, line in enumerate(stdout_lines):
            if "LLM Agent" in line or "Running LLM Agent" in line:
                in_llm_section = True
                llm_section_lines = []
            elif in_llm_section:
                if line.strip().startswith("=" * 60) or (line.strip() and not line.startswith(" ")):
                    # End of LLM section or new major section
                    if "RECOMMENDATION SUMMARY" in "\n".join(llm_section_lines):
                        break
                llm_section_lines.append(line)
        
        # Try to extract LLM agent information
        if llm_section_lines:
            llm_text = "\n".join(llm_section_lines)
            
            # Extract patterns (parameter restrictions/observations)
            patterns = []
            for line in llm_section_lines:
                if "Found" in line and "patterns" in line.lower():
                    match = re.search(r'Found\s+(\d+)\s+patterns?', line, re.IGNORECASE)
                    if match:
                        llm_agent_results["num_patterns"] = int(match.group(1))
                elif re.match(r'\s*-\s+[A-Za-z_]+:', line):
                    # Pattern line: "  - parameter: observation... (conf: X.XX)"
                    pattern_match = re.match(r'\s*-\s+([A-Za-z_]+):\s*(.+?)\s*\(conf:\s*([\d.]+)\)', line)
                    if pattern_match:
                        patterns.append({
                            "parameter": pattern_match.group(1),
                            "observation": pattern_match.group(2),
                            "confidence": float(pattern_match.group(3))
                        })
            
            if patterns:
                llm_agent_results["patterns"] = patterns
            
            # Extract config suggestions (parameter restrictions/changes)
            suggestions = []
            for line in llm_section_lines:
                if "Generated" in line and "suggestions" in line.lower():
                    match = re.search(r'Generated\s+(\d+)\s+suggestions?', line, re.IGNORECASE)
                    if match:
                        llm_agent_results["num_suggestions"] = int(match.group(1))
                elif "->" in line and "conf:" in line:
                    # Suggestion line: "  - parameter_path: current -> suggested (conf: X.XX)"
                    sugg_match = re.match(r'\s*-\s+([^:]+):\s*([^\s]+)\s*->\s*([^\s]+)\s*\(conf:\s*([\d.]+)\)', line)
                    if sugg_match:
                        suggestions.append({
                            "parameter_path": sugg_match.group(1).strip(),
                            "current_value": sugg_match.group(2),
                            "suggested_value": sugg_match.group(3),
                            "confidence": float(sugg_match.group(4))
                        })
            
            if suggestions:
                llm_agent_results["suggestions"] = suggestions
            
            # Extract recommendation summary
            summary_start = None
            summary_end = None
            for i, line in enumerate(llm_section_lines):
                if "RECOMMENDATION SUMMARY" in line:
                    summary_start = i + 1
                elif summary_start is not None and line.strip().startswith("-" * 60):
                    summary_end = i
                    break
            
            if summary_start is not None:
                summary_lines = llm_section_lines[summary_start:summary_end] if summary_end else llm_section_lines[summary_start:]
                summary_text = "\n".join(summary_lines).strip()
                if summary_text:
                    llm_agent_results["recommendation_summary"] = summary_text
        
        # Also try to load LLM agent results from a JSON file if it exists
        # (in case the Monte Carlo project saves agent results to a file)
        llm_results_file = repo / "data" / "llm_agent_results.json"
        if llm_results_file.exists():
            try:
                import json as json_lib
                with open(llm_results_file, 'r') as f:
                    saved_llm_results = json_lib.load(f)
                    # Merge saved results with parsed results (saved takes precedence)
                    if saved_llm_results:
                        llm_agent_results.update(saved_llm_results)
            except Exception as e:
                monte_carlo_results["llm_agent_parse_error"] = str(e)
        
        if llm_agent_results:
            monte_carlo_results["llm_agent_results"] = llm_agent_results
        
        # Try to load candidates_analysis.csv if it exists
        candidates_csv = repo / "data" / "candidates_analysis.csv"
        if candidates_csv.exists():
            try:
                import pandas as pd
                candidates_df = pd.read_csv(candidates_csv)
                if not candidates_df.empty:
                    # Extract top candidates
                    top_candidates = []
                    # Look for columns that might indicate ranking/score
                    score_cols = [c for c in candidates_df.columns if 'score' in c.lower() or 'rank' in c.lower() or 'quality' in c.lower()]
                    if score_cols:
                        # Sort by score if available
                        sort_col = score_cols[0]
                        top_df = candidates_df.nlargest(10, sort_col) if candidates_df[sort_col].dtype in ['float64', 'int64'] else candidates_df.head(10)
                    else:
                        top_df = candidates_df.head(10)
                    
                    for idx, row in top_df.iterrows():
                        candidate_info = {
                            "rank": idx + 1,
                            "candidate": str(row.get("Material_Composition", f"Candidate {idx + 1}")),
                        }
                        # Add any numeric columns as metrics
                        for col in candidates_df.columns:
                            if candidates_df[col].dtype in ['float64', 'int64']:
                                candidate_info[col] = float(row[col])
                        top_candidates.append(candidate_info)
                    
                    monte_carlo_results["top_candidates"] = top_candidates
                    monte_carlo_results["candidates_file"] = str(candidates_csv)
            except Exception as e:
                # If we can't parse the CSV, continue without it
                monte_carlo_results["candidates_parse_error"] = str(e)
        
        # Try to load optimization_history.csv if it exists
        history_csv = repo / "data" / "optimization_history.csv"
        if history_csv.exists():
            try:
                import pandas as pd
                history_df = pd.read_csv(history_csv)
                if not history_df.empty:
                    monte_carlo_results["optimization_history"] = {
                        "total_experiments": len(history_df),
                        "file_path": str(history_csv),
                    }
                    # Get latest cycle info if available
                    if "Cycle" in history_df.columns:
                        monte_carlo_results["optimization_history"]["latest_cycle"] = int(history_df["Cycle"].max())
            except Exception as e:
                monte_carlo_results["history_parse_error"] = str(e)
        
        return monte_carlo_results
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
