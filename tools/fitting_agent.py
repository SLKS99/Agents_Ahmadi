# fitting_agent.py - Spectropus curve fitting functionality for POLARIS agents

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Import from instruct module
try:
    from tools.instruct import get_prompt
except ModuleNotFoundError:
    # Fallback for when running from within tools directory
    from instruct import get_prompt

# ---------- LLM client (Gemini) ----------

# try:
#     import google.generativeai as genai  # type: ignore
# except ImportError:
#     genai = None


# class LLMClient:
#     """Lightweight wrapper for Gemini text and multimodal calls."""

#     def __init__(self, api_key: Optional[str] = None, model_id: str = "gemini-1.5-flash"):
#         key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
#         if not key:
#             raise ValueError(
#                 "No API key found. Provide api_key or set GOOGLE_API_KEY/GEMINI_API_KEY in your environment."
#             )
#         if genai is None:
#             raise ImportError("google-generativeai not installed. pip install google-generativeai")

#         genai.configure(api_key=key)
#         self.model = genai.GenerativeModel(model_id)

#     def generate(self, prompt: str, max_tokens: int = 1500) -> str:
#         """Text-only prompt. Returns plain text."""
#         try:
#             resp = self.model.generate_content(prompt, generation_config={"max_output_tokens": int(max_tokens)})
#             return getattr(resp, "text", "") or ""
#         except Exception as e:
#             logging.error(f"LLM text generation failed: {e}")
#             raise

#     def generate_multimodal(self, parts: List[Any], max_tokens: int = 1500) -> str:
#         """Multimodal prompt with [text, image, ...] parts."""
#         try:
#             resp = self.model.generate_content(parts, generation_config={"max_output_tokens": int(max_tokens)})
#             return getattr(resp, "text", "") or ""
#         except Exception as e:
#             logging.error(f"LLM multimodal generation failed: {e}")
#             raise

# ---------- LLM client with multiple providers ----------

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None

try:
    from openai import OpenAI  # OpenAI official client
except ImportError:
    OpenAI = None

try:
    import anthropic  # Anthropic client
except ImportError:
    anthropic = None


class LLMClient:
    """Wrapper for multiple LLM providers (Gemini, OpenAI, Anthropic)."""

    def __init__(
        self,
        provider: str = "gemini",
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        min_delay_seconds: Optional[float] = None,  # optional throttle between Gemini calls
    ):
        self.provider = provider.lower()

        if self.provider == "gemini":
            key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError("No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY.")
            if genai is None:
                raise ImportError("google-generativeai not installed. pip install google-generativeai")

            # Store API key and model IDs for reference
            self.api_key = key
            # Use available models - gemini-2.0-flash-lite for both text and image operations
            self.model_id_image = model_id or "gemini-2.5-flash-preview-image"  # For image analysis
            self.model_id_text = "gemini-2.0-flash-lite"  # For text-only calls
            self.model_id = self.model_id_image  # Default to image model
            
            # Store models separately for lazy loading
            self._model_image = None
            self._model_text = None

            # Optional throttle between Gemini calls to reduce RPD spikes
            # Default delay: 0.5 seconds (500ms) to prevent rate limit issues
            env_delay_ms = os.environ.get("GEMINI_MIN_DELAY_MS")
            env_delay_s = os.environ.get("GEMINI_MIN_DELAY_S")
            default_delay = 0.5  # Default 500ms delay between calls
            self.min_delay = (
                float(env_delay_ms) / 1000.0
                if env_delay_ms is not None
                else (float(env_delay_s) if env_delay_s is not None else (min_delay_seconds if min_delay_seconds is not None else default_delay))
            )
            # Track last call time for throttling
            self._last_call_ts = 0.0

            # Ensure API key is in environment (required by some versions of the library)
            os.environ['GOOGLE_API_KEY'] = key
            os.environ['GEMINI_API_KEY'] = key

            # Configure genai with API key (must be done before creating model)
            genai.configure(api_key=key)

            # Create models lazily when needed (don't create here)
            # Models will be created in generate() and generate_multimodal() methods

        elif self.provider == "openai":
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("No OpenAI API key found. Set OPENAI_API_KEY.")
            if OpenAI is None:
                raise ImportError("openai not installed. pip install openai")
            self.client = OpenAI(api_key=key)
            self.model_id = model_id or "gpt-4o-mini"

        elif self.provider == "anthropic":
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("No Anthropic API key found. Set ANTHROPIC_API_KEY.")
            if anthropic is None:
                raise ImportError("anthropic not installed. pip install anthropic")
            self.client = anthropic.Anthropic(api_key=key)
            self.model_id = model_id or "claude-3-haiku-20240307"

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _throttle(self):
        """Simple delay between Gemini calls to keep RPD down."""
        if self.provider != "gemini":
            return
        if self.min_delay and self.min_delay > 0:
            now = time.monotonic()
            elapsed = now - self._last_call_ts
            if elapsed < self.min_delay:
                time.sleep(self.min_delay - elapsed)
            self._last_call_ts = time.monotonic()

    def generate(self, prompt: str, max_tokens: int = 1500) -> str:
        """Text-only generation across providers. Uses text-only model (gemini-2.5-flash-lite)."""
        try:
            if self.provider == "gemini":
                self._throttle()
                # ALWAYS reconfigure genai before each call to ensure API key is set
                # This matches the working test_api_key.py pattern
                if not hasattr(self, 'api_key') or not self.api_key:
                    raise ValueError("API key not found in LLMClient instance")

                # Use text-only model for text generation
                model_id_to_use = self.model_id_text

                # Ensure environment has the key
                os.environ['GOOGLE_API_KEY'] = self.api_key
                os.environ['GEMINI_API_KEY'] = self.api_key

                # Always reconfigure - this is what makes test_api_key.py work
                # This MUST be done before creating the model
                genai.configure(api_key=self.api_key)

                # Use cached text model or create it
                if self._model_text is None:
                    try:
                        self._model_text = genai.GenerativeModel(model_id_to_use)
                    except Exception as model_error:
                        logging.error(f"Failed to create Gemini text model: {model_error}")
                        logging.error(f"Model ID: {model_id_to_use}")
                        logging.error(f"API key present: {bool(self.api_key)}, length: {len(self.api_key) if self.api_key else 0}")
                        raise

                # Make the API call - use text-only model
                resp = self._model_text.generate_content(prompt, generation_config={"max_output_tokens": int(max_tokens)})
                # Check if response has valid candidates with parts
                if resp.candidates and len(resp.candidates) > 0:
                    candidate = resp.candidates[0]
                    # Check finish_reason - 2 means BLOCKED
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                        raise ValueError(f"Gemini response was blocked (finish_reason=2). This may indicate content filtering or safety restrictions.")
                    # Check if candidate has parts
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                        if len(candidate.content.parts) > 0:
                            return candidate.content.parts[0].text
                # Fallback: try the text property, but handle errors gracefully
                try:
                    return resp.text
                except ValueError as e:
                    if "finish_reason" in str(e) or "Part" in str(e):
                        raise ValueError(f"Gemini response has no valid content. finish_reason may indicate blocking or filtering.")
                    raise

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                resp = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text if resp.content else ""

        except Exception as e:
            error_msg = str(e)
            # Enhanced error logging for API key issues
            if "API_KEY_INVALID" in error_msg or "API Key not found" in error_msg:
                logging.error(f"LLM text generation failed ({self.provider}): API key issue")
                logging.error(f"Error details: {error_msg}")
                if hasattr(self, 'api_key'):
                    logging.error(f"API key present in LLMClient: {bool(self.api_key)}")
                    logging.error(f"API key in environment: {bool(os.environ.get('GOOGLE_API_KEY'))}")
                else:
                    logging.error("API key not found in LLMClient instance")
            else:
                logging.error(f"LLM text generation failed ({self.provider}): {e}")
            raise

    def generate_multimodal(self, parts: List[Any], max_tokens: int = 1500) -> str:
        """Multimodal prompt (text+image). Uses image-capable model (gemini-2.5-flash-preview-image)."""
        try:
            if self.provider == "gemini":
                self._throttle()
                # ALWAYS reconfigure genai before each call to ensure API key is set
                # This matches the working test_api_key.py pattern
                if not hasattr(self, 'api_key') or not self.api_key:
                    raise ValueError("API key not found in LLMClient instance")

                # Use image-capable model for multimodal generation
                model_id_to_use = self.model_id_image

                # Ensure environment has the key
                os.environ['GOOGLE_API_KEY'] = self.api_key
                os.environ['GEMINI_API_KEY'] = self.api_key

                # Always reconfigure - this is what makes test_api_key.py work
                # This MUST be done before creating the model
                genai.configure(api_key=self.api_key)

                # Use cached image model or create it
                if self._model_image is None:
                    try:
                        self._model_image = genai.GenerativeModel(model_id_to_use)
                    except Exception as model_error:
                        logging.warning(f"Failed to create Gemini image model {model_id_to_use}: {model_error}")
                        logging.warning(f"Falling back to text-only model {self.model_id_text}")
                        # Fallback: try text model if image model fails
                        if self._model_text is None:
                            self._model_text = genai.GenerativeModel(self.model_id_text)
                        # Extract text part only for fallback
                        text_part = parts[0] if parts else ""
                        resp = self._model_text.generate_content(text_part, generation_config={"max_output_tokens": int(max_tokens)})
                        # Handle response same as below
                        if resp.candidates and len(resp.candidates) > 0:
                            candidate = resp.candidates[0]
                            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                                if len(candidate.content.parts) > 0:
                                    return candidate.content.parts[0].text
                        return resp.text

                # Make the API call - use image-capable model with retry for transient errors
                import time
                max_retries = 3
                retry_delay = 2  # seconds
                
                for attempt in range(max_retries):
                    try:
                        resp = self._model_image.generate_content(parts, generation_config={"max_output_tokens": int(max_tokens)})
                        break  # Success, exit retry loop
                    except Exception as api_error:
                        error_str = str(api_error).lower()
                        
                        # Check for transient errors (500, 503, rate limit)
                        is_transient = any(x in error_str for x in ['500', '503', 'internal', 'overloaded', 'rate limit', 'quota'])
                        
                        if is_transient and attempt < max_retries - 1:
                            logging.warning(f"Transient API error (attempt {attempt + 1}/{max_retries}): {api_error}")
                            logging.warning(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        
                        # If image model API call fails, fallback to text model with text-only input
                        if "image" in error_str or "multimodal" in error_str:
                            logging.warning(f"Image model API call failed: {api_error}")
                            logging.warning(f"Falling back to text-only model {self.model_id_text}")
                            if self._model_text is None:
                                self._model_text = genai.GenerativeModel(self.model_id_text)
                            # Extract text part only for fallback
                            text_part = parts[0] if parts else ""
                            resp = self._model_text.generate_content(text_part, generation_config={"max_output_tokens": int(max_tokens)})
                            break
                        else:
                            raise
                # Check if response has valid candidates with parts
                if resp.candidates and len(resp.candidates) > 0:
                    candidate = resp.candidates[0]
                    # Check finish_reason - 2 means BLOCKED
                    if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 2:
                        raise ValueError(f"Gemini response was blocked (finish_reason=2). This may indicate content filtering or safety restrictions.")
                    # Check if candidate has parts
                    if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                        if len(candidate.content.parts) > 0:
                            return candidate.content.parts[0].text
                # Fallback: try the text property, but handle errors gracefully
                try:
                    return resp.text
                except ValueError as e:
                    if "finish_reason" in str(e) or "Part" in str(e):
                        raise ValueError(f"Gemini response has no valid content. finish_reason may indicate blocking or filtering.")
                    raise

            elif self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": str(parts[0])}] +
                            [{"type": "image_url", "image_url": {"url": p}} for p in parts[1:] if isinstance(p, str)]
                        }
                    ],
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content

            elif self.provider == "anthropic":
                raise NotImplementedError("Anthropic does not yet support multimodal input in this wrapper.")

        except Exception as e:
            logging.error(f"LLM multimodal generation failed ({self.provider}): {e}")
            raise


# ---------- Peak guess dataclasses ----------

@dataclass
class PeakGuess:
    center: float
    height: float
    fwhm: Optional[float] = None
    prominence: Optional[float] = None


@dataclass
class PeakResult:
    peaks: List[PeakGuess]
    baseline: Optional[float] = None


# ---------- Plotting (for vision path) ----------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def save_plot_png(x: np.ndarray, y: np.ndarray, outfile: str, *, title: Optional[str] = None) -> str:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for save_plot_png")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'b-', linewidth=1.5, label='Spectrum')
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel("Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    
    # Add grid for easier peak identification
    ax.grid(True, alpha=0.3)
    
    # Mark the data range
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    ax.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
    
    # Add text annotation with key info for LLM
    info_text = f"X range: {x.min():.0f}-{x.max():.0f} nm\nY range: {y_min:.0f}-{y_max:.0f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    return outfile


def pick_good_peaks(
    out, metrics, x, y, window=(500, 850),
    min_height_snr=5.0,          # peak height ≥ SNR×RMSE
    min_area_frac=0.03,          # area fraction ≥ 3%
    fwhm_bounds=(6, 250),        # FWHM in nm
    center_margin_nm=0.5         # reject if center is pegged at bound within this margin
):
    """
    Returns: list of dicts [{id, center_nm, FWHM_nm, height, amplitude, area, frac}]
             for peaks that pass all tests.
    """
    lo, hi = window
    rmse = float(metrics.get('RMSE', np.nan))

    # parse peak ids present in metrics (p1, p2, p3, ...)
    peak_ids = sorted(
        {k.split('_')[0] for k in metrics if k.startswith('p') and k.endswith('_center')},
        key=lambda s: int(s[1:])
    )

    accepted = []
    for pid in peak_ids:
        c    = metrics.get(f'{pid}_center', np.nan)
        fwhm = metrics.get(f'{pid}_FWHM_est', np.nan)
        hgt  = metrics.get(f'{pid}_height', np.nan)
        frac = metrics.get(f'{pid}_frac', np.nan)

        # reject if parameter is pegged at its fit bounds
        pegged = False
        if out is not None and hasattr(out, "params"):
            pcenter = out.params.get(f'{pid}_center', None)
            if pcenter is not None and (np.isfinite(pcenter.min) and np.isfinite(pcenter.max)):
                if abs(pcenter.value - pcenter.min) <= center_margin_nm or abs(pcenter.value - pcenter.max) <= center_margin_nm:
                    pegged = True

        passes = (
            np.isfinite(c) and (lo <= c <= hi) and
            np.isfinite(fwhm) and (fwhm_bounds[0] <= fwhm <= fwhm_bounds[1]) and
            np.isfinite(rmse) and np.isfinite(hgt) and (hgt >= min_height_snr * rmse) and
            np.isfinite(frac) and (frac >= min_area_frac) and
            (not pegged)
        )
        if passes:
            accepted.append({
                'id': pid,
                'center_nm': float(c),
                'FWHM_nm'  : float(fwhm),
                'height'   : float(hgt),
                'amplitude': float(metrics.get(f'{pid}_amplitude', np.nan)),
                'area'     : float(metrics.get(f'{pid}_area', np.nan)),
                'frac'     : float(frac),
            })
    return accepted


def compute_peak_asymmetry(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Estimate left/right area asymmetry for each peak using partitioned windows.
    We partition the spectrum at midpoints between neighboring peak centers and
    integrate the baseline-subtracted signal on each side of the peak center.
    Returns list aligned with peaks: {left_area, right_area, asymmetry_ratio}.
    """
    if not len(peaks):
        return []
    # Sort peaks by center and keep mapping
    indexed = sorted(enumerate(peaks), key=lambda p: p[1].center)
    centers = [p.center for _, p in indexed]
    x_min, x_max = float(np.min(x)), float(np.max(x))

    # Precompute midpoints between adjacent peaks
    midpoints = []
    for i in range(len(centers) - 1):
        midpoints.append(0.5 * (centers[i] + centers[i + 1]))

    # Baseline-subtracted signal
    y_base = y - (baseline if baseline is not None else 0.0)

    results = [None] * len(peaks)
    for idx_pos, (orig_idx, pk) in enumerate(indexed):
        left_bound = midpoints[idx_pos - 1] if idx_pos > 0 else x_min
        right_bound = midpoints[idx_pos] if idx_pos < len(midpoints) else x_max

        # Mask window for this peak
        win_mask = (x >= left_bound) & (x <= right_bound)
        if not np.any(win_mask):
            results[orig_idx] = {"left_area": 0.0, "right_area": 0.0, "asymmetry_ratio": np.nan}
            continue

        x_win = x[win_mask]
        y_win = y_base[win_mask]

        # Split at peak center
        left_mask = x_win <= pk.center
        right_mask = x_win >= pk.center

        left_area = float(np.trapz(y_win[left_mask], x_win[left_mask])) if np.any(left_mask) else 0.0
        right_area = float(np.trapz(y_win[right_mask], x_win[right_mask])) if np.any(right_mask) else 0.0

        denom = left_area + right_area
        asym_ratio = (right_area - left_area) / denom if denom != 0 else 0.0

        results[orig_idx] = {
            "left_area": left_area,
            "right_area": right_area,
            "asymmetry_ratio": asym_ratio,
        }

    return results


# ---------- lmfit multi-peak fitting with retry ----------

try:
    import lmfit
    from lmfit.models import (
        ConstantModel, GaussianModel, VoigtModel, LorentzianModel,
        PseudoVoigtModel, SkewedGaussianModel, SkewedVoigtModel,
        ExponentialGaussianModel, SplitLorentzianModel
    )  # type: ignore
    _HAS_LMFIT = True
except Exception:
    _HAS_LMFIT = False


def select_peak_model(model_kind: str):
    """Select appropriate lmfit model based on model_kind string."""
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required")

    model_map = {
        'gaussian': GaussianModel,
        'lorentzian': LorentzianModel,
        'voigt': VoigtModel,
        'pseudovoigt': PseudoVoigtModel,
        'skewed_gaussian': SkewedGaussianModel,
        'skewed_voigt': SkewedVoigtModel,
        'exponential_gaussian': ExponentialGaussianModel,
        'split_lorentzian': SplitLorentzianModel
    }

    if model_kind not in model_map:
        raise ValueError(f"Unknown model_kind: {model_kind}. Available: {list(model_map.keys())}")

    return model_map[model_kind]


@dataclass
class PeakFitStats:
    r2: float
    rmse: float
    aic: float
    bic: float
    redchi: float
    nfev: int


@dataclass
class PeakFitResult:
    success: bool
    stats: PeakFitStats
    best_params: Dict[str, float]
    peaks: List[PeakGuess]          # updated centers/heights/FWHM after fit
    baseline: Optional[float]
    report: str
    model_kind: str                 # 'gaussian' or 'voigt'
    lmfit_result: Optional[Any] = None  # Store the actual lmfit result for accurate plotting


def save_fitting_plot_png(x: np.ndarray, y: np.ndarray, fit_result: PeakFitResult, outfile: str, *, title: Optional[str] = None) -> str:
    """Save a comprehensive fitting plot showing original data, fit, individual peaks, and residuals."""
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for save_fitting_plot_png")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top plot: Original data
    ax1.plot(x, y, 'b-', linewidth=2, label='Original data', alpha=0.8)

    # Use the actual lmfit result for the fit
    if hasattr(fit_result, 'lmfit_result') and fit_result.lmfit_result is not None:
        fit_y = fit_result.lmfit_result.best_fit
        
        # Plot individual peak components from lmfit
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
        try:
            components = fit_result.lmfit_result.eval_components(x=x)
            # Find the baseline component name (usually 'c_')
            base_comp_name = next((name for name in components.keys() if name.startswith('c_')), None)
            
            for i, (comp_name, comp_y) in enumerate(components.items()):
                if comp_name != base_comp_name:
                    # Add baseline back to individual peaks so they sit on the baseline
                    baseline_val = components[base_comp_name] if base_comp_name else 0
                    ax1.plot(x, comp_y + baseline_val, '--', color=colors[i % len(colors)],
                            alpha=0.6, linewidth=1, label=f'Peak {comp_name}')
        except:
            pass
    else:
        # Fallback to manual reconstruction
        fit_y = np.full_like(y, fit_result.baseline or 0.0)
        for i, peak in enumerate(fit_result.peaks):
            if peak.fwhm:
                sigma = peak.fwhm / 2.354820045
                peak_y = peak.height * np.exp(-((x - peak.center) / sigma) ** 2 / 2)
                fit_y += peak_y

    # Plot ONLY the one true best fit
    ax1.plot(x, fit_y, 'r-', linewidth=2, label='Total Fit')

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity')
    plot_title = title or f'Peak Fitting Results\nR² = {fit_result.stats.r2:.4f}'
    ax1.set_title(plot_title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Residuals - MUST use the same fit_y as above
    residuals = y - fit_y
    ax2.plot(x, residuals, 'g-', linewidth=1, label='Residuals')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    return outfile


def save_analysis_results(analysis_result: PeakResult, well_name: str, analysis_type: str = "numeric") -> str:
    """Save LLM analysis results to JSON file."""
    analysis_data = {
        "peaks": [{"center": p.center, "height": p.height, "fwhm": p.fwhm, "prominence": p.prominence}
                 for p in analysis_result.peaks],
        "baseline": analysis_result.baseline
    }
    filename = f'llm_analysis_{analysis_type}_{well_name}.json'
    with open(filename, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    return filename


def save_fitting_results(fit_result: PeakFitResult, well_name: str, read_num: int) -> str:
    """Save detailed fitting results to JSON file."""
    results = {
        'well': well_name,
        'read': read_num,
        'fitting_quality': {
            'success': fit_result.success,
            'r_squared': float(fit_result.stats.r2),
            'rmse': float(fit_result.stats.rmse),
            'reduced_chi_squared': float(fit_result.stats.redchi),
            'aic': float(fit_result.stats.aic),
            'bic': float(fit_result.stats.bic),
            'number_of_function_evaluations': int(fit_result.stats.nfev),
            'number_of_peaks': len(fit_result.peaks)
        },
        'peaks': [{"center": p.center, "height": p.height, "fwhm": p.fwhm, "prominence": p.prominence}
                 for p in fit_result.peaks],
        'baseline': float(fit_result.baseline) if fit_result.baseline else None,
        'model_kind': fit_result.model_kind
    }
    filename = f'fitting_results_{well_name}_read{read_num}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    return filename


def assess_fitting_quality(fit_result: PeakFitResult) -> Dict[str, str]:
    """Assess fitting quality and return assessment messages."""
    assessments = {}

    # R² assessment
    if fit_result.stats.r2 > 0.95:
        assessments['r2'] = " Excellent fit (R² > 0.95)"
    elif fit_result.stats.r2 > 0.90:
        assessments['r2'] = " Good fit (R² > 0.90)"
    elif fit_result.stats.r2 > 0.80:
        assessments['r2'] = " Fair fit (R² > 0.80)"
    else:
        assessments['r2'] = " Poor fit (R² < 0.80)"

    # Chi-squared assessment
    if fit_result.stats.redchi < 2.0:
        assessments['chi2'] = "Good chi-squared (reduced χ² < 2.0)"
    elif fit_result.stats.redchi < 5.0:
        assessments['chi2'] = " Acceptable chi-squared (reduced χ² < 5.0)"
    else:
        assessments['chi2'] = "High chi-squared (reduced χ² > 5.0)"

    return assessments


def save_all_wells_results(all_results: List[Dict[str, object]], filename: str = "results/all_wells_analysis.json") -> str:
    """Save all wells analysis results to a single comprehensive JSON file."""
    consolidated_data = {
        "analysis_summary": {
            "total_wells": len(all_results),
            "successful_fits": len([r for r in all_results if r['fit_result'].success]),
            "analysis_date": pd.Timestamp.now().isoformat(),
            "model_kind": all_results[0]['fit_result'].model_kind if all_results else "unknown"
        },
        "wells": {}
    }

    for result in all_results:
        well_name = result['well_name']
        fit_result = result['fit_result']
        x_arr = result['data']['x']
        y_arr = result['data']['y']

        # Extract peak information with all details
        peaks_data = []
        asym_data = compute_peak_asymmetry(x_arr, y_arr, fit_result.peaks, baseline=fit_result.baseline)
        for i, peak in enumerate(fit_result.peaks):
            asym = asym_data[i] if i < len(asym_data) else {"left_area": None, "right_area": None, "asymmetry_ratio": None}
            peak_data = {
                'peak_number': i + 1,
                'position_nm': float(peak.center),
                'intensity': float(peak.height),
                'fwhm_nm': float(peak.fwhm) if peak.fwhm else None,
                'prominence': float(peak.prominence) if peak.prominence else None,
                'left_area': asym.get("left_area"),
                'right_area': asym.get("right_area"),
                'asymmetry_ratio': asym.get("asymmetry_ratio"),
            }
            peaks_data.append(peak_data)

        # Create comprehensive metrics for pick_good_peaks
        metrics = fit_result.best_params.copy()
        metrics['RMSE'] = fit_result.stats.rmse

        # Add additional metrics that pick_good_peaks expects
        total_area = sum(peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 for peak in fit_result.peaks if peak.fwhm)
        for i, peak in enumerate(fit_result.peaks):
            prefix = f"p{i+1}"  # Use p1, p2, p3 format
            metrics[f'{prefix}_center'] = peak.center
            metrics[f'{prefix}_FWHM_est'] = peak.fwhm if peak.fwhm else np.nan
            metrics[f'{prefix}_height'] = peak.height
            peak_area = peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 if peak.fwhm else 0
            metrics[f'{prefix}_amplitude'] = peak_area
            metrics[f'{prefix}_area'] = peak_area
            metrics[f'{prefix}_frac'] = peak_area / total_area if total_area > 0 else 0

        # Use pick_good_peaks to filter quality peaks
        # Use actual data wavelength range instead of hardcoded values
        x_data = result['data']['x']
        wavelength_window = (float(x_data.min()), float(x_data.max()))
        good_peaks = pick_good_peaks(
            fit_result, metrics, x_data, result['data']['y'],
            window=wavelength_window, min_height_snr=3.0, min_area_frac=0.02  # More lenient thresholds
        )

        consolidated_data["wells"][well_name] = {
            "read": result['read'],
            "data_info": {
                "wavelength_range": [float(result['data']['x'].min()), float(result['data']['x'].max())],
                "intensity_range": [float(result['data']['y'].min()), float(result['data']['y'].max())],
                "data_points": len(result['data']['x'])
            },
            "llm_analysis": {
                "numeric_peaks": len(result['llm_numeric_result'].peaks),
                "image_peaks": len(result['llm_image_result'].peaks),
                "numeric_baseline": result['llm_numeric_result'].baseline,
                "image_baseline": result['llm_image_result'].baseline
            },
            "fitting_results": {
                "success": fit_result.success,
                "model_kind": fit_result.model_kind,
                "quality_metrics": {
                    "r_squared": float(fit_result.stats.r2),
                    "rmse": float(fit_result.stats.rmse),
                    "reduced_chi_squared": float(fit_result.stats.redchi),
                    "aic": float(fit_result.stats.aic),
                    "bic": float(fit_result.stats.bic),
                    "function_evaluations": int(fit_result.stats.nfev)
                },
                "baseline": float(fit_result.baseline) if fit_result.baseline else None,
                "total_peaks_found": len(peaks_data),
                "quality_peaks": good_peaks,
                "all_peaks": peaks_data
            },
            "quality_assessment": result['quality_assessment'],
            "files": result['files']
        }

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return filename


def export_peak_data_to_csv(all_results: List[Dict[str, object]], filename: str = "results/peak_data_export.csv") -> str:
    """Export peak data from quality peaks to a structured CSV file."""

    # Prepare data for DataFrame
    csv_data = []

    for result in all_results:
        well_name = result['well_name']
        read = result['read']
        composition = well_name  # Use well name as composition

        # Get quality peaks and metrics directly from the result
        # (These were already processed by save_all_wells_results)
        fit_result = result['fit_result']
        x_arr = result['data']['x']
        y_arr = result['data']['y']
        asym_data = compute_peak_asymmetry(x_arr, y_arr, fit_result.peaks, baseline=fit_result.baseline)

        # Check if this result has quality_peaks already processed
        if 'quality_peaks' in result:
            # Use pre-processed quality peaks
            quality_peaks = result['quality_peaks']
        else:
            # Fallback: create comprehensive metrics and use pick_good_peaks
            metrics = fit_result.best_params.copy()
            metrics['RMSE'] = fit_result.stats.rmse

            total_area = sum(peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 for peak in fit_result.peaks if peak.fwhm)
            for i, peak in enumerate(fit_result.peaks):
                prefix = f"p{i+1}"
                metrics[f'{prefix}_center'] = peak.center
                metrics[f'{prefix}_FWHM_est'] = peak.fwhm if peak.fwhm else np.nan
                metrics[f'{prefix}_height'] = peak.height
                peak_area = peak.height * peak.fwhm * np.sqrt(2 * np.pi) / 2.354820045 if peak.fwhm else 0
                metrics[f'{prefix}_amplitude'] = peak_area
                metrics[f'{prefix}_area'] = peak_area
                metrics[f'{prefix}_frac'] = peak_area / total_area if total_area > 0 else 0

            # Use pick_good_peaks to filter quality peaks
            # Use actual data wavelength range instead of hardcoded values
            x_data = result['data']['x']
            wavelength_window = (float(x_data.min()), float(x_data.max()))
            quality_peaks = pick_good_peaks(
                fit_result, metrics, x_data, result['data']['y'],
                window=wavelength_window, min_height_snr=3.0, min_area_frac=0.02
            )

        # Create row data
        row_data = {
            'Read': read,
            'Composition': composition,
            'Well': well_name,
            'R_squared': float(fit_result.stats.r2),
            'Total_Quality_Peaks': len(quality_peaks)
        }

        # Add peak data (up to 5 peaks)
        for i in range(5):  # Support up to 5 peaks
            if i < len(quality_peaks):
                peak = quality_peaks[i]
                row_data[f'Peak_{i+1}_Wavelength'] = peak['center_nm']
                row_data[f'Peak_{i+1}_Intensity'] = peak['height']
                row_data[f'Peak_{i+1}_FWHM'] = peak['FWHM_nm']
                row_data[f'Peak_{i+1}_Area'] = peak['area']
                # Map asymmetry from fitted peaks if available
                if i < len(asym_data):
                    row_data[f'Peak_{i+1}_LeftArea'] = asym_data[i].get("left_area")
                    row_data[f'Peak_{i+1}_RightArea'] = asym_data[i].get("right_area")
                    row_data[f'Peak_{i+1}_Asymmetry'] = asym_data[i].get("asymmetry_ratio")
            else:
                # Fill empty peaks with NaN
                row_data[f'Peak_{i+1}_Wavelength'] = np.nan
                row_data[f'Peak_{i+1}_Intensity'] = np.nan
                row_data[f'Peak_{i+1}_FWHM'] = np.nan
                row_data[f'Peak_{i+1}_Area'] = np.nan
                row_data[f'Peak_{i+1}_LeftArea'] = np.nan
                row_data[f'Peak_{i+1}_RightArea'] = np.nan
                row_data[f'Peak_{i+1}_Asymmetry'] = np.nan

        csv_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(csv_data)

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save to CSV
    df.to_csv(filename, index=False)

    return filename


def export_peak_data_from_json(json_filename: str, csv_filename: str = "results/peak_data_export.csv",
                              composition_csv: str = None) -> str:
    """Export peak data directly from consolidated JSON file to CSV with composition data."""

    # Load JSON data
    with open(json_filename, 'r') as f:
        data = json.load(f)

    # Load composition data if provided
    composition_data = None
    if composition_csv and os.path.exists(composition_csv):
        try:
            composition_data = pd.read_csv(composition_csv, index_col=0)
            print(f"Loaded composition data with columns: {list(composition_data.columns)}")
        except Exception as e:
            print(f"Warning: Could not load composition data: {e}")

    # Prepare data for DataFrame
    csv_data = []

    for well_name, well_data in data['wells'].items():
        read = well_data['read']

        # Get quality peaks directly from JSON
        quality_peaks = well_data['fitting_results']['quality_peaks']
        r_squared = well_data['fitting_results']['quality_metrics']['r_squared']

        # Create row data
        row_data = {
            'Read': read,
            'Well': well_name,
            'R_squared': float(r_squared),
            'Total_Quality_Peaks': len(quality_peaks)
        }

        # Add composition data if available
        if composition_data is not None and well_name in composition_data.columns:
            # Add each material composition as separate columns
            for material in composition_data.index:
                row_data[f'{material}'] = composition_data.loc[material, well_name]
        else:
            # Fallback: use well name as composition
            row_data['Composition'] = well_name

        # Add peak data (up to 5 peaks)
        for i in range(5):  # Support up to 5 peaks
            if i < len(quality_peaks):
                peak = quality_peaks[i]
                row_data[f'Peak_{i+1}_Wavelength'] = peak['center_nm']
                row_data[f'Peak_{i+1}_Intensity'] = peak['height']
                row_data[f'Peak_{i+1}_FWHM'] = peak['FWHM_nm']
                row_data[f'Peak_{i+1}_Area'] = peak['area']
            else:
                # Fill empty peaks with NaN
                row_data[f'Peak_{i+1}_Wavelength'] = np.nan
                row_data[f'Peak_{i+1}_Intensity'] = np.nan
                row_data[f'Peak_{i+1}_FWHM'] = np.nan
                row_data[f'Peak_{i+1}_Area'] = np.nan

        csv_data.append(row_data)

    # Create DataFrame
    df = pd.DataFrame(csv_data)

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    # Save to CSV
    df.to_csv(csv_filename, index=False)

    return csv_filename


def assess_fit_quality_with_llm(llm: LLMClient, fit_image_path: str, r2_value: float, well_name: str) -> bool:
    """Let LLM assess fit quality from the fitting plot image."""
    fit_assessment_prompt = f"""
    Analyze this peak fitting plot for well {well_name}. The plot shows:
    - Blue line: Original spectrum data
    - Red dashed line: Total fit
    - Colored dashed lines: Individual peak components
    - Bottom panel: Residuals (difference between data and fit)

    Current R² = {r2_value:.4f}

    Assess the fitting quality by examining:
    1. How well the red fit line matches the blue data
    2. Whether individual peaks capture the actual peak shapes
    3. If residuals are randomly distributed around zero
    4. Any systematic deviations or poor peak fits

    Return ONLY "good" if the fit is acceptable, or "poor" if it needs improvement.
    Focus on visual fit quality, not just the R² value.
    """

    try:
        # Load image for multimodal analysis
        from PIL import Image
        image = Image.open(fit_image_path)

        # Create multimodal parts list
        parts = [fit_assessment_prompt, image]
        response = llm.generate_multimodal(parts, max_tokens=50)
        assessment = response.strip().lower()
        # If LLM doesn't say "good" but R² is solid, fall back to numeric threshold
        if "good" in assessment:
            return True
        return r2_value > 0.90
    except Exception as e:
        print(f"  Error in LLM fit assessment: {e}, using R² threshold")
        return r2_value > 0.90  # Fallback to R² threshold


def llm_refine_peaks_from_residuals(llm: LLMClient, x: np.ndarray, y: np.ndarray,
                                   current_peaks: List[PeakGuess], residuals: np.ndarray,
                                   well_name: str, max_peaks: int = 3) -> PeakResult:
    """Let LLM refine peak positions based on fit residuals."""

    # Format current peaks as JSON-like structure
    current_peaks_json = [
        {"center": p.center, "height": p.height, "fwhm": p.fwhm if p.fwhm else 20.0}
        for p in current_peaks
    ]
    
    # Downsample residuals to include in the prompt so LLM can "see" them
    res_x, res_y = _downsample_xy(x, residuals, max_points=100)
    residuals_data = [{"x": float(rx), "res": float(ry)} for rx, ry in zip(res_x, res_y)]

    refinement_prompt = f"""You are a spectroscopy peak fitting assistant. Analyze the fit residuals and suggest refined peak parameters.

IMPORTANT: Return ONLY a valid JSON object. No explanations, no prose, no markdown. Just JSON.

Current fit for Well {well_name}:
- Wavelength range: {x.min():.1f} - {x.max():.1f} nm
- Max peaks allowed: {max_peaks}
- Current peak parameters: {json.dumps(current_peaks_json)}

Residual Statistics:
- Max positive residual: {residuals.max():.1f} (If high, the fit is too LOW at this position)
- Max negative residual: {residuals.min():.1f} (If very negative, the fit is too HIGH at this position)
- Residual std: {residuals.std():.1f}

Residual Data (Downsampled):
{json.dumps(residuals_data)}

TASK:
1. Look at the Residual Data. 
2. If you see large positive residuals (res > 10% of peak height), increase the 'height' of the peak at that 'x' position.
3. If you see a systematic "hump" in residuals where there is no peak, add a new peak.
4. Suggest refined centers, heights, and widths.

Output refined peak parameters in this EXACT JSON format:
{{"peaks": [{{"center": 780.0, "height": 1400.0, "fwhm": 50.0}}], "baseline": 0.0}}

Return ONLY the JSON object, nothing else."""

    try:
        response = llm.generate(refinement_prompt, max_tokens=500)
        
        # Try to extract JSON from response
        try:
            obj = _extract_json(response)
        except ValueError:
            # If no JSON found, the LLM returned prose - use original peaks
            print(f"  LLM returned prose instead of JSON, using original peaks")
            return PeakResult(peaks=current_peaks, baseline=None)

        # Validate that we got actual peak data
        peaks_data = obj.get("peaks", [])
        if not peaks_data or not isinstance(peaks_data, list):
            print(f"  LLM response missing peaks array, using original peaks")
            return PeakResult(peaks=current_peaks, baseline=None)

        refined_peaks = []
        for p in peaks_data:
            # Validate each peak has required fields
            if not isinstance(p, dict) or "center" not in p or "height" not in p:
                continue
            try:
                refined_peaks.append(PeakGuess(
                    center=float(p.get("center")),
                    height=float(p.get("height")),
                    fwhm=(float(p["fwhm"]) if p.get("fwhm") is not None else None),
                    prominence=(float(p["prominence"]) if p.get("prominence") is not None else None),
                ))
            except (ValueError, TypeError):
                # Skip invalid peaks
                continue

        if not refined_peaks:
            print(f"  No valid peaks in LLM response, using original peaks")
            return PeakResult(peaks=current_peaks, baseline=None)

        base = obj.get("baseline")
        return PeakResult(peaks=refined_peaks, baseline=(float(base) if base is not None else None))

    except Exception as e:
        print(f"  LLM refinement failed: {e}, using original peaks")
        return PeakResult(peaks=current_peaks, baseline=None)


def select_model_for_spectrum(llm: LLMClient, x: np.ndarray, y: np.ndarray, well_name: str) -> str:
    """Let LLM select the appropriate model type for the spectrum."""
    # Validate inputs
    if len(x) == 0 or len(y) == 0:
        raise ValueError(f"Empty arrays for well {well_name}: x has {len(x)} points, y has {len(y)} points")
    
    if len(x) != len(y):
        raise ValueError(f"Array length mismatch for well {well_name}: x has {len(x)} points, y has {len(y)} points")
    
    # Create a simple spectrum summary for LLM
    spectrum_summary = f"""
    Spectrum for Well {well_name}:
    - Wavelength range: {x.min():.1f} - {x.max():.1f} nm
    - Intensity range: {y.min():.1f} - {y.max():.1f}
    - Number of peaks visible: {len(np.where(y > y.max()*0.3)[0])}
    - Peak shape: {"Sharp" if np.std(y) > y.mean() else "Broad"}
    """

    model_prompt = f"""
    Based on this spectrum data, select the most appropriate lmfit model type:
    {spectrum_summary}

    Available models:
    - gaussian: For symmetric, bell-shaped peaks
    - lorentzian: For broader, more rounded peaks
    - voigt: For peaks with both Gaussian and Lorentzian character
    - pseudovoigt: Similar to Voigt but computationally faster
    - skewed_gaussian: For asymmetric peaks
    - skewed_voigt: For asymmetric peaks with mixed character

    Return ONLY the model name (e.g., "gaussian" or "voigt").
    """

    try:
        response = llm.generate(model_prompt, max_tokens=50)
        # Extract model name from response
        model_name = response.strip().lower()

        # Validate model name
        valid_models = ['gaussian', 'lorentzian', 'voigt', 'pseudovoigt', 'skewed_gaussian', 'skewed_voigt']
        if model_name in valid_models:
            return model_name
        else:
            print(f"LLM returned invalid model '{model_name}', defaulting to gaussian")
            return "gaussian"
    except Exception as e:
        print(f"Error in model selection: {e}, defaulting to gaussian")
        return "gaussian"


def run_complete_analysis(
    config: 'CurveFittingConfig',
    well_name: str,
    llm: LLMClient,
    reads: Union[int, List[int], str] = "auto",  # int, list[int], "auto", "all", or str like "2" / "1,3-5"
    max_peaks: int = 4,
    model_kind: Optional[str] = None,
    r2_target: float = 0.90,
    max_attempts: int = 3,
    save_plots: bool = True  # Set to False to skip saving final PNG files (LLM analysis still runs)
) -> Union[Dict[str, object], List[Dict[str, object]]]:
    """Run complete analysis workflow for a single well with flexible read selection."""

    # Determine which reads to analyze
    if isinstance(reads, str):
        if reads.lower() == "auto":
            curated = curate_dataset(config)
            available_reads = curated["reads"]
            target_reads = [available_reads[0]] if available_reads else [1]
        elif reads.lower() == "all":
            curated = curate_dataset(config)
            target_reads = curated["reads"]
        else:
            # Parse comma/range string like "2" or "1,3-5"
            target_reads = CurveFittingConfig._parse_int_list(reads)
    elif isinstance(reads, int):
        target_reads = [reads]
    elif isinstance(reads, list):
        target_reads = list(map(int, reads))
    else:
        raise ValueError(f"Invalid reads parameter: {reads}. Use int, list, 'auto', 'all', or comma/range string.")

    print(f"  Analyzing reads: {target_reads}")

    # Set up shared resources
    os.makedirs("analysis_output", exist_ok=True)
    if save_plots:
        os.makedirs("results", exist_ok=True)

    # LLM model selection (once per well, not per read)
    if model_kind is None:
        print(f"  Selecting model type for {well_name}...")
        # Use first read for model selection
        x_sample, y_sample = get_xy_for_well(config, well_name, read=target_reads[0])
        model_kind = select_model_for_spectrum(llm, x_sample, y_sample, well_name)
        print(f"  LLM selected model: {model_kind}")

    # Analyze all target reads efficiently
    all_read_results = []

    for read in target_reads:
        print(f"    Processing read {read}...")
        x, y = get_xy_for_well(config, well_name, read=read)

        # LLM numeric analysis (backup)
        sys_prompt_numeric = get_prompt("numeric")
        llm_result = llm_guess_peaks(llm, x, y, use_image=False, system_prompt=sys_prompt_numeric, max_peaks=max_peaks)

        # LLM image analysis (PRIMARY - visual analysis is more reliable)
        temp_spectrum_path = f'analysis_output/temp_spectrum_{well_name}_read{read}.png'
        save_plot_png(x, y, temp_spectrum_path, title=f'PL Spectrum - Well {well_name} Read {read}')

        sys_prompt_image = get_prompt("image")
        llm_image_result = llm_guess_peaks_from_image(llm, temp_spectrum_path, system_prompt=sys_prompt_image, max_peaks=max_peaks)

        # Clean up temp spectrum immediately
        try:
            os.remove(temp_spectrum_path)
        except:
            pass

        # Use IMAGE-based result for fitting (more reliable than numeric)
        # Fall back to numeric if image result has no peaks
        peaks_to_use = llm_image_result if llm_image_result.peaks else llm_result
        print(f"      Using {'image' if llm_image_result.peaks else 'numeric'}-based peaks: {len(peaks_to_use.peaks)} peaks")

        # lmfit fitting with retry logic
        fit_result = fit_peaks_lmfit_with_retry(
            x, y, peaks_to_use,
            model_kind=model_kind,
            r2_target=0.92,
            max_attempts=max_attempts
        )

        # LLM visual assessment (create temp plot, assess, delete)
        temp_plot_path = f'analysis_output/temp_fit_{well_name}_read{read}.png'
        save_fitting_plot_png(x, y, fit_result, temp_plot_path,
                              title=f'Initial Fit - Well {well_name} Read {read}')

        llm_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, fit_result.stats.r2, well_name)

        # Retry logic with alternative models
        retry_count = 0
        max_retries = max_attempts
        alternative_models = ['gaussian', 'voigt', 'lorentzian', 'pseudovoigt', 'skewed_gaussian']
        if model_kind in alternative_models:
            alternative_models.remove(model_kind)

        best_fit = fit_result

        while (fit_result.stats.r2 < r2_target or not llm_assessment) and retry_count < max_retries:
            retry_count += 1
            print(f"      Attempt {retry_count}: R²={fit_result.stats.r2:.3f}, LLM={'good' if llm_assessment else 'poor'}")

            alt_model = alternative_models[retry_count % len(alternative_models)]

            # LLM refinement based on residuals
            if fit_result.lmfit_result is not None:
                current_residuals = y - fit_result.lmfit_result.best_fit
                
                # Manual boost if residuals are huge (Residual peak > 20% of data peak)
                y_max = np.nanmax(y)
                res_max = np.nanmax(current_residuals)
                
                # Copy peaks to avoid modifying previous results
                boosted_peaks = [
                    PeakGuess(center=p.center, height=p.height, fwhm=p.fwhm, prominence=p.prominence)
                    for p in fit_result.peaks
                ]
                
                if res_max > 0.15 * y_max:
                    print(f"      Large residuals detected ({res_max:.1f}). Attempting manual peak boost.")
                    for pk in boosted_peaks:
                        # Find residual at peak center
                        c_idx = np.argmin(np.abs(x - pk.center))
                        res_at_center = current_residuals[c_idx]
                        if res_at_center > 0.05 * y_max:
                            # Boost height by the residual amount to force convergence
                            pk.height = (pk.height or 0) + res_at_center
                
                # Use BOOSTED peaks for LLM refinement input
                refined_peaks = llm_refine_peaks_from_residuals(
                    llm, x, y, boosted_peaks, current_residuals, well_name, max_peaks
                )
                print(f"      LLM refined peaks: {len(refined_peaks.peaks)} peaks")
            else:
                refined_peaks = llm_result

            try:
                alt_result = fit_peaks_lmfit_with_retry(
                    x, y, refined_peaks,
                    model_kind=alt_model,
                    r2_target=0.92,
                    max_attempts=max_attempts
                )

                if alt_result and alt_result.stats.r2 > best_fit.stats.r2:
                    save_fitting_plot_png(x, y, alt_result, temp_plot_path,
                                         title=f'Alt Fit ({alt_model}) - {well_name} Read {read}')
                    alt_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, alt_result.stats.r2, well_name)

                    if alt_result.stats.r2 > r2_target and alt_assessment:
                        print(f"      Success with {alt_model}: R²={alt_result.stats.r2:.3f}")
                        best_fit = alt_result
                        break
                    elif alt_result.stats.r2 > best_fit.stats.r2:
                        print(f"      Improved with {alt_model}: R²={alt_result.stats.r2:.3f}")
                        best_fit = alt_result

            except Exception as e:
                print(f"      Alternative model {alt_model} failed: {e}")
                continue

            fit_result = best_fit
            llm_assessment = assess_fit_quality_with_llm(llm, temp_plot_path, fit_result.stats.r2, well_name)

        # Clean up temp plot
        try:
            os.remove(temp_plot_path)
        except:
            pass

        fit_result = best_fit

        # Save final plot (only if requested)
        if save_plots:
            if len(target_reads) > 1:
                # Multiple reads: include read number in filename
                fitting_plot_file = save_fitting_plot_png(x, y, fit_result, f'results/fit_results_{well_name}_read{read}.png',
                                                         title=f'Peak Fitting - {well_name} Read {read}')
            else:
                # Single read: use simple filename
                fitting_plot_file = save_fitting_plot_png(x, y, fit_result, f'results/fit_results_{well_name}.png',
                                                         title=f'Peak Fitting - {well_name}')
            files_dict = {'fitting_plot': fitting_plot_file}
        else:
            files_dict = {}

        # Quality assessment
        quality_assessment = assess_fitting_quality(fit_result)

        # Store result
        result = {
            'well_name': well_name,
            'read': read,
            'data': {'x': x, 'y': y},
            'llm_numeric_result': llm_result,
            'llm_image_result': llm_image_result,
            'fit_result': fit_result,
            'files': files_dict,
            'quality_assessment': quality_assessment
        }
        all_read_results.append(result)

    # Return single result for backward compatibility, or list for multiple reads
    if len(all_read_results) == 1:
        return all_read_results[0]
    else:
        return all_read_results


# ---------- LLM peak guessing helpers ----------

def _downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return x[idx], y[idx]


def _extract_json(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return json.loads(s)
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        return json.loads(s[a : b + 1])
    raise ValueError("No JSON object found in LLM response.")


def build_peak_prompt_from_series(
    x: Iterable[float],
    y: Iterable[float],
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Model must return ONLY JSON:
    {
      "peaks": [
        { "center": number, "height": number, "fwhm": number|null, "prominence": number|null }
      ],
      "baseline": number|null
    }
    """
    header = (system_prompt.strip() + "\n\n") if system_prompt else \
        "Return ONLY JSON with keys 'peaks' (list) and 'baseline'. No prose.\n"
    series = {
        "x": list(map(float, x)),
        "y": list(map(float, y)),
        "max_peaks": int(max_peaks),
        "fields": ["center", "height", "fwhm", "prominence"],
    }
    return header + "Series:\n" + json.dumps(series)


def llm_guess_peaks_from_data(
    llm: LLMClient,
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
) -> PeakResult:
    xs, ys = _downsample_xy(np.asarray(x), np.asarray(y), max_points=512)
    prompt = build_peak_prompt_from_series(xs, ys, max_peaks=max_peaks, system_prompt=system_prompt)
    text = llm.generate(prompt, max_tokens=max_tokens)
    obj = _extract_json(text)
    peaks = [
        PeakGuess(
            center=float(p.get("center")),
            height=float(p.get("height")),
            fwhm=(float(p["fwhm"]) if p.get("fwhm") is not None else None),
            prominence=(float(p["prominence"]) if p.get("prominence") is not None else None),
        )
        for p in obj.get("peaks", [])
    ]
    base = obj.get("baseline")
    return PeakResult(peaks=peaks, baseline=(float(base) if base is not None else None))


def llm_guess_peaks_from_image(
    llm: LLMClient,
    image_path: str,
    *,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
) -> PeakResult:
    if genai is None:
        raise RuntimeError("google-generativeai is required for image analysis.")
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to load images.")

    instr = ((system_prompt.strip() + "\n\n") if system_prompt else "") + \
        ("Return ONLY JSON with keys 'peaks' (list) and 'baseline'. No prose.\n"
         f"Max peaks: {int(max_peaks)}\n")

    img = Image.open(image_path)
    try:
        text = llm.generate_multimodal([instr, img], max_tokens=max_tokens)
    finally:
        try:
            img.close()
        except Exception:
            pass

    obj = _extract_json(text)
    peaks = [
        PeakGuess(
            center=float(p.get("center")),
            height=float(p.get("height")),
            fwhm=(float(p["fwhm"]) if p.get("fwhm") is not None else None),
            prominence=(float(p["prominence"]) if p.get("prominence") is not None else None),
        )
        for p in obj.get("peaks", [])
    ]
    base = obj.get("baseline")
    return PeakResult(peaks=peaks, baseline=(float(base) if base is not None else None))


def llm_guess_peaks(
    llm: LLMClient,
    x: np.ndarray,
    y: np.ndarray,
    *,
    use_image: bool = False,
    image_path: Optional[str] = None,
    max_peaks: int = 5,
    system_prompt: Optional[str] = None,
    max_tokens: int = 500,
    temp_plot_path: str = "_tmp_curve.png",
) -> PeakResult:
    if use_image:
        if image_path is None:
            save_plot_png(x, y, temp_plot_path)
            image_path = temp_plot_path
        return llm_guess_peaks_from_image(
            llm, image_path, max_peaks=max_peaks, system_prompt=system_prompt, max_tokens=max_tokens
        )
    return llm_guess_peaks_from_data(
        llm, x, y, max_peaks=max_peaks, system_prompt=system_prompt, max_tokens=max_tokens
    )


def _estimate_sigma_from_fwhm(fwhm: float) -> float:
    return float(fwhm) / 2.354820045  # FWHM = 2*sqrt(2*ln2)*sigma


def _guess_sigma_from_span(x: np.ndarray, frac: float = 0.08) -> float:
    """Guess sigma as a fraction of the x span. Default 8% for typical PL peaks."""
    span = float(np.max(x) - np.min(x))
    return max(1e-6, frac * span)


def _height_to_gaussian_amplitude(height: float, sigma: float) -> float:
    # For GaussianModel, amplitude is area = height * sigma * sqrt(2*pi)
    return float(height) * float(sigma) * np.sqrt(2.0 * np.pi)


def _gaussian_height_from_amp(amplitude: float, sigma: float) -> float:
    return float(amplitude) / (float(sigma) * np.sqrt(2.0 * np.pi))


def _build_composite_model(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",       # 'gaussian' or 'voigt'
    center_window: Optional[float] = None,   # absolute window around initial center
    sigma_bounds: Optional[Tuple[float, float]] = None,
) -> Tuple[lmfit.Model, lmfit.Parameters]:
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin
    if center_window is None:
        center_window = 0.10 * span
    if sigma_bounds is None:
        # Allow wide sigma range - PL peaks can be quite broad (up to 40% of span)
        sigma_bounds = (max(1e-6, 0.005 * span), max(1e-6, 0.40 * span))

    model = ConstantModel(prefix="c_")
    params = model.make_params()
    
    # For baseline, use the minimum of the data or provided value
    # Constrain baseline to be reasonable (near the data minimum, not above data)
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    y_range = y_max - y_min
    
    base_val = float(baseline) if baseline is not None else y_min
    # Constrain baseline to be at most slightly above the data minimum
    # and at least somewhat below (to allow for negative baseline if needed)
    baseline_max = y_min + 0.1 * y_range  # Baseline can't be more than 10% above minimum
    baseline_min = y_min - 0.5 * y_range  # Allow some negative baseline
    params["c_c"].set(value=base_val, min=baseline_min, max=baseline_max)

    for i, pk in enumerate(peaks):
        prefix = f"p{i}_"
        comp = select_peak_model(model_kind)(prefix=prefix)
        model = model + comp

        c0 = float(pk.center)
        
        # Find the closest x index to the peak center
        center_idx = np.argmin(np.abs(x - c0))
        actual_height_at_center = float(y[center_idx]) - base_val  # Height above baseline
        
        # Estimate FWHM from the actual data if not provided by LLM
        if pk.fwhm and pk.fwhm > 0:
            sigma0 = _estimate_sigma_from_fwhm(pk.fwhm)
        else:
            # Try to estimate FWHM from the data by finding half-max points
            half_max = base_val + actual_height_at_center / 2
            
            # Find left half-max point
            left_idx = center_idx
            while left_idx > 0 and y[left_idx] > half_max:
                left_idx -= 1
            
            # Find right half-max point
            right_idx = center_idx
            while right_idx < len(y) - 1 and y[right_idx] > half_max:
                right_idx += 1
            
            # Calculate FWHM from data
            if right_idx > left_idx:
                estimated_fwhm = float(x[right_idx] - x[left_idx])
                # Sanity check: FWHM should be reasonable (5-200nm for PL)
                if 5 < estimated_fwhm < 200:
                    sigma0 = _estimate_sigma_from_fwhm(estimated_fwhm)
                else:
                    sigma0 = _guess_sigma_from_span(x)
            else:
                sigma0 = _guess_sigma_from_span(x)
        
        # Use actual data height - always trust the data over LLM for intensity
        if actual_height_at_center > 0:
            peak_height = actual_height_at_center
        else:
            # Fallback: use LLM height or estimate from data range
            peak_height = pk.height if pk.height and pk.height > 0 else y_range * 0.5
        
        amp0 = _height_to_gaussian_amplitude(peak_height, sigma0)

        p = comp.make_params()
        p[f"{prefix}amplitude"].set(value=float(amp0), min=0.0, max=np.inf)
        p[f"{prefix}center"].set(value=c0, min=c0 - center_window, max=c0 + center_window)
        p[f"{prefix}sigma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        if model_kind == "voigt" and f"{prefix}gamma" in p:
            p[f"{prefix}gamma"].set(value=float(sigma0), min=sigma_bounds[0], max=sigma_bounds[1])

        params.update(p)

    return model, params


def _fit_once(
    x: np.ndarray,
    y: np.ndarray,
    peaks: List[PeakGuess],
    baseline: Optional[float],
    *,
    model_kind: str = "gaussian",
) -> lmfit.model.ModelResult:
    model, params = _build_composite_model(x, y, peaks, baseline, model_kind=model_kind)
    
    # Add weights to emphasize peak matching
    # Higher y values get more weight to force better peak height matching
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    y_range = y_max - y_min + 1e-9
    y_norm = (y - y_min) / y_range
    # Higher weights at the top of peaks (up to 10x base weight)
    weights = 1.0 + 9.0 * (y_norm ** 2) 
    
    return model.fit(y, params, x=x, weights=weights, nan_policy="omit", max_nfev=10000)


def _score_fit(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    resid = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) + 1e-16
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(ss_res / max(1, len(y))))
    return r2, rmse


def _result_to_peaks(result: lmfit.model.ModelResult, model_kind: str) -> Tuple[List[PeakGuess], Optional[float], Dict[str, float]]:
    params = result.best_values
    baseline = params.get("c_c", None)

    out: List[PeakGuess] = []
    i = 0
    while True:
        # Try different prefix patterns used by lmfit
        prefixes_to_try = [f"p{i}_", f"g{i}_", f"v{i}_", f"l{i}_"]
        found_peak = False

        for prefix in prefixes_to_try:
            amp_key = f"{prefix}amplitude"
            center_key = f"{prefix}center"
            sigma_key = f"{prefix}sigma"

            if amp_key in params and center_key in params and sigma_key in params:
                amp = float(params[amp_key])
                cen = float(params[center_key])
                sig = float(params[sigma_key])

                # Calculate height and FWHM based on model type
                if model_kind in ['gaussian', 'skewed_gaussian']:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig
                elif model_kind in ['lorentzian', 'split_lorentzian']:
                    height = amp / (sig * np.pi)  # Lorentzian height from amplitude
                    fwhm = 2.0 * sig
                elif model_kind in ['voigt', 'pseudovoigt', 'skewed_voigt']:
                    # For Voigt, use gamma parameter if available
                    gamma_key = f"{prefix}gamma"
                    if gamma_key in params:
                        gamma = float(params[gamma_key])
                        fwhm = 3.6013 * gamma  # Approximation for Voigt FWHM
                    else:
                        fwhm = 2.354820045 * sig  # Fallback to Gaussian
                    height = _gaussian_height_from_amp(amp, sig)  # Approximation
                else:
                    height = _gaussian_height_from_amp(amp, sig)
                    fwhm = 2.354820045 * sig

                out.append(PeakGuess(center=cen, height=height, fwhm=fwhm, prominence=None))
                found_peak = True
                break

        if not found_peak:
            break
        i += 1

    return out, (float(baseline) if baseline is not None else None), {k: float(v) for k, v in params.items()}


def fit_peaks_lmfit_with_retry(
    x: np.ndarray,
    y: np.ndarray,
    seed: PeakResult,
    *,
    model_kind: str = "gaussian",    # 'gaussian' or 'voigt'
    r2_target: float = 0.90,
    max_attempts: int = 5,
    jitter_frac_center: float = 0.01,
    jitter_frac_sigma: float = 0.25,
    allow_baseline_refit: bool = True,
) -> PeakFitResult:
    """
    Fit multi-peak PL curves with lmfit. Retries with randomized restarts
    until R² >= r2_target or attempts are exhausted.
    """
    if not _HAS_LMFIT:
        raise RuntimeError("lmfit is required. pip install lmfit")

    rng = np.random.default_rng()
    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin

    # Removed random jittering - using LLM-based refinement instead

    best_result = None
    best_stats = None
    best_report = ""
    best_params: Dict[str, float] = {}
    best_peaks: List[PeakGuess] = []
    success = False

    attempt = 0
    current_seed = seed

    while attempt < max_attempts:
        attempt += 1
        try:
            result = _fit_once(x, y, current_seed.peaks, current_seed.baseline, model_kind=model_kind)
        except Exception:
            # Skip this attempt if fitting fails
            continue

        yhat = result.best_fit
        r2, rmse = _score_fit(y, yhat)
        stats = PeakFitStats(
            r2=r2,
            rmse=rmse,
            aic=float(result.aic),
            bic=float(result.bic),
            redchi=float(result.redchi),
            nfev=int(result.nfev),
        )
        peaks_out, base_out, params_out = _result_to_peaks(result, model_kind)

        if (best_result is None) or (stats.r2 > best_stats.r2):  # type: ignore
            best_result = result
            best_stats = stats
            best_report = result.fit_report(min_correl=0.3)
            best_params = params_out
            best_peaks = peaks_out

        if stats.r2 >= r2_target:
            success = True
            break

        # For subsequent attempts, we'll rely on model switching in run_complete_analysis

    if best_result is None or best_stats is None:
        raise RuntimeError("lmfit did not produce a valid result across attempts")

    return PeakFitResult(
        success=success,
        stats=best_stats,
        best_params=best_params,
        peaks=best_peaks,
        baseline=best_params.get("c_c", None),
        report=best_report,
        model_kind=model_kind,
        lmfit_result=best_result,  # Store the actual lmfit result
    )


# ---------- Dataset curation (reads/wells/wavelengths) ----------

@dataclass
class CurveFittingConfig:
    data_csv: Optional[str] = None
    composition_csv: Optional[str] = None

    start_wavelength: Optional[int] = None
    end_wavelength: Optional[int] = None
    wavelength_step_size: Optional[int] = None

    read_selection: Union[str, Iterable[int], None] = "all"
    wells_to_ignore: Union[str, Iterable[str], None] = None

    fill_na_value: float = np.nan

    @staticmethod
    def _parse_int_list(text: str) -> List[int]:
        items: List[int] = []
        for chunk in re.split(r"\s*,\s*", text.strip()):
            if not chunk:
                continue
            if re.match(r"^\d+-\d+$", chunk):
                a, b = map(int, chunk.split("-"))
                if a > b:
                    a, b = b, a
                items.extend(range(a, b + 1))
            elif chunk.isdigit():
                items.append(int(chunk))
            else:
                raise ValueError(f"Invalid integer/range token: '{chunk}'")
        return sorted(set(items))

    @staticmethod
    def _parse_str_list(text: str) -> List[str]:
        parts = [p.strip().upper() for p in re.split(r"\s*,\s*", text.strip()) if p.strip()]
        for p in parts:
            if not re.match(r"^[A-H](?:[1-9]|1[0-2])$", p):
                raise ValueError(f"Invalid well id: '{p}' (expected A1..H12)")
        return parts

    @classmethod
    def from_user_inputs(
        cls,
        data_csv: str,
        composition_csv: str,
        read_selection: Union[str, Iterable[int], None] = "all",
        wells_to_ignore: Union[str, Iterable[str], None] = None,
        start_wavelength: Optional[int] = None,
        end_wavelength: Optional[int] = None,
        wavelength_step_size: Optional[int] = None,
        fill_na_value: float = 0.0,
    ) -> CurveFittingConfig:
        if isinstance(read_selection, str) and read_selection.lower() != "all":
            read_selection = cls._parse_int_list(read_selection)
        elif isinstance(read_selection, Iterable) and not isinstance(read_selection, (str, bytes)):
            read_selection = list(map(int, read_selection))

        if isinstance(wells_to_ignore, str):
            wells_to_ignore = cls._parse_str_list(wells_to_ignore)
        elif wells_to_ignore is None:
            wells_to_ignore = []
        else:
            wells_to_ignore = [str(w).strip().upper() for w in wells_to_ignore]

        return cls(
            data_csv=data_csv,
            composition_csv=composition_csv,
            start_wavelength=start_wavelength,
            end_wavelength=end_wavelength,
            wavelength_step_size=wavelength_step_size,
            read_selection=read_selection,
            wells_to_ignore=wells_to_ignore,
            fill_na_value=fill_na_value,
        )


class CurveFitting:
    # Generic pattern to find boundaries of ANY read block
    READ_HEADER_PATTERN = re.compile(r"^Read\s+(\d+):", re.I)
    # Specific pattern for the target data type (PL)
    TARGET_TYPE_PATTERN = re.compile(r"EM\s+Spectrum", re.I)

    def __init__(self, config: CurveFittingConfig):
        self.cfg = config

    @staticmethod
    def load_csvs(data_path: str, comp_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not (data_path.lower().endswith(".csv") and comp_path.lower().endswith(".csv")):
            raise ValueError("Both data_path and comp_path must be .csv files")
        data = pd.read_csv(data_path, header=None)
        data = data.replace("OVRFLW", np.nan)
        composition = pd.read_csv(comp_path, index_col=0)
        return data, composition

    @classmethod
    def _find_read_block_starts(cls, data: pd.DataFrame) -> Dict[int, int]:
        """Find the start row of every 'Read N:' block to determine boundaries."""
        starts: Dict[int, int] = {}
        first_col = data.iloc[:, 0].astype(str)
        for idx, val in first_col.items():
            m = cls.READ_HEADER_PATTERN.match(val)
            if m:
                starts[int(m.group(1))] = idx
        if not starts:
            raise ValueError("No 'Read N:' blocks found in data CSV")
        return dict(sorted(starts.items()))

    @staticmethod
    def _slice_block(data: pd.DataFrame, start_row: int, end_row: Optional[int]) -> pd.DataFrame:
        """Slice a data block from the raw CSV, stopping at the first empty row."""
        end = end_row if end_row is not None else len(data)
        block_full = data.iloc[start_row + 2 : end].copy()
        
        # Find the first row that is entirely empty or NaNs
        # This stops the block before it hits summary tables at the end of the file
        empty_rows = block_full.isnull().all(axis=1)
        if empty_rows.any():
            first_empty = np.where(empty_rows)[0][0]
            block = block_full.iloc[:first_empty].copy()
        else:
            block = block_full.copy()

        if 0 in block.columns:
            block = block.drop(columns=[0])
        
        if len(block) > 0:
            new_header = block.iloc[0]
            block = block.iloc[1:]
            block.columns = new_header
            block = block.apply(pd.to_numeric, errors="coerce")
        return block

    @classmethod
    def parse_all_reads(cls, data: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        starts = cls._find_read_block_starts(data)
        read_indices = list(starts.keys())
        blocks: Dict[int, pd.DataFrame] = {}
        for i, r in enumerate(read_indices):
            start_row = starts[r]
            end_row = starts[read_indices[i + 1]] if i + 1 < len(read_indices) else None
            blocks[r] = cls._slice_block(data, start_row, end_row)
        return blocks

    @staticmethod
    def select_reads(blocks: Dict[int, pd.DataFrame], selection: Union[str, Iterable[int], None]) -> Dict[int, pd.DataFrame]:
        if selection is None or (isinstance(selection, str) and selection.lower() == "all"):
            return dict(sorted(blocks.items()))
        desired = sorted(set(map(int, selection)))
        return {k: v for k, v in blocks.items() if k in desired}

    @staticmethod
    def drop_wells(blocks: Dict[int, pd.DataFrame], wells_to_ignore: Iterable[str]) -> Dict[int, pd.DataFrame]:
        wells_to_ignore = [w.strip().upper() for w in wells_to_ignore or []]
        if not wells_to_ignore:
            return blocks
        cleaned: Dict[int, pd.DataFrame] = {}
        for k, df in blocks.items():
            keep_cols = [c for c in df.columns if str(c).strip().upper() not in wells_to_ignore]
            cleaned[k] = df[keep_cols]
        return cleaned

    @staticmethod
    def infer_wavelength_vector(df: pd.DataFrame) -> np.ndarray:
        """Infer wavelength vector from dataframe.
        
        First checks for a 'Wavelength' column, then checks first column if numeric,
        then falls back to index-based.
        """
        # Strategy 1: Check for explicit wavelength column names
        wl_col = None
        for col in df.columns:
            col_str = str(col).strip().upper()
            # More flexible matching - check if column name contains wavelength-related terms
            if any(term in col_str for term in ['WAVELENGTH', 'WL', 'LAMBDA', 'NM']) and \
               not any(term in col_str for term in ['WELL', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']):
                # Also check if it looks like numeric data (not well names)
                try:
                    sample_val = pd.to_numeric(df[col].iloc[0] if len(df) > 0 else None, errors='coerce')
                    if not np.isnan(sample_val) and sample_val > 100:  # Wavelengths are typically > 100nm
                        wl_col = col
                        break
                except:
                    pass
        
        # Strategy 2: Check first column if it looks like wavelengths (common CSV format)
        if wl_col is None and len(df.columns) > 0:
            first_col = df.columns[0]
            first_col_str = str(first_col).strip().upper()
            # If first column is not a well name, check if it contains wavelength values
            is_well_name = bool(re.match(r'^[A-H](?:[1-9]|1[0-2])$', first_col_str))
            if not is_well_name:
                try:
                    # Check if first column has numeric values in wavelength range
                    # Use more rows to be confident
                    sample_vals = pd.to_numeric(df[first_col].head(10), errors='coerce').values
                    valid_samples = sample_vals[~np.isnan(sample_vals)]
                    if len(valid_samples) >= 3:  # Need at least 3 valid samples
                        sample_min, sample_max = float(valid_samples.min()), float(valid_samples.max())
                        # If values are in reasonable wavelength range (100-2000 nm), use it
                        # Also check that values are increasing (wavelengths should be monotonic)
                        is_increasing = len(valid_samples) < 2 or valid_samples[1] > valid_samples[0]
                        if sample_min >= 100 and sample_max <= 2000 and is_increasing:
                            wl_col = first_col
                            print(f"  Detected wavelength column: '{first_col}' (first column, range: {sample_min:.1f}-{sample_max:.1f} nm)")
                except Exception as e:
                    pass
        
        # Strategy 3: Check ALL columns for wavelength-like numeric data
        if wl_col is None:
            for col in df.columns:
                col_str = str(col).strip().upper()
                # Skip well columns
                if re.match(r'^[A-H](?:[1-9]|1[0-2])$', col_str):
                    continue
                try:
                    # Check if column has numeric values in wavelength range
                    sample_vals = pd.to_numeric(df[col].head(10), errors='coerce').values
                    valid_samples = sample_vals[~np.isnan(sample_vals)]
                    if len(valid_samples) >= 3:
                        sample_min, sample_max = float(valid_samples.min()), float(valid_samples.max())
                        # Check if it's a wavelength-like column
                        if sample_min >= 100 and sample_max <= 2000:
                            # Check if values are roughly evenly spaced (wavelengths usually are)
                            if len(valid_samples) >= 2:
                                step = valid_samples[1] - valid_samples[0]
                                if step > 0 and step < 100:  # Reasonable step size
                                    wl_col = col
                                    print(f"  Detected wavelength column: '{col}' (range: {sample_min:.1f}-{sample_max:.1f} nm)")
                                    break
                except:
                    pass
        
        if wl_col is not None:
            wl_values = pd.to_numeric(df[wl_col], errors='coerce').values
            # Remove NaN values but preserve order
            valid_mask = ~np.isnan(wl_values)
            if np.any(valid_mask):
                valid_wl = wl_values[valid_mask]
                # Ensure wavelengths are in ascending order
                if len(valid_wl) > 1 and valid_wl[0] > valid_wl[-1]:
                    valid_wl = valid_wl[::-1]
                print(f"  Extracted wavelength range: {valid_wl.min():.1f} - {valid_wl.max():.1f} nm ({len(valid_wl)} points)")
                return valid_wl
        
        # Fallback to index-based (shouldn't happen if CSV is properly formatted)
        n = len(df)
        print(f"  WARNING: No wavelength column found, using index-based (0-{n-1}).")
        print(f"  Available columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        # Check if any column has wavelength-like values
        for col in df.columns[:5]:  # Check first 5 columns
            try:
                sample = pd.to_numeric(df[col].head(3), errors='coerce').values
                valid = sample[~np.isnan(sample)]
                if len(valid) > 0:
                    print(f"    Column '{col}': sample values {valid}")
            except:
                pass
        return np.arange(n)

    def build_wavelengths(self, exemplar_df: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        
        # FIRST: Try to infer wavelength from CSV column (most accurate)
        inferred_wl = self.infer_wavelength_vector(exemplar_df)
        
        # If we successfully inferred wavelengths from CSV, use them
        # (unless they seem obviously wrong - e.g., all zeros or starting at 0)
        if len(inferred_wl) > 0:
            wl_min, wl_max = float(inferred_wl.min()), float(inferred_wl.max())
            # Check if inferred wavelengths seem reasonable (not just index-based)
            if wl_min >= 100 and wl_max <= 2000:  # Reasonable wavelength range
                data_length = len(exemplar_df)
                # Ensure length matches
                if len(inferred_wl) == data_length:
                    return inferred_wl
                elif len(inferred_wl) > data_length:
                    # Trim to match data length
                    return inferred_wl[:data_length]
                else:
                    # Extend if needed (shouldn't happen, but handle it)
                    step = inferred_wl[1] - inferred_wl[0] if len(inferred_wl) > 1 else 1
                    extension = np.arange(inferred_wl[-1] + step, 
                                         inferred_wl[-1] + step * (data_length - len(inferred_wl) + 1),
                                         step)
                    return np.concatenate([inferred_wl, extension[:data_length - len(inferred_wl)]])
        
        # FALLBACK: Use config wavelength range if specified
        data_length = len(exemplar_df)
        if cfg.start_wavelength is not None and cfg.end_wavelength is not None and cfg.wavelength_step_size is not None:
            # Build wavelength array from range, but trim to match actual data length
            x = np.arange(cfg.start_wavelength, cfg.end_wavelength + cfg.wavelength_step_size, cfg.wavelength_step_size)
            # Ensure wavelength array matches data length
            if len(x) > data_length:
                x = x[:data_length]
            elif len(x) < data_length:
                # If shorter, extend with last step
                last_val = x[-1] if len(x) > 0 else cfg.start_wavelength
                extension = np.arange(last_val + cfg.wavelength_step_size,
                                     last_val + cfg.wavelength_step_size * (data_length - len(x) + 1),
                                     cfg.wavelength_step_size)
                x = np.concatenate([x, extension[:data_length - len(x)]])
            return x
        
        # LAST RESORT: Index-based (shouldn't happen with proper CSV)
        return np.arange(data_length)

    def stack_blocks(self, blocks: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], Dict[int, np.ndarray]]:
        # Filter out non-well columns (like "Wavelength", "WAVELENGTH", etc.)
        def is_well_column(col: str) -> bool:
            col_upper = str(col).strip().upper()
            # Exclude wavelength-related columns
            if col_upper in ['WAVELENGTH', 'WAVELENGTH (NM)', 'WAVELENGTH_NM', 'WL']:
                return False
            # Only include valid well names (A1-H12 pattern)
            return bool(re.match(r'^[A-H](?:[1-9]|1[0-2])$', col_upper))

        well_sets = []
        for df in blocks.values():
            well_cols = [str(c) for c in df.columns if is_well_column(str(c))]
            well_sets.append(set(well_cols))

        common_wells = sorted(set.intersection(*well_sets)) if well_sets else []
        if not common_wells:
            common_wells = sorted(set.union(*well_sets)) if well_sets else []

        read_indices = sorted(blocks.keys())
        num_reads = len(read_indices)
        
        # Find the maximum number of wavelength points across all reads
        # This handles cases where different reads have different data lengths
        read_lengths = [len(blocks[r]) for r in read_indices]
        max_wavelengths = max(read_lengths) if read_lengths else 0
        
        if max_wavelengths == 0:
            raise ValueError("No data found in any read blocks")
        
        num_wells = len(common_wells)
        
        # Build wavelength arrays PER READ (each read may have different wavelength ranges)
        # Store them in a dict keyed by read number
        read_wavelengths = {}
        for r in read_indices:
            read_df = blocks[r]
            read_wl = self.build_wavelengths(read_df)
            read_wavelengths[r] = read_wl
            print(f"  Read {r} wavelength range: {read_wl.min():.1f}-{read_wl.max():.1f} nm ({len(read_wl)} points)")
        
        # Use the longest read's wavelength array as the "master" for tensor dimensions
        # But we'll use per-read wavelengths when extracting data
        longest_read_idx = read_lengths.index(max_wavelengths)
        exemplar = blocks[read_indices[longest_read_idx]]
        x = self.build_wavelengths(exemplar)

        # Use the maximum length to avoid broadcasting errors
        tensor = np.full((num_reads, max_wavelengths, num_wells), fill_value=np.nan, dtype=float)
        
        for i, r in enumerate(read_indices):
            df = blocks[r].reindex(columns=common_wells)
            arr = df.to_numpy(dtype=float)
            actual_length = arr.shape[0]
            actual_wells = arr.shape[1]
            
            # Safety check: ensure well dimension matches
            if actual_wells != num_wells:
                raise ValueError(
                    f"Read {r} has {actual_wells} wells but expected {num_wells}. "
                    f"Common wells: {common_wells}, Read columns: {list(df.columns)}"
                )
            
            # Only assign up to the actual length of this read
            # Shorter reads will have NaN padding at the end (converted to fill_na_value)
            if actual_length <= max_wavelengths:
                tensor[i, :actual_length, :] = arr
            else:
                # If somehow longer (shouldn't happen), truncate
                tensor[i, :max_wavelengths, :] = arr[:max_wavelengths, :]

        tensor = np.nan_to_num(tensor, nan=self.cfg.fill_na_value)
        
        # Ensure wavelength array matches the maximum length exactly
        # This is critical to avoid broadcasting errors
        if len(x) != max_wavelengths:
            if len(x) < max_wavelengths:
                # Extend wavelength array by inferring step size from exemplar
                if len(x) > 1:
                    step = x[1] - x[0]
                    # Extend beyond the last value
                    num_extra = max_wavelengths - len(x)
                    extension = np.arange(x[-1] + step, x[-1] + step * (num_extra + 1), step)
                    x = np.concatenate([x, extension[:num_extra]])
                elif len(x) == 1:
                    # Single value - create evenly spaced array
                    x = np.linspace(x[0], x[0] + max_wavelengths - 1, max_wavelengths)
                else:
                    # Empty - create default range (shouldn't happen)
                    x = np.arange(max_wavelengths)
            else:
                # Truncate to match tensor size (shouldn't happen if using longest read)
                x = x[:max_wavelengths]
        
        # Final safety check - ensure exact match
        if len(x) != max_wavelengths:
            raise ValueError(f"Wavelength array length ({len(x)}) does not match tensor wavelength dimension ({max_wavelengths})")
        
        return tensor, x, common_wells, read_indices, read_wavelengths

    def curate_dataset(self) -> Dict[str, object]:
        if not self.cfg.data_csv or not self.cfg.composition_csv:
            raise ValueError("Both data_csv and composition_csv must be provided.")

        raw_data, composition = self.load_csvs(self.cfg.data_csv, self.cfg.composition_csv)
        all_blocks = self.parse_all_reads(raw_data)
        sel_blocks = self.select_reads(all_blocks, self.cfg.read_selection)
        if not sel_blocks:
            raise ValueError("No reads selected; check read_selection.")
        sel_blocks = self.drop_wells(sel_blocks, self.cfg.wells_to_ignore)
        tensor, wavelengths, wells, reads, read_wavelengths = self.stack_blocks(sel_blocks)

        comp_aligned = composition.copy()
        comp_aligned = comp_aligned.loc[:, [w for w in wells if w in comp_aligned.columns]]

        return {
            "tensor": tensor,
            "wavelengths": wavelengths,  # Master wavelength array (for compatibility)
            "read_wavelengths": read_wavelengths,  # Per-read wavelength arrays
            "wells": wells,
            "reads": reads,
            "composition": comp_aligned,
            "blocks": sel_blocks,
        }

    def get_xy(self, curated: Dict[str, object], well: str, read: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        well = str(well).strip().upper()
        wells: List[str] = curated["wells"]  # type: ignore
        if well not in wells:
            raise KeyError(f"Well '{well}' not in curated set: {wells}")
        reads: List[int] = curated["reads"]  # type: ignore
        if read is None:
            read = reads[0]
        if read not in reads:
            raise KeyError(f"Read {read} not in curated reads: {reads}")
        w_idx = wells.index(well)
        r_idx = reads.index(read)
        
        # Use per-read wavelength array if available, otherwise fall back to master
        read_wavelengths = curated.get("read_wavelengths", {})  # type: ignore
        if read_wavelengths and read in read_wavelengths:
            # Use this read's specific wavelength array
            x = read_wavelengths[read].copy()
            print(f"  Using read-specific wavelength array for read {read}: {x.min():.1f}-{x.max():.1f} nm")
        else:
            # Fall back to master wavelength array (for backward compatibility)
            x = curated["wavelengths"].copy()  # type: ignore
        
        y = curated["tensor"][r_idx, :, w_idx].copy()  # type: ignore
        
        # Ensure x and y have compatible lengths (y may be padded to max_wavelengths)
        # Trim x to match y's actual data length
        if len(x) > len(y):
            x = x[:len(y)]
        elif len(x) < len(y):
            # Extend x if needed (shouldn't happen, but handle it)
            step = x[1] - x[0] if len(x) > 1 else 1
            num_extra = len(y) - len(x)
            extension = np.arange(x[-1] + step, x[-1] + step * (num_extra + 1), step)
            x = np.concatenate([x, extension[:num_extra]])
        
        # Remove any padded zeros or NaNs at the beginning and end of the data 
        # (This happens when reads have different lengths or starting points)
        # Strategy: Find where valid data starts by looking for NaN/invalid values
        
        # Identify invalid values based on fill_na_value
        if np.isnan(self.cfg.fill_na_value):
            # NaN fill value: invalid = NaN
            invalid_mask = np.isnan(y)
        else:
            # Numeric fill value: invalid = NaN or equal to fill_na_value
            fill_val = self.cfg.fill_na_value
            invalid_mask = np.isnan(y) | (y == fill_val)
            # Also check for values very close to fill_na_value (within small tolerance)
            if not np.isnan(fill_val) and fill_val != 0:
                tolerance = max(abs(fill_val) * 0.001, 1e-10)
                invalid_mask |= (np.abs(y - fill_val) < tolerance)
        
        # Find first and last valid indices
        valid_mask = ~invalid_mask
        if np.any(valid_mask):
            valid_indices = np.where(valid_mask)[0]
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]
            
            # Trim both x and y to only include valid data points
            # This ensures wavelength array matches actual data range
            x_trimmed = x[first_valid : last_valid + 1]
            y_trimmed = y[first_valid : last_valid + 1]
            
            # Verify the trim makes sense (wavelengths should be monotonic)
            if len(x_trimmed) > 1:
                x_diff = np.diff(x_trimmed)
                if np.any(x_diff <= 0):
                    print(f"  Warning: Wavelength array not monotonic after trimming for {well} read {read}")
            
            print(f"  Trimmed data for {well} read {read}: removed {first_valid} leading and {len(y)-last_valid-1} trailing invalid points. "
                  f"Wavelength range: {x_trimmed.min():.1f}-{x_trimmed.max():.1f} nm ({len(x_trimmed)} points)")
            
            x = x_trimmed
            y = y_trimmed
        else:
            # No valid data found - this will be caught by validation below
            print(f"  Warning: No valid data found for {well} read {read} (all values are invalid)")
            pass
        
        # Apply wavelength filtering if specified
        cfg = self.cfg
        mask = np.ones(len(x), dtype=bool)
        
        # Filter by start/end wavelength
        if cfg.start_wavelength is not None:
            mask &= (x >= cfg.start_wavelength)
        if cfg.end_wavelength is not None:
            mask &= (x <= cfg.end_wavelength)
        
        # Apply mask
        x_filtered = x[mask]
        y_filtered = y[mask]
        
        # Check if filtering removed all data
        if len(x_filtered) == 0 and len(x) > 0:
            x_min, x_max = float(x.min()), float(x.max())
            raise ValueError(
                f"Wavelength filtering removed all data points for well {well}, read {read}!\n"
                f"  Data wavelength range: {x_min:.1f} - {x_max:.1f} nm\n"
                f"  Filter range: {cfg.start_wavelength if cfg.start_wavelength else 'none'} - "
                f"{cfg.end_wavelength if cfg.end_wavelength else 'none'} nm\n"
                f"  Solution: Disable wavelength filtering or adjust the range to match your data."
            )
        
        x = x_filtered
        y = y_filtered
        
        # Apply step size filtering if specified
        if cfg.wavelength_step_size is not None and cfg.wavelength_step_size > 1:
            # Keep every nth point
            step_indices = np.arange(0, len(x), cfg.wavelength_step_size)
            x = x[step_indices]
            y = y[step_indices]
        
        # Final validation: ensure we have data
        if len(x) == 0 or len(y) == 0:
            raise ValueError(
                f"No valid data points found for well {well}, read {read}. "
                f"This may be due to:\n"
                f"  - Wavelength filtering removing all points (start={cfg.start_wavelength}, end={cfg.end_wavelength})\n"
                f"  - All data points being NaN or invalid\n"
                f"  - Data trimming removing all points"
            )
        
        if len(x) != len(y):
            raise ValueError(
                f"Wavelength and intensity arrays have different lengths for well {well}, read {read}: "
                f"x has {len(x)} points, y has {len(y)} points"
            )
        
        return x, y


# ---------- Agent-callable helpers ----------

def build_agent_config(
    data_csv: str,
    composition_csv: str,
    read_selection: Union[str, Iterable[int], None] = "all",
    wells_to_ignore: Union[str, Iterable[str], None] = None,
    start_wavelength: Optional[int] = None,
    end_wavelength: Optional[int] = None,
    wavelength_step_size: Optional[int] = None,
    fill_na_value: float = 0.0,
) -> CurveFittingConfig:
    return CurveFittingConfig.from_user_inputs(
        data_csv=data_csv,
        composition_csv=composition_csv,
        read_selection=read_selection,
        wells_to_ignore=wells_to_ignore,
        start_wavelength=start_wavelength,
        end_wavelength=end_wavelength,
        wavelength_step_size=wavelength_step_size,
        fill_na_value=fill_na_value,
    )


def curate_dataset(config: CurveFittingConfig) -> Dict[str, object]:
    agent = CurveFitting(config)
    return agent.curate_dataset()


def get_xy_for_well(config: CurveFittingConfig, well: str, read: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    agent = CurveFitting(config)
    curated = agent.curate_dataset()
    return agent.get_xy(curated, well=well, read=read)