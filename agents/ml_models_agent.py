from typing import Any, Dict, Optional
import logging

from agents.base import BaseAgent
from tools.memory import MemoryManager

logger = logging.getLogger(__name__)

# Lazy import streamlit to avoid issues in headless mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except (ImportError, RuntimeError):
    STREAMLIT_AVAILABLE = False
    st = None


class MLModelsAgent(BaseAgent):
    """Agent wrapper for ML Models automation and routing."""

    def __init__(self, name: str = "ML Models", desc: str = "Runs ML model automation"):
        super().__init__(name, desc)
        self.memory = MemoryManager()

    def confidence(self, payload: Dict[str, Any]) -> float:
        conf = 0.1
        if payload and isinstance(payload, dict):
            action = payload.get("action") or payload.get("next_agent")
            if str(action).lower() in {"ml_models", "ml", "ml_models_agent"}:
                conf = 0.8
            if payload.get("results_json") or payload.get("json_path"):
                conf = max(conf, 0.6)
            if payload.get("results_csv") or payload.get("csv_path"):
                conf = max(conf, 0.6)

        if STREAMLIT_AVAILABLE and st is not None:
            if st.session_state.get("ml_auto_json_path") or st.session_state.get("ml_auto_csv_path"):
                conf = max(conf, 0.6)

        return conf

    def run_agent(self, memory: MemoryManager, payload: Optional[Dict[str, Any]] = None) -> Any:
        # If Streamlit is available, route to the ML Models page
        if STREAMLIT_AVAILABLE and st is not None:
            try:
                st.session_state.next_agent = "ml_models"
                st.switch_page("pages/ml_models.py")
                return {"success": True, "routed": True}
            except Exception:
                pass

        # Headless automation
        try:
            from tools.ml_automation import run_automated_ml_model

            model_choice = None
            auto_config = None
            json_path = None
            csv_path = None
            composition_csv = None

            if STREAMLIT_AVAILABLE and st is not None:
                model_choice = st.session_state.get("optimization_model_choice")
                auto_config = st.session_state.get("ml_model_config", {})
                json_path = st.session_state.get("ml_auto_json_path")
                csv_path = st.session_state.get("ml_auto_csv_path")
                composition_csv = st.session_state.get("ml_auto_composition_path")

            if payload and isinstance(payload, dict):
                json_path = payload.get("results_json") or payload.get("json_path") or json_path
                csv_path = payload.get("results_csv") or payload.get("csv_path") or csv_path
                composition_csv = payload.get("composition_csv") or composition_csv

            if not model_choice:
                model_choice = "Single-objective GP (scikit-learn)"

            result = run_automated_ml_model(
                model_choice=model_choice,
                json_path=json_path,
                csv_path=csv_path,
                composition_csv=composition_csv,
                auto_config=auto_config,
            )

            if memory and hasattr(memory, "log_event"):
                memory.log_event("ml_automation", {"result": result}, mode="ml_models")
            return result
        except Exception as e:
            logger.error(f"MLModelsAgent failed: {e}")
            if memory and hasattr(memory, "log_event"):
                memory.log_event("ml_automation", {"error": str(e)}, mode="ml_models")
            return {"success": False, "error": str(e)}
