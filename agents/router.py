
from typing import Any, Dict, List, Optional
import logging

from tools import socratic

logger = logging.getLogger(__name__)

# Lazy import streamlit to avoid issues in headless mode
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except (ImportError, RuntimeError):
    STREAMLIT_AVAILABLE = False
    st = None


class AgentRouter:
    """
    Central router coordinating which agent should run next.

    Modes (configured via `st.session_state.routing_mode`):
    - **Autonomous (LLM)**: use per‑agent `confidence` plus an LLM tie‑breaker
      that looks at current session context (files, hypothesis, experiment data).
    - **Manual Workflow**: follow the user‑defined ordered workflow in
      `st.session_state.manual_workflow`.

    The router does not render UI itself; pages/agents can call it when they
    want the “next” agent to run.
    """

    def __init__(self, agents: List[Any], fallback_agent: Optional[Any] = None):
        self.agents = agents
        self.fallback_agent = fallback_agent

    def _find_agent_by_name(self, name: str) -> Optional[Any]:
        for agent in self.agents:
            if getattr(agent, "name", "").lower() == name.lower():
                return agent
        return None

    def _llm_select_agent(self, payload: Dict[str, Any]) -> Optional[Any]:
        """
        Ask the LLM which agent should run next, given available agents and
        high‑level context from the current session.
        """
        agent_names = [getattr(a, "name", "Unnamed Agent") for a in self.agents]

        # Get context from session state if available
        uploaded_files = []
        last_hypothesis = None
        experimental_outputs = None
        experimental_constraints = {}
        curve_fitting_results = None
        
        if STREAMLIT_AVAILABLE and st is not None:
            try:
                uploaded_files = st.session_state.get("uploaded_files", [])
                last_hypothesis = st.session_state.get("last_hypothesis")
                experimental_outputs = st.session_state.get("experimental_outputs")
                experimental_constraints = st.session_state.get("experimental_constraints", {})
                curve_fitting_results = st.session_state.get("curve_fitting_results")
            except (RuntimeError, AttributeError, NameError):
                # Fallback to defaults if session state access fails
                pass

        context = {
            "payload": payload,
            "uploaded_files": uploaded_files,
            "last_hypothesis": last_hypothesis,
            "experimental_outputs": experimental_outputs,
            "experimental_constraints": experimental_constraints,
            "curve_fitting_results": curve_fitting_results is not None,
        }

        prompt = f"""
You are a routing controller for a lab-assistant application with multiple agents.

Available agents (choose exactly one):
- {"; ".join(agent_names)}

You are given JSON-like context about the current state (files, hypothesis, experiment data, etc.).
Decide which single agent from the list above should run *next*.

Context:
{context}

Return ONLY the exact name of the chosen agent from the list above, with no explanation, no formatting.
"""
        try:
            choice_raw = socratic.generate_text_with_llm(prompt).strip()
        except Exception as e:
            logger.warning(f"Unable to generate agent name from LLM: {e}")
            # Only show Streamlit warning if we're actually in a Streamlit context
            # Don't try to import st again - use the module-level one if available
            try:
                if STREAMLIT_AVAILABLE and st is not None and hasattr(st, 'warning'):
                    st.warning(f"Unable to generate agent name from LLM. Try again. Error: {e}")
            except (RuntimeError, AttributeError, NameError):
                # st might not be available in headless mode
                pass
            return None

        # Normalise and try to map back to a known agent
        for name in agent_names:
            if name.lower() in choice_raw.lower():
                return self._find_agent_by_name(name)

        return None

    def _route_manual(self) -> Optional[Any]:
        """
        Manual workflow: follow `st.session_state.manual_workflow` and
        `st.session_state.workflow_index`.
        """
        if not STREAMLIT_AVAILABLE:
            # In headless mode, use default workflow
            workflow = ["Hypothesis Agent", "Experiment Agent", "Curve Fitting", "Analysis Agent"]
            index = 0
        else:
            try:
                workflow = st.session_state.get(
                    "manual_workflow",
                    ["Hypothesis Agent", "Experiment Agent", "Curve Fitting", "Analysis Agent"],
                )
                index = st.session_state.get("workflow_index", 0)
            except (RuntimeError, AttributeError):
                workflow = ["Hypothesis Agent", "Experiment Agent", "Curve Fitting", "Analysis Agent"]
                index = 0

        if not workflow or index >= len(workflow):
            return None

        target_name = workflow[index]
        return self._find_agent_by_name(target_name)

    def route(self, payload: Dict[str, Any], memory: Any) -> Any:
        """
        Route to the next agent based on configured routing mode.

        - In autonomous mode, use agents' `confidence` plus an optional LLM
          decision using the current session context.
        - In manual mode, follow the user-configured workflow order.
        """
        # Store payload in memory so agents can access it
        try:
            if hasattr(memory, 'log_event'):
                memory.log_event(
                    "router",
                    {"current_payload": payload},
                    mode="router"
                )
        except Exception:
            pass
        
        if STREAMLIT_AVAILABLE:
            try:
                routing_mode = st.session_state.get("routing_mode", "Autonomous (LLM)")
            except (RuntimeError, AttributeError):
                routing_mode = "Autonomous (LLM)"
        else:
            # In headless mode, use autonomous routing
            routing_mode = "Autonomous (LLM)"

        # Manual workflow mode – ignore confidence and LLM, follow user order.
        if routing_mode == "Manual":
            agent = self._route_manual()
            if agent is None and self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            if agent is None:
                raise RuntimeError("Manual workflow is empty or exhausted.")

            # Advance workflow index for next call
            try:
                if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                    st.session_state.workflow_index = st.session_state.get("workflow_index", 0) + 1
                    st.session_state.agent_usage_counts["router"] = st.session_state.agent_usage_counts.get("router", 0) + 1
                    agent_key = getattr(agent, "name", "unknown").split()[0].lower()
                    st.session_state.agent_usage_counts[agent_key] = (
                        st.session_state.agent_usage_counts.get(agent_key, 0) + 1
                    )
            except (RuntimeError, AttributeError):
                pass
            return agent.run_agent(memory)

        # Autonomous mode: start with confidence-based scoring
        scored = []
        for agent in self.agents:
            try:
                conf = agent.confidence(payload)
                # Ensure confidence is a float (handle None)
                if conf is None:
                    conf = 0.0
                conf = float(conf)
                scored.append((conf, agent))
            except Exception as e:
                # If confidence method fails, assign low confidence
                logger.warning(f"Agent {getattr(agent, 'name', 'unknown')} confidence failed: {e}")
                scored.append((0.0, agent))
        
        if not scored:
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise RuntimeError("No agents available.")
        
        scored.sort(reverse=True, key=lambda x: x[0])

        score, top_agent = scored[0]

        # If all agents are uncertain, fall back
        if score is None or score < 0.4:
            if self.fallback_agent:
                try:
                    if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                        st.session_state.agent_usage_counts["router"] = st.session_state.agent_usage_counts.get("router", 0) + 1
                except (RuntimeError, AttributeError):
                    pass
                return self.fallback_agent.run_agent(memory)
            raise RuntimeError("No agent is confident enough.")

        # Let the LLM override / confirm the top choice using global context
        # Skip LLM selection in headless mode (watcher server) to avoid API key dependency
        llm_agent = None
        # Only use LLM if Streamlit is available AND we have an API key
        if STREAMLIT_AVAILABLE:
            try:
                import os
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    llm_agent = self._llm_select_agent(payload)
                else:
                    logger.debug("Skipping LLM agent selection - no API key in environment")
            except Exception as e:
                logger.warning(f"LLM agent selection failed: {e}")
        
        chosen_agent = llm_agent or top_agent

        try:
            # Update usage counts if Streamlit is available
            try:
                if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
                    st.session_state.agent_usage_counts["router"] = st.session_state.agent_usage_counts.get("router", 0) + 1
                    key = getattr(chosen_agent, "name", "unknown").split()[0].lower()
                    st.session_state.agent_usage_counts[key] = (
                        st.session_state.agent_usage_counts.get(key, 0) + 1
                    )
            except (RuntimeError, AttributeError):
                pass
            
            # Pass payload to run_agent if it accepts it
            import inspect
            sig = inspect.signature(chosen_agent.run_agent)
            if 'payload' in sig.parameters:
                return chosen_agent.run_agent(memory, payload=payload)
            else:
                return chosen_agent.run_agent(memory)
        except Exception as e:
            logger.error(f"Error running agent {getattr(chosen_agent, 'name', 'unknown')}: {e}")
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise