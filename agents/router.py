
from typing import Any, Dict, List, Optional

import streamlit as st

from tools import socratic


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

        uploaded_files = st.session_state.get("uploaded_files", [])
        last_hypothesis = st.session_state.get("last_hypothesis")
        experimental_outputs = st.session_state.get("experimental_outputs")
        experimental_constraints = st.session_state.get("experimental_constraints", {})

        context = {
            "payload": payload,
            "uploaded_files": uploaded_files,
            "last_hypothesis": last_hypothesis,
            "experimental_outputs": experimental_outputs,
            "experimental_constraints": experimental_constraints,
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
            st.warning(f"Unable to generate agent name from LLM. Try again. Error: {e}")
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
        workflow = st.session_state.get(
            "manual_workflow",
            ["Hypothesis Agent", "Experiment Agent", "Curve Fitting"],
        )
        index = st.session_state.get("workflow_index", 0)

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
        routing_mode = st.session_state.get("routing_mode", "Autonomous (LLM)")

        # Manual workflow mode – ignore confidence and LLM, follow user order.
        if routing_mode == "Manual":
            agent = self._route_manual()
            if agent is None and self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            if agent is None:
                raise RuntimeError("Manual workflow is empty or exhausted.")

            # Advance workflow index for next call
            st.session_state.workflow_index = st.session_state.get("workflow_index", 0) + 1
            st.session_state.agent_usage_counts["router"] += 1
            st.session_state.agent_usage_counts[getattr(agent, "name", "unknown").split()[0].lower()] = \
                st.session_state.agent_usage_counts.get(
                    getattr(agent, "name", "unknown").split()[0].lower(), 0
                ) + 1
            return agent.run_agent(memory)

        # Autonomous mode: start with confidence-based scoring
        scored = [
            (agent.confidence(payload), agent)
            for agent in self.agents
        ]
        scored.sort(reverse=True, key=lambda x: x[0])

        score, top_agent = scored[0]

        # If all agents are uncertain, fall back
        if score < 0.4:
            if self.fallback_agent:
                st.session_state.agent_usage_counts["router"] += 1
                return self.fallback_agent.run_agent(memory)
            raise RuntimeError("No agent is confident enough.")

        # Let the LLM override / confirm the top choice using global context
        llm_agent = self._llm_select_agent(payload)
        chosen_agent = llm_agent or top_agent

        try:
            st.session_state.agent_usage_counts["router"] += 1
            key = getattr(chosen_agent, "name", "unknown").split()[0].lower()
            st.session_state.agent_usage_counts[key] = (
                st.session_state.agent_usage_counts.get(key, 0) + 1
            )
            return chosen_agent.run_agent(memory)
        except Exception:
            if self.fallback_agent:
                return self.fallback_agent.run_agent(memory)
            raise