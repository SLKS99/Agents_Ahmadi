from typing import Dict, Any, Optional, List
import json
import os
import streamlit as st
from agents.base import BaseAgent
from tools.memory import MemoryManager
from tools import socratic
from tools.experiment_memory import get_experiment_memory

# Lazy import socratic module - only import when needed
_socratic_module = None

def _lazy_import_socratic():
    """Lazy import of socratic module to speed up module loading"""
    global _socratic_module
    if _socratic_module is None:
        from tools import socratic
        _socratic_module = socratic
    return _socratic_module


class AnalysisAgent(BaseAgent):
    """
    Analysis Agent that evaluates curve fitting results, relates them to hypothesis
    and experimental data, decides if more experiments are needed, and provides
    literature-backed explanations of results and impacts.
    """
    
    def __init__(self, name: str = "Analysis Agent", desc: str = "Analyzes curve fitting results and relates them to hypothesis and experiments"):
        super().__init__(name, desc)
        self.memory = MemoryManager()
    
    def confidence(self, payload: Dict[str, Any]) -> float:
        """
        Return confidence score for analysis tasks.
        High confidence when curve fitting results are available.
        """
        # Check if curve fitting results exist
        curve_fitting_results = self._get_curve_fitting_results()
        if curve_fitting_results:
            return 0.9
        
        # Medium confidence if hypothesis and experiment data exist
        try:
            import streamlit as st
            hypothesis = self.memory.view_component("hypothesis")
            if not hypothesis:
                try:
                    hypothesis = st.session_state.get("last_hypothesis")
                except (RuntimeError, AttributeError):
                    hypothesis = None
            experimental_outputs = None
            try:
                experimental_outputs = st.session_state.get("experimental_outputs")
            except (RuntimeError, AttributeError):
                pass
        except (ImportError, RuntimeError):
            hypothesis = None
            experimental_outputs = None
        
        if hypothesis and experimental_outputs:
            return 0.7
        
        # Low confidence otherwise
        return 0.3
    
    def _get_curve_fitting_results(self) -> Optional[Dict[str, Any]]:
        """Retrieve curve fitting results from files or session state."""
        # Try to get from session state first
        if hasattr(st.session_state, 'curve_fitting_results'):
            return st.session_state.curve_fitting_results
        
        # Try to find results JSON files in results directory
        results_dir = "results"
        if os.path.exists(results_dir):
            json_files = [f for f in os.listdir(results_dir) if f.endswith("_peak_fit_results.json")]
            if json_files:
                # Use the most recent file
                latest_file = max(json_files, key=lambda f: os.path.getmtime(os.path.join(results_dir, f)))
                file_path = os.path.join(results_dir, latest_file)
                try:
                    with open(file_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    st.warning(f"Could not load curve fitting results from {file_path}: {e}")
        
        return None
    
    def _get_gp_results(self) -> Optional[Dict[str, Any]]:
        """Retrieve Gaussian Process model results from session state."""
        return st.session_state.get("gp_results")
    
    def _get_hypothesis_context(self) -> str:
        """Build context string from hypothesis agent outputs."""
        context_parts = []
        
        # Get hypothesis
        hypothesis = self.memory.view_component("hypothesis") or st.session_state.get("last_hypothesis")
        if hypothesis:
            context_parts.append(f"**Hypothesis:**\n{hypothesis}\n")
        
        # Get clarified question
        clarified_q = self.memory.view_component("clarified_question")
        if clarified_q:
            context_parts.append(f"**Research Question:**\n{clarified_q}\n")
        
        # Get socratic questions
        socratic_q = self.memory.view_component("socratic_pass")
        if socratic_q:
            context_parts.append(f"**Socratic Analysis:**\n{socratic_q}\n")
        
        return "\n".join(context_parts) if context_parts else "No hypothesis context available."
    
    def _get_experimental_context(self) -> str:
        """Build context string from experimental agent outputs."""
        experimental_outputs = st.session_state.get("experimental_outputs")
        if not experimental_outputs:
            return "No experimental data available."
        
        context_parts = []
        
        if experimental_outputs.get("plan"):
            context_parts.append(f"**Experimental Plan:**\n{experimental_outputs['plan']}\n")
        
        if experimental_outputs.get("worklist"):
            context_parts.append(f"**Worklist:**\n{experimental_outputs['worklist'][:500]}...\n")
        
        return "\n".join(context_parts) if context_parts else "No experimental data available."
    
    def _summarize_curve_fitting_results(self, results: Dict[str, Any]) -> str:
        """Create a summary of curve fitting results for LLM analysis."""
        summary_parts = []
        
        # Extract key information from results
        if isinstance(results, dict):
            # Check if it's the format from run_complete_analysis
            if "results" in results:
                all_results = results["results"]
                summary_parts.append(f"**Total Wells Analyzed:** {len(all_results)}\n")
                
                # Count successful fits
                successful = sum(1 for r in all_results if r.get('fit_result', {}).get('success', False))
                summary_parts.append(f"**Successful Fits:** {successful}/{len(all_results)}\n")
                
                # Extract peak information
                peak_summaries = []
                for result in all_results[:10]:  # Limit to first 10 for summary
                    well_name = result.get('well_name', 'Unknown')
                    read = result.get('read', '')
                    fit_result = result.get('fit_result', {})
                    
                    if fit_result.get('success'):
                        r2 = fit_result.get('stats', {}).get('r2', 0)
                        peaks = fit_result.get('peaks', [])
                        peak_info = f"  - {well_name} (Read {read}): R²={r2:.3f}, {len(peaks)} peaks"
                        if peaks:
                            peak_centers = [p.get('center', 0) for p in peaks[:3]]
                            peak_info += f" at {[f'{c:.1f}nm' for c in peak_centers]}"
                        peak_summaries.append(peak_info)
                
                if peak_summaries:
                    summary_parts.append("**Sample Results:**\n" + "\n".join(peak_summaries) + "\n")
            else:
                # Try to extract from other formats
                summary_parts.append(f"**Results Summary:**\n{json.dumps(results, indent=2)[:1000]}...\n")
        
        return "\n".join(summary_parts) if summary_parts else "No detailed results available."
    
    def _summarize_gp_results(self, gp_results: Dict[str, Any]) -> str:
        """Create a summary of GP model results for LLM analysis."""
        summary_parts = []
        
        summary_parts.append(f"**Model Type:** {gp_results.get('model_type', 'Unknown')}\n")
        summary_parts.append(f"**Kernel:** {gp_results.get('kernel', 'Unknown')}\n")
        summary_parts.append(f"**Target Variable:** {gp_results.get('target', 'Unknown')}\n")
        summary_parts.append(f"**Cross-Validation R²:** {gp_results.get('cv_score', 0):.4f} ± {gp_results.get('cv_std', 0):.4f}\n")
        
        uncertainty_stats = gp_results.get('uncertainty_stats', {})
        if uncertainty_stats:
            summary_parts.append(f"**Uncertainty:** Mean={uncertainty_stats.get('mean', 0):.4f}, Max={uncertainty_stats.get('max', 0):.4f}\n")
        
        top_candidates = gp_results.get('top_candidates', [])
        if top_candidates:
            summary_parts.append(f"**Top Exploration Candidates:** {len(top_candidates)}\n")
            for i, candidate in enumerate(top_candidates[:3], 1):
                summary_parts.append(f"  {i}. {candidate.get('Candidate', 'Unknown')}: Predicted={candidate.get('Predicted Value', 'N/A')}, Uncertainty={candidate.get('Uncertainty', 'N/A')}\n")
        
        return "\n".join(summary_parts)
    
    def _summarize_monte_carlo_results(self, mc_results: Dict[str, Any]) -> str:
        """Create a summary of Monte Carlo Decision Tree results for LLM analysis."""
        summary_parts = []
        
        summary_parts.append(f"**Model Type:** Monte Carlo Decision Tree\n")
        
        # Optimization stats from stdout
        stats = mc_results.get("optimization_stats", {})
        if stats:
            summary_parts.append(f"**Total Cycles:** {stats.get('total_cycles', 'N/A')}\n")
            summary_parts.append(f"**Best Quality Achieved:** {stats.get('best_quality', 'N/A'):.2f}\n")
            summary_parts.append(f"**Total Improvement:** {stats.get('total_improvement_pct', 'N/A'):+.2f}%\n")
            summary_parts.append(f"**Average Improvement/Cycle:** {stats.get('avg_improvement_per_cycle_pct', 'N/A'):+.2f}%\n")
        
        # Top candidates
        top_candidates = mc_results.get("top_candidates", [])
        if top_candidates:
            summary_parts.append(f"**Top Exploration Candidates:** {len(top_candidates)}\n")
            for i, candidate in enumerate(top_candidates[:5], 1):
                candidate_str = candidate.get("candidate", "Unknown")
                # Add any quality/score metrics if available
                metrics = []
                for key, value in candidate.items():
                    if key not in ["rank", "candidate"] and isinstance(value, (int, float)):
                        metrics.append(f"{key}={value:.3f}")
                metrics_str = f" ({', '.join(metrics)})" if metrics else ""
                summary_parts.append(f"  {i}. {candidate_str}{metrics_str}\n")
        
        # Optimization history
        history = mc_results.get("optimization_history", {})
        if history:
            summary_parts.append(f"**Optimization History:** {history.get('total_experiments', 'N/A')} experiments")
            if history.get("latest_cycle"):
                summary_parts.append(f" (Latest Cycle: {history['latest_cycle']})")
            summary_parts.append("\n")
        
        # LLM Agent Restrictions/Constraints (if available)
        llm_agent = mc_results.get("llm_agent_results", {})
        if llm_agent:
            summary_parts.append(f"**LLM Agent Analysis (Parameter Restrictions/Constraints):**\n")
            
            # Patterns (learned restrictions)
            patterns = llm_agent.get("patterns", [])
            if patterns:
                summary_parts.append(f"**Learned Patterns ({len(patterns)}):**\n")
                for i, pattern in enumerate(patterns[:5], 1):
                    param = pattern.get("parameter", "Unknown")
                    obs = pattern.get("observation", "")
                    conf = pattern.get("confidence", 0)
                    summary_parts.append(f"  {i}. **{param}**: {obs} (confidence: {conf:.2f})\n")
            
            # Config suggestions (parameter restrictions/changes)
            suggestions = llm_agent.get("suggestions", [])
            if suggestions:
                summary_parts.append(f"**Parameter Restrictions/Changes Suggested ({len(suggestions)}):**\n")
                for i, sugg in enumerate(suggestions[:5], 1):
                    param_path = sugg.get("parameter_path", "Unknown")
                    current = sugg.get("current_value", "?")
                    suggested = sugg.get("suggested_value", "?")
                    conf = sugg.get("confidence", 0)
                    summary_parts.append(f"  {i}. **{param_path}**: {current} → {suggested} (confidence: {conf:.2f})\n")
            
            # Recommendation summary
            rec_summary = llm_agent.get("recommendation_summary")
            if rec_summary:
                summary_parts.append(f"**LLM Agent Recommendation Summary:**\n{rec_summary}\n")
            elif llm_agent.get("num_patterns") or llm_agent.get("num_suggestions"):
                summary_parts.append(f"LLM agent identified {llm_agent.get('num_patterns', 0)} patterns and "
                                   f"{llm_agent.get('num_suggestions', 0)} parameter restriction suggestions.\n")
        
        return "".join(summary_parts) if summary_parts else "Monte Carlo results available but no summary generated."
    
    def _analyze_results_with_llm(
        self,
        hypothesis_context: str,
        experimental_context: str,
        curve_fitting_summary: str,
        full_results: Optional[Dict[str, Any]] = None,
        gp_results: Optional[Dict[str, Any]] = None,
        monte_carlo_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze results, relate to hypothesis/experiments,
        decide on next steps, and provide literature-backed explanations.
        """
        socratic = _lazy_import_socratic()
        
        # Import analysis instructions
        from tools.instruct import ANALYSIS_INSTRUCTIONS
        
        # Summarize GP results if available
        gp_summary = self._summarize_gp_results(gp_results) if gp_results else "Not provided"
        
        # Summarize Monte Carlo results if available
        mc_summary = self._summarize_monte_carlo_results(monte_carlo_results) if monte_carlo_results else "Not provided"
        
        prompt = f"""
{ANALYSIS_INSTRUCTIONS}

## Hypothesis Context
{hypothesis_context}

## Experimental Context
{experimental_context}

## Curve Fitting Results Summary
{curve_fitting_summary}

## ML Model Results (Gaussian Process)
{gp_summary}

## ML Model Results (Monte Carlo Decision Tree)
{mc_summary}

## Full Results (if needed for detailed analysis)
{json.dumps(full_results, indent=2)[:2000] if full_results else "Not provided"}
"""
        
        try:
            analysis_response = socratic.generate_text_with_llm(prompt)
            return {
                "success": True,
                "analysis": analysis_response,
                "raw_response": analysis_response
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": None
            }
    
    def _parse_analysis_response(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse the LLM analysis response to extract structured information:
        - Hypothesis validation status
        - Need for more experiments
        - Literature explanations
        - Impact assessment
        """
        parsed = {
            "hypothesis_status": "unknown",  # "confirmed", "needs_revision", "rejected", "needs_more_data"
            "more_experiments_needed": False,
            "experiment_recommendations": [],
            "literature_explanations": "",
            "impact_assessment": "",
            "key_findings": []
        }
        
        # Simple parsing - look for key phrases
        analysis_lower = analysis_text.lower()
        
        # Check hypothesis status
        if "hypothesis is confirmed" in analysis_lower or "hypothesis is supported" in analysis_lower:
            parsed["hypothesis_status"] = "confirmed"
        elif "hypothesis needs revision" in analysis_lower or "hypothesis should be modified" in analysis_lower:
            parsed["hypothesis_status"] = "needs_revision"
        elif "hypothesis is rejected" in analysis_lower or "hypothesis is not supported" in analysis_lower:
            parsed["hypothesis_status"] = "rejected"
        elif "more data" in analysis_lower or "additional experiments" in analysis_lower:
            parsed["hypothesis_status"] = "needs_more_data"
        
        # Check if more experiments needed
        if "more experiments" in analysis_lower or "additional experiments" in analysis_lower:
            parsed["more_experiments_needed"] = True
        
        # Extract experiment recommendations (look for numbered lists or bullet points)
        import re
        exp_patterns = [
            r"experiment.*?:?\s*(.+?)(?:\n|$)",
            r"recommend.*?:?\s*(.+?)(?:\n|$)",
            r"should.*?:?\s*(.+?)(?:\n|$)"
        ]
        for pattern in exp_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.MULTILINE)
            if matches:
                parsed["experiment_recommendations"].extend(matches[:3])  # Limit to 3
        
        # Extract literature explanations (look for "literature" or "research" sections)
        lit_pattern = r"(?:literature|research|previous studies|according to).*?:(.+?)(?=\n\n|\n##|$)"
        lit_matches = re.findall(lit_pattern, analysis_text, re.IGNORECASE | re.DOTALL)
        if lit_matches:
            parsed["literature_explanations"] = lit_matches[0][:1000]  # Limit length
        
        # Extract impact assessment
        impact_pattern = r"(?:impact|implications|significance).*?:(.+?)(?=\n\n|\n##|$)"
        impact_matches = re.findall(impact_pattern, analysis_text, re.IGNORECASE | re.DOTALL)
        if impact_matches:
            parsed["impact_assessment"] = impact_matches[0][:1000]  # Limit length
        
        return parsed
    
    def _generate_new_question(
        self,
        research_goal: str,
        analysis_summary: str,
        current_hypothesis: str,
        completed_experiments: str,
    ) -> Optional[str]:
        """Generate a new research question using Hypothesis Agent approach."""
        socratic = _lazy_import_socratic()
        
        from tools.instruct import ANALYSIS_NEW_QUESTION_INSTRUCTIONS
        
        prompt = ANALYSIS_NEW_QUESTION_INSTRUCTIONS.format(
            research_goal=research_goal,
            analysis_summary=analysis_summary,
            completed_experiments=completed_experiments,
            current_hypothesis=current_hypothesis,
        )
        
        try:
            new_question = socratic.generate_text_with_llm(prompt)
            return new_question.strip() if new_question else None
        except Exception as e:
            st.error(f"Error generating new question: {e}")
            return None
    
    def run_agent(self, memory: MemoryManager) -> None:
        """Render UI and handle analysis agent interactions."""
        
        # Check for API key
        if not st.session_state.get("api_key"):
            st.warning("Please enter your API key in Settings before continuing.")
            st.stop()

        
        # Research Goal Input
        st.subheader("🎯 Research Goal")
        research_goal = st.text_area(
            "Enter your overall research goal:",
            value=st.session_state.get("research_goal", ""),
            height=100,
            help="This goal will guide the generation of new research questions and experiment recommendations.",
            key="research_goal_input"
        )
        st.session_state.research_goal = research_goal
        
        # Experiment Memory Display
        experiment_memory = get_experiment_memory()
        with st.expander("📚 Completed Experiments Memory", expanded=False):
            exp_summary = experiment_memory.get_experiment_summary()
            st.markdown(exp_summary)
            
            if st.button("🗑️ Clear Experiment Memory", use_container_width=True):
                experiment_memory.clear_memory()
                st.success("Experiment memory cleared!")
                st.rerun()
        
        # Get all context
        hypothesis_context = self._get_hypothesis_context()
        experimental_context = self._get_experimental_context()
        curve_fitting_results = self._get_curve_fitting_results()
        gp_results = self._get_gp_results()
        
        # Get Monte Carlo results
        monte_carlo_results = st.session_state.get("monte_carlo_results")
        
        # Display context
        with st.expander("📋 Context Information", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Hypothesis Context:**")
                st.text_area("", hypothesis_context, height=150, disabled=True, key="hyp_context_display")
            with col2:
                st.markdown("**Experimental Context:**")
                st.text_area("", experimental_context, height=150, disabled=True, key="exp_context_display")
        
        # Display GP results if available
        if gp_results:
            with st.expander("🤖 ML Model Results (Gaussian Process)", expanded=True):
                st.markdown(f"**Model Type:** {gp_results.get('model_type', 'Unknown')}")
                st.markdown(f"**Kernel:** {gp_results.get('kernel', 'Unknown')}")
                st.markdown(f"**Target Variable:** {gp_results.get('target', 'Unknown')}")
                st.markdown(f"**CV R² Score:** {gp_results.get('cv_score', 0):.4f} ± {gp_results.get('cv_std', 0):.4f}")
                
                if gp_results.get('top_candidates'):
                    st.markdown("**Top Exploration Candidates:**")
                    import pandas as pd
                    candidates_df = pd.DataFrame(gp_results['top_candidates'])
                    st.dataframe(candidates_df, use_container_width=True)
                
                uncertainty_stats = gp_results.get('uncertainty_stats', {})
                if uncertainty_stats:
                    st.markdown("**Uncertainty Statistics:**")
                    st.write(f"- Mean: {uncertainty_stats.get('mean', 0):.4f}")
                    st.write(f"- Std: {uncertainty_stats.get('std', 0):.4f}")
                    st.write(f"- Max: {uncertainty_stats.get('max', 0):.4f}")
        
        # Display Monte Carlo Decision Tree results if available
        monte_carlo_results = st.session_state.get("monte_carlo_results")
        if monte_carlo_results:
            with st.expander("🌳 ML Model Results (Monte Carlo Decision Tree)", expanded=True):
                stats = monte_carlo_results.get("optimization_stats", {})
                if stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Cycles", stats.get("total_cycles", "N/A"))
                    with col2:
                        st.metric("Best Quality", f"{stats.get('best_quality', 0):.2f}")
                    with col3:
                        st.metric("Total Improvement", f"{stats.get('total_improvement_pct', 0):+.2f}%")
                    with col4:
                        st.metric("Avg Improvement/Cycle", f"{stats.get('avg_improvement_per_cycle_pct', 0):+.2f}%")
                
                top_candidates = monte_carlo_results.get("top_candidates", [])
                if top_candidates:
                    st.markdown("**Top Exploration Candidates:**")
                    import pandas as pd
                    candidates_df = pd.DataFrame(top_candidates)
                    st.dataframe(candidates_df, use_container_width=True)
                
                history = monte_carlo_results.get("optimization_history", {})
                if history:
                    st.markdown(f"**Optimization History:** {history.get('total_experiments', 'N/A')} experiments")
                    if history.get("latest_cycle"):
                        st.markdown(f"Latest Cycle: {history['latest_cycle']}")
                
                # Display LLM Agent Restrictions/Constraints
                llm_agent = monte_carlo_results.get("llm_agent_results", {})
                if llm_agent:
                    st.markdown("---")
                    st.markdown("**🤖 LLM Agent Restrictions & Constraints**")
                    
                    patterns = llm_agent.get("patterns", [])
                    if patterns:
                        st.markdown(f"**Learned Patterns ({len(patterns)}):**")
                        for pattern in patterns[:5]:
                            param = pattern.get("parameter", "Unknown")
                            obs = pattern.get("observation", "")
                            conf = pattern.get("confidence", 0)
                            st.markdown(f"- **{param}**: {obs} _(confidence: {conf:.2f})_")
                    
                    suggestions = llm_agent.get("suggestions", [])
                    if suggestions:
                        st.markdown(f"**Parameter Restrictions Suggested ({len(suggestions)}):**")
                        import pandas as pd
                        sugg_df = pd.DataFrame([
                            {
                                "Parameter": s.get("parameter_path", "Unknown"),
                                "Current": str(s.get("current_value", "?")),
                                "Suggested": str(s.get("suggested_value", "?")),
                                "Confidence": f"{s.get('confidence', 0):.2f}"
                            }
                            for s in suggestions[:10]
                        ])
                        st.dataframe(sugg_df, use_container_width=True)
                    
                    rec_summary = llm_agent.get("recommendation_summary")
                    if rec_summary:
                        with st.expander("📋 LLM Agent Recommendation Summary", expanded=False):
                            st.markdown(rec_summary)
        
        if not curve_fitting_results:
            st.warning("⚠️ No curve fitting results found.")
            st.info("Please run the Curve Fitting Agent first to generate results for analysis.")
            st.info("Results should be in the `results/` directory as `*_peak_fit_results.json` files.")
            
            # Allow manual upload
            uploaded_file = st.file_uploader("Or upload a curve fitting results JSON file:", type=['json'])
            if uploaded_file:
                try:
                    curve_fitting_results = json.load(uploaded_file)
                    st.success("✅ Results file loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    st.stop()
            else:
                st.stop()
        
        # Summarize results
        curve_fitting_summary = self._summarize_curve_fitting_results(curve_fitting_results)
        
        st.subheader("Curve Fitting Results Summary")
        st.markdown(curve_fitting_summary)
        
        # Add GP results summary if available
        if gp_results:
            gp_summary = self._summarize_gp_results(gp_results)
            st.subheader("ML Model Results Summary (Gaussian Process)")
            st.markdown(gp_summary)
        
        # Add Monte Carlo Decision Tree results if available
        monte_carlo_results = st.session_state.get("monte_carlo_results")
        if monte_carlo_results:
            monte_carlo_summary = self._summarize_monte_carlo_results(monte_carlo_results)
            st.subheader("ML Model Results Summary (Monte Carlo Decision Tree)")
            st.markdown(monte_carlo_summary)
        
        # Analysis button
        if st.button("🔍 Analyze Results", type="primary", use_container_width=True):
            with st.spinner("Analyzing results with LLM..."):
                # Perform analysis
                analysis_result = self._analyze_results_with_llm(
                    hypothesis_context,
                    experimental_context,
                    curve_fitting_summary,
                    curve_fitting_results,
                    gp_results,
                    monte_carlo_results
                )
                
                if analysis_result["success"]:
                    # Parse the response
                    parsed = self._parse_analysis_response(analysis_result["analysis"])
                    
                    # Display full analysis
                    st.subheader("📊 Analysis Report")
                    st.markdown(analysis_result["analysis"])
                    
                    # Display structured insights
                    st.subheader("🔍 Key Insights")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hypothesis Status", parsed["hypothesis_status"].replace("_", " ").title())
                        st.metric("More Experiments Needed", "Yes" if parsed["more_experiments_needed"] else "No")
                    
                    with col2:
                        if parsed["experiment_recommendations"]:
                            st.markdown("**Experiment Recommendations:**")
                            for i, rec in enumerate(parsed["experiment_recommendations"], 1):
                                st.markdown(f"{i}. {rec}")
                    
                    # Literature explanations
                    if parsed["literature_explanations"]:
                        st.subheader("📚 Literature Context")
                        st.markdown(parsed["literature_explanations"])
                    
                    # Impact assessment
                    if parsed["impact_assessment"]:
                        st.subheader("💡 Impact Assessment")
                        st.markdown(parsed["impact_assessment"])
                    
                    # Save to memory
                    self.memory.insert_interaction(
                        "assistant",
                        analysis_result["analysis"],
                        "analysis_report",
                        "analysis"
                    )
                    
                    # Save structured data
                    st.session_state.analysis_results = {
                        "parsed": parsed,
                        "full_analysis": analysis_result["analysis"],
                        "timestamp": st.session_state.get("start_time", 0)
                    }
                    
                    st.success("✅ Analysis complete!")
                    
                    # Decision point: More experiments needed?
                    if parsed["more_experiments_needed"]:
                        st.info("🔄 The analysis suggests more experiments are needed.")
                        if st.button("🚀 Return to Experiment Agent", use_container_width=True):
                            st.session_state.next_agent = "experiment"
                            st.session_state.analysis_recommendations = parsed["experiment_recommendations"]
                            st.rerun()
                    
                    # Decision point: Hypothesis needs revision?
                    if parsed["hypothesis_status"] in ["needs_revision", "rejected"]:
                        st.info("📝 The analysis suggests the hypothesis may need revision.")
                        if st.button("🔬 Return to Hypothesis Agent", use_container_width=True):
                            st.session_state.next_agent = "hypothesis"
                            st.session_state.analysis_feedback = parsed
                            st.rerun()
                else:
                    st.error(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}")
        
        # Display previous analysis if available
        if st.session_state.get("analysis_results"):
            with st.expander("📜 Previous Analysis", expanded=False):
                prev_analysis = st.session_state.analysis_results.get("full_analysis", "")
                st.markdown(prev_analysis)
