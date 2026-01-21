from typing import Dict, Any

import streamlit as st
from agents.base import BaseAgent
from tools.memory import MemoryManager

# Lazy import socratic module - only import when needed
_socratic_module = None

def _lazy_import_socratic():
    """Lazy import of socratic module to speed up module loading"""
    global _socratic_module
    if _socratic_module is None:
        from tools import socratic
        _socratic_module = socratic
    return _socratic_module

class HypothesisAgent(BaseAgent):
    def __init__(self, name, desc, question: str):
        super().__init__("Hypothesis Agent", desc)
        self.question = question
        self.memory = MemoryManager()
        # Don't import socratic here - lazy load when needed

    def confidence(self, payload: Dict[str, Any]) -> float:
        pass

    def initial_process(self, question, experimental_mode=False, experimental_constraints=None):
        try:
            # Lazy import socratic module
            socratic = _lazy_import_socratic()
            
            if not st.session_state.api_key:
                st.warning("Please enter your API key in Settings before continuing.")
                st.info("**Make sure you're using a Google Gemini API key, not an OpenAI key.**")
                st.info("Get your Gemini API key from: https://makersuite.google.com/app/apikey")
                st.stop()

            # Validate question input
            if not question or not question.strip():
                st.error("Please provide a valid question to explore.")
                st.stop()

            # Generate clarified question with better error handling
            try:
                clarified_question = socratic.clarify_question(question)
            except ValueError as e:
                # API key error
                st.error(f"API Key Issue: {str(e)}. Please set API key in Settings or check environment variables.")
                st.stop()
            except Exception as e:
                st.error(f"Error generating clarified question: {str(e)}. Please check your API key and try again.")
                st.stop()

            if not clarified_question or not clarified_question.strip():
                st.error("Could not generate clarified question. The LLM returned an empty response.")
                st.warning("**Possible causes:**")
                st.warning("1. Invalid or expired Google Gemini API key")
                st.warning("2. Using OpenAI API key instead of Gemini key")
                st.warning("3. API quota exceeded")
                st.warning("4. Network connectivity issues")
                st.info("Get your Gemini API key from: https://makersuite.google.com/app/apikey")
                st.stop()

            # Generate socratic questions
            try:
                socratic_questions = socratic.socratic_pass(clarified_question)
            except Exception as e:
                st.error(f"Error generating socratic questions: {str(e)}. Please try again.")
                st.stop()
                
            if not socratic_questions or not socratic_questions.strip():
                st.error("Could not generate socratic questions. Please try again!")
                st.stop()

            if socratic_questions:
                socratic_answers = None
                try:
                    socratic_answers = socratic.socratic_answer_questions(clarified_question, socratic_questions)
                except Exception as e:
                    st.error(f"EXCEPTION in socratic answers: {e}")
                    st.stop()

                if experimental_mode and experimental_constraints:
                    thoughts = socratic.tot_generation_experimental_plan(
                    socratic_questions, clarified_question, experimental_constraints)
                else:
                    # Calling TOT with socratic answers (if available)
                    thoughts = socratic.tot_generation(socratic_questions, clarified_question, socratic_answers)
                    if not thoughts:
                        st.error("TOT Generation returned None or empty. Try again!")
                        st.stop()
            else:
                st.error("Could not generate socratic questions. Please try again!")
                st.stop()

            if len(thoughts) < 3:
                thoughts = list(thoughts) + [""] * (3-len(thoughts))

            return clarified_question, socratic_questions, thoughts[:3], socratic_answers

        except ValueError as e:
            # API Key related error
            st.error(f"Error: API Key Issue; Please set API key in Settings or check environment variables.")
            st.stop()
        except Exception as e:
            st.error(f"Error occurred: {e}; Please try again!")
            st.stop()

    def build_conversation_context(self, prompt_session_id: str = st.session_state.current_prompt_session_id):
        """ Build full conversation context from all interactions """
        context_parts = []

        # Get initial question
        initial_q = self.memory.view_component("initial_question")
        if initial_q:
            context_parts.append(f"Initial Question: {initial_q}")

        # Get clarified question
        clarified = self.memory.view_component("clarified_question")
        if clarified:
            context_parts.append(f"Clarified Question: {clarified}")

        # Get socratic pass
        socratic_pass = self.memory.view_component("socratic_pass")
        if socratic_pass:
            context_parts.append(f"Socratic Analysis: {socratic_pass}")

        # Get all thoughts
        thought1 = self.memory.view_component("first_thought_1")
        thought2 = self.memory.view_component("second_thought_1")
        thought3 = self.memory.view_component("third_thought_1")
        if thought1 or thought2 or thought3:
            thoughts = [t for t in [thought1, thought2, thought3] if t]
            context_parts.append(f"Initial Thoughts: {'; '.join(thoughts)}")

        # Get all selected options and their responses
        selected_options = []
        for i in st.session_state.interactions:
            if i.get("component") == "option_choice":
                selected_options.append(f"Selected: {i.get('message')}")
            elif i.get("component") == "next_step_option_1":
                selected_options.append(f"Option 1: {i.get('message')}")
            elif i.get("component") == "next_step_option_2":
                selected_options.append(f"Option 2: {i.get('message')}")
            elif i.get("component") == "next_step_option_3":
                selected_options.append(f"Option 3: {i.get('message')}")
            elif i.get("component") == "additional_question":
                selected_options.append(f"Additional Question: {i.get('message')}")

        if selected_options:
            context_parts.append(f"Conversation Flow: {'; '.join(selected_options[-10:])}")  # Last 10 interactions

        # Get socratic questions from iterations
        retry_q = self.memory.view_component("retry_thinking_question")
        if retry_q:
            context_parts.append(f"Latest Socratic Question: {retry_q}")

        return "\n\n".join(context_parts) if context_parts else "No previous context available."

    def get_context_for_followup(self):
        """ Get context from previous conversations for follow-up questions """
        if st.session_state.conversation_events:
            latest = self.memory.get_latest_history()
            payload = latest.get("payload", {})
            context = f"Previous question: {payload['question']}\n"

            if payload["thoughts"]:
                context += f"Previous thoughts: {latest['thoughts'][:200]}...\n"
            if payload["hypothesis"]:
                context += f"Previous hypothesis: {payload['hypothesis']}\n"

            return context
        else:
            return ""

    def generate_hypothesis_with_context(self, socratic_question, next_step_option, previous_option_1,
                                         previous_option_2, conversation_context):
        """ Generate hypothesis with full conversation context """
        try:
            socratic = _lazy_import_socratic()
            return socratic.hypothesis_synthesis(
                socratic_question, 
                next_step_option, 
                previous_option_1,
                previous_option_2,
                conversation_context
            )
        except Exception as e:
            st.error(f"Error generating hypothesis: {str(e)}. Please check your API key and try again.")
            st.stop()

    def run_agent(self, memory):
        # If stop button is pressed, jump straight to hypothesis
        if st.session_state.stop_hypothesis and st.session_state.stage != "analysis":
            with st.chat_message("assistant"):
                with st.spinner("Synthesizing hypothesis from current context..."):
                    try:
                        # Extract available data with fallbacks
                        soc_q = self.memory.view_component("retry_thinking_question") or self.memory.view_component(
                            "clarified_question") or "How can we continue exploring this hypothesis?"
                        picked = self.memory.view_component("next_step_option_1") or self.memory.view_component(
                            "first_thought_1") or self.memory.view_component("last_selected_option") or "Selected option"
                        prev1 = self.memory.view_component("next_step_option_2") or self.memory.view_component(
                            "second_thought_1") or self.memory.view_component("last_prev1") or "Previous option 1"
                        prev2 = self.memory.view_component("next_step_option_3") or self.memory.view_component(
                            "third_thought_1") or self.memory.view_component("last_prev2") or "Previous option 2"

                        # Ensure we have valid values
                        if not soc_q or not picked:
                            st.error(
                                "Insufficient context to generate hypothesis. Please go through the conversation flow first.")
                            st.session_state.stop_hypothesis = False
                            st.rerun()

                        # Build conversation context
                        context = self.build_conversation_context()
                        
                        # Generate hypothesis with conversation context
                        socratic = _lazy_import_socratic()
                        hypothesis = socratic.hypothesis_synthesis(soc_q, picked, prev1, prev2, context)

                        # Ensure hypothesis is never None
                        if hypothesis is None or not str(hypothesis).strip():
                            st.error("Error generating hypothesis. Please check your API key and try again.")
                            st.rerun()

                        st.markdown("**Hypothesis:**")
                        st.markdown(hypothesis)
                        
                        # Generate analysis report
                        with st.spinner("Generating analysis report..."):
                            socratic_question_for_analysis = self.memory.view_component("retry_thinking_question") or self.memory.view_component("socratic_pass") or soc_q
                            analysis_rubric = socratic.local_hypothesis_analysis_fallback(hypothesis, socratic_question_for_analysis)
                            
                            if analysis_rubric and str(analysis_rubric).strip():
                                st.markdown("**Analysis Report:**")
                                st.markdown(analysis_rubric)
                                self.memory.insert_interaction("assistant", analysis_rubric, "analysis_rubric", "hypothesis")

                    except Exception as e:
                        st.error(f"Error generating hypothesis: {str(e)}. Please check your API key and try again.")
                        st.rerun()

                st.success("🎉 Hypothesis generation complete (forced stop).")

            self.memory.insert_interaction("assistant", hypothesis, "hypothesis", "hypothesis")
            st.session_state.last_hypothesis = hypothesis
            st.session_state.hypothesis_ready = True
            st.session_state.stop_hypothesis = False
            st.session_state.stage = "analysis"
            st.rerun()

        # Normal stages
        if st.session_state.stage == "initial":
            st.write("Welcome to the hypothesis agent! Please enter a question that you would like to explore further.")

            question = st.chat_input("Ask a question...")

            if question:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        st.write("DEBUG: Calling initial_process...")

                        cl_question, soc_pass, thoughts_gen, soc_answers = self.initial_process(
                            question)

                        # Safe unpacking - always get exactly 3 items
                        first_thought = thoughts_gen[0] if len(thoughts_gen) > 0 else "Option 1: Continue exploring"
                        second_thought = thoughts_gen[1] if len(thoughts_gen) > 1 else "Option 2: Continue exploring"
                        third_thought = thoughts_gen[2] if len(thoughts_gen) > 2 else "Option 3: Continue exploring"

                        # Validate that thoughts are not empty
                        if not first_thought or not first_thought.strip():
                            first_thought = "Option 1: Continue exploring"
                        if not second_thought or not second_thought.strip():
                            second_thought = "Option 2: Continue exploring"
                        if not third_thought or not third_thought.strip():
                            third_thought = "Option 3: Continue exploring"

                        # Display everything inside the chat messages
                        st.markdown("**Clarified Question:**")
                        st.markdown(cl_question)

                        st.markdown("**Socratic Pass (Probing Questions):**")
                        st.markdown(soc_pass)

                        st.markdown("**Socratic Reasoning (LLM Answers to Its Own Question):**")
                        st.markdown(soc_answers)

                        st.markdown("**Generated Thoughts:**")
                        st.markdown(f"**1.** {first_thought}")
                        st.markdown(f"**2.** {second_thought}")
                        st.markdown(f"**3.** {third_thought}")

                        # Save interactions to both conversation_events and st.session_state.interactions
                        self.memory.insert_interaction("user", question, "initial_question", "hypothesis")
                        self.memory.insert_interaction("assistant", cl_question, "clarified_question", "hypothesis")
                        self.memory.insert_interaction("assistant", soc_pass, "socratic_pass", "hypothesis")
                        # CRITICAL: Save socratic answers so they persist after rerun
                        if soc_answers and soc_answers.strip():
                            self.memory.insert_interaction("assistant", soc_answers, "socratic_answers", "hypothesis")
                        self.memory.insert_interaction("assistant", first_thought, "first_thought_1", "hypothesis")
                        self.memory.insert_interaction("assistant", second_thought, "second_thought_1", "hypothesis")
                        self.memory.insert_interaction("assistant", third_thought, "third_thought_1", "hypothesis")

                        # Initialize round count if starting hypothesis stage
                        if "hypothesis_round_count" not in st.session_state:
                            st.session_state.hypothesis_round_count = 0
                        
                        st.session_state.stage = "refine"
                        st.rerun()

        elif st.session_state.stage == "refine":
            st.write("You are presented with three lines of distinct thoughts. Please choose the option that explores your initial question best.")
            user_choice = st.chat_input("Make a choice 1, 2, or 3...")

            if user_choice:
                if user_choice not in ["1", "2", "3"]:
                    st.warning("Please enter 1, 2, or 3.")
                    st.stop()

                with st.chat_message("user"):
                    st.markdown(user_choice)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if user_choice == "1":
                            picked = self.memory.view_component("first_thought_1")
                            prev1 = self.memory.view_component("second_thought_1")
                            prev2 = self.memory.view_component("third_thought_1")
                        elif user_choice == "2":
                            picked = self.memory.view_component("second_thought_1")
                            prev1 = self.memory.view_component("first_thought_1")
                            prev2 = self.memory.view_component("third_thought_1")
                        else:
                            picked = self.memory.view_component("third_thought_1")
                            prev1 = self.memory.view_component("first_thought_1")
                            prev2 = self.memory.view_component("second_thought_1")

                        # Show brief analysis of the selected option (not a full hypothesis report)
                        # This is just to help the user understand what they selected
                        st.markdown("**Selected Option:**")
                        st.markdown(picked)

                        st.markdown("---")

                        # Lazy import socratic module
                        socratic = _lazy_import_socratic()
                        clarified_question = self.memory.view_component("clarified_question")
                        if not clarified_question:
                            clarified_question = self.memory.view_component("initial_question") or "How can we continue exploring this?"
                        
                        # Build conversation context
                        context = self.build_conversation_context()
                        
                        result = socratic.retry_thinking_deepen_thoughts(
                            picked, prev1, prev2, clarified_question, context
                        )

                        if result is None or len(result) != 2:
                            soc_q = "How can we continue exploring this hypothesis?"
                            options = ["Why is this approach theoretically sound?",
                                       "What are the mechanistic advantages?",
                                       "How does this compare to alternatives?"]
                        else:
                            soc_q, options = result
                            # Ensure options is a list with exactly 3 items, filtering out None/empty
                            if not isinstance(options, list):
                                options = [str(o) for o in options] if options else []
                            # Filter out None, empty strings, and "None" strings
                            options = [opt for opt in options if
                                       opt and str(opt).strip() and str(opt).strip().lower() != "none"]
                            if len(options) < 3:
                                options = list(options) + [""] * (3 - len(options))
                            elif len(options) > 3:
                                options = options[:3]  # Take only first 3
                            # Ensure no None values
                            options = [
                                opt if opt and str(opt).strip() != "None" else f"Option {i + 1}: Continue exploring" for
                                i, opt in enumerate(options)]

                            # Ensure we only display exactly 3 options, and filter out None/empty values
                            valid_options = [opt for opt in options[:3] if
                                             opt and str(opt).strip() and str(opt).strip().lower() != "none"]
                            # Pad to 3 if needed
                            while len(valid_options) < 3:
                                valid_options.append(
                                    f"Option {len(valid_options) + 1}: Continue exploring this line of reasoning")
                            # Take only first 3
                            valid_options = valid_options[:3]

                            st.markdown("**Continuation Question:**")
                            st.markdown(
                                soc_q if soc_q and soc_q != "None" else "How can we continue exploring this hypothesis?")
                            st.markdown("**Next-Step Options:**")
                            for i, opt in enumerate(valid_options, 1):
                                opt_str = str(opt).strip()
                                if opt_str and opt_str.lower() != "none":
                                    st.markdown(f"{i}. {opt_str}")
                                else:
                                    st.markdown(f"{i}. Option {i}: Continue exploring this line of reasoning")

                        self.memory.insert_interaction("user", user_choice, "tot_choice", "hypothesis")
                        self.memory.insert_interaction("assistant", soc_q if soc_q and str(
                            soc_q).strip().lower() != "none" else "How can we continue exploring this hypothesis?",
                                           "retry_thinking_question", "hypothesis")
                        # Ensure we only save 3 options, using valid_options (defined above)
                        valid_options_to_save = valid_options[:3] if len(valid_options) >= 3 else valid_options + [
                            f"Option {len(valid_options) + i + 1}: Continue exploring" for i in
                            range(3 - len(valid_options))]
                        self.memory.insert_interaction("assistant", valid_options_to_save[0] if len(
                            valid_options_to_save) > 0 else "Option 1: Continue exploring", "next_step_option_1", "hypothesis")
                        self.memory.insert_interaction("assistant", valid_options_to_save[1] if len(
                            valid_options_to_save) > 1 else "Option 2: Continue exploring", "next_step_option_2", "hypothesis")
                        self.memory.insert_interaction("assistant", valid_options_to_save[2] if len(
                            valid_options_to_save) > 2 else "Option 3: Continue exploring", "next_step_option_3", "hypothesis")
                        
                        # Transition to hypothesis stage and initialize round count
                        if "hypothesis_round_count" not in st.session_state:
                            st.session_state.hypothesis_round_count = 0
                        
                        st.session_state.stage = "hypothesis"
                        st.rerun()


        elif st.session_state.stage == "hypothesis":

            if st.session_state.experimental_mode:
                st.session_state.stage = "refine"
                st.rerun()

            # Check round count and max rounds
            round_count = st.session_state.get("hypothesis_round_count", 0)
            max_rounds = st.session_state.get("max_hypothesis_rounds", 5)
            
            # If we've reached max rounds, suggest generating hypothesis
            if round_count >= max_rounds:
                st.warning(f"⚠️ Maximum exploration rounds ({max_rounds}) reached. Consider generating your hypothesis now.")
                if st.button("Generate Hypothesis Now", type="primary", use_container_width=True):
                    st.session_state.stop_hypothesis = True
                    st.rerun()

            st.write("**Standard Hypothesis Agent - Iterative Refinement Mode**")
            st.write("You can:")
            st.write("1. **Choose an option (1, 2, or 3)** → generates 3 new continuation options")
            st.write("2. **Ask an additional question** → triggers socratic questioning and TOT thinking")
            st.write("3. **Click 'Generate Hypothesis'** → synthesizes your hypothesis from all conversation")
            st.write(f"**Current Round:** {round_count}/{max_rounds}")

            # Load current options
            opt1 = self.memory.view_component("next_step_option_1")
            opt2 = self.memory.view_component("next_step_option_2")
            opt3 = self.memory.view_component("next_step_option_3")

            with st.expander("Current Next-Step Options", expanded=True):
                st.markdown(f"**1.** {opt1}")
                st.markdown(f"**2.** {opt2}")
                st.markdown(f"**3.** {opt3}")

            # Unified input
            user_input = st.chat_input("Pick option 1, 2, or 3, or ask an additional question...")

            # Additional question (explicit)
            with st.expander("💬 Ask Additional Question", expanded=False):
                additional_question = st.text_input(
                    "Ask a question to refine your thinking:",
                    key="additional_question_input"
                )

                if st.button("Ask Question"):
                    if additional_question.strip():
                        st.session_state.pending_additional_question = additional_question
                        st.rerun()

            # Option selection
            if user_input in ["1", "2", "3"]:
                # Increment round count
                round_count = st.session_state.get("hypothesis_round_count", 0)
                max_rounds = st.session_state.get("max_hypothesis_rounds", 5)
                
                # Check if we've exceeded max rounds
                if round_count >= max_rounds:
                    st.warning(f"Maximum rounds ({max_rounds}) reached. Generating hypothesis automatically...")
                    st.session_state.stop_hypothesis = True
                    st.rerun()
                
                picked = {"1": opt1, "2": opt2, "3": opt3}[user_input]
                prev = [opt1, opt2, opt3]
                prev.remove(picked)

                self.memory.insert_interaction("user", user_input, "option_choice", "hypothesis")
                self.memory.insert_interaction("assistant", picked, "last_selected_option", "hypothesis")
                self.memory.insert_interaction("assistant", prev[0], "last_prev1", "hypothesis")
                self.memory.insert_interaction("assistant", prev[1], "last_prev2", "hypothesis")

                with st.chat_message("assistant"):
                    with st.spinner("Generating continuation options..."):
                        socratic = _lazy_import_socratic()
                        context = self.build_conversation_context()
                        
                        # Check if hypothesis is ready using LLM
                        clarified_q = self.memory.view_component("clarified_question") or ""
                        socratic_q = self.memory.view_component("socratic_pass") or ""
                        previous_opts = f"{prev[0]}; {prev[1]}"
                        
                        # Import instruction for readiness check
                        from tools.instruct import HYPOTHESIS_READINESS_CHECK
                        readiness_prompt = HYPOTHESIS_READINESS_CHECK.format(
                            clarified_question=clarified_q[:500],
                            socratic_questions=socratic_q[:500],
                            round_count=round_count + 1,
                            selected_option=picked[:300],
                            previous_options=previous_opts[:300]
                        )
                        
                        try:
                            readiness_response = socratic.generate_text_with_llm(readiness_prompt).strip().upper()
                            is_ready = "READY" in readiness_response
                        except Exception as e:
                            # If LLM check fails, use round count as fallback
                            is_ready = (round_count + 1) >= max_rounds
                        
                        # If ready, generate hypothesis instead of new options
                        if is_ready:
                            st.info("🤖 LLM determined sufficient information gathered. Generating hypothesis...")
                            st.session_state.stop_hypothesis = True
                            st.rerun()
                        
                        _, new_opts = socratic.retry_thinking_deepen_thoughts(
                            picked, prev[0], prev[1],
                            self.memory.view_component("clarified_question"),
                            context
                        )

                        for i, opt in enumerate(new_opts[:3], 1):
                            self.memory.insert_interaction("assistant", opt, f"next_step_option_{i}", "hypothesis")
                            st.markdown(f"**{i}.** {opt}")

                # Increment round count
                st.session_state.hypothesis_round_count = round_count + 1
                st.session_state.stage = "hypothesis"
                st.rerun()

            # Additional question flow
            if st.session_state.get("pending_additional_question"):
                q = st.session_state.pop("pending_additional_question")
                with st.chat_message("user"):
                    st.markdown(q)

                socratic = _lazy_import_socratic()
                context = self.build_conversation_context()
                clarified = socratic.clarify_question(q)
                questions = socratic.socratic_pass(clarified)
                thoughts = socratic.tot_generation(questions, clarified)

                self.memory.insert_interaction("assistant", clarified, "clarified_question", "hypothesis")
                self.memory.insert_interaction("assistant", questions, "socratic_pass", "hypothesis")

                for i, t in enumerate(thoughts[:3], 1):
                    self.memory.insert_interaction("assistant", t, f"next_step_option_{i}", "hypothesis")

                st.session_state.stage = "hypothesis"
                st.rerun()

            # Generate final hypothesis

            if st.button("Generate Hypothesis", type="primary", use_container_width=True):
                socratic = _lazy_import_socratic()
                context = self.build_conversation_context()
                
                # Get the selected option and previous options
                socratic_q = self.memory.view_component("retry_thinking_question") or self.memory.view_component("socratic_pass") or "How can we test this hypothesis?"
                selected_option = self.memory.view_component("last_selected_option") or opt1
                prev1 = self.memory.view_component("last_prev1") or opt2
                prev2 = self.memory.view_component("last_prev2") or opt3
                
                # If no selected option, use the first option
                if not selected_option or selected_option == opt1:
                    selected_option = opt1
                    prev1 = opt2
                    prev2 = opt3
                
                with st.chat_message("assistant"):
                    with st.spinner("Synthesizing hypothesis from conversation..."):
                        hypothesis = self.generate_hypothesis_with_context(
                            socratic_q,
                            selected_option,
                            prev1,
                            prev2,
                            context
                        )

                        if not hypothesis or not str(hypothesis).strip():
                            st.error("Error generating hypothesis. Please check your API key and try again.")
                            st.stop()

                        st.markdown("**Hypothesis:**")
                        st.markdown(hypothesis)
                        
                        # Generate analysis report
                        with st.spinner("Generating analysis report..."):
                            socratic_question_for_analysis = self.memory.view_component("retry_thinking_question") or self.memory.view_component("socratic_pass") or socratic_q
                            analysis_rubric = socratic.local_hypothesis_analysis_fallback(hypothesis, socratic_question_for_analysis)
                            
                            if analysis_rubric and str(analysis_rubric).strip():
                                st.markdown("**Analysis Report:**")
                                st.markdown(analysis_rubric)
                                self.memory.insert_interaction("assistant", analysis_rubric, "analysis_rubric", "hypothesis")

                self.memory.insert_interaction("assistant", hypothesis, "hypothesis", "hypothesis")
                st.session_state.last_hypothesis = hypothesis
                st.session_state.hypothesis_ready = True
                st.session_state.stage = "analysis"
                st.rerun()

        elif st.session_state.stage == "analysis":
            # Transitions to Experiment agent
            if st.button("Generate Experimental Plan"):
                st.session_state.next_agent = "experiment"
                st.rerun()

            # Experimental mode should never reach analysis
            if st.session_state.experimental_mode:
                st.session_state.stage = "experimental_outputs"
                st.rerun()

            socratic = _lazy_import_socratic()
            hypothesis = self.memory.view_component("hypothesis")
            socratic_question = (
                    self.memory.view_component("retry_thinking_question")
                    or self.memory.view_component("socratic_pass")
                    or ""
            )

            if not hypothesis:
                st.error("No hypothesis found. Please generate a hypothesis first.")
                st.session_state.stage = "hypothesis"
                st.rerun()

            with st.chat_message("assistant"):
                with st.spinner("Analyzing Hypothesis and Producing Report..."):
                    analysis_rubric = socratic.local_hypothesis_analysis_fallback(
                        hypothesis, socratic_question)

                    if not analysis_rubric or not str(analysis_rubric).strip():
                        analysis_rubric = (
                            "Analysis generated but content is empty. "
                            "Please check your API key and try again.")

                st.markdown(analysis_rubric)

            self.memory.insert_interaction("assistant", analysis_rubric, "analysis_rubric", "hypothesis")
            st.success("Analysis complete!")

        elif st.session_state.stage == "followup":

            st.header("Follow-up Question")

            # Show previous context
            if st.session_state.conversation_history:
                latest = st.session_state.conversation_history[-1]
                with st.expander("Previous Conversation Context", expanded=True):
                    st.markdown(f"**Previous Question:** {latest['question']}")
                    if latest.get("hypothesis"):
                        hyp = latest["hypothesis"]
                        st.markdown(
                            f"**Previous Hypothesis:** {hyp[:300]}..."
                            if len(hyp) > 300 else f"**Previous Hypothesis:** {hyp}"
                        )

            followup_question = st.text_input(
                "Ask a follow-up question based on the previous hypothesis:",
                placeholder="e.g., 'What experimental methods would validate this hypothesis?'"
            )

            if st.button("🚀 Process Follow-up", type="primary"):
                if not followup_question.strip():
                    st.warning("Please enter a follow-up question.")
                    st.stop()

                socratic = _lazy_import_socratic()
                context = self.get_context_for_followup()
                contextual_question = f"{context}\n\nFollow-up question: {followup_question}"

                with st.spinner("Processing follow-up question..."):
                    clarified = socratic.clarify_question(contextual_question)
                    questions = socratic.socratic_pass(clarified)
                    answers = socratic.socratic_answer_questions(clarified, questions)
                    thoughts = socratic.tot_generation(questions, clarified, answers)

                    thoughts = (thoughts or [])[:3]
                    while len(thoughts) < 3:
                        thoughts.append("")

                # Display
                with st.chat_message("user"):
                    st.markdown(followup_question)

                with st.chat_message("assistant"):
                    st.markdown("**Clarified Question:**")
                    st.markdown(clarified)

                    st.markdown("**Socratic Pass:**")
                    st.markdown(questions)

                    if answers:
                        st.markdown("**Socratic Reasoning:**")
                        st.markdown(answers)

                    st.markdown("**Generated Thoughts:**")
                    for t in thoughts:
                        st.markdown(t)

                # Persist
                self.memory.insert_interaction("user", followup_question, "original_question", "hypothesis")
                self.memory.insert_interaction("assistant", clarified, "clarified_question", "hypothesis")
                self.memory.insert_interaction("assistant", questions, "socratic_pass", "hypothesis")
                if answers:
                    self.memory.insert_interaction("assistant", answers, "socratic_answers", "hypothesis")

                for i, t in enumerate(thoughts, 1):
                    self.memory.insert_interaction("assistant", t, f"next_step_option_{i}", "hypothesis")

                # Re-enter refinement loop
                st.session_state.stage = "refine"
                st.rerun()

