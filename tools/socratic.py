import os
import logging
import re

# Lazy import heavy dependencies - only load when actually needed
_genai = None
_edison_client = None
_edison_models = None
_given_vars = None

# Initialize API key from environment - will be updated by streamlit_app if needed
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
FUTUREHOUSE_API_KEY = os.getenv("FUTUREHOUSE_API_KEY")
GOOGLE_MODEL_ID = "gemini-2.5-flash-lite"

# Log API key status on import (first 10 chars only) - but only if API key exists
if GOOGLE_API_KEY:
    logging.info(f"API key loaded from environment on module import (starts with: {GOOGLE_API_KEY[:10]}...)")
else:
    logging.warning("No API key found in environment on module import. Will check again at runtime.")

def _lazy_import_genai():
    """
    Lazy import of google.generativeai to speed up module loading.
    
    Note: google.generativeai is deprecated in favor of google.genai.
    This code still works but will need migration in the future.
    See: https://github.com/google-gemini/deprecated-generative-ai-python
    """
    global _genai
    if _genai is None:
        import warnings
        # Suppress deprecation warning for now (package still works)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import google.generativeai as genai  # type: ignore
        _genai = genai
    return _genai


def _lazy_import_edison():
    """Lazy import of edison_client to speed up module loading"""
    global _edison_client, _edison_models
    if _edison_client is None:
        from edison_client import EdisonClient, JobNames
        from edison_client.models import TaskRequest, RuntimeConfig
        _edison_client = (EdisonClient, JobNames)
        _edison_models = (TaskRequest, RuntimeConfig)
    return _edison_client[0], _edison_client[1], _edison_models[0], _edison_models[1]


def _lazy_import_instructions():
    """Lazy import of instruction constants"""
    from tools.instruct import (CLARIFY_QUESTION_INSTRUCTIONS, SOCRATIC_PASS_INSTRUCTIONS, SOCRATIC_ANSWER_INSTRUCTIONS,
                                TOT_INSTRUCTIONS, RETRY_THINKING_INSTRUCTIONS, HYPOTHESIS_SYNTHESIS,
                                HYPOTHESIS_ANALYSIS_REPORT,
                                EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS)
def _lazy_import_instructions():
    """Lazy import of instruction constants"""
    from tools.instruct import (CLARIFY_QUESTION_INSTRUCTIONS, SOCRATIC_PASS_INSTRUCTIONS, SOCRATIC_ANSWER_INSTRUCTIONS,
                               TOT_INSTRUCTIONS, RETRY_THINKING_INSTRUCTIONS, HYPOTHESIS_SYNTHESIS, HYPOTHESIS_ANALYSIS_REPORT,
                               EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS)
    return (CLARIFY_QUESTION_INSTRUCTIONS, SOCRATIC_PASS_INSTRUCTIONS, SOCRATIC_ANSWER_INSTRUCTIONS, TOT_INSTRUCTIONS,
            RETRY_THINKING_INSTRUCTIONS, HYPOTHESIS_SYNTHESIS, HYPOTHESIS_ANALYSIS_REPORT,
            EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS)


def _lazy_import_given_vars():
    """Lazy import of given_variables module"""
    global _given_vars
    if _given_vars is None:
        import tools.given_variables as given_vars
        _given_vars = given_vars
    return _given_vars


def generate_text_with_llm(prompt: str) -> str:
    """Generating text using Gemini provided model and API Key"""
    try:
        # Get the latest API key - check in order: environment variables (most reliable), session state, cached key, module variable
        # Environment variables are most reliable for cross-page persistence
        api_key = None

        # First check environment variables (most reliable for persistence)
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Also try loading from .env file if available
        if not api_key:
            try:
                from dotenv import load_dotenv
                from pathlib import Path
                # Try loading from project root
                env_path = Path(__file__).parent.parent / '.env'
                load_dotenv(env_path)
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            except (ImportError, Exception):
                pass  # dotenv not available or file not found, continue

        # Then check session state (may have been updated in UI)
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, 'session_state') and hasattr(st.session_state, 'api_key'):
                    api_key = st.session_state.get('api_key')
            except (RuntimeError, AttributeError):
                # Not in Streamlit context, continue to other checks
                pass

        # If still no key, check fallback sources
        if not api_key:
            # Check cached key first (updated by UI)
            if hasattr(generate_text_with_llm, '_cached_api_key'):
                api_key = generate_text_with_llm._cached_api_key

            # Check module-level variable (may have been updated)
            if not api_key:
                api_key = GOOGLE_API_KEY

            # Check module-level variable (last resort)
            if not api_key:
                api_key = GOOGLE_API_KEY

        if not api_key:
            error_msg = "API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, or enter it in the UI (Options)."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Log model and key status (first 10 chars only for security)
        logging.info(f"Using model: {GOOGLE_MODEL_ID}, API key present: {bool(api_key)} (starts with: {api_key[:10] if api_key and len(api_key) > 10 else 'N/A'}...)")

        # Validate API key format (Gemini keys are typically 39 characters starting with specific patterns)
        if api_key and not (api_key.startswith(('AIza', 'AIzaSy')) and len(api_key) > 30):
            logging.warning(f"API key format looks unusual. Expected Gemini API key format. Key starts with: {api_key[:10] if api_key else 'None'}...")

        # CRITICAL: Check if API key is actually valid before proceeding
        if not api_key or not api_key.strip():
            raise ValueError("API key is empty or None. Please set your Google Gemini API key in Settings.")

        # Lazy import genai only when needed
        genai = _lazy_import_genai()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GOOGLE_MODEL_ID)
        response = model.generate_content(prompt)
        
        if not response:
            logging.error("Empty response from model")
            raise ValueError("Empty response from model")
        
        if not hasattr(response, 'text') or not response.text:
            logging.error(f"No text in response. Response object: {response}")
            raise ValueError("No text in model response")
        
        text = response.text
        logging.info(f"Successfully generated text (length: {len(text)})")
        return text
    except ValueError as e:
        logging.error(f"API key or response error: {e}")
        raise
    except Exception as e:
        logging.error(f"Error generating text: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise

def clarify_question(question: str) -> str:
    """Clarifying question given by user """
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        CLARIFY_QUESTION_INSTRUCTIONS = instructions[0]

        #Creating prompt for clarified question
        clarified_question_prompt = f"""
        {CLARIFY_QUESTION_INSTRUCTIONS}

        Question: {question}
        """
        llm_response = generate_text_with_llm(clarified_question_prompt)
        if llm_response is None or not llm_response.strip():
            logging.error("LLM returned empty response for clarify_question")
            return None
        return llm_response

    except ValueError as e:
        # API key error - re-raise with clear message
        logging.error(f"API key error in clarify_question: {e}")
        raise
    except Exception as e:
        logging.error(f"LLM clarifying question failed: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


def socratic_pass(clarified_question_response: str) -> str:
    """LLM is performing self questioning on clarified question response"""

    try:
        # Lazy import instructions and given_vars
        instructions = _lazy_import_instructions()
        SOCRATIC_PASS_INSTRUCTIONS = instructions[1]
        given_vars = _lazy_import_given_vars()

        socratic_pass_prompt = f"""
        {SOCRATIC_PASS_INSTRUCTIONS}
        User Question: {clarified_question_response}
        Socratic Principles: {given_vars.socratic_principles}
        """

        llm_response = generate_text_with_llm(socratic_pass_prompt)
        return llm_response

    except Exception as e:
        logging.error(f"SOCRATIC pass failed: {e}")

def socratic_answer_questions(clarified_question: str, probing_questions: str) -> str:
    """LLM answers the probing questions it previously asked itself, building deeper reasoning"""
    print("\n" + "="*80)
    print("DEBUG: INSIDE socratic_answer_questions")
    print(f"  clarified_question length: {len(clarified_question)}")
    print(f"  probing_questions length: {len(probing_questions)}")
    print("="*80 + "\n")
    try:
        # Pre-process probing questions to remove "Reasoning:" sections
        # This helps ensure the LLM focuses on answering the questions, not explaining why they were asked
        # Split by question patterns first, then clean each question
        import re
        
        # Pattern to match questions (lines that end with "?" and don't start with "Reasoning")
        question_pattern = r'^[^R].*\?'
        
        # Split into blocks - each question followed by its reasoning
        blocks = re.split(r'\n(?=\w)', probing_questions)  # Split on newline followed by word char
        
        cleaned_questions = []
        for block in blocks:
            lines = block.split('\n')
            question_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Skip reasoning lines
                if line_stripped.lower().startswith('reasoning:'):
                    break  # Stop at reasoning, don't include it
                # If line contains "Reasoning:" in middle, take only before it
                if 'Reasoning:' in line_stripped:
                    question_part = line_stripped.split('Reasoning:')[0].strip()
                    if question_part:
                        question_lines.append(question_part)
                    break  # Stop after this line
                # Add question lines
                if line_stripped and not line_stripped.lower().startswith('reasoning'):
                    question_lines.append(line_stripped)
            
            if question_lines:
                cleaned_questions.append('\n'.join(question_lines))
        
        # Join back together
        cleaned_questions_text = '\n\n'.join(cleaned_questions)
        
        # If we removed everything, fall back to original (shouldn't happen, but safety check)
        if not cleaned_questions_text.strip():
            cleaned_questions_text = probing_questions
            logging.warning("Could not clean probing questions, using original")
        else:
            logging.info(f"Cleaned probing questions (removed Reasoning sections)")
            logging.info(f"  Original length: {len(probing_questions)}, Cleaned length: {len(cleaned_questions_text)}")
            logging.info(f"  First 300 chars of cleaned: {cleaned_questions_text[:300]}...")
        
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        SOCRATIC_ANSWER_INSTRUCTIONS = instructions[2]  # SOCRATIC_ANSWER_INSTRUCTIONS is now at index 2
        
        socratic_answer_prompt = f"""
{SOCRATIC_ANSWER_INSTRUCTIONS}

User Question: {clarified_question}

Probing Questions: {cleaned_questions_text}
"""
        
        logging.info("Constructed socratic answer prompt")
        logging.info(f"  Prompt length: {len(socratic_answer_prompt)}")
        logging.info(f"  Instructions length: {len(SOCRATIC_ANSWER_INSTRUCTIONS)}")
        logging.info(f"  First 300 chars of full prompt: {socratic_answer_prompt[:300]}...")
        
        logging.info("Generating answers to socratic questions...")
        logging.info(f"  Clarified question length: {len(clarified_question)}")
        logging.info(f"  Probing questions length: {len(probing_questions)}")
        logging.info(f"  Cleaned questions length: {len(cleaned_questions_text)}")
        logging.info(f"  Cleaned questions preview: {cleaned_questions_text[:500]}...")
        
        try:
            llm_response = generate_text_with_llm(socratic_answer_prompt)
            logging.info(f"LLM call completed successfully")
        except Exception as llm_error:
            logging.error(f"LLM call failed with error: {llm_error}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
        
        logging.info(f"Raw LLM response received (length: {len(llm_response) if llm_response else 0})")
        if llm_response:
            logging.info(f"  First 500 chars of raw response: {llm_response[:500]}...")
        else:
            logging.error("LLM returned None or empty response")
            logging.error("  This is unusual - the LLM should return something, even if it's an error message")
            return None
        
        if llm_response:
            # CRITICAL: Check if LLM generated a hypothesis report instead of answers
            if "Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response:
                logging.error("LLM generated a hypothesis report instead of socratic answers!")
                logging.error("  Full response (first 1000 chars): {0}".format(llm_response[:1000]))
                logging.error("  TEMPORARILY ALLOWING THIS THROUGH SO USER CAN SEE IT")
                # TEMP: Don't return None, let it through so user can see what's happening
                # return None
            
            # Check for novelty/plausibility/testability which indicates a hypothesis report
            if "Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response:
                logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
                logging.error("  Full response (first 1000 chars): {0}".format(llm_response[:1000]))
                logging.error("  TEMPORARILY ALLOWING THIS THROUGH SO USER CAN SEE IT")
                # TEMP: Don't return None, let it through so user can see what's happening
                # return None
            
            # Check if LLM is just restating reasoning instead of answering
            # Look for patterns that indicate it's explaining why questions were asked
            reasoning_patterns = [
                "this question probes",
                "this question explores",
                "this question investigates",
                "this question examines",
                "this question seeks",
                "this question addresses",
                "reasoning:",
                "the reasoning behind"
            ]
            
            response_lower = llm_response.lower()
            reasoning_count = sum(1 for pattern in reasoning_patterns if pattern in response_lower)
            
            if reasoning_count >= 3:  # If multiple reasoning patterns found, likely not answering
                logging.warning(f"LLM response contains {reasoning_count} reasoning patterns - may be explaining why questions were asked instead of answering")
                logging.warning(f"  First 500 chars: {llm_response[:500]}...")
                # Don't return None - let it through but log warning
                # The instructions should handle this, but we want to track it
                # Still return the response so user can see what happened
            
            # Additional validation: Check if response actually answers questions vs just restating reasoning
            # Look for actual answers (specific materials, mechanisms, examples)
            answer_indicators = [
                "include", "examples include", "such as", "specifically", 
                "functional groups", "molecular", "classes of", "types of",
                "coordination", "interactions", "binding", "properties"
            ]
            answer_count = sum(1 for indicator in answer_indicators if indicator.lower() in response_lower)
            
            if reasoning_count >= 3 and answer_count < 2:
                logging.error(f"LLM response appears to be restating reasoning instead of answering!")
                logging.error(f"  Reasoning patterns: {reasoning_count}, Answer indicators: {answer_count}")
                logging.error(f"  First 500 chars: {llm_response[:500]}...")
                # Still return it, but log the issue - the instructions should handle this
            
            logging.info(f"Generated socratic answers (length: {len(llm_response)})")
            logging.info(f"  First 500 chars: {llm_response[:500]}...")
            logging.info(f"  Contains reasoning patterns: {reasoning_count}, Answer indicators: {answer_count}")
            
            # Final validation - if it looks like it's just restating reasoning, log but still return
            if reasoning_count >= 3 and answer_count < 2:
                logging.warning("WARNING: Response may be restating reasoning instead of answering")
                logging.warning("  Reasoning patterns: {0}, Answer indicators: {1}".format(reasoning_count, answer_count))
                logging.warning("  BUT still returning it so user can see what was generated")
        else:
            logging.error("Empty response from socratic answer generation")
            logging.error("  This means socratic answers will not be displayed")
            logging.error("  Returning None - fallback will use probing questions directly")
            
        print("\n" + "="*80)
        print("DEBUG: RETURNING from socratic_answer_questions")
        print(f"  Return value is None: {llm_response is None}")
        print(f"  Return value type: {type(llm_response)}")
        if llm_response:
            print(f"  Return value length: {len(llm_response)}")
            print(f"  First 300 chars: {llm_response[:300]}")
        print("="*80 + "\n")
        
        logging.info("=" * 80)
        logging.info(f"RETURNING from socratic_answer_questions")
        logging.info(f"  Return value is None: {llm_response is None}")
        logging.info(f"  Return value type: {type(llm_response)}")
        if llm_response:
            logging.info(f"  Return value length: {len(llm_response)}")
        logging.info("=" * 80)
        
        return llm_response
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"DEBUG: EXCEPTION in socratic_answer_questions: {e}")
        print("="*80 + "\n")
        logging.error(f"SOCRATIC answer generation failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None

def tot_generation_experimental_plan(socratic_pass_questioning: str, clarified_question: str, experimental_constraints: str) -> list:
    """LLM produces three distinct experimental plans given constraints"""
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS = instructions[6]

        tot_generation_prompt = f"""
        {EXPERIMENTAL_PLAN_TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Probing Questions: {socratic_pass_questioning}
        Experimental Constraints: {experimental_constraints}
        """

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if isinstance(llm_response, list):
            llm_response = " ".join(llm_response)
        elif llm_response is None:
            llm_response = ""

        # Parse experimental plans (look for "Experimental Plan" pattern)
        pattern = r"Experimental Plan\s*\d*\s*:\s*"
        raw_plans = re.split(pattern, llm_response, flags=re.IGNORECASE)

        plans = [p.strip() for p in raw_plans if p.strip()]

        # If no plans found, try alternative parsing
        if not plans or len(plans) == 0:
            numbered_pattern = r'^\d+\.?\s+'
            lines = llm_response.splitlines()
            for line in lines:
                line = line.strip()
                if line and (re.match(numbered_pattern, line) or line.startswith('-') or line.startswith('•')):
                    clean_plan = re.sub(numbered_pattern, '', line)
                    clean_plan = re.sub(r'^[-•]\s*', '', clean_plan)
                    if clean_plan and len(clean_plan) > 20:
                        plans.append(clean_plan)
            
            if not plans:
                sentences = re.split(r'(?<=[.!?])\s+', llm_response)
                plans = [s.strip() for s in sentences if len(s.strip()) > 30][:3]

        # Ensure we have at least 3 plans
        while len(plans) < 3:
            plans.append("")

        return plans[:3]

    except Exception as e:
        logging.error(f"Experimental plan TOT generation failed: {e}")
        return ["Error generating experimental plans", "Please check API key and try again", ""]


def tot_generation(socratic_pass_questioning: str, clarified_question: str, socratic_answers: str = None) -> list:
    """LLM produces three distinct lines of thought (rationale, predicted outcomes, and next step question)

    Args:
        socratic_pass_questioning: The probing questions generated
        clarified_question: The clarified user question
        socratic_answers: The answers to the probing questions (if available)
    """
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        TOT_INSTRUCTIONS = instructions[3]  # TOT_INSTRUCTIONS is now at index 3 (after SOCRATIC_ANSWER_INSTRUCTIONS)

        # If we have answers, use them; otherwise use questions
        reasoning_context = socratic_answers if socratic_answers else socratic_pass_questioning

        tot_generation_prompt = f"""
        {TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Socratic Answers: {reasoning_context}
        """
        
        if socratic_answers:
            logging.info(f"Generating TOT with socratic answers (length: {len(socratic_answers)})...")
        else:
            logging.info(f"Generating TOT with probing questions (answers not available)...")

        if socratic_answers:
            logging.info(f"Generating TOT with socratic answers (length: {len(socratic_answers)})...")
        else:
            logging.info(f"Generating TOT with probing questions (answers not available)...")

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if isinstance(llm_response, list):
            llm_response = " ".join(llm_response)  # combine if list
        elif llm_response is None:
            llm_response = ""  # avoid NoneType error
        
        # CRITICAL: Check if LLM generated a hypothesis report instead of thoughts
        if llm_response and ("Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response):
            logging.error("LLM generated a hypothesis report instead of TOT thoughts!")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []
        
        # Check for novelty/plausibility/testability which indicates a hypothesis report
        if llm_response and ("Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response):
            logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # CRITICAL: Check if LLM generated a hypothesis report instead of thoughts
        if llm_response and ("Hypothesis Report" in llm_response or "Hypothesis Evaluation Report" in llm_response):
            logging.error("LLM generated a hypothesis report instead of TOT thoughts!")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # Check for novelty/plausibility/testability which indicates a hypothesis report
        if llm_response and (
                "Novelty:" in llm_response or "Plausibility:" in llm_response or "Testability:" in llm_response):
            logging.error("LLM response contains hypothesis report markers (Novelty/Plausibility/Testability)")
            logging.error("  This should not happen - the LLM should only generate 3 thoughts")
            # Return empty list to trigger fallback
            return []

        # Improved pattern to match "Distinct Line of Thought" with optional number and colon
        # Matches: "Distinct Line of Thought 1:", "Distinct Line of Thought 1", "Distinct Line of Thought:", etc.
        # Use a more flexible pattern that handles variations (with or without numbers, with or without colons)
        pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"

        # First, try to find all occurrences of the pattern to verify we have 3 thoughts
        matches = list(re.finditer(pattern, llm_response, flags=re.IGNORECASE))

        logging.info(f"Found {len(matches)} 'Distinct Line of Thought' markers in response")

        if len(matches) >= 3:
            # We have at least 3 "Distinct Line of Thought" markers
            # Extract thoughts by finding text between markers
            thoughts = []
            for i, match in enumerate(matches):
                start_pos = match.end()  # Position after the marker
                # Find the end position (start of next marker, or end of string)
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(llm_response)

                thought_text = llm_response[start_pos:end_pos].strip()
                if thought_text and len(thought_text) > 10:
                    thoughts.append(thought_text)

            # If we still have issues, try split as fallback
            if len(thoughts) < 3:
                raw_thoughts = re.split(pattern, llm_response, flags=re.IGNORECASE)
                thoughts = [t.strip() for t in raw_thoughts if t.strip() and len(t.strip()) > 10]

            # Remove the first element if it's just preamble (text before first "Distinct Line of Thought")
            if thoughts and len(thoughts) > 0:
                first_thought = thoughts[0]
                # If the first element doesn't look like a thought (too short),
                # and we have more thoughts, remove it
                if len(first_thought) < 30 and len(thoughts) > 1:
                    # Check if it looks like preamble (common words that appear before thoughts)
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_thought.lower() for indicator in preamble_indicators) or (
                            first_thought and not first_thought[0].isupper()):
                        thoughts = thoughts[1:]

            # Ensure we have exactly 3 thoughts
            if len(thoughts) > 3:
                # Take the first 3 substantial thoughts
                substantial_thoughts = [t for t in thoughts if len(t.strip()) > 30]
                thoughts = substantial_thoughts[:3] if len(substantial_thoughts) >= 3 else thoughts[:3]
            elif len(thoughts) < 3 and len(thoughts) > 0:
                # If we have fewer than 3, check if any thought contains multiple "Distinct Line of Thought" markers
                expanded_thoughts = []
                for thought in thoughts:
                    # Check if this thought contains additional markers
                    sub_matches = list(re.finditer(pattern, thought, flags=re.IGNORECASE))
                    if len(sub_matches) > 0:
                        # Split this thought further
                        sub_split = re.split(pattern, thought, flags=re.IGNORECASE)
                        sub_split = [t.strip() for t in sub_split if t.strip() and len(t.strip()) > 10]
                        expanded_thoughts.extend(sub_split)
                    else:
                        expanded_thoughts.append(thought)

                if len(expanded_thoughts) >= 3:
                    thoughts = expanded_thoughts[:3]
        else:
            # Fallback: use simple split
            raw_thoughts = re.split(pattern, llm_response, flags=re.IGNORECASE)
            thoughts = [t.strip() for t in raw_thoughts if t.strip() and len(t.strip()) > 10]

            # Remove preamble if present
            if thoughts and len(thoughts) > 0:
                first_thought = thoughts[0]
                if len(first_thought) < 30 and len(thoughts) > 1:
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_thought.lower() for indicator in preamble_indicators) or not \
                    first_thought[0].isupper():
                        thoughts = thoughts[1:]

            # Limit to 3
            if len(thoughts) > 3:
                substantial_thoughts = [t for t in thoughts if len(t.strip()) > 30]
                thoughts = substantial_thoughts[:3] if len(substantial_thoughts) >= 3 else thoughts[:3]

        # If no thoughts found, try alternative parsing
        if not thoughts or len(thoughts) == 0:
            # Try splitting by numbered lists (1., 2., 3.) or bullet points
            numbered_pattern = r'^\d+\.?\s+'
            lines = llm_response.splitlines()
            for line in lines:
                line = line.strip()
                if line and (re.match(numbered_pattern, line) or line.startswith('-') or line.startswith('•')):
                    # Remove numbering/bullets
                    clean_thought = re.sub(numbered_pattern, '', line)
                    clean_thought = re.sub(r'^[-•]\s*', '', clean_thought)
                    if clean_thought and len(clean_thought) > 20:  # Only substantial thoughts
                        thoughts.append(clean_thought)

            # If still no thoughts, try splitting by sentences
            if not thoughts:
                sentences = re.split(r'(?<=[.!?])\s+', llm_response)
                thoughts = [s.strip() for s in sentences if len(s.strip()) > 30][
                    :3]  # Take up to 3 substantial sentences

        # Ensure we have at least 3 thoughts (pad if needed)
        while len(thoughts) < 3:
            thoughts.append("")

        # --- Debug print ---
        logging.info(f"Parsed {len(thoughts)} thoughts from TOT response")
        for i, t in enumerate(thoughts[:3], 1):
            if t:  # Only print non-empty thoughts
                logging.info(f"Thought {i} (length: {len(t)}): {t[:100]}...")
                print(f"\n--- Thought {i} ---\n{t}\n")

        # Final validation: ensure we have exactly 3 thoughts, each substantial
        final_thoughts = []
        for t in thoughts[:3]:
            if t and t.strip() and len(t.strip()) > 20:
                final_thoughts.append(t.strip())
            else:
                final_thoughts.append("")  # Pad with empty if needed

        # Ensure exactly 3
        while len(final_thoughts) < 3:
            final_thoughts.append("")

        return final_thoughts[:3]  # Return exactly 3 thoughts

    except Exception as e:
        logging.error(f"TOT generation failed: {e}")
        return ["Error generating thoughts", "Please check API key and try again", ""]


def retry_thinking_deepen_thoughts(line_of_thought: str, previous_thought_1: str, previous_thought_2: str,
                                   initial_clarified_question: str, conversation_context: str = ""):
    # produces continuation options based on user's selected option
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        RETRY_THINKING_INSTRUCTIONS = instructions[3]

        # Build context section if available
        context_section = ""
        if conversation_context and conversation_context.strip():
            context_section = f"""

        **Conversation Context (for understanding the flow):**
        {conversation_context}

        **CRITICAL: Use this context to understand what has been discussed, but your 3 continuation options MUST build directly on the selected option below, not on earlier parts of the conversation.**
        """

        tot_generation_prompt = f"""
        {RETRY_THINKING_INSTRUCTIONS}

        **CRITICAL: The user has JUST SELECTED the following option/hypothesis. Your task is to generate EXACTLY 3 continuation options that explore this selected option further. These are NOT new starting points - they must continue from what the user selected.**

        **SELECTED OPTION (CONTINUE FROM THIS - DO NOT IGNORE THIS):**
        {line_of_thought}

        **Previous Options (NOT selected - shown for context only, do NOT continue from these):**
            Previous Option 1: {previous_thought_1}
            Previous Option 2: {previous_thought_2}

        **Initial Question (for context only):** {initial_clarified_question}
        {context_section}

        **FINAL REMINDER:**
        - The user selected the option above (the "SELECTED OPTION")
        - Generate 3 options that CONTINUE exploring that selected option
        - Each option should explore a different aspect of the SAME selected option
        - Do NOT generate new TOT thoughts or start new directions
        - Do NOT ignore the selected option and generate unrelated thoughts
        - Build directly on the materials, mechanisms, or conditions mentioned in the selected option
        """

        # Log what we're sending to the LLM
        logging.info(f"retry_thinking_deepen_thoughts - Selected option (first 200 chars): {line_of_thought[:200]}")
        logging.info(f"retry_thinking_deepen_thoughts - Prompt length: {len(tot_generation_prompt)}")

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if llm_response is None:
            llm_response = ""

        # Log the raw response for debugging
        logging.info(f"retry_thinking_deepen_thoughts raw response (first 1000 chars): {llm_response[:1000]}")

        # Validate that the response actually references the selected option
        # Extract key terms from the selected option (improved to work with any chemical compound)
        import re as regex_module
        selected_lower = line_of_thought.lower()

        # Extract chemical compounds, molecular formulas, and key terms
        # Look for patterns like: Chemical names, formulas (e.g., CsPbI3, 2,2'-bipyridinium), cation markers (⁺, +)
        key_terms = []

        # Extract chemical abbreviations (e.g., XBDA, PPD, ATZ)
        abbrev_pattern = r'\b([A-Z]{2,6})\b'
        abbrevs = regex_module.findall(abbrev_pattern, line_of_thought)
        # Filter out action verbs and common words
        action_verbs = {'EXAMINING', 'EXPLORING', 'INVESTIGATING', 'QUANTIFYING', 'ANALYZING', 'COMPARE', 'EXAMINE'}
        abbrevs = [a for a in abbrevs if a.upper() not in action_verbs and 'CSPBI' not in a.upper()]
        key_terms.extend([a.lower() for a in abbrevs])

        # Extract words that look like chemical compounds (capitalized, contain numbers/symbols)
        chemical_pattern = r'\b[A-Z][a-z]*(?:\d+|[⁺₂₃₄])*(?:[A-Z][a-z]*(?:\d+|[⁺₂₃₄])*)*\b'
        chemicals = regex_module.findall(chemical_pattern, line_of_thought)
        # Filter out verbs
        chemicals = [c for c in chemicals if
                     c.lower() not in ['examining', 'exploring', 'investigating', 'quantifying', 'analyzing', 'compare',
                                       'examine', 'investigate']]
        key_terms.extend([c.lower() for c in chemicals if len(c) > 2])  # Filter short matches

        # Extract words with special characters that indicate chemical notation
        special_chem_pattern = r"[\w\d\-',]+(?:Cl₂|Br₂|I₂|⁺|₂|₃|₄)"
        special_chemicals = regex_module.findall(special_chem_pattern, line_of_thought)
        key_terms.extend([c.lower() for c in special_chemicals])

        # Extract hyphenated compound names (like 2,2'-bipyridinium, 1,4-bis...)
        hyphenated_pattern = r"\d+[,\-]\d+'?[,\-]?\w+"
        hyphenated = regex_module.findall(hyphenated_pattern, line_of_thought)
        key_terms.extend([c.lower() for c in hyphenated])

        # Remove duplicates, filter, and remove action verbs
        key_terms = list(set([term for term in key_terms if
                              len(term) > 3 and term not in ['examine', 'exploring', 'investigating', 'quantifying',
                                                             'analyzing', 'compare']]))

        logging.info(f"Extracted key terms from selected option: {key_terms[:10]}")  # Log first 10

        response_lower = llm_response.lower()
        if key_terms:
            # Check if ANY of the key terms appear in the response
            found_terms = [term for term in key_terms if term in response_lower]
            if len(found_terms) < max(1, len(key_terms) // 3):  # At least 1/3 of terms should be present
                logging.warning(f"WARNING: Generated response contains few/no key terms from selected option")
                logging.warning(f"Selected option mentions: {key_terms[:5]}, but response only found: {found_terms}")
                logging.warning(f"This suggests the LLM is generating unrelated thoughts instead of continuations!")
                logging.warning(f"Will use fallback to generate relevant continuations")

        # Parse response more robustly
        # Focus ONLY on extracting the 3 continuation options - IGNORE any socratic questions
        socratic_question = ""  # We don't want socratic questions for continuations
        options = []

        # Pattern to match "Distinct Line of Thought" format (same as tot_generation)
        thought_pattern = r"Distinct\s+Line\s+of\s+Thought\s*\d*\s*:?\s*"

        # Find all "Distinct Line of Thought" markers - these are what we want
        thought_matches = list(re.finditer(thought_pattern, llm_response, flags=re.IGNORECASE))

        # IGNORE any socratic questions - we only want the 3 continuation options
        # Don't extract any questions, just skip to the options

        # Extract options using "Distinct Line of Thought" markers (same logic as tot_generation)
        if len(thought_matches) >= 3:
            # Extract thoughts by finding text between markers
            for i, match in enumerate(thought_matches):
                start_pos = match.end()  # Position after the marker
                # Find the end position (start of next marker, or end of string)
                if i + 1 < len(thought_matches):
                    end_pos = thought_matches[i + 1].start()
                else:
                    end_pos = len(llm_response)

                option_text = llm_response[start_pos:end_pos].strip()
                # Clean up: remove any remaining "Distinct Line of Thought" markers that might be embedded
                option_text = re.sub(thought_pattern, "", option_text, flags=re.IGNORECASE).strip()
                if option_text and len(option_text) > 10:
                    options.append(option_text)

            logging.info(f"Extracted {len(options)} options from 'Distinct Line of Thought' markers")
        elif len(thought_matches) > 0:
            # We have some markers but less than 3, try split approach
            raw_options = re.split(thought_pattern, llm_response, flags=re.IGNORECASE)
            options = [opt.strip() for opt in raw_options if opt.strip() and len(opt.strip()) > 10]
            # Remove preamble if present
            if options and len(options) > 0:
                first_opt = options[0]
                if len(first_opt) < 30 and len(options) > 1:
                    preamble_indicators = ['here', 'following', 'below', 'are', 'the', 'three', 'distinct', 'lines',
                                           'thoughts']
                    if any(indicator in first_opt.lower() for indicator in preamble_indicators):
                        options = options[1:]

        # Fallback: if we don't have 3 options yet, try alternative parsing methods
        if len(options) < 3:
            # Try to find options by looking for numbered items or bullet points
            numbered_pattern = r'^\d+\.\s+'
            bullet_pattern = r'^[-•]\s+'

            for line in lines:
                line_stripped = line.strip()
                # Look for numbered options (1., 2., 3.)
                if re.match(numbered_pattern, line_stripped) or re.match(bullet_pattern, line_stripped):
                    # Remove numbering/bullet
                    clean_line = re.sub(numbered_pattern, '', line_stripped)
                    clean_line = re.sub(bullet_pattern, '', clean_line)
                    # Remove "Distinct Line of Thought" if present
                    clean_line = re.sub(thought_pattern, '', clean_line, flags=re.IGNORECASE).strip()
                    if clean_line and len(clean_line) > 15 and clean_line not in options:
                        options.append(clean_line)
                        if len(options) >= 3:
                            break
                # Also check for lines that start with "Option" or contain substantial content
                elif len(line_stripped) > 30 and not any(
                        kw in line_stripped.lower() for kw in ['socratic', 'question', 'reasoning']):
                    # Remove "Distinct Line of Thought" if present
                    clean_line = re.sub(thought_pattern, '', line_stripped, flags=re.IGNORECASE).strip()
                    if clean_line and len(clean_line) > 15 and clean_line not in options:
                        options.append(clean_line)
                        if len(options) >= 3:
                            break

            logging.info(f"After fallback parsing, found {len(options)} options")

        # Log what we extracted
        logging.info(f"retry_thinking_deepen_thoughts: Extracted {len(options)} options")
        for i, opt in enumerate(options, 1):
            logging.info(f"  Option {i} (length: {len(opt)}): {opt[:100]}...")

        # CRITICAL: Validate that the extracted options actually reference the selected option
        # Use the same key_terms we extracted earlier
        validated_options = []
        for opt in options:
            opt_lower = str(opt).lower()
            # Check if this option references materials/terms from the selected option
            # Use key_terms we extracted earlier with improved pattern matching
            if not key_terms:
                # If no key terms were extracted, accept the option
                validated_options.append(opt)
            else:
                # Check if option references ANY of the key terms from the selected option
                found_in_opt = [term for term in key_terms if term in opt_lower]
                if found_in_opt or len(key_terms) == 0:
                    validated_options.append(opt)
                else:
                    logging.warning(f"Rejected option that doesn't reference selected materials: {opt[:100]}...")
                    logging.warning(
                        f"Selected option contains: {key_terms[:5]}, but this option doesn't reference them")

        # If we lost options due to validation, create proper continuations
        if len(validated_options) < 3:
            logging.warning(
                f"Only {len(validated_options)} options passed validation. Creating continuations based on selected option.")

            # Extract the PRIMARY subject from the selected option (the main material/concept being discussed)
            # This should be what the option is ABOUT (the chemical compound), not action verbs
            primary_subject = None

            # Pattern 1: Look for text in parentheses (often contains full chemical names like "XBDA" or "1,4-bis...")
            paren_pattern = r'\(([^)]{5,100})\)'  # 5-100 chars in parentheses
            parens = re.findall(paren_pattern, line_of_thought)
            if parens:
                # Filter out things that are clearly not chemical names
                chem_names = [p for p in parens if
                              not any(word in p.lower() for word in ['e.g.', 'i.e.', 'such as', 'including'])]
                if chem_names:
                    primary_subject = chem_names[0]
                    logging.info(f"Extracted primary subject from parentheses: {primary_subject}")

            # Pattern 2: Look for abbreviated chemical names before parentheses (e.g., "XBDA" in "XBDA (1,4-bis...)")
            if not primary_subject:
                abbrev_pattern = r'\b([A-Z]{2,6})\s*\('  # 2-6 capital letters before parenthesis
                abbrevs = re.findall(abbrev_pattern, line_of_thought)
                if abbrevs:
                    # Filter out common non-chemical abbreviations
                    chem_abbrevs = [a for a in abbrevs if a not in ['CsPbI', 'ATZ', 'PPD'] or len(a) >= 3]
                    if chem_abbrevs:
                        primary_subject = chem_abbrevs[0]
                        logging.info(f"Extracted primary subject from abbreviation: {primary_subject}")

            # Pattern 3: Look for specific compound mentions
            if not primary_subject:
                # Look for explicit compound names after "Introducing", "Utilizing", etc.
                introducing_pattern = r"(?:Introducing|Utilizing|Incorporating|Using|Employing)\s+([A-Za-z0-9\-,'\(\)]+\s*(?:acid|amine|ammonium|cation|spacer)?)\s+(?:as|is|could|to)"
                match = re.search(introducing_pattern, line_of_thought, re.IGNORECASE)
                if match:
                    subject_candidate = match.group(1).strip()
                    # Make sure it's not a verb phrase
                    if not any(verb in subject_candidate.lower() for verb in
                               ['investigating', 'exploring', 'examining', 'preventing', 'introducing', 'comparing',
                                'water ingress', 'phase']):
                        primary_subject = subject_candidate
                        logging.info(f"Extracted primary subject from introducing pattern: {primary_subject}")

            # Pattern 4: Look for chemical formulas with subscripts (but not CsPbI3)
            if not primary_subject or len(primary_subject) < 5:
                chem_formula_pattern = r'\b[A-Z][a-z]?(?:\d+|[⁺⁻₀₁₂₃₄₅₆₇₈₉])+(?:[A-Z][a-z]?(?:\d+|[⁺⁻₀₁₂₃₄₅₆₇₈₉])*)*\b'
                formulas = re.findall(chem_formula_pattern, line_of_thought)
                # Filter out CsPbI3 (that's the base material)
                formulas = [f for f in formulas if 'cspbi' not in f.lower()]
                if formulas:
                    primary_subject = formulas[0]
                    logging.info(f"Extracted primary subject from formula: {primary_subject}")

            # Fallback: Take first 60 chars after first comma or "of"
            if not primary_subject or len(primary_subject) < 5 or any(
                    verb in primary_subject.lower() for verb in ['investigating', 'exploring', 'examining']):
                # Try to get the noun phrase, not the verb phrase
                if ',' in line_of_thought:
                    parts = line_of_thought.split(',')
                    if len(parts) > 1:
                        primary_subject = parts[1][:60].strip()
                else:
                    primary_subject = "the proposed approach"
                logging.info(f"Using fallback primary subject: {primary_subject}")

            # Validate primary_subject doesn't contain action verbs or generic phrases
            action_verbs = ['investigating', 'exploring', 'examining', 'quantifying', 'analyzing', 'examine',
                            'investigate', 'explore', 'compare', 'preventing', 'forming', 'creating']
            bad_phrases = ['water ingress', 'phase transition', 'phase segregation', 'bandgap tuning', 'the effect',
                           'the influence', 'the impact']
            primary_lower = primary_subject.lower() if primary_subject else ""

            # If primary_subject contains action verbs, bad phrases, or is too generic, use a better fallback
            is_bad = (any(verb in primary_lower for verb in action_verbs) or
                      any(phrase in primary_lower for phrase in bad_phrases) or
                      len(primary_subject) > 150)

            if is_bad:
                logging.warning(
                    f"Primary subject contains action verbs/bad phrases or is too long: {primary_subject[:100]}")
                # Try to extract just the chemical compound name from the selected option
                # Look for patterns like "stearic acid", "2-aminoethylphosphonic acid", etc.
                # Pattern: word ending in "acid", "amine", "ammonium", etc.
                chem_name_pattern = r'\b([a-z0-9\-,\(\)]+(?:acid|amine|ammonium|phosphonic|carboxylic|pyridine|phenyl|thiol))\b'
                chem_names = re.findall(chem_name_pattern, line_of_thought, re.IGNORECASE)
                if chem_names:
                    primary_subject = chem_names[0]
                    logging.info(f"Fixed primary subject to chemical name: {primary_subject}")
                else:
                    # Look for capitalized abbreviations
                    compound_pattern = r'\b([A-Z]{2,6}(?:-[A-Z]{2,6})?)\b'
                    compounds = re.findall(compound_pattern, line_of_thought)
                    # Filter out CsPbI and common words
                    compounds = [c for c in compounds if 'cspbi' not in c.lower() and c not in ['CsPbI', 'Pb', 'I']]
                    if compounds:
                        primary_subject = compounds[0]
                        logging.info(f"Fixed primary subject to abbreviation: {primary_subject}")
                    else:
                        primary_subject = "the proposed material"
                        logging.info(f"Using generic fallback: {primary_subject}")

            # Create SOCRATIC-STYLE continuation options (how/why/what questions)
            continuation_options = [
                f"How does {primary_subject} specifically interact at the molecular level with CsPbI₃, and what structural features enable these interactions?",
                f"Why is {primary_subject} expected to be more effective than alternative approaches, and what are the thermodynamic or kinetic advantages?",
                f"What are the potential challenges or limitations of using {primary_subject}, and how might these be addressed?"
            ]
            logging.info(f"Created socratic-style continuation options with subject: {primary_subject}")

            # Use validated options first, then fill with proper continuations
            options = validated_options + continuation_options[:3 - len(validated_options)]

        # Ensure we have exactly 3 options
        if len(options) < 3:
            remaining = 3 - len(options)
            generic_continuations = [
                "Analyze the electronic structure modifications that would result from this approach",
                "Explore experimental conditions and synthesis parameters for optimal phase control",
                "Investigate the relationship between structural ordering and photoluminescence narrowing"
            ]
            options.extend(generic_continuations[:remaining])
        elif len(options) > 3:
            options = options[:3]  # Take only first 3

        # Ensure all options are non-empty and substantial, and don't contain "Distinct Line of Thought" markers
        final_options = []
        for i, opt in enumerate(options):
            if opt and opt.strip() and len(opt.strip()) >= 5:
                # Clean up: remove any "Distinct Line of Thought" markers that might be at the start
                clean_opt = re.sub(thought_pattern, "", opt, flags=re.IGNORECASE).strip()
                if clean_opt and len(clean_opt) >= 5:
                    final_options.append(clean_opt)

        # Ensure we have exactly 3
        while len(final_options) < 3:
            final_options.append(f"Option {len(final_options) + 1}: Continue exploring")

        options = final_options[:3]

        # Final validation
        for i in range(len(options)):
            if not options[i] or len(options[i].strip()) < 5:
                options[i] = f"Option {i + 1}: Continue exploring this line of reasoning"

        # Log for debugging
        logging.info(
            f"retry_thinking_deepen_thoughts returning: {len(options)} continuation options (no socratic question for continuations)")

        # Always return empty socratic question for continuations - we don't want new questions
        return "", options
    except Exception as e:
        logging.error(f"Retry Thinking generation failed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

        # Create meaningful fallback options based on the selected hypothesis
        selected_text = line_of_thought[:200] if line_of_thought else ""

        # Try to extract key materials/mechanisms mentioned
        material_patterns = [r'(\w+[⁺²⁺]?)', r'([A-Z][a-z]+\w*[⁺²⁺]?)']
        materials = []
        for pattern in material_patterns:
            matches = re.findall(pattern, selected_text)
            materials.extend([m for m in matches if len(m) > 2 and m not in materials][:2])

        if materials:
            mat1 = materials[0] if len(materials) > 0 else "the proposed material"
            fallback_options = [
                f"Explore how {mat1} concentration and ratio affect the phase distribution and bandgap tuning",
                f"Investigate the specific mechanism by which {mat1} interacts with the inorganic framework",
                f"Compare {mat1} with alternative co-spacers to determine optimal properties for the target bandgap"
            ]
        else:
            fallback_options = [
                "Explore how the concentration and ratio of the proposed co-spacer affect phase distribution and bandgap tuning",
                "Investigate the specific mechanism of interaction with the inorganic framework to promote narrowed PL emission",
                "Compare with alternative co-spacers to determine optimal structural and electronic properties for the target bandgap"
            ]

        # Return empty socratic question and fallback options
        return "", fallback_options


def hypothesis_synthesis(socratic_question: str, next_step_option: str, previous_option_1: str,
                         previous_option_2: str, conversation_context: str) -> str:
    # Produce hypothesis based on user selected thoughts and other information
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        HYPOTHESIS_SYNTHESIS = instructions[4]

        hypothesis_synthesis_prompt = f"""
            {HYPOTHESIS_SYNTHESIS}
            Socratic Question: {socratic_question}
            Next-Step Option: {next_step_option}
            Previous Step Options: 
                Previous Option 1: {previous_option_1}
                Previous Option 2: {previous_option_2}
            Full Conversation Context: {conversation_context}
            """

        llm_response = generate_text_with_llm(hypothesis_synthesis_prompt)

        if llm_response is None or not llm_response.strip():
            return "Error generating hypothesis. Please check your API key and try again."

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis generation failed: {e}")
        return f"Error generating hypothesis: {str(e)}. Please check your API key and try again."


def local_hypothesis_analysis_fallback(hypothesis: str, socratic_question: str) -> str:
    try:
        # Lazy import instructions
        instructions = _lazy_import_instructions()
        HYPOTHESIS_ANALYSIS_REPORT = instructions[5]

        hypothesis_analysis_report_prompt = f"""
        {HYPOTHESIS_ANALYSIS_REPORT}
        Hypothesis: {hypothesis}
        Socratic Question: {socratic_question}
        """

        llm_response = generate_text_with_llm(hypothesis_analysis_report_prompt)

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis Analysis Report generation failed: {e}")
        # Fallback: return a simple error message
        return f"Error generating hypothesis analysis: {str(e)}. Please check your API key and try again."