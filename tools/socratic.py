import os
import google.generativeai as genai
from edison_client import EdisonClient, JobNames
from edison_client.models import TaskRequest, RuntimeConfig
import logging
from tools.instruct import (CLARIFY_QUESTION_INSTRUCTIONS, SOCRATIC_PASS_INSTRUCTIONS, TOT_INSTRUCTIONS,
                            RETRY_THINKING_INSTRUCTIONS, HYPOTHESIS_SYNTHESIS, HYPOTHESIS_ANALYSIS_REPORT)
import tools.given_variables as given_vars
import re

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
FUTUREHOUSE_API_KEY = os.getenv("FUTUREHOUSE_API_KEY")
GOOGLE_MODEL_ID = "gemini-2.5-flash"


def generate_text_with_llm(prompt: str) -> str:
    """Generating text using Gemini provided model and API Key"""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(GOOGLE_MODEL_ID)
        response = model.generate_content(prompt)
        text = response.text
        return text
    except Exception as e:
        logging.error(f"Error generating text: {e}")

def clarify_question(question: str) -> str:
    """Clarifying question given by user """
    try:
        #Creating prompt for clarified question
        clarified_question_prompt = f"""
        {CLARIFY_QUESTION_INSTRUCTIONS}
        
        Question: {question}
        """
        llm_response = generate_text_with_llm(clarified_question_prompt)
        return llm_response

    except Exception as e:
        logging.error(f"LLM clarifying question failed: {e}")

def socratic_pass(clarified_question_response: str) -> str:
    """LLM is performing self questioning on clarified question response"""

    try:
        socratic_pass_prompt = f"""
        {SOCRATIC_PASS_INSTRUCTIONS}
        User Question: {clarified_question_response}
        Socratic Principles: {given_vars.socratic_principles}
        """

        llm_response = generate_text_with_llm(socratic_pass_prompt)
        return llm_response

    except Exception as e:
        logging.error(f"SOCRATIC pass failed: {e}")

def tot_generation(socratic_pass_questioning: str, clarified_question: str) -> list:
    """LLM produces three distinct lines of thought (rationale, predicted outcomes, and next step question)"""
    try:
        tot_generation_prompt = f"""
        {TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Probing Questions: {socratic_pass_questioning}
        """

        llm_response = generate_text_with_llm(tot_generation_prompt)

        if isinstance(llm_response, list):
            llm_response = " ".join(llm_response)  # combine if list
        elif llm_response is None:
            llm_response = ""  # avoid NoneType error

        pattern = r"Distinct Line of Thought\s*\d*\s*:\s*"

        raw_thoughts = re.split(pattern, llm_response, flags=re.IGNORECASE)

        thoughts = [t.strip() for t in raw_thoughts if t.strip()]

        # --- Debug print ---
        for i, t in enumerate(thoughts, 1):
            print(f"\n--- Thought {i} ---\n{t}\n")

        return thoughts

    except Exception as e:
        logging.error(f"TOT generation failed: {e}")

def retry_thinking_deepen_thoughts(line_of_thought:str, previous_thought_1:str, previous_thought_2: str,
                                   initial_clarified_question:str):
    #produces another line of thought based on user's input
    try:
        tot_generation_prompt = f"""
        {RETRY_THINKING_INSTRUCTIONS}
        Line of Thought: {line_of_thought}
        Previous Thoughts:
            Previous Thought 1: {previous_thought_1}
            Previous Thought 2: {previous_thought_2}
        Initial Clarified Question: {initial_clarified_question}
        """

        llm_response = generate_text_with_llm(tot_generation_prompt)

        raw_text = llm_response.splitlines()
        socratic_question = raw_text[0]
        options = [raw_text[1], raw_text[2], raw_text[3]]

        return socratic_question, options
    except Exception as e:
        logging.error(f"Retry Thinking generation failed: {e}")

def hypothesis_synthesis(socratic_question:str, next_step_option:str, previous_option_1: str, previous_option_2: str) -> str:
    #Produce hypothesis based on user selected thoughts and other information
    try:
        hypothesis_synthesis_prompt = f"""
            {HYPOTHESIS_SYNTHESIS}
            Socratic Question: {socratic_question}
            Next-Step Option: {next_step_option}
            Previous Step Options: 
                Previous Option 1: {previous_option_1}
                Previous Option 2: {previous_option_2}
            """

        llm_response = generate_text_with_llm(hypothesis_synthesis_prompt)

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis generation failed: {e}")


def analyze_hypothesis(hypothesis:str, socratic_question:str, rubric_weights=None):

    if rubric_weights is None:
        rubric_weights = {"novelty": 1 / 3, "plausibility": 1 / 3, "testability": 1 / 3}

    client = EdisonClient(api_key=FUTUREHOUSE_API_KEY)

    # --- Step 1: Retrieve literature evidence
    task_data = TaskRequest(
        name=JobNames.CROW,
        query=socratic_question,
        runtime_config=RuntimeConfig(max_steps=5)
    )

    responses = client.run_tasks_until_done(task_data)
    evidence_summary = responses[0].formatted_answer.strip()

    # --- Step 2: Heuristic scoring (can later be replaced by LLM eval)
    # Novelty — less overlap with existing work = higher score
    prior_matches = len(re.findall(re.escape(hypothesis), evidence_summary, flags=re.IGNORECASE))
    novelty_score = max(0.0, 1.0 - min(prior_matches, 10) * 0.1)
    novelty_reason = (
        "High novelty: little to no direct matches found in literature."
        if novelty_score > 0.8 else
        "Moderate novelty: related ideas appear in literature."
        if 0.4 < novelty_score <= 0.8 else
        "Low novelty: hypothesis closely resembles existing findings."
    )

    # Plausibility — based on overlap of key concepts with evidence
    key_terms = re.findall(r"\b\w+\b", hypothesis)
    hits = sum(1 for term in key_terms if re.search(rf"\b{re.escape(term)}\b", evidence_summary, flags=re.IGNORECASE))
    plausibility_score = min(1.0, hits / len(key_terms))
    plausibility_reason = (
        "Strong conceptual support in related studies."
        if plausibility_score > 0.7 else
        "Partial conceptual alignment with existing evidence."
        if 0.4 < plausibility_score <= 0.7 else
        "Weak or minimal evidence supporting the hypothesis."
    )

    # Testability — presence of experimental/test language
    test_keywords = ["experiment", "measure", "observe", "compare", "test", "quantify", "analyze"]
    found_keywords = [kw for kw in test_keywords if re.search(rf"\b{kw}\b", evidence_summary, re.IGNORECASE)]
    testability_score = 1.0 if found_keywords else 0.5
    testability_reason = (
        "Evidence discusses measurable or experimental approaches."
        if found_keywords else
        "Limited direct mention of testable or measurable outcomes."
    )

    # --- Step 3: Weighted composite score
    scores = {
        "novelty": novelty_score,
        "plausibility": plausibility_score,
        "testability": testability_score
    }
    weighted_score = sum(scores[k] * rubric_weights[k] for k in rubric_weights)

    # --- Step 4: Generate readable Markdown report
    report = f"""
    ### 🧠 Hypothesis Evaluation Report

    **Hypothesis:**  
    {hypothesis}

    ---

    #### 📚 Evidence Summary:
    {evidence_summary}

    ---

    #### 🧩 Scoring Breakdown:

    | Criterion | Score | Explanation |
    |:-----------|:------:|:------------|
    | **Novelty** | {novelty_score:.2f} | {novelty_reason} |
    | **Plausibility** | {plausibility_score:.2f} | {plausibility_reason} |
    | **Testability** | {testability_score:.2f} | {testability_reason} |

    **Weighted Total Score:** {weighted_score:.2f}

    ---

    #### 🧾 Interpretation:
    - **> 0.80:** Highly original, plausible, and testable.  
    - **0.50 – 0.80:** Promising but may need refinement or stronger evidence.  
    - **< 0.50:** Likely weak, redundant, or insufficiently grounded.

    ---

    *Generated via FutureHouse API analysis and heuristic rubric evaluation.*
    """

    return {
        "evidence_summary": evidence_summary,
        "scores": scores,
        "weighted_score": weighted_score,
        "report": report.strip()
    }

def local_hypothesis_analysis_fallback(hypothesis:str, socratic_question:str) -> str:
    try:
        hypothesis_analysis_report_prompt = f"""
        {HYPOTHESIS_ANALYSIS_REPORT}
        Hypothesis: {hypothesis}
        Socratic Question: {socratic_question}
        """

        llm_response = generate_text_with_llm(hypothesis_analysis_report_prompt)

        return llm_response

    except Exception as e:
        logging.error(f"Hypothesis Analysis Report generation failed: {e}")





