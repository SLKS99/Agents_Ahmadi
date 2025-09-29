import logging
import os
import google.generativeai as genai
import logging
from tools.instruct import CLARIFY_QUESTION_INSTRUCTIONS, SOCRATIC_PASS_INSTRUCTIONS, TOT_INSTRUCTIONS
import tools.given_variables as given_vars
import re

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
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

def tot_generation(socratic_pass_questioning: str, clarified_question: str) -> list[str]:
    """LLM produces three distinct lines of thought (rationale, predicted outcomes, and next step question)"""
    try:
        tot_generation_prompt = f"""
        {TOT_INSTRUCTIONS}
        User Question: {clarified_question}
        Probing Questions: {socratic_pass_questioning}
        """

        llm_response = generate_text_with_llm(tot_generation_prompt)

        lines = llm_response.splitlines()
        thoughts = [lines.strip() for lines in lines if lines.strip()]

        return thoughts

    except Exception as e:
        logging.error(f"TOT generation failed: {e}")

def retry_thinking_deepen_thoughts(user_selection: str) -> str:
    #produces another line of thought based on user's input
    response = ""
    return response

def hypothesis_synthesis(user_selected_questions, information):
    #Produce hypothesis based on user selected thoughts and other information
    response = ""
    return response

def analyze_hypothesis(hypothesis):
    response = ""
    return response





