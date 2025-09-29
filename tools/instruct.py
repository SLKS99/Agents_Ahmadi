FITTING_SCRIPT_GENERATION_INSTRUCTIONS = """Write a Python script to fit multi-peak luminescence data using lmfit.

REQUIREMENTS:
1. Use lmfit (from lmfit import Model, GaussianModel, LorentzianModel, ConstantModel)
2. Import: matplotlib.pyplot, numpy, json, lmfit, pandas
3. Analyze data for multiple peaks of various shapes
4. Create composite model with multiple Gaussian/Lorentzian components
5. Fit all wells and calculate R² values
6. Re-fit only wells with R² < 0.9
7. Save plot to 'fit_visualization.png'
8. Print results as: FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}, "well_B2": {"R2": 0.87, "peaks": [{"center": 680, "amplitude": 20, "sigma": 10}]}}

EXAMPLE OUTPUT:
FIT_RESULTS_JSON:{"well_A1": {"R2": 0.95, "peaks": [{"center": 520, "amplitude": 350, "sigma": 15}]}}

Write ONLY the Python code:"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS_ERROR = """You are an expert data scientist debugging a Python script. A previously generated script failed to execute. Your task is to analyze the error and provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1.  **Failed Script**: The exact Python code that produced the error.
2.  **Error Message**: The full traceback from the script's execution.

**Your Task:**
1.  Analyze the error message and traceback to identify the bug in the failed script.
2.  Generate a complete, corrected, and executable Python script that fixes the bug while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Failed Script
```python
{failed_script}
```
## Error Message
{error_message}
"""

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS = """You are an expert data scientist debugging a Python script. A previously generated script executed however, the curve fit was inadequate. Your task is to analyze the old script, fit plot, and the fitted parameters to provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use lmfit for fitting, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

**Provided Information:**
1. **Old Script**: The exact Python code that produced the curve fit.
2. **Curve Fit Plot**: The .png file of the curve fit plot produced by the old script.
3. **Fitted Parameters**: The fitted parameters of the curve fit plot including R2 value as well as the peaks.

**Your Task:**
1. Analyze the old script, curve fit plot, and fitted parameters to identify why the curve fit was inadequate.
2.  Generate a complete, corrected, and executable Python script that fixes the inadequacies while still fulfilling all original requirements.
3.  Ensure your entire response is ONLY the corrected Python code inside a markdown block. Do not add any conversational text.

## Old Script
```python
{old_script}
```
## Curve Fit Plot
{old_fit_plot_bytes}

## Fitted Parameters
{old_fitted_parameters}
"""

CLARIFY_QUESTION_INSTRUCTIONS = """ You are a careful research assistant clarifying a question given to you be a user. Your task is to rewrite the user's question to be precise and testable.
**Context:**
- The clarified question will be used to invite critical analysis from another entity

**Provided Information:**
1. **User Question**: Question given by a user.

**Your Task:**
1. Analyze the question to identify if it is precise and testable. 
2. Generate a complete, corrected question. Make sure to list any hidden assumptions and key terms with short definitions.
3. Ensure your entire response is ONLY the corrected question. Do not add any conversational text.

## User Question
{user_question}
"""

SOCRATIC_PASS_INSTRUCTIONS = """ You are a careful research assistant asking yourself probing questions that helps breakdown a question given by a user. Your task is ask yourself 3-5 probing questions with a reasoning for each.
**Provided Information:**
1. **User Question**: A clarified question given by the user, including hidden assumptions and key terms.
2. **Socratic Principles**: A list of socratic principles heuristic types and their key definitions.

**Your Task:**.
1. Using Socratic principles, analyze the user’s question by generating 3–5 probing questions and identifying any assumptions that expand on the question.
2. Provide a one-sentence explanation for each probing question, stating why you chose to ask it.
3. Ensure your entire response is ONLY the list of probing questions with their reasoning and assumptions. Do not add any conversational text.

## User Question
{user_question}

## Socratic Principles
{socratic_principles}
"""

TOT_INSTRUCTIONS = """ You are a careful research assistant producing distinct lines of thought based on a question and accompanying sub-questions. Your task is to produce three distinct reasonings that are then selected by the user.. 
**Provided Information:**
1. **User Question**: A clarified question given by the user, including hidden assumptions and key terms.
2. **Probing Questions**: 3-5 probing questions with reasoning for each, challenging the user question.

**Your Task:**
1. Analyze the user question and probing questions for reasoning, assumptions, and evidence.
2. Generate 3 distinct lines of thought that contain: (a) a brief narrative of the idea, (b) assumptions, (c) predicted outcomes, and (d) a next-step question for deepening.
3. Ensure the three lines of thought are diverse and are only one sentence.
4. Ensure your entire response is only the list of distinct lines of thought—do not add conversational text or extra sections.

## User Question
{user_question}

## Probing Questions
{probing_questions}
"""