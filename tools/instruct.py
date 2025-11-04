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

TOT_INSTRUCTIONS = """ You are a careful research assistant producing distinct lines of thought based on a question and accompanying sub-questions. Your task is to produce three distinct reasoning that are then selected by the user.. 
**Provided Information:**
1. **User Question**: A clarified question given by the user, including hidden assumptions and key terms.
2. **Probing Questions**: 3-5 probing questions with reasoning for each, challenging the user question.

**Your Task:**
1. Analyze the user question and probing questions for reasoning, assumptions, and evidence.
2. Generate 3 distinct lines of thought that contain: (a) a brief narrative of the idea, (b) assumptions, (c) predicted outcomes, and (d) a next-step question for deepening.
3. Ensure the three lines of thought are diverse and are only one sentence. Make sure that each thought is completely different from the next one. 
4. Ensure your entire response is only the list of distinct lines of thought. Do not add conversational text or extra sections.
5. Ensure that each line of thought starts with "Distinct Line of Thought".

## User Question
{user_question}

## Probing Questions
{probing_questions}
"""

RETRY_THINKING_INSTRUCTIONS = """ You are a careful research assistant analyzing a given thought and producing a question based on your analysis. Your task is to produce a socratic question that is precise and testable and next-step options. 
**Provided Information:**
1. **Line of Thought**: A line of thought selected by the user that includes: (a) a brief narrative of the idea, (b) assumptions, (c) predicted outcomes, and (d) a next-step question for further exploration.
2. **Previous Thoughts**: Earlier lines of thought that were generated but not selected by the user.
3. **Initial Clarified Question**: The original question that the previous thoughts were based on.

**Your Task:**
1. Analyze the provided line of thought for its core idea, underlying assumptions, predicted outcomes, and next-step questions for further exploration.
2. Generate one Socratic question and two to three next-step options that guide the reasoning toward a precise, testable hypothesis for an experiment, based on the given line of thought and your analysis.
3. Ensure the question and next-step options are diverse, concise (one sentence each), and do not repeat ideas from previous thoughts. Exclude a "Next-Step Options:" heading as well.
4. Ensure each next-step option is completely distinct from the others. 
5. Ensure your entire response is only the list of distinct lines of thought. Do not add conversational text or extra sections.

## Line of Thought
{line_of_thought}

## Previous Thoughts
{previous_thoughts_1}
{previous_thoughts_2}

## Initial Clarified Question
{initial_clarified_question}
"""

HYPOTHESIS_SYNTHESIS = """ You are a careful research assistant analyzing a given option and socratic question to develop a hypothesis. Your task is to produce a precise and testable hypothesis for experimentation.
**Provided Information:**
1. **Socratic Question:** A clarified question developed from a distinct line of thought that the next-step options are based on.
2. **Next-Step Option:** The option selected by the user, which should be analyzed, explored further, and used to generate a hypothesis.
3. **Previous Options:** A list of next-step options that were not selected by the user.

**Your Task:**
1. Analyze the Socratic question and next-step option for their ideas, evidence, terminology, and other relevant details.
2. Generate a precise, testable hypothesis that includes both predictions and potential tests.
3. Ensure the hypothesis, predictions, and tests are each concise (one sentence each) and do not repeat ideas from the previous options.
4. Include a heading with a colon on a new line for each section (e.g., Hypothesis: ..., Predictions: ...).
5. Ensure your entire response is only the hypothesis, predictions, and tests. Do not add conversational text or extra sections.

## Socratic Question
{socratic_question}

## Next-Step Option
{next_step_option}

## Previous Step Options
{previous_step_option_1}
{previous_step_option_2}
"""

HYPOTHESIS_ANALYSIS_REPORT = """ You are a careful research assistant analyzing a hypothesis on a set of criteria and previous research. Your task is to generate a report grading the hypothesis.
**Provided Information:**
1. **Hypothesis:** A precise, testable statement that includes both predictions and proposed tests.
2. **Socratic Question:** The initial question on which the hypothesis is based.

**Your Task:**
1. Analyze the hypothesis, predictions, and tests, and evaluate them based on a search of scientific journals across the internet.
2. Generate a report that assesses the hypothesis in terms of novelty, plausibility, and testability.
3. Ensure your entire response is only the report. Do not add conversational text or extra sections.

## Hypothesis
{hypothesis}

## Socratic Question
{socratic_question}
"""
