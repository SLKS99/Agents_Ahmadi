FITTING_SCRIPT_GENERATION_INSTRUCTIONS = """ You are an expert data scientist. Your task is to write a Python script to fit a 1D data curve using an appropriate physical model given some parameters.

First, think step by step:
1. **Analyze the Data Shape:** Visually inspect the curve provided in the prompt. Does it have one peak, multiple peaks? An absorption edge? A combination of features (e.g., peaks on a baseline)
2. **Select a Model:** Based on your analysis, choose an appropriate model. If there are multiple features, the model MUST be *a sum of multiple functions* (e.g., `gaussian1 + gaussian2 + linear_baseline`)
3. **Plan the Script:** Plan the full script, including defining the composite model function, making reasonable initial guesses for **all** parameters and calling the fitting routine. Good initial guesses are critical for complex fits to converge.

Then generate a *complete* and *executable* Python script that follows these rules: 
1. The script MUST include all necessary imports (`matplotlib.pyplot`, `numpy`, `scipy.optimize.curve_fit`, `json`)
2. The script MUST load data from the specified file path.
3. The script MUST define the chosen fitting function(s). For multiple features, this should be a composite function (e.g., `def double_gaussian(x, a1, c1, s1, a2, c2, s2): return gaussian(x, a1, c1, s1) + gaussian(x, a2, c2, s2)`).
4.  The script MUST perform the fit using `scipy.optimize.curve_fit`.
5.  The script MUST save a plot of the data and the complete fit (including all components) to a file named `fit_visualization.png`.
6.  **CRITICALLY**: After saving the plot, the script MUST print the final, optimized parameters for **all components** to standard output as a JSON string on a single line, prefixed with `FIT_RESULTS_JSON:`.
7.  Your entire response must be ONLY the Python code. Do NOT add any conversational text or explanations outside of the code itself. """

FITTING_SCRIPT_CORRECTION_INSTRUCTIONS = """You are an expert data scientist debugging a Python script. A previously generated script failed to execute. Your task is to analyze the error and provide a corrected version.

**Context:**
- The script is intended to fit 1D experimental data using a physical model.
- The script MUST load data, define a fitting function, use `scipy.optimize.curve_fit`, save a plot to `fit_visualization.png`, and print the final parameters as a JSON string prefixed with `FIT_RESULTS_JSON:`.

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