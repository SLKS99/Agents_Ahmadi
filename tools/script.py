import os
import re
from io import BytesIO
import numpy as np
import pandas as pd
import json
import logging

from matplotlib import pyplot as plt
from transformers import pipeline

from executor.script_executor import ScriptExecutor
from .instruct import FITTING_SCRIPT_CORRECTION_INSTRUCTIONS, FITTING_SCRIPT_GENERATION_INSTRUCTIONS

class CurveFitting:

    MAX_SCRIPT_ATTEMPTS = 3
    LUM_READ_NUMBERS = ('1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,'
                        '33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,'
                        '62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,'
                        '91,92,93,94,95,96,97,98,99,100')

    def __init__(self, output_dir: str = "curve_fitting", executor_timeout: int = 60, wells_to_ignore: str = "",
                 start_wavelength: int = 500, end_wavelength: int = 850, wavelength_step_size: int = 1, time_step: int = 100,
                 number_of_reads: int = 100, luminescence_read_numbers: list = LUM_READ_NUMBERS):
         self.luminescence_read_numbers = luminescence_read_numbers
         self.wells_to_ignore = wells_to_ignore
         self.number_of_reads = number_of_reads
         self.time_step = time_step
         self.start_wavelength = start_wavelength
         self.end_wavelength = end_wavelength
         self.wavelength_step_size = wavelength_step_size
         self.output_dir = output_dir
         self.executor = ScriptExecutor(timeout=executor_timeout)

    #Loading in the .csv files
    def load_data(self, data_path: str, comp_path: str):
        if data_path.endswith(".csv") & comp_path.endswith(".csv"):
            data = pd.read_csv(data_path, header=None)
            data = data.replace("OVRFLW", np.nan)

            composition = pd.read_csv(comp_path, index_col=0)
        else:
            raise ValueError(f"{data_path} or {comp_path} is not a .csv file.")

        return data, composition

    # Parsing data into dictionary -> converting to readable data frame
    def data_simplification(self, data: pd.DataFrame, composition: pd.DataFrame):
        wells_to_ignore = self.wells_to_ignore
        number_of_reads = self.number_of_reads
        luminescence_read_numbers = self.luminescence_read_numbers
        # time_step = self.time_step
        start_wavelength = self.start_wavelength
        end_wavelength = self.end_wavelength
        wavelength_step_size = self.wavelength_step_size

        # Make a list of cells to reference later
        cells = []

        for i in range(1, 9):
            for j in range(1, 13):
                cells.append(chr(64 + i) + str(j))

        if not wells_to_ignore:
            for i in wells_to_ignore:
                cells.remove(i)

        for i in wells_to_ignore:
            composition = composition.drop(i, axis=1)

        rows = []

        for i in range(1, number_of_reads + 1):
            rows += data[data[data.columns[0]] == "Read " + str(i) + ":EM Spectrum"].index.tolist()
            rows += data[data[data.columns[0]] == "Results"].index.tolist()

        #Seperate into different dataframes

        #Make a list of names
        names = []

        for i in range(1, number_of_reads + 1):
            names.append("Read " + str(i))

        #Make a dictionary
        d = {}

        for c in names:
            split_name = c.split(" ")
            index = int(split_name[1])
            d[c] = data[rows[index - 1] + 2 : rows[index] - 1] #Take a section of values
            d[c] = d[c].drop([0], axis=1) #Drop the empty column
            new_header = d[c].iloc[0] #Grab the first row for the header
            d[c] = d[c][1:] #Take the data less the header row
            d[c].columns = new_header #Set the header row as the df header
            if not wells_to_ignore:
                for i in wells_to_ignore:
                    d[c] = d[c].drop(i, axis=1)
            d[c] = d[c].astype(float) #Make sure it is composed of numbers

        #Converting Dictionary into readable data frame

        #Convert top luminescence list into an array
        #luminescence_time = np.array(luminescence_read_numbers)
        #luminescence_time = [int(i) * time_step for i in luminescence_time]

        #Convert wavelength into an array
        luminescence_wavelength = np.arange(start_wavelength, end_wavelength + wavelength_step_size, wavelength_step_size)

        #Make a grid
        #nx, ny = np.meshgrid(luminescence_wavelength, luminescence_time)

        #Load information into a dataframe
        luminescence_df = pd.DataFrame()

        for i in luminescence_read_numbers:
            temp_df = d["Read " + str(i)]
            #Assuming temp_df needs to be modified or used as is
            luminescence_df = pd.concat([luminescence_df, temp_df], ignore_index=True)

        luminescence_df = luminescence_df.fillna(0.0)
        luminescence_vec = np.array(luminescence_df)

        ldata = luminescence_vec.reshape([100, 351, 98])
        dat = ldata[20, :, 2]
        y = dat/100
        x = luminescence_wavelength

        return dat, x, y

    def plot_data(self, curve_data, title_suffix="") -> bytes:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(curve_data)
        ax.set_title("1D Data" + title_suffix)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.grid(True, linestyle='--')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='jpeg', dpi=150)
        buf.seek(0)
        image_bytes = buf.getvalue()
        plt.close(fig)

        return image_bytes


    def generate_fitting_script(self, curve_data, x, y, data_path: str) -> str:

        delta = None
        center = None

        pipe = pipeline("text_generation", model="google/gemma-3-4b-pt", device=0)

        logging.info("Generating fitting script...")
        prompt = (
            f"{FITTING_SCRIPT_GENERATION_INSTRUCTIONS}"
            f"## Curve Data Preview:\n {curve_data}"
            f"## Data File Path\nThe script should load data from this absolute path: '{os.path.abspath(data_path)}'\n"
            f"Y value: {y}\n"
            f"X value: {x}\n"
        )

        response = pipe(prompt)
        script_content = response.text
        match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
        if match:
            script_content = match.group(1).strip()
        else:
            if script_content.strip().startswith("import"):
                script_content = script_content.strip()
            else:
                logging.error(f"LLM response did not contain a valid python code block. Response: {script_content[:500]}")
                raise ValueError("LLM failed to generate a Python script in a markdown block")

        if not script_content:
            raise ValueError("LLM generated an empty fitting script")

        return script_content

    def generate_and_execute_fitting_script_with_retry(self, curve_data, x, y, data_path: str) -> dict:

        last_error = "No script generated yet"
        fitting_script = None

        for attempt in range(1, self.MAX_SCRIPT_ATTEMPTS + 1):
            try:
                if attempt == 1:
                    #First initial script generation
                    print(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Generating initial fitting script...")
                    fitting_script = self.generate_fitting_script(curve_data, x, y, data_path)
                else:
                    print(f"Attempt {attempt}/{self.MAX_SCRIPT_ATTEMPTS}: Script failed. Requesting correction from LLM...")
                    correction_prompt = FITTING_SCRIPT_CORRECTION_INSTRUCTIONS.format(
                        failed_script = fitting_script,
                        error_msg = last_error
                    )

                    pipe = pipeline("text_generation", model="google/gemma-3-4b-pt")
                    response = pipe(correction_prompt)
                    script_content = response.text
                    match = re.search(r"```python\n(.*?)\n```", script_content, re.DOTALL)
                    if match:
                        script_content = match.group(1).strip()
                    else:
                        fitting_script = script_content.strip()

                #Execute current version of the script
                print(f"Executing script...")
                exec_result = self.executor.execute_script(fitting_script, working_dir=self.output_dir)

                if exec_result.get("status") == "success":
                    print(f"Script executed successfully.")
                    return {
                        "status": "success",
                        "execution_result": exec_result,
                        "final_script": fitting_script,
                        "attempt": attempt
                    }
                else:
                    last_error = exec_result.get("message", "Unknown execution error")
                    logging.error(f"Script execution attempt {attempt} failed with error: {last_error}")

            except Exception as e:
                last_error = f"An error occurred during script generation: {str(e)}"
                logging.error(last_error, exc_info=True)

        #If loop finishes without success
        print(f"Script generation failed after {self.MAX_SCRIPT_ATTEMPTS} attempts.")
        return {
            "status": "error",
            "message": f"Failed to generate a working script after {self.MAX_SCRIPT_ATTEMPTS} attempts. Last error: {last_error}",
            "last_script":fitting_script
        }

    def analyze_curve_fitting(self, data_path: str, comp_path: str) -> dict:
        logging.info(f"Starting curve fitting analysis for: {data_path} and {comp_path}...")

        try:
            #Step 0: Load Data and Visualize
            initial_curve_data, initial_comp = self.load_data(data_path, comp_path)
            curve_data, x, y = self.data_simplification(initial_curve_data, initial_comp)
            original_plot_bytes = self.plot_data(curve_data, "Original Curve Data")

            #Step 1 & 2: Generate and Execute Fitting Script with Retry Logic
            print(f"---- ANALYZING CURVE FITTING: SCRIPT GENERATION & EXECUTION ----")
            script_execution = self.generate_and_execute_fitting_script_with_retry(curve_data, x, y, data_path)

            if script_execution["status"] != "success":
                raise RuntimeError(script_execution["message"])

            execution_result = script_execution["execution_result"]

            #Step 3: Parse Results
            fit_parameters = {}
            for line in execution_result.get("stdout", "").splitlines():
                if line.startswith("FIT_RESULTS_JSON:"):
                    fit_parameters = json.loads(line.replace("FIT_RESULTS_JSON:", ""))
                    break
            if not fit_parameters:
                raise ValueError("Could not parse fitting parameters from script output")

            fit_plot_path = os.path.join(self.output_dir, "fit_visualization.png")
            with open(fit_plot_path, "rb") as f:
                fit_plot_bytes = f.read()

            #Add results into dictionary
            final_result = {"analysis_images": [
                {"label": "Original Data Plot", "data": original_plot_bytes},
                {"label": "Fit Visualization", "data": fit_plot_bytes},
            ], "status": "success", "fitting_parameters": fit_parameters}

            return final_result

        except Exception as e:
            logging.exception(f"Curve analysis failed with error: {e}")
            return {"status": "error", "message": str(e)}






