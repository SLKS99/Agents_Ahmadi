import json
from tools.script import CurveFitting

#Instantiate Curve Fitting Agent
curve_agent = CurveFitting()

#Define path to curve data and comp file
data_path = "PEASnPbI4-Toluene.csv"
comp_file = "2D-3D (1).csv"

#Run analysis
result = curve_agent.analyze_curve_fitting(data_path = data_path, comp_path = comp_file)

#Print results
print(f"--- Fitting and Analysis Summary ---\n")
print(result)

print("\n--- Fitted Parameters ---\n")
print(json.dumps(result.get("fitting parameters", {}), indent=2 ))


