import base64
from typing import Dict, Any

import streamlit as st
from agents.base import BaseAgent
from tools import socratic
from tools.memory import MemoryManager
import csv
import io

# Lazy import heavy modules
try:
    import requests
except ImportError:
    requests = None

class ExperimentAgent(BaseAgent):
    def __init__(self, name, desc, params_const):
        super().__init__("Experiment Agent", desc)
        self.params_const = params_const
        self.memory = MemoryManager()

    def confidence(self, params) -> float:
        """Return confidence score for experiment tasks."""
        # Low confidence by default (experiment agent usually needs user input)
        return 0.3

    def parse_worklist_from_plan(self, experimental_plan, materials):
        """Parse specific worklist details from experimental plan"""
        import re

        if not experimental_plan:
            return None

        # Look for worklist patterns in the experimental plan
        worklist_data = []

        # Pattern 1: CSV-style format like "A1,Cs_uL=20,BDA_uL=15,Solvent_uL=15"
        csv_pattern = r'([A-Z]\d{1,2}),([^;]+)'
        matches = re.findall(csv_pattern, experimental_plan)

        if matches:
            for well, mixture_str in matches:
                well_data = {'Well': well}

                # Parse mixture components like "Cs_uL=20,BDA_uL=15,Solvent_uL=15"
                components = mixture_str.split(',')
                for component in components:
                    component = component.strip()
                    if '=' in component:
                        mat, vol = component.split('=', 1)
                        mat = mat.strip()
                        try:
                            vol = float(vol.strip())
                            if mat in materials:
                                well_data[mat] = vol
                        except ValueError:
                            continue

                if len(well_data) > 1:  # Has well + at least one material
                    worklist_data.append(well_data)

        # If we found specific worklist data, return it
        if worklist_data:
            return worklist_data

        return None

    def generate_worklist(self, experimental_plan, plate_format="96-well", materials=None):
        """Generate a worklist CSV based on experimental plan - uses specific worklist if provided, otherwise creates varied mixing ratios"""
        import csv
        import io
        import re

        # Default materials if not provided
        if materials is None:
            materials = ["Cs_uL", "BDA_uL", "BDA_2_uL"]

        # Generate well IDs (Opentrons format: A01, A02, etc.)
        wells = []
        if plate_format == "96-well":
            for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                for col in range(1, 13):
                    wells.append(f"{row}{col:02d}")
        elif plate_format == "384-well":
            for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
                for col in range(1, 25):
                    wells.append(f"{row}{col:02d}")
        else:  # 24-well
            for row in ['A', 'B', 'C', 'D']:
                for col in range(1, 7):
                    wells.append(f"{row}{col:02d}")

        # Generate CSV content
        output = io.StringIO()
        fieldnames = ['Well'] + materials
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Get max volume from constraints
        max_vol = st.session_state.experimental_constraints["liquid_handling"]["max_volume_per_mixture"]

        # First, try to parse specific worklist from experimental plan
        parsed_worklist = self.parse_worklist_from_plan(experimental_plan, materials)
        if parsed_worklist:
            # Use the parsed specific worklist
            for well_data in parsed_worklist:
                writer.writerow(well_data)
            return output.getvalue()

        # Parse experimental plan to understand what to vary
        plan_lower = experimental_plan.lower() if experimental_plan else ""

        # Determine number of conditions to test based on plan
        # Look for keywords like "varying ratios", "different mixing ratios", "range of", etc.
        num_conditions = 24  # Default
        if "mixing ratio" in plan_lower or "ratio" in plan_lower:
            # Create a grid of mixing ratios
            if len(materials) == 2:
                # For 2 materials, create ratio variations (e.g., 90:10, 80:20, ..., 10:90)
                num_conditions = min(len(wells), 24)
                ratios = []
                for i in range(num_conditions):
                    # Create ratios from 5:95 to 95:5
                    ratio1 = 5 + (i * (90 // max(1, num_conditions - 1)))
                    ratio2 = 100 - ratio1
                    ratios.append((ratio1, ratio2))
            elif len(materials) >= 2:
                # For multiple materials, create systematic variations
                num_conditions = min(len(wells), 24)
            else:
                num_conditions = min(len(wells), 8)

        # Generate varied compositions based on experimental design
        num_wells = min(len(wells), num_conditions)

        for i, well in enumerate(wells[:num_wells]):
            row_data = {'Well': well}

            if len(materials) == 2:
                # Two materials: vary ratios systematically
                # Create ratios from 10:90 to 90:10 (varying material 1)
                if num_wells > 1:
                    # Create range from 10% to 90% in equal steps
                    ratio1_pct = 10 + (i * (80.0 / max(1, num_wells - 1)))
                else:
                    ratio1_pct = 50
                ratio2_pct = 100 - ratio1_pct

                # Calculate volumes with proper rounding
                vol1 = round((ratio1_pct / 100.0) * max_vol)
                vol2 = max_vol - vol1  # Ensure exact total

                # Make sure volumes are valid
                vol1 = max(0, min(max_vol, vol1))
                vol2 = max(0, min(max_vol, vol2))

                # Final check: ensure total is exactly max_vol
                if vol1 + vol2 != max_vol:
                    vol2 = max_vol - vol1

                row_data[materials[0]] = vol1
                row_data[materials[1]] = vol2

            elif len(materials) == 3:
                # Three materials: create systematic variations
                # Divide into groups: vary material 1, vary material 2, vary material 3
                groups = num_wells // 3
                if groups == 0:
                    groups = 1

                if i < groups:
                    # Vary material 1 (10-90% of max_vol)
                    mat1_vol = int(10 + (i * (80 / max(1, groups - 1))))
                    remaining = max_vol - mat1_vol
                    mat2_vol = remaining // 2
                    mat3_vol = remaining - mat2_vol
                elif i < groups * 2:
                    # Vary material 2
                    mat2_vol = int(10 + ((i - groups) * (80 / max(1, groups - 1))))
                    remaining = max_vol - mat2_vol
                    mat1_vol = remaining // 2
                    mat3_vol = remaining - mat1_vol
                else:
                    # Vary material 3
                    mat3_vol = int(10 + ((i - groups * 2) * (80 / max(1, groups - 1))))
                    remaining = max_vol - mat3_vol
                    mat1_vol = remaining // 2
                    mat2_vol = remaining - mat1_vol

                row_data[materials[0]] = max(0, mat1_vol)
                row_data[materials[1]] = max(0, mat2_vol)
                row_data[materials[2]] = max(0, mat3_vol)
            else:
                # For 4+ materials, create systematic grid
                # Distribute volumes with variations
                base_vol = max_vol // len(materials)
                variation = base_vol // 4

                for idx, mat in enumerate(materials):
                    # Create slight variations per material
                    vol = base_vol + (variation * (i % 3 - 1))  # -1, 0, or +1 variation
                    if idx == len(materials) - 1:
                        # Last material gets remainder to ensure total = max_vol
                        total_so_far = sum(row_data.get(m, 0) for m in materials[:-1])
                        vol = max_vol - total_so_far
                    row_data[mat] = max(0, min(max_vol, vol))

            writer.writerow(row_data)

        return output.getvalue()

    def upload_to_jupyter(self, server_url, token, file_content, filename, notebook_path):
        """Upload file to Jupyter server using Jupyter API"""
        try:
            # Clean up URL
            server_url = server_url.rstrip('/')
            if not server_url.startswith('http'):
                server_url = f"http://{server_url}"

            # Construct API endpoint
            api_path = f"{notebook_path}/{filename}"
            api_url = f"{server_url}/api/contents/{api_path}"

            # Prepare headers
            headers = {
                "Authorization": f"token {token}" if token else None
            }
            headers = {k: v for k, v in headers.items() if v is not None}

            # Prepare file content (base64 encoded for binary, plain text for text files)
            if filename.endswith('.csv') or filename.endswith('.py') or filename.endswith('.txt'):
                # Text files
                content_data = {
                    "type": "file",
                    "format": "text",
                    "content": file_content
                }
            else:
                # Binary files (base64 encoded)
                content_data = {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(
                        file_content.encode() if isinstance(file_content, str) else file_content).decode()
                }

            # Make PUT request to create/update file
            response = requests.put(
                api_url,
                json=content_data,
                headers=headers,
                timeout=10
            )

            if response.status_code in [200, 201]:
                return True, f"Successfully uploaded {filename} to {api_path}"
            else:
                return False, f"Failed to upload: {response.status_code} - {response.text}"

        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"
        except Exception as e:
            return False, f"Error uploading file: {str(e)}"

    def generate_opentrons_protocol(self, csv_filename, materials=None,
                                    csv_path="/var/lib/jupyter/notebooks/Dual GP 5AVA BDA/"):
        """Generate Opentrons Python protocol file that reads CSV and executes transfers
        Each material is in a separate tube, and materials are mixed in the destination well based on ratios"""
        if materials is None:
            materials = ["Cs_uL", "BDA_uL", "BDA_2_uL"]

        # Map materials to tube indices (one tube per material, no reservoirs)
        material_tube_mapping = []
        for idx, material in enumerate(materials):
            material_tube_mapping.append(f"        '{material}': {idx},  # Material in tube {idx}")

        protocol_code = f"""from opentrons import protocol_api
    import csv

    metadata = {{
        'protocolName': 'POLARIS Hypothesis Agent - Automated Liquid Handling',
        'author': 'POLARIS Hypothesis Agent',
        'description': 'Generated protocol for automated material mixing from separate tubes',
        'apiLevel': '2.13'
    }}

    def run(protocol: protocol_api.ProtocolContext):
        # Load labware
        left_tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', '1')
        tuberack = protocol.load_labware('opentrons_15_tuberack_nest_15ml_conical', '2')
        triple_cation_plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '3')

        # Load instruments
        left_pipette = protocol.load_instrument('p300_single', 'left', tip_racks=[left_tiprack])

        # Set flow rates
        left_pipette.flow_rate.aspirate = 30
        left_pipette.flow_rate.dispense = 150

        # Define z heights for tubes (all tubes use same z height)
        z_height = 117  # Standard z height for all tubes

        # Map each material to its tube index (one material per tube)
        material_to_tube = {{
    {chr(10).join(material_tube_mapping)}
        }}

        # Read CSV file
        csv_file_path = f'{csv_path}{csv_filename}'

        with open(csv_file_path, mode='r') as csvfile:
            csvreader = csv.DictReader(csvfile)

            # Process each well (row) in the CSV
            for row in csvreader:
                try:
                    well_id = row['Well']

                    # Collect all materials and volumes for this well
                    material_volumes = []
                    for material in material_to_tube.keys():
                        if material in row and row[material]:
                            try:
                                volume = float(row[material])
                                if volume > 0:
                                    material_volumes.append((material, volume))
                            except (ValueError, TypeError):
                                protocol.comment(f"Skipping {{material}} for well {{well_id}}: invalid volume")
                                continue

                    # If we have materials to transfer, mix them all into the same well
                    if material_volumes:
                        # Pick up a fresh tip for this well
                        left_pipette.pick_up_tip()

                        # Transfer each material to the same destination well
                        # This will mix them together in the correct ratios
                        for material, volume in material_volumes:
                            tube_index = material_to_tube[material]

                            # Transfer from source tube to destination well
                            left_pipette.transfer(
                                volume,
                                tuberack.wells()[tube_index].top(z=-z_height),
                                triple_cation_plate.wells_by_name()[well_id],
                                blowout_location='destination well',
                                blow_out=True,
                                new_tip='never',  # Use same tip for all materials in this well (mixing)
                                touch_tip=True
                            )

                        # Drop tip after mixing all materials for this well
                        left_pipette.drop_tip()

                except KeyError as e:
                    protocol.comment(f"Error processing row: missing key {{str(e)}}")
                    continue
                except Exception as e:
                    protocol.comment(f"Error processing row: {{str(e)}}")
                    continue
    """

        return protocol_code

    def generate_plate_layout(self, experimental_plan, plate_format="96-well", worklist_content=None):
        """Generate a visual plate layout based on actual worklist data"""

        layout = []
        layout.append("# Well Plate Layout")
        layout.append(f"# Format: {plate_format}")
        layout.append("")

        # Parse worklist to get actual compositions
        well_data = {}
        if worklist_content:
            try:
                reader = csv.DictReader(io.StringIO(worklist_content))
                for row in reader:
                    well_id = row.get('Well', '')
                    if well_id:
                        # Extract material volumes
                        mat_vols = []
                        for key, val in row.items():
                            if key != 'Well' and val:
                                try:
                                    vol = float(val)
                                    if vol > 0:
                                        mat_name = key.replace('_uL', '')
                                        mat_vols.append((mat_name, vol))
                                except:
                                    pass
                        if mat_vols:
                            well_data[well_id] = mat_vols
            except Exception as e:
                pass

        if plate_format == "96-well":
            layout.append("    01     02     03     04     05     06     07     08     09     10     11     12")
            for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                row_layout = [f"{row} "]
                for col in range(1, 13):
                    well_id = f"{row}{col:02d}"
                    if well_id in well_data and well_data[well_id]:
                        # Show actual composition
                        mat_vols = well_data[well_id]
                        total_vol = sum(v for _, v in mat_vols)

                        if len(mat_vols) == 2 and total_vol > 0:
                            # Show ratio for 2 materials (e.g., "70:30")
                            ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                            ratio2 = 100 - ratio1
                            content = f"{ratio1}:{ratio2}"
                        elif len(mat_vols) == 3 and total_vol > 0:
                            # Show ratios for 3 materials (e.g., "50:30:20")
                            ratios = [int((v / total_vol) * 100) for _, v in mat_vols]
                            content = ":".join([str(r) for r in ratios])
                        else:
                            # Show volumes for multiple materials
                            content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                            if len(mat_vols) > 2:
                                content += "+"
                    else:
                        content = "---"
                    row_layout.append(f"{content:>7}")
                layout.append(" ".join(row_layout))
        elif plate_format == "384-well":
            # Similar for 384-well
            layout.append("# 384-well plate layout (showing first 24 wells as example)")
            for row in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
                row_layout = [f"{row} "]
                for col in range(1, 25):
                    well_id = f"{row}{col:02d}"
                    if well_id in well_data and well_data[well_id]:
                        mat_vols = well_data[well_id]
                        total_vol = sum(v for _, v in mat_vols)
                        if len(mat_vols) == 2 and total_vol > 0:
                            ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                            ratio2 = 100 - ratio1
                            content = f"{ratio1}:{ratio2}"
                        else:
                            content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                    else:
                        content = "---"
                    row_layout.append(f"{content:>6}")
                layout.append(" ".join(row_layout))
        else:  # 24-well
            layout.append("    01     02     03     04     05     06")
            for row in ['A', 'B', 'C', 'D']:
                row_layout = [f"{row} "]
                for col in range(1, 7):
                    well_id = f"{row}{col:02d}"
                    if well_id in well_data and well_data[well_id]:
                        mat_vols = well_data[well_id]
                        total_vol = sum(v for _, v in mat_vols)
                        if len(mat_vols) == 2 and total_vol > 0:
                            ratio1 = int((mat_vols[0][1] / total_vol) * 100)
                            ratio2 = 100 - ratio1
                            content = f"{ratio1}:{ratio2}"
                        else:
                            content = "/".join([f"{int(v)}" for _, v in mat_vols[:2]])
                    else:
                        content = "---"
                    row_layout.append(f"{content:>7}")
                layout.append(" ".join(row_layout))

        layout.append("")
        layout.append("# Legend:")
        layout.append("# Numbers show mixing ratios (e.g., 70:30 = 70% Material 1, 30% Material 2)")
        layout.append("# Volumes shown as Material1:Material2 or Material1:Material2:Material3")

        return "\n".join(layout)

    def get_experimental_context(self):
        """Get experimental constraints as context for hypothesis generation"""
        constraints = st.session_state.experimental_constraints
        context = ""

        # Add explicit constraint header
        context += "=== EXPERIMENTAL CONSTRAINTS (STRICT - DO NOT DEVIATE) ===\n\n"

        if constraints["techniques"]:
            context += f"ONLY USE THESE EXPERIMENTAL TECHNIQUES (DO NOT MENTION OTHERS): {', '.join(constraints['techniques'])}\n"
            context += f"CRITICAL: Do NOT suggest, mention, or reference any techniques NOT in this list.\n"
            context += f"For example, if only 'time-resolved PL' is listed, DO NOT mention XRD, DFT, SEM, TEM, or any other technique.\n\n"
        else:
            context += "WARNING: No experimental techniques specified. Use only basic characterization methods.\n\n"

        if constraints["equipment"]:
            context += f"ONLY USE THIS EQUIPMENT (DO NOT SUGGEST ALTERNATIVES): {', '.join(constraints['equipment'])}\n\n"
        else:
            context += "No specific equipment constraints.\n\n"

        if constraints["parameters"]:
            context += f"ONLY FOCUS ON THESE PARAMETERS: {', '.join(constraints['parameters'])}\n"
            context += f"Do NOT introduce additional parameters not listed here.\n\n"
        else:
            context += "No specific parameter constraints.\n\n"

        if constraints["focus_areas"]:
            context += f"Primary focus areas: {', '.join(constraints['focus_areas'])}\n\n"

        # Add liquid handling context
        context += "=== LIQUID HANDLING CONSTRAINTS ===\n"
        lh = constraints["liquid_handling"]
        if lh["instruments"]:
            context += f"Liquid handling instruments: {', '.join(lh['instruments'])}\n"
        if lh["materials"]:
            context += f"Available materials: {', '.join(lh['materials'])}\n"
        context += f"Plate format: {lh['plate_format']}\n"
        context += f"Maximum volume per mixture: {lh['max_volume_per_mixture']} µL\n"
        context += f"Generate worklists and well plate layouts for automated liquid handling\n\n"

        # Add final reminder
        context += "=== REMINDER ===\n"
        context += "STRICTLY adhere to the listed techniques and equipment. Do NOT suggest alternatives or additional methods.\n"

        return context

    def run_agent(self, memory):
        constraints = st.session_state.experimental_constraints

        # Check for API key
        if not st.session_state.get("api_key"):
            st.warning("Please enter your API key in Settings before continuing.")
            st.stop()

        # Get experimental context
        experimental_context = self.get_experimental_context()
        
        # Try to get hypothesis from memory first, then check for manual inputs
        hypothesis = self.memory.view_component("hypothesis")
        if not hypothesis:
            hypothesis = st.session_state.get("manual_hypothesis", "")
        
        # Get clarified question - try memory first, then manual input
        clarified_question = self.memory.view_component("clarified_question")
        if not clarified_question:
            clarified_question = st.session_state.get("manual_clarified_question", "")
        if not clarified_question:
            clarified_question = "How can we test this research question?"
        
        # Get socratic questions - try memory first, then manual input
        socratic_questions = self.memory.view_component("socratic_pass")
        if not socratic_questions:
            socratic_questions = st.session_state.get("manual_socratic_questions", "")
        if not socratic_questions:
            socratic_questions = "What experimental approaches are needed?"
        
        # Check if we have minimum required inputs for LLM generation
        # We need both clarified_question and socratic_questions for tot_generation_experimental_plan
        has_required_inputs = clarified_question.strip() and socratic_questions.strip()
        
        if not has_required_inputs:
            st.warning("⚠️ Missing required inputs for experimental plan generation.")
            st.markdown("**Required inputs:**")
            st.markdown("1. ✅ **Clarified Question** - " + ("Provided" if clarified_question.strip() else "❌ Missing"))
            st.markdown("2. ✅ **Socratic Questions** - " + ("Provided" if socratic_questions.strip() else "❌ Missing"))
            st.markdown("\n**Optional inputs (helpful but not required):**")
            st.markdown("- Socratic Answers")
            st.markdown("- Three Lines of Thought")
            st.markdown("- Hypothesis")
            st.info("💡 **Tip:** Use the 'Manual Input' expander above to provide the required components, or generate a hypothesis using the Hypothesis Agent first.")
            st.stop()

        # Display what inputs we're using
        with st.expander("📋 Inputs Being Used", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Source:**")
                st.write("✅ Clarified Question: " + ("Memory" if self.memory.view_component("clarified_question") else "Manual Input"))
                st.write("✅ Socratic Questions: " + ("Memory" if self.memory.view_component("socratic_pass") else "Manual Input"))
                st.write("📝 Hypothesis: " + ("Memory" if self.memory.view_component("hypothesis") else ("Manual Input" if st.session_state.get("manual_hypothesis") else "Not provided")))
            with col2:
                st.markdown("**Preview:**")
                st.text_area("Clarified Question:", clarified_question[:200] + "..." if len(clarified_question) > 200 else clarified_question, height=60, disabled=True, key="preview_clarified")
                st.text_area("Socratic Questions:", socratic_questions[:200] + "..." if len(socratic_questions) > 200 else socratic_questions, height=60, disabled=True, key="preview_socratic")

        # Save manual inputs to memory if they were provided
        if st.session_state.get("manual_clarified_question"):
            self.memory.insert_interaction("user", st.session_state.manual_clarified_question, "clarified_question", "experiment")
        if st.session_state.get("manual_socratic_questions"):
            self.memory.insert_interaction("assistant", st.session_state.manual_socratic_questions, "socratic_pass", "experiment")
        if st.session_state.get("manual_socratic_answers"):
            self.memory.insert_interaction("assistant", st.session_state.manual_socratic_answers, "socratic_answers", "experiment")
        if st.session_state.get("manual_hypothesis"):
            self.memory.insert_interaction("assistant", st.session_state.manual_hypothesis, "hypothesis", "experiment")
            hypothesis = st.session_state.manual_hypothesis

        # Generate experimental plan using LLM
        with st.spinner("Generating experimental plan with LLM..."):
            try:
                experimental_plans = socratic.tot_generation_experimental_plan(
                    socratic_questions,
                    clarified_question,
                    experimental_context
                )
                
                if not experimental_plans or len(experimental_plans) == 0:
                    st.error("Failed to generate experimental plans. Please check your API key and try again.")
                    st.stop()
                
                # Use the first plan (or fallback to hypothesis if available)
                experimental_plan = experimental_plans[0] if experimental_plans[0] else (hypothesis or "Experimental plan generation failed. Please check your inputs and try again.")
                
                # Display the generated plan
                st.subheader("Generated Experimental Plan")
                with st.expander("View Full Experimental Plan", expanded=True):
                    st.markdown(experimental_plan)
                
                # Log the plan
                self.memory.insert_interaction("assistant", experimental_plan, "experimental_plan", "experiment")
                
            except Exception as e:
                st.error(f"Error generating experimental plan: {e}")
                st.info("Falling back to hypothesis-based worklist generation...")
                experimental_plan = hypothesis

        lh = constraints["liquid_handling"]
        plate_format = lh["plate_format"]
        materials = lh["materials"] or ["Cs_uL", "BDA_uL"]

        csv_materials = [
            m if m.endswith("_uL") else f"{m}_uL" for m in materials
        ]

        with st.spinner("Generating experimental artifacts (worklist, layout, protocol)..."):
            worklist = self.generate_worklist(experimental_plan, plate_format, csv_materials)
            layout = self.generate_plate_layout(experimental_plan, plate_format, worklist)

            protocol = None
            if "Opentrons" in lh["instruments"]:
                protocol = self.generate_opentrons_protocol(
                    "worklist.csv", csv_materials, lh["csv_path"]
                )

        # Display worklist preview
        st.subheader("Generated Worklist")
        with st.expander("View Worklist (CSV)", expanded=False):
            st.code(worklist, language="csv")

        # Display plate layout
        st.subheader("Plate Layout")
        with st.expander("View Plate Layout", expanded=False):
            st.code(layout, language="text")

        # Downloads
        st.subheader("Download Files")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "Download Worklist (.csv)",
                worklist,
                "worklist.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            st.download_button(
                "Download Plate Layout (.txt)",
                layout,
                "plate_layout.txt",
                "text/plain",
                use_container_width=True
            )

        with col3:
            if protocol:
                st.download_button(
                    "Download Opentrons Protocol (.py)",
                    protocol,
                    "opentrons_protocol.py",
                    "text/x-python",
                    use_container_width=True
                )
            else:
                st.info("No Opentrons protocol (Opentrons not in instruments list)")

        # Jupyter Upload (optional)
        if st.session_state.jupyter_config.get("upload_enabled"):
            st.subheader("Jupyter Integration")
            if st.button("Upload to Jupyter", use_container_width=True):
                success, message = self.upload_to_jupyter(
                    st.session_state.jupyter_config["server_url"],
                    st.session_state.jupyter_config["token"],
                    worklist,
                    "worklist.csv",
                    st.session_state.jupyter_config["notebook_path"]
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)

        # Save experimental outputs
        st.session_state.experimental_outputs = {
            "plan": experimental_plan,
            "worklist": worklist,
            "layout": layout,
            "protocol": protocol
        }
        
        self.memory.log_event("experiment_complete", {
            "plan": experimental_plan[:500],  # Truncate for storage
            "plate_format": plate_format,
            "materials": materials
        }, "experiment")
        
        # Record experiment in memory system
        try:
            from tools.experiment_memory import get_experiment_memory
            experiment_memory = get_experiment_memory()
            
            # Generate experiment ID from worklist
            import hashlib
            worklist_hash = hashlib.md5(worklist.encode()).hexdigest()[:8]
            experiment_id = f"exp_{worklist_hash}"
            
            # Extract composition information from worklist if available
            composition = {}
            if materials:
                composition = {mat: 0 for mat in materials}  # Placeholder - could parse from worklist
            
            # Check if experiment already exists
            if experiment_memory.has_experiment(experiment_id):
                st.info(f"ℹ️ Experiment {experiment_id} already exists in memory.")
            else:
                experiment_memory.add_experiment(
                    experiment_id=experiment_id,
                    description=f"Experimental plan: {experimental_plan[:200]}...",
                    composition=composition,
                    metadata={
                        "plate_format": plate_format,
                        "materials": materials,
                        "plan_length": len(experimental_plan),
                    }
                )
                st.success(f"✅ Experiment {experiment_id} recorded in memory!")
        except Exception as e:
            st.warning(f"Could not record experiment in memory: {e}")

        st.success("✅ Experimental planning complete!")