"""
Generate dummy spectral and composition data for demo workflow.
Creates Cytation-style CSV with peaks systematically varying 450-850 nm,
plus a composition CSV for GP training.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def generate_demo_spectral_csv(
    n_wells: int = 15,
    wl_min: float = 450,
    wl_max: float = 900,
    wl_step: float = 1,
    peak_center_min: float = 450,
    peak_center_max: float = 850,
    peak_height: float = 500,
    peak_fwhm: float = 50,
    output_path: str = None,
) -> str:
    """
    Generate a Cytation-style spectral CSV with synthetic Gaussian peaks.
    Peak centers vary systematically from peak_center_min to peak_center_max across wells.

    Args:
        n_wells: Number of wells (default 15)
        wl_min, wl_max, wl_step: Wavelength range
        peak_center_min, peak_center_max: Range of peak centers across wells
        peak_height: Peak amplitude
        peak_fwhm: Full-width at half-max (sigma ≈ fwhm/2.355)
        output_path: Optional path to save; if None, uses temp file

    Returns:
        Path to generated CSV file
    """
    wavelength = np.arange(wl_min, wl_max + wl_step / 2, wl_step)
    n_points = len(wavelength)
    sigma = peak_fwhm / 2.355

    # Well names: A1-A12, B1-B3 for 96-well style (15 wells)
    rows_96 = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cols_96 = list(range(1, 13))
    all_wells = [f"{r}{c}" for r in rows_96 for c in cols_96]
    well_names = all_wells[:n_wells]

    # Peak center for each well: linear from min to max
    peak_centers = np.linspace(peak_center_min, peak_center_max, n_wells)

    # Build data matrix: rows = wavelength, cols = wells
    data = np.zeros((n_points, n_wells))
    np.random.seed(42)
    baseline_noise = 5

    for j, (well, center) in enumerate(zip(well_names, peak_centers)):
        gaussian = peak_height * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
        noise = np.random.normal(0, baseline_noise, n_points)
        data[:, j] = gaussian + np.abs(noise)

    # Cytation header block - all rows must have same column count for pandas parsing
    # "Read 1:EM Spectrum" must be in column 0 (no leading comma)
    n_data_cols = 2 + n_wells  # empty + wavelength + wells
    pad = "," * max(0, n_data_cols - 2)  # pad 2-col rows to n_data_cols
    header_lines = [
        "Software Version,3.08.01" + pad,
        "" + "," * (n_data_cols - 1),
        "Plate Type,96 WELL PLATE (Use plate lid)" + pad,
        "Read,Fluorescence Spectrum" + pad,
        ",Full Plate" + "," * (n_data_cols - 2),
        f',"Emission Start: {int(wl_min)}/20 nm,  Stop: {int(wl_max)} nm,  Step: {wl_step} nm"' + pad,
        "," * (n_data_cols - 1),
        "Read 1:EM Spectrum" + "," * (n_data_cols - 1),
        "," * (n_data_cols - 1),
        ",Wavelength," + ",".join(well_names),
    ]

    # Data rows: leading comma, wavelength, then well values (matches Cytation)
    data_lines = []
    for i in range(n_points):
        row_vals = [str(int(wavelength[i]))] + [f"{v:.1f}" for v in data[i, :]]
        data_lines.append("," + ",".join(row_vals))

    csv_content = "\n".join(header_lines + data_lines)

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".csv", prefix="demo_spectral_")
        os.close(fd)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(csv_content)
    return output_path


def generate_demo_composition_csv(
    well_names: list,
    material_a: str = "PEA2PbI4",
    material_b: str = "FAPbI3",
    ratio_range: Tuple[float, float] = (0, 100),
    output_path: str = None,
) -> str:
    """
    Generate composition CSV with two materials.
    Material A fraction varies linearly across wells (for correlation with peak wavelength).

    Args:
        well_names: List of well names (e.g. ['A1','A2',...,'B3'])
        material_a: Name of first material
        material_b: Name of second material
        ratio_range: (min, max) for material A percentage
        output_path: Optional path; if None, uses temp file

    Returns:
        Path to generated CSV file
    """
    n = len(well_names)
    a_frac = np.linspace(ratio_range[0], ratio_range[1], n)
    b_frac = 100 - a_frac

    df = pd.DataFrame(
        {
            material_a: a_frac,
            material_b: b_frac,
        },
        index=well_names,
    ).T
    df.index.name = ""

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".csv", prefix="demo_composition_")
        os.close(fd)
    df.to_csv(output_path)
    return output_path


def generate_demo_worklist(comp_path: str, max_vol: float = 50) -> str:
    """
    Generate a worklist CSV string from the demo composition file.
    Used for experimental_outputs so Analysis Agent has full context.
    """
    df = pd.read_csv(comp_path, index_col=0)
    materials = list(df.index)
    wells = list(df.columns)
    rows = []
    for well in wells:
        row = {"Well": well}
        total = df[well].sum()
        if total <= 0:
            total = 1
        for m in materials:
            frac = df.loc[m, well] / total
            row[f"{m}_uL"] = round(frac * max_vol, 1)
        rows.append(row)
    import io
    import csv
    buf = io.StringIO()
    fieldnames = ["Well"] + [f"{m}_uL" for m in materials]
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def generate_demo_dataset(
    n_wells: int = 15,
    output_dir: str = None,
) -> Tuple[str, str]:
    """
    Generate both spectral and composition CSV files for the demo workflow.

    Returns:
        (spectral_csv_path, composition_csv_path)
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="polaris_demo_")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    spectral_path = os.path.join(output_dir, "demo_spectral_data.csv")
    comp_path = os.path.join(output_dir, "demo_composition.csv")

    rows_96 = ["A", "B", "C", "D", "E", "F", "G", "H"]
    cols_96 = list(range(1, 13))
    all_wells = [f"{r}{c}" for r in rows_96 for c in cols_96]
    well_names = all_wells[:n_wells]

    generate_demo_spectral_csv(
        n_wells=n_wells,
        output_path=spectral_path,
    )
    generate_demo_composition_csv(
        well_names=well_names,
        output_path=comp_path,
    )
    return spectral_path, comp_path
