
# ERIVN-ISM and MICMAC Analysis Tool

This repository provides a Python-based tool that performs full-scale **ERIVN-ISM** and **MICMAC analysis** on expert evaluations. The tool is designed for researchers and decision-makers who aim to analyze the interrelationships among factors using interval-valued neutrosophic logic and classify them based on their systemic roles.

---

## 🧠 What It Does

- Accepts expert evaluations in **linguistic SVNN format** from an Excel file.
- Aggregates expert opinions using the **Empirical Rule** to form **Interval-Valued Neutrosophic Numbers (IVNNs)**.
- Applies a **deneutrosophication** operator to generate a crisp decision matrix.
- Constructs:
  - Initial and Final Reachability Matrices
  - Hierarchical ISM structure with factor levels
  - MICMAC matrix with **Driving** and **Dependence** powers
- Classifies factors into:
  - **Driving**, **Dependent**, **Linkage**, and **Autonomous**
- Generates:
  - **MICMAC quadrant scatter plot**
  - **ISM level diagram** with directional arrows
- Exports all results to an Excel file (`ERIVN_ISM_MICMAC_Results.xlsx`)

---

## 📁 Input Format

- The tool prompts the user to select an **Excel (.xlsx)** file.
- The file should contain **expert evaluations** stacked vertically. If there are `n` factors and `q` experts, the total number of rows should be `q × n`, each of `n` columns.
- Evaluations must be coded using linguistic scale values (e.g., 0 to 4), as mapped to SVNN values in the script.

---

## ▶️ How to Run

1. Clone the repository or download the `main.py` file.
2. Make sure the following Python libraries are installed:
   - `numpy`
   - `pandas`
   - `matplotlib`
   - `openpyxl`
   - `tkinter`
3. Run the script:
   ```bash
   python main.py
   ```
4. Select your Excel file with expert inputs when prompted.

---

## 📊 Output Files

After execution, the following files will be generated:

- `ERIVN_ISM_MICMAC_Results.xlsx` — contains:
  - IVNN Matrix
  - Crisp Decision Matrix
  - Initial & Final Reachability Matrices
  - Factor Levels (ISM hierarchy)
  - MICMAC Results with classification
- `MICMAC_Results.pdf` — quadrant diagram with factor positioning
- `Factor_Levels.pdf` — ISM level diagram showing structural arrows

---

## 👤 Author

**Ahmet ÖZTEL**  
Bartın University  
Email: aoztel@bartin.edu.tr

---

## 📃 License

This project is open-source and available under the MIT License.
