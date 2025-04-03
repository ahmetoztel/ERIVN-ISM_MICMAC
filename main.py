import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font
import matplotlib.patches as patches
import math
# --- SVNN Linguistic Scale ---
svnn_scale = {
    0: {"T": 0.1, "I": 0.8, "F": 0.9},
    1: {"T": 0.35, "I": 0.6, "F": 0.7},
    2: {"T": 0.5, "I": 0.4, "F": 0.45},
    3: {"T": 0.8, "I": 0.2, "F": 0.15},
    4: {"T": 0.9, "I": 0.1, "F": 0.1}
}

# --- Dosya seçimi ve veri okuma ---
Tk().withdraw()
file_path = askopenfilename(title="Select the expert opinions file", filetypes=[("Excel files", "*.xlsx *.xls")])
data = pd.read_excel(file_path, header=None)
non_empty_columns = data.dropna(axis=1, how='all')
non_empty_rows = data.dropna(axis=0, how='all')

factor_count = non_empty_columns.shape[1]
expert_count = non_empty_rows.shape[0] // factor_count

expert_opinions = np.array(non_empty_columns).reshape((expert_count, factor_count, factor_count))

# --- IVNN ve Crisp Karar Matrisi oluşturma ---
IVNN_matrix = np.empty((factor_count, factor_count), dtype=object)
Crisp_matrix = np.zeros((factor_count, factor_count))

for i in range(factor_count):
    for j in range(factor_count):
        T_vals, I_vals, F_vals = [], [], []
        for exp in range(expert_count):
            value = int(expert_opinions[exp, i, j])
            svnn = svnn_scale.get(value, {"T": 0.0, "I": 1.0, "F": 1.0})
            T_vals.append(svnn["T"])
            I_vals.append(svnn["I"])
            F_vals.append(svnn["F"])

        T_mean, I_mean, F_mean = np.mean(T_vals), np.mean(I_vals), np.mean(F_vals)
        T_std, I_std, F_std = np.std(T_vals, ddof=1), np.std(I_vals, ddof=1), np.std(F_vals, ddof=1)
        T_L, T_U = max(0, T_mean - T_std), min(1, T_mean + T_std)
        I_L, I_U = max(0, I_mean - I_std), min(1, I_mean + I_std)
        F_L, F_U = max(0, F_mean - F_std), min(1, F_mean + F_std)

        IVNN_matrix[i, j] = {"T": [T_L, T_U], "I": [I_L, I_U], "F": [F_L, F_U]}

        numerator = (T_L + T_U + (1 - F_L) + (1 - F_U) + (T_L * T_U) + np.sqrt(abs((1 - F_L) * (1 - F_U)))) / 6
        denominator = ((1 - (I_L + I_U) / 2) * np.sqrt(abs((1 - I_L) * (1 - I_U)))) / 2
        Crisp_matrix[i, j] = numerator * denominator if denominator != 0 else 0

# --- Eşik Değeri ve Erişilebilirlik Matrisi ---
threshold_value = np.mean(Crisp_matrix)
IRM = np.zeros((factor_count, factor_count))
for i in range(factor_count):
    for j in range(factor_count):
        if Crisp_matrix[i, j] >= threshold_value or i == j:
            IRM[i, j] = 1

FRM = IRM.copy()
for k in range(factor_count):
    for i in range(factor_count):
        for j in range(factor_count):
            FRM[i, j] = max(FRM[i, j], min(FRM[i, k], FRM[k, j]))

# --- Backup FRM after transitive closure ---
FRM_backup = FRM.copy()

# --- MICMAC Analysis ---
DRIVING_POWER = FRM.sum(axis=1)
DEPENDENCE_POWER = FRM.sum(axis=0)
micmac_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(factor_count)],
    'Driving Power': DRIVING_POWER,
    'Dependence Power': DEPENDENCE_POWER
})

factor_colors = {
    'Driving': 'FF0000',
    'Linkage': '0000FF',
    'Dependent': '008000',
    'Autonomous': 'FFD700'
}

micmac_df['Factor Type'] = micmac_df.apply(lambda row: (
    'Driving' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] <= factor_count/2 else
    'Linkage' if row['Driving Power'] > factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Dependent' if row['Driving Power'] <= factor_count/2 and row['Dependence Power'] > factor_count/2 else
    'Autonomous'
), axis=1)

micmac_df['Color'] = micmac_df['Factor Type'].map(factor_colors)

# --- MICMAC Analysis Scatter Plot ---
plt.figure(figsize=(10, 8))

coordinate_groups = {}
for i in range(factor_count):
    coord = (DRIVING_POWER[i], DEPENDENCE_POWER[i])
    if coord not in coordinate_groups:
        coordinate_groups[coord] = []
    coordinate_groups[coord].append(f'{i+1}')

for coord, factors in coordinate_groups.items():
    color = f'#{micmac_df[micmac_df["Factor"] == f"Factor {factors[0]}"]["Color"].values[0]}'
    label = r"$C_{" + ",".join(factors) + r"}$"
    plt.scatter(coord[0], coord[1], color=color)
    plt.text(coord[0] + 0.1, coord[1] + 0.1, label, fontsize=12, color=color, fontweight='bold')

ax = plt.gca()

import matplotlib.ticker as ticker

# --- Faktör sayısı ve orta çizgi konumu ---
mid = factor_count / 2

# --- Ortaya quadrant çizgileri ---
plt.axvline(x=mid, color='black', linestyle='--')
plt.axhline(y=mid, color='black', linestyle='--')

# --- Eksen maksimumunu hesapla ---
# En yakın bir üst çift sayıya yuvarlayalım
raw_max = int(np.ceil(factor_count * 1.2))  # örn: 14 * 1.2 = 16.8 → 17
axis_max = math.ceil(factor_count / 2) * 2 + 2



# --- Eksen sınırlarını ayarla ---
plt.xlim(0, axis_max)
plt.ylim(0, axis_max)

# --- Tick aralıklarını 2 birim yap ---
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(2))

# --- Kare orantı ayarı ---
plt.gca().set_aspect('equal', adjustable='box')








plt.text(factor_count * 0.9, factor_count * 0.95, "Linkage", fontsize=24, color='blue')
plt.text(factor_count * 0.1, factor_count * 0.95, "Dependent", fontsize=24, color='green')
plt.text(factor_count * 0.9, factor_count * 0.4, "Driving", fontsize=24, color='red')
plt.text(factor_count * 0.1, factor_count * 0.4, "Autonomous", fontsize=24, color='orange')

plt.xlabel('Driving Power', color='black', fontsize=16)
plt.ylabel('Dependence Power', color='black', fontsize=16)
plt.title('MICMAC Analysis Results', color='black', fontsize=24)
plt.grid(True)

# BU 3 SATIRI SİL ⛔️
# plt.xlim(0, factor_count)
# plt.ylim(0, factor_count)
# plt.gca().set_aspect('equal', adjustable='datalim')

plt.tight_layout()
plt.savefig('MICMAC_Results.pdf', format='pdf', bbox_inches='tight')
plt.show()










# --- Factor Levels Determination ---
levels = []
remaining_factors = set(range(factor_count))
level = 1
while remaining_factors:
    current_level_factors = []
    for factor in remaining_factors:
        reachability_set = set(np.where(FRM[factor] == 1)[0])
        antecedent_set = set(np.where(FRM[:, factor] == 1)[0])
        intersection_set = reachability_set & antecedent_set
        if reachability_set == intersection_set:
            current_level_factors.append(factor)
    if not current_level_factors:
        break
    for factor in current_level_factors:
        levels.append({
            'Factor': factor + 1,
            'Level': level,
            'Reachability Set': list(map(int, np.where(FRM[factor] == 1)[0] + 1)),
            'Antecedent Set': list(map(int, np.where(FRM[:, factor] == 1)[0] + 1)),
            'Intersection Set': list(map(int, [x + 1 for x in (set(np.where(FRM[factor] == 1)[0]) & set(np.where(FRM[:, factor] == 1)[0]))]))
        })
        remaining_factors.remove(factor)
        FRM[factor, :] = 0
        FRM[:, factor] = 0
    level += 1

levels_df = pd.DataFrame(levels)

# --- Final Reachability Matrix with Trace ---
FRM_formatted = FRM_backup.astype(str)
for i in range(factor_count):
    for j in range(factor_count):
        if IRM[i, j] == 0 and FRM_backup[i, j] == 1:
            FRM_formatted[i, j] = "1*"

# --- Save to Excel ---
def format_ivnn(x):
    T = f"({x['T'][0]:.2f}, {x['T'][1]:.2f})"
    I = f"({x['I'][0]:.2f}, {x['I'][1]:.2f})"
    F = f"({x['F'][0]:.2f}, {x['F'][1]:.2f})"
    return f"T:{T} I:{I} F:{F}"

IVNN_str = np.vectorize(format_ivnn)(IVNN_matrix)

with pd.ExcelWriter('ERIVN_ISM_MICMAC_Results.xlsx', engine='openpyxl') as writer:
    micmac_df.drop(columns=['Color']).to_excel(writer, sheet_name='MICMAC Results', index=False)
    pd.DataFrame(Crisp_matrix).to_excel(writer, sheet_name='Crisp Decision Matrix', index=False)
    pd.DataFrame(IRM).to_excel(writer, sheet_name='Initial Reachability', index=False)
    pd.DataFrame(FRM_formatted).to_excel(writer, sheet_name='Final Reachability', index=False)
    levels_df.to_excel(writer, sheet_name='Factor Levels', index=False)
    pd.DataFrame(IVNN_str).to_excel(writer, sheet_name='IVNN Matrix', index=False)

# --- Plot Factor Levels (GÜNCELLENMİŞ VE DÜZGÜN OKLU HALİ) ---
unique_levels = sorted(levels_df['Level'].unique())
num_levels = len(unique_levels)
fig, ax = plt.subplots(figsize=(8, num_levels * 2))
level_colors = plt.get_cmap('tab20')

for idx, level in enumerate(unique_levels):
    y_bottom = idx
    ax.add_patch(patches.Rectangle((0, y_bottom), 8, 1, color=level_colors(idx % 20), alpha=0.3))
    ax.text(8.5, y_bottom + 0.5, f'Level {level}', fontsize=14, verticalalignment='center')

factor_positions = {}
factor_levels_dict = {}

for level in unique_levels:
    level_factors = levels_df[levels_df['Level'] == level]['Factor'].tolist()
    num_factors = len(level_factors)
    start_x = 4 - (num_factors - 1) / 2
    for i, factor in enumerate(level_factors):
        x_pos = start_x + i
        y_pos = level - 0.5
        size = 0.5
        ax.add_patch(patches.Rectangle((x_pos - size / 2, y_pos - size / 2), size, size, edgecolor='black', facecolor='white'))
        ax.text(x_pos, y_pos, f'{factor}', ha='center', va='center', fontsize=10)
        factor_positions[factor] = (x_pos, y_pos)
        factor_levels_dict[factor] = level

ok_sayisi = 0
for level in unique_levels:
    level_factors = sorted(levels_df[levels_df['Level'] == level]['Factor'].tolist())
    for idx in range(len(level_factors) - 1):
        factor_i = level_factors[idx]
        factor_j = level_factors[idx + 1]
        start_pos = factor_positions[factor_i]
        end_pos = factor_positions[factor_j]
        if FRM_backup[factor_i - 1, factor_j - 1] == 1:
            ax.annotate("", xy=(end_pos[0] - 0.3, end_pos[1]), xytext=(start_pos[0] + 0.3, start_pos[1]),
                        arrowprops=dict(arrowstyle="->", color='blue', lw=1))
            ok_sayisi += 1
        if FRM_backup[factor_j - 1, factor_i - 1] == 1:
            ax.annotate("", xy=(start_pos[0] + 0.3, start_pos[1]), xytext=(end_pos[0] - 0.3, end_pos[1]),
                        arrowprops=dict(arrowstyle="->", color='red', lw=1))
            ok_sayisi += 1

for i in range(factor_count):
    for j in range(factor_count):
        if FRM_backup[i, j] == 1 and i != j:
            factor_i = i + 1
            factor_j = j + 1
            level_i = factor_levels_dict[factor_i]
            level_j = factor_levels_dict[factor_j]
            if abs(level_j - level_i) == 1:
                start_pos = factor_positions[factor_i]
                end_pos = factor_positions[factor_j]
                ax.annotate("", xy=(end_pos[0], end_pos[1] + 0.3), xytext=(start_pos[0], start_pos[1] - 0.3),
                            arrowprops=dict(arrowstyle="->", color='black', lw=1))
                ok_sayisi += 1

print(f"\nToplam çizilen ok sayısı: {ok_sayisi}")
ax.set_xlim(0, 9)
ax.set_ylim(0, num_levels + 1)
ax.set_aspect('equal')
ax.axis('off')
plt.title('Factor Levels', fontsize=16)
plt.savefig('Factor_Levels.pdf', format='pdf', bbox_inches='tight')
plt.show()


print("\nAll ERIVN-ISM + MICMAC results saved to 'ERIVN_ISM_MICMAC_Results.xlsx'.")
