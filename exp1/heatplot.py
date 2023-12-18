import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def select_rows_from_csv(file_name):
    df = pd.read_csv(file_name)
    selected = df.sample(6)
    return selected.values

def row_to_matrix(row):
    return row.reshape(8, 8)

rows1 = select_rows_from_csv('data/uncondition/train_data.csv')
rows2 = select_rows_from_csv('data/uncondition/generated_samples_spd_ddpm.csv')
rows3 = select_rows_from_csv('data/uncondition/generated_samples_ddpm.csv')
labels = ['Test samples', 'SPD-DDPM', 'DDPM']

fig, axes = plt.subplots(3, 6, figsize=(15, 6))
for row_idx, rows in enumerate([rows1, rows2, rows3]):
    for col_idx in range(6):
        sns.heatmap(row_to_matrix(rows[col_idx]), ax=axes[row_idx, col_idx], cmap='coolwarm', cbar=False, annot=False)

        axes[row_idx, col_idx].set_xticks([])
        axes[row_idx, col_idx].set_yticks([])
        if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(labels[row_idx], rotation=0, ha='right', va='center', fontsize=20)
                axes[row_idx, col_idx].yaxis.labelpad = 15  

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sns.heatmap(np.random.rand(8,8), ax=axes[2, 5], cmap='coolwarm', cbar=True, annot=False, cbar_ax=cbar_ax)
axes[2, 5].set_xticks([])
axes[2, 5].set_yticks([])
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.savefig(f'heatplot.png')












