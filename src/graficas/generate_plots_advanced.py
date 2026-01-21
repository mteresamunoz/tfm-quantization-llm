
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi

# Define paths
base_dir = r"c:\Users\UJA\Desktop\HiTZ GreenAI TFM\Metrics llms"
files = {
    'Latxa 3.1 8b': os.path.join(base_dir, "metrics latxa 3.1 8b - Hoja 1.csv"),
    'Qwen 3 8b': os.path.join(base_dir, "metrics qwen 3 8b - Hoja 1.csv"),
    'Gemma 2 9b': os.path.join(base_dir, "metrics gemma 2 9b - Hoja 1.csv")
}

output_dir = os.path.join(base_dir, "advanced_tradeoff_plots")
os.makedirs(output_dir, exist_ok=True)

# Define column mappings
mappings = {
    'Latxa 3.1 8b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB) (Cargar modelo con la misma config de bnb, si no no funciona)': 'GPU inference (GB)',
        'Disk size (GB)': 'Disk size (GB)',
        'VRAM (durante ft) gb': 'GPU training (GB)',
        'Carbon emitted (kg CO2 eq.)': 'Carbon emitted (kg CO2 eq.)',
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa (min)': 'Training duration (min)'
    },
    'Qwen 3 8b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB)': 'GPU inference (GB)',
        'Tamaño modelo en disco(du -sh)': 'Disk size (GB)',
        'VRAM (durante ft)': 'GPU training (GB)',
        'Emisiones de CO₂e (g/query)': 'Carbon emitted (kg CO2 eq.)', 
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa(min)': 'Training duration (min)'
    },
    'Gemma 2 9b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB)': 'GPU inference (GB)',
        'Tamaño modelo en disco(du -sh)': 'Disk size (GB)',
        'VRAM (durante ft)': 'GPU training (GB)',
        'Emisiones de CO₂e (g/query)': 'Carbon emitted (kg CO2 eq.)',
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa': 'Training duration (min)'
    }
}

standard_columns = [
    'Method', 'Model', 
    'Mean accuracy', 'GPU inference (GB)', 'Disk size (GB)', 
    'GPU training (GB)', 'Carbon emitted (kg CO2 eq.)', 
    'Throughput (Token/s)', 'Efficiency (tok/kWh)', 'Training duration (min)'
]

dfs = []
for model_name, filepath in files.items():
    if not os.path.exists(filepath): continue
    df = pd.read_csv(filepath)
    df = df.rename(columns=mappings[model_name])
    df['Model'] = model_name
    cols_to_keep = [c for c in standard_columns if c in df.columns]
    df = df[cols_to_keep]
    for col in df.columns:
        if col not in ['Method', 'Model']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df['Method'] = final_df['Method'].str.replace('int4', 'fp4', regex=False).str.strip()
# EXCLUDE LoRA as requested for advanced plots
final_df = final_df[final_df['Method'] != 'LoRA']

# Setup style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['figure.dpi'] = 300
model_palette = {'Latxa 3.1 8b': '#1f77b4', 'Qwen 3 8b': '#d62728', 'Gemma 2 9b': '#2ca02c'}

# --- 1. TRADE-OFF SCATTER PLOTS ---
# A. Accuracy vs GPU Inference
# B. Accuracy vs Carbon
# C. Throughput vs Training Duration

tradeoffs = [
    ('GPU inference (GB)', 'Mean accuracy', 'Memory Footprint vs Accuracy'),
    ('Carbon emitted (kg CO2 eq.)', 'Mean accuracy', 'Carbon Footprint vs Accuracy'),
    ('Training duration (min)', 'Throughput (Token/s)', 'Training Cost vs Inference Speed')
]

for x_col, y_col, title in tradeoffs:
    plt.figure(figsize=(10, 6))
    
    # Filter data: X and Y must be valid
    plot_data = final_df.dropna(subset=[x_col, y_col])
    plot_data = plot_data[(plot_data[x_col] > 0) & (plot_data[y_col] > 0)]

    sns.scatterplot(
        data=plot_data, 
        x=x_col, 
        y=y_col, 
        hue='Model', 
        style='Method', 
        palette=model_palette,
        s=150,
        alpha=0.9,
        edgecolor='black'
    )
    
    plt.title(f'Trade-off: {title}', fontsize=14, fontweight='bold', pad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'tradeoff_{x_col.split()[0]}_vs_{y_col.split()[0]}.png'))
    plt.close()


# --- 2. RADAR CHARTS (Normalized "Better is Bigger") ---
# Metrics: Accuracy (High), Throughput (High), Efficiency (High), GPU (Low->High), Carbon (Low->High)
radar_metrics = {
    'Mean accuracy': 'max',          # Normalize to Max
    'Throughput (Token/s)': 'max',   # Normalize to Max
    'GPU inference (GB)': 'min',     # Invert: 1 - (val/max) or similar. Let's do (MAX - val) / (MAX - MIN) -> 1 is min(best), 0 is max(worst)
    'Carbon emitted (kg CO2 eq.)': 'min'
}

# Add inverted columns for radar
radar_df = final_df.copy()

processed_metrics = []
labels = []

# Global normalization ranges
for metric, direction in radar_metrics.items():
    if metric not in radar_df.columns: continue
    
    clean_series = radar_df[metric].dropna()
    if clean_series.empty: continue
    
    min_v = clean_series.min()
    max_v = clean_series.max()
    
    new_col = f'{metric}_norm'
    processed_metrics.append(new_col)
    
    if direction == 'max':
        # Simple 0-1 scaling: (val - min) / (max - min)
        # Or just val/max to keep origin at 0 absolute? 
        # Usually radar charts are better with min-max scaling to fill the shape.
        radar_df[new_col] = (radar_df[metric] - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        labels.append(metric)
    else:
        # Inverse: (max - val) / (max - min). So min value becomes 1, max becomes 0.
        radar_df[new_col] = (max_v - radar_df[metric]) / (max_v - min_v) if max_v > min_v else 0.5
        # User requested to remove (Inv)
        labels.append(metric)

# Function to draw radar
def create_radar(data, title, filename):
    categories = labels
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7) # Hide internal labels for clean look
    plt.ylim(0, 1.05)
    
    # Plot each method line
    # Depending on 'data', it might be rows (methods)
    linestyles = ['-', '--', '-.', ':']
    colors = plt.cm.tab10.colors
    
    for i, (idx, row) in enumerate(data.iterrows()):
        values = row[processed_metrics].values.flatten().tolist()
        values += values[:1]
        
        # Helper to avoid NaN in plotting
        values = [v if not pd.isna(v) else 0 for v in values]
        
        ax.plot(angles, values, linewidth=2, linestyle=linestyles[i % len(linestyles)], label=row['Method'])
        ax.fill(angles, values, alpha=0.05)

    plt.title(title, size=14, color='black', y=1.1, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Create one Radar Chart per Model (Comparing Methods)
for model in final_df['Model'].unique():
    model_data = radar_df[radar_df['Model'] == model].copy()
    
    # Filter for interesting methods to avoid clutter (e.g., Base, QLoRA NF4, PostQuant INT8)
    # Or just top 5? Or all? 
    # Let's try to plot all 7-8 methods, might be busy but comprehensive.
    # Selecting specific diverse set for clarity as requested "Compara... e.g., QLoRA INT4 destaca"
    # Let's stick to key ones + base
    # Updated method names based on CSV: 'Base model (BF16)', 'QLoRA_nf4DQ', 'LoRA', 'QLoRA_fp4', 'LoRA-Q_...'
    # Removed LoRA as requested
    key_methods = ['Base model (BF16)', 'QLoRA_nf4DQ', 'QLoRA_fp4']
    subset = model_data[model_data['Method'].isin(key_methods)]
    
    if not subset.empty:
        create_radar(subset, model, f'radar_{model}.png')


# --- 3. PARETO FRONTS (Skyline) ---
# Scatter with a line connecting optimal points (Max Accuracy, Min Resource)
# We plot standard scatter then overlay the front.

def plot_pareto(df, x_col, y_col, x_dir='min', y_dir='max', filename='pareto.png'):
    # x_dir: 'min' (lower is better) or 'max'
    # y_dir: 'max' (higher is better) usually Accuracy
    
    plt.figure(figsize=(10, 6))
    
    # Plot all points
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue='Model', style='Method', 
        palette=model_palette, s=150, alpha=0.9
    )
    
    # Just a simple scatter, no pareto overlay as requested

    plt.title(f'Trade-off: {y_col} vs {x_col}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# Plot Pareto for:
# 1. Acc vs GPU (Min GPU, Max Acc)
plot_pareto(final_df, 'GPU inference (GB)', 'Mean accuracy', x_dir='min', y_dir='max', filename='pareto_acc_gpu.png')

# 2. Acc vs Carbon (Min Carbon, Max Acc)
plot_pareto(final_df, 'Carbon emitted (kg CO2 eq.)', 'Mean accuracy', x_dir='min', y_dir='max', filename='pareto_acc_carbon.png')

print(f"Advanced plots generated in {output_dir}")
