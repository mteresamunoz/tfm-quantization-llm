import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your local CSV file
csv_path = '/gaueko1/users/mmartin/tfm-quantization-llm/results_csv/lm-harnessBasqueacc - latxa8b organizada.csv'

# Read CSV
df = pd.read_csv(csv_path, sep=',')

# Process key columns (adjust names if needed)
df['media_acc'] = df['media acc (lm harness)'].astype(str).str.replace(',', '.').astype(float)
df['disk_size_gb'] = pd.to_numeric(df['Tamaño modelo en disco(du -sh)'].astype(str).str.replace(',', '.'), errors='coerce')

# All models
all_models = df[['Modelo', 'media_acc', 'disk_size_gb']].dropna().copy()
all_models_acc_sorted = all_models.sort_values('media_acc', ascending=False)
all_models_disk_sorted = all_models.sort_values('disk_size_gb', ascending=False)

# QLORA and POST only
qlora_post = df[df['Modelo'].str.contains('qlora|post', case=False, na=False)][['Modelo', 'media_acc', 'disk_size_gb']].dropna().copy()
qlora_post_acc_sorted = qlora_post.sort_values('media_acc', ascending=False)
qlora_post_disk_sorted = qlora_post.sort_values('disk_size_gb', ascending=False)

# Function to create single chart with values on bars
def create_single_chart(df_sorted, metric_name, ylabel, color, filename):
    plt.figure(figsize=(12, 7))
    x = np.arange(len(df_sorted))
    bars = plt.bar(x, df_sorted[metric_name], color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    plt.title(f'{metric_name.replace("_", " ").title()} - {"All Models" if "all" in filename else "QLORA & POST"}\n(Sorted descending)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(x, df_sorted['Modelo'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # ADD VALUES ON TOP OF EACH BAR
    for bar, value in zip(bars, df_sorted[metric_name]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.005 if 'acc' in metric_name else 0.1),
                f'{value:.3f}' if 'acc' in metric_name else f'{value:.2f}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=0)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# Create 4 separate charts
print("ALL MODELS - ACC sorted (descending):")
print(all_models_acc_sorted[['Modelo', 'media_acc']])
create_single_chart(all_models_acc_sorted, 'media_acc', 'Mean ACC', 'skyblue', 'all_models_acc_sorted.png')

print("\\nALL MODELS - DISK SIZE sorted (descending):")
print(all_models_disk_sorted[['Modelo', 'disk_size_gb']])
create_single_chart(all_models_disk_sorted, 'disk_size_gb', 'Disk Size (GB)', 'lightcoral', 'all_models_disk_sorted.png')

print("\\nQLORA/POST - ACC sorted (descending):")
print(qlora_post_acc_sorted[['Modelo', 'media_acc']])
create_single_chart(qlora_post_acc_sorted, 'media_acc', 'Mean ACC', 'skyblue', 'qlora_post_acc_sorted.png')

print("\\nQLORA/POST - DISK SIZE sorted (descending):")
print(qlora_post_disk_sorted[['Modelo', 'disk_size_gb']])
create_single_chart(qlora_post_disk_sorted, 'disk_size_gb', 'Disk Size (GB)', 'lightcoral', 'qlora_post_disk_sorted.png')

print("✅ 4 charts generated with values on bars!")
