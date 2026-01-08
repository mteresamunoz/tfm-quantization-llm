import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Read the ACTUAL CSV file
df = pd.read_csv('/gaueko1/users/mmartin/tfm-quantization-llm/results_csv/lm-harnessBasqueacc - latxa8b organizada.csv')

print("CSV loaded successfully. Columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Clean and parse data (adjust column names based on your actual CSV)
# Assuming columns: 'model', 'media_acc', 'disk_size' or similar
# Update these based on your actual column names from the print above

# Parse accuracy (replace ',' with '.' and convert to float)
df['media_acc'] = df['media acc (lm harness)'].str.replace(',', '.').astype(float)  # Adjust column name

# Parse disk size (extract numbers, convert mb to gb)
df['disk_size_gb'] = df['Tamaño modelo en disco(du -sh)'].str.extract('(\d+(?:,\d+)?)(gb|mb)').apply(
    lambda x: float(x[0].replace(',', '.')) / 1000 if x[1] == 'mb' else float(x[0].replace(',', '.')), axis=1
)  # Adjust column name

# Clean model names
df['model'] = df['Modelo'].str.strip()  # Adjust column name

print("\nProcessed data:")
print(df[['model', 'media_acc', 'disk_size_gb']].round(4))

# QLoRA + PostQuant only
qlora_postquant = ['qlora8', 'qlora4', 'qlora_nf4', 'post_q8', 'post_q4', 'post_nf4', 'post_fp8']
df_qlora_post = df[df['model'].isin(qlora_postquant)]

# Configuration
plt.style.use('default')
fig_size = (12, 6)

def create_sorted_bar_chart(df, x_col, y_col, title, ylabel, filename):
    sorted_df = df.sort_values(y_col, ascending=False).dropna(subset=[y_col])
    
    plt.figure(figsize=fig_size)
    bars = plt.bar(range(len(sorted_df)), sorted_df[y_col], color='skyblue', edgecolor='navy')
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.xticks(range(len(sorted_df)), sorted_df[x_col], rotation=45, ha='right')
    
    max_y = max(sorted_df[y_col])
    for bar, value in zip(bars, sorted_df[y_col]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max_y, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# === ALL MODELS ===
create_sorted_bar_chart(df, 'model', 'media_acc', 'Average Accuracy - ALL MODELS (Highest to Lowest)', 'Accuracy (%)', 'all_models_accuracy.png')
create_sorted_bar_chart(df, 'model', 'disk_size_gb', 'Disk Size - ALL MODELS (Highest to Lowest)', 'Disk Size (GB)', 'all_models_disk_size.png')

# === QLoRA vs PostQuant ===
create_sorted_bar_chart(df_qlora_post, 'model', 'media_acc', 'Average Accuracy - QLoRA vs PostQuant', 'Accuracy (%)', 'qlora_postquant_accuracy.png')
create_sorted_bar_chart(df_qlora_post, 'model', 'disk_size_gb', 'Disk Size - QLoRA vs PostQuant', 'Disk Size (GB)', 'qlora_postquant_disk_size.png')

print("✅ Charts generated from CSV:")
print("- all_models_accuracy.png")
print("- all_models_disk_size.png") 
print("- qlora_postquant_accuracy.png")
print("- qlora_postquant_disk_size.png")
