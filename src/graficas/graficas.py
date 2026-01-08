import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/gaueko1/users/mmartin/tfm-quantization-llm/footprint_metricas/inference_emissions.csv')

# General configuration
plt.style.use('default')
fig_size = (12, 6)

# Helper function to create sorted bar chart
def create_sorted_bar_chart(df, x_col, y_col, title, ylabel, filename):
    # Sort data from highest to lowest
    sorted_df = df.sort_values(y_col, ascending=False)
    
    plt.figure(figsize=fig_size)
    bars = plt.bar(range(len(sorted_df)), sorted_df[y_col])
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add model labels rotated
    plt.xticks(range(len(sorted_df)), sorted_df[x_col], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar, value in zip(bars, sorted_df[y_col]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(sorted_df[y_col]), 
                f'{value:.0f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# FIGURE 1: Time per query (ms) - HIGHEST TO LOWEST
create_sorted_bar_chart(df, 'model', 'latency_ms', 'Time per Query (ms)', 'Latency (ms)', 'figure1_latency.png')

# FIGURE 2: Emissions per query (μg CO₂) - HIGHEST TO LOWEST
create_sorted_bar_chart(df, 'model', 'emissions_per_query_ug', 'Emissions per Query (μg CO₂)', 'Emissions (μg CO₂)', 'figure2_emissions.png')

# FIGURE 3: Throughput (tokens/second) - HIGHEST TO LOWEST
create_sorted_bar_chart(df, 'model', 'tokens_per_sec', 'Throughput (tokens/s)', 'Throughput (tokens/s)', 'figure3_throughput.png')

# FIGURE 4: Efficiency (tok/kWh) - HIGHEST TO LOWEST
create_sorted_bar_chart(df, 'model', 'tok_per_kwh', 'Energy Efficiency (tok/kWh)', 'Energy Efficiency (tok/kWh)', 'figure4_efficiency.png')

# FIGURE 5: VRAM peak (GB) - HIGHEST TO LOWEST
create_sorted_bar_chart(df, 'model', 'vram_peak_gb', 'Peak VRAM Usage (GB)', 'VRAM Peak (GB)', 'figure5_vram.png')

print("All figures generated successfully - sorted from HIGHEST to LOWEST")
