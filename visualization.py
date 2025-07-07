# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def create_safe_water_visualizations(df, title_suffix=""):
    """Creates time series plots for available water quality parameters."""
    print(f"\nCreating water visualizations for {title_suffix}...")
    water_params_plot_map = {
        'Water_Temp_C': ('Water Temperature (°C)', 'blue'),
        'pH': ('pH', 'green'),
        'DO_mgL': ('Dissolved Oxygen (mg/L)', 'cyan'),
        'Salinity_psu': ('Salinity (psu)', 'orange'),
        'Turbidity_NTU': ('Turbidity (NTU)', 'brown'),
        'Depth': ('Water Depth (m)', 'purple'),
    }

    available_params = {k: v for k, v in water_params_plot_map.items() if k in df.columns and df[k].notna().any()}
    if not available_params:
        print("No water parameters with data found to visualize.")
        return

    num_params = len(available_params)
    fig, axes = plt.subplots(num_params, 1, figsize=(15, 4 * num_params), squeeze=False)
    fig.suptitle(f'Water Quality Parameters Over Time {title_suffix}', fontsize=16)
    
    for i, (param, (label, color)) in enumerate(available_params.items()):
        ax = axes[i, 0]
        sns.lineplot(data=df, x='DateTime', y=param, ax=ax, color=color, linewidth=0.8)
        ax.set_title(label)
        ax.set_ylabel(label.split('(')[0].strip())
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def create_analysis_plots(df_combined, species_df):
    """Creates a dashboard of comprehensive analysis plots."""
    print("\nGenerating comprehensive analysis plots...")
    if df_combined.empty:
        print("Combined DataFrame is empty. Cannot generate plots.")
        return

    # Plot 1: Water Temperature Over Time with Detections
    fig = px.line(df_combined, x='DateTime', y='Water_Temp_C', title='Water Temperature & Detections')
    detections = df_combined[df_combined['has_camera_data']]
    fig.add_scatter(x=detections['DateTime'], y=detections['Water_Temp_C'], mode='markers', name='Animal Detection', marker=dict(color='red', size=5))
    fig.show()

    # Plot 2: Top Species Detection Frequency
    if not species_df.empty:
        top_species = species_df['Species'].value_counts().nlargest(10)
        fig = px.bar(top_species, x=top_species.index, y=top_species.values, title='Top 10 Species by Detection Events')
        fig.show()

    # Add more plots as desired (e.g., correlation matrix, daily detections)
    # Example: Correlation Matrix
    wq_corr_cols = [col for col in ['Water_Temp_C', 'pH', 'DO_mgL', 'Salinity_psu', 'Turbidity_NTU', 'Depth'] if col in df_combined.columns]
    if len(wq_corr_cols) > 1:
        corr_matrix = df_combined[wq_corr_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, title='Water Quality Correlation Matrix')
        fig.show()

def plot_gate_analysis(gate_analysis_df):
    """Visualizes detection rates by tide gate state."""
    fig = px.bar(
        gate_analysis_df,
        x=gate_analysis_df.index,
        y='Detection_Rate_Pct',
        title='Animal Detection Rate by Gate Opening Angle',
        labels={'x': 'Gate State', 'y': 'Detection Rate (%)'},
        color='Detection_Rate_Pct',
        color_continuous_scale='Viridis'
    )
    fig.show()