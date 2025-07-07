# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_environmental_factors(mtr_gate_df, hinge_gate_df, tidal_df, temp_df):
    """
    Generates plots for the environmental analysis summaries.
    Now creates separate plots for each tide gate.
    """
    print("\n--- Generating Environmental Factor Visualizations ---")
    
    # 1. Plot MTR Gate Analysis
    if mtr_gate_df is not None and not mtr_gate_df.empty:
        print(" -> Plotting: Detection Rate by MTR Gate Angle")
        fig_gate_mtr = px.bar(
            mtr_gate_df,
            x=mtr_gate_df.index,
            y='Detection_Rate_Pct',
            title='Detection Rate by MTR Gate Opening Angle',
            labels={'x': 'MTR Gate State', 'y': 'Detection Rate (%)'},
            color='Detection_Rate_Pct',
            color_continuous_scale='Viridis'
        )
        fig_gate_mtr.show()

    # 2. Plot Hinge Gate Analysis
    if hinge_gate_df is not None and not hinge_gate_df.empty:
        print(" -> Plotting: Detection Rate by Top Hinge Gate Angle")
        fig_gate_hinge = px.bar(
            hinge_gate_df,
            x=hinge_gate_df.index,
            y='Detection_Rate_Pct',
            title='Detection Rate by Top Hinge Gate Opening Angle',
            labels={'x': 'Top Hinge Gate State', 'y': 'Detection Rate (%)'},
            color='Detection_Rate_Pct',
            color_continuous_scale='Plasma' # Use a different color scale
        )
        fig_gate_hinge.show()
    
    # 3. Plot Tidal Analysis
    if tidal_df is not None and not tidal_df.empty:
        print(" -> Plotting: Detection Rate by Tidal Level")
        fig_tidal = px.bar(
            tidal_df,
            x=tidal_df.index,
            y='Detection_Rate_Pct',
            title='Detection Rate by Tidal Level',
            labels={'x': 'Tidal Level', 'y': 'Detection Rate (%)'}
        )
        fig_tidal.show()

def create_safe_water_visualizations(df, title_suffix=""):
    """
    Creates time series plots for available water quality parameters.
    This version is safer and checks for data before plotting.
    """
    print("\n--- Checking for Water Quality Time Series Plots ---")
    
    # Define the parameters we want to look for
    water_params_to_plot = {
        'Water_Temp_C': ('Water Temperature (°C)', 'blue'),
        'pH': ('pH', 'green'),
        'DO_mgL': ('Dissolved Oxygen (mg/L)', 'cyan'),
        'Salinity_psu': ('Salinity (psu)', 'orange'),
        'Turbidity_NTU': ('Turbidity (NTU)', 'brown'),
        'Depth': ('Water Depth (m)', 'purple'),
    }

    # Find which of these parameters are actually available in the DataFrame
    available_params = {k: v for k, v in water_params_to_plot.items() if k in df.columns and df[k].notna().any()}
    
    if not available_params:
        print(" -> Skipping water quality plots: No relevant data columns found.")
        return

    print(f" -> Plotting: Water Quality Parameters Over Time {title_suffix}")
    # Create one plot for each available parameter
    for param, (label, color) in available_params.items():
        fig = px.line(df, x='DateTime', y=param, title=label, labels={'y': label})
        fig.update_traces(line_color=color, line_width=1.5)
        fig.show()

def create_analysis_plots(df_combined, species_df):
    """
    Creates a dashboard of comprehensive analysis plots.
    This version is safer and checks for columns before plotting.
    """
    print("\n--- Checking for Comprehensive Analysis Plots ---")
    if df_combined.empty:
        print(" -> Skipping comprehensive plots: Combined DataFrame is empty.")
        return

    # Plot 1: Top Species Detection Frequency
    if not species_df.empty and 'Species' in species_df.columns:
        print(" -> Plotting: Top 10 Species by Detection Events")
        top_species = species_df['Species'].value_counts().nlargest(10)
        fig_species = px.bar(top_species, x=top_species.index, y=top_species.values, title='Top 10 Species by Detection Events', labels={'y': 'Number of Detections', 'x': 'Species'})
        fig_species.show()
    
    # Plot 2: Water Quality Correlation Matrix
    wq_corr_cols = [col for col in ['Water_Temp_C', 'pH', 'DO_mgL', 'Salinity_psu', 'Turbidity_NTU', 'Depth'] if col in df_combined.columns]
    if len(wq_corr_cols) > 1:
        print(" -> Plotting: Water Quality Correlation Matrix")
        corr_matrix = df_combined[wq_corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title='Water Quality Correlation Matrix', aspect="auto")
        fig_corr.show()

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