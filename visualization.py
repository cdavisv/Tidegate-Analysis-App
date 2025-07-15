# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os


# --- Add this helper function at the top of visualization.py ---
def save_plot(fig, filename):
    """Helper function to save a plot as both HTML and PNG."""
    # Create an 'output' directory if it doesn't exist
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths
    html_path = os.path.join(output_dir, f"{filename}.html")
    png_path = os.path.join(output_dir, f"{filename}.png")

    # Save the plot
    fig.write_html(html_path)
    print(f"   -> Saved interactive plot to: {html_path}")
    
    try:
        fig.write_image(png_path, scale=2)
        print(f"   -> Saved static image to: {png_path}")
    except Exception as e:
        print(f"   -> Could not save PNG. Is 'kaleido' installed? (pip install kaleido). Error: {e}")



def plot_environmental_factors(mtr_gate_df, hinge_gate_df, tidal_df, temp_df):
    """
    Generates and saves plots for the environmental analysis summaries.
    """
    print("\n--- Generating Environmental Factor Visualizations ---")
    
    if mtr_gate_df is not None and not mtr_gate_df.empty:
        print(" -> Plotting: Detection Rate by MTR Gate Angle")
        fig_gate_mtr = px.bar(
            mtr_gate_df, x=mtr_gate_df.index, y='Detection_Rate_Pct',
            title='Detection Rate by MTR Gate Opening Angle',
            labels={'x': 'MTR Gate State', 'y': 'Detection Rate (%)'},
            color='Detection_Rate_Pct', color_continuous_scale='Viridis'
        )
        #fig_gate_mtr.show()
        save_plot(fig_gate_mtr, "2a_mtr_gate_detection_rate") # <-- ADDED

    if hinge_gate_df is not None and not hinge_gate_df.empty:
        print(" -> Plotting: Detection Rate by Top Hinge Gate Angle")
        fig_gate_hinge = px.bar(
            hinge_gate_df, x=hinge_gate_df.index, y='Detection_Rate_Pct',
            title='Detection Rate by Top Hinge Gate Opening Angle',
            labels={'x': 'Top Hinge Gate State', 'y': 'Detection Rate (%)'},
            color='Detection_Rate_Pct', color_continuous_scale='Plasma'
        )
        #fig_gate_hinge.show()
        save_plot(fig_gate_hinge, "2b_hinge_gate_detection_rate") # <-- ADDED
    
    if tidal_df is not None and not tidal_df.empty:
        print(" -> Plotting: Detection Rate by Tidal Level")
        fig_tidal = px.bar(
            tidal_df, x=tidal_df.index, y='Detection_Rate_Pct',
            title='Detection Rate by Tidal Level',
            labels={'x': 'Tidal Level', 'y': 'Detection Rate (%)'}
        )
        #fig_tidal.show()
        save_plot(fig_tidal, "2c_tidal_level_detection_rate") # <-- ADDED

        
def create_safe_water_visualizations(df, title_suffix=""):
    """
    Creates time series plots for available water quality parameters.
    This version is safer and checks for data before plotting.
    """
    print("\n--- Checking for Water Quality Time Series Plots ---")
    
    water_params_to_plot = {
        'Water_Temp_C': ('Water Temperature (°C)', 'blue'),
        'pH': ('pH', 'green'),
        'DO_mgL': ('Dissolved Oxygen (mg/L)', 'cyan'),
        'Salinity_psu': ('Salinity (psu)', 'orange'),
        'Turbidity_NTU': ('Turbidity (NTU)', 'brown'),
        'Depth': ('Water Depth (m)', 'purple'),
    }

    available_params = {k: v for k, v in water_params_to_plot.items() if k in df.columns and df[k].notna().any()}
    
    if not available_params:
        print(" -> Skipping water quality plots: No relevant data columns found.")
        return

    print(f" -> Plotting: Water Quality Parameters Over Time {title_suffix}")
    for param, (label, color) in available_params.items():
        # --- MODIFIED LINE ---
        # Added markers=True to show exactly where the data points are.
        fig = px.line(df, x='DateTime', y=param, title=label, labels={'y': label}, markers=True)
        # --- END MODIFICATION ---
        fig.update_traces(line_color=color, line_width=1.5, marker=dict(size=4))
        #fig.show()
        save_plot(fig, f"3_water_quality_{param}{title_suffix}")


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
        #fig_species.show()
        save_plot(fig_species, "4a_top_10_species_events")
    
    # Plot 2: Water Quality Correlation Matrix
    wq_corr_cols = [col for col in ['Water_Temp_C', 'pH', 'DO_mgL', 'Salinity_psu', 'Turbidity_NTU', 'Depth'] if col in df_combined.columns]
    if len(wq_corr_cols) > 1:
        print(" -> Plotting: Water Quality Correlation Matrix")
        corr_matrix = df_combined[wq_corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title='Water Quality Correlation Matrix', aspect="auto")
        #fig_corr.show()
        save_plot(fig_corr, "4b_water_quality_correlation")

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
    #fig.show()
    save_plot(fig, "2_gate_analysis_detection_rate")



def plot_bird_analysis(summary_table, combined_df):
    """
    Creates detailed visualizations for the bird behavior analysis.
    """
    print("\n--- Generating Bird Behavior Visualizations ---")

    bird_detections_df = combined_df[combined_df['is_bird_detection']].copy()
    if bird_detections_df.empty:
        print(" -> Skipping bird plots: No bird detections to visualize.")
        return

    # --- Plot 1: Heatmap of Detection Hot-Spots ---
    # This visualizes the summary table, showing the best conditions for hunting.
    if not summary_table.empty:
        print(" -> Plotting: Heatmap of Bird Detection 'Hot-Spots'")
        fig_heatmap = px.imshow(
            summary_table,
            text_auto=True,
            labels=dict(x="Tidal Flow State", y="Gate Status", color="Detection Rate (%)"),
            title="<b>Bird Detection Rate (%) by Gate Status and Tidal Flow</b>"
        )
        fig_heatmap.update_xaxes(side="top")
        #fig_heatmap.show()
        save_plot(fig_heatmap, "5a_bird_detection_heatmap")

    # --- Plot 2: Granular Scatter Plot ---
    # This plot shows every single water measurement. Points are colored if a bird was
    # detected, directly showing the relationship between gate angle and tidal change.
    if 'Gate_Opening_MTR_Deg' in combined_df.columns and 'tidal_change_m_hr' in combined_df.columns:
        print(" -> Plotting: Granular Scatter Plot of Detections")

        # Downsample for performance if the dataframe is very large
        plot_df = combined_df.sample(n=5000) if len(combined_df) > 5000 else combined_df

        # FIX: Convert the DateTime index to a column so Plotly can access it.
        plot_df = plot_df.reset_index()

        fig_scatter = px.scatter(
            plot_df,
            x="Gate_Opening_MTR_Deg",
            y="tidal_change_m_hr",
            color="is_bird_detection",
            color_discrete_map={True: "red", False: "rgba(200, 200, 200, 0.2)"},
            title="<b>Bird Detections vs. Gate Angle and Tidal Change</b>",
            labels={
                "Gate_Opening_MTR_Deg": "Gate Opening Angle (Degrees)",
                "tidal_change_m_hr": "Tidal Change (meters/hour)",
                "is_bird_detection": "Bird Detected?"
            },
            hover_data=['DateTime']
        )
        # Add lines to indicate rising/falling tide and open/closed gate
        fig_scatter.add_hline(y=0, line_dash="dash", annotation_text="Slack Tide")
        fig_scatter.add_vline(x=5, line_dash="dash", annotation_text="Gate Closed")
        #fig_scatter.show()
        save_plot(fig_scatter, "5b_bird_detection_scatter")

def plot_species_analysis(species_summary_df):
    """
    Generates and saves a bar plot for the species analysis summary.
    """
    if species_summary_df is not None and not species_summary_df.empty:
        print("\n--- Generating Species Analysis Visualization ---")
        print(" -> Plotting: Top 15 Species by Total Count")
        
        plot_df = species_summary_df.head(15)
        
        fig = px.bar(
            plot_df,
            x=plot_df.index,
            y='Total_Count',
            title='Top 15 Species by Total Individual Count',
            labels={'x': 'Species', 'y': 'Total Individual Count'},
            color='Detection_Events',
            color_continuous_scale=px.colors.sequential.Viridis,
            hover_name=plot_df.index,
            hover_data={'Detection_Events': True, 'Total_Count': True}
        )
        
        #fig.show() # <-- Still shows the interactive plot
        save_plot(fig, "1_species_summary") # <-- ADDED: Saves the plot to files