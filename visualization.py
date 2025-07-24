# visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
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
        save_plot(fig_gate_mtr, "2a_mtr_gate_detection_rate")

    if hinge_gate_df is not None and not hinge_gate_df.empty:
        print(" -> Plotting: Detection Rate by Top Hinge Gate Angle")
        fig_gate_hinge = px.bar(
            hinge_gate_df, x=hinge_gate_df.index, y='Detection_Rate_Pct',
            title='Detection Rate by Top Hinge Gate Opening Angle',
            labels={'x': 'Top Hinge Gate State', 'y': 'Detection Rate (%)'},
            color='Detection_Rate_Pct', color_continuous_scale='Plasma'
        )
        save_plot(fig_gate_hinge, "2b_hinge_gate_detection_rate")
    
    if tidal_df is not None and not tidal_df.empty:
        print(" -> Plotting: Detection Rate by Tidal Level")
        fig_tidal = px.bar(
            tidal_df, x=tidal_df.index, y='Detection_Rate_Pct',
            title='Detection Rate by Tidal Level',
            labels={'x': 'Tidal Level', 'y': 'Detection Rate (%)'}
        )
        save_plot(fig_tidal, "2c_tidal_level_detection_rate")

        
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
        fig = px.line(df, x='DateTime', y=param, title=label, labels={'y': label}, markers=True)
        fig.update_traces(line_color=color, line_width=1.5, marker=dict(size=4))
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

    if not species_df.empty and 'Species' in species_df.columns:
        print(" -> Plotting: Top 10 Species by Detection Events")
        top_species = species_df['Species'].value_counts().nlargest(10)
        fig_species = px.bar(top_species, x=top_species.index, y=top_species.values, title='Top 10 Species by Detection Events', labels={'y': 'Number of Detections', 'x': 'Species'})
        save_plot(fig_species, "4a_top_10_species_events")
    
    wq_corr_cols = [col for col in ['Water_Temp_C', 'pH', 'DO_mgL', 'Salinity_psu', 'Turbidity_NTU', 'Depth'] if col in df_combined.columns]
    if len(wq_corr_cols) > 1:
        print(" -> Plotting: Water Quality Correlation Matrix")
        corr_matrix = df_combined[wq_corr_cols].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, title='Water Quality Correlation Matrix', aspect="auto")
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
    save_plot(fig, "2_gate_analysis_detection_rate")



def plot_bird_analysis(summary_table, combined_df):
    """
    Creates detailed visualizations for the bird behavior analysis.
    """
    print("\n--- Generating Bird Behavior Visualizations ---")

    bird_detections_df = combined_df[combined_df['is_animal_detection']].copy()
    if bird_detections_df.empty:
        print(" -> Skipping bird plots: No bird detections to visualize.")
        return

    if not summary_table.empty:
        print(" -> Plotting: Heatmap of Bird Detection 'Hot-Spots'")
        fig_heatmap = px.imshow(
            summary_table,
            text_auto=True,
            labels=dict(x="Tidal Flow State", y="Gate Status", color="Detection Rate (%)"),
            title="<b>Bird Detection Rate (%) by Gate Status and Tidal Flow</b>"
        )
        fig_heatmap.update_xaxes(side="top")
        save_plot(fig_heatmap, "5a_bird_detection_heatmap")

    if 'Gate_Opening_MTR_Deg' in combined_df.columns and 'tidal_change_m_hr' in combined_df.columns:
        print(" -> Plotting: Granular Scatter Plot of Detections")

        plot_df = combined_df.sample(n=5000) if len(combined_df) > 5000 else combined_df
        plot_df = plot_df.reset_index()

        fig_scatter = px.scatter(
            plot_df,
            x="Gate_Opening_MTR_Deg",
            y="tidal_change_m_hr",
            color="is_animal_detection",
            color_discrete_map={True: "red", False: "rgba(200, 200, 200, 0.2)"},
            title="<b>Bird Detections vs. Gate Angle and Tidal Change</b>",
            labels={
                "Gate_Opening_MTR_Deg": "Gate Opening Angle (Degrees)",
                "tidal_change_m_hr": "Tidal Change (meters/hour)",
                "is_bird_detection": "Bird Detected?"
            },
            hover_data=['DateTime']
        )
        fig_scatter.add_hline(y=0, line_dash="dash", annotation_text="Slack Tide")
        fig_scatter.add_vline(x=5, line_dash="dash", annotation_text="Gate Closed")
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
        save_plot(fig, "1_species_summary")

def _plot_tide_cycle_visualization(title, peak_gate_state, peak_tidal_state):
    """
    Helper function to generate and save a stylized plot of the tide cycle
    highlighting the point of peak bird activity.
    """
    time = np.linspace(0, 12, 300)
    depth = np.sin(time * np.pi / 6 - np.pi / 2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, depth, color='skyblue', linewidth=2)
    ax.fill_between(time, depth, -1, color='skyblue', alpha=0.3)

    positions = {
        'Low Slack': (0, -1),
        'Rising': (3, 0),
        'High Slack': (6, 1),
        'Falling': (9, 0)
    }

    for state, (t, d) in positions.items():
        ax.text(t, d + 0.15 if d >= 0 else d - 0.2, state, ha='center', fontsize=12, weight='bold')

    if peak_tidal_state in positions:
        peak_time, peak_depth = positions[peak_tidal_state]

        ax.plot(peak_time, peak_depth, '*', markersize=20, color='gold', markeredgecolor='black')
        ax.annotate(
            f"Peak Activity:\n{peak_gate_state}",
            xy=(peak_time, peak_depth),
            xytext=(peak_time, peak_depth + 0.5),
            ha='center',
            fontsize=11,
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            bbox=dict(boxstyle="round,pad=0.5", fc="gold", ec="black", lw=1, alpha=0.8)
        )

    ax.set_title(f"Peak Bird Activity Condition: {title}", fontsize=16, weight='bold')
    ax.set_xlabel("Tidal Cycle (~12 hours)")
    ax.set_ylabel("Relative Water Level")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(-1.5, 2.0)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()

    filename = f"hypothesis_visualization_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f" -> Saved hypothesis visualization to '{filename}'")
    plt.close()


def create_hypothesis_visualizations(df):
    """
    Generates and saves visualizations for each of the HYPOTHESIS TEST outputs,
    showing where peak bird activity occurs in the tidal cycle.
    """
    print("\n\n--- Generating Hypothesis Visualizations ---")
    if 'detailed_tidal_flow' not in df.columns or 'is_bird_detection' not in df.columns:
        print(" -> Skipping hypothesis visualizations: Required columns not found.")
        return

    analyses = {
        "MTR Gate": "Gate_Opening_MTR_Deg_category",
        "Top Hinge Gate": "Gate_Opening_Top_Hinge_Deg_category",
        "Combined Gate (Simple)": "simple_gate_category",
        "Combined Gate (Specific Combos)": "specific_gate_combo"
    }

    for title, gate_col in analyses.items():
        if gate_col not in df.columns:
            print(f" -> Skipping '{title}' visualization: Column '{gate_col}' not found.")
            continue

        summary_table = (
            df.groupby([gate_col, 'detailed_tidal_flow'])['is_bird_detection']
            .mean()
            .unstack()
            .fillna(0)
        )

        if summary_table.empty:
            continue

        peak_gate_state = summary_table.max(axis=1).idxmax()
        peak_tidal_state = summary_table.max(axis=0).idxmax()

        if title == "Combined Gate (Specific Combos)" and peak_gate_state == 'Other':
            other_df = df[
                (df[gate_col] == 'Other') & (df['detailed_tidal_flow'] == peak_tidal_state)
            ]
            if not other_df.empty:
                top_combo = other_df.groupby(['MTR_category', 'Hinge_category'])['is_bird_detection'].mean().idxmax()
                peak_gate_state = f"MTR: {top_combo[0]}\n& Hinge: {top_combo[1]}"


        _plot_tide_cycle_visualization(title, peak_gate_state, peak_tidal_state)

def create_tide_cycle_visualizations(df, tide_analysis_results):
    """
    Creates comprehensive visualizations of animal detections across tidal cycles.
    This version adds a fail-safe filter to exclude 'Unknown' from plots.
    """
    print("\n--- Generating Tide Cycle Visualizations ---")

    detection_by_tide, phase_detection, species_tide_table = tide_analysis_results

    # Plot 1: Bar chart of detection rate by general tidal state
    if detection_by_tide is not None and not detection_by_tide.empty:
        # --- FAIL-SAFE FIX: Explicitly filter the data for plotting ---
        plot_data = detection_by_tide[detection_by_tide.index != 'Unknown']
        print(" -> Plotting: Detection Rate by Tidal State (Rising, Falling, etc.)")
        
        if plot_data.empty:
            print("   -> Skipping plot: No data left after removing 'Unknown' state.")
        else:
            fig_state = px.bar(
                plot_data, # Use the filtered data
                x=plot_data.index,
                y='detection_rate',
                title='Detection Rate vs. Tidal State',
                labels={'x': 'Tidal State', 'detection_rate': 'Detection Rate'},
                color='detection_rate',
                color_continuous_scale='Blues'
            )
            save_plot(fig_state, "6a_detection_by_tidal_state")
        # --- END FIX ---

    # Plot 2: Polar chart showing detection rate by the phase of the tide
    if phase_detection is not None and not phase_detection.empty:
        print(" -> Plotting: Detection Rate by Tidal Phase (Polar Chart)")
        
        plot_data = phase_detection.copy()
        plot_data['phase_midpoint'] = plot_data.index.str.split('-').str[0].astype(float)
        plot_data['phase_degrees'] = plot_data['phase_midpoint'] * 360

        if not plot_data.empty:
            plot_data_closed = pd.concat([plot_data, plot_data.iloc[[0]]], ignore_index=True)
        else:
            plot_data_closed = plot_data

        fig_polar = px.line_polar(
            plot_data_closed,
            r='detection_rate',
            theta='phase_degrees',
            title='Detection Rate Across the Tidal Cycle',
            labels={'detection_rate': 'Detection Rate'},
            template='plotly_dark'
        )
        fig_polar.update_layout(
            polar=dict(
                angularaxis=dict(
                    tickvals=[0, 90, 180, 270],
                    ticktext=['Low Tide', 'Rising Tide', 'High Tide', 'Falling Tide'],
                    direction="clockwise"
                )
            )
        )
        save_plot(fig_polar, "6b_detection_by_tidal_phase")

    # Plot 3: Heatmap of tidal preferences for top species
    if species_tide_table is not None and not species_tide_table.empty:
        # --- FAIL-SAFE FIX: Remove 'Unknown' column if it exists ---
        plot_data_heatmap = species_tide_table.copy()
        if 'Unknown' in plot_data_heatmap.columns:
            plot_data_heatmap = plot_data_heatmap.drop(columns=['Unknown'])
        
        print(" -> Plotting: Heatmap of Species Tide Preferences")
        if plot_data_heatmap.empty or plot_data_heatmap.shape[1] == 0:
             print("   -> Skipping plot: No data left after removing 'Unknown' state.")
        else:
            fig_heatmap = px.imshow(
                plot_data_heatmap,
                text_auto=".1f",
                aspect="auto",
                title='Tidal State Preference by Species (% of Detections)',
                labels=dict(x="Tidal State", y="Species", color="Detection %"),
                color_continuous_scale='viridis'
            )
            save_plot(fig_heatmap, "6c_species_tide_preference_heatmap")
        # --- END FIX ---