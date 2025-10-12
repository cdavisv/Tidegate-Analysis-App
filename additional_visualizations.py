# additional_visualizations.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

def save_plot(fig, filename):
    """Helper function to save a plot as HTML."""
    output_dir = 'output_plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    html_path = os.path.join(output_dir, f"{filename}.html")
    fig.write_html(html_path)
    print(f"   -> Saved interactive plot to: {html_path}")

def create_method_comparison_visualizations(comprehensive_results, combined_df):
    """
    Creates visualizations comparing Camera Activity Pattern vs Wildlife Detection Efficiency methods.
    """
    print("\n--- Generating Method Comparison Visualizations ---")
    
    # 1. Data Overview Dashboard
    create_data_overview_dashboard(comprehensive_results)
    
    # 2. Analysis Method Comparison Charts
    create_analysis_method_comparison(comprehensive_results, combined_df)
    
    # 3. Species Detection Patterns
    create_species_pattern_comparison(comprehensive_results)
    
    # 4. Environmental Factor Effectiveness
    create_environmental_effectiveness_charts(combined_df)
    
    # 5. Time Series Analysis
    create_temporal_analysis_charts(combined_df)
    
    # 6. Camera Performance Dashboard
    create_camera_performance_dashboard(combined_df)

def create_data_overview_dashboard(comprehensive_results):
    """
    Creates a dashboard showing the key differences between analysis methods.
    """
    print(" -> Creating: Data Overview Dashboard")
    
    comparison = comprehensive_results['comparison']
    
    # Create metrics summary
    fig = go.Figure()
    
    # Add bar chart showing data breakdown
    categories = ['Total Time Periods', 'Camera Active Periods', 'Animal Detections']
    values = [comparison['total_periods'], comparison['camera_periods'], comparison['animal_detections']]
    colors = ['lightblue', 'orange', 'lightgreen']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f"{val:,}" for val in values],
        textposition='auto',
        name='Data Overview'
    ))
    
    fig.update_layout(
        title="<b>Dataset Overview: Camera vs Sensor Data</b>",
        yaxis_title="Number of Records",
        yaxis_type="log",  # Log scale to show the large differences
        template="plotly_white",
        height=500,
        annotations=[
            dict(
                x=1, y=comparison['camera_periods'] * 2,
                text=f"Camera Activity Rate: {comparison['camera_activity_rate']:.2f}%",
                showarrow=True,
                arrowhead=2,
                bgcolor="yellow",
                bordercolor="black"
            )
        ]
    )
    
    save_plot(fig, "7a_data_overview_dashboard")

def create_analysis_method_comparison(comprehensive_results, combined_df):
    """
    Creates side-by-side comparison of Camera Activity vs Detection Efficiency methods.
    """
    print(" -> Creating: Analysis Method Comparison Charts")
    
    # Get gate analysis data for comparison
    if 'Gate_Opening_MTR_Deg' not in combined_df.columns:
        print("   -> Skipping gate comparison: MTR gate data not available")
        return
    
    # Camera Activity Pattern rates - FIXED: Handle string boolean values
    combined_df_temp = combined_df.copy()
    # Convert string booleans to actual booleans
    if combined_df_temp['has_camera_data'].dtype == 'object':
        combined_df_temp['has_camera_data'] = combined_df_temp['has_camera_data'] == 'True'
    
    mtr_bins = [-1, 5, 39, 63, 88]
    mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
    combined_df_temp['gate_category'] = pd.cut(combined_df_temp['Gate_Opening_MTR_Deg'], bins=mtr_bins, labels=mtr_labels)
    
    activity_rates = combined_df_temp.groupby('gate_category', observed=True)['has_camera_data'].mean() * 100
    
    # Wildlife Detection Efficiency rates
    camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
    camera_obs['animal_detected'] = (
        camera_obs['Species'].notna() &
        (camera_obs['Notes'] != 'No animals detected')
    ).astype(int)
    efficiency_rates = camera_obs.groupby('gate_category', observed=True)['animal_detected'].mean() * 100
    
    # Create comparison chart
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Camera Activity Pattern Analysis<br>(When cameras were active)", 
            "Wildlife Detection Efficiency Analysis<br>(When cameras detected animals)"
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Camera Activity Pattern method
    fig.add_trace(
        go.Bar(
            x=activity_rates.index,
            y=activity_rates.values,
            name="Camera Activity Rate",
            marker_color="lightcoral",
            text=[f"{val:.1f}%" for val in activity_rates.values],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Wildlife Detection Efficiency method
    fig.add_trace(
        go.Bar(
            x=efficiency_rates.index,
            y=efficiency_rates.values,
            name="Detection Success Rate",
            marker_color="lightgreen",
            text=[f"{val:.1f}%" for val in efficiency_rates.values],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="<b>Analysis Method Comparison: Camera Activity vs Detection Efficiency</b>",
        height=600,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Camera Activity Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Detection Success Rate (%)", row=1, col=2)
    fig.update_xaxes(title_text="MTR Gate Position", row=1, col=1)
    fig.update_xaxes(title_text="MTR Gate Position", row=1, col=2)
    
    save_plot(fig, "7b_analysis_method_comparison")

def create_species_pattern_comparison(comprehensive_results):
    """
    Creates visualizations comparing species patterns between methods.
    """
    print(" -> Creating: Species Pattern Comparison")
    
    activity_species = comprehensive_results['camera_activity']['species_summary'].head(10)
    efficiency_species = comprehensive_results['detection_efficiency']['species_summary'].head(10)
    
    # Create side-by-side species comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Species Counts<br>(Camera Activity Method)", "Species Detection Rates<br>(Detection Efficiency Method)"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Camera Activity method - total counts
    fig.add_trace(
        go.Bar(
            y=activity_species.index,
            x=activity_species['Total_Count'],
            orientation='h',
            name="Total Count",
            marker_color="lightcoral",
            text=activity_species['Total_Count'].astype(int),
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Detection Efficiency method - detection rates
    if 'Detection_Rate_Pct' in efficiency_species.columns:
        fig.add_trace(
            go.Bar(
                y=efficiency_species.index,
                x=efficiency_species['Detection_Rate_Pct'],
                orientation='h',
                name="Detection Rate %",
                marker_color="lightgreen",
                text=[f"{val:.1f}%" for val in efficiency_species['Detection_Rate_Pct']],
                textposition='auto'
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        title="<b>Top 10 Species: Count vs Detection Rate Analysis</b>",
        height=600,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Total Individual Count", row=1, col=1)
    fig.update_xaxes(title_text="Detection Rate (%)", row=1, col=2)
    
    save_plot(fig, "7c_species_pattern_comparison")

def create_environmental_effectiveness_charts(combined_df):
    """
    Creates charts showing environmental factor effectiveness during camera operations.
    """
    print(" -> Creating: Environmental Factor Effectiveness Charts")
    
    # Filter to camera observations only - FIXED: Handle string boolean values
    combined_df_temp = combined_df.copy()
    if combined_df_temp['has_camera_data'].dtype == 'object':
        combined_df_temp['has_camera_data'] = combined_df_temp['has_camera_data'] == 'True'
    
    camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
    
    if camera_obs.empty:
        print("   -> No camera observations for environmental analysis")
        return
    
    # Create effectiveness metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("MTR Gate Position Activity", "Tidal Level Activity", "Temperature Activity", "Monthly Activity"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Gate effectiveness - Shows camera operational patterns
    if 'Gate_Opening_MTR_Deg' in camera_obs.columns:
        mtr_bins = [-1, 5, 39, 63, 88]
        mtr_labels = ['Closed (0-5°)', 'Partially Open (5-39°)', 'Open (39-63°)', 'Wide Open (>63°)']
        camera_obs['gate_category'] = pd.cut(camera_obs['Gate_Opening_MTR_Deg'], bins=mtr_bins, labels=mtr_labels)
        
        gate_counts = camera_obs['gate_category'].value_counts()
        fig.add_trace(
            go.Bar(
                x=gate_counts.index,
                y=gate_counts.values,
                name="MTR Gate Activity",
                marker_color="orange",
                text=gate_counts.values,
                textposition='auto'
            ),
            row=1, col=1
        )
    
    # Tidal effectiveness - Shows when cameras were most active
    if 'Depth' in camera_obs.columns:
        quantiles = camera_obs['Depth'].quantile([0.25, 0.75])
        camera_obs['tide_level'] = pd.cut(
            camera_obs['Depth'],
            bins=[camera_obs['Depth'].min()-1, quantiles[0.25], quantiles[0.75], camera_obs['Depth'].max()+1],
            labels=['Low Tide', 'Mid Tide', 'High Tide']
        )
        
        tide_counts = camera_obs['tide_level'].value_counts()
        fig.add_trace(
            go.Bar(
                x=tide_counts.index,
                y=tide_counts.values,
                name="Tidal Activity",
                marker_color="blue",
                text=tide_counts.values,
                textposition='auto'
            ),
            row=1, col=2
        )
    
    # Temperature effectiveness
    if 'Air_Temp_C' in camera_obs.columns:
        camera_obs['temp_bin'] = pd.cut(camera_obs['Air_Temp_C'], bins=5)
        temp_counts = camera_obs['temp_bin'].value_counts().sort_index()
        temp_labels = [f"{interval.left:.1f}-{interval.right:.1f}°C" for interval in temp_counts.index]
        
        fig.add_trace(
            go.Bar(
                x=temp_labels,
                y=temp_counts.values,
                name="Temperature Activity",
                marker_color="red",
                text=temp_counts.values,
                textposition='auto'
            ),
            row=2, col=1
        )
    
    # Monthly activity
    camera_obs['month'] = camera_obs['DateTime'].dt.month
    monthly_counts = camera_obs['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_labels = [month_names[m-1] for m in monthly_counts.index]
    
    fig.add_trace(
        go.Bar(
            x=month_labels,
            y=monthly_counts.values,
            name="Monthly Activity",
            marker_color="green",
            text=monthly_counts.values,
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title="<b>Camera Operational Patterns During Monitoring</b><br><sub>Shows when cameras were most active, not detection success rates</sub>",
        height=800,
        template="plotly_white",
        showlegend=False
    )
    
    save_plot(fig, "7d_environmental_effectiveness")

def create_temporal_analysis_charts(combined_df):
    """
    Creates temporal analysis showing camera activity patterns over time.
    """
    print(" -> Creating: Temporal Analysis Charts")
    
    # FIXED: Handle string boolean values
    combined_df_temp = combined_df.copy()
    if combined_df_temp['has_camera_data'].dtype == 'object':
        combined_df_temp['has_camera_data'] = combined_df_temp['has_camera_data'] == 'True'
    
    camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
    
    if camera_obs.empty:
        print("   -> No camera observations for temporal analysis")
        return
    
    # Hourly and daily patterns
    camera_obs['hour'] = camera_obs['DateTime'].dt.hour
    camera_obs['day_of_week'] = camera_obs['DateTime'].dt.day_name()
    camera_obs['month'] = camera_obs['DateTime'].dt.month
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Hourly Camera Activity", "Daily Camera Activity", "Monthly Camera Activity", "Detection Timeline"),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # Hourly pattern
    hourly_counts = camera_obs['hour'].value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=hourly_counts.index,
            y=hourly_counts.values,
            name="Hourly Activity",
            marker_color="purple",
            text=hourly_counts.values,
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # Day of week pattern
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_counts = camera_obs['day_of_week'].value_counts().reindex(day_order, fill_value=0)
    
    fig.add_trace(
        go.Bar(
            x=daily_counts.index,
            y=daily_counts.values,
            name="Daily Activity",
            marker_color="orange",
            text=daily_counts.values,
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # Monthly trend
    monthly_counts = camera_obs['month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_labels = [month_names[m-1] for m in monthly_counts.index]
    
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=monthly_counts.values,
            mode='lines+markers',
            name="Monthly Trend",
            line=dict(color="green", width=3),
            marker=dict(size=8),
            text=monthly_counts.values,
            textposition='top center'
        ),
        row=2, col=1
    )
    
    # Timeline scatter - Only show actual species detections
    species_obs = camera_obs[
        (camera_obs['Species'].notna()) & 
        (camera_obs['Notes'] != 'No animals detected')
    ]
    
    if not species_obs.empty:
        species_sample = species_obs.sample(min(100, len(species_obs)))  # Sample for better visualization
        fig.add_trace(
            go.Scatter(
                x=species_sample['DateTime'],
                y=species_sample['Species'],
                mode='markers',
                name="Animal Detection Timeline",
                marker=dict(size=8, color="red", opacity=0.7),
                text=species_sample['Species'],
                hovertemplate="<b>%{text}</b><br>%{x}<extra></extra>"
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="<b>Temporal Camera Activity Patterns</b><br><sub>Shows when cameras were operational during monitoring</sub>",
        height=800,
        template="plotly_white",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
    fig.update_xaxes(title_text="Day of Week", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    
    save_plot(fig, "7e_temporal_analysis")

def create_camera_performance_dashboard(combined_df):
    """
    Creates a dashboard showing camera performance metrics.
    """
    print(" -> Creating: Camera Performance Dashboard")
    
    # FIXED: Handle string boolean values
    combined_df_temp = combined_df.copy()
    if combined_df_temp['has_camera_data'].dtype == 'object':
        combined_df_temp['has_camera_data'] = combined_df_temp['has_camera_data'] == 'True'
    
    camera_obs = combined_df_temp[combined_df_temp['has_camera_data']].copy()
    all_data = combined_df_temp.copy()
    
    # Calculate performance metrics
    total_periods = len(all_data)
    camera_periods = len(camera_obs)
    activity_rate = (camera_periods / total_periods) * 100
    
    # Calculate actual detection success rate
    if not camera_obs.empty:
        animal_detections = len(camera_obs[
            (camera_obs['Species'].notna()) & 
            (camera_obs['Notes'] != 'No animals detected')
        ])
        detection_success = (animal_detections / camera_periods) * 100
    else:
        detection_success = 0
    
    # Create performance metrics visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Camera Activity Rate", "Wildlife Detection Success Rate", "Species Diversity", "Data Quality Metrics"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Camera activity rate gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=activity_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Camera Activity Rate (%)"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "gray"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=1
    )
    
    # Detection success rate gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=detection_success,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Wildlife Detection Success Rate (%)"},
            gauge={
                'axis': {'range': [None, 20]},  # Adjusted range for realistic values
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 20], 'color': "lightgreen"}
                ]
            }
        ),
        row=1, col=2
    )
    
    # Species diversity pie chart - Only use actual species detections
    if not camera_obs.empty:
        species_obs = camera_obs[
            (camera_obs['Species'].notna()) & 
            (camera_obs['Notes'] != 'No animals detected')
        ]
        
        if not species_obs.empty:
            top_species = species_obs['Species'].value_counts().head(8)
            others_count = species_obs['Species'].value_counts().iloc[8:].sum() if len(species_obs['Species'].value_counts()) > 8 else 0
            
            if others_count > 0:
                species_data = pd.concat([top_species, pd.Series({'Others': others_count})])
            else:
                species_data = top_species
            
            fig.add_trace(
                go.Pie(
                    labels=species_data.index,
                    values=species_data.values,
                    textinfo='label+percent',
                    textposition='inside'
                ),
                row=2, col=1
            )
    
    # Data quality metrics
    if 'Depth' in all_data.columns:
        quality_metrics = {
            'Camera Data': camera_periods,
            'Sensor Data': total_periods - camera_periods,
            'Missing Depth': all_data['Depth'].isna().sum(),
            'Valid Depth': all_data['Depth'].notna().sum()
        }
        
        fig.add_trace(
            go.Bar(
                x=list(quality_metrics.keys()),
                y=list(quality_metrics.values()),
                marker_color=['green', 'blue', 'red', 'lightblue'],
                text=[f"{val:,}" for val in quality_metrics.values()],
                textposition='auto'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="<b>Camera System Performance Dashboard</b>",
        height=800,
        template="plotly_white",
        showlegend=False
    )
    
    save_plot(fig, "7f_camera_performance_dashboard")

def create_all_additional_visualizations(comprehensive_results, combined_df):
    """
    Main function to create all additional visualizations.
    """
    print("\n" + "="*60)
    print("GENERATING ADDITIONAL COMPREHENSIVE VISUALIZATIONS")
    print("="*60)
    
    create_method_comparison_visualizations(comprehensive_results, combined_df)
    
    print("\n✅ All additional visualizations completed!")
    print("📁 Check the 'output_plots' folder for all generated visualizations:")
    print("   • 7a_data_overview_dashboard.html")
    print("   • 7b_analysis_method_comparison.html") 
    print("   • 7c_species_pattern_comparison.html")
    print("   • 7d_environmental_effectiveness.html")
    print("   • 7e_temporal_analysis.html")
    print("   • 7f_camera_performance_dashboard.html")