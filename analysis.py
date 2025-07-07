# analysis.py

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

def analyze_gate_impact(df):
    """
    Analyzes how the tide gate's opening angle affects wildlife detection rates.

    Returns:
        pd.DataFrame: A summary of detection rates by gate state.
        dict: Results of the Chi-square test.
    """
    if 'Gate_Opening_MTR_Deg' not in df.columns:
        print("Gate opening data not found. Skipping gate analysis.")
        return None, None

    df['gate_category'] = pd.cut(
        df['Gate_Opening_MTR_Deg'],
        bins=[-1, 5, 30, 60, 90],
        labels=['Closed (0-5°)', 'Partially Open (5-30°)', 'Open (30-60°)', 'Wide Open (>60°)']
    )

    gate_summary = df.groupby('gate_category').agg(
        Detection_Rate=('has_camera_data', 'mean')
    ).reset_index()
    gate_summary['Detection_Rate_Pct'] = gate_summary['Detection_Rate'] * 100

    # Chi-square test
    contingency_table = pd.crosstab(df['has_camera_data'], df['gate_category'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    test_results = {'chi2': chi2, 'p_value': p_value, 'dof': dof}
    
    print("--- Gate Impact Analysis ---")
    print(gate_summary)
    print(f"\nChi-square test p-value: {p_value:.4f}")
    
    return gate_summary, test_results

def analyze_temporal_patterns(df):
    """Analyzes hourly and seasonal detection patterns."""
    df['hour'] = df['DateTime'].dt.hour
    df['month'] = df['DateTime'].dt.month
    
    hourly_stats = df.groupby('hour')['has_camera_data'].mean().reset_index()
    monthly_stats = df.groupby('month')['has_camera_data'].mean().reset_index()
    
    print("\n--- Temporal Pattern Analysis ---")
    print("Peak Activity Hours (by detection rate):")
    print(hourly_stats.nlargest(3, 'has_camera_data'))
    
    return hourly_stats, monthly_stats

def run_glm_analysis(df):
    """
    Builds and summarizes a Generalized Linear Model (GLM) to predict detection probability.
    """
    predictors = ['Air_Temp_C', 'Tide_Level_In_m', 'Gate_Opening_MTR_Deg', 'Wind_Speed_km_h']
    available_predictors = [p for p in predictors if p in df.columns]
    
    if not available_predictors:
        print("No predictors available for GLM.")
        return None

    model_data = df[['has_camera_data'] + available_predictors].dropna()
    model_data['is_night'] = ((df['DateTime'].dt.hour < 6) | (df['DateTime'].dt.hour >= 18)).astype(int)
    
    formula = f"has_camera_data ~ {' + '.join(available_predictors)} + is_night"
    
    try:
        glm_model = smf.glm(formula=formula, data=model_data, family=sm.families.Binomial()).fit()
        print("\n--- GLM Results ---")
        print(glm_model.summary())
        return glm_model
    except Exception as e:
        print(f"Error fitting GLM: {e}")
        return None