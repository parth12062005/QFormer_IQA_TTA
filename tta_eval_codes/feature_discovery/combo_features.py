"""
Automatic combination feature generator.
"""
import numpy as np


def generate_combo_features(df, base_feature_names, rankings_df, top_k=20):
    """
    Generate ratio, product, and difference features from the top K base features.
    
    Args:
        df: DataFrame containing the base feature values for all images.
        base_feature_names: List of all base feature column names.
        rankings_df: DataFrame with base features ranked by absolute Spearman correlation.
        top_k: Number of top base features to use for combinations.
        
    Returns:
        DataFrame containing only the new combo features.
    """
    # Select top K base features
    top_features = rankings_df.nlargest(top_k, "abs_spearman")["feature"].tolist()
    # Filter to ensure they exist in df
    top_features = [f for f in top_features if f in df.columns]
    
    combo_data = {}
    
    n = len(top_features)
    for i in range(n):
        for j in range(i + 1, n):
            f1 = top_features[i]
            f2 = top_features[j]
            
            v1 = df[f1].values.astype(np.float64)
            v2 = df[f2].values.astype(np.float64)
            
            # Ratio
            ratio1 = v1 / (v2 + 1e-8)
            ratio2 = v2 / (v1 + 1e-8)
            combo_data[f"{f1}_div_{f2}"] = ratio1
            combo_data[f"{f2}_div_{f1}"] = ratio2
            
            # Product
            combo_data[f"{f1}_mul_{f2}"] = v1 * v2
            
            # Difference (normalized first to make it scale-invariant)
            std1, std2 = np.std(v1) + 1e-8, np.std(v2) + 1e-8
            v1_norm = (v1 - np.mean(v1)) / std1
            v2_norm = (v2 - np.mean(v2)) / std2
            combo_data[f"{f1}_sub_{f2}"] = v1_norm - v2_norm
            
    import pandas as pd
    combo_df = pd.DataFrame(combo_data, index=df.index)
    
    # Clean up inf/nan
    combo_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in combo_df.columns:
        if combo_df[col].isna().any():
            combo_df[col] = combo_df[col].fillna(combo_df[col].median())
            
    return combo_df
