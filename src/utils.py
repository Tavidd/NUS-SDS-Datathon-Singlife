import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir("./..")
REPO_PATH = os.getcwd().replace("\\", "/")
DATA_PATH = f"{REPO_PATH}/data"
RESPONSE_COL_NAME = "f_purchase_lh"
df = pd.read_csv(f"{DATA_PATH}/data.csv")

def get_feature_corr_with_response(feature_names: list) -> dict:
    """
    Returns the correlation of the input features with the response
    """
    correlations = {}
    for feature_name in feature_names:
        if not (feature_name in df.columns):
            print(f"Feature {feature_name} not found")
            continue
        feature_correlation = np.corrcoef(df[feature_name], df[RESPONSE_COL_NAME])[0, 1]
        correlations[feature_name] = feature_correlation
    return correlations
