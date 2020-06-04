import pandas as pd
from toxicity_model.config import config

def validate_input(input_data: pd.DataFrame) -> pd.DataFrame:
    """ Check inputs"""
    
    validated_data = input_data.copy()
    
    if input_data[[config.TEXT]].isnull().any().any():
        validated_data = validated_data.dropna(axis=0, subset = [config.TEXT])
        
    return validated_data