# data_preprocess.py
# Handles data cleaning, encoding, and type detection for synthetic data generation

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Union, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreprocessor:
    """Class for preprocessing tabular data for synthetic data generation."""
    
    def __init__(self):
        self.categorical_columns = []
        self.numerical_columns = []
        self.categorical_encoders = {}
        self.numerical_scalers = {}
        self.numerical_imputers = {}
        self.categorical_imputers = {}
        self.column_types = {}
        self.original_dtypes = {}
        self.one_hot_encoding_maps = {}
        
    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Automatically detect column types (numerical or categorical).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping column names to their detected types
        """
        column_types = {}
        
        for col in df.columns:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                # If it has few unique values relative to total rows, treat as categorical
                if df[col].nunique() < min(20, len(df) * 0.05):
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numerical'
            else:
                column_types[col] = 'categorical'
                
        logging.info(f"Detected {sum(1 for t in column_types.values() if t == 'numerical')} numerical and "
                     f"{sum(1 for t in column_types.values() if t == 'categorical')} categorical columns")
        
        return column_types
    
    def preprocess(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Preprocess the data for training a synthetic data model.
        
        Args:
            df: Input DataFrame
            column_types: Optional dictionary specifying column types. If None, they will be auto-detected.
            
        Returns:
            Preprocessed DataFrame ready for model training
        """
        # Store original data types
        self.original_dtypes = df.dtypes.to_dict()
        
        # Detect column types if not provided
        if column_types is None:
            self.column_types = self.detect_column_types(df)
        else:
            self.column_types = column_types
            
        # Separate columns by type
        self.numerical_columns = [col for col, col_type in self.column_types.items() 
                                 if col_type == 'numerical']
        self.categorical_columns = [col for col, col_type in self.column_types.items() 
                                   if col_type == 'categorical']
        
        # Create a copy of the dataframe to avoid modifying the original
        processed_df = df.copy()
        
        # Handle numerical columns
        for col in self.numerical_columns:
            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            values = df[col].values.reshape(-1, 1)
            imputer.fit(values)
            processed_df[col] = imputer.transform(values).flatten()
            self.numerical_imputers[col] = imputer
            
            # Normalize
            scaler = StandardScaler()
            values = processed_df[col].values.reshape(-1, 1)
            scaler.fit(values)
            processed_df[col] = scaler.transform(values).flatten()
            self.numerical_scalers[col] = scaler
        
        # Handle categorical columns
        for col in self.categorical_columns:
            # Handle missing values
            imputer = SimpleImputer(strategy='most_frequent')
            values = df[col].astype(str).values.reshape(-1, 1)
            imputer.fit(values)
            processed_df[col] = imputer.transform(values).flatten()
            self.categorical_imputers[col] = imputer
            
            # One-hot encode
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            values = processed_df[col].astype(str).values.reshape(-1, 1)
            encoder.fit(values)
            encoded = encoder.transform(values)
            
            # Create new columns for one-hot encoding
            categories = encoder.categories_[0]
            col_names = [f"{col}_{cat}" for cat in categories]
            encoded_df = pd.DataFrame(encoded, columns=col_names, index=processed_df.index)
            
            # Store encoding information for inverse transform
            self.categorical_encoders[col] = encoder
            self.one_hot_encoding_maps[col] = col_names
            
            # Add encoded columns and drop original
            processed_df = pd.concat([processed_df, encoded_df], axis=1)
            processed_df = processed_df.drop(col, axis=1)
        
        logging.info(f"Preprocessing complete. Output shape: {processed_df.shape}")
        return processed_df
    
    def inverse_transform(self, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """Convert preprocessed synthetic data back to original format.
        
        Args:
            synthetic_df: DataFrame of synthetic data in preprocessed format
            
        Returns:
            DataFrame with synthetic data in original format
        """
        result_df = pd.DataFrame(index=synthetic_df.index)
        
        # Inverse transform numerical columns
        for col in self.numerical_columns:
            if col in synthetic_df.columns:
                # Inverse scaling
                values = synthetic_df[col].values.reshape(-1, 1)
                unscaled_values = self.numerical_scalers[col].inverse_transform(values).flatten()
                result_df[col] = unscaled_values
        
        # Inverse transform categorical columns
        for col in self.categorical_columns:
            # Get the one-hot encoded columns for this category
            if col in self.one_hot_encoding_maps:
                col_names = self.one_hot_encoding_maps[col]
                
                # Check if all required columns exist
                if all(col_name in synthetic_df.columns for col_name in col_names):
                    # Extract one-hot encoded values
                    encoded_values = synthetic_df[col_names].values
                    
                    # Inverse transform
                    original_values = self.categorical_encoders[col].inverse_transform(encoded_values)
                    result_df[col] = original_values.flatten()
        
        # Convert back to original dtypes where possible
        for col, dtype in self.original_dtypes.items():
            if col in result_df.columns:
                try:
                    if pd.api.types.is_numeric_dtype(dtype):
                        result_df[col] = result_df[col].astype(dtype)
                except (ValueError, TypeError):
                    # If conversion fails, keep as is
                    pass
        
        logging.info(f"Inverse transformation complete. Output shape: {result_df.shape}")
        return result_df

    def fit_transform(self, df: pd.DataFrame, column_types: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Convenience method to fit preprocessor and transform data in one step.
        
        Args:
            df: Input DataFrame
            column_types: Optional dictionary specifying column types
            
        Returns:
            Preprocessed DataFrame
        """
        return self.preprocess(df, column_types)