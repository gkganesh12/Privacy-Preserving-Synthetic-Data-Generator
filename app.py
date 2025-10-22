# app.py
# Streamlit interface for Privacy-Preserving Synthetic Data Generator

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# Import custom modules
from data_preprocess import DataPreprocessor
from model_train import PrivacyPreservingCTGAN
from evaluate import SyntheticDataEvaluator

# Set page configuration
st.set_page_config(
    page_title="Privacy-Preserving Synthetic Data Generator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'real_data' not in st.session_state:
    st.session_state.real_data = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'column_types' not in st.session_state:
    st.session_state.column_types = None
if 'training_metrics' not in st.session_state:
    st.session_state.training_metrics = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None

# Helper functions
def save_dataframe(df: pd.DataFrame, filename: str) -> str:
    """Save DataFrame to CSV and return the path."""
    # Create 'output' directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"output/{timestamp}_{filename}"
    
    # Save to CSV
    df.to_csv(path, index=False)
    return path

def display_metrics(metrics: Dict[str, Any]) -> None:
    """Display evaluation metrics in a formatted way."""
    if not metrics:
        st.warning("No metrics available. Please run evaluation first.")
        return
    
    # Fidelity metrics
    if 'fidelity' in metrics:
        st.subheader("Fidelity Metrics")
        fidelity = metrics['fidelity']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'overall_fidelity' in fidelity and fidelity['overall_fidelity'] is not None:
                st.metric("Overall Fidelity", f"{fidelity['overall_fidelity']:.4f}")
            
            if 'ks_mean' in fidelity and fidelity['ks_mean'] is not None:
                st.metric("KS Similarity (Mean)", f"{fidelity['ks_mean']:.4f}")
        
        with col2:
            if 'correlation_similarity' in fidelity and fidelity['correlation_similarity'] is not None:
                st.metric("Correlation Similarity", f"{fidelity['correlation_similarity']:.4f}")
            
            if 'chi2_mean' in fidelity and fidelity['chi2_mean'] is not None:
                st.metric("Chi-Square Similarity (Mean)", f"{fidelity['chi2_mean']:.4f}")
        
        with col3:
            if 'pmse_similarity' in fidelity and fidelity['pmse_similarity'] is not None:
                st.metric("Propensity Similarity", f"{fidelity['pmse_similarity']:.4f}")
            
            if 'real_vs_synthetic_auc' in fidelity and fidelity['real_vs_synthetic_auc'] is not None:
                st.metric("Real vs Synthetic AUC", f"{fidelity['real_vs_synthetic_auc']:.4f}")
    
    # Privacy metrics
    if 'privacy' in metrics:
        st.subheader("Privacy Metrics")
        privacy = metrics['privacy']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'privacy_score' in privacy and privacy['privacy_score'] is not None:
                st.metric("Privacy Score", f"{privacy['privacy_score']:.4f}")
        
        with col2:
            if 'membership_inference_auc' in privacy and privacy['membership_inference_auc'] is not None:
                st.metric("Membership Inference AUC", f"{privacy['membership_inference_auc']:.4f}")
        
        with col3:
            if 'epsilon' in privacy and privacy['epsilon'] is not None:
                st.metric("Differential Privacy ε", f"{privacy['epsilon']:.2f}")
                if 'delta' in privacy and privacy['delta'] is not None:
                    st.metric("Differential Privacy δ", f"{privacy['delta']:.1e}")
    
    # Utility metrics
    if 'utility' in metrics:
        st.subheader("Utility Metrics")
        utility = metrics['utility']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'utility_ratio' in utility and utility['utility_ratio'] is not None:
                st.metric("Utility Ratio", f"{utility['utility_ratio']:.4f}")
        
        with col2:
            if 'real_model_accuracy' in utility and utility['real_model_accuracy'] is not None:
                st.metric("Real Model Accuracy", f"{utility['real_model_accuracy']:.4f}")
        
        with col3:
            if 'synthetic_model_accuracy' in utility and utility['synthetic_model_accuracy'] is not None:
                st.metric("Synthetic Model Accuracy", f"{utility['synthetic_model_accuracy']:.4f}")

# Main application
st.title("🔒 Privacy-Preserving Synthetic Data Generator")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a step:",
    ["1. Upload Dataset", "2. Train Model", "3. Generate Synthetic Data", "4. Evaluate & Visualize"]
)

# 1. Upload Dataset
if page == "1. Upload Dataset":
    st.header("Upload Dataset")
    
    upload_method = st.radio(
        "Choose upload method:",
        ["Upload CSV file", "Enter file path"]
    )
    
    data_loaded = False
    
    if upload_method == "Upload CSV file":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                data_loaded = True
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:  # Enter file path
        file_path = st.text_input("Enter the path to a CSV file:")
        
        if file_path and st.button("Load Data"):
            try:
                df = pd.read_csv(file_path)
                data_loaded = True
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    if data_loaded:
        st.session_state.real_data = df
        st.success(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display data info
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Statistics:**")
            st.write(f"- Rows: {df.shape[0]}")
            st.write(f"- Columns: {df.shape[1]}")
            st.write(f"- Missing values: {df.isna().sum().sum()}")
        
        with col2:
            st.write("**Column Types:**")
            dtypes = df.dtypes.value_counts().to_dict()
            for dtype, count in dtypes.items():
                st.write(f"- {dtype}: {count} columns")
        
        # Initialize preprocessor and detect column types
        preprocessor = DataPreprocessor()
        column_types = preprocessor.detect_column_types(df)
        
        # Store in session state
        st.session_state.preprocessor = preprocessor
        st.session_state.column_types = column_types
        
        # Display detected column types
        st.subheader("Detected Column Types")
        col_type_df = pd.DataFrame({
            'Column': list(column_types.keys()),
            'Type': list(column_types.values())
        })
        st.dataframe(col_type_df)
        
        st.info("✅ Dataset loaded and preprocessed. Proceed to the 'Train Model' step.")

# 2. Train Model
elif page == "2. Train Model":
    st.header("Train Synthetic Data Model")
    
    if st.session_state.real_data is None:
        st.warning("Please upload a dataset first.")
    else:
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_dp = st.checkbox("Use Differential Privacy", value=True)
            epochs = st.slider("Training Epochs", min_value=50, max_value=500, value=300, step=50)
            batch_size = st.slider("Batch Size", min_value=100, max_value=1000, value=500, step=100)
        
        with col2:
            if use_dp:
                epsilon = st.slider("Privacy Budget (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                delta = st.number_input("Privacy Failure Probability (δ)", 
                                       min_value=1e-7, max_value=1e-3, value=1e-5, format="%e")
                max_grad_norm = st.slider("Max Gradient Norm", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            else:
                epsilon = 0.0
                delta = 0.0
                max_grad_norm = 1.0
        
        # Train button
        if st.button("Train Model"):
            with st.spinner("Training model... This may take a few minutes."):
                try:
                    # Get data
                    df = st.session_state.real_data
                    preprocessor = st.session_state.preprocessor
                    column_types = st.session_state.column_types
                    
                    # Preprocess data
                    processed_df = preprocessor.fit_transform(df, column_types)
                    
                    # Get discrete columns for CTGAN
                    discrete_columns = [col for col in processed_df.columns 
                                       if any(col.startswith(f"{c}_") for c in preprocessor.categorical_columns)]
                    
                    # Initialize and train model
                    model = PrivacyPreservingCTGAN(
                        use_dp=use_dp,
                        epsilon=epsilon,
                        delta=delta,
                        max_grad_norm=max_grad_norm,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=True
                    )
                    
                    # Train model
                    training_metrics = model.train(processed_df, discrete_columns)
                    
                    # Store model and metrics in session state
                    st.session_state.model = model
                    st.session_state.training_metrics = training_metrics
                    
                    st.success("Model trained successfully!")
                    
                    # Display training metrics
                    st.subheader("Training Metrics")
                    st.write(f"Training time: {training_metrics['training_time']:.2f} seconds")
                    st.write(f"Epochs: {training_metrics['epochs']}")
                    
                    if use_dp:
                        st.write(f"Final privacy guarantee: (ε={training_metrics['epsilon']:.2f}, δ={training_metrics['delta']:.1e})")
                    
                    st.info("✅ Model trained successfully. Proceed to 'Generate Synthetic Data' step.")
                except Exception as e:
                    st.error(f"Error during training: {e}")

# 3. Generate Synthetic Data
elif page == "3. Generate Synthetic Data":
    st.header("Generate Synthetic Data")
    
    if st.session_state.model is None:
        st.warning("Please train a model first.")
    else:
        # Generation configuration
        st.subheader("Generation Configuration")
        
        # Number of rows to generate
        num_rows = st.slider(
            "Number of synthetic rows to generate", 
            min_value=100, 
            max_value=10000, 
            value=min(5000, len(st.session_state.real_data)),
            step=100
        )
        
        # Generate button
        if st.button("Generate Synthetic Data"):
            with st.spinner("Generating synthetic data..."):
                try:
                    # Get model and preprocessor
                    model = st.session_state.model
                    preprocessor = st.session_state.preprocessor
                    
                    # Generate synthetic data
                    synthetic_processed = model.generate(num_rows)
                    
                    # Inverse transform to original format
                    synthetic_data = preprocessor.inverse_transform(synthetic_processed)
                    
                    # Store in session state
                    st.session_state.synthetic_data = synthetic_data
                    
                    st.success(f"Generated {len(synthetic_data)} synthetic rows successfully!")
                    
                    # Save synthetic data
                    output_path = save_dataframe(synthetic_data, "synthetic_data.csv")
                    st.info(f"Synthetic data saved to: {output_path}")
                    
                    # Provide download link
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download Synthetic Data",
                            data=f,
                            file_name="synthetic_data.csv",
                            mime="text/csv"
                        )
                    
                    # Display data preview
                    st.subheader("Synthetic Data Preview")
                    st.dataframe(synthetic_data.head())
                    
                    st.info("✅ Synthetic data generated. Proceed to 'Evaluate & Visualize' step.")
                except Exception as e:
                    st.error(f"Error generating synthetic data: {e}")

# 4. Evaluate & Visualize
elif page == "4. Evaluate & Visualize":
    st.header("Evaluate & Visualize")
    
    if st.session_state.real_data is None or st.session_state.synthetic_data is None:
        st.warning("Please generate synthetic data first.")
    else:
        # Get data
        real_data = st.session_state.real_data
        synthetic_data = st.session_state.synthetic_data
        
        # Evaluation options
        st.subheader("Evaluation Options")
        
        # Target column for utility evaluation
        target_column = None
        if len(real_data.columns) > 0:
            target_column = st.selectbox(
                "Select target column for utility evaluation (optional):",
                [None] + list(real_data.columns)
            )
        
        # Evaluate button
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                try:
                    # Initialize evaluator
                    evaluator = SyntheticDataEvaluator()
                    
                    # Set column types if available
                    if st.session_state.column_types is not None:
                        evaluator.set_column_types(st.session_state.column_types)
                    
                    # Get privacy metrics if available
                    privacy_metrics = None
                    if st.session_state.training_metrics is not None and 'epsilon' in st.session_state.training_metrics:
                        privacy_metrics = {
                            'epsilon': st.session_state.training_metrics['epsilon'],
                            'delta': st.session_state.training_metrics['delta']
                        }
                    
                    # Run evaluation
                    results = evaluator.evaluate_all(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        privacy_metrics=privacy_metrics,
                        target_column=target_column
                    )
                    
                    # Store results
                    st.session_state.evaluation_results = results
                    
                    st.success("Evaluation completed successfully!")
                    
                    # Save evaluation results
                    os.makedirs('output', exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_path = f"output/{timestamp}_evaluation_results.json"
                    
                    # Convert numpy values to Python native types for JSON serialization
                    def convert_to_serializable(obj):
                        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                           np.uint8, np.uint16, np.uint32, np.uint64)):
                            return int(obj)
                        elif isinstance(obj, (np.float64, np.float16, np.float32)):
                            return float(obj)
                        elif isinstance(obj, (np.ndarray,)):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_to_serializable(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_serializable(i) for i in obj]
                        else:
                            return obj
                    
                    serializable_results = convert_to_serializable(results)
                    
                    with open(results_path, 'w') as f:
                        json.dump(serializable_results, f, indent=2)
                    
                    st.info(f"Evaluation results saved to: {results_path}")
                except Exception as e:
                    st.error(f"Error during evaluation: {e}")
        
        # Display evaluation results if available
        if st.session_state.evaluation_results is not None:
            display_metrics(st.session_state.evaluation_results)
            
            # Visualizations
            st.subheader("Visualizations")
            
            # Column selection for distribution plots
            if len(real_data.columns) > 0:
                # Get numerical columns
                numerical_cols = [col for col in real_data.columns 
                                if pd.api.types.is_numeric_dtype(real_data[col])]
                
                if len(numerical_cols) > 0:
                    selected_columns = st.multiselect(
                        "Select columns for distribution plots:",
                        options=real_data.columns,
                        default=numerical_cols[:min(5, len(numerical_cols))]
                    )
                    
                    if selected_columns:
                        # Create distribution plots
                        evaluator = SyntheticDataEvaluator()
                        fig = evaluator.plot_distributions(
                            real_data=real_data,
                            synthetic_data=synthetic_data,
                            columns=selected_columns,
                            figsize=(15, 10)
                        )
                        st.pyplot(fig)
                
                # Correlation heatmap
                if st.checkbox("Show Correlation Heatmaps", value=True):
                    evaluator = SyntheticDataEvaluator()
                    fig = evaluator.plot_correlation_comparison(
                        real_data=real_data,
                        synthetic_data=synthetic_data,
                        figsize=(15, 7)
                    )
                    st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Privacy-Preserving Synthetic Data Generator using CTGAN and Differential Privacy. "
    "This tool allows you to generate synthetic data that preserves statistical patterns "
    "while protecting sensitive information."
)

# Display current state
if st.sidebar.checkbox("Show Debug Info", value=False):
    st.sidebar.subheader("Current State")
    state_info = {
        "Data Loaded": st.session_state.real_data is not None,
        "Model Trained": st.session_state.model is not None,
        "Synthetic Data Generated": st.session_state.synthetic_data is not None,
        "Evaluation Complete": st.session_state.evaluation_results is not None
    }
    for key, value in state_info.items():
        st.sidebar.write(f"{key}: {value}")