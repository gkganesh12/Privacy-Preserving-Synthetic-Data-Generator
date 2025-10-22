# evaluate.py
# Implements metrics for evaluating synthetic data quality and privacy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from typing import Dict, List, Tuple, Union, Optional, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SyntheticDataEvaluator:
    """Evaluates synthetic data quality and privacy."""
    
    def __init__(self):
        self.metrics = {}
        self.column_types = {}
    
    def set_column_types(self, column_types: Dict[str, str]) -> None:
        """Set column types for evaluation.
        
        Args:
            column_types: Dictionary mapping column names to types ('numerical' or 'categorical')
        """
        self.column_types = column_types
    
    def evaluate_fidelity(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate how well synthetic data preserves statistical properties of real data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            
        Returns:
            Dictionary of fidelity metrics
        """
        logging.info("Evaluating fidelity metrics...")
        fidelity_metrics = {}
        
        # Ensure we're only comparing columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        
        # If column types weren't explicitly set, detect them
        if not self.column_types:
            self.column_types = {}
            for col in common_columns:
                if pd.api.types.is_numeric_dtype(real_data[col]):
                    self.column_types[col] = 'numerical'
                else:
                    self.column_types[col] = 'categorical'
        
        # 1. Kolmogorov-Smirnov test for numerical columns
        ks_scores = {}
        for col in common_columns:
            if col in self.column_types and self.column_types[col] == 'numerical':
                try:
                    # Drop NaN values for KS test
                    real_col = real_data[col].dropna()
                    synth_col = synthetic_data[col].dropna()
                    
                    if len(real_col) > 0 and len(synth_col) > 0:
                        ks_statistic, ks_pvalue = stats.ks_2samp(real_col, synth_col)
                        # Convert to similarity score (1 - statistic)
                        ks_scores[col] = 1.0 - ks_statistic
                except Exception as e:
                    logging.warning(f"Could not compute KS test for column {col}: {e}")
                    ks_scores[col] = None
        
        # Average KS similarity score
        valid_ks_scores = [score for score in ks_scores.values() if score is not None]
        if valid_ks_scores:
            fidelity_metrics['ks_mean'] = np.mean(valid_ks_scores)
            fidelity_metrics['ks_scores'] = ks_scores
        else:
            fidelity_metrics['ks_mean'] = None
            fidelity_metrics['ks_scores'] = {}
        
        # 2. Chi-square test for categorical columns
        chi2_scores = {}
        for col in common_columns:
            if col in self.column_types and self.column_types[col] == 'categorical':
                try:
                    # Get value counts
                    real_counts = real_data[col].value_counts(normalize=True)
                    synth_counts = synthetic_data[col].value_counts(normalize=True)
                    
                    # Align the indices
                    all_values = list(set(real_counts.index) | set(synth_counts.index))
                    real_counts = real_counts.reindex(all_values, fill_value=0)
                    synth_counts = synth_counts.reindex(all_values, fill_value=0)
                    
                    # Chi-square test
                    chi2_stat, chi2_pvalue = stats.chisquare(
                        f_obs=synth_counts * len(synthetic_data),
                        f_exp=real_counts * len(synthetic_data)
                    )
                    
                    # Convert to similarity score (using p-value)
                    chi2_scores[col] = chi2_pvalue
                except Exception as e:
                    logging.warning(f"Could not compute Chi-square test for column {col}: {e}")
                    chi2_scores[col] = None
        
        # Average Chi-square similarity score
        valid_chi2_scores = [score for score in chi2_scores.values() if score is not None]
        if valid_chi2_scores:
            fidelity_metrics['chi2_mean'] = np.mean(valid_chi2_scores)
            fidelity_metrics['chi2_scores'] = chi2_scores
        else:
            fidelity_metrics['chi2_mean'] = None
            fidelity_metrics['chi2_scores'] = {}
        
        # 3. Correlation similarity
        try:
            # Calculate correlation matrices
            numerical_cols = [col for col in common_columns 
                             if col in self.column_types and self.column_types[col] == 'numerical']
            
            if len(numerical_cols) >= 2:
                real_corr = real_data[numerical_cols].corr().values.flatten()
                synth_corr = synthetic_data[numerical_cols].corr().values.flatten()
                
                # Calculate correlation similarity (using Pearson correlation between correlation matrices)
                corr_similarity, _ = stats.pearsonr(real_corr, synth_corr)
                fidelity_metrics['correlation_similarity'] = corr_similarity
            else:
                fidelity_metrics['correlation_similarity'] = None
        except Exception as e:
            logging.warning(f"Could not compute correlation similarity: {e}")
            fidelity_metrics['correlation_similarity'] = None
        
        # 4. Propensity mean squared error (pMSE)
        try:
            # Create a combined dataset with a label indicating real (1) or synthetic (0)
            real_labeled = real_data.copy()
            real_labeled['is_real'] = 1
            synth_labeled = synthetic_data.copy()
            synth_labeled['is_real'] = 0
            
            # Combine datasets
            combined_data = pd.concat([real_labeled, synth_labeled], axis=0)
            
            # Select only common columns for classification
            X = combined_data[common_columns]
            y = combined_data['is_real']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train a classifier to distinguish real from synthetic
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            
            # Predict probabilities
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate AUC
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # For pMSE, we want a score where closer to 0.5 (random) is better
            # Convert AUC to a similarity score (0 = perfect distinguishability, 1 = indistinguishable)
            pmse_similarity = 1.0 - 2.0 * abs(auc - 0.5)
            fidelity_metrics['pmse_similarity'] = pmse_similarity
            fidelity_metrics['real_vs_synthetic_auc'] = auc
        except Exception as e:
            logging.warning(f"Could not compute propensity score: {e}")
            fidelity_metrics['pmse_similarity'] = None
            fidelity_metrics['real_vs_synthetic_auc'] = None
        
        # 5. Overall fidelity score (average of available metrics)
        available_scores = []
        if fidelity_metrics['ks_mean'] is not None:
            available_scores.append(fidelity_metrics['ks_mean'])
        if fidelity_metrics['chi2_mean'] is not None:
            available_scores.append(fidelity_metrics['chi2_mean'])
        if fidelity_metrics['correlation_similarity'] is not None:
            available_scores.append(fidelity_metrics['correlation_similarity'])
        if fidelity_metrics['pmse_similarity'] is not None:
            available_scores.append(fidelity_metrics['pmse_similarity'])
        
        if available_scores:
            fidelity_metrics['overall_fidelity'] = np.mean(available_scores)
        else:
            fidelity_metrics['overall_fidelity'] = None
        
        logging.info(f"Fidelity evaluation complete. Overall score: {fidelity_metrics['overall_fidelity']}")
        return fidelity_metrics
    
    def evaluate_privacy(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                         privacy_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate privacy protection of synthetic data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            privacy_metrics: Optional dictionary with privacy guarantees (epsilon, delta)
            
        Returns:
            Dictionary of privacy metrics
        """
        logging.info("Evaluating privacy metrics...")
        privacy_results = {}
        
        # Include formal privacy guarantees if provided
        if privacy_metrics is not None:
            if 'epsilon' in privacy_metrics:
                privacy_results['epsilon'] = privacy_metrics['epsilon']
            if 'delta' in privacy_metrics:
                privacy_results['delta'] = privacy_metrics['delta']
        
        # Membership inference attack
        try:
            # Ensure we're only comparing columns that exist in both datasets
            common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
            
            # Split real data into "training set" and "held out set"
            train_data, test_data = train_test_split(real_data, test_size=0.2, random_state=42)
            
            # Create a dataset for membership inference
            # Label 1: record was in training set
            # Label 0: record was not in training set (either test set or synthetic)
            train_labeled = train_data.copy()
            train_labeled['in_training'] = 1
            
            test_labeled = test_data.copy()
            test_labeled['in_training'] = 0
            
            # Sample same number of synthetic records as test records
            synth_sample = synthetic_data.sample(n=len(test_data), random_state=42)
            synth_labeled = synth_sample.copy()
            synth_labeled['in_training'] = 0
            
            # Combine datasets
            inference_data = pd.concat([train_labeled, test_labeled, synth_labeled], axis=0)
            
            # Select only common columns for classification
            X = inference_data[common_columns]
            y = inference_data['in_training']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Train a classifier for membership inference
            clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            clf.fit(X_train, y_train)
            
            # Evaluate on test set
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            privacy_results['membership_inference_accuracy'] = accuracy
            privacy_results['membership_inference_auc'] = auc
            
            # Privacy risk score (higher AUC = higher risk)
            # Convert to a 0-1 scale where 0 = high risk, 1 = low risk
            privacy_results['privacy_score'] = 1.0 - (auc - 0.5) * 2 if auc > 0.5 else 1.0
            
            logging.info(f"Membership inference results - AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            logging.warning(f"Could not compute membership inference attack: {e}")
            privacy_results['membership_inference_accuracy'] = None
            privacy_results['membership_inference_auc'] = None
            privacy_results['privacy_score'] = None
        
        return privacy_results
    
    def evaluate_utility(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                         target_column: str) -> Dict[str, Any]:
        """Evaluate utility of synthetic data for machine learning tasks.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            target_column: Target column for prediction task
            
        Returns:
            Dictionary of utility metrics
        """
        logging.info(f"Evaluating utility metrics for target column: {target_column}")
        utility_metrics = {}
        
        try:
            # Ensure target column exists in both datasets
            if target_column not in real_data.columns or target_column not in synthetic_data.columns:
                raise ValueError(f"Target column {target_column} not found in both datasets")
            
            # Get feature columns (excluding target)
            feature_columns = [col for col in real_data.columns if col != target_column]
            
            # Prepare real data
            X_real = real_data[feature_columns]
            y_real = real_data[target_column]
            
            # Split real data for testing
            X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
                X_real, y_real, test_size=0.2, random_state=42)
            
            # Prepare synthetic data for training
            X_synth = synthetic_data[feature_columns]
            y_synth = synthetic_data[target_column]
            
            # Train model on real data (baseline)
            real_model = RandomForestClassifier(n_estimators=100, random_state=42)
            real_model.fit(X_real_train, y_real_train)
            real_preds = real_model.predict(X_real_test)
            real_accuracy = accuracy_score(y_real_test, real_preds)
            
            # Train model on synthetic data, test on real
            synth_model = RandomForestClassifier(n_estimators=100, random_state=42)
            synth_model.fit(X_synth, y_synth)
            synth_preds = synth_model.predict(X_real_test)
            synth_accuracy = accuracy_score(y_real_test, synth_preds)
            
            # Calculate utility as ratio of synthetic to real performance
            utility_ratio = synth_accuracy / real_accuracy if real_accuracy > 0 else 0
            
            # Store results
            utility_metrics['real_model_accuracy'] = real_accuracy
            utility_metrics['synthetic_model_accuracy'] = synth_accuracy
            utility_metrics['utility_ratio'] = utility_ratio
            
            logging.info(f"Utility evaluation complete. Ratio: {utility_ratio:.4f}")
        except Exception as e:
            logging.warning(f"Could not compute utility metrics: {e}")
            utility_metrics['real_model_accuracy'] = None
            utility_metrics['synthetic_model_accuracy'] = None
            utility_metrics['utility_ratio'] = None
        
        return utility_metrics
    
    def plot_distributions(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                           columns: Optional[List[str]] = None, figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Plot distributions of real vs synthetic data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            columns: Optional list of columns to plot (defaults to all numerical columns)
            figsize: Figure size
            
        Returns:
            Matplotlib figure with distribution plots
        """
        # Ensure we're only comparing columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        
        # If no columns specified, use all numerical columns
        if columns is None:
            columns = [col for col in common_columns if pd.api.types.is_numeric_dtype(real_data[col])]
        else:
            # Filter to ensure all requested columns exist in both datasets
            columns = [col for col in columns if col in common_columns]
        
        # Determine grid size based on number of columns
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each column
        for i, col in enumerate(columns):
            if i < len(axes):
                ax = axes[i]
                
                if pd.api.types.is_numeric_dtype(real_data[col]):
                    # Numerical column - use histograms
                    sns.histplot(real_data[col], color='blue', alpha=0.5, label='Real', ax=ax, kde=True)
                    sns.histplot(synthetic_data[col], color='red', alpha=0.5, label='Synthetic', ax=ax, kde=True)
                else:
                    # Categorical column - use bar plots
                    real_counts = real_data[col].value_counts(normalize=True)
                    synth_counts = synthetic_data[col].value_counts(normalize=True)
                    
                    # Combine categories
                    all_cats = list(set(real_counts.index) | set(synth_counts.index))
                    real_counts = real_counts.reindex(all_cats, fill_value=0)
                    synth_counts = synth_counts.reindex(all_cats, fill_value=0)
                    
                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        'Real': real_counts,
                        'Synthetic': synth_counts
                    })
                    
                    # Plot
                    plot_df.plot(kind='bar', ax=ax, alpha=0.7)
                
                ax.set_title(col)
                ax.legend()
                ax.set_ylabel('Frequency')
                
                # Rotate x-axis labels for categorical variables
                if not pd.api.types.is_numeric_dtype(real_data[col]):
                    ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_comparison(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame,
                                   figsize: Tuple[int, int] = (15, 7)) -> plt.Figure:
        """Plot correlation matrices for real and synthetic data.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            figsize: Figure size
            
        Returns:
            Matplotlib figure with correlation heatmaps
        """
        # Get numerical columns that exist in both datasets
        common_columns = list(set(real_data.columns) & set(synthetic_data.columns))
        numerical_cols = [col for col in common_columns if pd.api.types.is_numeric_dtype(real_data[col])]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot correlation matrices
        if len(numerical_cols) >= 2:
            # Real data correlation
            real_corr = real_data[numerical_cols].corr()
            sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax1, vmin=-1, vmax=1)
            ax1.set_title('Real Data Correlation')
            
            # Synthetic data correlation
            synth_corr = synthetic_data[numerical_cols].corr()
            sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, vmin=-1, vmax=1)
            ax2.set_title('Synthetic Data Correlation')
        else:
            ax1.text(0.5, 0.5, 'Not enough numerical columns\nfor correlation analysis', 
                     ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.5, 'Not enough numerical columns\nfor correlation analysis', 
                     ha='center', va='center', fontsize=12)
            ax1.set_title('Real Data')
            ax2.set_title('Synthetic Data')
        
        plt.tight_layout()
        return fig
    
    def evaluate_all(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame, 
                     privacy_metrics: Optional[Dict[str, Any]] = None,
                     target_column: Optional[str] = None) -> Dict[str, Any]:
        """Run all evaluations and return combined results.
        
        Args:
            real_data: Original real data
            synthetic_data: Generated synthetic data
            privacy_metrics: Optional dictionary with privacy guarantees
            target_column: Optional target column for utility evaluation
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Fidelity metrics
        fidelity_metrics = self.evaluate_fidelity(real_data, synthetic_data)
        results['fidelity'] = fidelity_metrics
        
        # Privacy metrics
        privacy_metrics = self.evaluate_privacy(real_data, synthetic_data, privacy_metrics)
        results['privacy'] = privacy_metrics
        
        # Utility metrics (if target column provided)
        if target_column is not None and target_column in real_data.columns and target_column in synthetic_data.columns:
            utility_metrics = self.evaluate_utility(real_data, synthetic_data, target_column)
            results['utility'] = utility_metrics
        
        # Store results
        self.metrics = results
        
        return results