# Privacy-Preserving Synthetic Data Generator

A comprehensive system for generating realistic synthetic tabular data while preserving privacy using Differential Privacy techniques. This tool allows users to upload CSV datasets, train synthetic data generators (CTGAN with optional DP integration), generate privacy-safe synthetic samples, and evaluate the fidelity and privacy metrics of the generated data.

## Features

- **Data Preprocessing**: Automatic detection of numerical and categorical features, handling of missing values, normalization, and encoding
- **Model Training**: CTGAN from SDV with optional Differential Privacy (DP-SGD) using Opacus
- **Synthetic Data Generation**: Generate configurable number of synthetic rows with privacy guarantees
- **Evaluation Metrics**:
  - **Fidelity**: KS test, Chi-square test, correlation comparison, propensity classifier
  - **Privacy**: Membership inference attack, differential privacy guarantees
  - **Utility**: Train-on-synthetic, test-on-real ML performance
- **Visualization**: Distribution comparisons, correlation heatmaps
- **User Interface**: Interactive Streamlit application

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser with the following workflow:

1. **Upload Dataset**: Upload a CSV file or provide a file path
2. **Train Model**: Configure and train a CTGAN model with optional Differential Privacy
3. **Generate Synthetic Data**: Generate and download synthetic data
4. **Evaluate & Visualize**: Compute metrics and visualize comparisons between real and synthetic data

## Project Structure

- `app.py`: Streamlit interface for the application
- `data_preprocess.py`: Data cleaning, encoding, and type detection
- `model_train.py`: CTGAN with Differential Privacy integration
- `evaluate.py`: Fidelity and privacy metric functions
- `requirements.txt`: List of dependencies

## Privacy Considerations

This tool implements Differential Privacy (DP) to provide formal privacy guarantees for synthetic data generation. The privacy budget (ε) controls the privacy-utility trade-off:

- Lower ε values (e.g., 0.1-1.0) provide stronger privacy but may reduce data utility
- Higher ε values (e.g., 5.0-10.0) improve utility but offer weaker privacy guarantees

The membership inference attack metric helps evaluate the risk of privacy leakage in the generated data.

## Example Output

After evaluation, the system provides metrics such as:

- Fidelity Scores: KS similarity, correlation preservation
- Privacy Scores: Membership inference AUC, differential privacy parameters
- Utility Scores: ML model performance on synthetic vs. real data

## Dependencies

- sdv (Synthetic Data Vault)
- pandas, numpy
- scikit-learn, scipy
- matplotlib, seaborn
- streamlit
- opacus, torch

## License

MIT