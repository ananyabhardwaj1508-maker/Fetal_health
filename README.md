# Fetal Health Prediction Dashboard

A Streamlit-based machine learning dashboard for predicting fetal health using Artificial Neural Networks, Deep Neural Networks, and Recurrent Neural Networks.

## Overview

This application analyzes fetal health indicators and uses multiple ML models to classify fetal health status. It includes:

- **Model Implementations**: ANN, Deep NN, and RNN architectures
- **Feature Selection**: Automated feature selection using SelectKBest
- **Model Interpretability**: SHAP and LIME explanations for model predictions
- **Visualization**: Classification reports, confusion matrices, ROC curves, and PCA analysis
- **Clustering**: DBSCAN clustering analysis
- **Data Scaling**: Standard scaling for preprocessing

## Requirements

- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- shap
- lime
- xgboost

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ntcc
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run ntcc.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data

The application expects a `fetal_health.csv` file with:
- Feature columns with fetal health indicators
- Target column named `fetal_health` with values 1, 2, or 3

## Features

- Interactive model training and evaluation
- Real-time model predictions
- Model performance metrics (accuracy, precision, recall, F1-score)
- Feature importance visualization
- Confusion matrices and ROC curves
- Explainable AI with SHAP and LIME

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

Created for fetal health prediction analysis.
