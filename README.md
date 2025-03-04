# California Housing Price Predictor

## Overview
Web application to predict California house prices using a Random Forest Regressor model with KNN imputation.

## Prerequisites
- Python 3.8+
- pip (Python package manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/suraj2906/ve3-house-prediction
cd ve3-house-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install flask numpy joblib
```

## Model Download
Due to GitHub's 100 MB file size restriction, download the model files from:
[Google Drive Model Link](https://drive.google.com/file/d/1E0Gkzbb76EVnqzvZTlkgqGz2emzcrA3B/view?usp=sharing)

Required files:
- `rf_model.pkl` (From the Drive)

Place these files in the project root directory.

## Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Model Details
- **Algorithm**: Random Forest Regressor
- **Imputation**: K-Nearest Neighbors (KNN)
- **Features**: 8 housing characteristics
