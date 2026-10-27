# Machine Learning-Based Detection of Fraudulent Mobile Money Transactions in Zimbabwe

## Project Overview
This project uses machine learning to identify potentially fraudulent mobile money transactions. It compares Logistic Regression and Random Forest models and provides a Streamlit web application for fraud-risk analysis.

## Objectives
- Prepare transaction data for machine learning.
- Train and compare classification models.
- Evaluate model performance using suitable metrics.
- Develop a web application for fraud-risk prediction.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Files
- `app.py` — Web application.
- `fraud_detection_random_forest.joblib` — Saved machine learning model.
- `notebooks/fraud_detection_analysis.ipynb` — Model development and analysis.
- `requirements.txt` — Required Python packages.

## How to Run
Install the required packages:

`py -m pip install -r requirements.txt`

Start the application:

`py -m streamlit run app.py`

## Dataset
The project uses the synthetic PaySim mobile money transaction dataset. The dataset may need to be obtained separately and the file path adjusted to match the local computer.

## Limitations
The model was trained using synthetic data and has not been validated on real Zimbabwean mobile money transactions. Predictions indicate potential risk and do not prove fraud.

## Author
Student Name: [Enter Your Name]

Institution: [Enter Your Institution]