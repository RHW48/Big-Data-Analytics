# Wine Quality Prediction

This repository contains two implementations for predicting the quality of wine based on various chemical properties using Logistic Regression and Decision Tree models. The implementations are available as a Jupyter Notebook and a Streamlit web application.

## Features

- Load and preprocess the wine quality dataset.
- Perform exploratory data analysis (EDA).
- Train and evaluate Logistic Regression and Decision Tree models.
- Display model performance metrics and confusion matrices.

## File Structure

- `model_accuracy.ipynb`: Jupyter Notebook containing the full data analysis and model building process.
- `streamlit.py`: Main script containing the Streamlit application code.
- `requirements.txt`: List of dependencies required to run the applications.
- `README.md`: Documentation file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/wine-quality-prediction.git
   cd wine-quality-prediction
2. Install the required dependencies:
   ```bash
    pip install -r requirements.txt

## Usage
### Jupyter Notebook
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook model_accuracy.ipynb
2. Run the notebook cells to execute the data analysis, model training, and evaluation steps.

### Streamlit Application
1. Run the Streamlit application:
   ```bash
   streamlit run streamlit.py
2. Open your web browser and go to http://localhost:8501 to view the app.

### Data Source
The dataset used in these applications is the Wine Quality dataset from the UCI Machine Learning Repository.

## Models
### Logistic Regression
Logistic Regression is used to predict the quality of wine. The model is trained with the following parameters:
- max_iter=10000

### Decision Tree
A Decision Tree classifier is also used to predict wine quality. The model is trained with the default parameters.

## Evaluation Metrics
- Accuracy: The proportion of correctly classified samples.
- AUC (Area Under the Curve): The area under the ROC curve, which measures the ability of the model to distinguish between classes.
- Confusion Matrix: A table that describes the performance of a classification model by comparing the actual and predicted classes.

## Dependencies
- pandas
- scikit-learn
- seaborn
- matplotlib
- streamlit
