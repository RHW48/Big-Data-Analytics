# Titanic Survival Prediction with Logistic Regression using PySpark

This project demonstrates how to perform classification using Logistic Regression on the Titanic dataset with PySpark.

## Dataset

The dataset used in this project is the Titanic dataset, which can be found [here](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv).

## Features (X)

- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Gelar (Title extracted from names)

## Target (Y)

- Survived

## Project Steps

1. **Data Loading**
   - The dataset is loaded from a URL and saved as a CSV file.
   - The CSV file is then loaded into a Spark DataFrame.

2. **Data Preprocessing**
   - Extraction of titles from passenger names and creation of a new column 'Gelar'.
   - Selection of relevant columns.
   - Dropping rows with null values.

3. **Feature Engineering**
   - String indexing and one-hot encoding for categorical columns: Sex, Embarked, and Gelar (title).
   - Assembling feature columns into a single vector.

4. **Model Building**
   - Logistic Regression model is defined.
   - A pipeline is created with stages for indexing, encoding, assembling, and the logistic regression model.

5. **Model Training**
   - The dataset is split into training (70%) and testing (30%) sets.
   - The model is trained using the training set.

6. **Model Evaluation**
   - The model is tested on the testing set.
   - Predictions are made and evaluated using the ROC AUC metric.

## Prerequisites

- PySpark
- Pandas
- Regex

## Installation

To install PySpark, you can use pip:

```bash
pip install pyspark
