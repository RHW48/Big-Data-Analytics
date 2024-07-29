import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
 
# Load the data
st.title("Wine Quality Prediction")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=';')

st.write("### Dataset")
st.dataframe(data)  

st.write("### First 5 Rows of Data")
st.write(data.head())

st.write("### Data Description")
st.write(data.describe())

# Preprocess data
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Split data into features and target
X = data.drop('quality', axis=1)
y = data['quality']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
logreg = LogisticRegression(max_iter=10000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
auc_logreg = roc_auc_score(y_test, y_pred_logreg)

# Train Decision Tree model
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
auc_tree = roc_auc_score(y_test, y_pred_tree)

st.write("### Model Evaluation")

st.write("#### Logistic Regression")
st.write(f"Accuracy: {accuracy_logreg}")
st.write(f"AUC: {auc_logreg}")

st.write("#### Decision Tree")
st.write(f"Accuracy: {accuracy_tree}")
st.write(f"AUC: {auc_tree}")

# Plot confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
st.write("### Confusion Matrix for Logistic Regression")
fig, ax = plt.subplots()
sns.heatmap(cm_logreg, annot=True, fmt='.2f', xticklabels=["Bad Quality", "Good Quality"], yticklabels=["Bad Quality", "Good Quality"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Logistic Regression')
st.pyplot(fig)

# Plot confusion matrix for Decision Tree
cm_tree = confusion_matrix(y_test, y_pred_tree)
st.write("### Confusion Matrix for Decision Tree")
fig, ax = plt.subplots()
sns.heatmap(cm_tree, annot=True, fmt='.2f', xticklabels=["Bad Quality", "Good Quality"], yticklabels=["Bad Quality", "Good Quality"])
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix for Decision Tree')
st.pyplot(fig)