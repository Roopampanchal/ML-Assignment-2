import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

print("Libraries Imported Successfully!")
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income"
]

df = pd.read_csv(
    "dataset/adult.data",
    names=columns,
    skipinitialspace=True
)

print(df.head())

print("\nShape of Dataset:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

encoder = LabelEncoder()

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = encoder.fit_transform(df[column])

print("\nLabel Encoding Completed!")
print(df.head())

X = df.drop("income", axis=1)
y = df["income"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain-Test Split Completed!")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

y_pred = log_model.predict(X_test)

print("\nPredictions Completed!")
print(y_pred[:10])

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nEvaluation Metrics")
print("-----------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("AUC Score:", auc)
print("MCC Score:", mcc)


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("\nDecision Tree Model Trained Successfully!")
dt_pred = dt_model.predict(X_test)

print("\nDecision Tree Predictions Completed!")
print(dt_pred[:10])
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", dt_accuracy)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_pred)
dt_mcc = matthews_corrcoef(y_test, dt_pred)


print("\nDecision Tree Evaluation Metrics")
print("--------------------------------")
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1 Score :", dt_f1)
print("AUC Score:", dt_auc)
print("MCC Score:", dt_mcc)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("\nKNN Model Trained Successfully!")

knn_pred = knn_model.predict(X_test)

print("\nKNN Predictions Completed!")
print(knn_pred[:10])

knn_accuracy = accuracy_score(y_test, knn_pred)

print("\nKNN Accuracy:", knn_accuracy)

knn_precision = precision_score(y_test, knn_pred)
knn_recall = recall_score(y_test, knn_pred)
knn_f1 = f1_score(y_test, knn_pred)
knn_auc = roc_auc_score(y_test, knn_pred)
knn_mcc = matthews_corrcoef(y_test, knn_pred)

print("\nKNN Evaluation Metrics")
print("------------------------------")
print("Accuracy :", knn_accuracy)
print("Precision:", knn_precision)
print("Recall   :", knn_recall)
print("F1 Score :", knn_f1)
print("AUC Score:", knn_auc)
print("MCC Score:", knn_mcc)


nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

print("\nNaive Bayes Model Trained Successfully!")
nb_pred = nb_model.predict(X_test)

print("\nNaive Bayes Predictions Completed!")
print(nb_pred[:10])

nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Accuracy:", nb_accuracy)


nb_precision = precision_score(y_test, nb_pred)
nb_recall = recall_score(y_test, nb_pred)
nb_f1 = f1_score(y_test, nb_pred)
nb_auc = roc_auc_score(y_test, nb_pred)
nb_mcc = matthews_corrcoef(y_test, nb_pred)

print("\nNaive Bayes Evaluation Metrics")
print("------------------------------")
print("Accuracy :", nb_accuracy)
print("Precision:", nb_precision)
print("Recall   :", nb_recall)
print("F1 Score :", nb_f1)
print("AUC Score:", nb_auc)
print("MCC Score:", nb_mcc)


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Trained Successfully!")

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Predictions Completed!")
print(rf_pred[:10])

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred)
rf_mcc = matthews_corrcoef(y_test, rf_pred)

print("\nRandom Forest Evaluation Metrics")
print("--------------------------------")
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)
print("AUC Score:", rf_auc)
print("MCC Score:", rf_mcc)

results = pd.DataFrame({
    "ML Model": [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        dt_accuracy,
        knn_accuracy,
        nb_accuracy,
        rf_accuracy
    ],
    "AUC": [
        auc,
        dt_auc,
        knn_auc,
        nb_auc,
        rf_auc
    ],
    "Precision": [
        precision,
        dt_precision,
        knn_precision,
        nb_precision,
        rf_precision
    ],
    "Recall": [
        recall,
        dt_recall,
        knn_recall,
        nb_recall,
        rf_recall
    ],
    "F1 Score": [
        f1,
        dt_f1,
        knn_f1,
        nb_f1,
        rf_f1
    ],
    "MCC": [
        mcc,
        dt_mcc,
        knn_mcc,
        nb_mcc,
        rf_mcc
    ]
})

print("MODEL COMPARISON")
print(results)

results.to_csv("model_results.csv", index=False)

print("\nResults saved to model_results.csv")


os.makedirs("models", exist_ok=True)

joblib.dump(log_model, "models/logistic_regression.pkl")
joblib.dump(dt_model, "models/decision_tree.pkl")
joblib.dump(knn_model, "models/knn.pkl")
joblib.dump(nb_model, "models/naive_bayes.pkl")
joblib.dump(rf_model, "models/random_forest.pkl")

print("\nAll models saved successfully!")

X_test.to_csv("test_data.csv", index=False)

print("Test data saved successfully!")
