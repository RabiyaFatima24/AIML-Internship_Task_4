import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score
)


print("==========================================")
print("       LOGISTIC REGRESSION PROJECT")
print("==========================================")


# Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("\n--- FIRST 5 ROWS ---")
print(X.head())

print("\n--- DATASET SHAPE ---")
print(X.shape)

print("\n--- TARGET CLASSES ---")
print(data.target_names)

print("\n--- MISSING VALUES ---")
print(X.isnull().sum().sum())


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create and train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]


# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\n==========================================")
print("           MODEL EVALUATION")
print("==========================================")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("ROC-AUC:", round(roc_auc, 4))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- CONFUSION MATRIX ---")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

disp.plot()

plt.title("Logistic Regression Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()


# Threshold tuning
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\n==========================================")
print("           THRESHOLD TUNING")
print("==========================================")

for threshold in thresholds:
    threshold_predictions = (y_prob >= threshold).astype(int)

    threshold_precision = precision_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    threshold_recall = recall_score(
        y_test,
        threshold_predictions,
        zero_division=0
    )

    print(
        "Threshold:",
        threshold,
        "| Precision:",
        round(threshold_precision, 4),
        "| Recall:",
        round(threshold_recall, 4)
    )


# Explain sigmoid function
print("\n==========================================")
print("             SIGMOID FUNCTION")
print("==========================================")

print(
    "Logistic Regression uses the sigmoid function "
    "to convert model output into a probability between 0 and 1."
)

print(
    "A probability above the classification threshold "
    "is assigned to the positive class."
)


print("\n==========================================")
print("      LOGISTIC REGRESSION COMPLETED")
print("==========================================")

print("Output saved as: confusion_matrix.png")