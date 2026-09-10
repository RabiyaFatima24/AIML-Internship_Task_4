# Task 4: Logistic Regression

## Objective

Build a Logistic Regression model for binary classification using the Breast Cancer Wisconsin dataset.

## Dataset

Breast Cancer Wisconsin dataset provided by Scikit-learn.

The model predicts whether a tumor belongs to one of two classes.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Steps Performed

1. Loaded the Breast Cancer Wisconsin dataset.
2. Checked the dataset shape and missing values.
3. Split the data into training and testing sets.
4. Standardized the numerical features.
5. Trained a Logistic Regression model.
6. Made predictions on the test data.
7. Evaluated the model using accuracy, precision, recall, and ROC-AUC.
8. Created a confusion matrix.
9. Tested different classification thresholds.
10. Explained the sigmoid function used in Logistic Regression.

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC

## Threshold Tuning

Different probability thresholds were tested to understand how changing the threshold affects precision and recall.

## Sigmoid Function

Logistic Regression uses the sigmoid function to convert the model output into a probability between 0 and 1.

A probability above the selected classification threshold is assigned to the positive class.

## Output

- confusion_matrix.png - confusion matrix visualization.
- TASK_4_logistic_regression.py - Python implementation.

## Conclusion

Logistic Regression was successfully used for binary classification. Standardization, evaluation metrics, confusion matrix analysis, and threshold tuning helped understand the model's performance.