import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
 
'''
Functions to evaluate the model performance
In this functions, we aim to print sci-kit learn
library classification report and confusion matrix
and also get the results from tensorflow evaluate
method.
'''

# FUNCTION: extract labels from tf.data.Dataset
def get_true_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label.numpy())    # get a tensor for each label
    return np.concatenate(labels, axis=0)    # concatenate tensors and convert to numpy array

# FUNCTION: test the model on the test set
def test_model(model, test_dataset):
    return (model.evaluate(test_dataset))

# FUNCTION: calculate confusion matrix and classification report
def confusion_matrix_and_report(model, test_dataset, threshold=0.5):
    # labels one-hot-encoded
    Y_pred = model.predict(test_dataset)
    
    # get probabilities for the positive class
    y_pred_prob = Y_pred[:, 1]
    y_pred = (y_pred_prob > threshold).astype(int)

    # Obter labels verdadeiros
    y_true = get_true_labels(test_dataset)
    y_true = np.argmax(y_true, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_prob)

    print("accuracy: {}".format(accuracy))
    print("auc: {}".format(auc))

    # print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # print classification report
    print('Classification Report')
    target_names = ['Inadequate', 'Adequate']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return y_true, Y_pred, cm