import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
 
# FUNCTION: extract labels from tf.data.Dataset
def get_true_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label.numpy())    # get a tensor for each label
    return tf.concat(labels, axis=0)    # concatenates tensors

# FUNCTION: test the model on the test set
def test_model(model, test_dataset):
    return (model.evaluate(test_dataset))

# FUNCTION: calculate confusion matrix and classification report
def confusion_matrix_and_report(model, test_dataset, threshold):
    # labels one-hot-encoded
    Y_pred = model.predict(test_dataset)
    y_pred = np.argmax(Y_pred, axis=1)

    y_true = get_true_labels(test_dataset)
    y_true = np.argmax(y_true, axis=1)

    # labels binarias
    # Y_pred = model.predict(test_dataset)
    # y_pred = (Y_pred > threshold).astype(int)

    # y_true = get_true_labels(test_dataset)
    
    # print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # print classification report
    print('Classification Report')
    target_names = ['Inadequate', 'Adequate']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return y_true, Y_pred, cm