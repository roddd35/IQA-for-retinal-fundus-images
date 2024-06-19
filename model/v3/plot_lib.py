import itertools
import numpy as np
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc 
from sklearn.metrics import precision_recall_curve, average_precision_score 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten, Dense 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.preprocessing import image_dataset_from_directory 
from tensorflow.keras.metrics import AUC, Recall, Precision 

# FUNCTION: PLOT GRAPH AND SAVE 
def plot_metric_epochs(train_data, val_data, train_label, val_label, x_label, y_label, title, file_name, save_path):
    plt.plot(train_data, color='royalblue', label=train_label, linewidth=.85)
    plt.plot(val_data, color='orange', label=val_label, linewidth=.85)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(color='gray')
    plt.legend(loc='best')
    plt.savefig(save_path + file_name + ".png")
    plt.close()

# FUNCTION: CALL THE PLOT FUNCTION
def call_plot(history, save_path):
    plot_metric_epochs(history.history['loss'], history.history['val_loss'], "Training Loss", "Validation Loss", "Epochs", "Binary Cross-Entropy Loss", "Loss Comparison", "LossG3", save_path)
    plot_metric_epochs(history.history['accuracy'], history.history['val_accuracy'], "Training Accuracy", "Validation Accuracy", "Epochs", "Accuracy", "Accuracy Comparison", "AccuracyG3", save_path)
    plot_metric_epochs(history.history['auc'], history.history['val_auc'], "Training AUC", "Validation AUC", "Epochs", "AUC Value", "AUC Comparison", "AUCG3", save_path)
    plot_metric_epochs(history.history['precision'], history.history['val_precision'], "Training Precision", "Validation Precision", "Epochs", "Precision Value", "Precision Comparison", "PrecisionG3", save_path)
    plot_metric_epochs(history.history['recall'], history.history['val_recall'], "Training Recall", "Validation Recall", "Epochs", "Recall Value", "Recall Comparison", "RecallG3", save_path)

# FUNCTION: extract labels from tf.data.Dataset
def get_true_labels(dataset):
    labels = []
    for _, label in dataset:
        labels.append(label.numpy())    # get a tensor for each label
    return tf.concat(labels, axis=0)    # concatenates tensors

# FUNCTION: PLOT CONFUSION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão, sem normalização')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label verdadeiro')
    plt.xlabel('Label previsto')

# FUNCTION: plot confusion matrix and print classification report
def confusion_matrix_and_report(model, test_dataset, save_path, threshold):
    Y_pred = model.predict(test_dataset)
    # y_pred = np.argmax(Y_pred, axis=1)
    y_pred = (Y_pred > threshold).astype(int)

    y_true = get_true_labels(test_dataset)
    
    # print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # print classification report
    print('Classification Report')
    target_names = ['Inadequate', 'Adequate']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # plot confusion matrix
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix')
    plt.savefig(save_path + ".png")
    plt.clf()   # clean figure for the second plot

    return y_true, Y_pred

# FUNCTION: plot ROC Curve
def plotROCCurve(y_true, Y_pred, save_path):
    # use ravel() because it is a binary classification problem
    fpr_model, tpr_model, thresholds_model = roc_curve(y_true, Y_pred.ravel())
    auc_model = auc(fpr_model, tpr_model)

    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr_model, tpr_model, label='Model (area = {:.3f})'.format(auc_model))
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(save_path + ".png") 
    plt.clf()

# FUNCTION: plot PRECISION_RECALL Curve
def plotPRCurve(y_true, Y_pred, save_path):
    precision, recall, thresholds = precision_recall_curve(y_true, Y_pred.ravel())
    average_precision = average_precision_score(y_true, Y_pred.ravel())

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='Model (Avg Precision = {:.3f})'.format(average_precision))
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.savefig(save_path + ".png")
    plt.show()
    plt.clf()