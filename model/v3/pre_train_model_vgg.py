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
from tensorflow.keras.metrics import AUC, Recall, Precision, BinaryAccuracy

# FUNCTION: apply data augmentation to dataset (and normalize)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = image / 255.0
    return image, label

# FUNCTION: normalize images from dataset
def normalize(image, label):
    image = image / 255.0
    return image, label

# FUNCTION: apply data augmentation and normalize images
def prepare_dataset(dataset, apply_augmentation=False):
    if apply_augmentation:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# FUNCTION: create image dataset from directory
# label_mode='categorical' for multiple classes
def import_images(train_dir, test_dir, val_dir):
    img_size=(224, 224)
    batch_size=256
    label_mode='binary'

    train_dataset = image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='binary'
    )

    validation_dataset = image_dataset_from_directory(
        val_dir,
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='binary'
    )

    test_dataset = image_dataset_from_directory(
        test_dir,
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size,
        label_mode='binary'
    )

    train_dataset = prepare_dataset(train_dataset, apply_augmentation=True)
    validation_dataset = prepare_dataset(validation_dataset, apply_augmentation=False)
    test_dataset = prepare_dataset(test_dataset, apply_augmentation=False)

    return train_dataset, validation_dataset, test_dataset

# FUNCTION: MODEL FOR TRAINING
def train_model(num_classes, epochs, train_dataset, validation_dataset):
    # Load VGG16 Pre Trained CNN
    base_model = VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    # Freeze the base_model layers so they can not be changed
    for layer in base_model.layers:
        layer.trainable = False

    # Building the model and fine tuning
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(1, activation='sigmoid')  # use softmax activation for more than 2 classes and replace "1" with "num_classes"
    ])

    # Define model parameters
    # for multilabel model: use categorical_crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001), 
        loss='binary_crossentropy', 
        metrics=[
            'accuracy', 
            AUC(name='auc', curve='ROC'),
            Precision(name='precision'),
            Recall(name='recall'), 
        ]
    )

    # Define checkpoint, fit the model and save weights
    checkpoint_filepath = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv3/weights_vgg/.'

    # Define the callback to save weights
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, 
        save_weights_only=True, 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max'
    )

    # define the training model
    history = model.fit(
        train_dataset, 
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[model_callback]
    )

    model.load_weights(checkpoint_filepath)
    return history, model

# FUNCTION: TEST THE MODEL ON THE TEST SET
def test_model(model, test_dataset):
    return (model.evaluate(test_dataset))

# FUNCTION: PLOT GRAPH AND SAVE 
def plot_metric_epochs(train_data, val_data, train_label, val_label, x_label, y_label, title, file_name):
    plt.plot(train_data, color='royalblue', label=train_label, linewidth=.85)
    plt.plot(val_data, color='orange', label=val_label, linewidth=.85)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(color='gray')
    plt.legend(loc='best')
    plt.savefig("./graphsVGG/" + file_name + ".png")
    plt.close()

# FUNCTION: CALL THE PLOT FUNCTION
def call_plot(history):
    plot_metric_epochs(history.history['loss'], history.history['val_loss'], "Training Loss", "Validation Loss", "Epochs", "Binary Cross-Entropy Loss", "Loss Comparison", "LossG3")
    plot_metric_epochs(history.history['accuracy'], history.history['val_accuracy'], "Training Accuracy", "Validation Accuracy", "Epochs", "Accuracy", "Accuracy Comparison", "AccuracyG3")
    plot_metric_epochs(history.history['auc'], history.history['val_auc'], "Training AUC", "Validation AUC", "Epochs", "AUC Value", "AUC Comparison", "AUCG3")
    plot_metric_epochs(history.history['precision'], history.history['val_precision'], "Training Precision", "Validation Precision", "Epochs", "Precision Value", "Precision Comparison", "PrecisionG3")
    plot_metric_epochs(history.history['recall'], history.history['val_recall'], "Training Recall", "Validation Recall", "Epochs", "Recall Value", "Recall Comparison", "RecallG3")

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
def confusion_matrix_and_report(model, test_dataset):
    Y_pred = model.predict(test_dataset)
    # y_pred = np.argmax(Y_pred, axis=1)
    y_pred = (Y_pred > 0.5).astype(int)

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
    plt.savefig("./graphsVGG/confusion_matrix.png")
    plt.clf()   # clean figure for the second plot

    return y_true, Y_pred

# FUNCTION: plot ROC Curve
def plotROCCurve(y_true, Y_pred):
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
    plt.savefig("./graphsVGG/roc_curve.png") 
    plt.clf()

# FUNCTION: plot PRECISION_RECALL Curve
def plotPRCurve(y_true, Y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, Y_pred.ravel())
    average_precision = average_precision_score(y_true, Y_pred.ravel())

    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, label='Model (Avg Precision = {:.3f})'.format(average_precision))
    plt.grid(color='#EEEEEE', linewidth=0.7)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.savefig("./graphsVGG/pr_curve.png")
    plt.show()
    plt.clf()

# FUNCTION: MAIN
def main():
    train_path = "/home/rodrigocm/scratch/datasets/eyeq/images/train"
    test_path = "/home/rodrigocm/scratch/datasets/eyeq/images/test"
    val_path = "/home/rodrigocm/scratch/datasets/eyeq/images/validation"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Importando imagens: ")
    train_dataset, val_dataset, test_dataset = import_images(train_path, test_path, val_path)

    num_classes = 2
    epochs = 30
    history, model = train_model(num_classes, epochs, train_dataset, val_dataset)

    # plot training metrics
    call_plot(history=history)

    # plot result metrics
    y_true, Y_pred = confusion_matrix_and_report(model, test_dataset)
    plotPRCurve(y_true, Y_pred)
    plotROCCurve(y_true, Y_pred)

    # test model
    results = test_model(model=model, test_dataset=test_dataset)
    print("Resultados do teste:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")
main()
