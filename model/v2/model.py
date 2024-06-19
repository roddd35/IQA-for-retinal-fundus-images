import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC, Recall, Precision, BinaryAccuracy

# FUNCTION: IMPORT IMAGES FROM DATASET
# class_mode='categorical' for multiple classes
def import_images(train_dir, test_dir, val_dir):
    datagen_train = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    datagen_val_test = ImageDataGenerator(
        rescale=1./255,
    )

    train_generator = datagen_train.flow_from_directory(
        train_dir,
        batch_size=128,
        target_size=(224, 224),
        class_mode='binary',
        shuffle=True,
    )

    validation_generator = datagen_val_test.flow_from_directory(
        val_dir,
        batch_size=128,
        target_size=(224, 224),
        class_mode='binary',
        shuffle=False,
    )

    test_generator = datagen_val_test.flow_from_directory(
        test_dir,
        batch_size=128,
        target_size=(224, 224),
        class_mode='binary',
        shuffle=False,
    )

    return train_generator, test_generator, validation_generator

# FUNCTION: PRINT GENERATOR SIZE BY CLASS
def check_generators_size(generator):
    class_indices = generator.class_indices
    print("Mapeamento de classes:", class_indices)

    index_to_class = {v: k for k, v in class_indices.items()}

    class_counts = {class_name: 0 for class_name in class_indices.keys()}
    for label in generator.labels:
        class_name = index_to_class[label]
        class_counts[class_name] += 1

    print("Contagem de imagens por classe:", class_counts)

# FUNCTION: MODEL FOR TRAINING
def train_model(num_classes, epochs, train_generator, validation_generator):
    # Load VGG16 Pre Trained CNN
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base_model layers so they can not be changed
    for layer in base_model.layers:
        layer.trainable = False

    # Building the model and fine tuning
    model = Sequential([
        base_model,
        Flatten(),
        tf.keras.layers.Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(1, activation='sigmoid')  # use softmax activation for more than 2 classes and replace "1" with "num_classes"
    ])

    # Define model parameters
    # for multilabel model: use categorical_crossentropy loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=[
            'accuracy', 
            AUC(name='auc', curve='ROC'),
            Precision(name='precision'),
            Recall(name='recall'), 
            BinaryAccuracy(name='binary_accuracy', threshold=0.3)
        ]
    )

    # Define checkpoint, fit the model and save weights
    checkpoint_filepath = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv2/weights/.'

    # Define the callback to save weights
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, 
        save_weights_only=True, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min'
    )

    # compute class weights to get a better balance of the unbalanced dataset
    # for future training: pre-train on EyeQ dataset, finetune on a subset of BRSet with a more even class distribution
    weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights = {i: weights[i] for i in range(2)}

    # define the training model
    history = model.fit(
        train_generator, 
        epochs=epochs,
        class_weight=class_weights,
        validation_data=validation_generator,
        callbacks=[model_callback]
    )

    model.load_weights(checkpoint_filepath)
    return history, model

# FUNCTION: TEST THE MODEL ON THE TEST SET
def test_model(model, test_generator):
    return (model.evaluate(test_generator))

# FUNCTION: PLOT GRAPH AND SAVE 
def plot_metric_epochs(train_data, val_data, train_label, val_label, x_label, y_label, title, file_name):
    plt.plot(train_data, color='royalblue', label=train_label, linewidth=.85)
    plt.plot(val_data, color='orange', label=val_label, linewidth=.85)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(color='gray')
    plt.legend(loc='best')
    plt.savefig("./graphs/" + file_name + ".png")
    plt.close()

# FUNCTION: CALL THE PLOT FUNCTION
def call_plot(history):
    plot_metric_epochs(history.history['loss'], history.history['val_loss'], "Training Loss", "Validation Loss", "Epochs", "Binary Cross-Entropy Loss", "Loss Comparison", "LossG2")
    plot_metric_epochs(history.history['accuracy'], history.history['val_accuracy'], "Training Accuracy", "Validation Accuracy", "Epochs", "Accuracy", "Accuracy Comparison", "AccuracyG2")
    plot_metric_epochs(history.history['auc'], history.history['val_auc'], "Training AUC", "Validation AUC", "Epochs", "AUC Value", "AUC Comparison", "AUCG2")
    plot_metric_epochs(history.history['precision'], history.history['val_precision'], "Training Precision", "Validation Precision", "Epochs", "Precision Value", "Precision Comparison", "PrecisionG2")
    plot_metric_epochs(history.history['recall'], history.history['val_recall'], "Training Recall", "Validation Recall", "Epochs", "Recall Value", "Recall Comparison", "RecallG2")
    plot_metric_epochs(history.history['binary_accuracy'], history.history['val_binary_accuracy'], "Training Bin Accuracy", "Validation Bin Accuracy", "Epochs", "Bin Accuracy Value", "Bin Accuracy Comparison", "BinAccuracyG2")

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

# FUNCTION: PRINT CONFUSION MATRIX, CLASSIFICATION REPORT AND ROC CURVE
def confusion_matrix_and_report(model, test_generator):
    # use num_of_test_samples // batch_size + 1
    Y_pred = model.predict(test_generator)
    # y_pred = np.argmax(Y_pred, axis=1)
    y_pred = (Y_pred > 0.3).astype(int)
    
    # print confusion matrix
    cm = confusion_matrix(test_generator.classes, y_pred)
    print('Confusion Matrix')
    print(cm)

    # print  classification report
    print('Classification Report')
    target_names = ['Inadequate', 'Adequate']
    print(classification_report(test_generator.classes, y_pred, target_names=target_names, zero_division=0))

    # plot confusion matrix
    plt.figure(figsize=(10, 7))
    plot_confusion_matrix(cm, classes=target_names, title='Confusion Matrix')
    plt.savefig("./graphs/confusion_matrix.png")
    plt.clf()   # clean figure for the second plot

    # plot ROC Curve
    fpr_model, tpr_model, thresholds_model = roc_curve(test_generator.classes, Y_pred.ravel())
    auc_model = auc(fpr_model, tpr_model)

    plt.figure(figsize=(10, 7))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr_model, tpr_model, label='Model (area = {:.3f})'.format(auc_model))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig("./graphs/roc_curve.png") 

# FUNCTION: MAIN
def main():
    train_path = "/home/rodrigocm/scratch/datasets/brset/fundus_photos/train"
    test_path = "/home/rodrigocm/scratch/datasets/brset/fundus_photos/test"
    val_path = "/home/rodrigocm/scratch/datasets/brset/fundus_photos/validation"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Importando imagens: ")
    train_generator, val_generator, test_generator = import_images(train_path, test_path, val_path)

    print("\nDados gerador de treino:")
    check_generators_size(train_generator)

    print("\nDados gerador de validação:")
    check_generators_size(val_generator)

    print("\nDados gerador de teste:")
    check_generators_size(test_generator)

    num_classes = 2
    epochs = 25
    history, model = train_model(num_classes, epochs, train_generator, val_generator)

    call_plot(history=history)

    confusion_matrix_and_report(model, test_generator)

    results = test_model(model=model, test_generator=test_generator)
    print("Resultados do teste:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")
main()
