import itertools
import numpy as np
import pandas as pd 
import tensorflow as tf 
from matplotlib import pyplot as plt
from dataset_import_lib import import_images
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Flatten, Dense 
from tensorflow.keras.applications.vgg16 import VGG16 
from tensorflow.keras.metrics import AUC, Recall, Precision 
from plot_lib import call_plot, confusion_matrix_and_report, plotPRCurve, plotROCCurve

# FUNCTION: define pre-trained model
def load_pre_trained_model(): 
    base_model = VGG16(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    # Freeze the base_model layers so they can not be changed
    for layer in base_model.layers:
        layer.trainable = False

    # Building the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Define model parameters
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

    # Load pre-trained weights and freeze layers
    pre_trained_weights_path = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv3/weights_vgg/.'
    model.load_weights(pre_trained_weights_path)

    for layer in model.layers[-4:]:
        layer.trainable = True
    
    return model

# FUNCTION: fine tuning the pre-trained model
def fine_tune(fine_tune_model, train_dataset, val_dataset, epochs):
    fine_tuned_weights_filepath = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv3/weights_vgg_fine_tuned/.'

    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve='ROC'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )   

    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=fine_tuned_weights_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    history = fine_tune_model.fit(
        train_dataset, 
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[model_callback]
    )

    fine_tune_model.save_weights(fine_tuned_weights_filepath)

    return history, fine_tune_model

# FUNCTION: TEST THE MODEL ON THE TEST SET
def test_model(model, test_dataset):
    return (model.evaluate(test_dataset))

# FUNCTION: main
def main():
    train_path = '/home/rodrigocm/scratch/datasets/brset/fundus_photos/train'
    test_path = '/home/rodrigocm/scratch/datasets/brset/fundus_photos/test'
    val_path = '/home/rodrigocm/scratch/datasets/brset/fundus_photos/validation'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Importando imagens: ")
    train_dataset, val_dataset, test_dataset = import_images(train_path, 
                                                             test_path, 
                                                             val_path, 
                                                             img_size=(224, 224), 
                                                             batch_size=256, 
                                                             label_mode='binary')

    # load model and fine tune
    fine_tune_model = load_pre_trained_model()
    history, model = fine_tune(fine_tune_model=fine_tune_model, 
                               train_dataset=train_dataset, 
                               val_dataset=val_dataset, 
                               epochs=30)

    # plot training metrics
    call_plot(history=history, save_path="./graphsVGG_fine_tuned/")

    # plot result metrics
    y_true, Y_pred = confusion_matrix_and_report(model, 
                                                 test_dataset, 
                                                 "./graphsVGG_fine_tuned/confusion_matrix.png", 
                                                 threshold=0.75)
    plotPRCurve(y_true, Y_pred, save_path="./graphsVGG_fine_tuned/pr_curve.png")
    plotROCCurve(y_true, Y_pred, save_path="./graphsVGG_fine_tuned/roc_curve.png")

    # test model
    results = test_model(model=model, test_dataset=test_dataset)
    print("Resultados do teste:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")
main()
