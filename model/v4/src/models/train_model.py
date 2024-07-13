import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.metrics import AUC, Recall, Precision
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

# FUNCTION: check for each layer if it is frozen or not
def print_layer_info(model):
    for layer in model.layers:
        print(f'Layer: {layer.name}, Trainable: {layer.trainable}')

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

    # Load pre-trained weights and freeze layers
    # pre_trained_weights_path = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/weights_vgg/.'
    # model.load_weights(pre_trained_weights_path)
    
    new_model = Sequential()

    # Add all layers from the old model except the output layer
    for layer in model.layers[:-3]:
        new_model.add(layer)
    
    # Add new fully connected layers
    new_model.add(Dense(512, activation='relu'))
    new_model.add(tf.keras.layers.Dropout(0.5))
    new_model.add(Dense(2, activation='softmax'))

    model = new_model
    model.layers[0].trainable = False
    
    print_layer_info(model)
    
    return model

# FUNCTION: fine tuning the pre-trained model
def fine_tune(fine_tune_model, train_dataset, val_dataset, epochs):
    checkpoint_path = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/training_weights/cp.ckpt'
    os.makedirs(checkpoint_path, exist_ok=True)

    # Define model parameters
    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve='ROC'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )   

    # define call back for training weights
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
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

    fine_tune_model.load_weights(checkpoint_path)

    return history, fine_tune_model