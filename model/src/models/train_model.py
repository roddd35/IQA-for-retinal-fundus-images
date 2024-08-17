import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.metrics import AUC, Recall, Precision
from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.vgg16 import VGG16

'''
Functions to train the model
This file is structured in a way
that the user is loading a pre-trained
CNN and fine-tuning it to a new dataset.
If not necessary, the weights loading may
be excluded, and the training strategy
may be changed.
'''
'''
TREINAR AS CAMADAS CONVOLUCIONAIS
DO EYEQ PARA TENTAR DAR LOAD NOS PESOS
E CONSEGUIR UM RESULTADO MELHOR
AMANHA SEM FALTA!!!
'''

# FUNCTION: check for each layer if it is frozen or not
def print_layer_info(model):
    for layer in model.layers:
        print(f'Layer: {layer.name}, Trainable: {layer.trainable}')

# FUNCTION: define pre-trained model
def load_pre_trained_model(): 
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )

    # Building the model
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Load pre-trained weights and freeze layers
    pre_trained_weights_path = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/weights_resnet/.'
    model.load_weights(pre_trained_weights_path)
    
    new_model = Sequential()

    # Change the loaded model so it has a softmax output
    for layer in model.layers[:-1]:
        new_model.add(layer)
    
    # new_model.add(Dense(512, activation='relu'))
    # new_model.add(tf.keras.layers.Dropout(0.5))
    new_model.add(Dense(2, activation='softmax'))

    model = new_model
    model.layers[0].trainable = False
    
    print_layer_info(model)
    
    return model

# FUNCTION: fine tuning the pre-trained model
def fine_tune(fine_tune_model, train_dataset, val_dataset, fcEpochs, cnnEpochs):
    checkpoint_path = '/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/training_weights/resnet'
    checkpoint_file = os.path.join(checkpoint_path, 'cp.ckpt')
    os.makedirs(checkpoint_path, exist_ok=True)

    '''
    training first step
    '''
    # define callback for training weights
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # define model parameters for fully connected training
    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve='ROC'),
        ]
    )

    # fit fully connected
    history_fc = fine_tune_model.fit(
        train_dataset, 
        epochs=fcEpochs,
        validation_data=val_dataset,
        callbacks=[model_callback]
    )
    
    '''
    training second step
    '''
    # fine_tune_model.load_weights(checkpoint_file)
    fine_tune_model.layers[0].trainable = True

    print_layer_info(fine_tune_model)
    
    # define new model parameters for fine-tuning the entire model
    fine_tune_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve='ROC')
        ]
    )

    # fine-tuning the entire model
    history_cnn = fine_tune_model.fit(
        train_dataset,
        epochs=cnnEpochs,
        validation_data=val_dataset,
        callbacks=[model_callback]
    )

    fine_tune_model.load_weights(checkpoint_file)

    # concatenate histories
    history = history_fc.history
    for key in history:
        history[key].extend(history_cnn.history[key])

    return history, fine_tune_model