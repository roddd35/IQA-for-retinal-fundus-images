import os
import itertools
import numpy as np
import tensorflow as tf
from tensorflow.data import AUTOTUNE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

'''
Test script for IQA algorithm on BRSet
- Choose path to load images
- Choose path to load trained weights
- Reconfigure the neural network
- Execute the script to get results 
'''

'''
Testing model functions
'''
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
    # labels binarias
    Y_pred = model.predict(test_dataset)
    y_pred = (Y_pred > threshold).astype(int)

    y_true = get_true_labels(test_dataset).numpy()
    
    # print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix')
    print(cm)

    # print classification report
    print('Classification Report')
    target_names = ['Inadequate', 'Adequate']
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    # print other metrics, according to SKLearn calculation
    # includes threshold variation
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, Y_pred)
    
    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")

    return y_true, Y_pred, cm

'''
Importing model and loading weights
'''
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

    # Define model parameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.000001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc', curve='ROC'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )   
    
    return model

'''
Importing images
'''
# FUNCTION: apply data augmentation to dataset (and normalize)
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = preprocess_input(image)
    return image, label

# FUNCTION: normalize images from dataset
def normalize(image, label):
    image = preprocess_input(image)
    return image, label

# FUNCTION: apply data augmentation and normalize images
def prepare_dataset(dataset, apply_augmentation=False):
    if apply_augmentation:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset.cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# FUNCTION: create image dataset from directory
def import_images(train_dir, test_dir, val_dir, img_size, batch_size, label_mode):
    train_dataset = image_dataset_from_directory(
        train_dir,
        shuffle=True,
        batch_size=batch_size,
        image_size=img_size,
        label_mode=label_mode
    )

    validation_dataset = image_dataset_from_directory(
        val_dir,
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size,
        label_mode=label_mode
    )

    test_dataset = image_dataset_from_directory(
        test_dir,
        shuffle=False,
        batch_size=batch_size,
        image_size=img_size,
        label_mode=label_mode
    )

    train_dataset = prepare_dataset(train_dataset, apply_augmentation=False)
    validation_dataset = prepare_dataset(validation_dataset, apply_augmentation=False)
    test_dataset = prepare_dataset(test_dataset, apply_augmentation=False)

    return train_dataset, validation_dataset, test_dataset


'''
Main execution
'''
def main():
    train_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/train'
    test_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/test'
    val_path = '/home/rodrigocm/scratch/datasets/brset/selected_photos/validation'
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print("Importando imagens: ")
    train_dataset, val_dataset, test_dataset = import_images(train_path, 
                                                             test_path, 
                                                             val_path, 
                                                             img_size=(224, 224), 
                                                             batch_size=256, 
                                                             label_mode='binary')

    # load model and fine tune
    model = load_pre_trained_model()

    # get confusion matrix and classification report
    y_true, Y_pred, cm = confusion_matrix_and_report(model, 
                                                    test_dataset,
                                                    threshold=0.4)

    # test model
    results = test_model(model=model, test_dataset=test_dataset)
    print("Resultados do teste:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")
main()