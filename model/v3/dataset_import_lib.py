import itertools
import tensorflow as tf 
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory 

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

    train_dataset = prepare_dataset(train_dataset, apply_augmentation=True)
    validation_dataset = prepare_dataset(validation_dataset, apply_augmentation=False)
    test_dataset = prepare_dataset(test_dataset, apply_augmentation=False)

    return train_dataset, validation_dataset, test_dataset