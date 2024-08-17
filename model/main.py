import sys

sys.path.append('/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/src/data')
sys.path.append('/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/src/models')
sys.path.append('/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/src/visualization')

import itertools
import tensorflow as tf 
from data_import_lib import import_images, prepare_dataset, normalize, augment
from plot_lib import call_epochs_plot, call_plot_confusion_matrix, plot_pr_curve, plot_roc_curve
from evaluate_model import test_model, confusion_matrix_and_report
from train_model import load_pre_trained_model, fine_tune

'''
Execution algorithm for CNN Training on IQA on BRSet
    1. You can load a pre-trained CNN, fine-tune and evaluate
       or simply just train a new CNN for this problem.
    2. Test dataset follows 87:13 proportion, train and validation
       are balanced.
    3. Adjust parameters as needed
    4. [+] Second compile for the model
    5. [+] Finetune again with Resnet+EyeQ pre-trained model
'''

# FUNCTION: main
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
                                                             batch_size=16, 
                                                             label_mode='categorical')

    # load model and fine tune
    fine_tune_model = load_pre_trained_model()
    history, model = fine_tune(fine_tune_model=fine_tune_model, 
                               train_dataset=train_dataset, 
                               val_dataset=val_dataset, 
                               fcEpochs=10,
                               cnnEpochs=20)

    # plot training metrics
    call_epochs_plot(history=history, save_path="/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/images/fine_tune_graphs/resnet/")

    # plot result metrics
    y_true, Y_pred, cm = confusion_matrix_and_report(model, 
                                                    test_dataset,
                                                    threshold=0.2)

    lixo1, lixo2, lixo3 = confusion_matrix_and_report(model, 
                                                    val_dataset,
                                                    threshold=0.5)

    call_plot_confusion_matrix(y_true, Y_pred, save_path="/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/images/fine_tune_graphs/resnet/confusion_matrix", cm=cm)
    plot_roc_curve(y_true, Y_pred, save_path="/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/images/fine_tune_graphs/resnet/roc_curve")
    plot_pr_curve(y_true, Y_pred, save_path="/home/rodrigocm/IQA-Retinal-Project/algorithms/modelv4/data/images/fine_tune_graphs/resnet/pr_curve")

    # test model
    results = test_model(model=model, test_dataset=test_dataset)
    print("Resultados do teste:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value}")
main()