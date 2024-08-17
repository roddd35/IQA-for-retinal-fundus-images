# IQA for Retinal Fundus Images

### Rodrigo de Castro Michelassi
### Instituto de Matemática e Estatística da Universidade de São Paulo (IME-USP)

## General

<p align="justify">Research project being developed for the University of São Paulo, in Brasil, aiming to classify correctly the quality label of an eye-fundus images using Convolutional Neural Networks. This project was based on the brazilian dataset BRSet, put together by the Federal University of São Paulo, putting Brazil on the spot for Ophthalmology research; and the EyeQ dataset, a famous public dataset for IQA in eye pictures.</p>

<p align="justify">This project was first proposed at University of São Paulo Medical School (FMUSP), in which the final goal is to detect possible causes for Cognitive Decline, and Diabetes is one of the research scopes. On top of that, Computer Science students at IME-USP are being responsible for developing a deep learning algorithm capable of recognizing eye diseases, including Diabetic Retinopathy, really common on people with diabetes, and this classification should be held into account on the Cognitive Decline research.</p>
  
 <p align="justify">In order to optimize the results on the eye diseases classification problem and guarantee that no meaningful information to detect an eye disease is missing on the image, causing no harm to the feature extraction process, it is really important to assess the quality of the images used.</p>

## Labeling process

<p align="justify">The quality classification on BRSet is made based on the following metrics:</p>

> <b>Focus:</b> This parameter is graded as adequate when the focus is sufficient to identify third-generation branches within one optic disc diameter around the macula.
>
> <b>Illumination:</b> This parameter is graded as adequate when both of the following requirements are met:
> 
> 1) Absence of dark, bright, or washed-out areas that interfere with detailed grading;
>
> 2) In the case of peripheral shadows (e.g., due to pupillary constriction) the readable part should reach more than 80% of the whole image.
>
> <b>Image Field:</b> This parameter is graded as adequate when all the following requirements are met:
>
> 1) The optic disc is at least 1 disc diameter (DD) from the nasal edge;
> 
> 2) The macular center is at least 2 DD from the temporal edge;
> 
> 3) The superior and inferior temporal arcades are visible in a length of at least 2 DD
>
> <b>Artifacts:</b> The following artifacts are considered: haze, dust, and dirt. This parameter is graded as adequate when the image is sufficiently artifact-free to allow adequate grading.

<p align="justify">and we aim to classify the images between <b>Inadequate</b> (when the image present any of the above metrics) or <b>Adequate</b> (the image does not present any of the above metrics, and is good enough for abnormalities recognition).</p>

## Images Preview
<p align="center"><img src="img_analysis/eyeQuality.png" width="70%"/></p>
<p align="center">Examples of impaired/ungradable images. (A) Poor focus and clarity due to overall haze. (B) Poor macula visibility due to uneven illumination. (C) Poor optic disc visibility due to total blink. (D) Edge haze due to pupillary restriction. (E) Dust and dirt artifacts on the lens image capture system (near the center). (F) Lash artifact.</p>

## Pre-processing

<p align="justfify"></p>

## Neural Network

## Results

## Files description

## References
[1] Wang, Z., Bovik, A. C., and Lu, L. (2002). Why is image quality assessment so
difficult? In 2002 IEEE International Conference on Acoustics, Speech, and Signal
Processing, volume 4, pages IV–3313–IV–3316.

[2] Athar, S. and Wang, Z. (2019). A comprehensive performance evaluation of image
quality assessment algorithms. IEEE Access, 7:140030–140070.

[3] Bosse, S., Maniry, D., Wiegand, T., and Samek, W. (2016). A deep neural network for
image quality assessment. In 2016 IEEE International Conference on Image Processing
(ICIP), pages 3773–3777.

[4] Yang, J., Lyu, M., Qi, Z., and Shi, Y. (2023). Deep learning based image quality
assessment: A survey. Procedia Computer Science, 221:1000–1005. Tenth International
Conference on Information Technology and Quantitative Management (ITQM 2023).

[5] Mariana Batista Gonçalves, Luis Filipe Nakayama, Daniel Ferraz, Hanna Faber,
Edward Korot, Fernando Korn Malerbi, Caio Vinic ius Regatieri, Mauricio Maia,
Leo Anthony Celi, Pearse A. Keane, and Rub ens Belfort Jr. Image quality assessment
of retinal fundus photographs for diabetic retinopathy in the machine learning era:
a review. Eye, 2023.

[6] L. F. Nakayama, M. Goncalves, L. Zago Ribeiro, H. San tos, D. Ferraz, F. Malerbi,
L. A. Celi,and C. Regatieri. A brazilian multilabel ophthalmological dataset (BRSET)

[7] Huazhu Fu, Boyang Wang, Jianbing Shen, Shanshan Cui, Yanwu Xu, Jiang Liu, Ling Shao. 
Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces, in MICCAI, 2019.

[8] Bolla, M., Biswas, S. and Palanisamy, R. (2023) Deep Learning Based Quality Prediction of Retinal Fundus Images. Current Directions in Biomedical Engineering, Vol. 9 (Issue 1), pp. 706-709.
