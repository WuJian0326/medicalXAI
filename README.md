# Medical xAI

## Methodology

In this study, we utilized the Automated Cardiac Diagnosis Challenge (ACDC) dataset [22], which consists of cine-MRI sequences of 100 patients, each with four labeled regions of interest (ROIs): the left ventricle (LV), right ventricle (RV), myocardium (Myo). The dataset was divided into a training set (70 patients), a validation set (10 patients), and a test set (20 patients). 

We evaluated two AI medical image segmentation models, the CNN-based model and the Transformer-based model. These models were trained on the ACDC dataset and their performance was assessed using mean Intersection over Union (mIoU) and accuracy (Acc) metrics.

Detailed comparison and methods can be found in the original paper.

<img width="710" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/fb5c0723-5681-4cdd-bd63-1f90e7c247d2">



We also analyzed various types of artifacts, such as Gaussian noise, masks, and magnetic susceptibility artifacts, to understand their impact on the model's performance.

To improve the interpretability of these models, we integrated various xAI methods into a user-friendly interface, allowing users to visualize explanations and interact with the models.

<img width="710" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/802c5c79-9eb2-4fe9-999c-f3cb6e239897">


<img width="761" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/6cd58571-4e16-4290-afde-ab29e00d3997">


_Figure 3: Interface for comparing the output of an AI medical image segmentation model to the original image under various perturbations_

## Result & Discussion

We observed that perturbation-based approaches improved the interpretability and clinical integration of the AI medical image segmentation models. We compared the performance of the two models under different adversarial conditions and evaluated their robustness.

Our results demonstrate that Swin-Unet outperforms Unet in terms of performance and robustness under adversarial attacks.

<img width="483" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/6bca643a-99fe-402c-9d5d-d5b5d2e6ddeb">



<img width="483" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/8752b3ae-2541-40c8-9f72-7bc570139a62">



The magnetic susceptibility artifacts, a common problem in MRI imaging, can significantly impact the accuracy of the models. Our results show that Swin-Unet is more robust than Unet in handling these artifacts.

<img width="705" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/bb05bfb0-7964-47ad-b30f-1ce065e47dea">


<img width="634" alt="image" src="https://github.com/WuJian0326/medicalXAI/assets/92918078/b3a23d07-1c15-43dc-9ae7-d000128f8ccc">



In conclusion, Swin-Unet demonstrated superior performance and robustness under various perturbations, making it a promising model for medical image segmentation.
