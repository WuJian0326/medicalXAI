# Medical xAI

## Methodology

In this study, we utilized the Automated Cardiac Diagnosis Challenge (ACDC) dataset [22], which consists of cine-MRI sequences of 100 patients, each with four labeled regions of interest (ROIs): the left ventricle (LV), right ventricle (RV), myocardium (Myo). The dataset was divided into a training set (70 patients), a validation set (10 patients), and a test set (20 patients). 

We evaluated two AI medical image segmentation models, the CNN-based model and the Transformer-based model. These models were trained on the ACDC dataset and their performance was assessed using mean Intersection over Union (mIoU) and accuracy (Acc) metrics.

Detailed comparison and methods can be found in the original paper.

![](https://hackmd.io/_uploads/B1YfCuOOn.png)

_Figure 1: Perturbations used to analyze the critical operations of the AI medical image segmentation models_

We also analyzed various types of artifacts, such as Gaussian noise, masks, and magnetic susceptibility artifacts, to understand their impact on the model's performance.

To improve the interpretability of these models, we integrated various xAI methods into a user-friendly interface, allowing users to visualize explanations and interact with the models.

![](https://hackmd.io/_uploads/ByY40OuO2.png)
_Figure 2: Evaluation of the Swin-Unet model on medical image segmentation_

![](https://hackmd.io/_uploads/r1eUCO__3.png)

_Figure 3: Interface for comparing the output of an AI medical image segmentation model to the original image under various perturbations_

## Result & Discussion

We observed that perturbation-based approaches improved the interpretability and clinical integration of the AI medical image segmentation models. We compared the performance of the two models under different adversarial conditions and evaluated their robustness.

Our results demonstrate that Swin-Unet outperforms Unet in terms of performance and robustness under adversarial attacks.

![](https://hackmd.io/_uploads/S1b_0_uu2.png)


![](https://hackmd.io/_uploads/S1BtCddd3.png)



The magnetic susceptibility artifacts, a common problem in MRI imaging, can significantly impact the accuracy of the models. Our results show that Swin-Unet is more robust than Unet in handling these artifacts.

![](https://hackmd.io/_uploads/SJXoAd_dn.png)


![](https://hackmd.io/_uploads/ryQ20ud_2.png)


In conclusion, Swin-Unet demonstrated superior performance and robustness under various perturbations, making it a promising model for medical image segmentation.
