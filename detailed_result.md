This repository includes the experimental setup, code, and results for the rebuttal of our paper \#13751, "**Efficient Feature Unlearning Using Shuffling: Algorithm and Theoretical Analysis.**". It offers additional details addressing three key aspects raised by the reviewers:
- [*Model Complexity*](#details-for-responses-to-simplicity-of-mlp-model-and-choices-of-unlearned-feature): In response to Reviewers \#gLFb and \#F51p, we have integrated more advanced neural architectures for unlearning tabular datasets. We follow the paper "**Revisiting Deep Learning Models for Tabular Data**" levearging *ResNet* and the *FT-Transformer* and published in NeurIPS 2021, which have been shown to be effective for tabular data.
- [*Feature Selection and Unlearning Impact*](#details-for-responses-to-simplicity-of-mlp-model-and-choices-of-unlearned-feature): As suggested by Reviewers \#gLFb and \#F51p, we have analyzed the impact of unlearning by selecting both the two most important and the two least important features based on their Shapley value rankings. Additionally, per Reviewer \#gLFb's suggestion, we have evaluated the robustness of our unlearning method when removing features that are highly correlated with others.
- [*Applicability to Complex Tasks*](#details-for-responses-to-potential-in-complex-tasks): In response to Reviewer \#gLFb, we have extended our unlearning method to an image classification task using **CelebA**, a large-scale dataset featuring over 200K images, originally introduced in "Deep Learning Face Attributes in the Wild" (ICCV 2015).

The Table of Contents of this repository is shown below 
- [Details for Responses to *Simplicity of MLP model* and *Choices of unlearned feature*](#details-for-responses-to-simplicity-of-mlp-model-and-choices-of-unlearned-feature)
  - [Advanced Model Architectures](#advanced-model-architectures)
  - [Unlearning Features with Different Feature Importance](#unlearning-features-with-different-feature-importance)
  - [Unlearning Features that are Highly Correlated with Other Features](#unlearning-features-that-are-highly-correlated-with-other-features)
- [Details for Responses to *Potential in complex tasks*](#details-for-responses-to-potential-in-complex-tasks)
  - [Dataset Overview](#dataset-overview)
  - [Model Modification](#model-modification)
  - [Experiments](#experiments)



# Details for Responses to *Simplicity of MLP model* and *Choices of unlearned feature*

## Advanced Model Architectures

In addition to MLP, we have incorporated **ResNet** and **FT-Transformer** for tabular unlearning task.

[ResNet reference] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90. 


[FT-Transformer reference] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. 2021. Revisiting deep learning models for tabular data. In Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS '21). Curran Associates Inc., Red Hook, NY, USA, Article 1447, 18932–18943.



## Unlearning Features with Different Feature Importance

In the single-feature unlearning of the original manuscript, we selected the most important features for unlearning based on their Shapley values, computed using the ``shap`` package on the original model (prior to unlearning) to determine feature importance.


To further validate the effectiveness of our proposed method, we conducted additional experiments on each tabular dataset, focusing on:

- The **two most influential features**, as identified by their Shapley values.
- The **two least influential features**, to assess the impact of unlearning less critical information.

Each selected feature was unlearned across three different neural architectural frameworks:

1. MLP (as used in the original experiments)
2. ResNet
3. FT-Transformer (referred to as FtFormer below)

The evaluation metrics remain consistent with the original experiments, including:

1. Test Retention Index (TRI)
2. Efficiency Index (EI)
3. Robustness Against Shuffling Index (RASI)
4. SHAP Rentention Index (SRI)
5. SHAP Distance-To-Zero Index (SDI)

Experimental results are presented below.


### Results and Graphs

Each graph below presents the aggregated and averaged results across all datasets. For detailed results on each dataset, please refere to the ``results`` folder. Also, code for loading and analyzing these results is provided in ``evaluation_rebuttal.ipynb``.


**Key Findings: Our algorithm demonstrates exceptional efficiency in unlearning both the most and least influential features. This effectiveness is consistently observed across different architectural frameworks, including MLP, ResNet, and FtFormer.**




| <img src="imgs/TRI_mlp.png" alt="TRI MLP Results" width="500px"> | <img src="imgs/TRI_ftformer.png" alt="TRI FtFormer Results" width="500px"> | <img src="imgs/TRI_resnet.png" alt="TRI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>TRI MLP Results</em>                   |                <em>TRI FtFormer Results</em>                 |                 <em>TRI ResNet Results</em>                  |

| <img src="imgs/EI_mlp.png" alt="EI MLP Results" width="500px"> | <img src="imgs/EI_ftformer.png" alt="EI FtFormer Results" width="500px"> | <img src="imgs/EI_resnet.png" alt="EI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>EI MLP Results</em>                    |                 <em>EI FtFormer Results</em>                 |                  <em>EI ResNet Results</em>                  |

| <img src="imgs/RASI_mlp.png" alt="RASI MLP Results" width="500px"> | <img src="imgs/RASI_ftformer.png" alt="RASI FtFormer Results" width="500px"> | <img src="imgs/RASI_resnet.png" alt="RASI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                  <em>RASI MLP Results</em>                   |                <em>RASI FtFormer Results</em>                |                 <em>RASI ResNet Results</em>                 |

| <img src="imgs/SDI_mlp.png" alt="SDI MLP Results" width="500px"> | <img src="imgs/SDI_ftformer.png" alt="SDI FtFormer Results" width="500px"> | <img src="imgs/SDI_resnet.png" alt="SDI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>SDI MLP Results</em>                   |                <em>SDI FtFormer Results</em>                 |                 <em>SDI ResNet Results</em>                  |

| <img src="imgs/TRI_mlp.png" alt="SRI MLP Results" width="500px"> | <img src="imgs/SRI_ftformer.png" alt="SRI FtFormer Results" width="500px"> | <img src="imgs/SRI_resnet.png" alt="SRI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>SRI MLP Results</em>                   |                <em>SRI FtFormer Results</em>                 |                 <em>SRI ResNet Results</em>                  |




## Unlearning Features that are Highly Correlated with Other Features

We conducted experiments on highly correlated features to analyze the effects of unlearning interdependent features. Features with a correlation coefficient above 0.8 were selected. The code for computing feature correlations for each dataset is available in ``check_feature_correlation.ipynb``. These experiments were carried out using the same three architectural frameworks (MLP, FtFormer, ResNet) mentioned above. 

**Key Finding: Our algorithm exhibits remarkable efficiency in unlearning highly correlated features, a capability consistently demonstrated across MLP, ResNet, and FtFormer**

Highly correlated features in the *CALI* dataset:

- ``Latitude`` and ``Longitude``: 0.925
- `AveRooms` and `AveBedrms`: 0.848

Features Unlearned: **Latitude, Longitude, AveBedrms**

Highly correlated features in *CREDIT* dataset:

- `NumberOfTimes90DaysLate` and `NumberOfTime60-89DaysPastDueNotWorse`: 0.991
- `NumberOfTime30-59DaysPastDueNotWorse` and `NumberOfTime60-89DaysPastDueNotWorse`: 0.988
- `NumberOfTime30-59DaysPastDueNotWorse` and `NumberOfTimes90DaysLate`: 0.983

Features Unlearned: **NumberOfTimes90DaysLate, NumberOfTime30-59DaysPastDueNotWorse**


Highly correlated features in *MAGIC_TELE* dataset:

- `fConc`: and `fConc1`:: 0.975
- `fSize`: and `fConc`:: 0.847
- `fSize`: and `fConc1`:: 0.804

Features Unlearned: **fSize**

Experimental results are shown below.

### Results and Graphs

![CALI Latitude Results](imgs/CALI_Latitude.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'Latitude' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CALI Longitude Results](imgs/CALI_Longtitude.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'Longtitude' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CALI AveBedrms Results](imgs/CALI_Avgbed.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'AveBedrms' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CREDIT NumberOfTimes90DaysLate Results](imgs/CREDIT_90.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'NumberOfTimes90DaysLate' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![CREDIT NumberOfTime30-59DaysPastDueNotWorse Results](imgs/CREDIT_3059.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'NumberOfTime30-59DaysPastDueNotWorse' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![MAGIC_TELE fSize Results](imgs/TETL.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the 'fSize' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*




# Details for Responses to *Potential in complex tasks*

## Dataset Overview

CelebFaces Attributes Dataset (CelebA) is an extensive collection of over 200K celebrity images, each annotated with 40 binary facial attributes. The dataset captures a wide range of pose variations and background complexities. It offers remarkable diversity and volume, featuring 10,177 unique identities, 202,599 face images, and detailed annotations that include 5 landmark locations along with 40 binary attribute labels per image.


[CelebA reference] Z. Liu, P. Luo, X. Wang and X. Tang, "Deep Learning Face Attributes in the Wild," in 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, Chile, 2015, pp. 3730-3738, doi: 10.1109/ICCV.2015.425.

## Model Modification

For the image classification task, we leverage the ViT as backbone and MLP as classifier head.

[ViT reference] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby:
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021

## Experiments

In tabular datasets, each column represents a distinct feature, making feature selection and manipulation explicit. In contrast, for computer vision tasks, features are visually embedded within images and not explicitly structured as separate variables. For instance, in the CelebA dataset, facial attributes such as the nose, eyes, and gender are inherently captured within each image rather than existing as distinct columns.

To extend our unlearning method to the image classification setting, we focus on selectively unlearning specific visual features of the CelebA dataset, including:
- Nose
- Eye
- Nose and Eye

The classification tasks involve predicting the following labels:
- Gender (Male or Female)
- Big Nose
- Pointy Nose
- Eyeglasses
- Narrow Eyes

The detailed descriptions of the performed unlearning tasks are provided below

- **Unlearn Nose (Label Gender):**  
  Remove extraneous background details using the annotated nose locations.
- **Unlearn Eyes (Label Gender):**  
  Exclude the eye regions based on the provided annotations.
- **Unlearn Noise+Eyes (Label Gender):**  
  Simultaneously remove both the background noise and the eye regions.
- **Unlearn Nose (Label Big Nose):**  
  Remove nose features corresponding to a big nose.
- **Unlearn Nose (Label Pointy Nose):**  
  Remove nose features corresponding to a pointy nose.
- **Unlearn Eyes (Label Eyeglasses):**  
  Remove eye features associated with eyeglasses.
- **Unlearn Eyes (Label Narrow Eyes):**  
  Remove eye features associated with narrow eyes.


The following example illustrates how an original image is processed for our unlearning method and retrain-from-scratch model:

1. Shuffled Nose Feature: The nose region is shuffled while preserving the overall structure of the image.
2. Masked Nose Feature: The nose region is masked to remove its influence on the model.



![](imgs/example.png)


### Results and Graphs

Experimental results of the above seven image classification tasks are shown below. For detailed information on each task, please refer to the ``results_cv`` folder. Also, codes to load these dataset results are provided in ``evaluation_rebuttal_cv.ipynb``.

**Key Findings: Our algorithm demonstrates exceptional efficiency in unlearning various visual features in image classification tasks, outperforming both baseline methods..**


![Big_Nose nose Results](imgs/cv/Big_Nose_nose_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Big_Nose classification.*


![Eyeglasses eye Results](imgs/cv/Eyeglasses_eye_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Eyeglasses classification."*

![Male eye Results](imgs/cv/Male_eye_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Gender classification."*

![Male nose Results](imgs/cv/Male_nose_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Gender classification."*

![Male noseeye Results](imgs/cv/Male_noseeye_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the nose+eye feature for Gender classification"*

![Narrow_Eyes eye Results](imgs/cv/Narrow_Eyes_eye_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Narrow_Eyes classification."*

![Pointy_Nose nose Results](imgs/cv/Pointy_Nose_nose_metrics.png)
*Description: This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Pointy_Nose classification"*


<!-- # How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git

```
 -->
