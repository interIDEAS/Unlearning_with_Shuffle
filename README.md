This repository includes the experimental setup, code, and results for the rebuttal of our paper, "**Feature Unlearning: Theoretical Foundations and Practical Applications with Shuffling.**", aiming to provide more details of response in terms of

- *Simplicity of MLP model*: We have integrated more advanced neural architectures following the paper "**Revisiting Deep Learning Models for Tabular Data**" published in NeurIPS 2021, where the ResNet and the FT-Transformer were leveraged for tabular dataset.
- *Choices of unlearned feature and corresponding unlearning impacts*: Based on each feature's Shapley values, we select top two most important features and least two important features to unlearn. Additionally, we also examine our unlearning method's effectiveness and robustness when unlearning features that are highly correlated with other features.
- *Potential in complex tasks*: We have extended our unlearning method to an image classification task using **CelebA**, a large-scale dataset featuring over 200K images, from the paper "Deep Learning Face Attributes in the Wild" published in ICCV 2015. We further modify our model to use ViT as the backbone. 



# Details for Responses to *Simplicity of MLP model* and *Choices of unlearned feature*

## Advanced Model Architectures

In addition to MLP, we have incorporated **ResNet** and **FT-Transformer** for tabular unlearning task.

[ResNet reference] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90. 


[FT-Transformer reference] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. 2021. Revisiting deep learning models for tabular data. In Proceedings of the 35th International Conference on Neural Information Processing Systems (NIPS '21). Curran Associates Inc., Red Hook, NY, USA, Article 1447, 18932–18943.



## Unlearning Features with Different Feature Importance

In the submitted manuscript, the features selected for unlearning are those identified as having the highest feature importance. The feature importance is measured on the original model (before unlearning) through the Shapley value using the ``shap`` package.


To further demonstrate the effectiveness of our proposed method, we conducted additional experiments for each used tabular dataset which are focusing on:

- The **top two most influential features**, as determined by their Shapley values.
- The **least two influential features**, to assess the impact of unlearning less critical information.

Each of these features was unlearned across three different neural architectural frameworks:

1. MLP (as per the original experiments)
2. ResNet
3. FT-Transformer (referred to as FtFormer below)

The evaluation metrics remain the same as original experiments, including

1. Test Retention Index (TRI)
2. Efficiency Index (EI)
3. Robustness Against Shuffling Index (RASI)
4. SHAP Rentention Index (SRI)
5. SHAP Distance-To-Zero Index (SDI)

Experimental results are presented below.

### Results and Graphs

The following each graph presents the aggregated&averaged results across all datasets. For detailed information on each individual dataset, please refer to the ``results_processed`` folder. Also, codes to load these dataset results are provided in ``evaluation_rebuttal.ipynb``.

**Important: Our algorithm exhibits exceptional efficiency in unlearning features with the top two highest and lowest importance scores. This capability has been consistently validated across various architectural frameworks, including MLP, ResNet, and FtFormer.**




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

We have conducted experiments on features with high correlation coefficients to understand the effects of unlearning interdependent features. The threshold of identifying highly-correlated features is set to 0.8. The codes of calculating each dataset's feature correlation is provided in ``check_feature_correlation.ipynb``. These experiments were carried out using the same three architectural frameworks (MLP, FtFormer, ResNet) mentioned above. 

**Important: Our algorithm demonstrates remarkable efficiency in unlearning the features with high correlation, a capability that is consistently observed across multiple architectural frameworks, including MLP, ResNet, and FtFormer.**

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
*Description: This graph illustrates the impact of unlearning the 'Latitude' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CALI Longitude Results](imgs/CALI_Longtitude.png)
*Description: This graph illustrates the impact of unlearning the 'Longtitude' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CALI AveBedrms Results](imgs/CALI_Avgbed.png)
*Description: This graph illustrates the impact of unlearning the 'AveBedrms' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CREDIT NumberOfTimes90DaysLate Results](imgs/CREDIT_90.png)
*Description: This graph illustrates the impact of unlearning the 'NumberOfTimes90DaysLate' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![CREDIT NumberOfTime30-59DaysPastDueNotWorse Results](imgs/CREDIT_3059.png)
*Description: This graph illustrates the impact of unlearning the 'NumberOfTime30-59DaysPastDueNotWorse' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![MAGIC_TELE fSize Results](imgs/TETL.png)
*Description: This graph illustrates the impact of unlearning the 'fSize' feature using the MLP, ResNet, and FtFormer models. Our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*




 # Details for Responses to *Potential in complex tasks*

 ## Dataset Overview

TCelebFaces Attributes Dataset (CelebA) is an extensive collection of over 200K celebrity images, each annotated with 40 binary facial attributes. The dataset captures a wide range of pose variations and background complexities. It offers remarkable diversity and volume, featuring 10,177 unique identities, 202,599 face images, and detailed annotations that include 5 landmark locations along with 40 binary attribute labels per image.

## Experiment Description

In this experiment, we first train a model to classify gender/Big Nose/Pointy Nose/Eyeglasses/Narrow Eyes based on the celebrity images. Leveraging the dataset annotations, we then perform SEVEN distinct unlearning tasks:

- **Unlearn Noise (Classify Gender):**  
  Remove extraneous background details using the annotated noise locations.
- **Unlearn Eyes (Classify Gender):**  
  Exclude the eye regions based on the provided annotations.
- **Unlearn Noise+Eyes (Classify Gender):**  
  Simultaneously remove both the background noise and the eye regions.
- **Unlearn Nose (Classify Big Nose):**  
  Remove nose features corresponding to a big nose.
- **Unlearn Nose (Classify Pointy Nose):**  
  Remove nose features corresponding to a pointy nose.
- **Unlearn Eyes (Classify Eyeglasses):**  
  Remove eye features associated with eyeglasses.
- **Unlearn Eyes (Classify Narrow Eyes):**  
  Remove eye features associated with narrow eyes.

![Demostration of Image Unlearning](imgs/cv_unlearn_demo.png)
*Description: This graph demonstrates the unlearning process applied to the celebA dataset.

# How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git

```

