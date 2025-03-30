This repository includes the experimental setup, code, and results for our paper, "Feature Unlearning: Theoretical Foundations and Practical Applications with Shuffling." In response to reviewer feedback regarding the simplicity of our initial models, we have integrated more advanced architectures—specifically, FtFormer and ResNet—to demonstrate the robustness and scalability of our unlearning approach. Additionally, we have extended our method to address more complex tasks using an image dataset, CelebA, a large-scale dataset featuring over 200K images.
 # Feature Unlearning with CelebA: Removing a Feature from a Computer Vision Task
 ## Dataset Overvie

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

# Feature Unlearning: Leveraging Complex Model Architectures, Higher Feature Importance, and Correlation Analysis
 ## Expanded Model Architectures (Incorporating FtFormer and ResNet):
We have used three different models to demonstrate the efficacy and robustness of our feature unlearning approach:

- **MLP (Original)**: A classical neural network model consisting of multiple layers of perceptrons. It is widely used for tabular data and serves as our baseline model.
- **FtFormer**: A transformer-based model designed to handle sequential data.
- **ResNet**: A deep residual network known for its effectiveness in image processing tasks.

## Unlearning based on the Feature Importance
In the submitted manuscript, the features selected for unlearning are those identified as having the highest importance. To further demonstrate the effectiveness of our proposed method, we conducted additional experiments for each dataset which are focusing on:
- The **top two most influential features**, as determined by their Shapley values.
- The **least two influential features**, to assess the impact of unlearning less critical information.

Each of these features was unlearned across three different architectural frameworks:
1. MLP (as per the original experiments)
2. FtFormer
3. ResNet

### Results and Graphs

The following graph presents the aggregated results from all datasets. For detailed information on each individual dataset, please refer to the *results_processed* folder.

**Important: Our algorithm exhibits exceptional efficiency in unlearning features with the top two highest and lowest importance scores. This capability has been consistently validated across various architectural frameworks, including MLP, ResNet, and FtFormer.**






| <img src="imgs/TRI_mlp.png" alt="TRI MLP Results" width="500px"> | <img src="imgs/TRI_ftformer.png" alt="TRI FtFormer Results" width="500px"> | <img src="imgs/TRI_resnet.png" alt="TRI ResNet Results" width="500px"> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| <em>TRI MLP Results</em>                                            | <em>TRI FtFormer Results</em>                                        | <em>TRI ResNet Results</em>                                          |

| <img src="imgs/EI_mlp.png" alt="EI MLP Results" width="500px"> | <img src="imgs/EI_ftformer.png" alt="EI FtFormer Results" width="500px"> | <img src="imgs/EI_resnet.png" alt="EI ResNet Results" width="500px"> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| <em>EI MLP Results</em>                                            | <em>EI FtFormer Results</em>                                        | <em>EI ResNet Results</em>                                          |

| <img src="imgs/RASI_mlp.png" alt="RASI MLP Results" width="500px"> | <img src="imgs/RASI_ftformer.png" alt="RASI FtFormer Results" width="500px"> | <img src="imgs/RASI_resnet.png" alt="RASI ResNet Results" width="500px"> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| <em>RASI MLP Results</em>                                            | <em>RASI FtFormer Results</em>                                        | <em>RASI ResNet Results</em>                                          |

| <img src="imgs/SDI_mlp.png" alt="SDI MLP Results" width="500px"> | <img src="imgs/SDI_ftformer.png" alt="SDI FtFormer Results" width="500px"> | <img src="imgs/SDI_resnet.png" alt="SDI ResNet Results" width="500px"> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| <em>SDI MLP Results</em>                                            | <em>SDI FtFormer Results</em>                                        | <em>SDI ResNet Results</em>                                          |

| <img src="imgs/TRI_mlp.png" alt="SRI MLP Results" width="500px"> | <img src="imgs/SRI_ftformer.png" alt="SRI FtFormer Results" width="500px"> | <img src="imgs/SRI_resnet.png" alt="SRI ResNet Results" width="500px"> |
|:---------------------------------------------------------------:|:---------------------------------------------------------------:|:---------------------------------------------------------------:|
| <em>SRI MLP Results</em>                                            | <em>SRI FtFormer Results</em>                                        | <em>SRI ResNet Results</em>                                          |




## Unlearning based on the Feature Correaltion

We also conducted experiments on features with high correlation coefficients to understand the effects of unlearning interdependent features. These experiments were carried out using the same three architectural frameworks (MLP, FtFormer, ResNet)mentioned above. 

**Important: Our algorithm demonstrates remarkable efficiency in unlearning the features with high correlation, a capability that is consistently observed across multiple architectural frameworks, including MLP, ResNet, and FtFormer.**

Highly correlated features in *CALI* (threshold = 0.8):
- Latitude and Longitude: 0.925
- AveRooms and AveBedrms: 0.848
  
Features Unlearned: **Latitude, Longitude, AveBedrms**

Highly correlated features in *CREDIT* (threshold = 0.8):
- NumberOfTimes90DaysLate and NumberOfTime60-89DaysPastDueNotWorse: 0.991
- NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse: 0.988
- NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate: 0.983
  
Features Unlearned: **NumberOfTimes90DaysLate, NumberOfTime30-59DaysPastDueNotWorse**


Highly correlated features in *MAGIC_TELE* (threshold = 0.8):
- fConc: and fConc1:: 0.975
- fSize: and fConc:: 0.847
- fSize: and fConc1:: 0.804
  
Features Unlearned: **fSize**

Other datasets do not include features that exceed our set threshold (The code for checking feature correlation is available in: check_feature_correlation.ipynb).
 
### Results and Graphs
![CALI Latitude Results](imgs/CALI_Latitude.png)
*Description: This graph illustrates the impact of unlearning the 'Latitude' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."*
![CALI Longitude Results](imgs/CALI_Longtitude.png)
*Description: This graph illustrates the impact of unlearning the 'Longtitude' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![CALI AveBedrms Results](imgs/CALI_Avgbed.png)
*Description: This graph illustrates the impact of unlearning the 'AveBedrms' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![CREDIT NumberOfTimes90DaysLate Results](imgs/CREDIT_90.png)
*Description: This graph illustrates the impact of unlearning the 'NumberOfTimes90DaysLate' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![CREDIT NumberOfTime30-59DaysPastDueNotWorse Results](imgs/CREDIT_3059.png)
*Description: This graph illustrates the impact of unlearning the 'NumberOfTime30-59DaysPastDueNotWorse' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*
![MAGIC_TELE fSize Results](imgs/TETL.png)
*Description: This graph illustrates the impact of unlearning the 'fSize' feature using the MLP, ResNet, and FtFormer models. The red line represents the baseline, and our model's performance is consistent with observations reported in the main paper. It demonstrates robust performance across all evaluation criteria."s.*

## How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git
