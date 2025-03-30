This repository contains the experimental setup, code, and results for our paper on "Feature Unlearning: Theoretical Foundations and Practical Applications with Shuffling". In response to reviewer comments regarding the simplicity of our initial models, we have incorporated more complex models, namely FtFormer and ResNet, to demonstrate the robustness and scalability of our unlearning approach.

## More Complex Models:
We have used three different models to demonstrate the efficacy and robustness of our feature unlearning approach:

- **MLP (Multi-Layer Perceptron)**: A classical neural network model consisting of multiple layers of perceptrons. It is widely used for tabular data and serves as our baseline model.
- **FtFormer**: A transformer-based model designed to handle sequential data.
- **ResNet**: A deep residual network known for its effectiveness in image processing tasks.

## Unlearning based on the Feature Importance

Feature unlearning was conducted on multiple datasets. For each dataset, we identified and unlearned:
- The **top two most influential features**, as determined by their Shapley values.
- The **least two influential features**, to assess the impact of unlearning less critical information.

Each of these features was unlearned across three different architectural frameworks:
1. Baseline Model (as per the original experiments)
2. FtFormer
3. ResNet

### Results and Graphs
![MLP Results](imgs/different_feature_importance/TRI_mlp.pdf)
*Description: Graph showing the impact of unlearning the top and least influential features using the MLP model.*
![FtFormer Results](imgs/different_feature_importance/TRI_ftformer.pdf)
*Description: Graph depicting the outcomes of feature unlearning on the FtFormer model, focusing on interdependent feature sets.*
![ResNet Results](imgs/different_feature_importance/TRI_resnet.pdf)
*Description: Visual outcomes of unlearning using the ResNet model across various feature sets.*



## Unlearning based on the Feature Correaltion

We also conducted experiments on features with high correlation coefficients to understand the effects of unlearning interdependent features. These experiments were carried out using the same three architectural frameworks mentioned above. 

Highly correlated features in **CALI** (threshold = 0.8):
- Latitude and Longitude: 0.925
- AveRooms and AveBedrms: 0.848
Features Unlearned: **Latitude,Longitude,AveBedrms**

Highly correlated features in **CREDIT** (threshold = 0.8):
- NumberOfTimes90DaysLate and NumberOfTime60-89DaysPastDueNotWorse: 0.991
- NumberOfTime30-59DaysPastDueNotWorse and NumberOfTime60-89DaysPastDueNotWorse: 0.988
- NumberOfTime30-59DaysPastDueNotWorse and NumberOfTimes90DaysLate: 0.983
Features Unlearned: **NumberOfTimes90DaysLate,NumberOfTime30-59DaysPastDueNotWorse**


Highly correlated features in MAGIC_TELE (threshold = 0.8):
- fConc: and fConc1:: 0.975
- fSize: and fConc:: 0.847
- fSize: and fConc1:: 0.804
Features Unlearned: **fSize**

Other datasets do not include features that exceed our set threshold.
 
### Results and Graphs
![CALI Latitude Results](imgs/CALI_Latitude.png)
*Description: Graph showing the impact of unlearning the top and least influential features using the MLP model.*
![CALI Longitude Results](imgs/CALI_Longtitude.png)
*Description: Graph depicting the outcomes of feature unlearning on the FtFormer model, focusing on interdependent feature sets.*
![CALI AveBedrms Results](imgs/CALI_Avgbed.png)
*Description: Visual outcomes of unlearning using the ResNet model across various feature sets.*



## How to Run Experiments

To replicate our experiments or to run new experiments using the setup provided, follow the instructions below:

```bash
# Clone the repository
git clone https://github.com/your-repository-url.git
