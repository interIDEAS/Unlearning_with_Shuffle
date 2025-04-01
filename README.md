This repository includes the experimental setup, code, and results for the rebuttal of our paper \#13751, "**Efficient Feature Unlearning Using Shuffling: Algorithm and Theoretical Analysis.**". It offers additional details addressing three key aspects raised by the reviewers:

## Figure 1: : Advanced Model Architectures with the Most Important Features for Unlearning

| <img src="imgs/TRI_mlp.png" alt="TRI MLP Results" width="500px"> | <img src="imgs/TRI_ftformer.png" alt="TRI FtFormer Results" width="500px"> | <img src="imgs/TRI_resnet.png" alt="TRI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>TRI MLP Results for Top Two Important (identified by Shapley values) Features</em>                   |                <em>TRI FtFormer Results for Top Two Important(identified by Shapley values) Features</em>                 |                 <em>TRI ResNet Results for Top Two Important(identified by Shapley values) Features</em>                  |

| <img src="imgs/EI_mlp.png" alt="EI MLP Results" width="500px"> | <img src="imgs/EI_ftformer.png" alt="EI FtFormer Results" width="500px"> | <img src="imgs/EI_resnet.png" alt="EI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>EI MLP Results for Top Two Important (identified by Shapley values) Features</em>                    |                 <em>EI FtFormer Results for Top Two Important (identified by Shapley values) Features</em>                 |                  <em>EI ResNet Results for Top Two Important (identified by Shapley values) Features</em>                  |

| <img src="imgs/RASI_mlp.png" alt="RASI MLP Results" width="500px"> | <img src="imgs/RASI_ftformer.png" alt="RASI FtFormer Results" width="500px"> | <img src="imgs/RASI_resnet.png" alt="RASI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                  <em>RASI MLP Results for Top Two Important (identified by Shapley values) Features</em>                   |                <em>RASI FtFormer Results for Top Two Important (identified by Shapley values) Features</em>                |                 <em>RASI ResNet Results for Top Two Important (identified by Shapley values) Features</em>                 |

| <img src="imgs/SDI_mlp.png" alt="SDI MLP Results" width="500px"> | <img src="imgs/SDI_ftformer.png" alt="SDI FtFormer Results" width="500px"> | <img src="imgs/SDI_resnet.png" alt="SDI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>SDI MLP Results for Top Two Important (identified by Shapley values) Features</em>                   |                <em>SDI FtFormer Results for Top Two Important (identified by Shapley values) Features</em>                 |                 <em>SDI ResNet Results for Top Two Important (identified by Shapley values) Features</em>                  |

| <img src="imgs/TRI_mlp.png" alt="SRI MLP Results" width="500px"> | <img src="imgs/SRI_ftformer.png" alt="SRI FtFormer Results" width="500px"> | <img src="imgs/SRI_resnet.png" alt="SRI ResNet Results" width="500px"> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                   <em>SRI MLP Results for Top Two Important (identified by Shapley values) Features</em>                   |                <em>SRI FtFormer Results for Top Two Important (identified by Shapley values) Features</em>                 |                 <em>SRI ResNet Results for Top Two Important (identified by Shapley values) Features</em>                  |



## Figure 2: Advanced Model Architectures with the Highly Correlated  Features for Unlearning

![CALI Latitude Results](imgs/CALI_Latitude.png)
*(a) This graph illustrates the resulted evaluation metrics of unlearning the 'Latitude'(Correlation for``Latitude`` and ``Longitude`` is 0.925) feature using the MLP, ResNet, and FtFormer models.*
![CALI Longitude Results](imgs/CALI_Longtitude.png)
*(b) This graph illustrates the resulted evaluation metrics of unlearning the 'Longtitude' (Correlation for``Latitude`` and ``Longitude`` is 0.925) feature using the MLP, ResNet, and FtFormer models.*
![CALI AveBedrms Results](imgs/CALI_Avgbed.png)
*(c) This graph illustrates the resulted evaluation metrics of unlearning the 'AveBedrms' (Correlation for``AveRooms` and `AveBedrms`is 0.848)  feature using the MLP, ResNet, and FtFormer models.*
![CREDIT NumberOfTimes90DaysLate Results](imgs/CREDIT_90.png)
*(d) This graph illustrates the resulted evaluation metrics of unlearning the 'NumberOfTimes90DaysLate' (Correlation for - `NumberOfTimes90DaysLate` and `NumberOfTime60-89DaysPastDueNotWorse`is 0.991) feature using the MLP, ResNet, and FtFormer models.*
![CREDIT NumberOfTime30-59DaysPastDueNotWorse Results](imgs/CREDIT_3059.png)
*(e): This graph illustrates the resulted evaluation metrics of unlearning the 'NumberOfTime30-59DaysPastDueNotWorse' (Correlation for  `NumberOfTime30-59DaysPastDueNotWorse` and `NumberOfTime60-89DaysPastDueNotWorse` is 0.988) feature using the MLP, ResNet, and FtFormer models.*
![MAGIC_TELE fSize Results](imgs/TETL.png)
*(f) This graph illustrates the resulted evaluation metrics of unlearning the 'fSize'  (Correlation for `fSize`: and `fConc` is 0.847) feature using the MLP, ResNet, and FtFormer models.*




# Figure 3: Implementing Unlearning for Image Classification Tasks using the CelebA Dataset.
![](imgs/example.png)
![Big_Nose nose Results](imgs/cv/Big_Nose_nose_metrics.png)
*(a) This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Big_Nose classification.*

![Eyeglasses eye Results](imgs/cv/Eyeglasses_eye_metrics.png)
*(b) This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Eyeglasses classification."*

![Male eye Results](imgs/cv/Male_eye_metrics.png)
*(c) This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Gender classification."*

![Male nose Results](imgs/cv/Male_nose_metrics.png)
*(d) This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Gender classification."*

![Male noseeye Results](imgs/cv/Male_noseeye_metrics.png)
*(e) This graph illustrates the resulted evaluation metrics of unlearning ()the nose+eye feature for Gender classification"*

![Narrow_Eyes eye Results](imgs/cv/Narrow_Eyes_eye_metrics.png)
*(f) This graph illustrates the resulted evaluation metrics of unlearning the eye feature for Narrow_Eyes classification."*

![Pointy_Nose nose Results](imgs/cv/Pointy_Nose_nose_metrics.png)
*g This graph illustrates the resulted evaluation metrics of unlearning the nose feature for Pointy_Nose classification"*

