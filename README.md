# 👨‍👨‍👦‍👦🏻  Prediction of Intraoperative Hypotension Using Deep Learning Models Based on Non-invasive Monitoring Devices

 
### Abstract <br/>

 - _Intraoperative hypotension is associated with adverse outcomes. Predicting and proactively managing hypotension can reduce its incidence. Previously, hypotension prediction algorithms using artificial intelligence were developed for invasive arterial blood pressure monitors. This study tested whether routine non-invasive monitors could also predict intraoperative hypotension using deep learning algorithms._ <br/><br/>

### Dataset <br/>

- _An open-source database was used for algorithm development and internal validation (http://vitaldb.net/dataset). The open-source database consisted of 6388 patients who underwent non-cardiac surgery between June 2016 and August 2017 at Seoul National University Hospital,  South Korea, a tertiary referral hospital._ <br/>

- _Eligible patients selected from the open-source database were randomly divided into a training set of 80% and a test set of 20%. The training set was used to build the predictive model, and the test set was used to evaluate the final performance of the model. Patient data from the test set were not exposed during model training. A validation set (10% of the training set) was randomly selected from the training set to tune the hyperparameters of the trained model._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/8d7cd514-4ad4-45de-9362-bd54c36b83af)

- _Patient characteristics and dataset compositions between hypotensive and non-hypotensive cases are summarized in Supplementary Table S1. Surgeries included general, gynecological, thoracic, and genitourinary surgeries. The mean ± SD age was 57 ± 14 years, and 49% of the participants in the training set were men. The comparison variables were well-balanced between hypotensive and nonhypotensive cases._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/976e5e34-6325-47dd-9038-272d71f9cf4f)

### Data Preparation <br/>

- _All monitoring data were sampled from the source database at a rate of 100 Hz. Waveform data, such as electrocardiography, photoplethysmography, and capnography, were recorded continuously._ <br/>

- _However, non-invasive blood pressure data consisted of 1-point values with a mean measurement interval of 2.5 minutes and the bispectral index was recorded every second. Therefore, blood pressure and bispectral index data were rearranged at 100 Hz intervals by filling empty places with the preceding values to facilitate a seamless end-to-end training process._ <br/>

- _Our classification tasks involved binary classification of hypotension events (systolic blood pressure < 90 mmHg) or non-hypotension events (systolic blood pressure >= 90 mmHg) occurring after 5 min. These input signals were extracted as 100 Hz input data in units of 30s, sequentially shifted by 10 s, and then extracted repeatedly._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/bae90252-5929-4ff0-a82c-a7a7f21bf10d)

- _In the training dataset, the ratio of intraoperative hypotension to non-hypotension cases was highly unbalanced at 1:9. Therefore, a Focal Loss function with a weighting mechanism was used to address the class imbalance problem._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/acb0c526-3404-4e03-a206-e496a2b93157)

- _A Focal Loss function was applied during the training process to enable the model to learn intensively from rare hypotensive data. By dynamically adjusting the weighting of the training data, the Focal Loss function reoriented the learning process toward the minority class, thereby mitigating the detrimental effects of skewed data formulation._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/a8904154-c6ab-41f2-858b-fc21e23dcf07)

- _Normalization was performed to improve the efficiency of the deep learning models by training them on patterns. Scaling was performed using minimum-maximum normalization to address potential differences in the distributions of the training and test data._ <br/>
  
![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/bb782037-74d7-4093-b2ca-cbd04fc293a3)

### Model Development  <br/>

- _The deep learning model was built using the Multi-head Attention model for feature-wise information and the Globally Attentive Locally Recurrent (GALR) model for time-wise information._ <br/>

- _The Multi-head Attention model made predictions by considering the importance ratio of the attributes in the input data. The GALR model evaluated the importance of time in input data. Finally, a fully connected model was developed by combining the Multi-head Attention and GALR models._ <br/>

- _This approach enabled the model to leverage the strengths of both the feature and time axes, thereby enhancing its ability to effectively predict hypotension events._ <br/><br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/5cfa0405-3c45-4969-be8c-6734ea259358)

### Metrix <br/>

- _The primary outcome was the area under the receiver operating characteristic curve (AUROC) for the classification models._ <br/>
![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/692bf8d2-c697-4b2b-874e-5c7089dc1ab5)

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/f977f06f-2e4c-4eaa-98df-6ed5e6591bab)

- _The secondary outcomes included the accuracy, sensitivity, and specificity of the established model._ <br/>
  
![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/9a96485a-922a-49c9-aa8b-27b7d6b2ecec)
![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/a49e4c83-6879-4d45-8d43-9250b0d48c1c)

- _The study outcomes were presented with 95% confidence intervals (CIs) estimated using the bootstrapping method. Descriptive statistics were used to describe patient characteristics and were expressed as means ± standard deviation (SD) or absolute numbers (proportion), as appropriate. The chi-square test for categorical variables and t-test for continuous variables were performed for comparative analysis._ <br/><br/>

### Result <br/>

- _The fully connected model, which combines the Multi-head Attention and the GALR model with a Focal Loss function, achieved the highest AUROC of 0.917 (95% CI, 0.915–0.918) for the test set of the original data and 0.833 (95% CI, 0.830–0.836) for the external validation dataset. The secondary outcomes, including accuracy, sensitivity, and specificity, are presented in Table 1._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/b0a3e2a4-4a1b-4d92-9615-a8cff4255585)

- _We introduced attention mechanisms into our architecture. Figure 4 illustrates the attention map analysis employed to visualize the features and time domains utilized for decision-making in hypotension events. According to the attention map, our algorithm utilized data from each monitor with weights ranging from 5% to 22% for determining hypotension._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/e410d4d0-234a-411e-978a-458e6460e9df)

- _Among them, the bispectral index had the highest weight (20%) in predicting hypotension, and mean  blood pressure had the highest weight (22%) in predicting non-hypotension._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/eabca6c0-db7b-4047-b75b-cec407595fcf)

- _Figure 3 is a monitor equipped with the algorithm, in which the circle changes from green to red, 
indicating that the patient is likely to develop hypotension in 5 min._ <br/>

![image](https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/assets/91899326/224db891-bb33-4a37-a16c-bcbeae16cea1)

## Conclusion <br/>

- _In conclusion, a deep learning algorithm based on routine non-invasive monitoring could predict 
the occurrence of intraoperative hypotension in diverse patients and surgeries. Our findings can
expand the application of medical AI to a broad population undergoing surgery with only basic 
monitoring. Future research should test whether this algorithm is highly accurate when applied 
prospectively in clinical practice and whether it could improve postoperative outcomes._ <br/>


- _The dataset used in our study has certain limitations._ <br/>
  1) _Firstly, the amount of data is quite limited for conducting sufficient training._ <br/>
  2) _Secondly, there is an imbalance in the distribution among classes. Such a limited quantity of data or a distribution biased towards specific classes can lead to overfitting problems during the learning process._ <br/><br/><br/>

--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/Google Colab-black?style=flat-square&logo=Google Colab&logoColor=yellow"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-FF9900?style=flat-square&logo=PyTorch&logoColor=EE4C2C"/>
  <img src="https://img.shields.io/badge/Numpy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
</p>

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; AI Hub Dataset : Korean Face Image <br/>
