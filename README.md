# 💉  Prediction of Intraoperative Hypotension Using Deep Learning Models Based on Non-invasive Monitoring Devices
<br/>
 
### Overview <br/>

 - _Intraoperative hypotension is associated with adverse outcomes. Predicting and proactively managing hypotension can reduce its incidence. Previously, hypotension prediction algorithms using artificial intelligence were developed for invasive arterial blood pressure monitors. This study tested whether routine non-invasive monitors could also predict intraoperative hypotension using deep learning algorithms._ <br/><br/>

### Participants and Dataset <br/>

- _An open-source database was used for algorithm development and internal validation (http://vitaldb.net/dataset). The open-source database consisted of 6388 patients who underwent non-cardiac surgery between June 2016 and August 2017 at Seoul National University Hospital,  South Korea, a tertiary referral hospital._ <br/>

- _Eligible patients selected from the open-source database were randomly divided into a training set of 80% and a test set of 20%. The training set was used to build the predictive model, and the test set was used to evaluate the final performance of the model. Patient data from the test set were not exposed during model training. A validation set (10% of the training set) was randomly selected from the training set to tune the hyperparameters of the trained model._ <br/>

<p align="center">
  <img width="50%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig1.png?raw=true">
  <br>
  <em> Figure 1) Composition of dataset.</em>
</p> 
<br>

- _Patient characteristics and dataset compositions between hypotensive and non-hypotensive cases are summarized in Supplementary Table 1. Surgeries included general, gynecological, thoracic, and genitourinary surgeries. The mean ± SD age was 57 ± 14 years, and 49% of the participants in the training set were men. The comparison variables were well-balanced between hypotensive and nonhypotensive cases._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig2.png?raw=true">
  <br>
  <em> Table 1) Patient characteristics and dataset composition between hypotensive and non-hypotensive cases.</em>
</p> 
<br>

- _We externally validated the final model using a dataset from another hospital. The dataset for 
external validation consisted of 421 patients who underwent non-cardiac surgery in April 2022 at the 
Samsung Medical Center, South Korea, a tertiary referral hospital. All monitoring data for external 
validation were extracted and processed in the same manner as the data used for model development._ <br/><br/>

### Data Preparation <br/>

- _All monitoring data were sampled from the source database at a rate of 100 Hz. Waveform data, such as electrocardiography, photoplethysmography, and capnography, were recorded continuously._ <br/>

- _However, non-invasive blood pressure data consisted of 1-point values with a mean measurement interval of 2.5 minutes and the bispectral index was recorded every second. Therefore, blood pressure and bispectral index data were rearranged at 100 Hz intervals by filling empty places with the preceding values to facilitate a seamless end-to-end training process._ <br/>

- _Our classification tasks involved binary classification of hypotension events (systolic blood pressure < 90 mmHg) or non-hypotension events (systolic blood pressure >= 90 mmHg) occurring after 5 min. These input signals were extracted as 100 Hz input data in units of 30s, sequentially shifted by 10 s, and then extracted repeatedly._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig3.png?raw=true">
  <br>
  <em> Figure 2) Study design and data processing. Five types of non-invasive biosignal (input data) were extracted as 100 Hz input data in units of 30s, sequentially shifted by 10 s, and then extracted repeatedly.</em>
</p> 
<br>

- _In the training dataset, the ratio of intraoperative hypotension to non-hypotension cases was highly unbalanced at 1:9. Therefore, a Focal Loss function with a weighting mechanism was used to address the class imbalance problem. A Focal Loss function was applied during the training process to enable the model to learn intensively from rare hypotensive data. By dynamically adjusting the weighting of the training data, the Focal Loss function reoriented the learning process toward the minority class, thereby mitigating the detrimental effects of skewed data formulation._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig4.png?raw=true">
  <br>
  <em> Figure 3) Comparison of Focal Loss with different γ values to Cross-Entropy Loss for class imbalance correction.</em>
</p> 
<br>

- _Normalization was performed to improve the efficiency of the deep learning models by training them on patterns. Scaling was performed using minimum-maximum normalization to address potential differences in the distributions of the training and test data._ <br/>
  
<p align="center">
  <img width="35%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig5.png?raw=true">
  <br>
  <em> Figure 4) Minimum-maximum normalization formula.</em>
</p> 
<br>

### Model Development  <br/>

- _The deep learning model was built using the Multi-head Attention model for feature-wise information and the Globally Attentive Locally Recurrent (GALR) model for time-wise information. The Multi-head Attention model made predictions by considering the importance ratio of the attributes in the input data. The GALR model evaluated the importance of time in input data. Finally, a fully connected model was developed by combining the Multi-head Attention and GALR models._ <br/>

- _This approach enabled the model to leverage the strengths of both the feature and time axes, thereby enhancing its ability to effectively predict hypotension events._ <br/><br/>

<p align="center">
  <img width="90%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig6.png?raw=true">
  <br>
  <em> Figure 5) Concept of the final prediction model. Final model is a combination of the Multi-head Attention model and the Globally Attentive Locally Recurrent model with a Focal Loss function.</em>
</p> 
<br>

### Metrix <br/>

- _The primary outcome was the area under the receiver operating characteristic curve (AUROC) for the classification models._ <br/>
<p align="center">
  <img width="70%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig7.png?raw=true">
  <br>
  <em> Figure 6) ROC Curve analysis demonstrating model performance in classification tasks, with larger AUROC indicating superiority.
</p> 
<br>

- _The secondary outcomes included the accuracy, sensitivity, and specificity of the established model._ <br/>
  
<p align="center">
  <img width="70%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig8.png?raw=true">
  <br>
  <em> Figure 7) Formulas for accuracy, sensitivity, and specificity.</em>
</p> 
<br>

- _The study outcomes were presented with 95% confidence intervals (CIs) estimated using the bootstrapping method._ <br/><br/>

### Results <br/>

- _The fully connected model, which combines the Multi-head Attention and the GALR model with a Focal Loss function, achieved the highest AUROC of 0.917 (95% CI, 0.915–0.918) for the test set of the original data and 0.833 (95% CI, 0.830–0.836) for the external validation dataset. The secondary outcomes, including accuracy, sensitivity, and specificity, are presented in Table 2._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig9.png?raw=true">
  <br>
  <em> Table 2) Performance of the final model.</em>
</p> 
<br>

- _We introduced attention mechanisms into our architecture. Figure 8 illustrates the attention map analysis employed to visualize the features and time domains utilized for decision-making in hypotension events. According to the attention map, our algorithm utilized data from each monitor with weights ranging from 5% to 22% for determining hypotension._ <br/>

<p align="center">
  <img width="90%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig10_.png?raw=true">
  <br>
  <em> Figure 8)  Illustrates Attention Map Weights for Feature Importance in Hypotension Event Prediction.</em>
</p> 
<br>

- _Among them, the bispectral index had the highest weight (20%) in predicting hypotension, and mean  blood pressure had the highest weight (22%) in predicting non-hypotension._ <br/>

<p align="center">
  <img width="75%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig11.png?raw=true">
  <br>
  <em> Figure 9)  Illustrates Attention Map Weights for Feature Importance in Hypotension Event Prediction.</em>
</p> 
<br>

- _Figure 10 is a monitor equipped with the algorithm, in which the circle changes from green to red, 
indicating that the patient is likely to develop hypotension in 5 min._ <br/>

<p align="center">
  <img width="60%" src="https://github.com/qortmdgh4141/GALR_Globally-Attentive-Locally-Recurrent-Model/blob/main/image/fig12.png?raw=true">
  <br>
  <em> Figure 10) A monitor equipped with the algorithm, in which the circle changes from green to red, indicating that the patients is likely to develop hypotension in 5 min.</em>
</p> 
<br>

### Conclusion <br/> 

- _In conclusion, a deep learning algorithm based on routine non-invasive monitoring could predict the occurrence of intraoperative hypotension in diverse patients and surgeries. Our findings can expand the application of medical AI to a broad population undergoing surgery with only basic monitoring. Future research should test whether this algorithm is highly accurate when applied prospectively in clinical practice and whether it could improve postoperative outcomes._ <br/><br/><br/>

--------------------------
### 💻 S/W Development Environment
<p>
  <img src="https://img.shields.io/badge/Windows 10-0078D6?style=flat-square&logo=Windows&logoColor=white"/>
  <img src="https://img.shields.io/badge/NVIDIA-black?style=flat-square&logo=NVIDIA&logoColor=76B900"/>
  <img src="https://img.shields.io/badge/Visual Studio-5C2D91?style=flat-square&logo=Visual studio&logoColor=white"/> 
</p>  
<p>
  <img src="https://img.shields.io/badge/PyCharm-66FF00?style=flat-square&logo=PyCharm&logoColor=black"/>
  <img src="https://img.shields.io/badge/Anaconda-e9e9e9?style=flat-square&logo=Anaconda&logoColor=44A833"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
</p>
<p>
   <img src="https://img.shields.io/badge/PyTorch-FF9900?style=flat-square&logo=PyTorch&logoColor=EE4C2C"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=Numpy&logoColor=blue"/>
  <img src="https://img.shields.io/badge/scikit learn-blue?style=flat-square&logo=scikitlearn&logoColor=F7931E"/>
</p>   

### 💾 Datasets used in the project
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dataset from Non-cardiac Surgery Patients (Seoul National University Hospital, South Korea) <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Dataset from Non-cardiac Surgery Patients (Samsung Medical Center, South Korea) <br/>
