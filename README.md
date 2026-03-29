# SPECTR---Deep-learning-for-protein-electrophoresis
SPECTR is a project conducted by the biochemistry lab of the CHU of Angers and initiated by Floris Chabrun and Xavier Dieu
The goal is to achieve level-expert serum protein electrophoresis interpretation through the use of deep learning models


Then, we aim to give a broader interpretation for protein electrophoresis (PSE) than it's current use. Using these models to detect previously undiagnosed diseases when using EPS.

<img width="1850" height="651" alt="image" src="https://github.com/user-attachments/assets/1d653f61-8c89-448d-9b27-352ac403343d" />


Here is the link through the article : https://doi.org/10.1093/clinchem/hvab133



##### To help expand the use of EPS in the diagnosis of more diseases, we transitioned from a supervised learning approach (cf article) to unsupervised learning, employing an autoencoder as the model architecture. Code follow in two parts :

The first part 'AE_traintest.py' concern the training of the model. It include training and validation part

The 2nd part 'AE_post_steps' and 'AE_clinicalanalysis' concern the use of the pre-trained model above. The objective is to perform supervised analyses using various machine learning algorithms on the latent space of the autoencoder. 

###### The AE_post_steps code is intended for routine interpretation of EPS using this autoencoder (e.g., classification of beta/gamma fractions, monoclonal peaks, hemolysis, etc.). 

###### The AE_clinicalanalysis code is designed to apply this method to the detection of novel pathologies (e.g., Alzheimer’s disease, COVID-19), based on the assumption that disease-specific proteins are released into the bloodstream during these pathologies and are therefore detectable by EPS. 
