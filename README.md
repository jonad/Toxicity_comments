# Recurrent Neural Network for sentence-level Text Classification

This project is about building and evaluating recurrent neural network models
for sentence-level text classification. The final models detect toxicity in
short texts as well as the type of toxicity, which include the following
categories: severe toxicity, obscene, identity attack, insult, and threat.
The final models can be used for filtering online posts and comments,
social media policing, and user education.
<br>
### Links
- [The deployed models](TODO)

### Sections
- [Dataset Summary](#dataset-summary)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)

## Dataset Summary
[back to top](#sections)

-  1.8+ million user comments dataset was downloaded from the [Kaggle competition labeled 'Jigsaw Unintended Bias in Toxicity Classification'](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification).
- The dataset consists of 1.8+ million user comments that have been hand-labeled by human raters for toxicity levels.
- The dataset also includes the following toxicity types: severe toxicity,  obscene, threat, insult, and identity attack.
<br>
<br>

## Exploratory Data Analysis
[back to top](#sections)

###  Toxicity class distribution
![](./images/data_distribution.png)

### Correlation heatmap of types
<br>
<br>

![](./images/correlations.png)

<br>
<br>

## Models
[back to top](#sections)

### Long Short-Term Memory Model (LSTM)
![](./images/lstm.jpg)

<br >

### Bidirectional Long Short-Term Memory Model (BiLSTM)

<br >
![](./images/bilstm.jpg)

<br >

### BiLSTM with Attention Mechanism
![](./images/attention.jpg)

<br>
<br>

## Training
[back to top](#sections)

### Learning Curves

![](./images/training.png)

<br>
<br>

## Evaluation
[back to top](#sections)
<br >

### ROC-AUC Toxicity
![](./images/toxicity.png)

### ROC-AUC Severe Toxicity
![](./images/severe_toxicity.png)
### ROC-AUC Obscene
![](./images/obscene.png)
### ROC-AUC Identity Attack
![](./images/identity_attack.png)
### ROC-AUC Insult
![](./images/insult.png)
### ROC-AUC Threat
![](./images/threat.png)
<br >
## Testing
[back to top](#sections)
<br >

![](./images/test1.png)

![](./images/t2.png)

![](./images/t3.png)




