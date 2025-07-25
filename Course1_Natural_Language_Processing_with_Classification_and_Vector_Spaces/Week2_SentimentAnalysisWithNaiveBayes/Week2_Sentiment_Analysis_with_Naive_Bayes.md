# Week 2. Sentiment Analysis with Naive Bayes

## Course Overview
**Learn the theory behind Bayes' rule for conditional probabilities, then apply it toward building a Naive Bayes tweet classifier of your own!**  
*Coursera - [DeepLearning.AI](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)*

---

## Learning Objectives
- [x] [Probability and Bayes' Rule](#1-probability-and-bayes-rule)
- [x] [Naive Bayes Introduction](#2-naive-bayes-introduction)
- [x] [Laplacian Smoothing](#3-laplacian-smoothing)
- [x] [Log-Likelihood](#4-log-likelihood)
- [x] [Training Naive Bayes](#5-training-naive-bayes)
- [x] [Testing Naive Bayes](#6-testing-naive-bayes)
- [x] [Naive Bayes Assumptions](#7-naive-bayes-assumptions)
- [x] [Error Analysis](#8-error-analysis)   
- [x] [Applications of Naive Bayes](#9-applications-of-naive-bayes)


---

## 1. Probability and Bayes' Rule

![Probability and Bayes' Rule](images/week2_1_ProbabilityAndBayesRule.png)
![Probability and Bayes' Rule](images/week2_2_ProbabilityAndBayesRule.png)
![Probability and Bayes' Rule](images/week2_3_ProbabilityAndBayesRule.png)


---

## 2. Naive Bayes Introduction

![Naive Bayes Introduction](images/week2_4_NaiveBayesIntroduction.png)
![Naive Bayes Introduction](images/week2_5_NaiveBayesIntroduction.png)


---

## 3. Laplacian Smoothing

![Laplacian Smoothing](images/week2_6_LaplacianSmoothing.png)
![Laplacian Smoothing](images/week2_7_LaplacianSmoothing.png)

---

## 4. Log-Likelihood

![Log-Likelihood](images/week2_8_LogLikelihood.png)
![Log-Likelihood](images/week2_9_LogLikelihood.png)
![Log-Likelihood](images/week2_10_LogLikelihood.png)
![Log-Likelihood](images/week2_10_1_LogLikelihood.png)
![Log-Likelihood](images/week2_11_LogLikelihood.png)
![Log-Likelihood](images/week2_12_LogLikelihood.png)
![Log-Likelihood](images/week2_13_LogLikelihood.png)
![Log-Likelihood](images/week2_14_LogLikelihood.png)
![Log-Likelihood](images/week2_15_LogLikelihood.png)

---

## 5. Training Naive Bayes

![Training Naive Bayes](images/week2_16_TrainingNaiveBayes.png)
![Training Naive Bayes](images/week2_17_TrainingNaiveBayes.png)
![Training Naive Bayes](images/week2_18_TrainingNaiveBayes.png)
![Training Naive Bayes](images/week2_19_TrainingNaiveBayes.png)
![Training Naive Bayes](images/week2_20_TrainingNaiveBayes.png)
![Training Naive Bayes](images/week2_21_TrainingNaiveBayes.png)



---

## 6. Testing Naive Bayes

![Testing Naive Bayes](images/week2_22_TestingNaiveBayes.png)
![Testing Naive Bayes](images/week2_23_TestingNaiveBayes.png)
![Testing Naive Bayes](images/week2_24_TestingNaiveBayes.png)
![Testing Naive Bayes](images/week2_25_TestingNaiveBayes.png)


---

## 6. Application of Naive Bayes

![Application of Naive Bayes](images/week2_26_ApplicationOfNaiveBayes.png)
![Application of Naive Bayes](images/week2_27_ApplicationOfNaiveBayes.png)

### 6.1 Log-Likelihood in Naive Bayes for Multiclass Classification

This note explains how **Naive Bayes** computes **log-likelihood** for each class in a **multiclass classification** problem. We will go through a concrete example from training to testing, and compute log-likelihoods step by step.

#### 6.1.1 Problem Setup

We have a simple dataset with two numerical features and three classes: **Class 0, Class 1, Class 2**. We use **Gaussian Naive Bayes**, which assumes that each feature follows a normal distribution conditioned on the class.

#### 6.1.2 Training Data:

| Sample | Feature 1 | Feature 2 | Class |
|--------|-----------|-----------|-------|
| A      | 1.0       | 2.0       | 0     |
| B      | 1.2       | 1.9       | 0     |
| C      | 4.5       | 4.2       | 1     |
| D      | 4.7       | 4.0       | 1     |
| E      | 8.0       | 8.0       | 2     |
| F      | 7.8       | 8.2       | 2     |

#### 6.1.3 Step 1: Training

We compute:

1. **Prior probabilities** \( P(C_k) \) for each class.
2. **Mean** and **variance** of each feature for each class.

#### Prior Probabilities:

- \( P(C=0) = 2/6 \)
- \( P(C=1) = 2/6 \)
- \( P(C=2) = 2/6 \)

#### Gaussian Parameters:

- **Class 0:**
  - Feature 1: \( \mu = 1.1 \), \( \sigma^2 = 0.01 \)
  - Feature 2: \( \mu = 1.95 \), \( \sigma^2 = 0.0025 \)
- **Class 1:**
  - Feature 1: \( \mu = 4.6 \), \( \sigma^2 = 0.01 \)
  - Feature 2: \( \mu = 4.1 \), \( \sigma^2 = 0.01 \)
- **Class 2:**
  - Feature 1: \( \mu = 7.9 \), \( \sigma^2 = 0.01 \)
  - Feature 2: \( \mu = 8.1 \), \( \sigma^2 = 0.01 \)

#### 6.1.4 Step 2: Testing

Now, we want to classify a test sample:

Test input: \( x = [4.6, 4.0] \)

We compute the log-likelihood for each class:

\[
\log P(C_k | x) \propto \log P(C_k) + \sum_{i=1}^{2} \log P(x_i | C_k)
\]

###### Class 0:

- Feature 1 is far from \( \mu = 1.1 \), so \( \log P(x_1 | C=0) \approx -\infty \)
- Feature 2 is far from \( \mu = 1.95 \), so \( \log P(x_2 | C=0) \approx -\infty \)
- Total log-likelihood: very negative

###### Class 1:

- Feature 1:
  \[
  \log P(4.6 | \mu=4.6, \sigma^2=0.01) = -\frac{1}{2} \log(2\pi \cdot 0.01) \approx 1.38
  \]
- Feature 2:
  \[
  \log P(4.0 | \mu=4.1, \sigma^2=0.01) = 1.38 - \frac{(0.1)^2}{2 \cdot 0.01} = 1.38 - 0.5 = 0.88
  \]
- Prior:
  \[
  \log P(C=1) = \log(1/3) \approx -1.0986
  \]
- Total log-likelihood:
  \[
  \log P(C=1 | x) \approx -1.0986 + 1.38 + 0.88 = 1.1614
  \]

###### Class 2:

- Feature 1 and 2 are far from \( \mu = 7.9 \), \( \mu = 8.1 \)
- So \( \log P(x_i | C=2) \approx -\infty \)
- Total log-likelihood: very negative

#### 6.1.5 Prediction

We choose the class with the highest log-likelihood:

\[
\hat{y} = \arg\max_k \log P(C_k | x)
\]

The predicted class is **Class 1**.

### 6.2 Maximum Log-Likelihood Value

In Gaussian Naive Bayes, the maximum log-likelihood of a single feature occurs when the feature equals the mean:

\[
\max \log P(x_i | C_k) = -\frac{1}{2} \log(2\pi \sigma^2)
\]

###### Examples:

- If \( \sigma^2 = 1 \), then:
  \[
  \max \log P(x_i | C_k) = -\frac{1}{2} \log(2\pi) \approx -0.9189
  \]
- If \( \sigma^2 = 0.01 \), then:
  \[
  \max \log P(x_i | C_k) = -\frac{1}{2} \log(2\pi \cdot 0.01) \approx 1.38
  \]

For multiple features, the total log-likelihood is the sum of individual log-likelihoods:

\[
\log P(x | C_k) = \sum_{i=1}^{d} \log P(x_i | C_k)
\]

### 6.3 Summary

- Naive Bayes calculates:
  \[
  \log P(C_k) + \sum_{i=1}^{d} \log P(x_i | C_k)
  \]
  for each class \( C_k \).
- The predicted label is the one with the highest total log-likelihood.
- Log-likelihood is derived from the Gaussian probability density function:
  \[
  \log P(x_i | C_k) = -\frac{1}{2} \log(2\pi \sigma^2) - \frac{(x_i - \mu)^2}{2\sigma^2}
  \]
- The maximum value depends on the feature variance \( \sigma^2 \), and occurs when \( x_i = \mu \).


---

## 7. Naive Bayes Assumptions

![Naive Bayes Assumptions](images/week2_28_NaiveBayesAssumptions.png)


---

## 8. Error Analysis

![Error Analysis](images/week2_29_ErrorAnalysis.png)

## 9. Naive Bayes Multi-classification Example

Naive Bayes naturally extends to multiclass classification problems. Here's how it works:

### 9.1 Key Concepts for Multiclass Naive Bayes

1. **Multiple Classes**: Instead of just binary classification (positive/negative), we can have K classes: C₁, C₂, ..., Cₖ

2. **Class Probabilities**: For each class k, we compute:
   ```
   P(Cₖ | x) ∝ P(Cₖ) × P(x | Cₖ)
   ```

3. **Feature Independence**: The naive assumption still applies - features are independent given the class.

### 9.2 Training Process

For each class Cₖ:
- Calculate **prior probability**: P(Cₖ) = (number of samples in class k) / (total samples)
- Calculate **likelihood parameters** for each feature given the class
- For text classification: word frequencies in each class
- For Gaussian NB: mean and variance of each feature per class

### 9.3 Prediction Process

For a new sample x:

1. **Calculate log-likelihood for each class**:
   ```
   log P(Cₖ | x) = log P(Cₖ) + Σᵢ log P(xᵢ | Cₖ)
   ```

2. **Choose the class with highest log-likelihood**:
   ```
   ŷ = argmax_k [log P(Cₖ | x)]
   ```

### 9.4 Example: Email Classification

Consider classifying emails into 3 categories:
- **Class 0**: Spam
- **Class 1**: Personal  
- **Class 2**: Work

**Training Data**:
- 100 spam emails
- 150 personal emails  
- 50 work emails

**Prior Probabilities**:
- P(Spam) = 100/300 = 0.33
- P(Personal) = 150/300 = 0.50
- P(Work) = 50/300 = 0.17

**For a new email with words ["meeting", "urgent", "deadline"]**:

1. Calculate likelihood for each word in each class
2. Sum log-likelihoods: 
   - log P(Spam | email) = log(0.33) + log P("meeting"|Spam) + log P("urgent"|Spam) + log P("deadline"|Spam)
   - log P(Personal | email) = log(0.50) + log P("meeting"|Personal) + log P("urgent"|Personal) + log P("deadline"|Personal)
   - log P(Work | email) = log(0.17) + log P("meeting"|Work) + log P("urgent"|Work) + log P("deadline"|Work)

3. Predict the class with highest total log-likelihood

### 9.5 Advantages of Multiclass Naive Bayes

- **Scalable**: Easily handles any number of classes
- **Efficient**: Linear time complexity in number of classes
- **Probabilistic**: Provides confidence scores for each class
- **Simple**: Same algorithm, just extended to K classes instead of 2

### 9.6 Applications

- **Document Classification**: News articles into categories (sports, politics, technology, etc.)
- **Spam Filtering**: Email classification (spam, ham, promotional, etc.)
- **Sentiment Analysis**: Multi-level sentiment (very negative, negative, neutral, positive, very positive)
- **Language Detection**: Identifying the language of text documents
