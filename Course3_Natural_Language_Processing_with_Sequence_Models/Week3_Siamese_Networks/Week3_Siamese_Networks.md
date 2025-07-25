# Week 1: Sentiment Analysis with Logistic Regression

## Course Overview
**Learn to extract features from text into numerical vectors, then build a binary classifier for tweets using a logistic regression!**  
*Coursera - [DeepLearning.AI](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)*

---

## Learning Objectives
- [x] [Supervised ML & Sentiment Analysis](#1-supervised-ml--sentiment-analysis)
- [x] [Vocabulary & Feature Extraction](#2-vocabulary--feature-extraction)
- [x] [Feature Extraction with Frequencies](#3-feature-extraction-with-frequencies)
- [x] [Calculating word frequencies](#4-preprocessing)
- [x] [Putting it all together](#5-putting-it-all-together)
- [x] [Logistic Regression](#6-logistic-regression)


---

## 1. Supervised ML & Sentiment Analysis

![Supervised ML](images/week1_1_SupervisedML.png)
![Supervised ML](images/week1_2_SupervisedML.png)


---

## 2. Vocabulary & Feature Extraction

![Vocabulary & Feature Extraction](images/week1_3_VocabularyAndFeatureExtraction.png)


---

## 3. Feature Extraction with Frequencies

![Feature Extraction with Frequencies](images/week1_4_FeatureExtractionWithFrequencies.png)
![Feature Extraction with Frequencies](images/week1_5_FeatureExtractionWithFrequencies.png)
![Feature Extraction with Frequencies](images/week1_6_FeatureExtractionWithFrequencies.png)

---

## 4. Preprocessing

![Preprocessing](images/week1_7_Preprocessing.png)

---

## 5. Putting it all together

![Putting it all together](images/week1_8_PuttingItAllTogether.png)
![Putting it all together](images/week1_9_PuttingItAllTogether.png)


---

## 6. Logistic Regression

![Logistic Regression](images/week1_10_LogisticRegression.png)
![Logistic Regression](images/week1_11_LogisticRegression.png)
![Logistic Regression](images/week1_12_LogisticRegression.png)
![Logistic Regression](images/week1_13_LogisticRegression.png)
![Logistic Regression](images/week1_14_LogisticRegression.png)
![Logistic Regression](images/week1_15_LogisticRegression.png)
![Logistic Regression](images/week1_16_LogisticRegression.png)
![Logistic Regression](images/week1_17_LogisticRegression.png)

### Optinal Logistic Regression: Gradient
#### Derivation of the Sigmoid Function

First, we calculate the derivative of the sigmoid function:

\[
h(x) = \frac{1}{1 + e^{-x}}
\]
To compute the derivative \( h'(x) \), we apply the **quotient rule**:

The quotient rule says that for a function \( f(x) = \frac{u(x)}{v(x)} \), the derivative is:

\[
f'(x) = \frac{u'(x)v(x) - u(x)v'(x)}{[v(x)]^2}
\]

Here, we set:

- \( u(x) = 1 \Rightarrow u'(x) = 0 \)
- \( v(x) = 1 + e^{-x} \Rightarrow v'(x) = -e^{-x} \)

Now apply the quotient rule:

\[
h'(x) = \left( \frac{1}{1 + e^{-x}} \right)'
= \frac{-(1 + e^{-x})'}{(1 + e^{-x})^2}
= \frac{-\left(0 + (-x)' \cdot e^{-x} \right)}{(1 + e^{-x})^2}
= \frac{-(-1)e^{-x}}{(1 + e^{-x})^2}
= \frac{e^{-x}}{(1 + e^{-x})^2}
\]

Now, recall:
\[
h(x) = \frac{1}{1 + e^{-x}}, \quad 1 - h(x) = \frac{e^{-x}}{1 + e^{-x}}
\]

So we can write:
\[
h'(x) = h(x) \cdot (1 - h(x))
\]

#### Derivation of the Logistic Regression Cost Function Gradient
We start with the cost function for logistic regression:

\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[
y^{(i)} \log(h(x^{(i)}, \theta)) + (1 - y^{(i)}) \log(1 - h(x^{(i)}, \theta))
\right]
\]

We compute the partial derivative with respect to \( \theta_j \):

\[
    \frac{\partial}{\partial \theta_j} J(\theta) = \frac{\partial}{\partial \theta_j} \left(- \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h(x^{(i)}, \theta)) + (1 - y^{(i)}) \log(1 - h(x^{(i)}, \theta)) \right]\right)
\]

\[
= -\frac{1}{m} \sum_{i=1}^m \left[
y^{(i)} \frac{\partial}{\partial \theta_j} \log(h(x^{(i)}, \theta)) + 
(1 - y^{(i)}) \frac{\partial}{\partial \theta_j} \log(1 - h(x^{(i)}, \theta))
\right]
\]

Using the chain rule:

\[
= -\frac{1}{m} \sum_{i=1}^m \left[
\frac{y^{(i)}}{h(x^{(i)}, \theta)} \frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta) + \frac{1 - y^{(i)}}{1 - h(x^{(i)}, \theta)} \frac{\partial}{\partial \theta_j} (- h(x^{(i)}, \theta))
\right]
\]

\[
= -\frac{1}{m} \sum_{i=1}^m \left[
\frac{y^{(i)}}{h(x^{(i)}, \theta)} \frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta) - \frac{1 - y^{(i)}}{1 - h(x^{(i)}, \theta)} \frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta)
\right]
\]

Now factor out \( \frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta) \). Since \( h(x^{(i)}, \theta) = \sigma(\theta^T x^{(i)}) \), and we know:

\[
\frac{\partial}{\partial \theta_j} h(x^{(i)}, \theta) = h(x^{(i)}, \theta)(1 - h(x^{(i)}, \theta)) x_j^{(i)}
\]

So:

\[
= -\frac{1}{m} \sum_{i=1}^m \left[
\left( \frac{y^{(i)}}{h(x^{(i)}, \theta)} - \frac{1 - y^{(i)}}{1 - h(x^{(i)}, \theta)} \right)
h(x^{(i)}, \theta)(1 - h(x^{(i)}, \theta)) x_j^{(i)}
\right]
\]

Now simplify the expression inside the parentheses:

\[
\left( \frac{y^{(i)}}{h(x^{(i)}, \theta)} - \frac{1 - y^{(i)}}{1 - h(x^{(i)}, \theta)} \right)
= \frac{y^{(i)} (1 - h(x^{(i)}, \theta)) - (1 - y^{(i)}) h(x^{(i)}, \theta)}{h(x^{(i)}, \theta)(1 - h(x^{(i)}, \theta))}
\]

Then the numerator becomes:

\[
y^{(i)} - y^{(i)} h(x^{(i)}, \theta) - h(x^{(i)}, \theta) + y^{(i)} h(x^{(i)}, \theta)
= y^{(i)} - h(x^{(i)}, \theta)
\]

So the full expression simplifies to:

\[
\frac{y^{(i)} - h(x^{(i)}, \theta)}{h(x^{(i)}, \theta)(1 - h(x^{(i)}, \theta))} \cdot h(x^{(i)}, \theta)(1 - h(x^{(i)}, \theta)) = y^{(i)} - h(x^{(i)}, \theta)
\]

Thus the final gradient is:

\[
\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m} \sum_{i=1}^m \left( h(x^{(i)}, \theta) - y^{(i)} \right) x_j^{(i)}
\]







