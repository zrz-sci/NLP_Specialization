# Week 2: Part of Speech Tagging and Hidden Markov Models

## Course Overview
**Learn about Markov chains and Hidden Markov models, then use them to create part-of-speech tags for a Wall Street Journal text corpus!**  
*Coursera - [DeepLearning.AI](https://www.deeplearning.ai/courses/natural-language-processing-specialization/)*

---

## Learning Objectives

- [x] [Part of Speech Tagging](#1-part-of-speech-tagging)
- [x] [Markov Chains](#2-markov-chains)
- [x] [Hidden Markov Models](#3-hidden-markov-models)
- [x] [Calculating Probabilities](#4-calculating-probabilities)
- [x] [Populating the Transition Matrix](#5-populating-the-transition-matrix)
- [x] [Populating the Emission Matrix](#6-populating-the-emission-matrix)
- [x] [The Viterbi Algorithm](#7-the-viterbi-algorithm)
- [x] [The Viterbi Algorithm Initialization](#8-the-viterbi-algorithm-initialization)
- [x] [The Viterbi Algorithm Forward Pass](#9-the-viterbi-algorithm-forward-pass)
- [x] [The Viterbi Algorithm Backward Pass](#10-the-viterbi-algorithm-backward-pass)


---

## 1. Part of Speech Tagging

![Part of Speech Tagging](images/week2_1_PartOfSpeechTagging.png)
![Part of Speech Tagging](images/week2_2_PartOfSpeechTagging.png)
![Part of Speech Tagging](images/week2_3_PartOfSpeechTagging.png)
![Part of Speech Tagging](images/week2_4_PartOfSpeechTagging.png)

---

## 2. Markov Chains

![Markov Chains](images/week2_5_MarkovChains.png)
![Markov Chains](images/week2_6_MarkovChains.png)


---

## 3. Hidden Markov Models

![Hidden Markov Models](images/week2_7_HiddenMarkovModels.png)
![Hidden Markov Models](images/week2_8_HiddenMarkovModels.png)

---

## 4. Calculating Probabilities

![Calculating Probabilities](images/week2_9_CalculatingProbabilities.png)

---

## 5. Populating the Transition Matrix

![Populating the Transition Matrix](images/week2_10_PopulatingTransitionMatrix.png)
![Populating the Transition Matrix](images/week2_11_PopulatingTransitionMatrix.png)

In a real-world example, you might not want to apply smoothing to the initial probabilities in the first row of the transition matrix. That's because if you apply smoothing to that row by adding a small value to possibly zeroed valued entries. You'll effectively allow a sentence to start with any parts of speech tag, including punctuation

---

## 6. Populating the Emission Matrix

![Populating the Emission Matrix](images/week2_12_PopulatingEmissionMatrix.png)

---

## 7. The Viterbi Algorithm

![The Viterbi Algorithm](images/week2_13_ViterbiAlgorithm.png)
![The Viterbi Algorithm](images/week2_14_ViterbiAlgorithm.png)

---

## 8. The Viterbi Algorithm Initialization

![The Viterbi Algorithm Initialization](images/week2_15_ViterbiAlgorithmInitialization.png)
![The Viterbi Algorithm Initialization](images/week2_16_ViterbiAlgorithmInitialization.png)

---

## 9. The Viterbi Algorithm Forward Pass

![The Viterbi Algorithm Forward Pass](images/week2_17_ViterbiAlgorithmForward.png)
![The Viterbi Algorithm Forward Pass](images/week2_18_ViterbiAlgorithmForward.png)

## 10. The Viterbi Algorithm Backward

![The Viterbi Algorithm Backward Pass](images/week2_19_ViterbiAlgorithmBackward.png)
![The Viterbi Algorithm Backward Pass](images/week2_20_ViterbiAlgorithmBackward.png)
![The Viterbi Algorithm Backward Pass](images/week2_21_ViterbiAlgorithmBackward.png)
![The Viterbi Algorithm Backward Pass](images/week2_22_ViterbiAlgorithmBackward.png)


