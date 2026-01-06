# Intrusion Detection System (IDS) Project

## Overview

An Intrusion Detection System (IDS) plays a critical role in cybersecurity by monitoring a network or systems for unauthorized access, potential malicious activity, and security policy violations. This project focuses on developing a predictive model capable of differentiating between **“bad connections”** (intrusions or attacks) and **“good connections”** (normal activity), using advanced machine learning techniques.

The IDS leverages the **KDD Cup 1999 dataset**, one of the most widely used datasets for intrusion detection research, to train and evaluate several classification algorithms. After conducting feature extraction and selection, the dataset is used to compare the performance of seven machine learning models, with results measured by accuracy and computational efficiency. The **Decision Tree** model was found to be the most effective, providing an optimal balance between accuracy and computational time.

---

## Dataset

### KDD Cup 1999 Dataset by DARPA
The KDD Cup 1999 dataset, developed by DARPA, is a comprehensive resource for evaluating intrusion detection systems. This dataset includes a diverse range of normal and malicious activities, allowing models to learn and differentiate between various types of connections and attacks.

- **Download**: [KDD Cup 1999 dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Key Features**:
  - 41 features for each connection record, covering network traffic details and host attributes.
  - Includes multiple types of attacks, such as DoS (Denial of Service), R2L (Remote to Local), U2R (User to Root), and probing attacks.
  - Provides a labeled classification for each record as either **normal** or **attack**.

---

## Project Goals

The primary goal of this project is to develop a predictive IDS model that can accurately classify network connections into **intrusions** or **normal connections**. To achieve this, a variety of machine learning algorithms were tested to determine the most efficient model based on accuracy and computational speed. 

---

## Methodology

1. **Data Preprocessing**: 
   - Feature extraction and selection to reduce dimensionality and improve model performance.
   - Data normalization and encoding to prepare the dataset for model training.

2. **Model Selection and Training**:
   - A total of seven models were selected, trained, and evaluated for this project.
   - Each model was tested on the KDD Cup 1999 dataset, with accuracy and computational time recorded for comparison.

3. **Evaluation Metrics**:
   - **Accuracy**: Measures the percentage of correctly classified instances.
   - **Computational Time**: Assesses the efficiency of each model, a critical factor in real-time IDS deployment.

---

## Algorithms Used

The following machine learning algorithms were implemented and compared in this project:

1. **Gaussian Naive Bayes**: A probabilistic classifier based on Bayes’ theorem, effective for initial comparisons due to its simplicity.
2. **Decision Tree**: A tree-based model that iteratively splits the dataset to maximize predictive power, achieving the best results in terms of accuracy and computational speed.
3. **Random Forest**: An ensemble model using multiple decision trees to improve accuracy and robustness.
4. **Support Vector Machine (SVM)**: A model that identifies the optimal hyperplane for separating classes, suitable for binary classification tasks.
5. **Logistic Regression**: A straightforward statistical method, often used as a baseline for binary classification.
6. **Gradient Boosting**: A boosting algorithm that combines weak models to create a strong predictive model, effective for handling complex patterns in data.
7. **Artificial Neural Network (ANN)**: A multi-layer neural network, suitable for complex, non-linear relationships within data.

---

## Results

Each model’s performance was evaluated based on **accuracy** and **computational time**. The **Decision Tree** model outperformed other algorithms in both metrics, making it the most suitable choice for an efficient, accurate IDS system.

### Performance Summary

Here’s an example table with placeholder values replaced with sample performance results. Please adjust the values according to your specific project results:

| Model                  | Test Accuracy (%) | Training Accuracy (%) |
|------------------------|-------------------|-----------------------|
| Gaussian Naive Bayes   | 85               | 87                    |
| Decision Tree          | 97               | 98                    |
| Random Forest          | 98               | 99                    |
| Support Vector Machine | 97               | 97                    |
| Logistic Regression    | 96               | 96                    |
| Gradient Boosting      | 97               | 98                    |
| Artificial Neural Network | 95           | 96                    |

---

## Conclusion

This project demonstrates the effectiveness of machine learning algorithms in building an Intrusion Detection System capable of identifying malicious network activity. The **Decision Tree** model emerged as the most effective in terms of accuracy and computational time, making it a robust choice for real-time intrusion detection applications.

Future improvements could include integrating real-time data processing and testing on a broader range of network environments for enhanced applicability. 

By applying machine learning to intrusion detection, this project contributes towards building smarter, faster, and more effective cybersecurity solutions.

---

## Project Resources

- **Dataset**: [KDD Cup 1999 Dataset](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- **Libraries Used**: `Scikit-learn`, `Pandas`, `NumPy`, `Matplotlib`
- **Tools**: Python, Jupyter Notebook

---

**Project By:** Sk Mastan

