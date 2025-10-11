[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![University of Florence](https://i.imgur.com/1NmBfH0.png)](https://ingegneria.unifi.it)

📄 **[Read the Full Report](https://github.com/mattiamarilli/SVMSMO/blob/main/report/SMO_Report.pdf)**

Custom SMO SVM Implementation
---

## Overview
This project implements a **custom SVM (Support Vector Machine) with Sequential Minimal Optimization - SMO** in Python and compares it with `scikit-learn` SVC on LIBSVM datasets (`a1a`, `a2a`...). The goal is to understand the internal workings of SMO and evaluate performance on real datasets in terms of **accuracy** and **training time**.

The main phases considered are:  
1. **Training**: Training the SVM model with linear or RBF kernel.  
2. **Prediction**: Evaluating performance on test data.  

Two main versions are compared:  
1. **Custom SMO SVM**: Manual implementation of the SMO algorithm.  
2. **Scikit-learn SVC**: Standard implementation using the RBF kernel.

---

## Key Features

- **Linear and RBF Kernels**: Full support for both kernel types.  
- **KKT Violation Selection**: Intelligent update of alphas.  
- **Optimized Bias Update**: Efficient calculation of the bias term.  
- **Automatic scikit-learn Comparison**: Measures runtime and accuracy for multiple `C` values.  

---

## Implementation

### Custom SMO SVM
- Trains the model by optimizing two alphas at a time according to KKT conditions.  
- Maintains an error cache to speed up computations.  
- Selects the final support vectors after training.

### Scikit-learn SVC
- Standard SVM implementation with RBF kernel.  
- Training and prediction using the `fit` and `predict` API.

---

## Performance Evaluation

### Metrics
- **Execution Time**: Time needed to train the model.  
- **Accuracy**: Percentage of correct predictions on test data.

### Results
- Custom SMO achieves comparable accuracy to scikit-learn.  
- Scikit-learn is generally faster on large datasets due to internal optimizations.

---

