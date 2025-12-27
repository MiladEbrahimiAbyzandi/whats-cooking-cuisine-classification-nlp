# 🍲 What's Cooking? — Cuisine Classification (Kaggle)
**Text Classification | TF–IDF + Logistic Regression | Neural Baseline with Learned Embeddings**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%7C%20Embeddings-informational)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview
This project tackles Kaggle’s **What’s Cooking?** competition: predict a recipe’s cuisine (20 classes) from its ingredient list. We analyze **39,774 recipes** across **20 cuisines**, run EDA, build two modeling pipelines (classical ML + neural), and compare validation and Kaggle test performance.

**Key result:** A **multinomial logistic regression** model with **TF–IDF** features outperformed a neural embedding baseline and was selected as the final submission model. 

---

## Dataset
- **Source:** Kaggle “What’s Cooking?” (recipes with ingredient lists)
- **Size:** 39,774 recipes, 20 cuisines; ~10 ingredients per recipe on average fileciteturn1file0L17-L19  
- **Notes from EDA:** moderate class imbalance; word clouds show common ingredients (e.g., salt) vs cuisine-specific signals 


---

## Methods

### 1) TF–IDF + Multinomial Logistic Regression (Final)
**Feature engineering**
- Clean text (lowercase, remove special chars, normalize spacing)
- Merge ingredients into a single string per recipe
- Vectorize using **TF–IDF** with **unigrams + bigrams** to capture multi-word ingredients (e.g., “soy sauce”) 
- Filter rare features (appear in < 3 recipes) → final space: **23,593 features** 

**Modeling**
- Multinomial logistic regression with **L2 regularization** to reduce overfitting on sparse, rare features 
- Hyperparameter tuning via CV over **C ∈ {1, 3, 5, 7, 10}** fileciteturn1file2L4-L6
- Tested `class_weight="balanced"`; it reduced validation accuracy by ~1 point, so the final model uses default weighting 

### 2) Neural Baseline: Learned Embeddings + Global Average Pooling
**Representation**
- Word-level tokenization with max vocab size **15,000** and padded sequence length **40**   
- Learned embeddings (not pretrained) to capture domain-specific ingredient co-occurrence patterns 

**Architecture**
- `Embedding(64) → GlobalAveragePooling1D → Dense(128, ReLU) → Softmax`  
- 64-d embedding chosen over 128 for similar accuracy with less complexity 
- Avg pooling preferred over max pooling; alternatives (Flatten, Conv1D+MaxPool) underperformed 

---

## Experiment: preserving multi-word ingredients
We tested an alternate cleaning strategy that keeps multi-word ingredients intact by replacing spaces with underscores (e.g., `soy_sauce`). 
Result: **no meaningful improvement** for either logistic regression or the neural model. 

---

## Results
### Performance summary
| Model | Tokenization | Val Acc | Kaggle Test Acc |
|---|---|---:|---:|
| **TF–IDF + Logistic Regression** | Word-level | **0.7822** | **0.7915** |
| Neural Net (Embedding + GlobalAvgPool) | Word-level | 0.7743 | 0.7754 |
| TF–IDF + Logistic Regression | Ingredient-level (`soy_sauce`) | 0.7791 | 0.7874 |
| Neural Net (Embedding + GlobalAvgPool) | Ingredient-level (`soy_sauce`) | 0.7742 | 0.7754 |

**Final model:** **TF–IDF + Logistic Regression (word-level)** — selected for best generalization and Kaggle score. fileciteturn1file1L24-L30

---

## Figures (generated in notebook)
- **Cuisine word clouds** (EDA): highlight common vs discriminative ingredients  
- **Neural network architecture**: Embedding → GlobalAvgPool → Dense → Softmax   

---

## How to run
```bash
# 1) Clone
git clone https://github.com/MiladEbrahimiAbyzandi/whats-cooking-cuisine-classification-nlp.git
cd whats-cooking-cuisine-classification

# 2) Create environment (example)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Launch notebook
jupyter notebook
```

---

## Authors
- **Milad Ebrahimi Abyazandi**

Course context: DATA6100 (Fall 2025).
