# Bird Classification with Transfer Learning (TensorFlow)

Two image-classification pipelines in TensorFlow/Keras:
- **Birds vs Squirrels** — 3-class baseline with Xception, rapid convergence on a small balanced set.
- **Birder** — fine-grained **358-class** bird species classifier with InceptionV3, two-phase fine-tuning and Top-K metrics.

---

## Introduction

This project compares transfer-learning setups on two tasks:
1. **Birds vs Squirrels (3-class)** — fast, balanced dataset for establishing a strong baseline.  
2. **Birder (358-class)** — large output space with few samples per class (~20 train / ~10 val), emphasizing regularization, Top-K accuracy, and careful fine-tuning.

When building vision models, two risks exist:
1. **Underfitting** on complex, fine-grained classes.  
2. **Overfitting** when data per class is small.

We mitigate these via pretrained CNNs, controlled unfreezing, augmentation, and early stopping.

---

## Business / Research Objectives

- Deliver a **reproducible** transfer-learning pipeline with TFRecords or folder datasets.  
- Quantify benefits of **fine-tuning** and **Top-K** metrics for fine-grained recognition.  
- Provide **readable class mappings** to aid error analysis (names ↔ indices).

---

## Data

- **Birds vs Squirrels (3-class):** small, balanced dataset (TFRecords or folders).  
- **Birder (358-class):** fine-grained species dataset with few images per class.  
- Pipelines use `tf.data` with map/batch/prefetch; see scripts for expected paths.

Class mapping files:
- `birdNames.txt` — common names for 358 classes  
- `birdLabs.txt` — label IDs / indices for 358 classes

---

## Analytical Approach

1. **Preprocessing & Augmentation**  
   - Resize/pad to **299×299**, Xception/Inception preprocessing.  
   - Augmentations (random flip/zoom/contrast/rotation) on the 358-class task.

2. **Models**  
   - **Birds vs Squirrels (3-class):** **Xception** (ImageNet, frozen) → GAP → Dropout(0.5) → Dense(256, ReLU) → Dense(3, Softmax).  
   - **Birder (358-class):** **InceptionV3** (ImageNet) → GAP → Dense(512, ReLU) → BN → Dropout(0.2) → Dense(256, ReLU) → BN → Dropout(0.2) → Dense(358, Softmax). Two-phase training: head-only, then unfreeze top ~30 layers. Metrics: Accuracy, **Top-5/10/20**.

3. **Training Strategy**  
   - Adam optimizer, early stopping, model checkpointing.  
   - Fine-tuning balances compute with accuracy; full unfreeze avoided for runtime.

---

## Results (Summary)

- **Birds vs Squirrels:** ~**98%** train / **98%** val accuracy.  
- **Birder (358-class):** **Top-1 ≈ 46%**, **Top-5 ≈ 76%**, **Top-10 ≈ 85%**, **Top-20 ≈ 91%** (Avg Top-K ≈ 74%).  

*Exact curves/ablations are in the report.*

---

## Tools & Libraries

- **TensorFlow/Keras**, **tf.data**, **NumPy**, **Matplotlib**  
- Pretrained CNNs: **Xception**, **InceptionV3**

---

## How to Run

> Ensure your dataset paths (TFRecords or folders) match the expected locations inside each script.

### 1) Install
```bash
pip install tensorflow numpy matplotlib
