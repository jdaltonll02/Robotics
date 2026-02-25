
# Perception Loss Functions

---

## Semantic Segmentation: Cross-Entropy Loss

Given an input image $x$ and ground truth label map $y$, the model predicts a probability distribution $\hat{y}$ over $C$ classes for each pixel $i$:

$$
\hat{y}_{i,c} = P(y_i = c \mid x; \theta)
$$

where $\theta$ are the model parameters.

The one-hot ground truth for pixel $i$ is $y_{i,c} \in \{0,1\}$.

The pixelwise cross-entropy loss is:

$$
\mathcal{L}_{CE}^{(i)} = - \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}
$$

The total loss over all $N$ pixels is:

$$
\mathcal{L}_{CE} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}_{CE}^{(i)} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log \hat{y}_{i,c}
$$

**Derivation:**
The cross-entropy loss measures the Kullback-Leibler divergence between the true and predicted distributions for each pixel. For a one-hot label $y_{i,c}$, only the true class contributes to the sum.

---

## Depth Estimation: L1 Loss

Given ground truth depth $y_i$ and predicted depth $\hat{y}_i$ for each pixel $i$:

$$
\mathcal{L}_{L1} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$

**Derivation:**
The L1 loss penalizes the absolute difference between predicted and true depth, encouraging robustness to outliers compared to L2 loss.

---

## General Modeling Steps

1. **Data Preparation:**
	- Input images $x$ and ground truth labels $y$ are collected from simulation or real-world datasets.
2. **Model Forward Pass:**
	- The neural network $f_\theta(x)$ outputs $\hat{y}$ (class probabilities or depth values).
3. **Loss Computation:**
	- For segmentation: use $\mathcal{L}_{CE}$ as above.
	- For depth: use $\mathcal{L}_{L1}$ as above.
4. **Backpropagation:**
	- Compute gradients $\nabla_\theta \mathcal{L}$ and update parameters $\theta$ using an optimizer (e.g., Adam).
5. **Evaluation:**
	- Metrics such as pixel accuracy, mean IoU (for segmentation), or RMSE (for depth) are computed on validation data.

---

For further details, see code references in `perception/train.py` and `perception/model.py`.
