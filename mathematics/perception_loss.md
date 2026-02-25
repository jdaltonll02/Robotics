# Perception Loss Functions

## Semantic Segmentation

Cross-entropy loss:
$$
\mathcal{L}_{CE} = -\sum_{c} y_c \log \hat{y}_c
$$

## Depth Estimation

L1 loss:
$$
\mathcal{L}_{L1} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|
$$
