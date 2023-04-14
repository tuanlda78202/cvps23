# Segmentation Metrics

### Pixel accuracy

Pixel accuracy measures how many pixels are predicted correctly. In `binary cases`:  

$\text{PixelAcc} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$  

In `multi-class cases`, it can be calculated from Confusion matrix, by dividing the sum of diagonal elements (True Positives for all classes) with the total number of pixels.

---

### Sørensen–Dice coefficient

Dice evaluates the overlap rate of prediction results and ground truth; equals to F1 score in definition.  

$\text{Dice}  = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$  

---

### Precision

Describes the purity of our positive detections relative to the ground truth.  

$\text{Precision}  = \frac{\text{TP}}{\text{TP} + \text{FP}}$  

---

### Recall

Describes the completeness of our positive predictions relative to the ground truth.  

$\text{Recall}  = \frac{\text{TP}}{\text{TP} + \text{FN}}$  

---

### Specificity

Also known as True Negative Rate (TNR)  

$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$  