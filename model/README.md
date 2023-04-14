# Evaluation
- [Evaluation](#evaluation)
  - [Losses](#losses)
    - [U^2-Net](#u2-net)
  - [Metrics](#metrics)
    - [PR curve](#pr-curve)
    - [F-measure](#f-measure)
    - [MAE](#mae)
    - [Weighted F-measure](#weighted-f-measure)
    - [S-measure](#s-measure)
    - [Relax Boundary F-measure](#relax-boundary-f-measure)

## Losses
### U^2-Net 
Training loss is defined as:

$$L = \sum_{m=1}^{M}w_{side}^{(m)}l_{side}^{(m)} + w_{fuse}l_{fuse}$$

where $l_{size}^{(m)}$ ($M$ = 6) is the loss of the side output saliency map $S_{side}^{(m)}$ and $l_{fuse}$ is the loss of the final fusion output saliency map $S_{fuse}$. $w_{side}^{(m)}$ and $w_{fuse}$ are the weights of each loss term. For each term $l$, using the standard binary cross-entropy to calculate the loss:

$$l = -\sum_{(r,c)}^{(H,W)}[P_{G(r,c)}log(P_{S(r,c)})+ (1-P_{G(r,c)}log(1-P_{S(r,c)})]$$

where $(r, c)$ is the pixel coordinates and $(H, W)$ is image size: height and width. $P_{G(r,c)}$ and $P_{S(r,c)}$ denote the pixel values of the GT and the predicted saliency probability map, respectively.

---
## Metrics
The outputs of the deep salient object methods are usually probability maps that have the same spatial resolution with the input images. Each pixel of the predicted saliency maps has a value within the range of 0 and 1 (or [0, 255]). The ground truth are usually *binary* masks, in which each pixel is either 0 or 1 (or 0 and 255) where 0 indicates the background pixels and 1 indicates the foreground salient object pixels.

### PR curve
It plotted based on a set of precision-recall pairs. Given a predicted saliency probability map, its precision and recall scores are computed by comparing its thresholded binary mask against the GT mask. The precision and recall of a dataset are computed by averaging the  precision and recall scores of those saliency maps. By varying the thresholds from 0 to 1, we can obtain a set of average precision-recall pairs of the dataset.
* $\text{Precision}  = \frac{\text{TP}}{\text{TP} + \text{FP}}$ (describes the purity of positive detections relative to the GT)
* $\text{Recall}  = \frac{\text{TP}}{\text{TP} + \text{FN}}$    (describes the completeness of positive predictions relative to the GT)



### F-measure 
$F_{\beta}$ is used to comprehensively evaluate both precision and recall as:

$$F_{\beta} = \frac{(1+\beta^{2}) \text{ . Precision} \text{ . Recall}}{\beta^2 \text{ . Precision + Recall}}$$

Default $\beta$ = 0.3 and maximum ($maxF_{\beta}$) for better.

### MAE 
Mean Absolute Error which denotes the average per-pixel difference between a predicted saliency map and its ground truth mask. It is defined as:
$$MAE = \frac{1}{H \text{ x } W} \sum_{r=1}^{H} \sum_{c=1}^{W} |P(r,c) - G(r,c)|$$

where $P$ and $G$ are the probability map of the salient object detection and the corresponding ground truth respectively, $(H, W )$ and $(r, c)$ are the (height, width) and the pixel coordinates.

### Weighted F-measure  
$F_{\beta}{w}$ is utilized as a complementary measure to $maxF_{\beta}$ for overcoming the possible unfair comparison caused by “interpolation flaw, dependency flaw and equal-importance flaw”. It is defined as:

$$ F_{\beta}^{w} = (1+\beta^2) \frac{Precision^{w} \text{ . } Recall^{w}}{\beta^2 \text{ . } Precision^{w} \text{ + } Recall^{w}} $$

### S-measure 
$S_m$ is used to evaluate the structure similarity of the predicted non-binary saliency map and the GT. The S-measure is defined as the weighted sum of region-aware $S_r$ and object-aware $S_o$ structural similarity:

$$S = (1 - \alpha)S_r + \alpha S_o$$

Default $\alpha$ = 0.5

### Relax Boundary F-measure
$relax F_{\beta}^{b}$ is utilized to quantitatively evaluate boundaries’ quality of the predicted saliency maps. 

Given a saliency probability map $P \in$ [0, 1], its binary mask $P_{bw}$ is obtained by a simple thresholding operation (threshold is set to 0.5). Then, the $XOR(P_{bw}, P_{erd})$ operation is conducted to obtain its one pixel wide boundary, where $P_{erd}$ denotes the eroded binary mask of $P_{bw}$. The boundaries of ground truth mask are obtained in the same way. The computation of relaxed boundary F-measure $relax F_{\beta}^{b}$ is similar to equation of F-measure $F_{\beta}$. 

The difference is that $relax Precision^{b}$ and $relax Recall^{b}$ other than $Precision$ and $Recall$ are used in equation F-measure $F_{\beta}$. 


* The definition of relaxed boundary precision ($relax Precision^{b}$) is the fraction of predicted boundary pixels within a range of ρ pixels from ground truth boundary pixels. 
* The relaxed boundary recall ($relax Recall^{b}$) is defined as the fraction of GT boundary pixels that are within ρ pixels of predicted boundary pixels.

Default ρ = 3. 
