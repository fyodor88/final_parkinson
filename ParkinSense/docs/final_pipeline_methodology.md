# ParkinSense Domain-Normalized MediaPipe Pipeline Methodology

## 1. Overview & Pipeline Flowchart
The ParkinSense domain-normalized gait classification pipeline processes RGB video sequences to estimate Parkinson's Disease (PD) probability using physical domain normalization.

```
RGB Video
   │
   ▼
MediaPipe Pose (33 3D World Landmarks)
   │
   ▼
Domain Normalization Layer (Pelvis Centering + Body Scale + Canonical Frame Alignment)
   │
   ▼
MediaPipe-to-SMPL Rotation Estimator (24 3x3 Rotation Matrices)
   │
   ▼
SMPL Feature Extractor (290 Summary Statistics: mean, std, min, max across 72 channels + fps/num_frames)
   │
   ▼
Normalized Random Forest Classifier (`models/pd_detector_mediapipe_normalized.pkl`)
   │
   ▼
Camera-Level P(PD) ──► Subject-Level Probability Aggregation ──► Subject PD Diagnosis
```

---

## 2. Mathematical Formulation

1. **Landmark Filtering**:
   Landmark coordinates $\mathbf{p}_k(t) \in \mathbb{R}^3$ are retained if visibility $v_k(t) \ge 0.50$.
2. **Pelvis Centering**:
   $$\mathbf{p}'_k(t) = \mathbf{p}_k(t) - \frac{\mathbf{p}_{\text{L\_hip}}(t) + \mathbf{p}_{\text{R\_hip}}(t)}{2}$$
3. **Body Scale Normalization**:
   $$\mathbf{p}''_k(t) = \frac{\mathbf{p}'_k(t)}{\|\mathbf{p}_{\text{neck}}(t) - \mathbf{p}_{\text{pelvis}}(t)\|}$$
4. **Canonical Frame Alignment**:
   $$\mathbf{p}'''_k(t) = R_{\text{canonical}}(t)^T \mathbf{p}''_k(t)$$
   where $R_{\text{canonical}}(t) = [\mathbf{x}_{\text{shoulder}}, \mathbf{y}_{\text{torso}}, \mathbf{z}_{\text{normal}}] \in SO(3)$.
5. **SMPL Axis-Angle Extraction**:
   For each joint $i \in \{0 \ldots 23\}$, $R_i(t) \in SO(3)$ is estimated via bone directions and converted to axis-angle $\mathbf{r}_i(t) = \text{Log}(R_i(t)) \in \mathbb{R}^3$.
6. **Feature Matrix**:
   Statistical summary (mean, std, min, max) across $T$ frames yields 288 pose features + `fps` + `num_frames` = 290 features.
