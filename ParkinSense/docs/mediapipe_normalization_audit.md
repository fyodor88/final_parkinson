# MediaPipe Pipeline Mathematical Audit & Baseline Analysis

## 1. Landmark Representation & Coordinate Systems
- **Landmark Source**: MediaPipe Pose 3D `world_landmarks` (expressed in real-world 3D meters).
- **Native Axes**:
  - $+X$: Points towards anatomical right.
  - $+Y$: Points downwards (gravity direction).
  - $+Z$: Points forward (towards camera / screen).
- **Landmark Confidence**: Per-landmark `visibility` score $\in [0, 1]$.
- **Current Visibility Threshold**: `_VISIBILITY_THRESHOLD = 0.30`. Landmarks below 0.30 return `None` and fall back to identity rotation matrices ($I_{3 \times 3}$).

## 2. Joint Mapping & Rotation Matrix Construction
- **Joint Representation**: 24-joint SMPL layout ($1$ global orientation + $23$ body pose joints).
- **Global Orientation (Joint 0 - Pelvis)**:
  - Lateral vector: $x_{\text{raw}} = \text{L\_Hip} - \text{R\_Hip}$
  - Superior vector: $y_{\text{raw}} = \text{Shoulder\_Mid} - \text{Pelvis\_Mid}$
  - Rotation matrix built via Gram-Schmidt orthogonalisation: $R_{\text{pelvis}} = [x \mid y \mid z]$ with $\det(R_{\text{pelvis}}) = +1$.
- **Bone Rotations (Joints 1–23)**:
  - Primary bone direction vector: $y_{\text{axis}} = \text{normalize}(\text{joint}_{\text{child}} - \text{joint}_{\text{parent}})$
  - Secondary lateral hint vector: $x_{\text{hint}}$ (e.g. shoulder or hip span vector).
  - Built via Gram-Schmidt orthogonalisation enforcing $R^T R = I$ and $\det(R) = +1$.

## 3. Feature Extraction (290 Features)
- **Axis-Angle Conversion**: For each frame, 24 rotation matrices ($3 \times 3$) are converted via Rodrigues formula to $72$ axis-angle rotation vectors ($24 \times 3$).
- **Summary Statistics**: Across all $N$ frames, $4$ statistics are computed per channel:
  $$\text{pose}_i \rightarrow [\mu(\text{pose}_i), \sigma(\text{pose}_i), \min(\text{pose}_i), \max(\text{pose}_i)] \quad \text{for } i \in \{0, \dots, 71\}$$
  yielding $72 \times 4 = 288$ pose features.
- **Metadata Features**: `fps` ($1$) and `num_frames` ($1$) are prepended, yielding **exactly 290 features**.

## 4. Current Domain Mismatch Vulnerabilities
1. **Global Translation**: Coordinates vary if person walks towards camera or across screen.
2. **Subject Scale**: Heights/limb lengths directly scale raw landmark distance vectors.
3. **Camera Orientation**: Pitch/yaw angle of camera alters global world axes $X, Y, Z$.
4. **Duration & Frame Rate**: `num_frames` and `fps` enter directly into RF decision nodes trained on 60 FPS / 150-frame sequences.
