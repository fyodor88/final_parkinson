# ParkinSense 🧠🚶

> **AI-Powered Parkinson's Disease Detection & UPDRS Severity Estimation from Walking Videos**
> Complete Web Application Integrating MediaPipe & 4D-Humans Gait Extraction

---

## 📋 Integrated System Architecture

ParkinSense integrates two selectable gait-extraction pipelines converging onto a single 290-dimensional feature contract, centralized machine learning prediction engine, and clinical results dashboard:

```
                      Browser / React Web Frontend
                      (Selectable Pipeline UI Cards)
                                    │
                                    ▼ HTTP Video Upload & pipeline param
                        Flask Backend API Server
                        (backend/app.py :5001)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          Pipeline: MediaPipe                 Pipeline: 4D-Humans
   (Lightweight Pose Extraction)       (3D Mesh Surface Reconstruction)
   • 33 3D world landmarks             • HMR2 + PHALP tracking
   • Fast, local CPU / Mac             • Remote Ubuntu GPU (RTX 2050 via SSH)
   • Joint angles & keyframes          • 24-joint SMPL rotation sequence
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                                    ▼
                        Feature Adapter & Validator
                      (Strict 290-Dim Feature Contract)
             [fps, num_frames, 72 channels × (mean, std, min, max)]
                                    │
                                    ▼
                     Centralized Prediction Service
                      (backend/prediction.py)
            • Parkinson's Classifier: models/pd_detector.pkl
            • Severity Estimator:     models/random_forest.pkl
            • Explainability:         ModelExplainer (SMPL joints)
                                    │
                                    ▼
                      Unified Result JSON Schema
                                    │
                                    ▼
                    Interactive Results Dashboard
      (Classification %, UPDRS Score, Timeseries Charts, Keyframes)
```

---

## 🎯 Selectable Extraction Pipelines

| Feature | MediaPipe Pose Pipeline | 4D-Humans Mesh Pipeline |
| :--- | :--- | :--- |
| **Model Type** | 33 3D landmark geometric tracking | 3D human body surface mesh (HMR2 / SMPL) |
| **Hardware** | Runs locally on CPU / MacBook Air M1 | Requires NVIDIA GPU (RTX 2050 via SSH/local) |
| **Execution Time** | ~5–10 seconds | ~30–60 seconds (GPU accelerated) |
| **Remote Server** | Not required (100% offline ready) | Configured via `.env` SSH settings |
| **Feature Output** | Standard 290-dim feature vector | Standard 290-dim feature vector |
| **Prediction Model** | Shared `pd_detector.pkl` & `random_forest.pkl` | Shared `pd_detector.pkl` & `random_forest.pkl` |
| **Dashboard** | Shared clinical results dashboard | Shared clinical results dashboard |

---

## 📁 Repository Structure

```
ParkinSense/
├── frontend/                        ← React + Vite Web Frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx           ← Navigation & service status indicators
│   │   │   ├── UploadZone.jsx       ← Dual pipeline selectable cards & uploader
│   │   │   ├── ProcessingScreen.jsx ← Real-time progress bar & stage tracking
│   │   │   └── ResultsDashboard.jsx ← Unified results dashboard with timeseries & charts
│   │   ├── App.jsx                  ← Main state orchestrator
│   │   └── main.jsx
│   ├── dist/                        ← Production SPA build (served by Flask)
│   └── package.json
│
├── backend/                         ← Unified Backend API
│   ├── app.py                       ← Flask server, endpoints & background worker
│   ├── extractors.py                ← MediaPipe & 4D-Humans gait extractor abstractions
│   ├── adapter.py                   ← Feature adapter interface
│   ├── pipeline.py                  ← Central orchestrator & pipeline dispatcher
│   ├── prediction.py                ← Centralized prediction engine & explainability
│   └── fourd_humans.py              ← SSH GPU & local 4D-Humans execution client
│
├── src/                             ← Core Computer Vision & ML Modules
│   ├── features/
│   │   ├── feature_contract.py      ← Strict 290-dim contract definition & validation
│   │   ├── feature_adapter.py       ← Feature vector alignment & validation
│   │   ├── smpl_feature_extractor.py← 290-dim SMPL axis-angle summary statistics
│   │   └── explainability.py        ← ModelExplainer mapping to 24 SMPL joints
│   ├── pose/
│   │   ├── mediapipe_pose.py        ← MediaPipe pose landmark extraction
│   │   ├── mediapipe_smpl.py        ← MediaPipe to SMPL rotation mapping
│   │   └── pose_client.py           ← Pose service client
│   └── inference/
│       └── mediapipe_predict.py     ← MediaPipe standalone predictor
│
├── 4D-Humans/                       ← 4D-Humans / HMR2 Codebase
│   ├── track.py                     ← PHALP tracking & SMPL parameter generation
│   └── hmr2/                        ← Human Mesh Recovery 2 architecture
│
├── models/                          ← Locked Trained Production Models
│   ├── pd_detector.pkl              ← Binary Parkinson's detector (290 features)
│   └── random_forest.pkl            ← UPDRS gait severity estimator (290 features)
│
├── tests/                           ← Pytest Test Suite
│   ├── test_unified_pipeline.py     ← Pipeline selection, contract & API tests
│   ├── test_mediapipe_pipeline.py   ← MediaPipe regression tests
│   └── test_prediction_mapping.py   ← Model score mapping tests
│
├── .env.example                     ← Environment configuration template
├── requirements.txt                 ← Python package dependencies
└── README.md                        ← Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

Python 3.11 is required (due to MediaPipe legacy solutions API compatibility):

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the template:

```bash
cp .env.example .env
```

To enable 4D-Humans via the remote GPU server, configure your remote SSH details in `.env`:
```ini
POSE_SERVICE_MODE=ssh
FOURD_HUMANS_SSH_HOST=<GPU_SERVER_IP>
FOURD_HUMANS_SSH_USER=pranav
FOURD_HUMANS_SSH_PORT=22
FOURD_HUMANS_REMOTE_DIR=~/ParkinSense/4D-Humans
```
*(If the GPU server is offline, MediaPipe runs locally without any configuration. For offline 4D-Humans testing without GPU, set `POSE_SERVICE_MODE=mock`)*

### 3. Starting the Backend Server

Run the Flask server:

```bash
python backend/app.py
```
*Server starts at `http://localhost:5001`. Serves both the REST API and the compiled React frontend.*

### 4. Running the Frontend (Development)

If working on the frontend source code:

```bash
cd frontend
npm install
npm run dev
```
*Development server starts at `http://localhost:5173` with hot-reloading.*

To build for production:
```bash
npm run build
```
*(The production bundle in `frontend/dist` is automatically served by Flask).*

---

## 🧪 Testing & Quality Assurance

Run the full automated test suite:

```bash
pytest tests/ -v
```

All **39 tests** validate:
- ✅ MediaPipe and 4D-Humans pipeline selection & dispatch
- ✅ 290-dimensional feature contract validation & NaN/Inf rejection
- ✅ FeatureAdapter reordering, shape validation, and contract conformation
- ✅ Centralized `PredictionEngine` scoring & clinical explainability
- ✅ Consistent result schemas across both pipelines
- ✅ 4D-Humans GPU unavailable error handling (graceful friendly error, never fakes inference)
- ✅ Flask API `/api/health`, `/api/predict`, and `/api/status` endpoints
- ✅ MediaPipe regression behavior & landmark quality validation
- ✅ SMPLFeatureExtractor input validation (empty df, negative fps, NaN/Inf rejection)
- ✅ Model artifact provenance (`models/pd_detector.pkl` and `models/random_forest.pkl`)
- ✅ Candidate model metadata and independent external evaluation isolation

---

## 🔬 ML Research & External Benchmark Evaluation

A standalone, data-leakage-free evaluation script assesses models on the independent TULIP benchmark:

```bash
python scripts/evaluate_tulip_external.py
```

### Key Research Findings:
- **Internal Cross-Validation (DNE 5-Fold StratifiedGroupKFold, 58 subjects):** Models achieve 85%–91% accuracy and AUC 0.94–0.97.
- **Independent External Benchmark (TULIP, 14 subjects, 50 videos):** Models trained strictly on DNE exhibit severe out-of-domain degradation due to the representation gap between 3D SMPL mesh fitting (DNE) and 2D-to-3D heuristic pose estimations (MediaPipe).
- **Production Guardrail:** `models/pd_detector.pkl` is safely retained as the production classifier because no candidate passes independent external validation without domain failure. Detailed research findings are cataloged in `data/processed/improved_model/final_completion_report.md`.

---

## 🛠️ Troubleshooting & FAQ

- **Remote GPU is offline (`GPUServiceUnavailableError`):** The 4D-Humans GPU server is optional. Local MediaPipe gait analysis runs 100% offline on macOS CPU/M1 without external dependencies.
- **Frontend build:** Ensure Node.js is in PATH (`export PATH="$PATH:/usr/local/bin:/opt/homebrew/bin"`) before running `npm --prefix frontend run build`.
- **Port conflicts:** The backend runs on port 5001 by default (`http://localhost:5001`) to avoid conflicts with macOS AirPlay Receiver on port 5000.

