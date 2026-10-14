/* ParkinSense — Web Frontend Client Application Logic */

document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = window.location.origin;

  // UI Elements
  const dropzone = document.getElementById('dropzone');
  const videoInput = document.getElementById('videoInput');
  const previewContainer = document.getElementById('previewContainer');
  const videoPreview = document.getElementById('videoPreview');
  const btnRemoveVideo = document.getElementById('btnRemoveVideo');
  const btnAnalyze = document.getElementById('btnAnalyze');
  const uploadForm = document.getElementById('uploadForm');
  const modeSelect = document.getElementById('modeSelect');
  const serviceStatusBadge = document.getElementById('serviceStatusBadge');
  const serviceStatusText = document.getElementById('serviceStatusText');

  // Dashboard state containers
  const placeholderState = document.getElementById('placeholderState');
  const loadingState = document.getElementById('loadingState');
  const errorState = document.getElementById('errorState');
  const resultsDashboard = document.getElementById('resultsDashboard');

  // Loading elements
  const loadingStageTitle = document.getElementById('loadingStageTitle');
  const loadingStageDesc = document.getElementById('loadingStageDesc');

  // Results elements
  const pdDiagnosis = document.getElementById('pdDiagnosis');
  const pdProbability = document.getElementById('pdProbability');
  const riskBadge = document.getElementById('riskBadge');
  const updrsScore = document.getElementById('updrsScore');
  const updrsDesc = document.getElementById('updrsDesc');
  const indicatorsList = document.getElementById('indicatorsList');
  const metaVideo = document.getElementById('metaVideo');
  const metaFrames = document.getElementById('metaFrames');
  const metaMode = document.getElementById('metaMode');

  let selectedFile = null;

  // 1. Initial Health Check
  checkBackendHealth();

  async function checkBackendHealth() {
    try {
      const resp = await fetch(`${API_BASE}/health`);
      if (resp.ok) {
        const data = await resp.json();
        const mode = data.pose_service ? data.pose_service.mode : 'unknown';
        serviceStatusText.textContent = `Backend Ready (${mode.toUpperCase()} Mode)`;
        serviceStatusBadge.querySelector('.status-dot').classList.remove('offline');
      } else {
        throw new Error('Backend HTTP error');
      }
    } catch (err) {
      serviceStatusText.textContent = 'Backend Offline';
      serviceStatusBadge.querySelector('.status-dot').classList.add('offline');
    }
  }

  // 2. File Selection & Drag Drop
  videoInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'var(--primary)';
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.style.borderColor = 'rgba(14, 165, 233, 0.4)';
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(14, 165, 233, 0.4)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    const url = URL.createObjectURL(file);
    videoPreview.src = url;
    dropzone.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    btnAnalyze.disabled = false;
  }

  btnRemoveVideo.addEventListener('click', () => {
    selectedFile = null;
    videoPreview.src = '';
    videoInput.value = '';
    dropzone.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    btnAnalyze.disabled = true;
  });

  // 3. Form Submission & Predict API Call
  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    // Show Loading state
    showState('loading');
    loadingStageTitle.textContent = 'Processing 4D-Humans Pose & Gait Features...';
    loadingStageDesc.textContent = `Mode: ${modeSelect.value.toUpperCase()} — Extracting SMPL rotation matrices & evaluating Random Forest models.`;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('mode', modeSelect.value);

    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok || data.status === 'error') {
        throw new Error(data.message || 'Prediction failed');
      }

      // Render Results
      renderResults(data);
      showState('results');

    } catch (err) {
      console.error('Prediction error:', err);
      document.getElementById('errorMessage').textContent = err.message;
      showState('error');
    }
  });

  document.getElementById('btnRetry').addEventListener('click', () => {
    showState('placeholder');
  });

  // Render Data into Dashboard
  function renderResults(data) {
    const pd = data.parkinsons_detection;
    const sev = data.severity_estimation;
    const exp = data.explainability;

    // Parkinson's Detection Card
    pdDiagnosis.textContent = pd.diagnosis;
    pdProbability.textContent = `${pd.probability}%`;

    riskBadge.textContent = `${pd.risk_level} Risk`;
    riskBadge.className = `badge risk-badge ${pd.risk_level.toLowerCase()}`;

    if (pd.label === 1) {
      pdDiagnosis.style.color = '#f43f5e';
    } else {
      pdDiagnosis.style.color = '#10b981';
    }

    // Severity Card
    updrsScore.textContent = `UPDRS Gait Score ${sev.updrs_gait_score}`;
    updrsDesc.textContent = sev.description;

    // Explainability Indicators
    indicatorsList.innerHTML = '';
    const indicators = exp.important_gait_indicators || [];

    if (indicators.length === 0) {
      indicatorsList.innerHTML = '<li class="indicator-item">No specific gait anomalies detected.</li>';
    } else {
      indicators.forEach((item) => {
        const li = document.createElement('li');
        li.className = 'indicator-item';
        li.innerHTML = `
          <div>
            <div class="indicator-title">${item.indicator}</div>
            <div class="indicator-sub">Anatomical Region: ${item.joint_name} (${item.feature})</div>
          </div>
          <div class="importance-pill">Impact: ${item.importance}</div>
        `;
        indicatorsList.appendChild(li);
      });
    }

    // Metadata
    metaVideo.textContent = `Video: ${data.video_file}`;
    metaFrames.textContent = `${data.num_frames} frames @ ${data.fps} fps`;
    metaMode.textContent = `Pose Mode: ${data.pose_service_mode.toUpperCase()}`;
  }

  function showState(state) {
    placeholderState.classList.add('hidden');
    loadingState.classList.add('hidden');
    errorState.classList.add('hidden');
    resultsDashboard.classList.add('hidden');

    if (state === 'placeholder') placeholderState.classList.remove('hidden');
    if (state === 'loading') loadingState.classList.remove('hidden');
    if (state === 'error') errorState.classList.remove('hidden');
    if (state === 'results') resultsDashboard.classList.remove('hidden');
  }
});
