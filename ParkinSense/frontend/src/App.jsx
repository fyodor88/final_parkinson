import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadZone from './components/UploadZone';
import ProcessingScreen from './components/ProcessingScreen';
import ResultsDashboard from './components/ResultsDashboard';

export default function App() {
  const [step, setStep] = useState('upload'); // 'upload' | 'processing' | 'results'
  const [jobId, setJobId] = useState(null);
  const [videoFile, setVideoFile] = useState(null);
  const [resultData, setResultData] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  const handleStartAnalysis = async (file, pipeline = 'mediapipe', submode = '') => {
    setErrorMsg('');
    setJobId(null);
    setResultData(null);
    setVideoFile(file);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('pipeline', pipeline);
    if (submode) {
      formData.append('mode', submode);
    }

    try {
      setStep('processing');
      const params = new URLSearchParams();
      params.append('pipeline', pipeline);
      if (submode) {
        params.append('mode', submode);
      }
      const response = await fetch(`/api/predict?${params.toString()}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.message || `Server error (HTTP ${response.status})`);
      }

      const data = await response.json();

      if (data.status === 'processing' && data.job_id) {
        setJobId(data.job_id);
      } else if (data.status === 'success') {
        // Synchronous response
        setResultData(data);
        setStep('results');
      } else {
        throw new Error(data.message || 'Failed to start video analysis.');
      }
    } catch (err) {
      console.error('Analysis start error:', err);
      setErrorMsg(err.message || 'Failed to upload video to backend API.');
      setStep('upload');
    }
  };

  const handleProcessingComplete = (result) => {
    setResultData(result);
    setStep('results');
  };

  const handleProcessingError = (errMsg) => {
    setErrorMsg(errMsg || 'Processing failed. Please try again.');
    setStep('upload');
  };

  const handleReset = () => {
    setStep('upload');
    setJobId(null);
    setVideoFile(null);
    setResultData(null);
    setErrorMsg('');
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col justify-between selection:bg-sky-100 selection:text-sky-900">
      {/* Top Header */}
      <div>
        <Navbar />

        {/* Main Content Area */}
        <main className="pb-16">
          {step === 'upload' && (
            <UploadZone
              onStartAnalysis={handleStartAnalysis}
              errorMsg={errorMsg}
            />
          )}

          {step === 'processing' && jobId && (
            <ProcessingScreen
              jobId={jobId}
              videoName={videoFile?.name || 'walking.mp4'}
              onComplete={handleProcessingComplete}
              onError={handleProcessingError}
            />
          )}

          {step === 'results' && resultData && (
            <ResultsDashboard
              result={resultData}
              onReset={handleReset}
            />
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p>© {new Date().getFullYear()} ParkiSense Research Group • B.Tech Computer Science & Engineering</p>
          <p className="text-slate-400">
            Powered by MediaPipe Pose · SMPL-like Kinematics · Random Forest ML
          </p>
        </div>
      </footer>
    </div>
  );
}
