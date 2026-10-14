import React, { useEffect, useState } from 'react';
import { CheckCircle2, Loader2, AlertCircle, RefreshCw, Cpu, Activity, Database, FileText } from 'lucide-react';

export default function ProcessingScreen({ jobId, videoName, onComplete, onError }) {
  const [jobStatus, setJobStatus] = useState({
    status: 'processing',
    step_index: 1,
    total_steps: 4,
    progress_percent: 15,
    current_step: 'Initializing AI pipeline...',
    error: null,
  });

  const STEPS = [
    { idx: 1, title: 'Video Uploaded & Validated', desc: 'Walking video prepared for pose extraction', icon: FileText },
    { idx: 2, title: 'MediaPipe Pose Extraction', desc: 'Detecting 3D world landmarks for all body joints', icon: Cpu },
    { idx: 3, title: '290 Gait Feature Extraction', desc: 'Computing kinematic variance, joint rotations & SMPL-like representation', icon: Activity },
    { idx: 4, title: 'AI Prediction & Explainability', desc: 'Evaluating the gait classifier and preparing observed movement measurements', icon: Database },
  ];

  useEffect(() => {
    let timer = null;

    const pollStatus = async () => {
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (res.ok) {
          const data = await res.json();
          setJobStatus(data);

          if (data.status === 'completed' && data.result) {
            clearInterval(timer);
            setTimeout(() => {
              onComplete(data.result);
            }, 600);
          } else if (data.status === 'failed') {
            clearInterval(timer);
            onError(data.error || 'Pipeline execution failed.');
          }
        }
      } catch (err) {
        console.error('Error polling status:', err);
      }
    };

    pollStatus();
    timer = setInterval(pollStatus, 1500);

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobId, onComplete, onError]);

  const currentIdx = jobStatus.step_index || 1;
  // Only show real progress from server; do not fake a percentage
  const progressPercent = jobStatus.progress_percent || null;

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <div className="medical-card p-8 bg-white shadow-sm border border-slate-200 text-center">
        
        {/* Header Title */}
        <div className="w-12 h-12 rounded-full bg-sky-50 border border-sky-200 flex items-center justify-center mx-auto text-sky-600 mb-4">
          <Loader2 className="w-6 h-6 animate-spin" />
        </div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900">
          Analyzing Walking Video
        </h2>
        <p className="text-xs text-slate-500 mt-1">
          Job ID: <span className="font-mono text-slate-700 font-medium">{jobId}</span> • Video: <span className="font-medium text-slate-700">{videoName}</span>
        </p>

        {/* Progress Bar */}
        <div className="mt-6 mb-8 max-w-md mx-auto">
          <div className="flex justify-between text-xs text-slate-500 font-medium mb-1.5">
            <span>Overall Progress</span>
            <span>{progressPercent !== null ? `${progressPercent}%` : 'Processing...'}</span>
          </div>
          <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden border border-slate-200">
            <div
              className="h-full bg-sky-600 transition-all duration-500 ease-out rounded-full"
              style={{ width: progressPercent !== null ? `${progressPercent}%` : '100%',
                       animation: progressPercent === null ? 'pulse 1.5s ease-in-out infinite' : 'none' }}
            ></div>
          </div>
          <p className="text-xs font-medium text-sky-700 mt-2 italic">
            "{jobStatus.current_step || 'Processing video...'}"
          </p>
        </div>

        {/* Step-by-Step Vertical List */}
        <div className="space-y-4 text-left max-w-lg mx-auto pt-4 border-t border-slate-100">
          {STEPS.map((step) => {
            const isFinished = step.idx < currentIdx || jobStatus.status === 'completed';
            const isCurrent = step.idx === currentIdx && jobStatus.status === 'processing';
            const isPending = step.idx > currentIdx && jobStatus.status === 'processing';
            const StepIcon = step.icon;

            return (
              <div
                key={step.idx}
                className={`flex items-start space-x-3.5 p-3 rounded-lg transition-all ${
                  isCurrent
                    ? 'bg-sky-50/70 border border-sky-200'
                    : isFinished
                    ? 'bg-slate-50/60 border border-transparent'
                    : 'opacity-50'
                }`}
              >
                <div className="mt-0.5 flex-shrink-0">
                  {isFinished ? (
                    <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                  ) : isCurrent ? (
                    <Loader2 className="w-5 h-5 text-sky-600 animate-spin" />
                  ) : (
                    <div className="w-5 h-5 rounded-full border border-slate-300 flex items-center justify-center text-[10px] font-semibold text-slate-400">
                      {step.idx}
                    </div>
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-semibold ${isCurrent ? 'text-sky-900' : isFinished ? 'text-slate-800' : 'text-slate-500'}`}>
                    {step.title}
                  </p>
                  <p className="text-xs text-slate-500 truncate mt-0.5">
                    {step.desc}
                  </p>
                </div>
              </div>
            );
          })}
        </div>

        {/* Failure Error Display */}
        {jobStatus.status === 'failed' && (
          <div className="mt-6 p-4 rounded-lg bg-rose-50 border border-rose-200 text-left">
            <div className="flex items-start space-x-3 text-rose-800">
              <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-sm">Pipeline Execution Error</p>
                <p className="text-xs mt-1 text-rose-700">{jobStatus.error || 'An error occurred during video processing.'}</p>
                <button
                  onClick={() => onError(jobStatus.error)}
                  className="mt-3 inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-rose-600 text-white text-xs font-semibold hover:bg-rose-700"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Try Again</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
