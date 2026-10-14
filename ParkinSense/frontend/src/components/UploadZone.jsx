import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileVideo,
  Trash2,
  ArrowRight,
  Info,
  AlertCircle,
  Cpu,
  Layers,
  CheckCircle2,
  Sparkles,
  Server,
  Zap,
} from 'lucide-react';

export default function UploadZone({ onStartAnalysis, errorMsg }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [pipeline, setPipeline] = useState('mediapipe'); // 'mediapipe' | 'fourd_humans'
  const [submode, setSubmode] = useState(''); // '' (default real) | 'mock'
  const [validationError, setValidationError] = useState('');
  const fileInputRef = useRef(null);

  const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/mkv'];
  const MAX_SIZE_MB = 150;

  const handleFileSelect = (file) => {
    setValidationError('');
    if (!file) return;

    // Validate size
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_SIZE_MB) {
      setValidationError(`File size (${sizeMb.toFixed(1)} MB) exceeds limit of ${MAX_SIZE_MB} MB.`);
      return;
    }

    // Validate extension/type
    const ext = file.name.split('.').pop().toLowerCase();
    const validExts = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
    if (!validExts.includes(ext)) {
      setValidationError(`Unsupported file format .${ext}. Please upload MP4, MOV, AVI, MKV, or WEBM.`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setValidationError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (selectedFile) {
      onStartAnalysis(selectedFile, pipeline, submode);
    }
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      {/* Page Header */}
      <div className="text-center mb-8">
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900">
          Upload Walking Video
        </h2>
        <p className="mt-2 text-sm text-slate-600 max-w-xl mx-auto">
          ParkinSense extracts 3D gait kinematics to compute objective motor indicators and estimate Parkinson&apos;s Disease classification via centralized trained models.
        </p>
      </div>

      {/* Main Form Card */}
      <form onSubmit={handleSubmit} className="medical-card p-6 sm:p-8 bg-white shadow-sm border border-slate-200">
        
        {/* Error Banners */}
        {(validationError || errorMsg) && (
          <div className="mb-6 p-4 rounded-lg bg-rose-50 border border-rose-200 flex items-start space-x-3 text-rose-800">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold">Analysis Error</p>
              <p className="mt-0.5">{validationError || errorMsg}</p>
            </div>
          </div>
        )}

        {/* Pipeline Selection Section */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <label className="block text-xs font-bold uppercase tracking-wider text-slate-700">
              Select Gait Extraction Pipeline
            </label>
            <span className="text-[11px] text-slate-500">
              Both converge on common 290-D prediction
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Card 1: MediaPipe */}
            <div
              onClick={() => setPipeline('mediapipe')}
              className={`relative rounded-xl p-4 cursor-pointer border-2 transition-all duration-200 ${
                pipeline === 'mediapipe'
                  ? 'border-sky-500 bg-sky-50/40 shadow-sm ring-1 ring-sky-500/20'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                      pipeline === 'mediapipe'
                        ? 'bg-sky-500 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    <Zap className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-slate-900">MediaPipe</h4>
                    <p className="text-[11px] text-slate-500">Pose-Based Extraction</p>
                  </div>
                </div>

                {pipeline === 'mediapipe' && (
                  <CheckCircle2 className="w-5 h-5 text-sky-600 flex-shrink-0" />
                )}
              </div>

              <p className="mt-3 text-xs text-slate-600 leading-relaxed">
                Fast, lightweight 33 3D world landmark tracking. Runs locally in seconds on your Mac/CPU without external GPU requirements.
              </p>

              <div className="mt-3 pt-2.5 border-t border-slate-200/80 flex items-center justify-between text-[11px]">
                <span className="inline-flex items-center text-slate-500">
                  <Cpu className="w-3.5 h-3.5 mr-1 text-slate-400" />
                  Local / CPU Ready
                </span>
                <span className="font-semibold text-sky-700 bg-sky-100/70 px-2 py-0.5 rounded-full text-[10px]">
                  Recommended
                </span>
              </div>
            </div>

            {/* Card 2: 4D-Humans */}
            <div
              onClick={() => setPipeline('fourd_humans')}
              className={`relative rounded-xl p-4 cursor-pointer border-2 transition-all duration-200 ${
                pipeline === 'fourd_humans'
                  ? 'border-purple-600 bg-purple-50/40 shadow-sm ring-1 ring-purple-600/20'
                  : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50/50'
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-2.5">
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                      pipeline === 'fourd_humans'
                        ? 'bg-purple-600 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-600'
                    }`}
                  >
                    <Layers className="w-5 h-5" />
                  </div>
                  <div>
                    <h4 className="text-sm font-bold text-slate-900">4D-Humans</h4>
                    <p className="text-[11px] text-slate-500">3D SMPL Mesh Extraction</p>
                  </div>
                </div>

                {pipeline === 'fourd_humans' && (
                  <CheckCircle2 className="w-5 h-5 text-purple-600 flex-shrink-0" />
                )}
              </div>

              <p className="mt-3 text-xs text-slate-600 leading-relaxed">
                Full 3D surface human mesh reconstruction via HMR2 + PHALP tracking. Requires an NVIDIA GPU server (RTX 2050 via SSH).
              </p>

              <div className="mt-3 pt-2.5 border-t border-slate-200/80 flex items-center justify-between text-[11px]">
                <span className="inline-flex items-center text-slate-500">
                  <Server className="w-3.5 h-3.5 mr-1 text-slate-400" />
                  GPU Server Required
                </span>
                <span className="font-semibold text-purple-700 bg-purple-100/70 px-2 py-0.5 rounded-full text-[10px]">
                  3D Kinematics
                </span>
              </div>
            </div>
          </div>

          {/* 4D-Humans Dev/Mock Option */}
          {pipeline === 'fourd_humans' && (
            <div className="mt-2.5 p-2.5 rounded-lg bg-slate-50 border border-slate-200 flex items-center justify-between text-xs text-slate-600">
              <span className="text-[11px]">
                Execution Mode: <strong>{submode === 'mock' ? 'Mock (Testing/Offline)' : 'Real Remote GPU (SSH)'}</strong>
              </span>
              <button
                type="button"
                onClick={() => setSubmode(submode === 'mock' ? '' : 'mock')}
                className="text-[11px] font-medium text-purple-700 hover:text-purple-800 underline"
              >
                {submode === 'mock' ? 'Switch to Real GPU' : 'Enable Mock Mode for Tests'}
              </button>
            </div>
          )}
        </div>

        {/* Dropzone or Selected File */}
        {!selectedFile ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-8 sm:p-12 text-center cursor-pointer transition-all duration-150 ${
              isDragging
                ? 'border-sky-500 bg-sky-50/50'
                : 'border-slate-300 hover:border-slate-400 bg-slate-50/50 hover:bg-slate-100/50'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,.mp4,.mov,.avi,.mkv,.webm"
              className="hidden"
              onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            />
            <div className="w-14 h-14 rounded-full bg-white border border-slate-200 shadow-sm flex items-center justify-center mx-auto text-sky-600 mb-4">
              <UploadCloud className="w-7 h-7" />
            </div>
            <h3 className="text-base font-semibold text-slate-800">
              Drag and drop your walking video here
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Supports MP4, MOV, AVI, MKV, or WEBM up to 150 MB
            </p>
            <div className="mt-5">
              <button
                type="button"
                className="inline-flex items-center px-4 py-2 text-xs font-semibold rounded-lg bg-white border border-slate-300 text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
              >
                Select Video File
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-200 gap-4">
              <div className="flex items-center space-x-3 overflow-hidden w-full sm:w-auto">
                <div className="w-10 h-10 rounded-lg bg-sky-100 border border-sky-200 flex items-center justify-center text-sky-700 flex-shrink-0">
                  <FileVideo className="w-5 h-5" />
                </div>
                <div className="truncate">
                  <p className="text-sm font-semibold text-slate-900 truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-slate-500">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || 'Video'}
                  </p>
                </div>
              </div>

              <button
                type="button"
                onClick={handleRemove}
                className="inline-flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium text-rose-700 bg-rose-50 hover:bg-rose-100 border border-rose-200 rounded-lg transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                <span>Remove Video</span>
              </button>
            </div>

            {/* Video Player Preview */}
            {previewUrl && (
              <div className="rounded-xl overflow-hidden bg-black border border-slate-200 max-h-72 flex justify-center">
                <video
                  src={previewUrl}
                  controls
                  className="max-h-72 w-auto object-contain"
                />
              </div>
            )}
          </div>
        )}

        {/* Submit Action */}
        <div className="mt-6 pt-6 border-t border-slate-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="text-xs text-slate-600">
            Selected Pipeline:{' '}
            <span className="font-semibold text-slate-900">
              {pipeline === 'mediapipe' ? 'MediaPipe (Fast Local Pose)' : '4D-Humans (3D SMPL Mesh)'}
            </span>
          </div>

          <button
            type="submit"
            disabled={!selectedFile}
            className={`w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-2.5 rounded-lg text-sm font-semibold text-white shadow-sm transition-all ${
              selectedFile
                ? pipeline === 'mediapipe'
                  ? 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 cursor-pointer'
                  : 'bg-purple-600 hover:bg-purple-700 active:bg-purple-800 cursor-pointer'
                : 'bg-slate-300 cursor-not-allowed'
            }`}
          >
            <span>Run {pipeline === 'mediapipe' ? 'MediaPipe' : '4D-Humans'} Analysis</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </form>

      {/* Clinical Disclaimer */}
      <div className="mt-8 text-center flex items-center justify-center space-x-2 text-xs text-slate-500">
        <Info className="w-4 h-4 text-slate-400 flex-shrink-0" />
        <span>
          ParkinSense is a research prototype intended for gait analysis research and educational purposes. It is not a clinical diagnostic tool and should not be used to diagnose or make medical decisions.
        </span>
      </div>
    </div>
  );
}
