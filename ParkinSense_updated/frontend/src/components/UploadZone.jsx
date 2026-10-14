import React, { useState, useRef } from 'react';
import { UploadCloud, FileVideo, Trash2, ArrowRight, Info, AlertCircle, Settings2 } from 'lucide-react';

export default function UploadZone({ onStartAnalysis, errorMsg }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [mode, setMode] = useState('local');
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
      onStartAnalysis(selectedFile, mode);
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
          ParkiSense extracts 3D SMPL pose dynamics and 290 clinical gait features to evaluate Parkinson's Disease likelihood and UPDRS motor severity.
        </p>
      </div>

      {/* Main Form Card */}
      <form onSubmit={handleSubmit} className="medical-card p-6 sm:p-8 bg-white shadow-sm border border-slate-200">
        
        {/* Error Banners */}
        {(validationError || errorMsg) && (
          <div className="mb-6 p-4 rounded-lg bg-rose-50 border border-rose-200 flex items-start space-x-3 text-rose-800">
            <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
            <div className="text-sm">
              <p className="font-semibold">Upload Error</p>
              <p className="mt-0.5">{validationError || errorMsg}</p>
            </div>
          </div>
        )}

        {!selectedFile ? (
          /* Dropzone */
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
          /* Video Selected Card & Preview */
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

        {/* Processing Mode Selection */}
        <div className="mt-6 pt-6 border-t border-slate-200 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center space-x-2 text-xs text-slate-600">
            <Settings2 className="w-4 h-4 text-slate-400" />
            <span className="font-medium text-slate-700">Execution Mode:</span>
            <select
              value={mode}
              onChange={(e) => setMode(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-800 text-xs rounded-md px-2.5 py-1 font-medium focus:ring-2 focus:ring-sky-500 focus:outline-none"
            >
              <option value="local">Local GPU (NVIDIA RTX 2050 — Real HMR2)</option>
              <option value="auto">Auto-detect (Local GPU → SSH → Mock)</option>
              <option value="mock">Mock Baseline (Development / Testing)</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={!selectedFile}
            className={`w-full sm:w-auto inline-flex items-center justify-center space-x-2 px-6 py-2.5 rounded-lg text-sm font-semibold text-white shadow-sm transition-all ${
              selectedFile
                ? 'bg-sky-600 hover:bg-sky-700 active:bg-sky-800 cursor-pointer'
                : 'bg-slate-300 cursor-not-allowed'
            }`}
          >
            <span>Analyze Video</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </form>

      {/* Clinical Disclaimer */}
      <div className="mt-8 text-center flex items-center justify-center space-x-2 text-xs text-slate-500">
        <Info className="w-4 h-4 text-slate-400 flex-shrink-0" />
        <span>
          ParkiSense is a B.Tech research prototype designed for computer-aided gait analysis and is not intended to provide a formal clinical diagnosis.
        </span>
      </div>
    </div>
  );
}
