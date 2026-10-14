import React from 'react';
import { 
  Activity, 
  Brain, 
  CheckCircle2, 
  AlertTriangle, 
  ShieldAlert, 
  RotateCcw, 
  FileText, 
  BarChart2, 
  Info,
  TrendingUp,
  Layers
} from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts';

export default function ResultsDashboard({ result, onReset }) {
  if (!result) return null;

  const pdInfo = result.parkinsons_detection || {};
  const sevInfo = result.severity_estimation || {};
  const explainability = result.explainability || {};
  const metadata = result.metadata || {};

  const diagnosis = pdInfo.diagnosis || 'Analysis Completed';
  const pdProb = pdInfo.probability ?? 0;
  const healthyProb = pdInfo.class_probabilities?.Healthy ?? (100 - pdProb);
  const isParkinsons = pdInfo.label === 1 || pdProb >= 50;
  const riskLevel = pdInfo.risk_level || 'Normal';
  const updrsScore = sevInfo.updrs_gait_score ?? 0;
  const updrsDesc = sevInfo.description || 'Normal Gait (UPDRS 0: No movement difficulty)';

  // Prepare chart data for clinical explainability indicators
  const indicators = explainability.important_gait_indicators || [];
  const chartData = indicators.map((item) => ({
    name: item.joint_name || item.indicator || 'Joint Indicator',
    indicator: item.indicator || 'Gait Indicator',
    importance: Math.round((item.importance || 0) * 1000) / 10, // Convert to percentage score
    rawImportance: item.importance || 0,
    joint: item.joint_name || 'Anatomical Joint',
    desc: item.description || '',
  }));

  // Risk Level Badge Styling
  const getRiskBadge = (risk) => {
    switch (risk.toUpperCase()) {
      case 'HIGH':
        return 'bg-rose-50 text-rose-700 border-rose-200';
      case 'MEDIUM':
        return 'bg-amber-50 text-amber-700 border-amber-200';
      default:
        return 'bg-emerald-50 text-emerald-700 border-emerald-200';
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      {/* Top Banner / Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-200">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-2xl font-bold tracking-tight text-slate-900">Analysis Results Dashboard</h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-100 text-slate-700 font-medium border border-slate-200">
              Job: {result.job_id}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Video: <span className="font-semibold text-slate-700">{result.video_file}</span> • Processing Time: {result.processing_time_seconds}s
          </p>
        </div>

        <button
          onClick={onReset}
          className="inline-flex items-center justify-center space-x-2 px-4 py-2 rounded-lg bg-white border border-slate-300 text-slate-700 text-xs font-semibold hover:bg-slate-50 shadow-sm transition-colors"
        >
          <RotateCcw className="w-4 h-4 text-slate-500" />
          <span>Analyze Another Video</span>
        </button>
      </div>

      {/* Demo / Mock Notice Banner */}
      {result.is_demo && (
        <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 flex items-start space-x-3 text-amber-900 text-xs">
          <Info className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-bold text-amber-900">DEMO / MOCK PIPELINE EXECUTION</p>
            <p className="mt-0.5 text-amber-800">
              {result.demo_notice || "4D-Humans GPU stack is currently offline. Analysis was performed using sample SMPL gait dynamics to verify end-to-end pipeline features & ML inference for this job."}
            </p>
          </div>
        </div>
      )}

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        
        {/* Card 1: Parkinson's Detection Likelihood */}
        <div className="medical-card p-6 bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Primary Diagnosis</span>
              <div className={`p-2 rounded-lg ${isParkinsons ? 'bg-amber-50 text-amber-700' : 'bg-emerald-50 text-emerald-700'}`}>
                {isParkinsons ? <AlertTriangle className="w-5 h-5" /> : <CheckCircle2 className="w-5 h-5" />}
              </div>
            </div>

            <h3 className="text-2xl font-extrabold text-slate-900 mt-3">
              {diagnosis}
            </h3>

            <div className="mt-4 flex items-baseline space-x-2">
              <span className="text-4xl font-extrabold text-slate-900">{pdProb.toFixed(1)}%</span>
              <span className="text-xs text-slate-500 font-medium">Parkinson's Likelihood</span>
            </div>

            {/* Visual Bar */}
            <div className="mt-4">
              <div className="flex justify-between text-xs text-slate-500 mb-1">
                <span>Healthy ({healthyProb.toFixed(1)}%)</span>
                <span>Parkinson's ({pdProb.toFixed(1)}%)</span>
              </div>
              <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden flex border border-slate-200">
                <div className="h-full bg-emerald-500" style={{ width: `${healthyProb}%` }}></div>
                <div className="h-full bg-amber-500" style={{ width: `${pdProb}%` }}></div>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs">
            <span className="text-slate-500 font-medium">Model Risk Level:</span>
            <span className={`px-2.5 py-0.5 rounded-full font-bold border ${getRiskBadge(riskLevel)}`}>
              {riskLevel} Risk
            </span>
          </div>
        </div>

        {/* Card 2: UPDRS Severity Estimation */}
        <div className="medical-card p-6 bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">UPDRS Gait Severity</span>
              <div className="p-2 rounded-lg bg-sky-50 text-sky-700">
                <Brain className="w-5 h-5" />
              </div>
            </div>

            <div className="mt-3 flex items-baseline space-x-2">
              <span className="text-4xl font-extrabold text-slate-900">Score {updrsScore}</span>
              <span className="text-xs text-slate-500 font-medium">MDS-UPDRS Part III</span>
            </div>

            <p className="text-sm font-semibold text-slate-800 mt-3 leading-snug">
              {updrsDesc}
            </p>

            <p className="text-xs text-slate-500 mt-2">
              Evaluated using Random Forest multi-class severity model on 290 SMPL gait channels.
            </p>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>UPDRS Scale Range:</span>
            <span className="font-semibold text-slate-700">0 (Normal) to 4 (Severe)</span>
          </div>
        </div>

        {/* Card 3: Video & Gait Kinematic Metadata */}
        <div className="medical-card p-6 bg-white border border-slate-200 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500">Video Kinematics</span>
              <div className="p-2 rounded-lg bg-indigo-50 text-indigo-700">
                <Layers className="w-5 h-5" />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                <p className="text-xs text-slate-500">Frames Analyzed</p>
                <p className="text-lg font-bold text-slate-900">{metadata.num_frames || 'N/A'}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                <p className="text-xs text-slate-500">Sampling Rate</p>
                <p className="text-lg font-bold text-slate-900">{metadata.fps ? `${metadata.fps} FPS` : 'N/A'}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                <p className="text-xs text-slate-500">Gait Features</p>
                <p className="text-lg font-bold text-slate-900">{metadata.features_extracted || 290}</p>
              </div>
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                <p className="text-xs text-slate-500">Pose Model</p>
                <p className="text-lg font-bold text-slate-900">SMPL 24-Joint</p>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Pipeline Engine:</span>
            <span className="font-semibold text-slate-700">4D-Humans HMR2</span>
          </div>
        </div>

      </div>

      {/* Explainability Section — "Why this prediction?" */}
      <div className="medical-card p-6 sm:p-8 bg-white border border-slate-200 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 flex items-center space-x-2">
              <BarChart2 className="w-5 h-5 text-sky-600" />
              <span>Model Explainability — Clinical Gait Indicators</span>
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Top anatomical joint motion features contributing to the Random Forest model's prediction.
            </p>
          </div>

          <span className="text-xs text-slate-500 bg-slate-100 px-3 py-1 rounded-full border border-slate-200 self-start sm:self-auto font-medium">
            Random Forest Feature Importance
          </span>
        </div>

        {/* Feature Importance Bar Chart */}
        {chartData.length > 0 ? (
          <div className="space-y-6">
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis type="number" unit="%" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis dataKey="name" type="category" width={180} tick={{ fontSize: 11, fill: '#334155', fontWeight: 600 }} />
                  <Tooltip
                    formatter={(value) => [`${value}% Score`, 'Importance']}
                    contentStyle={{ backgroundColor: '#ffffff', borderColor: '#cbd5e1', borderRadius: '0.5rem', fontSize: '12px' }}
                  />
                  <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={index === 0 ? '#0284c7' : index === 1 ? '#0369a1' : '#075985'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Indicator Details Table */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t border-slate-100">
              {indicators.map((ind, idx) => (
                <div key={idx} className="p-4 rounded-xl bg-slate-50 border border-slate-200 flex items-start space-x-3">
                  <div className="w-8 h-8 rounded-lg bg-sky-100 text-sky-800 font-bold flex items-center justify-center text-xs flex-shrink-0">
                    #{idx + 1}
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h4 className="text-sm font-bold text-slate-900">{ind.indicator}</h4>
                      <span className="text-[10px] font-semibold text-slate-500 bg-white px-2 py-0.5 rounded border border-slate-200">
                        {ind.joint_name}
                      </span>
                    </div>
                    <p className="text-xs text-slate-600 mt-1">{ind.description}</p>
                    <div className="mt-2 flex items-center space-x-3 text-[11px] text-slate-500">
                      <span>Feature: <code className="font-mono text-slate-700">{ind.feature}</code></span>
                      <span>•</span>
                      <span>Importance: <strong>{((ind.importance || 0) * 100).toFixed(2)}%</strong></span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-slate-500 text-sm">
            No specific joint explainability indicators available for this model run.
          </div>
        )}
      </div>

      {/* Medical Disclaimer Box */}
      <div className="p-4 rounded-xl bg-slate-100 border border-slate-200 flex items-start space-x-3 text-slate-600 text-xs">
        <Info className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold text-slate-800">Research Prototype Notice</p>
          <p className="mt-0.5">
            ParkiSense predictions are derived from computer vision motion metrics and trained machine learning models for academic research. This interface does not replace clinical evaluation by a qualified neurologist.
          </p>
        </div>
      </div>
    </div>
  );
}
