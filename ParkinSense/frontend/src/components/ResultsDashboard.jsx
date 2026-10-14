import React, { useMemo } from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart2,
  Brain,
  CheckCircle2,
  FileText,
  Info,
  Layers,
  RotateCcw,
  ShieldAlert,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const formatNumber = (value, digits = 2, suffix = '') => {
  if (value === null || value === undefined || value === '') {
    return 'N/A';
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 'N/A';
  }
  return `${numeric.toFixed(digits)}${suffix}`;
};

const getFiniteNumber = (value, fallback = null) => {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
};

export default function ResultsDashboard({ result, onReset }) {
  if (!result) return null;

  const pdInfo = result.parkinsons_detection || {};
  const metadata = result.metadata || {};
  const modelInfo = result.model_information || {};
  const datasetPerformance = result.dataset_performance || {};
  const frames = Array.isArray(result.key_gait_frames) ? result.key_gait_frames : [];

  const isMediaPipe = (result.pipeline || '').toLowerCase() === 'mediapipe';
  const pipelineLabel = isMediaPipe ? 'MediaPipe Pose' : (result.pose_service_mode === 'mock' ? '4D-Humans HMR2 (Demo SMPL)' : '4D-Humans HMR2 (3D Mesh)');
  const diagnosis = pdInfo.diagnosis || 'Analysis complete';
  const pdProbability = getFiniteNumber(pdInfo.probability, null);
  const healthyProbability = getFiniteNumber(
    pdInfo.class_probabilities?.Healthy,
    pdInfo.label === 1 && pdProbability !== null ? 100 - pdProbability : pdProbability,
  );
  const parkinsonsProbability = getFiniteNumber(
    pdInfo.class_probabilities?.Parkinsons,
    pdInfo.label === 1 ? pdProbability : (pdProbability === null ? null : 100 - pdProbability),
  );
  const confidence = getFiniteNumber(
    pdInfo.confidence,
    pdInfo.label === 1 ? parkinsonsProbability : healthyProbability,
  );
  const predictedLabel = diagnosis || (pdInfo.label === 1 ? 'Parkinsons likely' : 'Healthy / control');
  const disclaimerText = result.disclaimer || 'ParkinSense is a research prototype intended for gait analysis research and educational purposes. It is not a clinical diagnostic tool and should not be used to diagnose or make medical decisions.';
  const pipelineSteps = result.pipeline === 'mediapipe'
    ? ['Video', 'MediaPipe Pose', '33 Landmarks', 'Gait Feature Extraction', '290-D Feature Vector', 'Random Forest', 'Assessment']
    : ['Video', 'HMR2 / 4D-Humans', 'SMPL Pose Sequence', 'Gait Feature Extraction', 'Random Forest', 'Assessment'];

  const summaryMetrics = useMemo(() => {
    const metrics = [];

    if (result.num_frames != null || metadata.num_frames != null) {
      metrics.push({ label: 'Usable frames analyzed', value: formatNumber(result.num_frames ?? metadata.num_frames, 0), help: 'Frames with detected pose landmarks' });
    }

    if (result.video_duration_seconds != null || metadata.video_duration_seconds != null) {
      metrics.push({ label: 'Video duration', value: formatNumber(result.video_duration_seconds ?? metadata.video_duration_seconds, 2, ' s'), help: 'Actual video length' });
    }

    if (result.processing_time_seconds != null) {
      metrics.push({ label: 'Processing time', value: formatNumber(result.processing_time_seconds, 2, ' s'), help: 'Total backend pipeline time' });
    }

    if (result.pose_detection_quality != null || metadata.pose_detection_quality != null) {
      metrics.push({ label: 'Pose detection quality', value: formatNumber(result.pose_detection_quality ?? metadata.pose_detection_quality, 1, '%'), help: 'Average landmark visibility' });
    }

    if (result.fps != null || metadata.fps != null) {
      metrics.push({ label: 'FPS', value: formatNumber(result.fps ?? metadata.fps, 2, ' fps'), help: 'Video sampling rate' });
    }

    if (confidence !== null) {
      metrics.push({ label: 'Model confidence', value: formatNumber(confidence, 1, '%'), help: 'Confidence in the displayed class' });
    }

    return metrics;
  }, [confidence, metadata, result]);

  const gaitSummary = useMemo(() => {
    const gaitMetrics = result.gait_metrics || {};
    const kneesLeft = frames.map((frame) => getFiniteNumber(frame.knee_angle_left)).filter((num) => num !== null);
    const kneesRight = frames.map((frame) => getFiniteNumber(frame.knee_angle_right)).filter((num) => num !== null);
    const leftArmSwing = frames.map((frame) => getFiniteNumber(frame.left_arm_movement)).filter((num) => num !== null);
    const rightArmSwing = frames.map((frame) => getFiniteNumber(frame.right_arm_movement)).filter((num) => num !== null);
    const asymmetry = frames.map((frame) => getFiniteNumber(frame.asymmetry)).filter((num) => num !== null);
    const mean = (values) => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
    const leftKnee = getFiniteNumber(gaitMetrics.left_knee_angle_mean, mean(kneesLeft));
    const rightKnee = getFiniteNumber(gaitMetrics.right_knee_angle_mean, mean(kneesRight));
    const leftArm = getFiniteNumber(gaitMetrics.left_arm_movement_mean, mean(leftArmSwing));
    const rightArm = getFiniteNumber(gaitMetrics.right_arm_movement_mean, mean(rightArmSwing));
    const armDifference = getFiniteNumber(gaitMetrics.arm_movement_difference_mean,
      leftArm !== null && rightArm !== null ? Math.abs(leftArm - rightArm) : null);
    const sideDifference = leftKnee !== null && rightKnee !== null ? Math.abs(leftKnee - rightKnee) : null;
    const meanAsymmetry = getFiniteNumber(gaitMetrics.left_right_asymmetry_mean, mean(asymmetry));
    const meanStepWidth = getFiniteNumber(gaitMetrics.step_width_mean, null);

    return [
      { label: 'Left knee angle', value: leftKnee === null ? null : formatNumber(leftKnee, 1, '°') },
      { label: 'Right knee angle', value: rightKnee === null ? null : formatNumber(rightKnee, 1, '°') },
      { label: 'Left arm movement', value: leftArm === null ? null : formatNumber(leftArm, 4, ' m/frame') },
      { label: 'Right arm movement', value: rightArm === null ? null : formatNumber(rightArm, 4, ' m/frame') },
      { label: 'Arm movement difference', value: armDifference === null ? null : formatNumber(armDifference, 4, ' m/frame') },
      { label: 'Left/right knee angle difference', value: sideDifference === null ? null : formatNumber(sideDifference, 1, '°') },
      { label: 'Mean left/right asymmetry', value: meanAsymmetry === null ? null : formatNumber(meanAsymmetry, 2, '°') },
      { label: 'Mean step width', value: meanStepWidth === null ? null : formatNumber(meanStepWidth, 3, ' m') },
    ].filter((metric) => metric.value !== null);
  }, [frames, result.gait_metrics]);

  const chartData = useMemo(() => {
    const series = Array.isArray(result.gait_timeseries) && result.gait_timeseries.length
      ? result.gait_timeseries
      : frames;
    return series
      .map((frame) => ({
        time: getFiniteNumber(frame.timestamp, null),
        kneeLeft: getFiniteNumber(frame.knee_left ?? frame.knee_angle_left, null),
        kneeRight: getFiniteNumber(frame.knee_right ?? frame.knee_angle_right, null),
        leftArm: getFiniteNumber(frame.left_arm ?? frame.left_arm_movement, null),
        rightArm: getFiniteNumber(frame.right_arm ?? frame.right_arm_movement, null),
      }))
      .filter((point) => point.time !== null && (point.kneeLeft !== null || point.kneeRight !== null || point.leftArm !== null || point.rightArm !== null));
  }, [frames, result.gait_timeseries]);

  const datasetRows = [
    { label: 'Accuracy', value: datasetPerformance.accuracy },
    { label: 'Sensitivity', value: datasetPerformance.sensitivity },
    { label: 'Specificity', value: datasetPerformance.specificity },
    { label: 'Precision', value: datasetPerformance.precision },
    { label: 'F1 score', value: datasetPerformance.f1_score },
    { label: 'ROC AUC', value: datasetPerformance.roc_auc },
  ].filter((item) => item.value !== null && item.value !== undefined && Number.isFinite(Number(item.value)));

  const firstFrame = frames[0] || {};
  const imageA = firstFrame.original_image_path || firstFrame.zoom_image_path || null;
  const imageB = firstFrame.image_path || null;
  const metricValue = (label) => gaitSummary.find((metric) => metric.label === label)?.value || 'Not available';

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <div className="flex flex-col gap-4 pb-4 border-b border-slate-200 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-2xl font-bold tracking-tight text-slate-900">Analysis Results Dashboard</h2>
            <span className="rounded-full border border-slate-200 bg-slate-100 px-2.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-slate-700">
              Job {result.job_id || 'N/A'}
            </span>
            <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold tracking-wide border ${
              isMediaPipe
                ? 'bg-sky-50 text-sky-700 border-sky-200'
                : 'bg-purple-50 text-purple-700 border-purple-200'
            }`}>
              {pipelineLabel}
            </span>
          </div>
          <p className="mt-1 text-xs text-slate-500">
            Video: <span className="font-semibold text-slate-700">{result.video_file || 'Uploaded video'}</span>
          </p>
        </div>

        <button
          type="button"
          onClick={onReset}
          className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          <RotateCcw className="h-4 w-4 text-slate-500" />
          Analyze Another Video
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-500">Assessment Result</span>
            <div className={`rounded-lg p-2 ${pdInfo.label === 1 ? 'bg-amber-50 text-amber-700' : 'bg-emerald-50 text-emerald-700'}`}>
              {pdInfo.label === 1 ? <AlertTriangle className="h-5 w-5" /> : <CheckCircle2 className="h-5 w-5" />}
            </div>
          </div>

          <h3 className="mt-4 text-2xl font-extrabold text-slate-900">{diagnosis}</h3>
          <p className="mt-2 text-sm text-slate-500">Predicted class: <span className="font-semibold text-slate-700">{predictedLabel}</span></p>

          <div className="mt-5 flex items-baseline gap-2">
            <span className="text-4xl font-extrabold text-slate-900">{confidence !== null ? `${confidence.toFixed(1)}%` : 'N/A'}</span>
            <span className="text-[11px] uppercase tracking-wide text-slate-500" title="How strongly the trained model favored the displayed class">Model Confidence</span>
          </div>

          <div className="mt-5">
            <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500">
              <span>Healthy</span>
              <span>Parkinson&apos;s</span>
            </div>
            {healthyProbability !== null && parkinsonsProbability !== null && (
              <div className="flex h-2.5 overflow-hidden rounded-full border border-slate-200 bg-slate-100" aria-label={`Healthy ${healthyProbability} percent, Parkinson's ${parkinsonsProbability} percent`}>
                <div className="h-full bg-emerald-500" style={{ width: `${Math.max(0, Math.min(100, healthyProbability))}%` }} />
                <div className="h-full bg-amber-500" style={{ width: `${Math.max(0, Math.min(100, parkinsonsProbability))}%` }} />
              </div>
            )}
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
              <p className="text-emerald-800">Healthy: <strong>{formatNumber(healthyProbability, 1, '%')}</strong></p>
              <p className="text-right text-amber-800">Parkinson&apos;s: <strong>{formatNumber(parkinsonsProbability, 1, '%')}</strong></p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-500">Analysis Quality &amp; Summary</span>
            <div className="rounded-lg bg-sky-50 p-2 text-sky-700"><Layers className="h-5 w-5" /></div>
          </div>

          <div className="mt-5 space-y-3">
            {summaryMetrics.length > 0 ? summaryMetrics.map((metric) => (
              <div key={metric.label} className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs text-slate-500">{metric.label}</span>
                  <span className="text-sm font-bold text-slate-900">{metric.value}</span>
                </div>
                {metric.help && <p className="mt-1 text-[10px] text-slate-500">{metric.help}</p>}
              </div>
            )) : <p className="text-sm text-slate-500">No summary metrics available for this result.</p>}
          </div>
        </div>

      </div>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm" aria-labelledby="evidence-heading">
        <div className="mb-2 flex items-center gap-2">
          <Activity className="h-5 w-5 text-sky-600" />
          <h3 id="evidence-heading" className="text-lg font-bold text-slate-900">Why did ParkinSense make this assessment?</h3>
        </div>
        <p className="mb-5 text-sm text-slate-600">These are observed movement measurements from this video, not explanations of which features caused the model&apos;s prediction.</p>

        <div className="grid gap-4 lg:grid-cols-3">
          <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <h4 className="text-xs font-bold uppercase tracking-wide text-slate-700">Knee movement</h4>
            <div className="mt-3 space-y-2 text-sm">
              <p className="flex justify-between gap-2"><span>Left mean angle</span><strong>{metricValue('Left knee angle')}</strong></p>
              <p className="flex justify-between gap-2"><span>Right mean angle</span><strong>{metricValue('Right knee angle')}</strong></p>
              <p className="flex justify-between gap-2"><span>Side-to-side difference</span><strong>{metricValue('Left/right knee angle difference')}</strong></p>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-600">The angle describes how much the knee bends or extends while walking. A validated healthy reference distribution is not available for these angle measurements.</p>
          </article>

          <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <h4 className="text-xs font-bold uppercase tracking-wide text-slate-700">Arm swing</h4>
            <div className="mt-3 space-y-2 text-sm">
              <p className="flex justify-between gap-2"><span>Left mean movement</span><strong>{metricValue('Left arm movement')}</strong></p>
              <p className="flex justify-between gap-2"><span>Right mean movement</span><strong>{metricValue('Right arm movement')}</strong></p>
              <p className="flex justify-between gap-2"><span>Difference</span><strong>{metricValue('Arm movement difference')}</strong></p>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-600">These are mean wrist displacements between detected frames, not total arm-swing ranges. Difference is calculated from the two displayed sequence means. Reference comparison is not available.</p>
          </article>

          <article className="rounded-xl border border-slate-200 bg-slate-50 p-4">
            <h4 className="text-xs font-bold uppercase tracking-wide text-slate-700">Left/right asymmetry</h4>
            <div className="mt-3 space-y-2 text-sm">
              <p className="flex justify-between gap-2"><span>Mean knee-angle difference</span><strong>{metricValue('Mean left/right asymmetry')}</strong></p>
              <p className="flex justify-between gap-2"><span>Underlying left angle</span><strong>{metricValue('Left knee angle')}</strong></p>
              <p className="flex justify-between gap-2"><span>Underlying right angle</span><strong>{metricValue('Right knee angle')}</strong></p>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-600">This summarizes the measured difference between left and right knee angles. It is an observation and does not indicate a diagnosis.</p>
          </article>
        </div>

        <div className="mt-4 rounded-lg border border-sky-100 bg-sky-50 p-3 text-xs text-sky-900">
          Observed gait evidence describes measurements; it does not establish why the classifier selected its output. Step width describes the side-to-side distance between the feet. The available validated dataset file contains model-evaluation statistics, not reference distributions for these gait measurements.
        </div>
      </section>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <BarChart2 className="h-5 w-5 text-sky-600" />
            <h3 className="text-lg font-bold text-slate-900">Observed Gait Metrics</h3>
          </div>
          <span className="rounded-full border border-slate-200 bg-slate-100 px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-slate-600">
            Actual extracted values
          </span>
        </div>

        {gaitSummary.length > 0 ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {gaitSummary.map((metric) => (
              <div key={metric.label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500">{metric.label}</p>
                <p className="mt-2 text-xl font-extrabold text-slate-900">{metric.value}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No gait metrics were exposed for this analysis run.</p>
        )}
      </div>

      {chartData.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-bold text-slate-900"><Activity className="h-5 w-5 text-sky-600" />Knee Angle Over Time</h3>
            <div className="h-64 w-full">
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="time" domain={[0, 'dataMax']} tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={(value) => `${Number(value).toFixed(1)}s`} label={{ value: 'Video time (seconds)', position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }} label={{ value: 'Knee angle (degrees)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip labelFormatter={(value) => `${Number(value).toFixed(2)} s`} formatter={(value, name) => [`${Number(value).toFixed(1)}°`, name]} />
                  <Legend />
                  {frames.slice(0, 3).map((frame, index) => <ReferenceLine key={`knee-marker-${frame.frame_index ?? index}`} x={Number(frame.timestamp)} stroke="#64748b" strokeDasharray="4 3" label={`K${index + 1}`} />)}
                  <Line type="monotone" dataKey="kneeLeft" stroke="#0ea5e9" strokeWidth={2} dot={false} name="Left knee" />
                  <Line type="monotone" dataKey="kneeRight" stroke="#f59e0b" strokeWidth={2} dot={false} name="Right knee" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="mb-4 flex items-center gap-2 text-lg font-bold text-slate-900"><Activity className="h-5 w-5 text-sky-600" />Arm Movement Over Time</h3>
            <div className="h-64 w-full">
              <ResponsiveContainer>
                <LineChart data={chartData} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="time" domain={[0, 'dataMax']} tick={{ fontSize: 10, fill: '#64748b' }} tickFormatter={(value) => `${Number(value).toFixed(1)}s`} label={{ value: 'Video time (seconds)', position: 'insideBottom', offset: -4 }} />
                  <YAxis tick={{ fontSize: 10, fill: '#64748b' }} label={{ value: 'Wrist displacement (m/frame)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip labelFormatter={(value) => `${Number(value).toFixed(2)} s`} formatter={(value, name) => [`${Number(value).toFixed(4)} m/frame`, name]} />
                  <Legend />
                  {frames.slice(0, 3).map((frame, index) => <ReferenceLine key={`arm-marker-${frame.frame_index ?? index}`} x={Number(frame.timestamp)} stroke="#64748b" strokeDasharray="4 3" label={`K${index + 1}`} />)}
                  <Line type="monotone" dataKey="leftArm" stroke="#22c55e" strokeWidth={2} dot={false} name="Left arm" />
                  <Line type="monotone" dataKey="rightArm" stroke="#f97316" strokeWidth={2} dot={false} name="Right arm" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {!result.gait_timeseries && chartData.length > 0 && (
        <p className="-mt-6 text-xs text-slate-500">This stored result contains selected key-frame samples only; newly processed MediaPipe videos include sampled measurements across the full sequence.</p>
      )}

      {frames.length > 0 && (
        <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm" aria-labelledby="moments-heading">
          <div className="mb-1 flex items-center gap-2">
            <Activity className="h-5 w-5 text-sky-600" />
            <h3 id="moments-heading" className="text-lg font-bold text-slate-900">Key Movement Moments</h3>
          </div>
          <p className="mb-5 text-sm text-slate-600">Frames are selected from measured gait variation, separated in time, and shown with their extracted video timestamps.</p>
          <div className="grid gap-5 lg:grid-cols-3">
            {frames.slice(0, 3).map((frame, index) => (
              <article key={frame.frame_index ?? frame.timestamp ?? index} className="overflow-hidden rounded-xl border border-slate-200 bg-slate-50">
                <div className="flex items-center justify-between border-b border-slate-200 px-4 py-3">
                  <h4 className="text-xs font-bold uppercase tracking-wide text-slate-800">Moment {index + 1}</h4>
                  <span className="font-mono text-sm font-semibold text-slate-700">{formatNumber(frame.timestamp, 2, ' s')}</span>
                </div>
                <div className="grid grid-cols-2 gap-2 p-3">
                  <figure>
                    {frame.original_image_path || frame.zoom_image_path ? <img src={frame.original_image_path || frame.zoom_image_path} alt={`Original video frame at ${frame.timestamp} seconds`} className="h-36 w-full rounded-lg border border-slate-200 bg-white object-contain" /> : <div className="flex h-36 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-xs text-slate-500">Original frame unavailable</div>}
                    <figcaption className="mt-1 text-center text-[10px] uppercase tracking-wide text-slate-500">Original frame</figcaption>
                  </figure>
                  <figure>
                    {frame.image_path ? <img src={frame.image_path} alt={`Detected pose at ${frame.timestamp} seconds`} className="h-36 w-full rounded-lg border border-slate-200 bg-white object-contain" /> : <div className="flex h-36 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-xs text-slate-500">Pose image unavailable</div>}
                    <figcaption className="mt-1 text-center text-[10px] uppercase tracking-wide text-slate-500">Pose overlay</figcaption>
                  </figure>
                </div>
                <div className="px-4 pb-4">
                  <p className="text-sm font-semibold text-slate-900">{frame.title || frame.primary_indicator || 'Measured gait variation'}</p>
                  <p className="mt-2 text-xs leading-5 text-slate-600">{frame.analysis_summary || frame.reason_selected || 'Measured movement variation at this timestamp.'}</p>
                  <p className="mt-2 text-[10px] leading-4 text-slate-500">Selection reason: {frame.reason_selected || 'Selected using the backend gait-variation score.'}</p>
                </div>
              </article>
            ))}
          </div>
        </section>
      )}

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-sky-600" />
            <h3 className="text-lg font-bold text-slate-900">Pose Analysis</h3>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">Original Frame</div>
            {imageA ? (
              <img src={imageA} alt="Original video frame" className="h-64 w-full rounded-lg border border-slate-200 bg-white object-contain" />
            ) : (
              <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-sm text-slate-500">Original frame unavailable</div>
            )}
          </div>

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
            <div className="mb-2 text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500">Detected Pose</div>
            {imageB ? (
              <img src={imageB} alt="Detected pose overlay" className="h-64 w-full rounded-lg border border-slate-200 bg-white object-contain" />
            ) : (
              <div className="flex h-64 items-center justify-center rounded-lg border border-dashed border-slate-300 bg-white text-sm text-slate-500">Detected pose image unavailable</div>
            )}
          </div>
        </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
              <p className="font-semibold text-slate-800">What the system observed</p>
              <p className="mt-2">Body landmarks were tracked, lower- and upper-limb movement was measured, and gait features were calculated from the video.</p>
            </div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
              <p className="font-semibold text-slate-800">Why this matters</p>
              <p className="mt-2">The extracted movement features are inputs to the machine-learning model. The displayed frame itself does not establish a diagnosis.</p>
            </div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center gap-2">
          <ShieldAlert className="h-5 w-5 text-sky-600" />
          <h3 className="text-lg font-bold text-slate-900">Analysis Pipeline</h3>
        </div>

        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          {pipelineSteps.map((step, index, arr) => (
            <React.Fragment key={step}>
              <div className="rounded-full border border-sky-200 bg-sky-50 px-3 py-2 text-center text-xs font-semibold text-sky-800">{step}</div>
              {index < arr.length - 1 && <span className="text-slate-400">↓</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center gap-2">
          <Brain className="h-5 w-5 text-sky-600" />
          <h3 className="text-lg font-bold text-slate-900">Model Information</h3>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          <div className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">Model</p><p className="mt-1 font-semibold text-slate-900">{modelInfo.model || 'Random Forest'}</p></div>
          <div className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">Features</p><p className="mt-1 font-semibold text-slate-900">{modelInfo.n_features || metadata.features_extracted || 'N/A'}</p></div>
          <div className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">Trees</p><p className="mt-1 font-semibold text-slate-900">{modelInfo.n_trees || 'N/A'}</p></div>
          <div className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">Model classes</p><p className="mt-1 font-semibold text-slate-900">{Array.isArray(modelInfo.classes) && modelInfo.classes.length ? modelInfo.classes.map((value) => value === 0 || value === '0' ? '0 Healthy' : value === 1 || value === '1' ? '1 Parkinsons' : String(value)).join(' · ') : 'N/A'}</p></div>
          <div className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">Pose pipeline</p><p className="mt-1 font-semibold text-slate-900">{pipelineLabel}</p></div>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="mb-5 flex items-center gap-2">
          <Info className="h-5 w-5 text-sky-600" />
          <h3 className="text-lg font-bold text-slate-900">Model Evaluation</h3>
        </div>

        <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-[11px] text-amber-800">
          These results come from the repository&apos;s validated dataset evaluation and are separate from the specific video prediction shown above.
        </div>

        {datasetRows.length > 0 ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {datasetRows.map((row) => (
              <div key={row.label} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                <p className="text-[11px] uppercase tracking-[0.12em] text-slate-500">{row.label}</p>
                <p className="mt-2 text-xl font-extrabold text-slate-900">{formatNumber(row.value * 100, 2, '%')}</p>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-500">No dataset evaluation metrics were available in the project.</p>
        )}
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-start gap-3 text-slate-600">
          <Info className="mt-0.5 h-5 w-5 text-slate-400" />
          <div>
            <p className="text-sm font-semibold text-slate-800">Research prototype disclaimer</p>
            <p className="mt-1 text-sm">{disclaimerText}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
