import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Server, ShieldCheck } from 'lucide-react';

export default function Navbar() {
  const [backendHealth, setBackendHealth] = useState(null);
  const [gpuHealth, setGpuHealth] = useState(null);

  const checkHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendHealth(data.status === 'healthy');
      } else {
        setBackendHealth(false);
      }
    } catch {
      setBackendHealth(false);
    }

    try {
      const res = await fetch('/api/health/gpu');
      if (res.ok) {
        const data = await res.json();
        setGpuHealth(data.available);
      } else {
        setGpuHealth(false);
      }
    } catch {
      setGpuHealth(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo & Title */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-lg bg-sky-600 flex items-center justify-center text-white shadow-sm">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-lg font-bold tracking-tight text-slate-900">ParkiSense</h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200">
                Research v2.0
              </span>
            </div>
            <p className="text-xs text-slate-500 hidden sm:block">
              AI-Assisted Video-Based Gait Analysis
            </p>
          </div>
        </div>

        {/* System Health Tags */}
        <div className="flex items-center space-x-3 text-xs">
          {/* API Backend Tag */}
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200">
            <Server className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-600 font-medium hidden md:inline">API Backend:</span>
            <span className="flex items-center space-x-1">
              <span className={`w-2 h-2 rounded-full ${backendHealth === true ? 'bg-emerald-500' : backendHealth === false ? 'bg-rose-500' : 'bg-amber-400'}`}></span>
              <span className="font-semibold text-slate-700">
                {backendHealth === true ? 'Online' : backendHealth === false ? 'Offline' : 'Checking'}
              </span>
            </span>
          </div>

          {/* GPU Server Tag */}
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-full bg-slate-50 border border-slate-200">
            <Cpu className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-slate-600 font-medium hidden md:inline">GPU Service:</span>
            <span className="flex items-center space-x-1">
              <span className={`w-2 h-2 rounded-full ${gpuHealth === true ? 'bg-emerald-500' : gpuHealth === false ? 'bg-amber-500' : 'bg-slate-300'}`}></span>
              <span className="font-semibold text-slate-700">
                {gpuHealth === true ? 'GPU Active' : gpuHealth === false ? 'Unavailable' : 'Checking'}
              </span>
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
