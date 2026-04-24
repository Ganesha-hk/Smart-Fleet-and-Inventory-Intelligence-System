import React from 'react';
import { motion } from 'framer-motion';
import { Bell, AlertTriangle, ShieldCheck, Zap, Trash2, CheckCircle2 } from 'lucide-react';
import useFleetStore from '../store/useFleetStore';

const MotionDiv = motion.div;

const Notifications = () => {
  const { alerts, clearAlerts, quickActionAlert, removeAlert } = useFleetStore();

  return (
    <div className="max-w-5xl mx-auto space-y-10 pb-16">
      <div className="flex justify-between items-center bg-gray-950/20 p-8 rounded-3xl border border-gray-800/50 backdrop-blur-sm">
        <div>
          <h2 className="text-4xl font-black tracking-tighter text-gray-100">Inbound <span className="text-accent-blue">Intelligence</span></h2>
          <p className="text-gray-500 mt-2 font-medium">Real-time system monitoring and predictive fleet alerts.</p>
        </div>
        <button 
          onClick={clearAlerts} 
          className="group flex items-center gap-2 px-6 py-3 bg-gray-900 border border-gray-800 rounded-2xl text-xs font-bold uppercase tracking-widest transition-all hover:bg-red-500/10 hover:border-red-500/30 hover:text-red-400"
        >
          <Trash2 className="w-4 h-4 transition-transform group-hover:scale-110" />
          Purge All
        </button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {alerts.map((alert, index) => (
          <MotionDiv
            key={alert.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="group relative bg-gray-950/40 border border-gray-800/50 rounded-3xl overflow-hidden hover:border-gray-700 transition-all duration-300"
          >
            {/* Status Accent Line */}
            <div className={`absolute top-0 left-0 bottom-0 w-1 ${
              alert.type === 'critical' ? 'bg-red-500' : 
              alert.type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
            }`} />

            <div className="p-8 flex flex-col xl:flex-row xl:items-center gap-8">
              {/* Icon & Type */}
              <div className="flex items-center gap-6 shrink-0">
                <div className={`p-4 rounded-2xl ${
                  alert.type === 'critical' ? 'bg-red-500/10 text-red-500' : 
                  alert.type === 'warning' ? 'bg-yellow-500/10 text-yellow-500' : 
                  'bg-blue-500/10 text-blue-500'
                }`}>
                  {alert.type === 'critical' ? <AlertTriangle className="w-8 h-8" /> : 
                   alert.type === 'warning' ? <ShieldCheck className="w-8 h-8" /> : 
                   <Zap className="w-8 h-8" />}
                </div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-black uppercase tracking-widest px-2 py-0.5 rounded-md ${
                      alert.type === 'critical' ? 'bg-red-500/20 text-red-400 border border-red-500/30' : 
                      alert.type === 'warning' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' : 
                      'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                    }`}>
                      {alert.type}
                    </span>
                    <span className="text-xs font-mono text-gray-600">{alert.vehicle_id}</span>
                  </div>
                  <h4 className="text-xl font-bold text-gray-100 tracking-tight">{alert.title}</h4>
                  <p className="text-xs text-gray-500 font-medium tracking-tight">Sentinel Hub • {alert.time}</p>
                </div>
              </div>

              {/* Core Content */}
              <div className="flex-1 space-y-4">
                <div className="flex flex-wrap gap-4">
                  <div className="bg-gray-900/50 px-4 py-2 rounded-xl border border-gray-800/50">
                    <p className="text-[9px] text-gray-600 font-bold uppercase">Delay Risk</p>
                    <p className={`text-sm font-black ${alert.delayRisk > 70 ? 'text-red-400' : 'text-accent-blue'}`}>
                      {Math.round(alert.delayRisk)}%
                    </p>
                  </div>
                  <div className="bg-gray-900/50 px-4 py-2 rounded-xl border border-gray-800/50">
                    <p className="text-[9px] text-gray-600 font-bold uppercase">Risk Level</p>
                    <p className="text-sm font-black text-gray-300 uppercase tracking-tighter">{alert.riskLevel}</p>
                  </div>
                </div>
                
                <div className="border-l-2 border-gray-800 pl-4 space-y-2">
                  <p className="text-sm text-gray-400 leading-relaxed italic">{alert.message}</p>
                  <div className="flex flex-wrap gap-2 pt-1">
                    {alert.reasons?.map((reason) => (
                      <span key={`${alert.id}-${reason}`} className="text-[10px] text-gray-500 bg-gray-900 px-2 py-1 rounded-lg border border-gray-800">
                        {reason}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex xl:flex-col items-center justify-end gap-3 shrink-0">
                <button 
                  onClick={() => quickActionAlert(alert.id)} 
                  className="w-full xl:w-auto px-6 py-2.5 bg-accent-blue text-white rounded-xl text-xs font-bold uppercase tracking-widest transition-all hover:bg-blue-600 hover:shadow-[0_0_20px_rgba(59,130,246,0.3)]"
                >
                  Investigate
                </button>
                <button 
                  onClick={() => removeAlert(alert.id)} 
                  className="p-2.5 text-gray-600 rounded-xl transition-all hover:bg-gray-800 hover:text-red-400"
                >
                  <Trash2 className="w-5 h-5" />
                </button>
              </div>
            </div>
          </MotionDiv>
        ))}
      </div>

      {alerts.length === 0 && (
        <MotionDiv 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="py-32 text-center bg-gray-950/20 rounded-[3rem] border border-gray-800/50"
        >
          <div className="w-20 h-20 bg-gray-900 rounded-full flex items-center justify-center mx-auto mb-6 shadow-inner border border-gray-800">
            <ShieldCheck className="w-10 h-10 text-gray-700" />
          </div>
          <h3 className="text-2xl font-black text-gray-200 tracking-tighter">Systems Secure</h3>
          <p className="text-gray-500 mt-2 font-medium">All predictive models are currently running within nominal parameters.</p>
        </MotionDiv>
      )}
    </div>
  );
};

export default Notifications;
