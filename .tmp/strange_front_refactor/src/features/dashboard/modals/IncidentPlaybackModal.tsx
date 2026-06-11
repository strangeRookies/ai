import { Pause, Play, Volume2 } from 'lucide-react';
import type { IncidentAlert } from '../types/dashboard';

interface IncidentPlaybackModalProps {
  incident: IncidentAlert;
  isPlaying: boolean;
  playbackProgress: number;
  playbackStreamUrl?: string;
  onClose: () => void;
  onPlaybackProgressChange: (value: number) => void;
  onTogglePlaying: () => void;
}

export function IncidentPlaybackModal({
  incident,
  isPlaying,
  playbackProgress,
  playbackStreamUrl,
  onClose,
  onPlaybackProgressChange,
  onTogglePlaying,
}: IncidentPlaybackModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 backdrop-blur-sm p-4">
      <div className="w-full max-w-3xl bg-[#071329] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl flex flex-col">
        <div className="px-5 py-3.5 bg-[#061224] border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />
            <h3 className="text-sm font-extrabold text-white">Incident playback</h3>
            <span className="text-[10px] text-slate-400 bg-slate-800 border border-slate-700 px-2 py-0.5 rounded font-mono">
              {incident.camera} / {incident.time}
            </span>
          </div>
          <button onClick={onClose} className="text-xs font-bold text-slate-400 hover:text-white px-2 py-1 rounded bg-[#020817] border border-slate-800 cursor-pointer">Close</button>
        </div>
        <div className="relative aspect-video bg-black overflow-hidden">
          <img src={playbackStreamUrl} alt="Playback stream" className="w-full h-full object-cover contrast-125 brightness-75" />
          <div className="absolute top-1/3 left-1/3 w-1/3 h-1/3 border-2 border-rose-500 rounded bg-rose-500/5 flex flex-col justify-between p-2">
            <span className="text-[9px] font-bold text-white bg-rose-600 px-1.5 rounded uppercase self-start">{incident.type}</span>
            <span className="text-[10px] text-rose-400 font-extrabold text-center animate-pulse">{incident.label}</span>
          </div>
        </div>
        <div className="bg-[#061224] p-4 space-y-3">
          <div className="space-y-1">
            <div className="flex justify-between text-[10px] font-mono text-slate-400">
              <span>00:{playbackProgress.toString().padStart(2, '0')}</span>
              <span className="text-rose-400 font-bold">Focus point (00:30)</span>
              <span>01:00</span>
            </div>
            <div className="relative pt-1">
              <input type="range" min="0" max="60" value={playbackProgress} onChange={(event) => onPlaybackProgressChange(Number(event.target.value))} className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500" />
              <div className="absolute top-1 left-1/2 -translate-x-1/2 w-2 h-2.5 bg-rose-500 rounded-full border border-white" />
            </div>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button onClick={onTogglePlaying} className="p-2 hover:bg-slate-800 rounded-xl text-slate-300 hover:text-white cursor-pointer">
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <div className="flex items-center gap-1.5">
                <Volume2 className="w-4 h-4 text-slate-400" />
                <span className="text-[10px] text-slate-500">Audio</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
