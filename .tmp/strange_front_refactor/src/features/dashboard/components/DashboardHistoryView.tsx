import { Download, Search, Video } from 'lucide-react';
import type { IncidentAlert } from '../types/dashboard';

interface DashboardHistoryViewProps {
  filteredHistory: readonly IncidentAlert[];
  searchCamera: string;
  searchDate: 'today' | 'week' | 'month';
  searchKeyword: string;
  cameraOptions: readonly { id: string; name: string }[];
  onOpenIncident: (alert: IncidentAlert) => void;
  onSearchCameraChange: (value: string) => void;
  onSearchDateChange: (value: 'today' | 'week' | 'month') => void;
  onSearchKeywordChange: (value: string) => void;
}

export function DashboardHistoryView({
  filteredHistory,
  searchCamera,
  searchDate,
  searchKeyword,
  cameraOptions,
  onOpenIncident,
  onSearchCameraChange,
  onSearchDateChange,
  onSearchKeywordChange,
}: DashboardHistoryViewProps) {
  return (
    <div className="flex-1 p-6 space-y-6 overflow-y-auto max-w-5xl flex flex-col">
      <div>
        <h2 className="text-base font-extrabold text-white">Incident playback archive</h2>
        <p className="text-xs text-slate-400 mt-1">
          Filter saved incidents by period, camera, and keyword.
        </p>
      </div>
      <div className="bg-[#071329] border border-slate-800 p-4 rounded-2xl">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 tracking-wider">Period</label>
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { id: 'today', label: 'Today' },
                { id: 'week', label: '7 days' },
                { id: 'month', label: '30 days' },
              ].map((period) => (
                <button
                  key={period.id}
                  onClick={() => onSearchDateChange(period.id as 'today' | 'week' | 'month')}
                  className={`py-2 rounded-lg text-[10px] font-bold border transition-all cursor-pointer ${
                    searchDate === period.id
                      ? 'bg-blue-600/10 border-blue-500 text-blue-400'
                      : 'border-slate-800 text-slate-400 hover:border-slate-700'
                  }`}
                >
                  {period.label}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 tracking-wider">Camera</label>
            <select
              value={searchCamera}
              onChange={(event) => onSearchCameraChange(event.target.value)}
              className="w-full px-3 py-2 bg-[#020817] border border-slate-800 rounded-lg text-xs text-slate-300"
            >
              <option value="All">All cameras</option>
              {cameraOptions.map((camera) => (
                <option key={camera.id} value={camera.name}>
                  {camera.name}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 tracking-wider">Keyword</label>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
              <input
                type="text"
                value={searchKeyword}
                onChange={(event) => onSearchKeywordChange(event.target.value)}
                placeholder="FALL, room, corridor"
                className="w-full pl-9 pr-4 py-2 bg-[#020817] border border-slate-800 rounded-lg text-xs text-white placeholder-slate-600"
              />
            </div>
          </div>
        </div>
      </div>
      <div className="flex-1 bg-[#071329] border border-slate-800 rounded-2xl overflow-hidden flex flex-col">
        <div className="px-5 py-3 bg-slate-900/30 border-b border-slate-800 flex justify-between text-xs text-slate-400">
          <span className="font-semibold">Results ({filteredHistory.length})</span>
          <span className="text-[10px]">Recent archive view</span>
        </div>
        <div className="flex-1 overflow-y-auto divide-y divide-slate-800">
          {filteredHistory.length === 0 ? (
            <div className="py-20 text-center">
              <p className="text-xs text-slate-500">No playback records match this filter.</p>
            </div>
          ) : (
            filteredHistory.map((log) => (
              <div key={log.id} className="p-4 flex items-center justify-between hover:bg-slate-800/10">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center text-slate-400">
                    <Video className="w-4 h-4" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">{log.label}</h4>
                    <div className="flex items-center gap-2 text-[9px] text-slate-500 mt-1 font-mono">
                      <span>Location: {log.camera}</span>
                      <span>/</span>
                      <span>2026-05-{log.timestamp % 2 === 0 ? '25' : '26'} {log.time}</span>
                    </div>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => onOpenIncident(log)}
                    className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg text-[10px] cursor-pointer"
                  >
                    Open
                  </button>
                  <button
                    onClick={() => alert('MP4 download ready')}
                    className="p-1.5 hover:bg-slate-800 text-slate-400 hover:text-white rounded border border-slate-800 cursor-pointer"
                    title="Download"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
