import { Flame, Video } from 'lucide-react';
import { AiDangerPanel } from '../../../components/dashboard/AiDangerPanel';
import type { AiEvent } from '../../../shared/types/aiEvent';
import type { LiveCamera } from '../data/cameras';
import { LiveCameraGrid } from './LiveCameraGrid';

interface DashboardHomeViewProps {
  acknowledgedAiEventIds: ReadonlySet<string>;
  dangerAiEvents: readonly AiEvent[];
  focusedLiveCameras: readonly LiveCamera[];
  onCameraSelect: (camera: LiveCamera) => void;
  onConfirmAiEvent: (event: AiEvent) => void;
  onEmergency: () => void;
  onFocusAiEvent: (event: AiEvent) => void;
}

export function DashboardHomeView({
  acknowledgedAiEventIds,
  dangerAiEvents,
  focusedLiveCameras,
  onCameraSelect,
  onConfirmAiEvent,
  onEmergency,
  onFocusAiEvent,
}: DashboardHomeViewProps) {
  return (
    <div className="flex-1 flex overflow-hidden">
      <div className="flex-1 p-4 overflow-y-auto space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-sm font-extrabold text-white flex items-center gap-2">
            <Video className="w-4 h-4 text-blue-400" />
            Live CCTV Monitoring
          </h2>
          <span className="text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-2.5 py-0.5 rounded border border-emerald-500/20">
            Dashboard connected
          </span>
        </div>

        <LiveCameraGrid
          cameras={[...focusedLiveCameras]}
          onCameraClick={onCameraSelect}
        />
      </div>

      <div className="w-72 bg-[#020817] border-l border-slate-800/50 flex flex-col flex-shrink-0">
        <div className="flex-1 bg-[#071329] m-3 mb-0 rounded-xl flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-800/50">
            <h3 className="text-base font-bold text-white">Real-time AI alerts</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            <AiDangerPanel
              events={dangerAiEvents}
              acknowledgedEventIds={acknowledgedAiEventIds}
              onFocus={onFocusAiEvent}
              onConfirm={onConfirmAiEvent}
            />
          </div>
        </div>
        <button
          onClick={onEmergency}
          className="mx-3 my-3 py-4 bg-[#dc2626] hover:bg-red-500 text-white font-extrabold rounded-xl text-sm flex items-center justify-center gap-2 transition-all cursor-pointer"
        >
          <Flame className="w-4 h-4 fill-white/20" />
          Emergency dispatch
        </button>
      </div>
    </div>
  );
}
