import { Flame, Video } from 'lucide-react';
import type { AiConnectionState } from '../../../hooks/useAiEvents';
import { AiDangerPanel } from '../../../components/dashboard/AiDangerPanel';
import type { AiEvent } from '../../../shared/types/aiEvent';
import type { LiveCamera } from '../data/cameras';
import { LiveCameraGrid } from './LiveCameraGrid';

interface DashboardHomeViewProps {
  acknowledgedAiEventIds: ReadonlySet<string>;
  connectionState: AiConnectionState;
  dangerAiEvents: readonly AiEvent[];
  focusedLiveCameras: readonly LiveCamera[];
  onCameraSelect: (camera: LiveCamera) => void;
  onConfirmAiEvent: (event: AiEvent) => void;
  onEmergency: () => void;
  onFocusAiEvent: (event: AiEvent) => void;
}

export function DashboardHomeView({
  acknowledgedAiEventIds,
  connectionState,
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
            실시간 CCTV 모니터링
          </h2>
          <span className="text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-2.5 py-0.5 rounded border border-emerald-500/20">
            카메라 상태를 실시간으로 확인하고 있습니다.
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
            <h3 className="text-base font-bold text-white">실시간 AI 이벤트</h3>
          </div>
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            <AiDangerPanel
              events={dangerAiEvents}
              acknowledgedEventIds={acknowledgedAiEventIds}
              connectionState={connectionState}
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
          긴급 출동 요청
        </button>
      </div>
    </div>
  );
}
