import { AlertTriangle, Camera, Check, Clock } from 'lucide-react';
import type { AiEvent } from '../../hooks/useAiEvents';
import { formatAiEventLabel } from '../../shared/utils/aiAlerts';

interface AiAlertCardProps {
  readonly event: AiEvent;
  readonly acknowledged?: boolean;
  readonly onFocus?: (event: AiEvent) => void;
  readonly onConfirm?: (event: AiEvent) => void;
}

export function AiAlertCard({ event, acknowledged = false, onFocus, onConfirm }: AiAlertCardProps) {
  if (event.event_type === 'Normal') {
    return null;
  }

  const time = new Date(event.timestamp * 1000).toLocaleTimeString();
  const confidence = Number.isFinite(event.confidence) ? event.confidence : event.score;
  const confidencePercent = Math.round(Math.max(confidence, 0) * 100);

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onFocus?.(event)}
      onKeyDown={(keyEvent) => {
        if (keyEvent.key === 'Enter' || keyEvent.key === ' ') {
          keyEvent.preventDefault();
          onFocus?.(event);
        }
      }}
      className={`relative mb-3 w-full overflow-hidden rounded-xl border p-4 text-left backdrop-blur-md transition-colors ${
        acknowledged
          ? 'border-slate-700 bg-slate-900/50 opacity-70'
          : 'border-red-500/50 bg-red-900/30 shadow-[0_0_15px_rgba(239,68,68,0.3)]'
      }`}
    >
      {!acknowledged && <div className="absolute inset-0 bg-red-500/10 animate-pulse" />}

      <div className="relative z-10 flex items-start gap-3">
        <div className={`rounded-full p-2 ${acknowledged ? 'bg-emerald-500/15 text-emerald-300' : 'bg-red-500/20 text-red-400'}`}>
          {acknowledged ? <Check className="h-5 w-5" /> : <AlertTriangle className="h-5 w-5" />}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <h4 className={`text-base font-bold ${acknowledged ? 'text-slate-300' : 'text-red-300'}`}>
              {acknowledged ? '담당자가 확인한 이벤트입니다.' : '이상 상황이 감지되었습니다.'}
            </h4>
            <span className="rounded-md border border-red-500/20 bg-red-950/50 px-2 py-1 font-mono text-xs text-red-300/80">
              {formatAiEventLabel(event)}
            </span>
          </div>

          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-red-200/70">
            <span className="flex items-center gap-1">
              <Camera className="h-3.5 w-3.5" />
              {event.camera_id}
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-3.5 w-3.5" />
              {time}
            </span>
            <span className="text-[10px] text-slate-400">
              신뢰도 {Number.isFinite(confidencePercent) ? `${confidencePercent}%` : '확인 중'}
            </span>
          </div>

          <div className="mt-3 flex items-center justify-between gap-3">
            <span className="text-[10px] font-semibold tracking-wide text-slate-400">
              {acknowledged ? '확인 완료' : '미확인 이벤트'}
            </span>
            {!acknowledged && (
              <button
                type="button"
                onClick={(clickEvent) => {
                  clickEvent.stopPropagation();
                  onConfirm?.(event);
                }}
                className="rounded-lg bg-emerald-600 px-3 py-1.5 text-[10px] font-extrabold text-white transition-colors hover:bg-emerald-500"
              >
                확인 처리
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
