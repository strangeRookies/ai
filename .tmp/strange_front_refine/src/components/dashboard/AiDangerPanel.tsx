import type { ReactNode } from 'react';
import { Shield } from 'lucide-react';
import type { AiConnectionState } from '../../hooks/useAiEvents';
import { aiEventFingerprint } from '../../shared/utils/aiAlerts';
import type { AiEvent } from '../../hooks/useAiEvents';
import { AiAlertCard } from './AiAlertCard';

interface AiDangerPanelProps {
  readonly events: readonly AiEvent[];
  readonly acknowledgedEventIds: ReadonlySet<string>;
  readonly onFocus: (event: AiEvent) => void;
  readonly onConfirm: (event: AiEvent) => void;
  readonly connectionState?: AiConnectionState;
  readonly fallback?: ReactNode;
}

function EmptyState({ connectionState }: { connectionState?: AiConnectionState }) {
  if (connectionState === 'connecting' || connectionState === 'idle') {
    return (
      <div className="py-8 text-center text-slate-500">
        <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-xs font-semibold text-slate-300">이벤트 정보를 불러오는 중입니다.</p>
        <p className="text-[10px] mt-1 text-slate-600">실시간 이벤트 채널에 연결하고 있습니다.</p>
      </div>
    );
  }

  if (connectionState === 'disconnected' || connectionState === 'error') {
    return (
      <div className="py-8 text-center text-slate-500">
        <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p className="text-xs font-semibold text-slate-300">이벤트 채널 연결 상태를 확인해주세요.</p>
        <p className="text-[10px] mt-1 text-slate-600">연결이 복구되면 새로운 이벤트가 여기에 표시됩니다.</p>
      </div>
    );
  }

  return (
    <div className="py-8 text-center text-slate-500">
      <Shield className="w-8 h-8 mx-auto mb-2 opacity-50" />
      <p className="text-xs font-semibold text-slate-300">현재 감지된 이상 상황이 없습니다.</p>
      <p className="text-[10px] mt-1 text-slate-600">실시간 모니터링 중입니다.</p>
    </div>
  );
}

export function AiDangerPanel({
  events,
  acknowledgedEventIds,
  onFocus,
  onConfirm,
  connectionState,
  fallback,
}: AiDangerPanelProps) {
  if (events.length === 0) {
    return fallback ?? <EmptyState connectionState={connectionState} />;
  }

  return (
    <>
      {events.map((event) => (
        <AiAlertCard
          key={aiEventFingerprint(event)}
          event={event}
          acknowledged={acknowledgedEventIds.has(aiEventFingerprint(event))}
          onFocus={onFocus}
          onConfirm={onConfirm}
        />
      ))}
    </>
  );
}
