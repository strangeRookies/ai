import { AlertTriangle, Expand, Minimize2, Signal, SignalZero, Video } from 'lucide-react';
import { useState } from 'react';
import type { LiveCamera } from '../data/cameras';
import { useFullscreenCamera } from '../hooks/useFullscreenCamera';

interface LiveCameraGridProps {
  cameras: LiveCamera[];
  className?: string;
  compact?: boolean;
  onCameraClick?: (camera: LiveCamera) => void;
}

function statusStyle(camera: LiveCamera) {
  if (camera.connectionStatus === 'offline') {
    return {
      border: 'border-slate-700',
      badge: 'bg-slate-700/80 text-slate-200 border-slate-600',
      dot: 'bg-slate-400',
      label: '연결 끊김',
      icon: SignalZero,
    };
  }
  if (camera.connectionStatus === 'connecting') {
    return {
      border: 'border-amber-500/60',
      badge: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
      dot: 'bg-amber-400 animate-pulse',
      label: '연결 중',
      icon: Signal,
    };
  }
  if (camera.eventStatus === 'danger') {
    return {
      border: 'border-rose-500/80 shadow-[0_0_0_1px_rgba(244,63,94,0.45)]',
      badge: 'bg-rose-500/15 text-rose-200 border-rose-500/40',
      dot: 'bg-rose-500 animate-ping',
      label: camera.eventLabel || '이상 상황',
      icon: AlertTriangle,
    };
  }
  if (camera.eventStatus === 'warning') {
    return {
      border: 'border-amber-500/70',
      badge: 'bg-amber-500/15 text-amber-200 border-amber-500/35',
      dot: 'bg-amber-400 animate-pulse',
      label: camera.eventLabel || '주의 필요',
      icon: AlertTriangle,
    };
  }
  return {
    border: 'border-slate-800',
    badge: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/25',
    dot: 'bg-emerald-400 animate-pulse',
    label: '모니터링 중',
    icon: Signal,
  };
}

function gridClass(count: number) {
  if (count <= 1) return 'grid-cols-1';
  if (count === 2) return 'grid-cols-1 xl:grid-cols-2';
  return 'grid-cols-1 md:grid-cols-2';
}

function connectionLabel(status: LiveCamera['connectionStatus']) {
  if (status === 'online') return '정상';
  if (status === 'connecting') return '연결 중';
  return '오프라인';
}

function CameraStream({ camera }: { camera: LiveCamera }) {
  const [failed, setFailed] = useState(false);
  const unavailable = failed || camera.connectionStatus === 'offline';

  return (
    <>
      {!failed && (
        <img
          src={camera.streamUrl}
          alt={`${camera.name} 라이브 영상`}
          className={`absolute inset-0 h-full w-full object-cover ${unavailable ? 'opacity-25 grayscale' : ''}`}
          onError={() => setFailed(true)}
        />
      )}
      {unavailable && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#030712]/90 text-slate-500">
          <SignalZero className="w-10 h-10 mb-3 text-slate-600" />
          <span className="text-xs font-extrabold tracking-wider text-slate-300">카메라 연결 끊김</span>
          <span className="mt-1 text-[10px] text-slate-500">
            {failed ? '카메라 연결 상태를 확인해주세요.' : '카메라 연결을 기다리는 중입니다.'}
          </span>
        </div>
      )}
    </>
  );
}

export function LiveCameraGrid({ cameras, className = '', compact = false, onCameraClick }: LiveCameraGridProps) {
  const visibleCameras = cameras.slice(0, 4);
  const {
    exitFullscreen,
    isFullscreenCamera,
    registerCameraElement,
    toggleFullscreen,
  } = useFullscreenCamera();

  return (
    <div className={`grid ${gridClass(visibleCameras.length)} gap-3 ${className}`}>
      {visibleCameras.map((camera) => {
        const style = statusStyle(camera);
        const StatusIcon = style.icon;
        const fullscreen = isFullscreenCamera(camera.id);

        return (
          <div
            ref={registerCameraElement(camera.id)}
            key={camera.id}
            className={`group overflow-hidden rounded-xl border ${style.border} bg-[#0f172a] text-left transition-colors cursor-pointer hover:border-blue-500/50 focus:outline-none focus:ring-2 focus:ring-blue-500/60`}
            onClick={() => onCameraClick?.(camera)}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') onCameraClick?.(camera);
            }}
          >
            <div className={`relative bg-black ${compact ? 'aspect-video' : visibleCameras.length === 1 ? 'aspect-[16/8]' : 'aspect-video'}`}>
              <CameraStream camera={camera} />

              <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/75 px-2 py-1 text-[10px] font-extrabold text-rose-300 backdrop-blur">
                <span className="h-1.5 w-1.5 rounded-full bg-rose-500 animate-pulse" />
                실시간
              </div>

              <div className={`absolute right-2 top-2 flex items-center gap-1.5 rounded border px-2 py-1 text-[10px] font-extrabold backdrop-blur ${style.badge}`}>
                <StatusIcon className="h-3 w-3" />
                {style.label}
              </div>

              {fullscreen && (
                <button
                  type="button"
                  title="전체 화면 종료"
                  onClick={(event) => {
                    event.stopPropagation();
                    void exitFullscreen();
                  }}
                  className="absolute bottom-2 right-2 z-10 flex items-center gap-1 rounded bg-black/70 px-2 py-1 text-[10px] font-bold text-white backdrop-blur hover:bg-black/85"
                >
                  <Minimize2 className="h-3 w-3" />
                  닫기
                </button>
              )}

              {camera.eventStatus === 'danger' && camera.connectionStatus !== 'offline' && (
                <div className="absolute inset-0 border-2 border-rose-500 bg-rose-600/10">
                  <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-rose-600 px-3 py-1 text-xs font-extrabold text-white shadow-lg">
                    {camera.eventLabel || '이상 상황'}
                  </div>
                </div>
              )}
            </div>

            <div className="flex items-center justify-between gap-3 px-3 py-2.5">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <Video className="h-3.5 w-3.5 flex-shrink-0 text-blue-400" />
                  <p className="truncate text-xs font-extrabold text-white">{camera.name}</p>
                </div>
                <p className="mt-0.5 truncate text-[10px] font-semibold text-slate-500">{camera.location}</p>
              </div>
              <div className="flex flex-shrink-0 items-center gap-1.5">
                <div className="flex items-center gap-1.5 rounded bg-slate-950/70 px-2 py-1 text-[10px] font-bold text-slate-300">
                  <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
                  {connectionLabel(camera.connectionStatus)}
                </div>
                <button
                  type="button"
                  title={fullscreen ? '전체 화면 종료' : '전체 화면으로 보기'}
                  onClick={(event) => {
                    event.stopPropagation();
                    void toggleFullscreen(camera.id);
                  }}
                  className="p-1.5 rounded bg-slate-900 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                >
                  {fullscreen ? <Minimize2 className="h-3 w-3" /> : <Expand className="h-3 w-3" />}
                </button>
              </div>
            </div>
          </div>
        );
      })}

      {visibleCameras.length === 0 && (
        <div className="flex aspect-video items-center justify-center rounded-xl border border-dashed border-slate-700 bg-[#0f172a] text-xs font-semibold text-slate-500">
          표시할 카메라가 없습니다.
        </div>
      )}
    </div>
  );
}
