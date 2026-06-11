import type { LiveCamera } from '../../features/dashboard/data/cameras';
import type { AiEvent } from '../../hooks/useAiEvents';

export type AiAlertPage = 'admin' | 'company' | 'personal';

export function isAiAlertEnabledRoute(page: AiAlertPage) {
  return page !== 'admin';
}

export function isDangerAiEvent(event: AiEvent) {
  return event.event_type !== 'Normal';
}

export function aiEventKey(event: AiEvent) {
  const trackId = event.track_id ?? 'no-track';
  return `${event.camera_id}:${event.event_type}:${event.timestamp}:${trackId}`;
}

export { aiEventFingerprint } from './aiEventFeed';

export function formatAiEventLabel(event: AiEvent): string {
  const upper = event.event_type.trim().toUpperCase();
  const korean = getEventTypeKorean(event.event_type);
  return `${upper} (${korean})`;
}

export function getSeverityTone(severity: string): 'critical' | 'warning' | 'info' {
  const upper = severity.toUpperCase();
  if (upper === 'CRITICAL' || upper === 'HIGH') return 'critical';
  if (upper === 'MEDIUM') return 'warning';
  return 'info';
}

export function findCameraForAiEvent(cameras: readonly LiveCamera[], event: AiEvent) {
  const eventTokens = cameraIdTokens(event.camera_id);
  return cameras.find((camera) => {
    const cameraTokens = new Set([
      normalizeCameraToken(camera.id),
      normalizeCameraToken(camera.name),
      normalizeCameraToken(camera.location),
      ...cameraIdTokens(camera.id),
      ...cameraIdTokens(camera.name),
    ]);
    return eventTokens.some((token) => cameraTokens.has(token));
  });
}

export function getEventTypeKorean(type: string): string {
  const upper = type.toUpperCase();
  if (upper.includes('FALL')) return '낙상';
  if (upper.includes('FAINT')) return '실신';
  if (upper.includes('COLLAPSE')) return '쓰러짐';
  if (upper.includes('VIOLENCE') || upper.includes('FIGHT')) return '폭력';
  if (upper.includes('CROWD')) return '군집';
  if (upper.includes('FIRE')) return '화재';
  if (upper.includes('INTRUSION')) return '침입';
  return type;
}

export function markAiDangerCameras(cameras: readonly LiveCamera[], events: readonly AiEvent[]) {
  return cameras.map((camera) => {
    const matchingEvent = events.find((event) => {
      if (!isDangerAiEvent(event)) return false;
      const cameraObj = findCameraForAiEvent(cameras, event);
      return cameraObj?.id === camera.id;
    });

    if (!matchingEvent) {
      return camera;
    }

    const typeUpper = matchingEvent.event_type.toUpperCase();
    const koreanLabel = getEventTypeKorean(matchingEvent.event_type);
    return {
      ...camera,
      eventStatus: 'danger' as const,
      eventLabel: `${typeUpper} (${koreanLabel})`,
    };
  });
}

export function focusCameraFirst(cameras: readonly LiveCamera[], focusedCameraId: string | null) {
  if (!focusedCameraId) {
    return [...cameras];
  }
  const focused = cameras.find((camera) => camera.id === focusedCameraId);
  if (!focused) {
    return [...cameras];
  }
  return [focused, ...cameras.filter((camera) => camera.id !== focusedCameraId)];
}

function cameraIdTokens(value: string) {
  const normalized = normalizeCameraToken(value);
  const numeric = normalized.match(/\d+/)?.[0];
  if (!numeric) {
    return [normalized];
  }
  const unpadded = String(Number(numeric));
  const padded = unpadded.padStart(2, '0');
  return [normalized, `camera${unpadded}`, `cam${unpadded}`, `cctv${unpadded}`, `cctv${padded}`];
}

function normalizeCameraToken(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}
