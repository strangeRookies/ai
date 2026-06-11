import { useEffect, useMemo, useState } from 'react';
import type { AiEvent } from '../../../shared/types/aiEvent';
import {
  aiEventFingerprint,
  findCameraForAiEvent,
  getEventTypeKorean,
  getSeverityTone,
} from '../../../shared/utils/aiAlerts';
import type { LiveCamera } from '../data/cameras';
import type { IncidentAlert } from '../types/dashboard';

interface UseDashboardAlertsParams {
  acknowledgedAiEventIds: ReadonlySet<string>;
  dangerAiEvents: readonly AiEvent[];
  liveCameras: readonly LiveCamera[];
  onConfirmAiEvent: (event: AiEvent) => void;
}

interface HistoryFilters {
  searchDate: 'today' | 'week' | 'month';
  searchCamera: string;
  searchKeyword: string;
}

export function useDashboardAlerts({
  acknowledgedAiEventIds,
  dangerAiEvents,
  liveCameras,
  onConfirmAiEvent,
}: UseDashboardAlertsParams) {
  const [alerts, setAlerts] = useState<IncidentAlert[]>([]);

  useEffect(() => {
    if (dangerAiEvents.length === 0) return;

    setAlerts((prev) => {
      let updated = [...prev];
      let changed = false;

      for (const event of dangerAiEvents) {
        const fingerprint = aiEventFingerprint(event);
        const exists = updated.some((alert) => alert.id === fingerprint);

        if (!exists) {
          const cameraObj = findCameraForAiEvent(liveCameras, event);
          const cameraName = cameraObj ? cameraObj.name : event.camera_id;
          const timeString = new Date(event.timestamp * 1000).toTimeString().split(' ')[0];
          const eventType = event.event_type.toUpperCase();
          const label = `${eventType} (${getEventTypeKorean(event.event_type)}) Alert`;
          const severity = getSeverityTone(event.severity);
          const isAcknowledged = acknowledgedAiEventIds.has(fingerprint);

          updated = [
            {
              id: fingerprint,
              time: timeString,
              timestamp: event.timestamp * 1000,
              camera: cameraName,
              type: eventType,
              label,
              severity,
              status: isAcknowledged ? 'resolved' : 'new',
            },
            ...updated,
          ];
          changed = true;
        }
      }

      updated = updated.map((alert) => {
        if (acknowledgedAiEventIds.has(alert.id) && alert.status === 'new') {
          changed = true;
          return { ...alert, status: 'resolved' as const };
        }
        return alert;
      });

      return changed ? updated : prev;
    });
  }, [acknowledgedAiEventIds, dangerAiEvents, liveCameras]);

  const activeTenMinAlerts = useMemo(
    () => alerts.filter((alert) => Date.now() - alert.timestamp <= 10 * 60 * 1000),
    [alerts],
  );

  const getFilteredHistory = useMemo(
    () => (filters: HistoryFilters) => alerts.filter((alert) => {
      if (
        filters.searchKeyword
        && !alert.label.toLowerCase().includes(filters.searchKeyword.toLowerCase())
        && !alert.camera.includes(filters.searchKeyword)
      ) {
        return false;
      }

      if (filters.searchCamera !== 'All' && alert.camera !== filters.searchCamera) return false;

      const age = Date.now() - alert.timestamp;
      if (filters.searchDate === 'today' && age > 86400000) return false;
      if (filters.searchDate === 'week' && age > 7 * 86400000) return false;
      if (filters.searchDate === 'month' && age > 30 * 86400000) return false;
      return true;
    }),
    [alerts],
  );

  const resolveAlert = (id: string) => {
    setAlerts((prev) => prev.map((alert) => (
      alert.id === id ? { ...alert, status: 'resolved' as const } : alert
    )));

    if (!id.includes(':')) return;
    const matchingEvent = dangerAiEvents.find((event) => aiEventFingerprint(event) === id);
    if (matchingEvent) onConfirmAiEvent(matchingEvent);
  };

  return {
    alerts,
    activeTenMinAlerts,
    getFilteredHistory,
    resolveAlert,
  };
}
