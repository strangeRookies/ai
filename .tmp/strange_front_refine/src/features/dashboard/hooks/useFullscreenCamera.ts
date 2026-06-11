import { useCallback, useEffect, useRef, useState } from 'react';

type CameraElement = HTMLDivElement | null;

function isTrackedFullscreenElement(
  refs: Record<string, CameraElement>,
  element: Element | null,
) {
  if (!element) return false;
  return Object.values(refs).some((ref) => ref === element || (ref?.contains(element) ?? false));
}

export function useFullscreenCamera() {
  const refs = useRef<Record<string, CameraElement>>({});
  const [activeCameraId, setActiveCameraId] = useState<string | null>(null);

  const syncFullscreenState = useCallback(() => {
    const currentElement = document.fullscreenElement;
    if (!currentElement) {
      setActiveCameraId(null);
      return;
    }

    const nextActiveId = Object.entries(refs.current).find(([, element]) => (
      element === currentElement || (element?.contains(currentElement) ?? false)
    ))?.[0] ?? null;

    setActiveCameraId(nextActiveId);
  }, []);

  const exitFullscreen = useCallback(async () => {
    if (!document.fullscreenElement) {
      setActiveCameraId(null);
      return;
    }

    try {
      await document.exitFullscreen();
    } catch {
      setActiveCameraId(null);
    }
  }, []);

  const enterFullscreen = useCallback(async (cameraId: string) => {
    const target = refs.current[cameraId];
    if (!target || !target.requestFullscreen) return;

    if (document.fullscreenElement && activeCameraId && activeCameraId !== cameraId) {
      await exitFullscreen();
    }

    if (document.fullscreenElement === target || target.contains(document.fullscreenElement)) {
      setActiveCameraId(cameraId);
      return;
    }

    try {
      await target.requestFullscreen();
      setActiveCameraId(cameraId);
    } catch {
      syncFullscreenState();
    }
  }, [activeCameraId, exitFullscreen, syncFullscreenState]);

  const toggleFullscreen = useCallback(async (cameraId: string) => {
    if (activeCameraId === cameraId && document.fullscreenElement) {
      await exitFullscreen();
      return;
    }
    await enterFullscreen(cameraId);
  }, [activeCameraId, enterFullscreen, exitFullscreen]);

  const registerCameraElement = useCallback((cameraId: string) => (element: CameraElement) => {
    refs.current[cameraId] = element;

    if (!element && activeCameraId === cameraId && isTrackedFullscreenElement(refs.current, document.fullscreenElement)) {
      void exitFullscreen();
    }
  }, [activeCameraId, exitFullscreen]);

  useEffect(() => {
    document.addEventListener('fullscreenchange', syncFullscreenState);
    document.addEventListener('fullscreenerror', syncFullscreenState);

    return () => {
      document.removeEventListener('fullscreenchange', syncFullscreenState);
      document.removeEventListener('fullscreenerror', syncFullscreenState);

      if (isTrackedFullscreenElement(refs.current, document.fullscreenElement)) {
        void document.exitFullscreen().catch(() => undefined);
      }
    };
  }, [syncFullscreenState]);

  return {
    activeCameraId,
    enterFullscreen,
    exitFullscreen,
    isFullscreenActive: activeCameraId !== null,
    isFullscreenCamera: (cameraId: string) => activeCameraId === cameraId,
    registerCameraElement,
    toggleFullscreen,
  };
}
