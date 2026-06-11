# Dashboard Refactor Summary

## Changed files

- `src/features/dashboard/pages/UserDashboard.tsx`
- `src/features/dashboard/hooks/useDashboardAlerts.ts`
- `src/features/dashboard/types/dashboard.ts`
- `src/features/dashboard/utils/dashboardStatus.ts`
- `src/features/dashboard/components/DashboardHomeView.tsx`
- `src/features/dashboard/components/DashboardAlertsView.tsx`
- `src/features/dashboard/components/DashboardHistoryView.tsx`
- `src/features/dashboard/components/DashboardCameraManagementView.tsx`
- `src/features/dashboard/components/DashboardMyPageView.tsx`
- `src/features/dashboard/components/DashboardQnaView.tsx`
- `src/features/dashboard/modals/AddCameraModal.tsx`
- `src/features/dashboard/modals/NewInquiryModal.tsx`
- `src/features/dashboard/modals/IncidentPlaybackModal.tsx`
- `scripts/verify-ai-acknowledge-contract.mjs`

## Extracted components and hooks

- `DashboardHomeView`: live camera area + AI danger panel
- `DashboardAlertsView`: recent alert board
- `DashboardHistoryView`: filterable playback history
- `DashboardCameraManagementView`: registered camera management
- `DashboardMyPageView`: profile / password / notification / account tabs
- `DashboardQnaView`: inquiry list and detail view
- `AddCameraModal`: camera registration modal
- `NewInquiryModal`: inquiry creation modal
- `IncidentPlaybackModal`: playback overlay
- `useDashboardAlerts`: alert sync, history filtering, acknowledgement state sync
- `dashboard.ts`: shared dashboard-specific types
- `dashboardStatus.ts`: menu/category constants and small formatting helpers

## Preserved behavior

- `UserDashboard.tsx` remains the top-level container/orchestrator.
- Existing MQTT/EQMS -> backend -> frontend alert flow still enters through `useAiAlertActions`.
- Alert cards, history, and playback still derive from the same AI event stream.
- Camera cards still render through `LiveCameraGrid`.
- No new fake detected state was added during this refactor.

## Next improvements

- Extract fullscreen camera state into a dedicated hook/component instead of relying on local card fullscreen only.
- Normalize dashboard copy back to product-approved Korean strings after encoding cleanup.
- Tighten alert deduplication and acknowledgement UX now that the container file is smaller.
- Add buildable tests around the extracted views and the new `useDashboardAlerts` hook.
