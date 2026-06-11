import { readFileSync } from 'node:fs';

const apiSource = readFileSync('src/app/api/aiIncidentRequests.ts', 'utf8');
const hookSource = readFileSync('src/hooks/useAiAlertActions.ts', 'utf8');
const dashboardSource = readFileSync('src/features/dashboard/pages/UserDashboard.tsx', 'utf8');
const alertsHookSource = readFileSync('src/features/dashboard/hooks/useDashboardAlerts.ts', 'utf8');

const checks = [
  ['preFrames is 150', apiSource.includes('preFrames: 150')],
  ['postFrames is 150', apiSource.includes('postFrames: 150')],
  ['totalFrames is 300', apiSource.includes('totalFrames: 300')],
  ['ack endpoint is used', apiSource.includes('/api/incidents/${encodeURIComponent(request.eventId)}/acknowledge-and-record')],
  ['POST method is used', apiSource.includes("method: 'POST'")],
  ['Confirm calls backend API', hookSource.includes('acknowledgeAndRequestRecording(event, username)')],
  ['Confirm acknowledges local event', hookSource.includes('next.add(fp)') || hookSource.includes('next.add(aiEventFingerprint')],
  ['dashboard uses alert actions hook', dashboardSource.includes('useAiAlertActions({ userType, username, liveCameras, focusHome })')],
  ['dashboard uses extracted alerts hook', dashboardSource.includes('useDashboardAlerts({')],
  ['dashboard alerts hook uses fingerprint ids', alertsHookSource.includes('aiEventFingerprint(event)')],
];

const failed = checks.filter(([, passed]) => !passed);
if (failed.length > 0) {
  for (const [name] of failed) {
    console.error(`FAIL ${name}`);
  }
  process.exit(1);
}

for (const [name] of checks) {
  console.log(`PASS ${name}`);
}
