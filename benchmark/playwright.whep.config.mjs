export default {
  testDir: '.',
  testMatch: 'webrtc_whep_smoke.spec.mjs',
  timeout: Number(process.env.WHEP_SMOKE_TIMEOUT_MS || '30000') + 10000,
  use: {
    browserName: 'chromium',
    channel: 'chrome',
    headless: true,
  },
};
