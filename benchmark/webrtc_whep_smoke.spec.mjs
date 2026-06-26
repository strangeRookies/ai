import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const port = Number(process.env.WHEP_SMOKE_PAGE_PORT || '8099');
const streamPath = process.env.WHEP_STREAM_PATH || 'cam_01';
const whepUrl = process.env.WHEP_URL || `http://localhost:8889/${streamPath}/whep`;
const timeoutMs = Number(process.env.WHEP_SMOKE_TIMEOUT_MS || '30000');

function startStaticServer() {
  const server = http.createServer(async (request, response) => {
    const requestPath = decodeURIComponent(new URL(request.url, `http://127.0.0.1:${port}`).pathname);
    const filePath = path.normalize(path.join(repoRoot, requestPath));
    if (!filePath.startsWith(repoRoot)) {
      response.writeHead(403).end('forbidden');
      return;
    }
    try {
      const body = await fs.readFile(filePath);
      response.writeHead(200, { 'Content-Type': filePath.endsWith('.html') ? 'text/html' : 'application/octet-stream' });
      response.end(body);
    } catch {
      response.writeHead(404).end('not found');
    }
  });
  return new Promise(resolve => {
    server.listen(port, '127.0.0.1', () => resolve(server));
  });
}

test('MediaMTX WHEP stream reaches browser video playback', async ({ page }) => {
  const server = await startStaticServer();
  try {
    page.on('console', message => console.log(`[browser:${message.type()}] ${message.text()}`));
    await page.goto(`http://127.0.0.1:${port}/benchmark/webrtc_whep_smoke.html?url=${encodeURIComponent(whepUrl)}`);
    await expect.poll(
      async () => page.evaluate(() => window.webrtcSmokeResult),
      { timeout: timeoutMs, intervals: [1000] },
    ).toMatchObject({
      whepPostOk: true,
      iceConnected: true,
      ontrack: true,
      bytesReceivedIncreased: true,
      framesDecodedIncreased: true,
      videoPlaying: true,
    });
    console.log(`final smoke result ${JSON.stringify(await page.evaluate(() => window.webrtcSmokeResult))}`);
  } finally {
    server.close();
  }
});
