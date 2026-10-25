// Usage: node run.js <outdir> [json options overrides]
// Serves the mod root (so /res/*.wgsl resolves), runs the harness in headless Chromium
// (SwiftShader WebGPU) and writes PNGs + a log.
const { chromium } = require('playwright');
const http = require('http');
const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '../..');
const outDir = path.resolve(process.argv[2] || 'out');
const overrides = process.argv[3] ? JSON.parse(process.argv[3]) : {};
fs.mkdirSync(outDir, { recursive: true });

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/tools/harness/index.html';
  const f = path.join(root, p);
  if (!f.startsWith(root) || !fs.existsSync(f)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'Content-Type': f.endsWith('.html') ? 'text/html' : 'text/plain' });
  res.end(fs.readFileSync(f));
});

(async () => {
  await new Promise(r => server.listen(0, r));
  const port = server.address().port;
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--use-angle=swiftshader', '--use-webgpu-adapter=swiftshader', '--ignore-gpu-blocklist', '--enable-webgpu-developer-features'] });
  const page = await browser.newPage();
  page.on('console', m => console.log('[page]', m.text()));
  page.on('pageerror', e => console.log('[pageerror]', e.message));
  await page.goto(`http://localhost:${port}/`);
  await page.waitForFunction('window.harnessReady === true');
  const opts = Object.assign({
    halfRes: true, rays: 2, steps: 20, rayLength: 450, thickness: 30, intensity: 1.0, bounce: 0.6,
    aoRadius: 90, aoStrength: 0.6, sky: 0.5, emissive: 0.5, bleed: 1.0, historyFrames: 24, clampK: 3.0, denoise: 0.5,
    frames: 32, capture: [0, 31], orbit: false,
  }, overrides);
  const t0 = Date.now();
  const result = await page.evaluate(o => window.runTest(o), opts);
  console.log('elapsed ms', Date.now() - t0);
  console.log('stats', JSON.stringify(result.stats));
  for (const m of result.log) console.log('LOG', m);
  for (const m of result.errors) console.log('GPU ERROR', m);
  for (const [name, url] of Object.entries(result.images)) {
    fs.writeFileSync(path.join(outDir, name + '.png'), Buffer.from(url.split(',')[1], 'base64'));
  }
  console.log('wrote', Object.keys(result.images).join(', '));
  await browser.close(); server.close();
})().catch(e => { console.error(e); process.exit(1); });
