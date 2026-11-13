import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';

const [, , webRoot, chrome, screenshot] = process.argv;
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright-core');
const root = resolve(webRoot);
const contentTypes = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm', '.ttf': 'font/ttf' };
const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname;
  if (path === '/favicon.ico') {
    response.writeHead(204).end();
    return;
  }
  const file = resolve(root, `.${path === '/' ? '/index.html' : path}`);
  if (!file.startsWith(root + sep)) {
    response.writeHead(404).end();
    return;
  }
  try {
    const bytes = await readFile(file);
    response.writeHead(200, { 'Content-Type': contentTypes[extname(file)] ?? 'application/octet-stream' });
    response.end(bytes);
  } catch {
    response.writeHead(404).end();
  }
});
await new Promise((done, fail) => server.listen(0, '127.0.0.1', done).once('error', fail));

let browser;
try {
  browser = await chromium.launch({ executablePath: chrome, args: ['--enable-unsafe-swiftshader'] });
  const page = await browser.newPage({ viewport: { width: 1160, height: 720 } });
  const errors = [];
  const failedRequests = [];
  page.on('response', response => {
    if (response.status() >= 400) failedRequests.push(new URL(response.url()).pathname);
  });
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/?backend=webgl2`);
  await page.waitForFunction(() => {
    return performance.getEntriesByName('web.first.presented').length || document.body.dataset.wasm === 'failed';
  }, null, { timeout: 20000 }).catch(() => {});
  const result = await page.evaluate(() => ({
    wasm: document.body.dataset.wasm,
    context: document.body.dataset.graphicsContext,
    marks: performance.getEntriesByType('mark').map(mark => mark.name).filter(name => name.startsWith('web.')),
    errorText: document.body.dataset.wasm === 'failed' ? document.body.innerText : undefined,
  }));
  await page.screenshot({ path: screenshot });
  console.log(JSON.stringify({ ...result, errors, failedRequests }, null, 2));
  if (result.wasm !== 'loaded' || result.context !== 'webgl2' || !result.marks.includes('web.first.presented') || errors.length || failedRequests.length) {
    process.exitCode = 1;
  }
} finally {
  await browser?.close();
  server.close();
}
