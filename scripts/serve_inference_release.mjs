// Exact-file download server for a private BuildHub preview, behind its login gateway.
// Never point this at a repository root or credentials directory.
import { createServer } from 'node:http';
import { createReadStream, statSync } from 'node:fs';
import { resolve } from 'node:path';

const [directory, portText = '7872'] = process.argv.slice(2);
if (!directory) throw new Error('Usage: node serve_inference_release.mjs RELEASE_MODEL_DIR PORT');
const names = ['adapter.pt', 'config.json', 'radio_moge_inference-0.1.0-py3-none-any.whl',
  'README.md', 'THIRD_PARTY.md'];
const files = new Map(names.map(name => {
  const path = resolve(directory, name);
  const stat = statSync(path);
  if (!stat.isFile()) throw new Error(`Not a file: ${name}`);
  return [`/${name}`, { path, name, stat }];
}));
const server = createServer((req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  const file = files.get(pathname);
  if (!file) { res.writeHead(404); return res.end(); }
  if (!['GET', 'HEAD'].includes(req.method)) {
    res.writeHead(405, { Allow: 'GET, HEAD' }); return res.end();
  }
  const { path, name, stat } = file;
  const etag = `"${stat.size}-${stat.mtimeMs}"`;
  let start = 0, end = stat.size - 1, status = 200;
  if (req.headers.range && (!req.headers['if-range'] || req.headers['if-range'] === etag)) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
    if (match && (match[1] || match[2])) {
      start = match[1] ? Number(match[1]) : Math.max(0, stat.size - Number(match[2]));
      end = match[1] && match[2] ? Math.min(Number(match[2]), stat.size - 1) : stat.size - 1;
    } else { start = stat.size; }
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) ||
        start > end || start >= stat.size) {
      res.writeHead(416, { 'Content-Range': `bytes */${stat.size}`, 'Content-Length': '0' });
      return res.end();
    }
    status = 206;
  }
  const headers = {
    'Content-Type': 'application/octet-stream',
    'Content-Disposition': `attachment; filename="${name}"`,
    'Content-Length': String(end - start + 1), 'Accept-Ranges': 'bytes',
    'Cache-Control': 'private, no-cache, no-transform', ETag: etag,
    'X-Content-Type-Options': 'nosniff',
  };
  if (status === 206) headers['Content-Range'] = `bytes ${start}-${end}/${stat.size}`;
  res.writeHead(status, headers);
  if (req.method === 'HEAD') return res.end();
  const stream = createReadStream(path, { start, end, highWaterMark: 64 * 1024 });
  stream.on('error', () => res.destroy());
  res.on('close', () => stream.destroy());
  stream.pipe(res);
});
server.maxConnections = 4;
server.headersTimeout = 15000;
server.requestTimeout = 0;
server.listen(Number(portText), '0.0.0.0', () => console.log(`Release files on port ${portText}`));
