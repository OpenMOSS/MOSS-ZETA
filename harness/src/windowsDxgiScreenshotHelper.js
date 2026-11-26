const path = require('node:path');
const { spawn } = require('node:child_process');
const readline = require('node:readline');
const { performance } = require('node:perf_hooks');

let child = null, starting = null, clock = null, readyInfo = null, serial = 0, busy = false;
const pending = new Map();
function skip(message) { return Object.assign(new Error(message), { nonFatalCaptureSkip: true, captureBackend: 'windows-dxgi' }); }
function budgetMs() {
  const value = Number(process.env.AGI_GTA_CAPTURE_DEADLINE_MS || 100);
  if (!Number.isFinite(value) || value <= 0) throw Error('Invalid AGI_GTA_CAPTURE_DEADLINE_MS');
  return value;
}
function closeWindowsDxgiScreenshotHelper() {
  const old = child; child = null; starting = null; clock = null; busy = false;
  for (const p of pending.values()) { clearTimeout(p.timer); clearTimeout(p.watchdog); p.reject(skip('DXGI helper closed')); }
  pending.clear();
  if (old && !old.killed) old.kill();
}
function rpc(body, timeout) {
  const id = String(++serial), began = performance.now();
  const deadline = began + timeout;
  return new Promise((resolve, reject) => {
    const p = { resolve, reject, began, deadline, type: body.type, expired: false };
    p.timer = setTimeout(() => { p.expired = true; reject(skip('DXGI capture deadline exceeded')); }, timeout);
    // A late reply clears the reservation but is never delivered to a newer request.
    p.watchdog = setTimeout(() => closeWindowsDxgiScreenshotHelper(), Math.max(3000, timeout + 1000));
    pending.set(id, p);
    child.stdin.write(`${JSON.stringify({ ...body, id })}\n`, (error) => {
      if (error && pending.has(id)) closeWindowsDxgiScreenshotHelper();
    });
  });
}
async function warmWindowsDxgiScreenshotHelper() {
  if (starting) return starting;
  if (child && clock) return { readyInfo, clock };
  starting = (async () => {
    const executable = process.env.AGI_GTA_CAPTURE_PYTHON || process.env.AGI_GTA_GAMEPAD_PYTHON || 'python';
    const script = path.join(__dirname, '..', 'scripts', 'windows-dxgi-capture.py');
    const proc = spawn(executable, ['-u', script], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] });
    child = proc;
    let stderr = '';
    proc.stderr.on('data', (data) => { stderr = (stderr + data.toString()).slice(-2000); });
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => { reject(Error('DXGI startup timed out')); closeWindowsDxgiScreenshotHelper(); }, 15000);
      const lines = readline.createInterface({ input: proc.stdout });
      lines.on('line', (line) => {
        let response;
        try { response = JSON.parse(line); } catch { return; }
        if (response.event === 'ready') { readyInfo = response; clearTimeout(timer); resolve(); return; }
        const p = pending.get(String(response.id));
        if (!p) return;
        pending.delete(String(response.id)); clearTimeout(p.timer); clearTimeout(p.watchdog);
        if (p.type === 'capture') busy = false;
        const ended = performance.now();
        if (p.expired) return;
        if (ended > p.deadline) { p.reject(skip('DXGI reply arrived after deadline')); return; }
        if (!response.ok) { p.reject(skip(`DXGI capture: ${response.error}`)); return; }
        p.resolve({ response, began: p.began, ended });
      });
      proc.once('error', (error) => { clearTimeout(timer); reject(error); if (child === proc) closeWindowsDxgiScreenshotHelper(); });
      proc.once('exit', () => { clearTimeout(timer); reject(Error(`DXGI helper exited: ${stderr}`)); if (child === proc) closeWindowsDxgiScreenshotHelper(); });
    });
    const samples = [];
    for (let i = 0; i < 5; i++) {
      const r = await rpc({ type: 'clock' }, 1000);
      samples.push({ offsetMs: r.response.qpcMs - (r.began + r.ended) / 2, rttMs: r.ended - r.began });
    }
    clock = samples.sort((a, b) => a.rttMs - b.rttMs)[0];
    clock.uncertaintyMs = clock.rttMs / 2;
    return { readyInfo, clock };
  })().finally(() => { starting = null; });
  return starting;
}
async function captureWindowsDxgiScreenshot(options = {}) {
  const begin = performance.now(), budget = budgetMs();
  let startupTimer;
  try {
    await Promise.race([warmWindowsDxgiScreenshotHelper(), new Promise((_, reject) => {
      startupTimer = setTimeout(() => reject(skip('DXGI not warm before capture deadline')), budget);
    })]);
  } finally { clearTimeout(startupTimer); }
  const remaining = budget - (performance.now() - begin);
  if (remaining <= 0) throw skip('DXGI startup consumed capture deadline');
  if (busy) throw skip('Previous DXGI capture still in flight');
  busy = true;
  const { response: r, began, ended } = await rpc({ type: 'capture', displayId: options.displayId,
    width: options.width || 1280, height: options.height || 720, quality: options.quality || 85 }, remaining);
  const buffer = Buffer.from(r.data, 'base64');
  const ready = performance.now(), framePerfMs = r.frameQpcMs - clock.offsetMs;
  if (ready - begin > budget) throw skip('DXGI image decoding exceeded capture deadline');
  return { buffer, mimeType: 'image/jpeg', width: r.width, height: r.height, backend: 'windows-dxgi',
    display: { id: options.displayId, sourceId: options.displayId },
    contentBounds: { left: r.contentLeft, top: r.contentTop, width: r.contentWidth, height: r.contentHeight },
    cursorIncluded: r.cursorIncluded, cursor: r.cursorX >= 0 ? { x: r.cursorX, y: r.cursorY } : null,
    timing: { helper_total_ms: r.helperTotalMs, helper_jpeg_encode_ms: r.jpegEncodeMs,
      helper_base64_ms: r.base64Ms, ipc_round_trip_ms: ended - began,
      queue_or_transport_ms: Math.max(0, ended - began - r.helperTotalMs),
      dxgi_acquire_ms: r.acquireMs, dxgi_resize_ms: r.resizeMs,
      frame_perf_ms: framePerfMs, frame_timestamp_source: r.frameTimestampSource,
      frame_age_at_request_ms: begin - framePerfMs, frame_age_at_ready_ms: ready - framePerfMs,
      capture_request_to_ready_ms: ready - begin, capture_deadline_ms: budget,
      capture_clock_uncertainty_ms: clock.uncertaintyMs, capture_clock_offset_ms: clock.offsetMs }
  };
}
process.once('exit', closeWindowsDxgiScreenshotHelper);
module.exports = { captureWindowsDxgiScreenshot, warmWindowsDxgiScreenshotHelper, closeWindowsDxgiScreenshotHelper };
