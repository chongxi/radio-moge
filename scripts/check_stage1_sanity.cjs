// Exercise real exported clouds in a browser, including raw PLY preservation.
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const linked = state => {
  for (const key of ['cameras', 'targets'])
    for (let b = 1; b < 3; b++)
      state[key][0].forEach((v, i) => assert(Math.abs(v - state[key][b][i]) < 1e-8));
};
(async () => {
  const base = process.argv[2], output = process.argv[3];
  fs.mkdirSync(output, {recursive: true});
  const manifest = await (await fetch(base + 'manifest.json')).json();
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_EXECUTABLE,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader', '--use-angle=swiftshader']});
  try {
    const page = await browser.newPage({viewport: {width: 1600, height: 1050}, acceptDownloads: true});
    const errors = [], failed = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('response', r => {if (r.status() >= 400) failed.push([r.url(), r.status()]);});
    await page.goto(base);
    await page.waitForFunction(() => document.body.dataset.ready === 'true');
    const first = await page.evaluate(() => __sanity()); linked(first);
    assert.equal(first.sources.length, 8); assert.equal(first.samples.length, 51);
    assert.equal(await page.locator('.cloud canvas').count(), 3);
    assert(first.counts.every(n => n > 32));
    const pixels = await page.locator('#view0 canvas').evaluate(c => {
      const copy = document.createElement('canvas'); copy.width = c.width; copy.height = c.height;
      const ctx = copy.getContext('2d'); ctx.drawImage(c, 0, 0);
      const p = ctx.getImageData(0, 0, c.width, c.height).data, colors = new Set();
      for (let i = 0; i < p.length; i += 64) colors.add(`${p[i]},${p[i+1]},${p[i+2]}`);
      return colors.size;
    });
    assert(pixels > 100);
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    const box = await page.locator('#view0 canvas').boundingBox();
    await page.mouse.move(box.x + box.width * .4, box.y + box.height * .5);
    await page.mouse.down(); await page.mouse.move(box.x + box.width * .6, box.y + box.height * .6, {steps: 8}); await page.mouse.up();
    const rotated = await page.evaluate(() => __sanity()); linked(rotated);
    assert.notDeepEqual(rotated.cameras[0], first.cameras[0]);
    await page.mouse.wheel(0, 90); await page.waitForTimeout(150);
    const zoomed = await page.evaluate(() => __sanity()); linked(zoomed);
    assert.notDeepEqual(zoomed.cameras[0], rotated.cameras[0]);
    await page.locator('#scale').selectOption('relative');
    assert.match(await page.locator('#protocol').innerText(), /不能据此判断尺度/);
    const event = page.waitForEvent('download'); await page.locator('#download0').click();
    const download = await event, binary = fs.readFileSync(await download.path());
    const end = binary.indexOf('end_header\n') + 11;
    assert.equal(binary.length - end, first.counts[0] * 15);
    const cloud = manifest.samples.find(s => s.id === first.sample).clouds.reference.reference;
    const original = zlib.gunzipSync(Buffer.from(await (await fetch(base + cloud.file)).arrayBuffer()));
    for (let i = 0; i < first.counts[0]; i++) for (let k = 0; k < 3; k++) {
      assert.equal(binary.readFloatLE(end + i * 15 + k * 4), original.readFloatLE(i * 12 + k * 4));
      assert.equal(binary[end + i * 15 + 12 + k], original[first.counts[0] * 12 + i * 3 + k]);
    }
    await page.locator('#overlay').check(); assert(await page.locator('#card1').isHidden());
    assert(await page.locator('#card2').isHidden());
    await page.screenshot({path: path.join(output, 'overlay.png'), fullPage: true});
    await page.locator('#overlay').uncheck(); await page.locator('#color').selectOption('rgb');
    await page.locator('#scale').selectOption('raw'); await page.locator('#support').selectOption('full');
    const full = await page.evaluate(() => __sanity());
    assert(full.counts[1] >= first.counts[1]);
    await page.locator('#support').selectOption('reference');
    const visited = [];
    for (const dataset of first.sources) {
      await page.locator('#dataset').selectOption(dataset);
      const selected = manifest.samples.filter(s => s.dataset === dataset);
      for (const s of selected) {
        await page.locator('#sample').selectOption(s.id);
        await page.waitForFunction(id => document.body.dataset.ready === 'true' && __sanity().sample === id, s.id);
        const state = await page.evaluate(() => __sanity()); linked(state);
        assert(state.counts.every(n => n > 32));
        if (s.label_kind === 'teacher_pseudo') assert.match(await page.locator('#reference-note').innerText(), /不是独立 GT/);
        if (s.excluded || s.native_flag || s.external_flag) assert.match(await page.locator('#reference-note').innerText(), /标记或排除/);
        if (s.id === selected[0].id) await page.screenshot({path: path.join(output, dataset + '.png'), fullPage: true});
        visited.push(s.id);
      }
      console.log('checked', dataset, selected.length);
    }
    await page.locator('#dataset').selectOption('all');
    await page.waitForFunction(() => document.body.dataset.ready === 'true');
    await page.locator('#search').fill(String(manifest.samples[0].ordinal));
    await page.waitForFunction(id => document.body.dataset.ready === 'true' && __sanity().sample === id, first.sample);
    assert.equal(await page.locator('#sample option').count(), 1);
    await page.locator('#search').fill('');
    await page.waitForFunction(() => document.body.dataset.ready === 'true');
    await page.locator('#next').click();
    await page.waitForFunction(id => document.body.dataset.ready === 'true' && __sanity().sample === id, manifest.samples[1].id);
    await page.locator('#prev').click();
    await page.waitForFunction(id => document.body.dataset.ready === 'true' && __sanity().sample === id, first.sample);
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []); assert.deepEqual(failed, []);
    const result = {stage: 'passed', images: visited.length, sources: first.sources, pixels,
      synchronizedOrbitZoom: true, overlay: true, fixedSupportAndFullSwitch: true,
      rawPLYExactEvenDuringNormalizedDisplay: true, sourceFilter: true, search: true,
      previousNext: true, mobileNoOverflow: true, errors, failedRequests: failed};
    fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally {await browser.close();}
})().catch(e => {console.error(e); process.exit(1);});
