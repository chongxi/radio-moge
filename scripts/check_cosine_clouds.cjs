// Real browser validation of the self-contained page, without a web server.
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const linked = s => {
  for (const key of ['cameras', 'targets'])
    for (let branch = 1; branch < s[key].length; branch++)
      s[key][0].forEach((v, i) => assert(Math.abs(v - s[key][branch][i]) < 1e-8));
};
(async () => {
  const output = process.argv[3]; fs.mkdirSync(output, {recursive: true});
  const browser = await chromium.launch({headless: true,
    executablePath: process.env.CHROMIUM_EXECUTABLE,
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--enable-unsafe-swiftshader', '--use-angle=swiftshader']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1120}, acceptDownloads: true});
    const errors = [], network = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (/^https?:/.test(r.url())) network.push(r.url()); });
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    await page.waitForFunction(() => document.body.dataset.ready === 'true');
    const first = await page.evaluate(() => __cloudTest()); linked(first);
    const branches = await page.evaluate(() => CLOUD_DATA.names.length);
    assert.equal(await page.locator('.cloud canvas').count(), branches);
    assert(first.counts.every(n => n > 10000));
    const coloredPixels = await page.locator('#view0 canvas').evaluate(c => {
      const canvas = document.createElement('canvas'); canvas.width = c.width; canvas.height = c.height;
      const ctx = canvas.getContext('2d'); ctx.drawImage(c, 0, 0);
      const pixels = ctx.getImageData(0, 0, c.width, c.height).data, colors = new Set();
      for (let i = 0; i < pixels.length; i += 64) colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
      return colors.size;
    });
    assert(coloredPixels > 100);
    await page.screenshot({path: path.join(output, 'desktop.png'), fullPage: true});
    const box = await page.locator('#view0 canvas').boundingBox();
    await page.mouse.move(box.x + box.width * .4, box.y + box.height * .5);
    await page.mouse.down(); await page.mouse.move(box.x + box.width * .6, box.y + box.height * .6, {steps: 8}); await page.mouse.up();
    const rotated = await page.evaluate(() => __cloudTest()); linked(rotated);
    assert.notDeepEqual(rotated.cameras[0], first.cameras[0]);
    await page.mouse.wheel(0, 90); await page.waitForTimeout(150);
    const zoomed = await page.evaluate(() => __cloudTest()); linked(zoomed);
    assert.notDeepEqual(zoomed.cameras[0], rotated.cameras[0]);
    await page.locator('#overlay').check(); assert(await page.locator('#card1').isHidden());
    for (let i = 2; i < branches; i++) assert(await page.locator('#card'+i).isHidden());
    await page.locator('#reset').click();
    await page.screenshot({path: path.join(output, 'overlay.png'), fullPage: true});
    await page.locator('#camera').selectOption('shared');
    const shared = await page.evaluate(() => __cloudTest());
    assert.notDeepEqual(shared.positions[0], first.positions[0]);
    await page.locator('#camera').selectOption('own');
    await page.locator('#overlay').uncheck(); await page.locator('#color').selectOption('rgb');
    await page.locator('#density').selectOption('2');
    const sparse = await page.evaluate(() => __cloudTest());
    assert(sparse.counts[0] < first.counts[0] * .3);
    await page.locator('#density').selectOption('1');
    const downloaded = page.waitForEvent('download'); await page.locator('#download0').click();
    const download = await downloaded, binary = fs.readFileSync(await download.path());
    const headerEnd = binary.indexOf('end_header\n') + 11;
    assert.equal(binary.length - headerEnd, first.counts[0] * 15);
    assert.match(binary.subarray(0, headerEnd).toString(), /binary_little_endian/);
    const visited = [];
    for (let i = 0; i < 20; i++) {
      await page.locator('#sample').selectOption(String(i));
      await page.waitForFunction(n => document.body.dataset.ready === 'true' && __cloudTest().sample === CLOUD_DATA.samples[n].id, i);
      const state = await page.evaluate(() => __cloudTest()); linked(state);
      assert(state.counts.every(n => n > 1000)); visited.push(state.sample);
    }
    await page.locator('#next').click();
    await page.waitForFunction(id => __cloudTest().sample === id && document.body.dataset.ready === 'true', first.sample);
    await page.setViewportSize({width: 390, height: 844});
    await page.screenshot({path: path.join(output, 'mobile.png'), fullPage: true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(errors, []); assert.deepEqual(network, []);
    const result = {stage: 'passed', images: visited.length, branches, coloredPixels, firstCounts: first.counts,
      synchronizedOrbitZoom: true, overlay: true, cameraSwitch: true, plyVerified: true,
      mobileNoOverflow: true, externalRequests: network.length, errors};
    fs.writeFileSync(path.join(output, 'browser-verification.json'), JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
