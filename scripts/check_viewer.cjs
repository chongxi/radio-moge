// Optional browser check: install playwright separately, then pass URL and screenshot directory.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
(async()=>{
  const url=process.argv[2]||'http://127.0.0.1:7860';
  const output=process.argv[3]||'/tmp/radio-moge-viewer-check';fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
  try {
    const page=await browser.newPage({viewport:{width:1512,height:1100},deviceScaleFactor:1});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const gridURL=new URL(url);gridURL.hash='mode=grid';
    await page.goto(gridURL.href,{waitUntil:'domcontentloaded'});
    await page.waitForSelector('main[aria-busy="false"]',{timeout:60000});
    const manifest=await (await page.request.get(new URL('manifest.json',url).href)).json();
    assert.equal(await page.locator('#total-images').textContent(),'290');
    assert.equal(await page.locator('.panel canvas').count(),6);
    for(const branch of ['before','after','teacher']){
      const scores=manifest.samples.map(sample=>sample.metrics[branch].abs_rel);
      const average=scores.reduce((sum,value)=>sum+value,0)/scores.length;
      assert.equal(await page.locator('#'+branch+'-score').textContent(),average.toFixed(3));
    }
    assert((await page.locator('#run-steps').textContent()).includes(String(manifest.run.steps)));
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    await page.locator('#canvas-after').hover({position:{x:50,y:50}});
    assert.match(await page.locator('#pixel-values').textContent(),/After training: [0-9.]+ m/);
    const checked=await page.evaluate(()=>{
      const i=50*state.sample.width+50;
      return {pixel:state.data.after[i],before:state.data.before[i],channels:state.manifest.channels,bytes:state.data.after.byteLength};
    });
    assert(checked.pixel>0 && checked.before>0);assert.equal(checked.bytes,640*480*4);
    await page.locator('#search').fill('kitchen');
    const matches=await page.locator('.scene-item').count();assert(matches>0 && matches<290);
    await page.locator('.scene-item').first().click();await page.waitForSelector('main[aria-busy="false"]');
    assert.match(await page.locator('#scene-name').textContent(),/kitchen/);
    await page.locator('[data-mode="swipe"]').click();
    await page.locator('#swipe-right').selectOption('teacher');
    await page.locator('#swipe-range').fill('25');await page.locator('#swipe-range').dispatchEvent('input');
    assert.equal(await page.locator('#split-value').textContent(),'25%');
    await page.locator('#scale').selectOption('fixed');assert.equal(await page.locator('#far').textContent(),'10.00 m');
    await page.screenshot({path:path.join(output,'swipe.png'),fullPage:true});
    await page.locator('#search').fill('nonexistent-scene-xyz');assert(await page.locator('#empty').isVisible());
    await page.locator('#clear-search').click();assert.equal(await page.locator('.scene-item').count(),290);
    await page.locator('#sort').selectOption('worst');
    const ordered=await page.evaluate(()=>state.filtered.map(s=>s.metrics.after.abs_rel));
    assert(ordered.every((v,i)=>i===0||ordered[i-1]>=v));
    await page.locator('#scene-filter').selectOption('bedroom');await page.locator('.scene-item').first().click();
    await page.waitForSelector('main[aria-busy="false"]');
    await page.locator('[data-mode="grid"]').click();await page.locator('#error-reference').selectOption('ground_truth');
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    assert.equal(errors.length,0,errors.join('\n'));
    fs.writeFileSync(path.join(output,'browser-check.json'),JSON.stringify({passed:true,images:290,kitchenMatches:matches,errors,checked},null,2));
    console.log('Browser checks passed: 290 images, decoded float32 depths, metrics, filters, swipe, errors, mobile, no JS exceptions.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
