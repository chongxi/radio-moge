// Real WebGL check using Chromium's software GPU when no display GPU is available.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
function assertLinked(cameras){
  for(const key of ['position','target']) cameras[0][key].forEach((v,i)=>
    assert(Math.abs(v-cameras[1][key][i])<1e-10,'3D cameras must remain synchronized'));
}
(async()=>{
  const output=process.argv[3]||'/tmp/radio-moge-3d-check';fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1512,height:1100},deviceScaleFactor:1,acceptDownloads:true});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.goto(process.argv[2]||'http://127.0.0.1:7862',{waitUntil:'domcontentloaded'});
    await page.waitForSelector('#cloud-view[aria-busy="false"]',{timeout:90000});
    assert(!await page.locator('#cloud-view').isHidden());
    assert.equal(await page.locator('.cloud-stage canvas').count(),2);
    const initial=await page.evaluate(()=>cloudViewer.debug());
    assert(initial.counts.after>10000);assert(initial.counts.teacher>10000);
    assertLinked(initial.cameras);
    // A renderer must produce image content, not just mount an empty canvas.
    const renderColors=await page.locator('#cloud-after canvas').evaluate(c=>{
      const flat=document.createElement('canvas');flat.width=c.width;flat.height=c.height;
      const ctx=flat.getContext('2d');ctx.drawImage(c,0,0);
      const rgba=ctx.getImageData(0,0,c.width,c.height).data,colors=new Set();
      for(let i=0;i<rgba.length;i+=128)colors.add(rgba[i]+','+rgba[i+1]+','+rgba[i+2]);
      return colors.size;
    });assert(renderColors>50);
    await page.screenshot({path:path.join(output,'clouds-desktop.png'),fullPage:true});
    await page.locator('#cloud-after canvas').scrollIntoViewIfNeeded();
    const box=await page.locator('#cloud-after canvas').boundingBox();
    await page.mouse.move(box.x+box.width*.5,box.y+box.height*.5);await page.mouse.down();
    await page.mouse.move(box.x+box.width*.7,box.y+box.height*.6,{steps:10});await page.mouse.up();
    const rotated=await page.evaluate(()=>cloudViewer.debug());
    assert.notDeepEqual(rotated.cameras[0].position,initial.cameras[0].position);
    assertLinked(rotated.cameras);
    await page.mouse.wheel(0,100);await page.waitForTimeout(200);
    const zoomed=await page.evaluate(()=>cloudViewer.debug());
    assert.notDeepEqual(zoomed.cameras[0].position,rotated.cameras[0].position);
    assertLinked(zoomed.cameras);
    await page.locator('#cloud-layout').selectOption('overlay');
    assert(await page.locator('#cloud-card-teacher').isHidden());
    assert.equal(await page.locator('#cloud-color').inputValue(),'model');
    await page.screenshot({path:path.join(output,'clouds-overlay.png'),fullPage:true});
    await page.locator('#cloud-camera').selectOption('shared');
    assert.match(await page.locator('#cloud-projection-note').textContent(),/diagnostic/);
    await page.locator('#cloud-density').selectOption('1');
    const full=await page.evaluate(()=>cloudViewer.debug());assert(full.counts.after>initial.counts.after*3.8);
    const pendingDownload=page.waitForEvent('download');await page.locator('#download-after').click();
    const download=await pendingDownload;assert(download.suggestedFilename().endsWith('.ply'));
    const buffer=fs.readFileSync(await download.path()),end=buffer.indexOf('end_header\n')+11;
    assert.equal(buffer.length-end,full.counts.after*15);
    assert.match(buffer.subarray(0,end).toString(),/binary_little_endian/);
    const sharedAfter=full.positions.after;
    await page.locator('#cloud-camera').selectOption('native');
    const native=await page.evaluate(()=>cloudViewer.debug());
    assert.notDeepEqual(native.positions.after,sharedAfter);
    await page.locator('#cloud-layout').selectOption('side');await page.locator('#cloud-color').selectOption('rgb');
    await page.locator('#cloud-density').selectOption('2');
    await page.locator('#next').click();await page.waitForFunction(()=>cloudViewer?.debug().sample==='nyuv2_0004');
    await page.waitForSelector('#cloud-view[aria-busy="false"]');
    assert.notEqual((await page.evaluate(()=>cloudViewer.debug())).sample,initial.sample);
    await page.locator('[data-mode="grid"]').click();assert(await page.locator('#cloud-view').isHidden());
    await page.locator('[data-mode="cloud"]').click();await page.waitForSelector('#cloud-view[aria-busy="false"]');
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'clouds-mobile.png'),fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
    if(await page.locator('#experiment-comparison').count() && await page.locator('#experiment-comparison').isVisible()){
      const sample=await page.evaluate(()=>state.sample.id);
      const oldFingerprint=await page.evaluate(()=>state.manifest.run.checkpoint_sha256);
      await page.locator('#experiment-select').selectOption('mlp');
      await page.waitForURL('**/mlp/**');
      await page.waitForSelector('#cloud-view[aria-busy="false"]',{timeout:90000});
      assert.equal(await page.evaluate(()=>state.sample.id),sample);
      assert.notEqual(await page.evaluate(()=>state.manifest.run.checkpoint_sha256),oldFingerprint);
      assert.equal(await page.locator('#agreement-image').textContent(),
        await page.evaluate(()=>state.comparison.current.teacher_agreement.samples[state.sample.id].abs_rel.toFixed(4)));
      assert.match(await page.locator('#experiment-note').textContent(),/step-0 initialization/);
      await page.locator('#experiment-select').selectOption('fusion');
      await page.waitForURL(url=>!url.pathname.includes('/mlp/'));
      await page.waitForSelector('#cloud-view[aria-busy="false"]',{timeout:90000});
      assert.equal(await page.evaluate(()=>state.sample.id),sample);
      assert.equal(await page.evaluate(()=>state.manifest.run.checkpoint_sha256),oldFingerprint);
      assert.equal(await page.locator('#experiment-results tr').count(),4);
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'clouds-check.json'),JSON.stringify({passed:true,initial,renderColors,fullCounts:full.counts,errors},null,2));
    console.log('3D checks passed: rendered RGB clouds, linked orbit/zoom, overlay, native/shared cameras, full density, binary PLY, navigation, mobile.');
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
