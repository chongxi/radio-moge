// Validate actual offline WebGL, all masks, synchronized views, and PLY export.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
function linked(s){for(const key of ['cameras','targets'])for(let side=1;side<3;side++)s[key][0].forEach((v,i)=>assert(Math.abs(v-s[key][side][i])<1e-8))}
(async()=>{
  const output=process.argv[3],expectations=JSON.parse(fs.readFileSync(process.argv[4]));fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,
    args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1536,height:1120},acceptDownloads:true});
    const errors=[],network=[];page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    await page.waitForFunction(()=>document.body.dataset.ready==='true');
    const first=await page.evaluate(()=>__ppdEdgesTest());linked(first);
    assert.equal(first.ppdFiltered,false);assert.equal(first.threshold,2);
    assert.equal(first.counts[0],first.counts[2]);assert(first.counts[1]<first.counts[0]);
    assert(first.previewSubsetExact);assert.equal(await page.locator('.cloud canvas').count(),3);
    const colors=[];
    for(let side=0;side<3;side++)colors.push(await page.locator('#view'+side+' canvas').evaluate(c=>{
      const flat=document.createElement('canvas');flat.width=c.width;flat.height=c.height;
      const ctx=flat.getContext('2d');ctx.drawImage(c,0,0);
      const pixels=ctx.getImageData(0,0,c.width,c.height).data,colors=new Set();
      for(let i=0;i<pixels.length;i+=64)colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
      return colors.size;
    }));assert(colors.every(n=>n>100));
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    const box=await page.locator('#view1 canvas').boundingBox();
    await page.mouse.move(box.x+box.width*.4,box.y+box.height*.4);await page.mouse.down();
    await page.mouse.move(box.x+box.width*.6,box.y+box.height*.6,{steps:8});await page.mouse.up();
    const rotated=await page.evaluate(()=>__ppdEdgesTest());linked(rotated);assert.notDeepEqual(rotated.cameras[0],first.cameras[0]);
    await page.mouse.wheel(0,100);await page.waitForTimeout(100);
    const zoomed=await page.evaluate(()=>__ppdEdgesTest());linked(zoomed);assert.notDeepEqual(zoomed.cameras[0],rotated.cameras[0]);
    await page.locator('#side').click();await page.locator('#color').selectOption('depth');
    await page.screenshot({path:path.join(output,'side-depth.png'),fullPage:true});
    await page.locator('#removed').check();const over=await page.evaluate(()=>__ppdEdgesTest());
    linked(over);assert.deepEqual(over.counts,first.counts);assert.equal(over.removed[1],first.counts[0]-first.counts[1]);
    await page.screenshot({path:path.join(output,'removed-red.png'),fullPage:true});
    await page.locator('#removed').uncheck();
    for(const [button,count,positions] of [['#download1',first.counts[1],first.positions[1]],['#download-removed',first.counts[0]-first.counts[1],null]]){
      const pending=page.waitForEvent('download');await page.locator(button).click();
      const download=await pending,blob=fs.readFileSync(await download.path());
      const end=blob.indexOf('end_header\n')+11;assert(end>11);assert.equal(blob.length-end,count*15);
      assert.match(blob.subarray(0,end).toString(),new RegExp('element vertex '+count+'\\n'));
      if(positions)for(let i=0;i<positions.length;i++)assert(Math.abs(blob.readFloatLE(end+Math.floor(i/3)*15+(i%3)*4)-positions[i])<1e-5);
    }
    await page.locator('#ppd-filter').check();
    let comparisons=0;
    for(const expected of expectations.cases){
      const [index,,resolution]=expected.id.split('-');
      await page.locator('#resolution').selectOption(resolution.split('x')[0]);
      await page.locator('#sample').selectOption(String(+index));
      await page.waitForFunction(id=>document.body.dataset.ready==='true'&&__ppdEdgesTest().sample===id,expected.id);
      for(let threshold=0;threshold<4;threshold++){
        await page.locator('#threshold').selectOption(String(threshold));
        const s=await page.evaluate(()=>__ppdEdgesTest());linked(s);assert(s.previewSubsetExact);
        assert.deepEqual(s.counts,[expected.base_count,...expected.filtered_counts[threshold]]);
        assert.equal(s.counts[1]+s.removed[1],s.counts[0]);assert.equal(s.counts[2]+s.removed[2],s.counts[0]);
        comparisons++;
      }
    }
    await page.locator('#resolution').selectOption('480');await page.locator('#sample').selectOption('6');
    await page.waitForFunction(()=>document.body.dataset.ready==='true'&&__ppdEdgesTest().sample==='06-nyuv2-480x640');
    await page.locator('#threshold').selectOption('2');await page.locator('#ppd-filter').uncheck();
    await page.locator('#color').selectOption('rgb');await page.locator('#side').click();
    await page.screenshot({path:path.join(output,'scene-06-side.png'),fullPage:true});
    const dense=await page.evaluate(()=>__ppdEdgesTest());await page.locator('#density').selectOption('2');
    const sparse=await page.evaluate(()=>__ppdEdgesTest());assert(sparse.counts[0]<dense.counts[0]*.3);
    await page.locator('#density').selectOption('1');await page.locator('#color').selectOption('model');
    await page.setViewportSize({width:390,height:844});await page.locator('#oblique').click();
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const result={passed:true,cases:expectations.cases.length,thresholdComparisons:comparisons,
      webglColors:colors,firstCounts:first.counts,synchronizedOrbitZoom:true,removedOverlay:true,
      ppdSameFilter:true,previewSubsetCoordinatesExact:true,plyCountAndCoordinatesVerified:true,
      mobileNoOverflow:true,externalRequests:network.length,errors};
    fs.writeFileSync(path.join(output,'browser-verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
