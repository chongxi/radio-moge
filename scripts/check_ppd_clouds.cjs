// Validate the actual offline WebGL viewer, its camera controls and PLY export.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
function linked(s){for(const key of ['cameras','targets'])s[key][0].forEach((v,i)=>assert(Math.abs(v-s[key][1][i])<1e-8))}
(async()=>{
  const output=process.argv[3];fs.mkdirSync(output,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,
    args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1100},acceptDownloads:true});
    const errors=[],network=[];page.on('pageerror',e=>errors.push(e.message));
    page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
    await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
    await page.waitForFunction(()=>document.body.dataset.ready==='true');
    const first=await page.evaluate(()=>__ppdCloudTest());linked(first);
    assert.deepEqual(first.methods,[2,3]);assert(first.counts.every(n=>n>1000));
    assert.equal(await page.locator('.cloud canvas').count(),2);
    const colors=await page.locator('#view0 canvas').evaluate(c=>{
      const flat=document.createElement('canvas');flat.width=c.width;flat.height=c.height;
      const ctx=flat.getContext('2d');ctx.drawImage(c,0,0);
      const pixels=ctx.getImageData(0,0,c.width,c.height).data,colors=new Set();
      for(let i=0;i<pixels.length;i+=64)colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
      return colors.size;
    });assert(colors>100);
    await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
    const box=await page.locator('#view0 canvas').boundingBox();
    await page.mouse.move(box.x+box.width*.4,box.y+box.height*.4);await page.mouse.down();
    await page.mouse.move(box.x+box.width*.6,box.y+box.height*.6,{steps:8});await page.mouse.up();
    const rotated=await page.evaluate(()=>__ppdCloudTest());linked(rotated);assert.notDeepEqual(rotated.cameras[0],first.cameras[0]);
    await page.mouse.wheel(0,100);await page.waitForTimeout(150);
    const zoomed=await page.evaluate(()=>__ppdCloudTest());linked(zoomed);assert.notDeepEqual(zoomed.cameras[0],rotated.cameras[0]);
    await page.locator('#side').click();linked(await page.evaluate(()=>__ppdCloudTest()));
    await page.locator('#color').selectOption('normal');
    await page.screenshot({path:path.join(output,'geometry-side.png'),fullPage:true});
    await page.locator('#overlay').check();assert(await page.locator('#card1').isHidden());
    await page.locator('#reset').click();await page.screenshot({path:path.join(output,'overlay.png'),fullPage:true});
    await page.locator('#overlay').uncheck();await page.locator('#color').selectOption('rgb');
    const methods=[];
    for(let i=0;i<5;i++){await page.locator('#right').selectOption(String(i));const s=await page.evaluate(()=>__ppdCloudTest());assert.equal(s.methods[1],i);assert(s.counts[1]>1000);methods.push(i)}
    await page.locator('#right').selectOption('3');
    await page.locator('#density').selectOption('2');const sparse=await page.evaluate(()=>__ppdCloudTest());assert(sparse.counts[0]<first.counts[0]*.3);
    await page.locator('#density').selectOption('1');
    const pending=page.waitForEvent('download');await page.locator('#download1').click();
    const download=await pending,blob=fs.readFileSync(await download.path());
    const end=blob.indexOf('end_header\n')+11;assert(end>11);
    assert.match(blob.subarray(0,end).toString(),/binary_little_endian/);
    assert.equal(blob.length-end,first.counts[1]*15);
    for(let i=0;i<first.positions[1].length;i++)assert(Math.abs(blob.readFloatLE(end+Math.floor(i/3)*15+(i%3)*4)-first.positions[1][i])<1e-5);
    const visited=[];
    for(const resolution of ['768','480']){
      await page.locator('#resolution').selectOption(resolution);
      for(let index=0;index<16;index++){
        await page.locator('#sample').selectOption(String(index));
        const id=String(index).padStart(2,'0')+'-'+(index<12?'nyuv2':'structured3d')+'-'+resolution+'x'+(resolution==='768'?1024:640);
        await page.waitForFunction(id=>document.body.dataset.ready==='true'&&__ppdCloudTest().sample===id,id);
        const s=await page.evaluate(()=>__ppdCloudTest());linked(s);assert(s.counts.every(n=>n>1000));visited.push(s.sample);
      }
    }
    await page.locator('#resolution').selectOption('768');await page.locator('#sample').selectOption('15');
    await page.waitForFunction(()=>document.body.dataset.ready==='true'&&__ppdCloudTest().sample==='15-structured3d-768x1024');
    await page.locator('#color').selectOption('normal');await page.locator('#reset').click();
    await page.screenshot({path:path.join(output,'empty-room-geometry.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
    const result={passed:true,cases:visited.length,methods:methods.length,webglColors:colors,firstCounts:first.counts,
      synchronizedOrbitZoom:true,geometryLighting:true,sideView:true,overlay:true,plyCountAndCoordinatesVerified:true,
      mobileNoOverflow:true,externalRequests:network.length,errors};
    fs.writeFileSync(path.join(output,'browser-verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
