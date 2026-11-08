// Exercise the delivered offline 3D page, including geometry and downloads.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright-core');
const linked=s=>s.cameras[0].forEach((v,i)=>assert(Math.abs(v-s.cameras[1][i])<1e-8));
(async()=>{
 const output=process.argv[3];fs.mkdirSync(output,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_EXECUTABLE,
  args:['--no-sandbox','--disable-dev-shm-usage','--enable-unsafe-swiftshader','--use-angle=swiftshader']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1150},acceptDownloads:true}),errors=[],network=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url())});
  await page.goto(pathToFileURL(path.resolve(process.argv[2])).href);
  await page.waitForFunction(()=>document.body.dataset.ready==='true');
  const first=await page.evaluate(()=>__ppdCacheTest());linked(first);assert.deepEqual(first.methods,[0,1]);assert.equal(first.frame,179);
  assert(first.counts.every(n=>n>1000));assert.equal(await page.locator('.view canvas').count(),2);
  const colors=await page.locator('#view0 canvas').evaluate(c=>{
   const flat=document.createElement('canvas');flat.width=c.width;flat.height=c.height;const ctx=flat.getContext('2d');ctx.drawImage(c,0,0);
   const data=ctx.getImageData(0,0,c.width,c.height).data,set=new Set();for(let i=0;i<data.length;i+=64)set.add(`${data[i]},${data[i+1]},${data[i+2]}`);return set.size;
  });assert(colors>100);
  await page.screenshot({path:path.join(output,'desktop.png'),fullPage:true});
  const box=await page.locator('#view0 canvas').boundingBox();
  await page.mouse.move(box.x+box.width*.4,box.y+box.height*.4);await page.mouse.down();await page.mouse.move(box.x+box.width*.6,box.y+box.height*.6,{steps:6});await page.mouse.up();
  const rotated=await page.evaluate(()=>__ppdCacheTest());linked(rotated);assert.notDeepEqual(first.cameras[0],rotated.cameras[0]);
  await page.mouse.wheel(0,100);await page.waitForTimeout(150);const zoomed=await page.evaluate(()=>__ppdCacheTest());linked(zoomed);assert.notDeepEqual(rotated.cameras[0],zoomed.cameras[0]);
  await page.locator('#side').click();linked(await page.evaluate(()=>__ppdCacheTest()));
  const visited=[];
  for(const frame of [0,29,59,89,119,149,179,209,239,269,299,359]){
   await page.locator('#frame').selectOption(String(frame));await page.waitForFunction(f=>document.body.dataset.ready==='true'&&__ppdCacheTest().frame===f,frame);
   for(let method=1;method<4;method++){await page.locator('#method').selectOption(String(method));const s=await page.evaluate(()=>__ppdCacheTest());linked(s);assert.deepEqual(s.methods,[0,method]);assert(s.counts.every(n=>n>1000));visited.push([frame,method])}
  }
  await page.locator('#common-mask').uncheck();assert((await page.evaluate(()=>__ppdCacheTest())).counts[1]>=first.counts[1]);
  await page.locator('#common-mask').check();await page.locator('#color').selectOption('model');await page.locator('#front').click();
  await page.screenshot({path:path.join(output,'fixed-first-frame-drift.png'),fullPage:true});
  const selected=await page.evaluate(()=>__ppdCacheTest()),pending=page.waitForEvent('download');await page.locator('#download1').click();
  const download=await pending,blob=fs.readFileSync(await download.path()),end=blob.indexOf('end_header\n')+11;
  assert(end>11);assert.equal(blob.length-end,selected.counts[1]*15);
  for(let i=0;i<selected.positions[1].length;i++)assert(Math.abs(blob.readFloatLE(end+Math.floor(i/3)*15+i%3*4)-selected.positions[1][i])<1e-5);
  await page.locator('#chart').click({position:{x:55,y:30}});await page.waitForFunction(()=>document.body.dataset.ready==='true'&&__ppdCacheTest().frame===0);
  await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(output,'mobile.png'),fullPage:true});
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
  const result={passed:true,frame_method_pairs:visited.length,webgl_colors:colors,synchronized_orbit_zoom:true,mask_toggle:true,
   chart_selection:true,preview_ply_coordinates_verified:true,mobile_no_overflow:true,external_requests:0,errors};
  fs.writeFileSync(path.join(output,'browser-verification.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exitCode=1});
