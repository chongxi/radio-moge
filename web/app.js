'use strict';
const $ = id => document.getElementById(id);
const names = {rgb:'RGB image', before:'Before training', after:'After training', teacher:'MoGe-2', ground_truth:'Measured depth', error:'After · relative error'};
const panelKeys = ['rgb','before','after','teacher','ground_truth','error'];
const cache = new Map();
const state = {manifest:null, filtered:[], sample:null, data:null, rendered:{}, mode:'grid', generation:0, controller:null, split:50};
const fmt = (v,n=3) => Number.isFinite(v) ? v.toFixed(n) : '—';
const pretty = text => text.replaceAll('_',' ');
const valid = v => Number.isFinite(v) && v > 0;
function element(tag, className, text) { const node=document.createElement(tag); if(className)node.className=className; if(text!==undefined)node.textContent=text; return node; }
function status(text) { $('load-status').textContent=text; $('load-status').hidden=!text; }
function updateURL() { if(!state.sample)return; const params=new URLSearchParams({image:state.sample.id, mode:state.mode}); history.replaceState(null,'','#'+params.toString()); }
function createPanels(){
  panelKeys.forEach(key=>{
    const card=element('article','panel '+key); const heading=element('div','panel-title');
    heading.append(element('span','',names[key]));
    heading.append(element('small','tag',key==='before'?'STEP 0':key==='after'?'STEP '+state.manifest.run.steps:key==='rgb'?'INPUT':key==='ground_truth'?'SENSOR':key==='teacher'?'TEACHER':'0–50%'));
    const wrap=element('div','image-wrap'); wrap.dataset.key=key;
    const canvas=element('canvas'); canvas.id='canvas-'+key; canvas.setAttribute('aria-label',names[key]);
    wrap.append(canvas,element('i','cross-h'),element('i','cross-v')); card.append(heading,wrap); $('grid-view').append(card);
    wrap.addEventListener('pointermove',event=>inspect(event,wrap)); wrap.addEventListener('pointerleave',clearInspection);
  });
}
function renderList(){
  const term=$('search').value.trim().toLowerCase(), type=$('scene-filter').value;
  state.filtered=state.manifest.samples.filter(s=>(!type||s.scene_type===type)&&(!term||[s.scene,s.id,s.scene_type].join(' ').toLowerCase().includes(term)));
  const order=$('sort').value;
  if(order!=='original')state.filtered.sort((a,b)=>{
    const av=a.metrics.after.abs_rel, bv=b.metrics.after.abs_rel;
    if(!Number.isFinite(av))return 1;if(!Number.isFinite(bv))return -1;
    return (order==='worst'?-1:1)*(av-bv);
  });
  $('result-count').textContent=state.filtered.length+' / '+state.manifest.samples.length;
  $('empty').hidden=state.filtered.length>0; $('scene-list').replaceChildren();
  const fragment=document.createDocumentFragment();
  state.filtered.forEach(s=>{
    const button=element('button','scene-item'+(state.sample?.id===s.id?' active':'')); button.dataset.id=s.id;
    button.setAttribute('aria-label',s.scene+', '+s.id); button.setAttribute('aria-current',String(state.sample?.id===s.id));
    const thumb=element('img'); thumb.src=s.thumbnail;thumb.alt='';thumb.loading='lazy';thumb.width=160;thumb.height=120;
    const content=element('span','scene-text');content.append(element('span','scene-title',pretty(s.scene)),element('span','scene-meta',s.id.replace('nyuv2_','#')+' · error '+fmt(s.metrics.after.abs_rel)));
    button.append(thumb,content); button.addEventListener('click',()=>select(s)); fragment.append(button);
  });
  $('scene-list').append(fragment);
  updateNavigation();
}
function updateNavigation(){
  const index=state.filtered.findIndex(s=>s.id===state.sample?.id);
  $('position').textContent=index<0?'Not in filter':(index+1)+' / '+state.filtered.length;
  $('previous').disabled=index<=0; $('next').disabled=index<0||index>=state.filtered.length-1;
}
function navigate(offset){const index=state.filtered.findIndex(s=>s.id===state.sample?.id); const sample=state.filtered[index+offset];if(sample)select(sample);}
function showMetrics(){
  $('metrics-body').replaceChildren();
  ['before','after','teacher'].forEach(name=>{
    const m=state.sample.metrics[name],tr=element('tr','metric-'+name);
    tr.append(element('td','',names[name]),element('td','',fmt(m.abs_rel)),element('td','',fmt(m.rmse)),element('td','',Number.isFinite(m.delta1)?fmt(m.delta1*100,1)+'%':'—'));
    $('metrics-body').append(tr);
  });
  const failures=Object.keys(state.sample.failures);
  $('failure-note').hidden=!failures.length;
  $('failure-note').textContent=failures.map(k=>names[k]+': focal recovery failed; depth marked invalid.').join(' ');
}
async function fetchSample(sample, signal){
  if(cache.has(sample.id))return cache.get(sample.id);
  const [response, bitmap]=await Promise.all([fetch(sample.depth,{signal}), fetch(sample.rgb,{signal}).then(async r=>{if(!r.ok)throw Error('RGB image unavailable');return createImageBitmap(await r.blob());})]);
  if(!response.ok) {bitmap.close();throw Error('Depth data unavailable ('+response.status+')');}
  const buffer=await response.arrayBuffer(), pixels=sample.width*sample.height;
  if(buffer.byteLength!==pixels*4*4){bitmap.close();throw Error('Depth file has an unexpected size. Use the provided viewer server so gzip is decoded correctly.');}
  const data={rgb:bitmap};
  state.manifest.channels.forEach((key,i)=>{data[key]=new Float32Array(buffer,i*pixels*4,pixels);});
  cache.set(sample.id,data);
  while(cache.size>5){const first=cache.keys().next().value;cache.get(first).rgb.close();cache.delete(first);}
  return data;
}
async function select(sample){
  const generation=++state.generation;
  state.controller?.abort(); state.controller=new AbortController(); state.sample=sample;state.data=null;
  document.querySelector('main').setAttribute('aria-busy','true');
  $('image-id').textContent=sample.id.toUpperCase()+' · HELD-OUT / '+pretty(sample.scene_type).toUpperCase();
  $('scene-name').textContent=pretty(sample.scene);status('Loading depth maps…'); clearInspection();showMetrics();updateURL();
  document.querySelectorAll('.scene-item').forEach(b=>{const active=b.dataset.id===sample.id;b.classList.toggle('active',active);b.setAttribute('aria-current',String(active));});
  updateNavigation();
  try{
    const data=await fetchSample(sample,state.controller.signal);
    if(generation!==state.generation)return;
    state.data=data;render();status('');
    document.querySelector('main').setAttribute('aria-busy','false');
  }catch(error){if(error.name!=='AbortError'&&generation===state.generation)status('Could not load this image: '+error.message);}
}
function bounds(){return $('scale').value==='fixed'?[0,10]:state.sample.range;}
function depthImage(values,width,height,lo,hi){
  const out=new ImageData(width,height),colors=state.manifest.colormap,den=Math.max(hi-lo,1e-6);
  for(let i=0;i<values.length;i++){
    const value=values[i],color=valid(value)?colors[Math.max(0,Math.min(255,Math.floor((value-lo)/den*255)))]:[255,0,255],j=i*4;
    out.data[j]=color[0];out.data[j+1]=color[1];out.data[j+2]=color[2];out.data[j+3]=255;
  }
  return out;
}
function errorImage(predicted,target,width,height){
  const out=new ImageData(width,height);
  for(let i=0;i<target.length;i++){
    const j=i*4;
    if(!valid(target[i])){out.data.set([255,0,255,255],j);continue;}
    const ratio=valid(predicted[i])?Math.abs(predicted[i]-target[i])/target[i]:1;
    const t=Math.min(1,ratio/.5),a=t<.7?[22,29,43]:[249,115,22],b=t<.7?[249,115,22]:[255,242,178],u=t<.7?t/.7:(t-.7)/.3;
    out.data[j]=a[0]+u*(b[0]-a[0]);out.data[j+1]=a[1]+u*(b[1]-a[1]);out.data[j+2]=a[2]+u*(b[2]-a[2]);out.data[j+3]=255;
  }
  return out;
}
function render(){
  if(!state.data)return;
  const {width,height}=state.sample,[lo,hi]=bounds();
  $('near').textContent=fmt(lo,2)+' m';$('far').textContent=fmt(hi,2)+' m';
  panelKeys.forEach(key=>{
    const canvas=$('canvas-'+key);canvas.width=width;canvas.height=height;
    canvas.parentElement.style.aspectRatio=width+'/'+height;
    const context=canvas.getContext('2d');
    if(key==='rgb')context.drawImage(state.data.rgb,0,0,width,height);
    else if(key==='error')context.putImageData(errorImage(state.data.after,state.data[$('error-reference').value],width,height),0,0);
    else context.putImageData(depthImage(state.data[key],width,height,lo,hi),0,0);
    state.rendered[key]=canvas;
  });
  drawSwipe();
}
function drawSwipe(){
  if(!state.data)return;
  const {width,height}=state.sample,canvas=$('swipe-canvas');canvas.width=width;canvas.height=height;
  $('swipe-stage').style.aspectRatio=width+'/'+height;
  const context=canvas.getContext('2d'),left=$('swipe-left').value,right=$('swipe-right').value;
  context.drawImage(state.rendered[right],0,0);context.save();context.beginPath();context.rect(0,0,width*state.split/100,height);context.clip();context.drawImage(state.rendered[left],0,0);context.restore();
  $('left-tag').textContent=names[left];$('right-tag').textContent=names[right];
  $('swipe-divider').style.left=state.split+'%';$('split-value').textContent=Math.round(state.split)+'%';
}
function setMode(mode){
  state.mode=mode==='swipe'?'swipe':'grid';
  $('grid-view').hidden=state.mode!=='grid';$('swipe-view').hidden=state.mode!=='swipe';
  document.querySelectorAll('[data-mode]').forEach(b=>{const active=b.dataset.mode===state.mode;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  updateURL();drawSwipe();
}
function inspect(event,wrap){
  if(!state.data)return;
  const rect=wrap.getBoundingClientRect(),{width,height}=state.sample;
  const x=Math.max(0,Math.min(width-1,Math.floor((event.clientX-rect.left)/rect.width*width)));
  const y=Math.max(0,Math.min(height-1,Math.floor((event.clientY-rect.top)/rect.height*height)));
  $('pixel-position').textContent='Pixel '+x+', '+y;
  $('pixel-values').replaceChildren();
  state.manifest.channels.forEach(key=>{const v=state.data[key][y*width+x],entry=element('span');entry.append(document.createTextNode(names[key]+': '),element('b','',valid(v)?fmt(v,2)+' m':'missing'));$('pixel-values').append(entry);});
  document.querySelectorAll('.cross-h').forEach(line=>{line.style.display='block';line.style.top=(y+.5)/height*100+'%';});
  document.querySelectorAll('.cross-v').forEach(line=>{line.style.display='block';line.style.left=(x+.5)/width*100+'%';});
}
function clearInspection(){
  $('pixel-position').textContent='Hover over an image to inspect depth';$('pixel-values').replaceChildren();
  document.querySelectorAll('.cross-h,.cross-v').forEach(line=>line.style.display='none');
}
function wire(){
  ['search','scene-filter','sort'].forEach(id=>$(id).addEventListener(id==='search'?'input':'change',renderList));
  $('clear-search').addEventListener('click',()=>{$('search').value='';$('scene-filter').value='';renderList();});
  $('previous').addEventListener('click',()=>navigate(-1));$('next').addEventListener('click',()=>navigate(1));
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>setMode(button.dataset.mode)));
  $('scale').addEventListener('change',render);$('error-reference').addEventListener('change',render);
  ['swipe-left','swipe-right'].forEach(id=>$(id).addEventListener('change',drawSwipe));
  $('swipe-range').addEventListener('input',()=>{state.split=Number($('swipe-range').value);drawSwipe();});
  let dragging=false;
  const drag=event=>{const r=$('swipe-stage').getBoundingClientRect();state.split=Math.max(0,Math.min(100,(event.clientX-r.left)/r.width*100));$('swipe-range').value=state.split;drawSwipe();};
  $('swipe-stage').addEventListener('pointerdown',event=>{dragging=true;$('swipe-stage').setPointerCapture(event.pointerId);drag(event);});
  $('swipe-stage').addEventListener('pointermove',event=>{if(dragging)drag(event);inspect(event,$('swipe-stage'));});
  $('swipe-stage').addEventListener('pointerup',()=>dragging=false);$('swipe-stage').addEventListener('pointercancel',()=>dragging=false);$('swipe-stage').addEventListener('pointerleave',clearInspection);
  document.addEventListener('keydown',event=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;if(event.key==='ArrowRight'){event.preventDefault();navigate(1);}if(event.key==='ArrowLeft'){event.preventDefault();navigate(-1);}});
}
async function init(){
  const response=await fetch('manifest.json');if(!response.ok)throw Error('Evaluation index is unavailable');
  state.manifest=await response.json();
  if(state.manifest.schema!==1||state.manifest.channels.join(',')!=='before,after,teacher,ground_truth')throw Error('The evaluation export is not ready or has an unsupported format.');
  const m=state.manifest;
  $('total-images').textContent=m.run.test_images;
  $('scene-count').textContent=m.run.test_scenes+' held-out scenes · '+(m.run.validation_images?m.run.validation_images+' separate validation images':'80 / 20 split');
  $('run-steps').textContent=m.run.steps+'-step checkpoint';
  ['before','after','teacher'].forEach(name=>$(name+'-score').textContent=fmt(m.summary[name].abs_rel));
  $('protocol-text').textContent=m.run.protocol;$('baseline-text').textContent='Before-training weights: '+m.run.baseline+'.';
  $('model-info').textContent=m.run.student+' → frozen decoder. Teacher: '+m.run.teacher+'. Training images: '+m.run.train_images+'.';
  [...new Set(m.samples.map(s=>s.scene_type))].sort().forEach(type=>$('scene-filter').add(new Option(pretty(type),type)));
  const colors=new ImageData(256,1);m.colormap.forEach((c,i)=>colors.data.set([...c,255],i*4));
  const bar=$('colorbar'),ctx=bar.getContext('2d'),temp=document.createElement('canvas');temp.width=256;temp.height=1;temp.getContext('2d').putImageData(colors,0,0);ctx.drawImage(temp,0,0,256,12);
  createPanels();wire();renderList();
  const hash=new URLSearchParams(location.hash.slice(1));setMode(hash.get('mode'));
  await select(m.samples.find(s=>s.id===hash.get('image'))||m.samples[0]);
}
init().catch(error=>{status('');$('fatal').hidden=false;$('fatal').textContent='Could not open the viewer: '+error.message;});
