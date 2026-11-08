import * as THREE from './lib/three.module.min.js';
import {OrbitControls} from './lib/OrbitControls.js';

const D=window.PPD_CACHE,$=id=>document.getElementById(id),views=[],colors=['#f5ad69','#63dfbd','#83acff'];
let current=null,clouds=[],syncing=false,generation=0;
const linear=x=>x<=10.31475?x/3294.6:((x/255+.055)/1.055)**2.4;
const labels=['每帧标定','每 10 帧标定','每 30 帧标定','只标定首帧'];
for(const [mode,label] of [['cpu_every','逐帧标定 → CPU 点云'],['gpu_every','逐帧标定 → GPU 点云'],['gpu_10','每 10 帧标定 → GPU'],['gpu_first','只标定首帧 → GPU']]){
 const box=document.createElement('div');box.className='stat';box.textContent=label;const b=document.createElement('b');b.textContent=D.timings[mode].all_ms.mean.toFixed(1)+' ms';box.append(b);$('stats').append(box);
}
for(const s of D.samples){const o=document.createElement('option');o.value=s.frame;o.textContent=`${s.frame} · ${(s.frame/30).toFixed(2)} s`;$('frame').append(o)}
for(let i=1;i<4;i++){const o=document.createElement('option');o.value=i;o.textContent=labels[i];$('method').append(o)}
$('frame').value='179';
function render(){for(const v of views){const w=v.host.clientWidth,h=v.host.clientHeight;v.renderer.setSize(w,h,false);v.camera.aspect=w/h;v.camera.updateProjectionMatrix();v.renderer.render(v.scene,v.camera)}}
function sync(origin){if(syncing)return;syncing=true;for(const v of views)if(v!==origin){v.camera.position.copy(origin.camera.position);v.camera.quaternion.copy(origin.camera.quaternion);v.controls.target.copy(origin.controls.target);v.controls.update()}syncing=false;render()}
for(let i=0;i<2;i++){
 const host=$('view'+i),scene=new THREE.Scene();scene.background=new THREE.Color('#101923');
 const camera=new THREE.PerspectiveCamera(48,1,.01,2000),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.append(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=.02;controls.maxDistance=500;
 const v={host,scene,camera,renderer,controls};views.push(v);controls.addEventListener('change',()=>sync(v));new ResizeObserver(render).observe(host);
}
function clear(){for(const v of views)for(const child of [...v.scene.children]){v.scene.remove(child);child.geometry?.dispose();child.material?.dispose()}clouds=[]}
function build(){
 if(!current)return;clear();const {sample:s,depths,mask,rgb}=current,k=D.camera,methods=[0,+$('method').value],shared=$('common-mask').checked;
 for(let side=0;side<2;side++){
  const method=methods[side],depth=depths[method],positions=[],rgbColors=[],cs=[],color=new THREE.Color(),tint=new THREE.Color(side?colors[method-1]:'#f6e0c9');
  for(let y=0;y<120;y++)for(let x=0;x<160;x++){
   const j=y*160+x,z=depth[j];if((!method||shared)&&!mask[j]||!Number.isFinite(z)||z<=0)continue;
   positions.push(((x*4+.5)/640-k[0][2])/k[0][0]*z,-((y*4+.5)/480-k[1][2])/k[1][1]*z,-z);
   rgbColors.push(rgb[j*3],rgb[j*3+1],rgb[j*3+2]);
   if($('color').value==='rgb')cs.push(linear(rgb[j*3]),linear(rgb[j*3+1]),linear(rgb[j*3+2]));
   else if($('color').value==='model')cs.push(tint.r,tint.g,tint.b);
   else{const t=Math.max(0,Math.min(1,(z-s.range[0])/(s.range[1]-s.range[0])));color.setHSL(.66*(1-t),.78,.56);cs.push(color.r,color.g,color.b)}
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(cs,3));
  views[side].scene.add(new THREE.Points(geometry,new THREE.PointsMaterial({vertexColors:true,size:2,sizeAttenuation:false})));
  clouds.push({method,positions,rgb:rgbColors,count:positions.length/3});
  $('detail'+side).textContent=`${(positions.length/3).toLocaleString()} 个预览点 · 固定首帧相机 · ${method&&!shared?'当前有限深度':'当前参照有效区域'}`;
 }
 const metric=s.metrics[D.names[methods[1]]];$('right-title').textContent=labels[methods[1]];
 $('detail1').textContent+=` · 平均相对偏差 ${(metric.abs_rel*100).toFixed(2)}% · 深度中位比 ${metric.median_depth_ratio.toFixed(3)}`;
 $('error').textContent=`当前帧：${metric.calibration_updated?'本帧更新标定':'本帧复用标定'}；相对逐帧参照的像素误差 P95 为 ${(metric.p95_pixel_rel*100).toFixed(2)}%。`;
 render();drawChart();
}
function reset(mode='oblique'){
 if(!clouds.length)return;const bounds=[];for(let axis=0;axis<3;axis++){const a=[];for(const c of clouds)for(let i=axis;i<c.positions.length;i+=3)a.push(c.positions[i]);a.sort((x,y)=>x-y);bounds.push([a[Math.floor(a.length*.005)],a[Math.floor(a.length*.995)]])}
 const v=views[0],center=new THREE.Vector3(...bounds.map(([a,b])=>(a+b)/2)),direction=new THREE.Vector3(...(mode==='front'?[0,0,1]:mode==='side'?[1,.12,.08]:[.42,.22,1.15])).normalize();
 const right=new THREE.Vector3().crossVectors(new THREE.Vector3(0,1,0),direction).normalize(),up=new THREE.Vector3().crossVectors(direction,right),tanV=Math.tan(THREE.MathUtils.degToRad(v.camera.fov/2)),aspect=Math.min(...views.map(v=>v.host.clientWidth/v.host.clientHeight)),p=new THREE.Vector3(),distances=[];
 for(const c of clouds)for(let i=0;i<c.positions.length;i+=3){p.set(c.positions[i],c.positions[i+1],c.positions[i+2]).sub(center);distances.push(p.dot(direction)+Math.max(Math.abs(p.dot(right))/(tanV*aspect),Math.abs(p.dot(up))/tanV))}
 distances.sort((a,b)=>a-b);const distance=Math.max(.1,distances[Math.floor(distances.length*.995)]);v.controls.target.copy(center);v.camera.position.copy(center).addScaledVector(direction,distance*1.14);v.camera.lookAt(center);v.controls.update();sync(v);
}
function drawChart(){
 const canvas=$('chart'),ratio=Math.min(devicePixelRatio,2),w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext('2d');c.scale(ratio,ratio);
 const max=Math.max(...D.curves.flat())*1.08,left=45,top=10,bottom=20;const yy=x=>h-bottom-x/max*(h-top-bottom),xx=i=>left+i/359*(w-left-8);
 c.font='11px system-ui';c.fillStyle='#93a9bd';c.fillText((max*100).toFixed(0)+'%',4,14);c.fillText('0%',8,h-bottom);c.fillText('0 s',left,h-4);c.fillText('12 s',w-34,h-4);
 for(let j=0;j<3;j++){c.strokeStyle=colors[j];c.lineWidth=1.4;c.beginPath();D.curves[j].forEach((v,i)=>{i?c.lineTo(xx(i),yy(v)):c.moveTo(xx(i),yy(v))});c.stroke()}
 if(current){c.strokeStyle='#edf4fa';c.setLineDash([3,3]);c.beginPath();c.moveTo(xx(current.sample.frame),top);c.lineTo(xx(current.sample.frame),h-bottom);c.stroke()}
}
async function load(){
 const token=++generation;document.body.dataset.ready='false';$('status').textContent='解压点云…';const sample=D.samples.find(s=>s.frame===+$('frame').value);
 const bytes=Uint8Array.from(atob(sample.payload),x=>x.charCodeAt(0)),buffer=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer(),n=160*120;
 if(buffer.byteLength!==n*20)throw Error('点云数据长度错误');if(token!==generation)return;
 current={sample,depths:Array.from({length:4},(_,i)=>new Float32Array(buffer,i*n*4,n)),mask:new Uint8Array(buffer,n*16,n),rgb:new Uint8Array(buffer,n*17,n*3)};
 $('rgb').src=sample.rgb;$('frame-label').textContent=`第 ${sample.frame} 帧 · ${(sample.frame/30).toFixed(2)} 秒 / 12 秒`;
 build();reset();document.body.dataset.ready='true';$('status').textContent='拖动旋转 · 滚轮缩放 · 右键平移 · 两侧相机同步';
}
function download(side){const c=clouds[side],header=`ply\nformat binary_little_endian 1.0\ncomment predicted scale; x right y up z backward; stride 4\nelement vertex ${c.count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`,buffer=new ArrayBuffer(c.count*15),v=new DataView(buffer);for(let j=0;j<c.count;j++)for(let k=0;k<3;k++){v.setFloat32(j*15+k*4,c.positions[j*3+k],true);v.setUint8(j*15+12+k,c.rgb[j*3+k])}const a=document.createElement('a'),url=URL.createObjectURL(new Blob([header,buffer]));a.href=url;a.download=`frame-${current.sample.frame}-${D.names[c.method]}-preview.ply`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function fail(e){$('status').textContent='加载失败：'+e.message;console.error(e)}
$('frame').onchange=()=>load().catch(fail);for(const id of ['method','color','common-mask'])$(id).onchange=()=>{build();reset()};
for(const mode of ['front','side'])$(mode).onclick=()=>reset(mode);$('reset').onclick=()=>reset();for(let i=0;i<2;i++)$('download'+i).onclick=()=>download(i);
$('chart').onclick=e=>{const frame=Math.max(0,Math.min(359,(e.offsetX-45)/($('chart').clientWidth-53)*359)),nearest=D.samples.reduce((a,b)=>Math.abs(b.frame-frame)<Math.abs(a.frame-frame)?b:a);$('frame').value=nearest.frame;load().catch(fail)};
new ResizeObserver(drawChart).observe($('chart'));
window.__ppdCacheTest=()=>({ready:document.body.dataset.ready,frame:current?.sample.frame,methods:clouds.map(c=>c.method),counts:clouds.map(c=>c.count),positions:clouds.map(c=>c.positions.slice(0,9)),cameras:views.map(v=>v.camera.position.toArray())});
load().catch(fail);
