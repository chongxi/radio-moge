import * as THREE from './lib/three.module.min.js';
import {OrbitControls} from './lib/OrbitControls.js';

const D=window.PPD_CLOUDS,$=id=>document.getElementById(id),views=[];
let current=null,clouds=[],syncing=false,generation=0;
const methodReference=i=>[0,2,4].includes(i)?'native':'adapter';
const linear=x=>x<=10.31475?x/3294.6:((x/255+.055)/1.055)**2.4;

for(const s of D.samples.filter(s=>s.height===768)){
  const option=document.createElement('option');option.value=s.index;
  option.textContent=String(s.index+1).padStart(2,'0')+' · '+s.scene.replace('nyuv2:','').replace('structured3d:scene:','空房 ');
  $('sample').append(option);
}
for(const [id,selected] of [['left','2'],['right','3']]){
  D.labels.forEach((label,i)=>{const o=document.createElement('option');o.value=i;o.textContent=label;$(id).append(o)});
  $(id).value=selected;
}

function render(){for(const v of views){const w=v.host.clientWidth,h=v.host.clientHeight;if(!w||!h)continue;v.renderer.setSize(w,h,false);v.camera.aspect=w/h;v.camera.updateProjectionMatrix();v.renderer.render(v.scene,v.camera)}}
function synchronize(origin){if(syncing)return;syncing=true;for(const v of views)if(v!==origin){v.camera.position.copy(origin.camera.position);v.camera.quaternion.copy(origin.camera.quaternion);v.controls.target.copy(origin.controls.target);v.controls.update()}syncing=false;render()}
for(let i=0;i<2;i++){
  const host=$('view'+i),scene=new THREE.Scene();scene.background=new THREE.Color('#101923');
  const camera=new THREE.PerspectiveCamera(48,1,.01,1000);
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.append(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;
  controls.minDistance=.03;controls.maxDistance=300;
  const view={host,scene,camera,renderer,controls};views.push(view);
  controls.addEventListener('change',()=>synchronize(view));new ResizeObserver(render).observe(host);
  renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();fail(new Error('WebGL 上下文丢失，请重新打开页面'))});
}
function clear(){for(const v of views)for(const child of [...v.scene.children]){v.scene.remove(child);child.geometry?.dispose();child.material?.dispose()}clouds=[]}
function point(z,k,x,y,s){return [((x+.5)/s.width-k[0][2])/k[0][0]*z,-((y+.5)/s.height-k[1][2])/k[1][1]*z,-z]}
function reset(mode='oblique'){
  if(!current)return;const s=current.sample,k=s.cameras.native,[lo,hi]=s.range,d=(lo+hi)/2;
  const extent=Math.max(d/k[0][0],d/k[1][1],hi-lo),v=views[0];
  v.controls.target.set(0,0,-d);
  if(mode==='front')v.camera.position.set(0,0,-d+extent*1.25);
  else if(mode==='side')v.camera.position.set(extent*1.3,extent*.12,-d+extent*.08);
  else v.camera.position.set(extent*.42,extent*.22,-d+extent*1.15);
  v.camera.lookAt(v.controls.target);v.controls.update();synchronize(v);
}
function build(){
  if(!current)return;clear();const {sample:s,depths,masks,rgb}=current;
  const mode=$('color').value,step=+$('density').value,over=$('overlay').checked;
  const sw=Math.ceil(s.width/s.stride),sh=Math.ceil(s.height/s.stride),methods=[+$('left').value,+$('right').value];
  for(let side=0;side<2;side++){
    const method=methods[side],ref=methodReference(method),k=s.cameras[ref],mask=masks[ref],depth=depths[method];
    const positions=[],colors=[],rgbColors=[],tint=new THREE.Color(side?'#54e4c3':'#ffb35f'),c=new THREE.Color();
    for(let y=0;y<sh;y+=step)for(let x=0;x<sw;x+=step){
      const j=y*sw+x,z=depth[j];if(!mask[j]||!Number.isFinite(z)||z<=0)continue;
      const p=point(z,k,x*s.stride,y*s.stride,s);positions.push(...p);
      rgbColors.push(rgb[3*j],rgb[3*j+1],rgb[3*j+2]);
      if(mode==='rgb')colors.push(linear(rgb[j*3]),linear(rgb[j*3+1]),linear(rgb[j*3+2]));
      else if(mode==='model')colors.push(tint.r,tint.g,tint.b);
      else if(mode==='depth'){const t=Math.min(1,Math.max(0,(z-s.range[0])/(s.range[1]-s.range[0])));c.setHSL(.66*(1-t),.78,.56);colors.push(c.r,c.g,c.b)}
      else{
        let light=.75;
        if(x+1<sw&&y+1<sh&&mask[j+1]&&mask[j+sw]){
          const a=point(depth[j+1],k,(x+1)*s.stride,y*s.stride,s),b=point(depth[j+sw],k,x*s.stride,(y+1)*s.stride,s);
          const ax=a[0]-p[0],ay=a[1]-p[1],az=a[2]-p[2],bx=b[0]-p[0],by=b[1]-p[1],bz=b[2]-p[2];
          const nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx,n=Math.hypot(nx,ny,nz);
          if(n>1e-12)light=.16+.84*Math.abs((nx*.3+ny*.4+nz*.8660254)/n);
        }
        colors.push(light*.88,light*.94,light);
      }
    }
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
    const material=new THREE.PointsMaterial({vertexColors:true,size:+$('size').value,sizeAttenuation:false,transparent:over,opacity:over?.70:1,depthWrite:!over});
    views[over?0:side].scene.add(new THREE.Points(geometry,material));
    clouds.push({method,positions,rgb:rgbColors,count:positions.length/3,reference:ref});
    $('count'+side).textContent=(positions.length/3).toLocaleString()+' 点 · 原分辨率 '+s.full_counts[D.names[method]].toLocaleString()+' 点';
    $('title'+side).textContent=D.labels[method];
  }
  $('card1').hidden=over;$('panels').classList.toggle('overlay',over);
  if(over)$('title0').textContent='叠加 · 橙：'+D.labels[methods[0]]+' / 绿：'+D.labels[methods[1]];
  $('projection').textContent='左侧尺度/相机：'+(methodReference(methods[0])==='native'?'原生 MoGe':'我们的 MoGe 路线')+'　｜　右侧尺度/相机：'+(methodReference(methods[1])==='native'?'原生 MoGe':'我们的 MoGe 路线')+'。坐标单位为预测的米制尺度。';
  render();
}
async function load(){
  const token=++generation;clear();current=null;document.body.dataset.ready='false';$('status').textContent='正在解压点云…';
  const sample=D.samples.find(s=>s.index===+$('sample').value&&s.height===+$('resolution').value);
  if(!sample)throw Error('场景不存在');$('rgb').src=sample.rgb;$('scene').textContent=sample.scene;
  $('scene-id').textContent=(sample.index+1)+' / 16 · '+sample.width+' × '+sample.height+' · 空间特征 cos '+sample.features_cos.toFixed(4);
  const bytes=Uint8Array.from(atob(sample.payload),c=>c.charCodeAt(0));
  const buffer=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const n=Math.ceil(sample.width/sample.stride)*Math.ceil(sample.height/sample.stride);
  if(buffer.byteLength!==n*25)throw Error('点云载荷长度错误');
  const depths=Array.from({length:5},(_,i)=>new Float32Array(buffer,i*n*4,n));
  const masks={native:new Uint8Array(buffer,n*20,n),adapter:new Uint8Array(buffer,n*21,n)};
  const rgb=new Uint8Array(buffer,n*22,n*3);if(token!==generation)return;
  current={sample,depths,masks,rgb};build();reset();document.body.dataset.ready='true';
  $('status').textContent='拖动旋转 · 滚轮缩放 · 右键拖动平移 · 两侧视角同步';
}
function download(side){
  const c=clouds[side];if(!c)return;
  const header=`ply\nformat binary_little_endian 1.0\ncomment predicted meters; x right y up z backward; display stride ${current.sample.stride*(+$('density').value)}\nelement vertex ${c.count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const buffer=new ArrayBuffer(c.count*15),v=new DataView(buffer);
  for(let j=0;j<c.count;j++)for(let k=0;k<3;k++){v.setFloat32(j*15+k*4,c.positions[j*3+k],true);v.setUint8(j*15+12+k,c.rgb[j*3+k])}
  const link=document.createElement('a'),url=URL.createObjectURL(new Blob([header,buffer],{type:'application/octet-stream'}));
  link.href=url;link.download=current.sample.id+'-'+D.names[c.method]+'-preview.ply';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function fail(error){$('status').textContent='载入失败：'+error.message;document.body.dataset.ready='false';console.error(error)}
for(const id of ['sample','resolution'])$(id).onchange=()=>load().catch(fail);
for(const id of ['left','right','color','density'])$(id).onchange=build;
for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('sample').value=(+$('sample').value+delta+16)%16;load().catch(fail)};
$('size').oninput=build;$('overlay').onchange=()=>{if($('overlay').checked)$('color').value='model';build()};
$('reset').onclick=()=>reset();$('front').onclick=()=>reset('front');$('side').onclick=()=>reset('side');
for(let i=0;i<2;i++)$('download'+i).onclick=()=>download(i);
window.__ppdCloudTest=()=>({ready:document.body.dataset.ready,sample:current?.sample.id,methods:clouds.map(c=>c.method),counts:clouds.map(c=>c.count),positions:clouds.map(c=>c.positions.slice(0,9)),cameras:views.map(v=>v.camera.position.toArray()),targets:views.map(v=>v.controls.target.toArray())});
load().catch(fail);
