import * as THREE from './lib/three.module.min.js';
import {OrbitControls} from './lib/OrbitControls.js';

const D=window.PPD_EDGES,$=id=>document.getElementById(id),views=[];
let current=null,clouds=[],removedCloud=null,syncing=false,generation=0;
const linear=x=>x<=10.31475?x/3294.6:((x/255+.055)/1.055)**2.4;
const labels=['moge-original','moge-edge','ppd'];
for(const s of D.samples.filter(s=>s.height===480)){
  const option=document.createElement('option');option.value=s.index;
  option.textContent=String(s.index+1).padStart(2,'0')+' · '+s.scene.replace('nyuv2:','').replace('structured3d:scene:','空房 ');
  $('sample').append(option);
}
function render(){for(const v of views){const w=v.host.clientWidth,h=v.host.clientHeight;if(!w||!h)continue;v.renderer.setSize(w,h,false);v.camera.aspect=w/h;v.camera.updateProjectionMatrix();v.renderer.render(v.scene,v.camera)}}
function synchronize(origin){if(syncing)return;syncing=true;for(const v of views)if(v!==origin){v.camera.position.copy(origin.camera.position);v.camera.quaternion.copy(origin.camera.quaternion);v.controls.target.copy(origin.controls.target);v.controls.update()}syncing=false;render()}
for(let i=0;i<3;i++){
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
function clear(){for(const v of views)for(const child of [...v.scene.children]){v.scene.remove(child);child.geometry?.dispose();child.material?.dispose()}clouds=[];removedCloud=null}
function point(z,k,x,y,s){return [((x+.5)/s.width-k[0][2])/k[0][0]*z,-((y+.5)/s.height-k[1][2])/k[1][1]*z,-z]}
function reset(mode='oblique'){
  if(!current||!clouds.length)return;const v=views[0],bounds=[];
  // Use both unfiltered depths for one shared framing, independent of threshold.
  const {sample:s,depths,mask}=current,sw=Math.ceil(s.width/s.stride),all=[];
  for(const depth of depths)for(let j=0;j<depth.length;j+=4)if(mask[j])all.push(...point(depth[j],s.camera,(j%sw)*s.stride,Math.floor(j/sw)*s.stride,s));
  for(let axis=0;axis<3;axis++){const values=[];for(let i=axis;i<all.length;i+=3)values.push(all[i]);values.sort((a,b)=>a-b);bounds.push([values[Math.floor(values.length*.005)],values[Math.floor(values.length*.995)]])}
  const center=new THREE.Vector3(...bounds.map(([a,b])=>(a+b)/2));
  const direction=new THREE.Vector3(...(mode==='front'?[0,0,1]:mode==='side'?[1,.12,.08]:[.82,.18,.85])).normalize();
  const right=new THREE.Vector3().crossVectors(new THREE.Vector3(0,1,0),direction).normalize(),up=new THREE.Vector3().crossVectors(direction,right);
  const tanV=Math.tan(THREE.MathUtils.degToRad(v.camera.fov/2)),aspect=Math.min(...views.map(w=>w.host.clientWidth/w.host.clientHeight));
  const distances=[],p=new THREE.Vector3();
  for(let i=0;i<all.length;i+=3){p.set(all[i],all[i+1],all[i+2]).sub(center);distances.push(p.dot(direction)+Math.max(Math.abs(p.dot(right))/(tanV*aspect),Math.abs(p.dot(up))/tanV))}
  distances.sort((a,b)=>a-b);const distance=Math.max(.1,distances[Math.floor(distances.length*.995)]);
  v.controls.target.copy(center);v.camera.position.copy(center).addScaledVector(direction,distance*1.14);
  v.camera.lookAt(v.controls.target);v.controls.update();synchronize(v);
}
function addCloud(side,positions,colors,size){
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  const material=new THREE.PointsMaterial({vertexColors:true,size,sizeAttenuation:false});
  views[side].scene.add(new THREE.Points(geometry,material));
}
function drawRemovedMap(){
  const {sample:s,mask,bits,rgb}=current,sw=Math.ceil(s.width/s.stride),sh=Math.ceil(s.height/s.stride),flag=1<<+$('threshold').value;
  const canvas=$('removed-map');canvas.width=sw;canvas.height=sh;
  const context=canvas.getContext('2d'),pixels=context.createImageData(sw,sh);
  for(let j=0;j<mask.length;j++){
    const removed=mask[j]&&(bits[0][j]&flag);
    for(let c=0;c<3;c++)pixels.data[j*4+c]=removed?([255,50,75][c]*.8+rgb[j*3+c]*.2):rgb[j*3+c]*.72;
    pixels.data[j*4+3]=255;
  }
  context.putImageData(pixels,0,0);
}
function build(){
  if(!current)return;clear();const {sample:s,depths,mask,bits,rgb}=current;
  const index=+$('threshold').value,flag=1<<index,mode=$('color').value,step=+$('density').value;
  const sw=Math.ceil(s.width/s.stride),sh=Math.ceil(s.height/s.stride),ppdFiltered=$('ppd-filter').checked;
  for(let side=0;side<3;side++){
    const which=side===2?1:0,filtered=side===1||(side===2&&ppdFiltered),depth=depths[which];
    const positions=[],colors=[],rgbColors=[],ids=[],deleted=[],deletedRgb=[],deletedColors=[];
    const tint=new THREE.Color(['#b7c7d9','#71dfbf','#83baff'][side]),c=new THREE.Color();
    for(let y=0;y<sh;y+=step)for(let x=0;x<sw;x+=step){
      const j=y*sw+x,z=depth[j];if(!mask[j]||!Number.isFinite(z)||z<=0)continue;
      const p=point(z,s.camera,x*s.stride,y*s.stride,s);
      if(filtered&&(bits[which][j]&flag)){
        deleted.push(...p);deletedRgb.push(rgb[j*3],rgb[j*3+1],rgb[j*3+2]);deletedColors.push(1,.07,.16);continue;
      }
      positions.push(...p);ids.push(j);rgbColors.push(rgb[j*3],rgb[j*3+1],rgb[j*3+2]);
      if(mode==='rgb')colors.push(linear(rgb[j*3]),linear(rgb[j*3+1]),linear(rgb[j*3+2]));
      else if(mode==='model')colors.push(tint.r,tint.g,tint.b);
      else {const t=Math.min(1,Math.max(0,(z-s.range[0])/(s.range[1]-s.range[0])));c.setHSL(.66*(1-t),.78,.56);colors.push(c.r,c.g,c.b)}
    }
    addCloud(side,positions,colors,+$('size').value);
    if($('removed').checked&&deleted.length)addCloud(side,deleted,deletedColors,+$('size').value+.5);
    const item={positions,rgb:rgbColors,ids,count:positions.length/3,removed:deleted.length/3};clouds.push(item);
    if(side===1)removedCloud={positions:deleted,rgb:deletedRgb,count:deleted.length/3};
    const stats=s.counts[which?'ppd_adapter':'moge_adapter'][index];
    $('total'+side).textContent=filtered?'删除 '+(stats.removed_fraction*100).toFixed(2)+'%':'保留原始有效点';
    $('count'+side).textContent=(filtered?stats.kept:s.base_count).toLocaleString()+' / '+s.base_count.toLocaleString()+' 点（完整分辨率） · 预览 '+item.count.toLocaleString();
  }
  $('title2').textContent=ppdFiltered?'PPD · 同样过滤':'PPD · 未过滤';
  const t=D.thresholds[index]*100,m=s.counts.moge_adapter[index],p=s.counts.ppd_adapter[index];
  $('stats').textContent=`当前阈值 ${t}%：MoGe 删除 ${m.removed.toLocaleString()} 点（${(m.removed_fraction*100).toFixed(2)}%）。PPD 自己的深度按同一规则会删除 ${p.removed.toLocaleString()} 点（${(p.removed_fraction*100).toFixed(2)}%）${ppdFiltered?'，右侧已应用。':'，右侧暂未应用。'}`;
  drawRemovedMap();render();
}
async function load(){
  const token=++generation;clear();current=null;document.body.dataset.ready='false';$('status').textContent='正在解压点云…';
  const sample=D.samples.find(s=>s.index===+$('sample').value&&s.height===+$('resolution').value);
  if(!sample)throw Error('场景不存在');$('rgb').src=sample.rgb;$('scene').textContent=sample.scene;
  $('scene-id').textContent=(sample.index+1)+' / 16 · '+sample.width+' × '+sample.height+' · 过滤后再抽样预览';
  const bytes=Uint8Array.from(atob(sample.payload),c=>c.charCodeAt(0));
  const buffer=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const n=Math.ceil(sample.width/sample.stride)*Math.ceil(sample.height/sample.stride);
  if(buffer.byteLength!==n*14)throw Error('点云载荷长度错误');
  const depths=[new Float32Array(buffer,0,n),new Float32Array(buffer,n*4,n)],mask=new Uint8Array(buffer,n*8,n);
  const bits=[new Uint8Array(buffer,n*9,n),new Uint8Array(buffer,n*10,n)],rgb=new Uint8Array(buffer,n*11,n*3);
  if(token!==generation)return;current={sample,depths,mask,bits,rgb};build();reset();document.body.dataset.ready='true';
  $('status').textContent='拖动旋转 · 滚轮缩放 · 右键平移 · 三个视角同步';
}
function download(side){
  const c=side==='removed'?removedCloud:clouds[side];if(!c)return;
  const suffix=side==='removed'?'moge-removed':labels[side];
  const header=`ply\nformat binary_little_endian 1.0\ncomment predicted meters; x right y up z backward; preview stride ${current.sample.stride*(+$('density').value)}\ncomment ${suffix}; edge rtol=${D.thresholds[+$('threshold').value]}; ppd_filtered=${$('ppd-filter').checked}\nelement vertex ${c.count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const buffer=new ArrayBuffer(c.count*15),v=new DataView(buffer);
  for(let j=0;j<c.count;j++)for(let k=0;k<3;k++){v.setFloat32(j*15+k*4,c.positions[j*3+k],true);v.setUint8(j*15+12+k,c.rgb[j*3+k])}
  const link=document.createElement('a'),url=URL.createObjectURL(new Blob([header,buffer],{type:'application/octet-stream'}));
  link.href=url;link.download=current.sample.id+'-'+suffix+'-t'+D.thresholds[+$('threshold').value]+'-preview.ply';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function fail(error){$('status').textContent='载入失败：'+error.message;$('status').classList.add('error');document.body.dataset.ready='false';console.error(error)}
for(const id of ['sample','resolution'])$(id).onchange=()=>load().catch(fail);
for(const id of ['threshold','color','density','removed','ppd-filter'])$(id).onchange=build;
for(const [id,delta] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('sample').value=(+$('sample').value+delta+16)%16;load().catch(fail)};
$('size').oninput=build;for(const mode of ['oblique','front','side'])$(mode).onclick=()=>reset(mode);
for(let i=0;i<3;i++)$('download'+i).onclick=()=>download(i);$('download-removed').onclick=()=>download('removed');
function subsetExact(){
  if(clouds.length!==3)return false;const offsets=new Int32Array(current.mask.length).fill(-1);
  clouds[0].ids.forEach((id,i)=>{offsets[id]=i});
  return clouds[1].ids.every((id,j)=>{const i=offsets[id];return i>=0&&[0,1,2].every(k=>clouds[1].positions[j*3+k]===clouds[0].positions[i*3+k])});
}
window.__ppdEdgesTest=()=>({ready:document.body.dataset.ready,sample:current?.sample.id,threshold:+$('threshold').value,
  counts:clouds.map(c=>c.count),removed:clouds.map(c=>c.removed),positions:clouds.map(c=>c.positions.slice(0,9)),
  cameras:views.map(v=>v.camera.position.toArray()),targets:views.map(v=>v.controls.target.toArray()),
  ppdFiltered:$('ppd-filter').checked,overlay:$('removed').checked,
  previewSubsetExact:subsetExact()});
load().catch(fail);
