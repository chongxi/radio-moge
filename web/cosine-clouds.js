import * as THREE from './lib/three.module.min.js';
import {OrbitControls} from './lib/OrbitControls.js';
const D=window.CLOUD_DATA,$=id=>document.getElementById(id),names=D.names,labels=D.labels,tints=[0xffae72,0x44d5b1];
let current=null,views=[],clouds=[],busy=false,token=0;
const linear=v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;};
$('sample').innerHTML=D.samples.map((s,i)=>`<option value="${i}">${i+1}. ${s.scene} · ${s.id}</option>`).join('');
$('summary').innerHTML=names.map((n,i)=>`<div><span>${labels[i]}</span><strong>${D.summary[n].sensor.abs_rel.toFixed(4)}</strong><small>20 个固定场景 · 传感器 AbsRel ↓</small></div>`).join('');
function render(){views.forEach(v=>{if(v.host.clientWidth){v.renderer.setSize(v.host.clientWidth,v.host.clientHeight,false);v.camera.aspect=v.host.clientWidth/v.host.clientHeight;v.camera.updateProjectionMatrix();v.renderer.render(v.scene,v.camera);}});}
function sync(v){if(busy)return;busy=true;views.forEach(x=>{if(x!==v){x.camera.position.copy(v.camera.position);x.camera.quaternion.copy(v.camera.quaternion);x.controls.target.copy(v.controls.target);x.controls.update();}});busy=false;render();}
for(let i=0;i<2;i++){
 const host=$('view'+i),scene=new THREE.Scene();scene.background=new THREE.Color('#101c25');
 const camera=new THREE.PerspectiveCamera(48,1,.01,1000),renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));host.appendChild(renderer.domElement);
 const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=false;controls.minDistance=.05;controls.maxDistance=100;
 const v={host,scene,camera,renderer,controls};views.push(v);controls.addEventListener('change',()=>sync(v));new ResizeObserver(render).observe(host);
 renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();$('status').textContent='图形上下文丢失，请重新打开展示。';});
}
function reset(front=false){if(!current)return;const [lo,hi]=current.sample.range,k=current.sample.cameras[1],d=(lo+hi)/2,extent=Math.max(d/k[0][0],d/k[1][1],hi-lo),v=views[0];v.controls.target.set(0,0,-d);v.camera.position.set(front?0:extent*.37,front?0:extent*.20,front?extent*.15:-d+extent*.95);v.camera.lookAt(v.controls.target);v.controls.update();sync(v);}
function clear(){for(const v of views){for(const o of [...v.scene.children]){v.scene.remove(o);o.geometry?.dispose();o.material?.dispose();}}clouds=[];}
function build(){if(!current)return;clear();const {sample:s,planes,masks,rgb}=current,mode=$('color').value,step=Number($('density').value),overlay=$('overlay').checked;
 for(let b=0;b<2;b++){
  const k=s.cameras[$('camera').value==='shared'?1:b],pos=[],cols=[],colors=[],ids=[],base=s.stride,sw=Math.ceil(s.width/base),sh=Math.ceil(s.height/base),tint=new THREE.Color(tints[b]);
  for(let y=0;y<sh;y+=step)for(let x=0;x<sw;x+=step){const j=y*sw+x,z=planes[b][j];if(!masks[b][j]||!Number.isFinite(z)||z<=0)continue;
   const px=x*base,py=y*base;pos.push(((px+.5)/s.width-k[0][2])/k[0][0]*z,-((py+.5)/s.height-k[1][2])/k[1][1]*z,-z);ids.push(j);const p=(py*s.width+px)*4;colors.push(rgb[p],rgb[p+1],rgb[p+2]);
   if(mode==='model')cols.push(tint.r,tint.g,tint.b);else if(mode==='depth'){const t=Math.max(0,Math.min(1,(z-s.range[0])/(s.range[1]-s.range[0]))),c=new THREE.Color().setHSL(.65*(1-t),.78,.53);cols.push(c.r,c.g,c.b);}else cols.push(linear(rgb[p]),linear(rgb[p+1]),linear(rgb[p+2]));
  }
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geometry.setAttribute('color',new THREE.Float32BufferAttribute(cols,3));const material=new THREE.PointsMaterial({size:Number($('size').value),sizeAttenuation:false,vertexColors:true,transparent:overlay,opacity:overlay?.72:1,depthWrite:!overlay});const object=new THREE.Points(geometry,material);views[overlay?0:b].scene.add(object);clouds.push({positions:pos,rgb:colors,count:ids.length});$('count'+b).textContent=ids.length.toLocaleString()+' 点 · FOV '+(2*Math.atan(.5/k[0][0])*180/Math.PI).toFixed(1)+'°';
 }
 $('card1').hidden=overlay;$('title0').textContent=overlay?'叠加 · 橙：Cosine 最佳 / 绿：MoGe-2':labels[0];$('panels').classList.toggle('overlay',overlay);$('projection').textContent=$('camera').value==='shared'?'共用教师内参的诊断投影；保留各自预测深度。':'各自恢复的相机内参；共同原点、米制坐标，无尺度对齐或点云配准。';render();
}
async function load(index){const id=++token;current=null;clear();$('status').textContent='正在载入点云…';document.body.dataset.ready='false';const s=D.samples[index];$('rgb').src=s.rgb;$('scene').textContent=s.scene;$('scene-id').textContent=(index+1)+' / '+D.samples.length+' · '+s.id+' · '+s.width+' × '+s.height;
 const bytes=Uint8Array.from(atob(s.payload),c=>c.charCodeAt(0));const buf=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();const n=Math.ceil(s.width/s.stride)*Math.ceil(s.height/s.stride),planes=Array.from({length:2},(_,i)=>new Float32Array(buf,i*n*4,n)),masks=Array.from({length:2},(_,i)=>new Uint8Array(buf,n*8+i*n,n));
 const img=new Image();img.src=s.rgb;await img.decode();const canvas=document.createElement('canvas');canvas.width=s.width;canvas.height=s.height;const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0);const rgb=ctx.getImageData(0,0,s.width,s.height).data;if(id!==token)return;current={sample:s,planes,masks,rgb};
 $('scores').innerHTML=names.map((name,i)=>`<tr><td>${labels[i]}</td><td>${s.metrics[name].sensor.abs_rel.toFixed(4)}</td><td>${s.metrics[name].sensor.rmse.toFixed(3)} m</td><td>${s.metrics[name].teacher.abs_rel.toFixed(4)}</td></tr>`).join('');build();reset();$('status').textContent='拖动任一视图旋转 · 滚轮缩放 · 右键拖动平移 · 两个视图同步';document.body.dataset.ready='true';
}
function download(b){const c=clouds[b];if(!c)return;const header=`ply\nformat binary_little_endian 1.0\ncomment meters; x right y up z backward\nelement vertex ${c.count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`,buffer=new ArrayBuffer(c.count*15),v=new DataView(buffer);for(let j=0;j<c.count;j++)for(let k=0;k<3;k++){v.setFloat32(j*15+k*4,c.positions[j*3+k],true);v.setUint8(j*15+12+k,c.rgb[j*3+k]);}const a=document.createElement('a'),url=URL.createObjectURL(new Blob([header,buffer]));a.href=url;a.download=current.sample.id+'-'+names[b]+'.ply';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('sample').onchange=()=>load(Number($('sample').value)).catch(fail);$('prev').onclick=()=>{let n=(Number($('sample').value)+D.samples.length-1)%D.samples.length;$('sample').value=n;load(n).catch(fail);};$('next').onclick=()=>{let n=(Number($('sample').value)+1)%D.samples.length;$('sample').value=n;load(n).catch(fail);};
for(const key of ['color','density','camera'])$(key).onchange=build;$('size').oninput=build;$('overlay').onchange=()=>{if($('overlay').checked)$('color').value='model';build();};$('reset').onclick=()=>reset();$('front').onclick=()=>reset(true);for(let i=0;i<2;i++)$('download'+i).onclick=()=>download(i);
function fail(e){$('status').textContent='载入失败：'+e.message;console.error(e);}
window.__cloudTest=()=>({ready:document.body.dataset.ready,sample:current?.sample.id,counts:clouds.map(c=>c.count),cameras:views.map(v=>v.camera.position.toArray()),targets:views.map(v=>v.controls.target.toArray()),positions:clouds.map(c=>c.positions.slice(0,9)),summary:D.summary});load(0).catch(fail);
