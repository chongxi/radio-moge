import * as THREE from './lib/three.module.min.js';
import {OrbitControls} from './lib/OrbitControls.js';
import {buildCloud} from './cloud-math.mjs';

const $=id=>document.getElementById(id);
const srgb=Array.from({length:256},(_,v)=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;});
const colors={after:0xe98b50,teacher:0x39cbb4};

export function createCloudViewer(metadata,manifest){
  if(metadata.checkpoint_sha256!==manifest.run.checkpoint_sha256 ||
     metadata.samples.length!==manifest.samples.length ||
     metadata.samples.some((s,i)=>s.id!==manifest.samples[i].id))
    throw Error('3D data does not match this evaluation checkpoint.');
  const views=[], clouds={};
  let syncing=false, current=null, generation=0, active=false;
  function draw(){if(active)views.forEach(v=>v.renderer.render(v.scene,v.camera));}
  function resize(){if(!active)return;views.forEach(v=>{
    if(v.host.clientWidth===0)return;
    v.renderer.setSize(v.host.clientWidth,v.host.clientHeight,false);
    v.camera.aspect=v.host.clientWidth/v.host.clientHeight;v.camera.updateProjectionMatrix();
  });draw();}
  function sync(from){
    if(syncing)return;syncing=true;
    views.forEach(v=>{if(v===from)return;v.camera.position.copy(from.camera.position);
      v.camera.quaternion.copy(from.camera.quaternion);v.controls.target.copy(from.controls.target);
      v.controls.update();});
    syncing=false;draw();
  }
  for(const branch of ['after','teacher']){
    const host=$('cloud-'+branch),scene=new THREE.Scene();scene.background=new THREE.Color('#111e25');
    const camera=new THREE.PerspectiveCamera(50,1,.01,1000);
    const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));
    renderer.domElement.setAttribute('aria-label',branch==='after'?'Trained model 3D point cloud':'MoGe-2 3D point cloud');
    renderer.domElement.tabIndex=0;host.append(renderer.domElement);
    const controls=new OrbitControls(camera,renderer.domElement);
    controls.enableDamping=false;controls.minDistance=.05;controls.maxDistance=100;
    const view={branch,host,scene,camera,renderer,controls};views.push(view);
    controls.addEventListener('change',()=>sync(view));
    renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();
      $('cloud-status').textContent='3D graphics context lost. Reload the page to restore it.';});
    new ResizeObserver(resize).observe(host);
  }
  function clear(){
    views.forEach(v=>{for(const item of [...v.scene.children]){v.scene.remove(item);
      item.geometry?.dispose();item.material?.dispose();}});
    for(const k of Object.keys(clouds))delete clouds[k];
  }
  function paint(){
    if(!current)return;
    for(const v of views) for(const object of v.scene.children){
      if(!object.isPoints)continue;
      const {branch,indices}=object.userData,values=current.data[branch],mode=$('cloud-color').value;
      const attr=new Float32Array(indices.length*3);
      const [lo,hi]=current.sample.range,den=Math.max(hi-lo,1e-6);
      const modelColor=new THREE.Color(colors[branch]);
      for(let j=0;j<indices.length;j++){
        const i=indices[j];let c;
        if(mode==='model'){attr.set(modelColor.toArray(),j*3);continue;}
        if(mode==='depth')c=manifest.colormap[Math.max(0,Math.min(255,Math.floor((values[i]-lo)/den*255)))];
        else c=current.rgb.subarray(i*4,i*4+3);
        attr.set([srgb[c[0]],srgb[c[1]],srgb[c[2]]],j*3);
      }
      object.geometry.setAttribute('color',new THREE.BufferAttribute(attr,3));
      object.material.size=Number($('cloud-size').value);
    }
    draw();
  }
  function addPoints(view,branch){
    const {positions,indices}=clouds[branch],geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
    const material=new THREE.PointsMaterial({size:Number($('cloud-size').value),vertexColors:true,
      sizeAttenuation:false});
    const object=new THREE.Points(geometry,material);object.userData={branch,indices};view.scene.add(object);
  }
  function rebuild(){
    if(!current)return;
    clear();
    const {sample,data,record,masks}=current,stride=Number($('cloud-density').value);
    for(const branch of ['after','teacher']){
      const k=$('cloud-camera').value==='shared'?record.teacher.intrinsics:record[branch].intrinsics;
      clouds[branch]=buildCloud(data[branch],k,masks[branch],sample.width,sample.height,stride);
      $('cloud-count-'+branch).textContent=clouds[branch].indices.length.toLocaleString()+' points';
      const fov=2*Math.atan(.5/k[0][0])*180/Math.PI;
      $('cloud-info-'+branch).textContent='Horizontal FOV '+fov.toFixed(1)+'° · meters';
    }
    const overlay=$('cloud-layout').value==='overlay';
    $('cloud-card-teacher').hidden=overlay;$('cloud-panels').classList.toggle('overlay',overlay);
    $('cloud-title-after').textContent=overlay?'Overlay · trained model + MoGe-2':'Trained model';
    addPoints(views[0],'after');addPoints(views[overlay?0:1],'teacher');
    $('cloud-projection-note').textContent=$('cloud-camera').value==='shared'
      ? 'Shared MoGe-2 intrinsics: isolates depth differences. This is a diagnostic reprojection.'
      : 'Each model’s recovered intrinsics, as in MoGe. Same origin and meter scale; no alignment.';
    paint();resize();
  }
  function preset(kind='orbit'){
    if(!current)return;
    const {sample,record}=current,depth=Math.max(.1,(sample.range[0]+sample.range[1])/2);
    const k=record.teacher.intrinsics,extent=Math.max(depth/k[0][0],depth/k[1][1],sample.range[1]-sample.range[0]);
    const v=views[0];v.controls.target.set(0,0,-depth);
    if(kind==='front')v.camera.position.set(0,0,Math.max(0,extent*.25));
    else v.camera.position.set(extent*.42,extent*.22,-depth+extent*1.05);
    v.camera.far=Math.max(1000,sample.range[1]*10);v.camera.updateProjectionMatrix();
    v.camera.lookAt(v.controls.target);v.controls.update();sync(v);
  }
  async function load(sample,data){
    const ownGeneration=++generation;current=null;clear();draw();
    $('cloud-view').setAttribute('aria-busy','true');$('cloud-status').textContent='Loading camera geometry…';
    const record=metadata.samples.find(s=>s.id===sample.id);
    try{
      const masks={};
      await Promise.all(['after','teacher'].map(async branch=>{
        const response=await fetch(record[branch].mask);if(!response.ok)throw Error('Validity mask unavailable');
        const bitmap=await createImageBitmap(await response.blob());
        if(bitmap.width!==sample.width||bitmap.height!==sample.height){bitmap.close();throw Error('Mask dimensions differ');}
        const canvas=document.createElement('canvas');canvas.width=sample.width;canvas.height=sample.height;
        const ctx=canvas.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();
        const rgba=ctx.getImageData(0,0,canvas.width,canvas.height).data;
        masks[branch]=Uint8Array.from({length:sample.width*sample.height},(_,i)=>rgba[i*4]>127?1:0);
      }));
      if(ownGeneration!==generation)return;
      const canvas=document.createElement('canvas');canvas.width=sample.width;canvas.height=sample.height;
      const ctx=canvas.getContext('2d');ctx.drawImage(data.rgb,0,0);
      current={sample,data,record,masks,rgb:ctx.getImageData(0,0,canvas.width,canvas.height).data};
      rebuild();preset();$('cloud-status').textContent='Drag either view to rotate · scroll to zoom · right-drag to pan · views are linked';
      $('cloud-view').setAttribute('aria-busy','false');
    }catch(error){if(ownGeneration===generation){$('cloud-status').textContent='3D unavailable: '+error.message;
      $('cloud-view').setAttribute('aria-busy','false');}throw error;}
  }
  function download(branch){
    if(!current)return;
    const {positions,indices}=clouds[branch],rgb=current.rgb;
    const header=`ply\nformat binary_little_endian 1.0\ncomment meters; x right y up z backward\nelement vertex ${indices.length}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
    const buffer=new ArrayBuffer(indices.length*15),view=new DataView(buffer);
    indices.forEach((i,j)=>{for(let c=0;c<3;c++){view.setFloat32(j*15+c*4,positions[j*3+c],true);
      view.setUint8(j*15+12+c,rgb[i*4+c]);}});
    const link=document.createElement('a'),url=URL.createObjectURL(new Blob([header,buffer],{type:'application/octet-stream'}));
    link.href=url;link.download=current.sample.id+'-'+branch+'-'+$('cloud-camera').value+'.ply';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  ['cloud-density','cloud-camera'].forEach(id=>$(id).addEventListener('change',rebuild));
  $('cloud-layout').addEventListener('change',()=>{if($('cloud-layout').value==='overlay')$('cloud-color').value='model';rebuild();});
  $('cloud-color').addEventListener('change',paint);$('cloud-size').addEventListener('input',paint);
  $('cloud-reset').addEventListener('click',()=>preset());$('cloud-front').addEventListener('click',()=>preset('front'));
  ['after','teacher'].forEach(branch=>$('download-'+branch).addEventListener('click',()=>download(branch)));
  return {load,show(value){active=value;views.forEach(v=>v.controls.enabled=value);resize();},
    debug(){return {sample:current?.sample.id,counts:Object.fromEntries(Object.entries(clouds).map(([k,v])=>[k,v.indices.length])),
      positions:Object.fromEntries(Object.entries(clouds).map(([k,v])=>[k,Array.from(v.positions.slice(0,3))])),
      cameras:views.map(v=>({position:v.camera.position.toArray(),target:v.controls.target.toArray()}))};}};
}
