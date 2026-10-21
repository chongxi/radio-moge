import assert from 'node:assert/strict';
import {buildCloud,pointAt} from '../web/cloud-math.mjs';
const k=[[.8,0,.4],[0,1.2,.6],[0,0,1]],depth=new Float32Array([1,2,3,2,3,4]);
const cloud=buildCloud(depth,k,new Uint8Array([1,0,1,1,1,1]),3,2,1);
assert.equal(cloud.indices.length,5);assert(!cloud.indices.includes(1));
for(let j=0;j<cloud.indices.length;j++){
  const i=cloud.indices[j],p=cloud.positions.subarray(j*3,j*3+3),z=-p[2];
  assert(Math.abs((p[0]/z*k[0][0]+k[0][2])*3-.5-i%3)<1e-6);
  assert(Math.abs((-p[1]/z*k[1][1]+k[1][2])*2-.5-Math.floor(i/3))<1e-6);
}
assert.deepEqual(pointAt(2,[[1,0,.5],[0,1,.5],[0,0,1]],0,0,1,1),[0,-0,-2]);
const invalid=buildCloud(new Float32Array([NaN,Infinity,-1,0]),k,new Uint8Array([1,1,1,1]),2,2,1);
assert.equal(invalid.indices.length,0);
console.log('Point-cloud math passed: metric depth, pixel centers, axis conversion, mask and nonfinite filtering.');
