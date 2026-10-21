// The only coordinate conversion: OpenCV camera frame -> WebGL (x, -y, -z).
// Intrinsics are normalized to image width/height, with pixel-center sampling.
export function pointAt(depth, k, x, y, width, height) {
  return [((x + .5) / width - k[0][2]) / k[0][0] * depth,
          -((y + .5) / height - k[1][2]) / k[1][1] * depth, -depth];
}

export function buildCloud(depth, k, mask, width, height, stride=2) {
  if (depth.length !== width*height || mask.length !== width*height ||
      ![1,2,4].includes(stride) || !k.flat().every(Number.isFinite) ||
      k[0][0] <= 0 || k[1][1] <= 0) throw Error('Invalid point-cloud data');
  const positions=[], indices=[];
  for(let y=0;y<height;y+=stride) for(let x=0;x<width;x+=stride){
    const i=y*width+x,z=depth[i];
    if(!mask[i] || !Number.isFinite(z) || z<=0) continue;
    positions.push(...pointAt(z,k,x,y,width,height));indices.push(i);
  }
  return {positions:new Float32Array(positions),indices:new Uint32Array(indices)};
}
