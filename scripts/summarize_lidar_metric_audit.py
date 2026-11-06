"""Saved-array robustness and bounded native-unit checks; never replace official scores."""
import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
from pathlib import Path
import zlib

import numpy as np
from PIL import Image

from audit_lidar_metric import independent_nearest, publish, read, ref, reference_metrics, rows


def pixel_map(camera, hw=(576,768)):
    fx,fy,cx,cy,k1,k2,p1,p2=camera['parameters']
    h,w=hw
    v,u=np.indices((h,w),dtype=np.float64)
    x=(u-cx*w/1920)/(fx*w/1920)
    y=(v-cy*h/1440)/(fy*h/1440)
    radius=x*x+y*y
    radial=1+k1*radius+k2*radius**2
    xd=x*radial+2*p1*x*y+p2*(radius+2*x*x)
    yd=y*radial+p1*(radius+2*y*y)+2*p2*x*y
    return (fx*xd+cx).astype(np.float32)*np.float32(256/1920), (fy*yd+cy).astype(np.float32)*np.float32(192/1440)


def native_selected(path, indices, codec):
    result={}
    lz4=ctypes.CDLL(ctypes.util.find_library('lz4'))
    lz4.LZ4_decompress_safe.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_int]
    lz4.LZ4_decompress_safe.restype=ctypes.c_int
    with Path(path).open('rb') as f:
        for i in range(max(indices)+1):
            header=f.read(4)
            if len(header)!=4: raise ValueError('Native frame missing')
            count=int.from_bytes(header,'little')
            if not 0<count<500000: raise ValueError('Invalid native frame size')
            if i not in indices:
                f.seek(count,1)
                continue
            data=f.read(count)
            if len(data)!=count: raise ValueError('Native frame truncated')
            if codec=='lz4_uint16_mm':
                destination=ctypes.create_string_buffer(192*256*2)
                source=ctypes.create_string_buffer(data)
                size=lz4.LZ4_decompress_safe(source,destination,len(data),len(destination))
                if size != 192*256*2: raise ValueError('Native uint16 frame dimensions differ')
                raw=np.frombuffer(destination.raw,dtype='<u2').reshape(192,256)
                depth=raw.astype(np.float32)/1000
            elif codec=='deflate_float32_metres':
                depth=np.frombuffer(zlib.decompress(data,wbits=-zlib.MAX_WBITS),dtype='<f4').reshape(192,256)
            else:
                raise ValueError('Unsupported selected codec')
            result[i]=depth.copy()
    return result


def summarize(args):
    output=args.output.resolve()
    report=read(output/'model/report.json')
    images=read(output/'model/per-image.json')
    records=rows(report['manifest']['path'])
    tie_axes=[]
    for n,m in [(768,640),(576,480)]:
        grid=np.arange(n,dtype=np.float32)[None]
        actual=np.asarray(Image.fromarray(grid).resize((m,1),Image.Resampling.NEAREST))[0]
        coordinates=(np.arange(m,dtype=np.float64)+.5)*n/m
        ideal=np.floor(coordinates)
        different=actual!=ideal
        all_ties=bool(np.all(np.abs(coordinates[different]-np.rint(coordinates[different]))<1e-10))
        if not all_ties or np.any(np.abs(actual[different]-ideal[different])!=1):
            raise ValueError('Resampling disagreement is not solely a nearest-neighbor tie')
        tie_axes.append(dict(source=n,target=m,differing_indices=int(different.sum()),all_equidistant_ties=all_ties))
    alt={name:[] for name in ['parent','final']}
    details=[]
    for record,item in zip(records,images,strict=True):
        if record['image'] != item['image']: raise ValueError('Image order changed')
        prepared=np.load(record['depth'],allow_pickle=False)
        target=independent_nearest(prepared,(480,640))
        with np.load(item['arrays']['path'],allow_pickle=False) as arrays:
            details.append(dict(id=record['id'],different_pixels=int(np.sum(target!=arrays['sensor'])),
                changed_support_pixels=int(np.sum((target>0)!=(arrays['sensor']>0)))))
            for name in alt:
                alt[name].append(reference_metrics(arrays[name],target)['abs_rel'])
    primary={name:report['summary'][name]['independent_abs_rel'] for name in alt}
    alternative={name:float(np.mean(values)) for name,values in alt.items()}
    scene_diffs={}
    for item in images:
        scene_diffs.setdefault(item['scene'],[]).append(item['models']['final']['independent']['abs_rel']-item['models']['parent']['independent']['abs_rel'])
    changes=np.array([np.mean(v) for v in scene_diffs.values()])
    rng=np.random.default_rng(43)
    bootstrap=changes[rng.integers(len(changes),size=(10000,len(changes)))].mean(axis=1)
    source_evidence=read(Path(report['manifest']['path']).parent/'scene-evidence.json')
    selected=[]
    for codec in ['lz4_uint16_mm','deflate_float32_metres']:
        scene=next(k for k,v in sorted(source_evidence.items()) if v['decoder']['codecs']==[codec])
        subset=[r for r in records if r['scene']==scene]
        native=native_selected(subset[0]['source_depth'],[r['frame_index'] for r in subset],codec)
        for record in subset:
            d=native[record['frame_index']]
            mx,my=pixel_map(record['camera']['camera'])
            ix,iy=np.rint(mx).astype(int),np.rint(my).astype(int)
            inside=(ix>=0)&(iy>=0)&(ix<256)&(iy<192)
            sample=np.zeros(mx.shape,np.float32)
            sample[inside]=d[iy[inside],ix[inside]]
            prepared=np.load(record['depth'],allow_pickle=False)
            support=prepared>0
            delta=np.abs(sample[support]-prepared[support])
            selected.append(dict(id=record['id'],codec=codec,source=record['source_depth'],
                prepared_depth=ref(record['depth']),supported_pixels=int(support.sum()),
                unequal_supported_pixels=int(np.count_nonzero(delta)),maximum_difference_metres=float(delta.max()),
                native_min_metres=float(d[d>0].min()),native_max_metres=float(d.max()),
                anonymization_support_reused_not_revalidated=True))
    publish(output/'resampling-details.json',details)
    result=dict(stage='saved_array_lidar_robustness_and_native_units_complete',created_at=datetime.now(timezone.utc).isoformat(),
        model_audit=ref(output/'model/report.json'),cpu_audit=ref(output/'cpu/report.json'),
        primary_independent_abs_rel=primary,
        resampling=dict(axes=tie_axes,all_disagreements_are_equidistant_nearest_ties=True,
            affected_depth_pixels=sum(x['different_pixels'] for x in details),
            changed_support_pixels=sum(x['changed_support_pixels'] for x in details),
            alternative_tie_convention_abs_rel=alternative,
            alternative_relative_change_percent=100*(alternative['final']/alternative['parent']-1),
            primary_relative_change_percent=100*(primary['final']/primary['parent']-1),
            changes_published_metric=False,details=ref(output/'resampling-details.json')),
        source_decode_checks=selected,source_unit_and_supported_value_checks_exact=all(x['unequal_supported_pixels']==0 for x in selected),
        primary_formula_error_found=False,
        exploratory_scene_differences=dict(scenes=len(changes),better=int(np.sum(changes<0)),worse=int(np.sum(changes>0)),
            bootstrap_replicates=10000,seed=43,mean=float(changes.mean()),
            percentile95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
            validation_repeatedly_used=True,independent_test_or_significance_claim=False),
        diagnostic_scale_only=dict(parent_abs_rel=report['summary']['parent']['diagnostic_scaled_abs_rel'],
            final_abs_rel=report['summary']['final']['diagnostic_scaled_abs_rel'],
            relative_change_percent=100*(report['summary']['final']['diagnostic_scaled_abs_rel']/report['summary']['parent']['diagnostic_scaled_abs_rel']-1),
            replaces_primary=False,real_world_planarity_proven=False),
        limitations=['The native-unit check covers eight frames in two scenes, both actual compressed formats.',
            'No independent physical RGB/depth calibration or sensor-confidence guarantee is established.',
            'RGB/depth mapping is supported by the publisher alignment convention; numeric reproduction is not physical calibration.',
            'Published metric includes inferred focal, metric scale and errors in the iPhone reference; it is not a flatness metric.',
            'The nearest-neighbor alternative is a sensitivity diagnostic, not a corrected benchmark or a favorable new selection.'],
        model_inference=False,optimizer_updates=0,auditor=ref(__file__))
    publish(output/'robustness-report.json',result)
    import json
    print(json.dumps(dict(report=ref(output/'robustness-report.json'),resampling=result['resampling'],source_checks_exact=result['source_unit_and_supported_value_checks_exact'],scenes=result['exploratory_scene_differences'],scale=result['diagnostic_scale_only'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    summarize(parser.parse_args())
