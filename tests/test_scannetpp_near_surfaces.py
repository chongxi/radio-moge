import numpy as np
import pytest

from probe_scannetpp_native_near_surfaces import analytic_hits, sample_pixels


def planes(depths):
    vertices=np.array([[[-5,-5,z],[5,-5,z],[0,5,z]] for z in depths],dtype=np.float64).reshape(-1,3)
    return vertices,np.arange(len(vertices)).reshape(-1,3)


def test_parameter_is_not_always_camera_z_and_translation_is_preserved():
    vertices,faces=planes([2])
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    origin=np.array([1e6,-1e6,1e6])
    vertices=vertices @ rotation.T+origin
    ray=np.array([1,.5,1.]) @ rotation.T
    for direction,expected in ((ray,2),(ray/np.linalg.norm(ray),3)):
        hit=analytic_hits(vertices,faces,origin,direction)['first_hits'][0]
        assert hit['t']==pytest.approx(expected,abs=1e-8)
        z=((origin+hit['t']*direction-origin) @ rotation)[2]
        assert z==pytest.approx(2,abs=1e-8)
        assert abs(z-float(np.float32(z)))<1e-7


def test_raw_nearest_hit_does_not_skip_sub_near_plane_or_clamp():
    vertices,faces=planes([-1,0,.005,.010001,2])
    result=analytic_hits(vertices,faces,[0,0,0],[0,0,1])
    assert result['first_hits'][0]['t']==pytest.approx(.005)
    assert result['first_hits'][0]['face_id']==2
    assert result['first_beyond_t']['0.01']['t']==pytest.approx(.010001)
    assert result['first_beyond_t']['1.0']['t']==pytest.approx(2)


def test_face_ids_chunking_double_sided_winding_and_equal_depth_ties():
    vertices,faces=planes([3,1,1,4])
    baseline=analytic_hits(vertices,faces,[0,0,0],[0,0,1],chunk=1)
    for chunk in (2,4,17):
        assert analytic_hits(vertices,faces,[0,0,0],[0,0,1],chunk=chunk)==baseline
    assert [h['face_id'] for h in baseline['first_hits'][:2]]==[1,2]
    reverse=analytic_hits(vertices,faces[:,::-1],[0,0,0],[0,0,1])
    assert [h['t'] for h in reverse['first_hits']]==[1,1,3,4]


def test_boundary_outside_parallel_degenerate_and_negative_hits():
    vertices=np.array([[0,0,1],[1,0,1],[0,1,1]],float)
    faces=np.array([[0,1,2],[0,0,0]])
    for ray in ([0,0,1],[.5,0,1]):
        hit=analytic_hits(vertices,faces,[0,0,0],ray)['first_hits'][0]
        assert hit['boundary_hit'] and hit['ray_triangle_residual_m']<1e-12
    for ray in ([.8,.8,1],[1,0,0],[0,0,-1]):
        assert analytic_hits(vertices,faces,[0,0,0],ray)['first_hits']==[]
    with pytest.raises(ValueError):
        analytic_hits(vertices,faces,[0,0,0],[0,0,0])


def test_pixels_include_supported_quantiles_and_invalid_center_without_duplicates():
    depth=np.arange(36,dtype=np.float32).reshape(6,6)/10+.01
    valid=np.ones_like(depth,dtype=bool)
    valid[3,3]=False
    result=sample_pixels(depth,valid)
    assert len(result)<=6 and len({(y,x) for y,x,_ in result})==len(result)
    assert any((y,x)==(3,3) and 'invalid_nearest_center' in why for y,x,why in result)
    assert sum(any('supported_depth_quantile' in s for s in why) for _,_,why in result)==3
