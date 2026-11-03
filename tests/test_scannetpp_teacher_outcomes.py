from copy import deepcopy
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from audit_scannetpp_teacher_outcomes import bind_outcome, checked_metrics, run, validate_generation
from radio_moge.corpus_admission import file_ref
from radio_moge.geometry_cache import GeometryPackWriter, canonical_json, publish_json


def fixture(root):
    config = dict(root=str(root), manifest_sha256='a' * 64, expected_counts={'scannetpp_dslr': 4},
                  teacher=dict(repo='fixture', revision='b' * 40))
    rows = [dict(corpus_ordinal=i, image=str(root / f'rgb-{i}.png'), mask=str(root / f'mask-{i}.png'),
        sha256=f'{i:064x}', dataset='scannetpp_dslr', split='train', official_split='nvs_sem_train',
        group='scannetpp:' + str(i // 2), scene=str(i // 2),view=f'{i}.png',width=8,height=8) for i in range(4)]
    with sqlite3.connect(root / 'inventory.sqlite') as db:
        db.execute('CREATE TABLE images(ordinal,image,row_json,scene_group)')
        db.executemany('INSERT INTO images VALUES (?,?,?,?)',
                       [(i,r['image'],json.dumps({k:v for k,v in r.items() if k!='corpus_ordinal'}),r['group'])
                        for i,r in enumerate(rows)])
    publish_json(root / 'inventory.json', dict(stage='inventory_complete',
        manifest_sha256=config['manifest_sha256'],dataset_counts=config['expected_counts'],
        inventory_sha256=file_ref(root / 'inventory.sqlite')['sha256']))
    audit=root/'native'
    audit.mkdir()
    scenes=[]
    for scene in ('0','1'):
        path=audit/(scene+'.json')
        publish_json(path,dict(scene=scene,stage='needs_review',images_in_training_manifest=2))
        scenes.append(dict(scene=scene,stage='needs_review',report=path.name,sha256=file_ref(path)['sha256']))
    publish_json(audit/'report.json',dict(stage='complete',manifest_sha256=config['manifest_sha256'],
        expected_scenes=2,scenes_completed=2,scenes_needing_review=2,scenes=scenes))
    route_path=root/'scannetpp-review-teacher-routing-20261102.json'
    publish_json(route_path,dict(dataset='scannetpp_dslr',label_kind='teacher_pseudo',
        manifest_sha256=config['manifest_sha256'],metric_calibrated=False,fill_native_holes=False,
        training_admission=False,scene_counts={'0':2,'1':2},scope_images=4,source_audit=file_ref(audit/'report.json')))
    route_ref=file_ref(route_path)
    original_policy=dict(dataset='scannetpp_dslr',label_kind='teacher_pseudo',teacher=config['teacher'],
        manifest_sha256=config['manifest_sha256'],checkpoint_sha256='c'*64,
        candidate_routing_sha256=route_ref['sha256'],native_geometry_read=False,native_metric_gt=False,
        force_projection=True,minimum_valid_fraction=.05,generator_sha256='old',
        anonymity='original mask==255 AND 5x5 erosion AND teacher support; no resizing/filling')
    directories=('scannetpp-review-teacher-recovered-01','scannetpp-review-teacher-cache-02')
    reports=[]
    for index,name in enumerate(directories):
        directory=root/name
        directory.mkdir()
        policy=dict(original_policy,**(dict(generator_sha256='new',recovery_helper_sha256='helper') if index else {}))
        policy_hash=hashlib.sha256(canonical_json(policy)).hexdigest()
        writer=GeometryPackWriter(directory/'packs',policy)
        for row in rows[:2] if index==0 else rows[2:3]:
            stats=dict(anonymous_mask_sha256='d'*64,valid_pixels=64,total_pixels=64,
                valid_fraction=1.,teacher_valid_fraction=1.,projection_max_scaled_error=1e-7,
                projection_max_abs_error_m=1e-6)
            meta=dict(dataset='scannetpp_dslr',label_kind='teacher_pseudo',camera_kind='teacher_estimated',
                metric_calibrated=False,teacher_revision=config['teacher']['revision'],policy_sha256=policy_hash,
                image_sha256=row['sha256'],native_source_review_unresolved=True,source_quality=stats,
                source_id=row['scene']+'/'+row['view'],
                conversion='pinned_moge2_infer_forced_projection_with_original_anonymity_mask',
                source_proofs=[dict(source='fixture/model.pt',sha256='c'*64),
                    dict(source=row['image'],sha256=row['sha256']),dict(source=row['mask'],sha256='d'*64),
                    dict(source=route_ref['path'],sha256=route_ref['sha256'])])
            writer.add(row,np.ones((8,8),np.float32),np.ones((8,8),bool),np.eye(3),meta)
        writer.close()
        gen=dict(stage='complete',mode='generate',training_started=False,training_admission=False,
            optimizer_updates=0,completed=2 if index==0 else 1,verified_cache_members=2 if index==0 else 1,
            rejected=index,expected_scope=2,policy_sha256=policy_hash)
        if index==0:
            scope=directory/'scope.jsonl'
            scope.write_text(''.join(json.dumps(dict(ordinal=r['corpus_ordinal'],sha256=r['sha256']))+'\n' for r in rows[:2]))
            gen.update(recovered_verified_subset=True,scope_sha256=file_ref(scope)['sha256'])
        else:
            gen.update(teacher_frozen=True,recovered_subset=reports[0],recovered_skipped=2,original_scope=4)
            publish_json(directory/'rejections.json',[dict(image=rows[3],reason='insufficient_teacher_and_anonymous_support')])
        publish_json(directory/'report.json',gen)
        reports.append(file_ref(directory/'report.json'))
    preflight=root/'scannetpp-review-teacher-preflight-02'
    preflight.mkdir()
    publish_json(preflight/'report.json',dict(stage='complete',passed=True,mode='preflight',
        verified_cache_members=4,rejected=0,policy_sha256=policy_hash))
    publish_json(preflight/'visual-review.json',dict(stage='visual_review_complete',passed=True,
        policy_sha256=policy_hash,preflight_report_sha256=file_ref(preflight/'report.json')['sha256'],
        reviewed_scenes=['0','1'],evidence=[route_ref]))
    return config,rows,route_ref


def test_recovered_and_remainder_count_every_original_once_without_clearing_native_flags(tmp_path):
    config,_,_=fixture(tmp_path)
    result=run(config,'audit')
    assert result['outcome_counts']=={'teacher_candidate':3,'teacher_rejection_needs_review':1}
    assert result['expected_scope']==4 and result['scenes']==2
    assert sorted(result['candidate_policy_counts'].values())==[1,2]
    assert not result['native_source_flags_cleared'] and not result['source_quality_accepted']
    assert not result['pack_payload_hashes_rechecked'] and not result['independent_array_readback_performed']
    review=json.loads((tmp_path/'audit/review-candidates.json').read_text())
    assert {r['corpus_ordinal'] for r in review}=={0,1,2}


def test_partial_readback_and_wrong_recovery_binding_are_rejected(tmp_path):
    fixture(tmp_path)
    rec_ref=file_ref(tmp_path/'scannetpp-review-teacher-recovered-01/report.json')
    rec=json.loads((tmp_path/'scannetpp-review-teacher-recovered-01/report.json').read_text())
    fresh=json.loads((tmp_path/'scannetpp-review-teacher-cache-02/report.json').read_text())
    for key,value in [('stage','generating_cache'),('verified_cache_members',0),('teacher_frozen',False),
                      ('recovered_skipped',1),('original_scope',5),('recovered_subset',{})]:
        with pytest.raises(ValueError):
            validate_generation(rec,dict(fresh,**{key:value}),rec_ref,4)


def test_duplicate_changed_and_wrongly_recovered_rgb_fail(tmp_path):
    _,rows,_=fixture(tmp_path)
    with sqlite3.connect(':memory:') as db:
        db.execute('CREATE TABLE images(ordinal,row_json,outcome,subset,policy,evidence_json,entry_key,support,p01_ratio,p99_ratio,max_ratio,reason)')
        db.execute('INSERT INTO images(ordinal,row_json) VALUES (?,?)',(0,json.dumps(rows[0])))
        for row,subset in [(dict(rows[0],view='different.png'),'recovered'),(rows[0],'remainder'),
                           (dict(rows[0],corpus_ordinal=9),'recovered')]:
            with pytest.raises(ValueError,match='RGB outcome'):
                bind_outcome(db,row,subset,{0:'hash'},'policy',{})
        bind_outcome(db,rows[0],'recovered',{0:'hash'},'policy',{},key='key',metrics=(1.,1.,1.,1.))
        with pytest.raises(ValueError,match='duplicate'):
            bind_outcome(db,rows[0],'recovered',{0:'hash'},'policy',{})


@pytest.mark.parametrize('field,value',[
    ('native_source_review_unresolved',False),('metric_calibrated',True),('teacher_revision','wrong'),
    ('source_proofs',[]),('projection_max_scaled_error',1.),('teacher_valid_fraction',.5),
    ('anonymous_mask_sha256','bad'),('valid_pixels',63)])
def test_teacher_cannot_clear_native_flags_or_change_masks_provenance_and_projection(tmp_path,field,value):
    _,_,route_ref=fixture(tmp_path)
    directory=tmp_path/'scannetpp-review-teacher-cache-02/packs'
    policy=json.loads((directory/'policy.json').read_text())
    receipt=json.loads((directory/'pack-000000.json').read_text())
    entry=deepcopy(receipt['entries'][0])
    meta=entry['geometry_metadata']
    if field in meta['source_quality']:
        meta['source_quality'][field]=value
    else:
        meta[field]=value
    entry['key']=hashlib.sha256(canonical_json(meta)).hexdigest()
    with pytest.raises(ValueError):
        checked_metrics(entry,policy,receipt['policy_sha256'],route_ref)
