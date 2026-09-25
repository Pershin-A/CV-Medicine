import json
from pathlib import Path
import pytest
from geometry import (
    empty_geometry, validate_geometry, geometry_sidecar, read_geometry,
    save_geometry, geometry_counts, is_newer_canvas_revision,
)


def test_roundtrip_and_all_annotation_types(tmp_path):
    g = empty_geometry(400, 800)
    g['spine']['disc_lines'] = [{'id':'l0','points':[[140.4, 180],[248, 181]]}]
    g['spine']['iliac_crests'] = {'image_left':[45,700], 'image_right':[342,701]}
    g['spine']['foreign_objects'] = [{'id':'f','kind':'metal','bbox':[90,20,105,78]}]
    g['hip']['landmarks'] = {
        'greater_trochanter':[45,120], 'femoral_neck':[200,220], 'ischial_bone':[320,320],
    }
    g['hip']['lesser_trochanter'] = [[120,310],[140,310],[135,330],[119,329]]
    g['hip']['lesser_trochanter_traces']['trochanter'] = [
        {'id':'pencil1','points':[[120,310],[122,311],[125,312],[129,310]]},
    ]
    g['hip']['lesser_trochanter_traces']['adjacent_bone'] = [
        {'id':'pencil2','points':[[118,304],[117,307],[119,310]]},
    ]
    g['hip']['roi_box'] = [30,40,240,700]
    g['image_view'] = {'mode':'threshold', 'threshold_8bit':173}
    g['complete'] = {'spine':True,'hip':False}
    normalized = validate_geometry(g, 400,800)
    path = geometry_sidecar(tmp_path,'study/scan.dcm')
    save_geometry(path, normalized, 'study/scan.dcm', 'tester')
    assert read_geometry(path,400,800) == normalized
    assert geometry_counts(normalized) == {
        'disc_lines':1,'iliac_points':2,'foreign_objects':1,
        'hip_landmarks':3,'lesser_trochanter':1, 'trochanter_traces':1,
        'bone_contour_traces':1, 'hip_roi':1, 'threshold_8bit':173,
    }
    lines = (tmp_path/'geometry'/'geometry_history.jsonl').read_text().splitlines()
    assert len(lines)==1 and json.loads(lines[0])['geometry']==normalized


def test_no_sidecar_returns_empty(tmp_path):
    g=read_geometry(tmp_path/'missing.json',320,600)
    assert g==empty_geometry(320,600)


@pytest.mark.parametrize('bad', [
    lambda g: g.update(image_width=199),
    lambda g: g['spine'].update(disc_lines=[{'points':[[1,2],[1,2]]}]),
    lambda g: g['spine']['iliac_crests'].update(image_left=[float('nan'),1]),
    lambda g: g['hip'].update(roi_box=[-1,0,10,10]),
    lambda g: g['hip'].update(lesser_trochanter=[[1,1],[2,2]]),
    lambda g: g['spine'].update(foreign_objects=[{'kind':'bad','bbox':[1,1,10,10]}]),
    lambda g: g.update(image_view={'mode':'threshold','threshold_8bit':256}),
    lambda g: g.update(image_view={'mode':'invalid','threshold_8bit':20}),
    lambda g: g['hip']['lesser_trochanter_traces'].update(trochanter=[{'points':[[1,2]]}]),
    lambda g: g['hip']['lesser_trochanter_traces'].update(adjacent_bone=[{'points':[[1,2],[1,2]]}]),
])
def test_rejects_invalid(bad):
    g=empty_geometry(200,200)
    bad(g)
    with pytest.raises(ValueError):
        validate_geometry(g,200,200)


def test_path_uses_hash_and_avoids_traversal(tmp_path):
    p=geometry_sidecar(tmp_path,'../../private/a.dcm')
    assert p.parent == tmp_path / 'geometry'
    assert p.suffix == '.json' and len(p.stem)==64


def test_normalizes_bbox_coordinate_order():
    g=empty_geometry(40,60)
    g['hip']['roi_box']=[30,50,10,20]
    assert validate_geometry(g,40,60)['hip']['roi_box']==[10,20,30,50]


def test_previous_version_geometry_remains_compatible():
    g=empty_geometry(100,200)
    # A v5.2 sidecar had a polygon, but no freehand trace or per-image view.
    g['hip']['lesser_trochanter']=[[10,10],[20,10],[18,18]]
    del g['hip']['lesser_trochanter_traces']
    del g['image_view']
    new=validate_geometry(g,100,200)
    assert new['hip']['lesser_trochanter']==[[10,10],[20,10],[18,18]]
    assert new['hip']['lesser_trochanter_traces']=={'trochanter':[],'adjacent_bone':[]}
    assert new['image_view']=={'mode':'original','threshold_8bit':128}


def test_threshold_is_saved_per_image(tmp_path):
    a,b=empty_geometry(200,300),empty_geometry(200,300)
    a['image_view']={'mode':'threshold','threshold_8bit':43}
    b['image_view']={'mode':'threshold','threshold_8bit':205}
    for rel,g in [('a/1.dcm',a),('b/2.dcm',b)]:
        save_geometry(geometry_sidecar(tmp_path,rel),validate_geometry(g,200,300),rel,'tester')
    assert read_geometry(geometry_sidecar(tmp_path,'a/1.dcm'),200,300)['image_view']['threshold_8bit']==43
    assert read_geometry(geometry_sidecar(tmp_path,'b/2.dcm'),200,300)['image_view']['threshold_8bit']==205


def test_out_of_order_canvas_revisions_cannot_restore_deleted_objects():
    original=empty_geometry(40,60)
    first={**original, '_client_revision': 1200}
    older={**original, '_client_revision': 1100}
    duplicate={**original, '_client_revision': 1200}
    newer={**original, '_client_revision': 1300}
    assert is_newer_canvas_revision(first,-1)
    assert not is_newer_canvas_revision(older,1200)
    assert not is_newer_canvas_revision(duplicate,1200)
    assert is_newer_canvas_revision(newer,1200)
    for bad in (None, {}, {'_client_revision': None}, {'_client_revision': True},
                {'_client_revision': float('nan')}):
        assert not is_newer_canvas_revision(bad,1200)
    assert validate_geometry(newer,40,60)==original, 'Client revision must never be stored in saved annotation'
