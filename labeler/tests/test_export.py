import json
from pathlib import Path
import pandas as pd
import pytest
from geometry import empty_geometry, geometry_sidecar, save_geometry
from export_geometry import export


def test_export_jsonl_and_summary(tmp_path):
    root=tmp_path/'out';root.mkdir()
    rel='study/a.dcm';g=empty_geometry(300,400)
    g['spine']['iliac_crests']['image_left']=[20,300]
    g['complete']['spine']=True
    sidecar=geometry_sidecar(root,rel)
    save_geometry(sidecar,g,rel,'tester')
    pd.DataFrame([{'relative_path':rel,'label':'SPINE','side':'',
                   'geometry_path':sidecar.relative_to(root).as_posix()}]).to_csv(root/'labels.csv',index=False)
    n=export(root,root/'all.jsonl',root/'all.csv')
    assert n==1
    rec=json.loads((root/'all.jsonl').read_text().strip())
    assert rec['geometry']['spine']['iliac_crests']['image_left']==[20.0,300.0]
    csv=pd.read_csv(root/'all.csv')
    assert csv.loc[0,'iliac_points']==1
    assert csv.loc[0,'threshold_8bit']==128
    assert csv.loc[0,'trochanter_traces']==0
    assert csv.loc[0,'bone_contour_traces']==0
    assert bool(csv.loc[0,'spine_complete'])


def test_export_rejects_path_escape(tmp_path):
    root=tmp_path/'out';root.mkdir()
    pd.DataFrame([{'relative_path':'x','geometry_path':'../../x.json'}]).to_csv(root/'labels.csv',index=False)
    with pytest.raises(ValueError):
        export(root,root/'all.jsonl',root/'all.csv')
