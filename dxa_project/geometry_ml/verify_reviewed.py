"""Check the requested correction scope, pixel preservation and regenerated ancestry."""
import csv,json,hashlib
from pathlib import Path
import pydicom,numpy as np
from openpyxl import load_workbook

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/retrained_20260929'
    backup=root/'dxa_project/backups/artifact_review_20260929'
    def rows(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    labels=rows(root/'Размеченные/labels.csv');ref={r['relative_path']:r for r in rows(root/'dxa_project/outputs/manifest.csv')}
    plan=json.loads((out/'correction_plan.json').read_text(encoding='utf-8'));checks=[]
    for change in plan['changes']:
        r=labels[change['number']-1];assert r['relative_path']==change['relative_path'];assert r['metal']=='1'
        assert ref[r['relative_path']]['spine_artifact']=='1'
        g=json.loads((root/'Размеченные'/r['geometry_path']).read_text(encoding='utf-8'))
        assert len(g['spine']['foreign_objects'])==change['visual_boxes']
        for folder in ('Исследования','Размеченные'):
            current=pydicom.dcmread(root/folder/r['relative_path']);before=pydicom.dcmread(backup/folder/r['relative_path'])
            assert np.array_equal(current.pixel_array,before.pixel_array)
            block=current.private_block(0x0011,'DXA_MANUAL_LABELER');value=current[block.get_tag(6)].value
            if isinstance(value,bytes):value=value.decode('ascii').strip()
            assert str(value)=='1'
            assert json.loads(current[block.get_tag(10)].value)['spine_artifact']==1
            assert json.loads(current[block.get_tag(9)].value)==g
        checks.append({'number':change['number'],'flag':1,'boxes':len(g['spine']['foreign_objects'])})
    affected_studies=set(c['study'] for c in plan['excel_cells'].values())
    study_members=[{'number':i,'flag':ref[r['relative_path']]['spine_artifact']} for i,r in enumerate(labels,1)
                   if r['label']=='SPINE' and ref[r['relative_path']]['reference_study_uid'] in affected_studies]
    assert all(r['flag']=='1' for r in study_members),'Per-study workbook correction disagrees with a sibling image'
    mismatches=[]
    for i,r in enumerate(labels,1):
        if r['label']!='SPINE':continue
        g=json.loads((root/'Размеченные'/r['geometry_path']).read_text(encoding='utf-8'))
        if int(float(ref[r['relative_path']]['spine_artifact']))!=int(bool(g['spine']['foreign_objects'])):mismatches.append(i)
    assert not mismatches
    report={'requested_scans':checks,'remaining_artifact_mismatches':mismatches,'affected_study_spine_members':study_members,
            'original_and_annotated_pixels_preserved':True,'geometry_matches_embedded_metadata':True}
    (out/'correction_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
