"""Compare visibility/framing and optional geometry abstention on cached inference."""
import json
from pathlib import Path
from .evaluate import with_ci

def main():
    root=Path(__file__).resolve().parents[2];out=root/'dxa_project/outputs/improvements_v2'
    evaluation=out/'pipeline/evaluation';results=json.loads((evaluation/'predictions.json').read_text(encoding='utf-8'))
    rows=[]
    for i,r in enumerate(results,1):
        if r['truth_region']=='SPINE' or r['predicted_region']!=r['truth_region']:continue
        p=json.loads((evaluation/f'prediction_{i:03}.json').read_text(encoding='utf-8'))
        framing=r['predicted']['hip_position'];geometry=framing
        if framing==0 and p['landmark_geometry_check']['status']!='plausible':geometry=None
        rows.append({'study':r['study'],'truth':r['truth']['hip_position'],'framing':framing,'framing_and_geometry':geometry,
                     'score':r['scores']['hip_position']})
    report={}
    for rule in ('framing','framing_and_geometry'):
        usable=[{'study':r['study'],'truth':r['truth'],'prediction':r[rule],'score':r['score']}
                for r in rows if r['truth'] is not None and r[rule] is not None]
        report[rule]={'coverage':len(usable)/len(rows) if rows else None,'abstentions':len(rows)-len(usable),**with_ci(usable,200)}
    (out/'geometry_rule_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
