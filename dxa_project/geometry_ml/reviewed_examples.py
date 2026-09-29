"""Versioned module examples from the completed reviewed-data light pipeline."""
import csv,json,html,hashlib
from PIL import Image
import numpy as np
from .retrain_reviewed import ROOT,OUT
from .data import load_records,read_dicom_image
from .predict import predict_file
from .model_examples import score
from .build_team_report import filtered
from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.pilot_30 import _overlay

def main():
    public=ROOT/'dxa_project/team_demo';assets=public/'reviewed_examples_20260929';assets.mkdir(exist_ok=True)
    folder=OUT/'light';saved=json.loads((folder/'evaluation/predictions.json').read_text(encoding='utf-8'))
    records={r.relative_path:r for r in load_records(ROOT)}
    with (ROOT/'Размеченные/labels.csv').open(encoding='utf-8-sig',newline='') as f:numbers={r['relative_path'].replace('\\','/'):i for i,r in enumerate(csv.DictReader(f),1)}
    cache={}
    for i,row in enumerate(saved,1):
        r=records[row['relative_path']];pred=json.loads((folder/f'evaluation/prediction_{i:03}.json').read_text(encoding='utf-8'))
        actual=pred
        if pred['region']!=r.region:pred=predict_file(r.source_path,folder,force_region=r.region)
        image=read_dicom_image(r.source_path);gt=prepare_geometry(json.loads(r.geometry_path.read_text(encoding='utf-8')),r.region)
        cache[r.relative_path]=(image,gt,pred,actual)
    sections=[('router','Определение области (router)'),('lines','Межпозвоночные линии (spine lines)'),('crests','Подвздошные точки (iliac crests)'),
              ('artifact','Артефакты (artifact)'),('points','Три точки бедра (hip landmarks)'),('roi','Область интереса (ROI)'),('mask','Малый вертел (segmentation)')]
    parts=['<h1>Простая модель после исправления разметки: успехи и ошибки</h1><p>Это иллюстрации на контрольных оригиналах, не дополнительный тест. Для геометрических модулей задаётся верная область, чтобы отдельно оценить их работу. Номера соответствуют labels.csv. По возможности пять успехов и пять ошибок без повторов пикселей.</p>',
           '<p>Условия отбора: точки бедра ≤10 мм, подвздошные точки ≤5 мм, одинаковое число линий и отклонение ≤5 мм, IoU ROI ≥0,85, Dice маски ≥0,70; рамки артефактов сопоставляются при IoU ≥0,5. Это допуски для иллюстраций.</p>']
    summary={}
    for key,title in sections:
        candidates=[]
        for row in saved:
            rel=row['relative_path'];r=records[rel]
            if key!='router' and (r.region=='SPINE')!=(key in ('lines','crests','artifact')):continue
            image,gt,pred,actual=cache[rel]
            if key=='router':
                passed=actual['region']==r.region;severity=1-max(actual['router_probabilities'].values());text=f'Эталон {r.region}; модель {actual["region"]}.';target=True
            else:
                passed,severity,text,target=score(key,gt,pred['geometry'],r)
                if key=='points':target=any(v is not None for v in gt['hip']['landmarks'].values())
                if key=='crests':target=any(v is not None for v in gt['spine']['iliac_crests'].values())
            candidates.append({'rel':rel,'passed':passed,'severity':severity,'text':text,'target':target,'hash':hashlib.sha256(image.tobytes()).hexdigest()})
        def unique(pool):
            seen=set();result=[]
            for item in pool:
                if item['hash'] not in seen:seen.add(item['hash']);result.append(item)
            return result
        good=unique(sorted((c for c in candidates if c['passed']),key=lambda c:(not c['target'],c['severity'])))[:5]
        bad=unique(sorted((c for c in candidates if not c['passed']),key=lambda c:(not c['target'],-c['severity'])))[:5]
        imperfect=[] if good else unique(sorted(candidates,key=lambda c:c['severity']))[:5]
        parts.append(f'<h2>{title}</h2><p>Успешных по указанному допуску: {sum(c["passed"] for c in candidates)} / {len(candidates)}.</p>')
        summary[key]={'success_count':int(sum(c['passed'] for c in candidates)),'images':len(candidates),'example_numbers':[]}
        for category,pool in (('Успехи',good),('Лучшие, но ещё неточные',imperfect),('Ошибки',bad)):
            if not pool:continue
            parts.append(f'<h3>{category}: {len(pool)}</h3>')
            for j,c in enumerate(pool):
                rel=c['rel'];r=records[rel];image,gt,pred,actual=cache[rel];number=numbers[rel]
                base=Image.fromarray((image*255).astype('uint8')).convert('RGB');panels=[]
                for panel_index,g in enumerate((None,gt,actual['geometry'] if key=='router' else pred['geometry'])):
                    panel=base.copy()
                    if g is not None:panel=_overlay(panel,filtered(g,key) if key!='router' else g,actual['region'] if key=='router' and panel_index==2 else r.region)
                    panel.thumbnail((320,430));panels.append(panel)
                canvas=Image.new('RGB',(960,max(p.height for p in panels)),(12,20,28))
                for k,panel in enumerate(panels):canvas.paste(panel,(k*320+(320-panel.width)//2,0))
                category_id='ok' if c['passed'] else 'best_imperfect' if category=='Лучшие, но ещё неточные' else 'error'
                filename=f'{key}_{category_id}_{j}.png';canvas.save(assets/filename)
                parts.append(f'<article><h4>Снимок №{number}</h4><p>{html.escape(c["text"])}</p><p>Исходный снимок · разметка · предсказание</p><img loading="lazy" src="reviewed_examples_20260929/{html.escape(filename)}"></article>')
                summary[key]['example_numbers'].append(number)
    (public/'retraining_examples_20260929.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>DXA — примеры</title><style>body{background:#101923;color:#e8eff5;font:16px system-ui;margin:30px}img{max-width:100%}article{border:1px solid #35485b;padding:18px;margin:20px 0;max-width:1000px}h2{margin-top:45px}</style>'+''.join(parts)+'</html>',encoding='utf-8')
    (public/'retraining_examples_20260929.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'sections':len(summary),'examples':sum(len(r['example_numbers']) for r in summary.values())}))
if __name__=='__main__':main()
