"""Fifteen actual test illustrations selected to cover target labels, not accuracy."""
import json
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from .final_assembly import ROOT,OUT,save


def choose_examples(rows,count=5):
    cells=sorted({(task,value) for r in rows for task,value in r['truth'].items() if value in (0,1)})
    matrix=np.array([[int(r['truth'].get(task)==value) for r in rows] for task,value in cells],dtype=float)
    # Cover every available positive and negative with exactly five test cases.
    rng=np.random.default_rng(42);cost=np.array([100*bool(r.get('synthetic'))+10*(r['truth_region']!=r['predicted_region']) for r in rows])+rng.random(len(rows))*.001
    constraints=LinearConstraint(np.vstack([np.ones(len(rows)),matrix]),[count]+[1]*len(cells),[count]+[np.inf]*len(cells))
    result=milp(cost,integrality=np.ones(len(rows)),bounds=Bounds(0,1),constraints=constraints,options={'time_limit':20})
    if result.x is None:
        raise RuntimeError('Cannot cover all available target polarities in five examples; report needs explicit relaxed selection')
    selected=[r for r,x in zip(rows,result.x) if x>.5]
    absent={task:[v for v in (0,1) if (task,v) not in cells] for task in sorted({k for r in rows for k in r['truth']})}
    return selected,{k:v for k,v in absent.items() if v}


def draw(ax,image,g,analysis=None):
    ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.axis('off')
    if g is None:return
    for line in g['spine']['disc_lines']:
        p=np.asarray(line['points']);ax.plot(p[:,0],p[:,1],color='#00d9ff',lw=1.4)
    for obj in g['spine']['foreign_objects']:
        from matplotlib.patches import Rectangle
        x,y,x2,y2=obj['bbox'];ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor='#ff9911',lw=1.6))
    for part,field in [('spine','iliac_crests'),('hip','landmarks')]:
        for k,p in g[part][field].items():
            if p is not None:
                ax.scatter(*p,c='#00ff65',s=15);ax.text(p[0]+3,p[1],{'image_left':'L','image_right':'R','greater_trochanter':'1','femoral_neck':'2','ischial_bone':'3'}[k],color='#00ff65',fontsize=9)
    box=g['hip']['roi_box']
    if box:
        from matplotlib.patches import Rectangle
        x,y,x2,y2=box;ax.add_patch(Rectangle((x,y),x2-x,y2-y,fill=False,edgecolor='#c88aff',lw=1.5))
    pixels=g['hip'].get('lesser_trochanter_pixels',[])
    if pixels:
        p=np.asarray(pixels);ax.scatter(p[:,0],p[:,1],c='red',s=1,alpha=.55)
    if analysis:
        chain=[analysis.get('upper_fragment'),*analysis.get('axes',[]),analysis.get('lower_fragment')]
        for axis in chain:
            if not axis or not axis.get('axis_points'):continue
            p=np.asarray(axis['axis_points']);ax.plot(p[:,0],p[:,1],color='#ffda33',lw=1)
        if analysis.get('global_axis_points'):
            p=np.asarray(analysis['global_axis_points']);ax.plot(p[:,0],p[:,1],color='#ff55e8',lw=2)


def build(records,output,augmented,labels,protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .data import read_dicom_image
    from dxa_project.augmentation.core import prepare_geometry
    tested=json.loads((output/'predictions.json').read_text(encoding='utf-8'));lookup={r.relative_path:(i,r) for i,r in enumerate(records,1)}
    from .predict import predict_file
    extra_predictions={}
    calibration=json.loads((output/'calibration.json').read_text(encoding='utf-8'))
    for region in ('SPINE','LEG_LEFT','LEG_RIGHT'):
        pool=[r for r in tested if r['truth_region']==region]
        _,absent=choose_examples(pool)
        candidates={}
        for task,values in absent.items():
            for value in values:
                matching=[r for r in augmented if r.region==region and protocol['partition_by_path'][r.source_id]=='test' and labels[r.relative_path].get(task)==value]
                for r in sorted(matching,key=lambda r:r.relative_path)[:3]:candidates[r.relative_path]=r
        for r in candidates.values():
            threshold=calibration['rotation_threshold_mm2']/(r.spacing_mm[0]*r.spacing_mm[1])
            pred=predict_file(r.source_path,output.parent,rotation_area_threshold_px2=threshold,spacing_override_mm=r.spacing_mm)
            truth={k:labels[r.relative_path].get(k) for k in pool[0]['truth']}
            tested.append({'relative_path':r.relative_path,'study':r.study,'truth_region':region,'predicted_region':pred['region'],
                           'truth':truth,'predicted':pred['quality_flags'],'scores':pred['quality_scores'],'synthetic':True})
            lookup[r.relative_path]=(None,r);extra_predictions[r.relative_path]=pred
    save(output/'example_augmented_predictions.json',extra_predictions)
    asset=ROOT/'dxa_project/team_demo/final_model_20260929_assets';asset.mkdir(exist_ok=True)
    sections=['## 15 примеров работы на test','',
              'В каждой области выбраны ровно пять примеров, чтобы покрыть оба класса каждого таргета. При отсутствии класса в исходном test добавлен явно помеченный синтетический снимок из test-исследования. Подбор иллюстративный; итоговые метрики рассчитаны только на исходном test. **0 — нарушения нет, 1 — нарушение есть, null — ответ не определён.** Эталон — разметка, а не ответ модели.', '',
              'Слева входной снимок, в центре ручная геометрия, справа предсказание. Голубые линии — разделители; зелёные точки — ориентиры (1 большой вертел, 2 шейка бедра, 3 седалищная кость, L/R подвздошные точки); фиолетовая рамка — ROI; красное — малый вертел; оранжевые рамки — артефакты. На позвоночнике жёлтые линии — локальные оси, розовая — общая ось.', '']
    audit={}
    for region,title in [('SPINE','Позвоночник'),('LEG_LEFT','Левая нога'),('LEG_RIGHT','Правая нога')]:
        selected,absent=choose_examples([r for r in tested if r['truth_region']==region]);audit[region]={'paths':[r['relative_path'] for r in selected],'synthetic_paths':[r['relative_path'] for r in selected if r.get('synthetic')],'missing_polarities_in_examples':absent}
        sections+=['### '+title,'']
        if absent:sections+=['В test отсутствуют классы: '+str(absent)+'. Эти примеры нельзя показать без замены test.', '']
        for number,row in enumerate(selected,1):
            i,r=lookup[row['relative_path']];pred=extra_predictions[r.relative_path] if i is None else json.loads((output/f'prediction_{i:03}.json').read_text(encoding='utf-8'))
            image=read_dicom_image(r.source_path);gt=prepare_geometry(json.loads(r.geometry_path.read_text(encoding='utf-8')),r.region)
            fig,axes=plt.subplots(1,3,figsize=(12,7));draw(axes[0],image,None);draw(axes[1],image,gt);draw(axes[2],image,pred['geometry'],pred.get('vertebral_axes'))
            for ax,label in zip(axes,['Вход','Ручная разметка','Модель: '+pred['region']]):ax.set_title(label,fontsize=12)
            fig.tight_layout();name=f'{region.lower()}_{number:02}.png';fig.savefig(asset/name,dpi=125,bbox_inches='tight');plt.close(fig)
            save(asset/name.replace('.png','.json'),{'source':r.relative_path,'truth':row['truth'],'prediction':pred})
            sections+=[f'#### {title}, пример {number}','',f'![{title}, пример {number}](team_demo/final_model_20260929_assets/{name})','',
                       '| Таргет | Разметка | Модель | Score |','|---|---:|---:|---:|']
            for key,value in row['truth'].items():
                score=pred['quality_scores'].get(key);guess=pred['quality_flags'].get(key)
                sections.append(f'|{key}|{value if value is not None else "null"}|{guess if guess is not None else "null"}|{score:.4f}|' if score is not None else f'|{key}|{value if value is not None else "null"}|{guess if guess is not None else "null"}|—|')
            if region=='SPINE':sections+=['',f'Угол с вертикалью: {pred.get("spine_axis_angle_deg")}°. Сколиоз — отдельный классификатор по локальной метке; это не угловой порог.']
            if row.get('synthetic'):sections+=['','**Синтетический пример из удержанного test-исследования.** Использован для демонстрации отсутствующего в исходном test класса; в итоговые метрики не включён.']
            sections+=['',f'Исходник: `{r.relative_path}`. Полный JSON: `{name.replace(".png",".json")}` рядом с иллюстрацией.','']
    save(output/'example_selection.json',audit);return sections
