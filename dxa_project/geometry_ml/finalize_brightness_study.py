"""Audit decoded geometry and finish the saved brightness experiment (no training)."""
import json
import numpy as np
from PIL import Image, ImageDraw
from .spine_brightness_study import ROOT, DEST, read, filter_bright
from .data import load_records, load_augmented_records, DxaDataset
from .spine_penalty_study import target
from .report_reviewed import table


def geometry_audit(lines, record):
    g = read(record.geometry_path)
    h, w = g['image_height'], g['image_width']
    aspect = (w-1)*record.spacing_mm[1]/((h-1)*record.spacing_mm[0])
    gt, visible = target(record)
    gt = gt[:int(visible.sum())]
    ref_gap = np.diff(gt[:,0])*np.sqrt(1+((gt[:-1,1]+gt[1:,1])/2/aspect)**2)
    minimum = 1/(1/ref_gap.mean()+2)
    values = []
    for line in lines:
        a, b = sorted(line['points'], key=lambda p:p[0])
        slope = (b[1]-a[1])/max(b[0]-a[0],1e-6)
        values.append([(a[1]+((w-1)/2-a[0])*slope)/(h-1), slope*(w-1)/(h-1)/aspect])
    if len(values)<2:
        return {'defined':False, 'all_pass':False, 'extra':len(lines)>len(gt)}
    v = np.array(sorted(values)); y,s = v.T
    gaps = np.diff(y)*np.sqrt(1+((s[:-1]+s[1:])/2)**2)
    close = gaps < minimum-1e-6
    top = y[0]*np.sqrt(1+s[0]**2) > gaps[:2].mean()+1e-6
    bottom = (1-y[-1])*np.sqrt(1+s[-1]**2) > gaps[-2:].mean()+1e-6
    return {'defined':True,'all_pass':not(close.any() or top or bottom),
            'extra':len(lines)>len(gt),'close':bool(close.any()),'top':bool(top),'bottom':bool(bottom),
            'minimum_over_height':float(minimum),'min_gap_over_height':float(gaps.min())}


def run():
    report = read(DEST/'report.json'); predictions = read(DEST/'predicted_lines.json')
    protocol = read(DEST/'protocol.json')
    originals = load_records(ROOT)
    records = originals+load_augmented_records(ROOT,ROOT/'dxa_project/outputs/augmented_15000_20260929',originals)
    by = {r.relative_path:r for r in records}; test=[by[p] for p in protocol['partitions']['test']]
    audits={}
    for name, pred in predictions.items():
        rows=[dict(relative_path=r.relative_path,**geometry_audit(pred[r.relative_path],r)) for r in test]
        extra=[v for v in rows if v['extra']]
        audits[name]={'images':len(rows),'defined':sum(v['defined'] for v in rows),
                     'all_pass':sum(v['all_pass'] for v in rows),
                     'extra_images':len(extra),'extra_all_pass':sum(v['all_pass'] for v in extra),
                     'close':sum(v.get('close',False) for v in rows),
                     'top':sum(v.get('top',False) for v in rows),
                     'bottom':sum(v.get('bottom',False) for v in rows),'details':rows}
    report['decoded_geometry_audit']=audits
    contour_audits={}
    manual={r['relative_path']:r['axes'] for r in report['axis_details']['manual']}
    for name,rows in report['axis_details'].items():
        outcomes=[]
        for row in rows:
            axes=row['axes'];ref=manual[row['relative_path']]
            if not axes.get('upper_fragment') or not axes.get('lower_fragment') or not axes['axes']:
                outcomes.append({'defined':False,'pass':False});continue
            lengths=np.array([v['length_mm'] for v in axes['axes']])
            reference=np.mean([v['length_mm'] for v in ref['axes']])
            record=by[row['relative_path']]
            height=(read(record.geometry_path)['image_height']-1)*record.spacing_mm[0]
            minimum=height/(height/reference+2)
            passed=(lengths.min()>=minimum and axes['upper_fragment']['length_mm']<=lengths[:2].mean()
                    and axes['lower_fragment']['length_mm']<=lengths[-2:].mean())
            outcomes.append({'defined':True,'pass':bool(passed)})
        contour_audits[name]={'images':len(outcomes),'defined':sum(v['defined'] for v in outcomes),'all_pass':sum(v['pass'] for v in outcomes)}
    report['contour_geometry_audit']=contour_audits
    # Show exactly the tensor used by the model, including quantiles after resize.
    size=__import__('torch').load(DEST/'raw_new_loss.pt',map_location='cpu',weights_only=False)['size']
    dataset=DxaDataset(test,size); assets=ROOT/'dxa_project/team_demo/spine_brightness_assets'
    for i in list(range(5))+list(range(34,39)):
        image,t=dataset[i];r=test[i]
        for name,v in report['results'].items():
            filtered,_=filter_bright(image[None],[t],v['fraction'])
            a=filtered[0].numpy()*np.array([.229,.224,.225])[:,None,None]+np.array([.485,.456,.406])[:,None,None]
            l,top=t['pad_left'],t['pad_top']; dw=round(t['width']*t['scale']);dh=round(t['height']*t['scale'])
            a=np.clip(a[:,top:top+dh,l:l+dw],0,1).transpose(1,2,0)
            pic=Image.fromarray((a*255).round().astype('uint8')).resize((t['width'],t['height']),Image.Resampling.NEAREST)
            draw=ImageDraw.Draw(pic)
            for line in predictions[name][r.relative_path]:draw.line([tuple(p) for p in line['points']],fill='lime',width=2)
            pic.thumbnail((300,430));pic.save(assets/f'{i}_{name}.png')
    heading='<section><h2>Выводы по завершённому тесту</h2><p><b>Жёсткий фильтр яркости не улучшил результат.</b> При одинаковых новых штрафах исходник дал ошибку 4,23 мм, полноту 82,4% и точность 67,2%; фильтры 10–20% дали 5,55–6,53 мм, 64,7–70,2% и 45,1–49,6%. Дополнительный контроль 15% с прежней потерей тоже ухудшил полноту: 42,8% против 66,6% у исходника. Это результат четырёх эпох дообучения декодера с замороженным энкодером, а не доказательство бесполезности любых способов подавления фона.</p><p>Новые штрафы на исходнике уменьшили недобор с 75,3% до 0% и ошибку пар с 8,52 до 4,23 мм. Однако число линий выше эталона встречается в 77,9% случаев. Неточная разметка может пропускать реальные линии, поэтому сам перебор не объявляется ошибкой. Ниже проверены ограничения для <b>всех</b> предсказанных линий, включая дополнительные. Мягкие штрафы обучаются по кандидатам, сопоставленным с эталонными линиями: дополнительные пики пока могут обходить эти штрафы.</p></section>'
    geometry= '<h2>Проверка геометрии всех декодированных линий</h2><p>Здесь длины приближены по центрам линий и их нормалям с учётом размера пикселя; это не полные контурные оси. Проверяем короткие межлинейные интервалы и длины крайних фрагментов относительно двух соседних полных интервалов. «Все условия» — одновременно все три ограничения. Столбец дополнительных линий показывает, сколько случаев с числом выше эталона прошли эти условия.</p>'+table(['Вариант','Все условия /154','Сверх эталона','Из них все условия','Близкие линии','Длинный верх','Длинный низ'],[[k,v['all_pass'],v['extra_images'],v['extra_all_pass'],v['close'],v['top'],v['bottom']] for k,v in audits.items()])
    geometry+='<h3>Проверка полных контурных осей на 30 примерах</h3><p>Здесь использованы реальные длины построенных осей, а адаптивный минимум — по средним длинам осей из ручных линий того же изображения. Даже ручная разметка может описывать нарушенную укладку, поэтому выполнение идеальных ограничений не обязательно для всех эталонов.</p>'+table(['Линии','Всего','Оси определены','Все ограничения'],[[k,v['images'],v['defined'],v['all_pass']] for k,v in contour_audits.items()])
    cis='<h2>95% интервалы неопределённости</h2><p>Групповой bootstrap: 200 повторов, пересэмплирование исследований, не отдельных аугментаций. При 20 тестовых исследованиях интервалы ориентировочные. Полнота и точность линий — средние по изображениям; ошибка пар учитывает только сопоставленные линии, поэтому её нельзя читать отдельно от недобора и полноты. Порог 5 мм зависит от принятых размеров пикселя.</p>'+table(['Вариант','Недобор: оценка [95% ДИ], %','Полнота: оценка [95% ДИ], %'],[[k,*[f"{v['test'][m]*100:.1f} [{v['ci95'][m][0]*100:.1f}; {v['ci95'][m][1]*100:.1f}]" for m in ['under_fraction','recall_5mm']]] for k,v in report['results'].items()])
    recommendation='<h2>Что использовать дальше</h2><p>Кандидат для следующего исследования — исходный вход и spatial-декодер с новыми штрафами. Перед включением в общий пайплайн нужно штрафовать геометрию всех пиков карты, а не только эталонных кандидатов, и проверить дополнительные линии вручную. Декодер пока ограничен семью линиями. Для подавления фона разумнее проверить мягкую маску или дополнительный канал маски, сохраняя исходное изображение, затем частично разморозить энкодер.</p><p>Новое смешивание крайних осей реализовано. Разрыв общей цепочки на проверенных случаях — 0 пикселей. На 30 одинаковых снимках угол определён для 29 случаев с новыми линиями против 25 с контрольными; ошибка относительно того же алгоритма на ручных линиях — 0,91° против 1,09°. Это сравнение линий при новом алгоритме осей. Само изменение весов сдвигает угол в среднем на 0,24° на новых линиях; независимого эталона осей нет, поэтому улучшение истинной точности этим тестом не доказано. 13 из 30 случаев имеют флаг проверки контуров или крайних фрагментов.</p>'
    page=ROOT/'dxa_project/team_demo/spine_brightness_study_20260929.html'
    html=page.read_text(encoding='utf-8')
    if '<section><h2>Выводы по завершённому тесту' not in html:
        html=html.replace('<h2>Метрики на одинаковом тесте</h2>',heading+'<h2>Метрики на одинаковом тесте</h2>')
        html=html.replace('<h2>Крайние оси и непрерывность</h2>',geometry+cis+'<h2>Крайние оси и непрерывность</h2>')
        html=html.replace('<h2>Одни и те же снимки: входы и предсказанные линии</h2>',recommendation+'<h2>Одни и те же снимки: входы и предсказанные линии</h2><p>В столбцах моделей показан фактический вход после изменения размера и фильтрации, восстановленный до размера исходника для наложения координат. В эталонном столбце — исходный снимок.</p>')
    html=html.replace('Перебор</th>','Число выше эталона</th>')
    audit_note='<p id="final-audit-note"><b>Ограничение новых штрафов:</b> на исходнике все приближённые ограничения проходят только 27/154 предсказаний; среди 120 случаев с дополнительными линиями — только 1. По полным контурным осям проходят 6/30 новых предсказаний против 5/30 контрольных и 19/30 ручных. Значит, уменьшение недобора пока не решило задачу корректного разделения позвонков. Нужен штраф по всем пикам и проверка лишних линий.</p><p>Ошибка линий в таблицах — среднее вертикальное отклонение в мм на трёх абсциссах (25%, 50%, 75% ширины), после взаимно однозначного оптимального сопоставления. Это не расстояние между полными контурами и не метрика угла сколиоза.</p>'
    if 'id="final-audit-note"' not in html:html=html.replace('<h2>Что использовать дальше</h2>',audit_note+'<h2>Что использовать дальше</h2>')
    page.write_text(html,encoding='utf-8')
    (DEST/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'geometry':{k:{a:b for a,b in v.items() if a!='details'} for k,v in audits.items()}},ensure_ascii=False))


if __name__=='__main__':run()
