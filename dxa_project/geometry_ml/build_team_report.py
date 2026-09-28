"""Build a portable Russian team report with real held-out model examples."""
from pathlib import Path
from copy import deepcopy
import csv, json, html, shutil
import numpy as np
import torch
from PIL import Image
from .data import load_records, read_dicom_image
from .train import split_records
from .predict import predict_file, prepare_input, _load_model
from .evaluate import truth_flags
from dxa_project.augmentation.core import prepare_geometry
from dxa_project.augmentation.pilot_30 import _overlay
from dxa_project.augmentation.visualize_spine_axes import draw_axes

REGIONS={'SPINE':'Позвоночник','LEG_LEFT':'Левое бедро','LEG_RIGHT':'Правое бедро'}
TASKS={'spine_position':'Укладка позвоночника','spine_axis':'Наклон позвоночника',
       'spine_artifact':'Артефакты позвоночника','hip_position':'Позиционирование бедра',
       'hip_roi':'Отступы области интереса бедра','hip_rotation':'Ротация бедра'}
MODULES={'router':'Определение области тела (router)', 'lines':'Линии позвоночника (spine lines)',
         'crests':'Точки подвздошных костей (iliac crests)', 'artifact':'Посторонние объекты (artifact)',
         'points':'Опорные точки бедра (hip landmarks)', 'roi':'Область интереса бедра (hip ROI)',
         'mask':'Маска малого вертела (hip segmentation)'}
MARGINS={'top':'сверху','bottom':'снизу','lateral':'сбоку'}

def flag(v):return 'неопределённо' if v is None else 'нарушение' if v else 'норма'

def filtered(g,key):
    g=deepcopy(g)
    if key in ('lines','crests','artifact'):
        if key!='lines':g['spine']['disc_lines']=[]
        if key!='crests':g['spine']['iliac_crests']={k:None for k in g['spine']['iliac_crests']}
        if key!='artifact':g['spine']['foreign_objects']=[]
    else:
        if key!='roi':g['hip']['roi_box']=None
        if key!='points':g['hip']['landmarks']={k:None for k in g['hip']['landmarks']}
        if key!='mask':
            g['hip']['lesser_trochanter_pixels']=[]
            g['hip']['lesser_trochanter_traces']={'trochanter':[],'bone':[]}
    return g

def main():
    root=Path(__file__).resolve().parents[2]; project=root/'dxa_project'
    run=project/'outputs/geometry_ml_augmented_5epochs'; out=project/'team_demo'
    out.mkdir(exist_ok=True); (out/'images').mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    evaluation=read(run/'evaluation/report.json'); training=read(run/'report.json')
    synthetic=read(run/'synthetic_evaluation/report.json')
    # Only aggregate metrics go into the portable publication directory.
    (out/'metrics.json').write_text(json.dumps({'originals':evaluation,'synthetic':synthetic},ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(run/'learning_curves.png',out/'images/learning_curves.png')
    _,valid=split_records(load_records(root),0)
    saved={r['relative_path']:r for r in read(run/'evaluation/predictions.json')}
    with (project/'outputs/manifest.csv').open(encoding='utf-8-sig',newline='') as f: reference={r['relative_path']:r for r in csv.DictReader(f)}
    def unique(records):
        import hashlib
        seen=set(); result=[]
        for r in records:
            digest=hashlib.sha256(read_dicom_image(r.source_path).tobytes()).hexdigest()
            if digest not in seen:seen.add(digest);result.append(r)
        return result
    spine=unique([r for r in valid if r.region=='SPINE'])
    hips=unique([r for r in valid if r.region!='SPINE'])
    pick=lambda rows:[rows[int(i)] for i in np.linspace(0,len(rows)-1,5)]
    spine5=pick(spine); hip5=pick(hips)
    wrong=[r for r in valid if saved[r.relative_path]['predicted_region']!=r.region]
    router5=unique((wrong[:1]+[spine[0],hips[0],hips[-1],spine[-1]]))
    router5=unique(router5+valid)[:5]
    full5=unique(wrong[:1]+spine5[:2]+hip5[:3])[:5]
    selected={r.relative_path:r for r in spine5+hip5+router5+full5}
    aliases={path:f'Пример {i+1:02d}' for i,path in enumerate(selected)}
    (project/'outputs/team_report_source_map.json').write_text(json.dumps(aliases,ensure_ascii=False,indent=2),encoding='utf-8')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache={}
    def example(record):
        if record.relative_path in cache:return cache[record.relative_path]
        image=read_dicom_image(record.source_path); base=Image.fromarray((image*255).astype(np.uint8)).convert('RGB')
        truth=prepare_geometry(read(record.geometry_path),record.region)
        prediction=predict_file(record.source_path,run,force_region=record.region)
        actual=predict_file(record.source_path,run)
        with torch.no_grad():
            task='spine' if record.region=='SPINE' else 'hip'
            model,size=_load_model(task,run,device)
            tensor,meta=prepare_input(image,size,flipped=record.region=='LEG_RIGHT')
            maps=torch.sigmoid(model(tensor[None].to(device))['spatial'][0]).cpu().numpy()
        result=(base,truth,prediction,actual,maps,meta);cache[record.relative_path]=result;return result
    def save(img,name):
        img.thumbnail((600,650)); img.save(out/'images'/f'{name}.png');return f'images/{name}.png'
    def heat(base,maps,meta,indices):
        probability=maps[indices].max(axis=0)
        dw=round(meta['width']*meta['scale']);dh=round(meta['height']*meta['scale'])
        small=Image.fromarray((probability*255).astype(np.uint8)).crop((meta['left'],meta['top'],meta['left']+dw,meta['top']+dh))
        values=np.asarray(small.resize(base.size,Image.Resampling.BILINEAR))/255
        if meta['flipped']:values=np.fliplr(values)
        original=np.asarray(base).astype(float)
        color=np.zeros_like(original);color[:,:,0]=255; color[:,:,1]=80
        alpha=values[:,:,None]*.75
        return Image.fromarray((original*(1-alpha)+color*alpha).astype(np.uint8))
    def picture(src,title):return f'<figure><img src="{src}" loading="lazy"><figcaption>{html.escape(title)}</figcaption></figure>'
    parts=['<h1>DXA: как работает модель и что получилось</h1>',
           '<p>Тестовый эксперимент: 15 000 аугментированных изображений, четыре нейросети, по пять эпох обучения на GPU. Проверка качества — на 100 исходных снимках из 19 исследований вне обучения. Это первая рабочая версия; некоторые проверки пока ошибаются часто.</p>',
           '<nav><a href="#modules">Модули и примеры</a> · <a href="#full">Пять полных проходов</a> · <a href="#metrics">Метрики</a> · <a href="#problems">Что исправлять</a></nav>',
           '<h2>Что делает система</h2><ol><li>Определение области тела (router): позвоночник, левое или правое бедро.</li><li>Модель позвоночника (spine): линии между позвонками и верхние точки подвздошных костей. По ним отдельный геометрический алгоритм строит оси и проверяет укладку.</li><li>Детектор посторонних объектов (artifact): ищет рамки артефактов на позвоночнике.</li><li>Общая модель бедра (hip): три опорные точки, область интереса (ROI) и пиксельная маска малого вертела. Правое бедро зеркально приводится к левому внутри модели.</li><li>Правила превращают координаты, расстояния и площадь в ответы: норма (0), нарушение (1), иногда неопределённо.</li></ol>',
           '<p><strong>Важно:</strong> отсутствие найденной точки может означать как отсутствие структуры на снимке, так и ошибку модели. Сейчас это одна из главных причин ложных тревог.</p>',
           '<h2 id="modules">Примеры отдельных модулей</h2><p>В каждом модуле по пять реальных снимков из отложенной части. Для геометрических модулей область тела задана правильно вручную, чтобы отдельно увидеть качество самого модуля. Полные проходы ниже используют реальный выбор маршрутизатора. Примеры взяты по фиксированному правилу из разных частей списка; это иллюстрации, а не дополнительная тестовая выборка.</p>']
    for key,title in MODULES.items():
        records=router5 if key=='router' else spine5 if key in ('lines','crests','artifact') else hip5
        parts.append(f'<h3>{title}</h3>')
        for n,r in enumerate(records,1):
            base,truth,pred,actual,maps,meta=example(r); stem=f'{key}_{n}'
            parts.append(f'<article><h4>{aliases[r.relative_path]}</h4><div class="grid">')
            parts.append(picture(save(base.copy(),stem+'_original'),'Исходное изображение'))
            if key=='router':
                parts.append('</div><p>Правильная область: <b>'+REGIONS[r.region]+'</b>. Выбор модели: <b>'+REGIONS[actual['region']]+'</b>. '+('Верно.' if actual['region']==r.region else 'Ошибка выбора области.')+'</p>')
                parts.append('<p>Оценки модели: '+ '; '.join(f'{REGIONS[k]}: {v:.1%}' for k,v in actual['router_probabilities'].items())+'. Это оценки уверенности, а не гарантия правильности.</p>')
            else:
                parts.append(picture(save(_overlay(base,filtered(truth,key),r.region),stem+'_truth'),'Ручная / производная эталонная разметка'))
                parts.append(picture(save(_overlay(base,filtered(pred['geometry'],key),r.region),stem+'_prediction'),'Предсказание после порогов'))
                parts.append('</div>')
                if key in ('lines','crests','points','mask'):
                    indices={'lines':[0],'crests':[1,2],'points':[0,1,2],'mask':[3]}[key]
                    parts.append('<details><summary>Показать карту отклика модели до порогов</summary>'+picture(save(heat(base,maps,meta,indices),stem+'_heat'),'Чем ярче красный, тем больше отклик нейросети. Это не окончательная разметка.')+'</details>')
                if key=='lines': caption=f"Число линий: эталон {len(truth['spine']['disc_lines'])}, модель {len(pred['geometry']['spine']['disc_lines'])}. Низкий Dice линий означает, что их положение и толщина ещё плохо совпадают с эталоном."
                elif key=='crests':caption='Найдено верхних точек: '+str(sum(v is not None for v in pred['geometry']['spine']['iliac_crests'].values()))+' из 2. Пропуски влияют на укладку.'
                elif key=='artifact':caption=f"Рамок: эталон {len(truth['spine']['foreign_objects'])}, модель {len(pred['geometry']['spine']['foreign_objects'])}. Метка наличия артефакта и точность всех рамок — разные проверки."
                elif key=='points':caption='Найдено опорных точек: '+str(sum(v is not None for v in pred['geometry']['hip']['landmarks'].values()))+' из 3. 1 — большой вертел, 2 — шейка, 3 — седалищная кость.'
                elif key=='roi':caption='Отступы предсказанной области интереса (ROI): '+', '.join(f'{MARGINS[k]}: {v/10:.2f} см' for k,v in pred['roi_margins_mm'].items())+'. Порог: сверху/снизу 3 см, сбоку 2 см. Масштаб пикселя номинальный.'
                else:caption=f"Площадь предсказанной маски: {pred['lesser_trochanter_area_px2']} пикселей; красная заливка — маска, красный пунктир в эталоне — контур при нулевой производной площади."
                parts.append('<p>'+html.escape(caption)+'</p>')
            parts.append('</article>')
    parts+=['<h2 id="full">Пять примеров полного прохождения</h2>', '<p>Все промежуточные изображения построены по выходам уже обученных моделей. Ручная разметка показана отдельно для сравнения. Ошибки не скрыты.</p>']
    full_summary=[]
    for n,r in enumerate(full5,1):
        base,truth,forced,actual,maps,meta=example(r);stem=f'full_{n}'
        parts.append(f'<article><h3>Полный проход {n} — {aliases[r.relative_path]}</h3><p>Истинная область: {REGIONS[r.region]}; модель выбрала: {REGIONS[actual["region"]]}.</p><div class="grid">')
        parts.append(picture(save(base.copy(),stem+'_original'),'1. Исходный снимок'))
        parts.append(picture(save(_overlay(base,truth,r.region),stem+'_truth'),'Эталонная визуальная разметка'))
        if actual['region']=='SPINE':
            for key,caption in [('lines','2. Найденные разделительные линии'),('crests','3. Найденные верхние точки костей'),('artifact','4. Найденные рамки артефактов')]:
                parts.append(picture(save(_overlay(base,filtered(actual['geometry'],key),'SPINE'),stem+'_'+key),caption))
            parts.append(picture(save(draw_axes(_overlay(base,actual['geometry'],'SPINE'),actual['vertebral_axes'],detailed=True),stem+'_axes'),'5. Контуры и оси, построенные по предсказанным линиям'))
        else:
            for key,caption in [('points','2. Найденные три опорные точки'),('roi','3. Предсказанная область интереса'),('mask','4. Предсказанный малый вертел')]:
                parts.append(picture(save(_overlay(base,filtered(actual['geometry'],key),actual['region']),stem+'_'+key),caption))
            parts.append(picture(save(_overlay(base,actual['geometry'],actual['region']),stem+'_all'),'5. Вся предсказанная разметка'))
        truth_binary=truth_flags(reference[r.relative_path],r.region,truth)
        parts.append('</div><h4>Итоговые ответы</h4><table><tr><th>Проверка</th><th>Эталон</th><th>Модель</th><th>Сравнение</th></tr>')
        for key,value in truth_binary.items():
            guess=actual['quality_flags'].get(key) if actual['region']==r.region else None
            verdict='Нельзя сравнить: другая область / нет ответа' if guess is None or value is None else 'Верно' if guess==value else 'Ошибка'
            parts.append(f'<tr><td>{TASKS[key]}</td><td>{flag(value)}</td><td>{flag(guess)}</td><td>{verdict}</td></tr>')
        parts.append('</table>')
        if 'spine_axis_angle_deg' in actual:parts.append('<p>Наклон оси к вертикали: '+str(actual['spine_axis_angle_deg'])+'°. Нарушение при |угол| > 5°.</p>')
        if 'roi_margins_mm' in actual:parts.append('<p>Отступы области интереса (ROI): '+', '.join(f'{MARGINS[k]}: {v/10:.2f} см' for k,v in actual['roi_margins_mm'].items())+'.</p>')
        parts.append('</article>')
        full_summary.append({'example':n,'truth_region':r.region,'predicted_region':actual['region'],'truth':truth_binary,'prediction':actual['quality_flags']})
    (out/'full_examples.json').write_text(json.dumps(full_summary,ensure_ascii=False,indent=2),encoding='utf-8')
    parts+=['<h2 id="metrics">Как понимать метрики</h2>', '<p>Во всех проверках положительный класс — <b>нарушение (1)</b>. Слово «позитив» в плане аугментации, напротив, означало качественный снимок. Эти два значения нельзя смешивать.</p>',
           '<table><tr><th>Метрика</th><th>Простыми словами</th><th>Как читать</th></tr>'+''.join('<tr><td>'+a+'</td><td>'+b+'</td><td>'+c+'</td></tr>' for a,b,c in [
           ('Чувствительность (sensitivity)','Какую долю реальных нарушений нашли.','1 — нашли все; 0 — пропустили все. Низкая чувствительность означает пропуски проблем.'),
           ('Специфичность (specificity)','Какую долю качественных снимков признали качественными.','1 — нет ложных тревог; около 0 — почти каждый хороший снимок ошибочно забракован.'),
           ('F1-мера (F1)','Баланс между найденными нарушениями и ложными тревогами.','От 0 до 1, больше лучше. Даже высокая чувствительность не спасает F1 при множестве ложных тревог.'),
           ('ROC AUC','Насколько хорошо оценка модели ставит проблемные снимки выше хороших.','1 — идеальное ранжирование; 0,5 — случайный уровень. Высокий AUC не гарантирует хороший выбранный порог.'),
           ('PR AUC / AP','Качество ранжирования при редких нарушениях.','Больше лучше; случайный ориентир зависит от доли нарушений. AP — вариант усреднения precision-recall, используемый здесь.'),
           ('Сбалансированная точность (balanced accuracy)','Среднее чувствительности и специфичности.','1 — идеально; 0,5 — случайный ориентир даже при дисбалансе классов.'),
           ('Dice и IoU','Насколько совпали предсказанные пиксели или прямоугольники с эталоном.','1 — полное совпадение, 0 — нет пересечения. Тонкие линии сильно штрафуются даже за небольшой сдвиг.'),
           ('Ошибка точек (keypoint distance)','Расстояние от найденной точки до правильной.','Меньше лучше. Миллиметры здесь рассчитаны по номинальному размеру пикселя.'),
           ('Покрытие (coverage)','Для какой доли снимков есть определённый ответ по проверке.','Низкое покрытие означает много ответов «неопределённо». Высокое качество на малой покрытой части может вводить в заблуждение.'),
           ('95% доверительный интервал (CI)','Насколько неопределённа оценка на этом небольшом наборе.','Широкий интервал означает мало информации. Это интервал метрики, а не вероятность диагноза конкретного снимка.')])+'</table>',
           '<h3>Результаты на исходных снимках</h3><p>100 файлов из 19 отложенных исследований. Метрики задач считаются только для определённых ответов; покрытие показывает исключённую долю. Исходные изображения содержат точные дубли внутри частей: 499 файлов, 263 уникальных массива; между train и validation точных дублей нет. Поэтому число файлов не равно числу независимых наблюдений. Интервалы получены повторной выборкой исследований, а не отдельных аугментаций.</p>']
    def metric_table(report):
        rows=['<table><tr><th>Область</th><th>Проверка</th><th>N ответов</th><th>Покрытие</th><th>Чувствительность</th><th>Специфичность</th><th>F1 [95% ДИ]</th><th>ROC AUC [95% ДИ]</th></tr>']
        for m in report['metrics']:
            def value(k,ci=False):
                v=(m.get('metrics') or {}).get(k)
                if v is None:return '—'
                s=f'{v:.3f}'; interval=m.get('ci95',{}).get(k)
                return s+(f' [{interval[0]:.3f}; {interval[1]:.3f}]' if ci and interval else '')
            rows.append(f'<tr><td>{REGIONS[m["region"]]}</td><td>{TASKS[m["task"]]}</td><td>{m["n"]}</td><td>{m["coverage"]:.1%}</td><td>{value("sensitivity")}</td><td>{value("specificity")}</td><td>{value("f1",True)}</td><td>{value("roc_auc",True)}</td></tr>')
        return ''.join(rows)+'</table>'
    parts.append(metric_table(evaluation))
    parts+=['<p>«—» означает, что метрика не определена, например в выборке нет реальных нарушений. Это не нулевая ошибка. Отдельное поле позиционирования бедра отсутствует у организаторов: его эталон вычислен по ручным точкам. Остальные бинарные эталоны — исправленные метки организаторов.</p>',
            f'<p>Определение области (router): правильно 97 из 100; macro-F1 = {evaluation["router_macro_f1"]:.3f}. Обработка файлов успешна для 100 из 100. Среднее время: {evaluation["seconds_per_file_mean"]:.3f} с, p95: {evaluation["seconds_per_file_p95"]:.3f} с на прогретых моделях.</p>',
            '<h3>Геометрия: совпала ли сама разметка?</h3><table><tr><th>Модуль</th><th>Результат</th><th>Что это значит</th></tr>',
            '<tr><td>Позвоночник (spine)</td><td>Dice линий 0,172; ошибка точек 12,7 мм</td><td>Линии совпадают плохо. Даже если некоторые итоговые ответы верны, геометрия ещё ненадёжна.</td></tr>',
            '<tr><td>Бедро (hip)</td><td>Dice вертела 0,775; IoU ROI 0,914; ошибка точек 26,9 мм</td><td>Маска и прямоугольник лучше точек. Хорошая ROI не доказывает, что все анатомические структуры найдены.</td></tr>',
            '<tr><td>Артефакты (artifact)</td><td>Recall рамок при IoU≥0,5: 0,379; precision: 0,862</td><td>Найденные рамки чаще правильные, но многие эталонные объекты пропущены. Это совместимо с высоким F1 наличия хотя бы одного артефакта.</td></tr></table>',
            '<h3>Отдельно: устойчивость к аугментациям</h3><p>135 производных снимков только от источников вне обучения. Не смешиваем эти результаты с исходными данными: преобразования и их метки искусственные, и не отражают всю сложность реальных случаев.</p>',metric_table(synthetic),
            '<h2 id="problems">Что хорошо и что нужно исправить</h2><ul><li>Область тела определяется хорошо, но три ошибки маршрутизации могут направить снимок в неправильную ветку.</li><li>ROI на небольшой исходной выборке имеет F1=1. Смотрите доверительный интервал: это не доказательство идеального качества.</li><li>Позиционирование бедра почти всегда даёт ложную тревогу: специфичность всего 0,031 / 0,100. Нужно улучшить точки и отдельно откалибровать пороги на данных для настройки.</li><li>Реальное нарушение оси позвоночника не выявлено: F1=0, AUC=0,5. Хорошая синтетическая проверка поворотов этого не отменяет.</li><li>Артефакты хорошо распознаются по наличию на оригиналах, но хуже локализуются и плохо переносят преобразования.</li><li>Ротация правого бедра слабая. Следует проверить маску малого вертела и соответствие площади реальной ротации.</li><li>Размер пикселя номинальный; у заранее увеличенных сканов реальный масштаб может отличаться. Пороговые расстояния в сантиметрах требуют калибровки.</li><li>После аугментации нужна проверка верхних точек подвздошных костей. Текущий набор сохранён для честного сравнения после доразметки.</li></ul>',
            '<p>Для любого нарушения в целом: F1=0,534 и ROC AUC=0,886 при покрытии 90%. Специфичность лишь 0,088 — система слишком часто бракует хорошие снимки. Пока это эксперимент для разработки, а не готовый контроль качества.</p>',
            '<h2>Обучение и следующий вариант</h2><p>Замороженный ResNet18 для определения области; ResNet18 с U-Net декодерами для геометрии; Faster R-CNN MobileNetV3 FPN для артефактов. Размер входа 256×256, пять эпох. Следующий вариант кода: ResNet50 с теми же выходами и Faster R-CNN ResNet50 FPN v2. Он проверен на исполняемость, но ещё не обучен.</p>',
            picture('images/learning_curves.png','Кривые обучения: router — область тела; spine — позвоночник; hip — бедро; artifact — посторонние объекты. Train — обучение, validation — отложенная проверка. Для детектора показана только train loss.'),
            '<p><a href="metrics.json">Полные агрегированные метрики, PR AUC, balanced accuracy и интервалы</a> · <a href="full_examples.json">Итоговые ответы пяти полных примеров</a></p>']
    style='body{font:16px system-ui;line-height:1.55;background:#111a23;color:#ecf3fa;margin:30px auto;max-width:1450px;padding:20px}a{color:#96ceff}h2{margin-top:45px}article{background:#1b2938;padding:18px;margin:18px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}figure{margin:0}img{width:100%;max-height:600px;object-fit:contain;background:#071018}figcaption{font-size:14px;margin:8px 0}table{border-collapse:collapse;display:block;overflow:auto;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #425569;vertical-align:top}nav{position:sticky;top:0;background:#111a23;padding:12px;z-index:2}@media(max-width:800px){.grid{grid-template-columns:1fr}}'
    (out/'index.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DXA — отчёт для команды</title><style>'+style+'</style>'+''.join(parts)+'</html>',encoding='utf-8')
    print(json.dumps({'team_report':str(out/'index.html'),'module_examples':35,'full_examples':len(full5),'unique_sources':len(selected)},ensure_ascii=False))

if __name__=='__main__':main()
