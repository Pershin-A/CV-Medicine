"""Concrete contract of the local FastAPI implementation, including all routes."""
import json
from .final_assembly import ROOT,OUT,save


def build():
    from dxa_project.service.app import create_app
    app=create_app(OUT/'api_reference');schema=app.openapi();save(ROOT/'dxa_project/team_demo/final_model_20260929_assets/openapi.json',schema)
    descriptions={
      '/v1/health':'Состояние, активная версия, занятость внешним обучением.',
      '/v1/model/versions':'POST Register JSON: checkpoints, protocol, epochs, activate → версия модели; GET → список версий.',
      '/v1/model/versions/{version}/activate':'version в URL → активированная прошедшая проверку версия.',
      '/v1/model/predict':'Predict JSON + wait_seconds → результат single, состояние batch или 202 job_id.',
      '/v1/model/partial_fit':'PartialFit JSON → 202 job_id и количество примеров; дообучение пяти основных модулей.',
      '/v1/data/import':'query path на сервере → описание импортированного изображения и image_id.',
      '/v1/data/uploads':'multipart поле file, DICOM ≤128 MiB → описание изображения и image_id.',
      '/v1/data/images/{id}':'image_id → метаданные импортированного изображения.',
      '/v1/data/images/{id}/preview.png':'image_id → PNG исходного изображения.',
      '/v1/data/images/{id}/annotation':'Annotation JSON → новая версия разметки; expected_version проверяет конфликт.',
      '/v1/data/images/{id}/annotation/{version}':'image_id/version → метаданные и полная геометрия разметки.',
      '/v1/data/images/{id}/annotation/{version}/overlay.png':'image_id/version → PNG наложения ручной разметки.',
      '/v1/data/queue':'POST Enqueue JSON → 202 задание аугментации с добавлением в очередь; GET query status → список очереди.',
      '/v1/data/augmentation':'Enqueue JSON → 202 задание аугментации без добавления в очередь.',
      '/v1/data/queue/{id}':'queue_id → элемент помечается excluded; reserved нельзя удалить.',
      '/v1/jobs/{id}':'job_id → статус, прогресс, результат или ошибка.',
      '/v1/jobs/{id}/cancel':'job_id → отмена только queued; иначе 409.',
      '/v1/jobs/{id}/logs':'offset≥0, limit=1..1000 → events, next_offset, total.',
      '/v1/jobs/{id}/export':'format=csv/json/zip → файл завершённого задания; иначе 409.',
      '/v1/results/{id}':'result_id → сохранённый результат предсказания.',
      '/v1/results/{id}/files/{name}':'name: overlay.png, mask.png, geometry.json, result.json → соответствующий файл.'}
    lines=['## API: вход, выход и обращение','',
      'Сервис FastAPI принимает **однокадровые серые DICOM**, а не JPEG/PNG-фотографии. Передавать можно путь к DICOM или папке на сервере, идентификаторы импортированных файлов либо загрузить DICOM через multipart. Upload и импорт требуют StudyInstanceUID. При передаче папки рекурсивно выбираются файлы .dcm/.dicom. Слово «фотография» в логе — порядковый номер обрабатываемого DICOM.', '',
      'Адрес по умолчанию: `http://127.0.0.1:8765`. Интерактивная документация: `/docs`; полная машинная схема: `/openapi.json` (копия сохранена в assets отчёта).', '',
      '### Запуск и регистрация готовой версии','',
      '```powershell',r'$env:PYTHONPATH="$PWD\analysis_20260929\runtime;$PWD"',r'.venv\Scripts\python.exe -m dxa_project.service --device auto','```','',
      'При первом запуске зарегистрировать bundle методом **POST `/v1/model/versions`**. Обученные файлы копируются в отдельную версию сервиса. Следующий JSON с activate=true явно активирует её; в ходе исследования действующая версия не переключалась.', '',
      '```json',json.dumps({'checkpoints':str(OUT/'bundle').replace('\\','/'),'protocol':str(OUT/'protocol.json').replace('\\','/'),'activate':True},ensure_ascii=False,indent=2),'```','',
      '`epochs` — необязательный словарь бюджета будущего partial_fit: router=12, spine/hip/artifact/hip_points=24 по умолчанию. Дополнительные scoliosis, hip_mask и spine_crests при partial_fit сохраняются без дообучения. Активация прошедшего partial_fit-кандидата выполняется отдельным POST `/v1/model/versions/{version}/activate`.', '',
      '### Предсказание и получение результата','',
      'Тело POST `/v1/model/predict?wait_seconds=0`:','',
      '```json',json.dumps({'paths':['C:/path/on/server/scan.dcm'],'image_ids':[],'mode':'single','model_version':None,'request_id':'example-001'},indent=2),'```','',
      'Для папки или нескольких файлов mode=batch. paths и image_ids суммируются, повторные пути удаляются; допустимо 1–10 000 файлов. model_version=null использует активную версию. single требует ровно один файл. request_id необязателен: одинаковый ключ и содержимое возвращают прежнее задание, иной payload с тем же ключом отклоняется. Неизвестные поля JSON отклоняются.', '',
      'wait_seconds по умолчанию 10, допустимо 0–30. HTTP 200: в single при успехе сам JSON результата, в batch состояние завершённого задания; если single не дал успешного результата — состояние задания с errors. HTTP 202: `{ "job_id": "...", "status": "queued" }` (статус может быть running). Опрос GET `/v1/jobs/{job_id}` до completed/completed_with_errors/failed; идентификаторы успешных результатов — result.result_ids. Затем GET `/v1/results/{result_id}`.', '',
      'Пример Python-клиента для удалённого файла:', '',
      '```python','import requests, time','base = "http://127.0.0.1:8765"','with open("scan.dcm", "rb") as f:','    response = requests.post(base + "/v1/data/uploads", files={"file": ("scan.dcm", f)})','response.raise_for_status()','image_id = response.json()["image_id"]','response = requests.post(base + "/v1/model/predict", params={"wait_seconds": 0},','                         json={"image_ids": [image_id], "mode": "single"})','response.raise_for_status()','job_id = response.json()["job_id"]','while True:','    response = requests.get(base + f"/v1/jobs/{job_id}")','    response.raise_for_status()','    job = response.json()','    if job["status"] in ("completed", "completed_with_errors", "failed", "cancelled", "interrupted"):','        break','    time.sleep(1)','if job["status"] in ("completed", "completed_with_errors"):','    for result_id in job["result"]["result_ids"]:','        print(requests.get(base + f"/v1/results/{result_id}").json())','else:','    print(job)','```','',
      'Для локального пути обращаться к серверу можно без импорта. Путь относится к файловой системе сервера и должен попадать в разрешённые корни. Дополнительные корни задаются DXA_ALLOWED_ROOTS через `;`. Для удалённого клиента нужен upload или уже существующий image_id.', '',
      '### Поля результата','',
      '| Поле | Содержимое |','|---|---|',
      '| result_id, source, model_version | Идентификатор результата, исходник, точная версия весов. |',
      '| region, router_probabilities | SPINE / LEG_LEFT / LEG_RIGHT и словарь softmax для трёх областей. |',
      '| quality_flags | 0 — нет нарушения, 1 — есть, null — ответ не определён. Только таргеты выбранной области. |',
      '| quality_scores | Непрерывные оценки для ранжирования; не клинические вероятности. |',
      '| spine_axis_angle_deg, vertebral_axes | Угол общей оси с вертикалью и восстановленная геометрия осей для SPINE. |',
      '| scoliosis | sigmoid score, validation threshold, описание целевой метки. |',
      '| geometry | Разделители, подвздошные точки, рамки артефактов либо точки бедра и ROI; координаты исходного DICOM, (0,0) сверху слева. |',
      '| spacing_mm, spacing_basis | Масштаб (Y,X) и его источник; при отсутствии PixelSpacing — номинальные 1.05/0.6 мм. |',
      '| lesser_trochanter_area_px2 / _mm2 | Площадь маски малого вертела в пикселях / номинальных мм² для бедра. |',
      '| overlay_url, geometry_url, annotation_legend | Ссылки на артефакты и подписи ориентиров. |','',
      'В API-JSON массив пикселей малого вертела сокращён: вместо него geometry.hip.lesser_trochanter_mask_png указывает на mask.png. Полный geometry.json содержит пиксели. overlay.png показывает разметку поверх снимка; mask.png — бинарную маску. Ошибка отдельного файла отражается в result.errors задания и не останавливает остальные.', '',
      'GET `/v1/jobs/{job_id}/logs?offset=0&limit=100` возвращает события с номером, общим числом, файлом, временем и статусом started/completed/failed. Во время работы выводятся «Фотография 1/N: обработка», «готово» или «ошибка». Следующая страница — offset=next_offset. GET `/v1/jobs/{job_id}/export?format=zip` выдаёт артефакты всех результатов; csv — флаги и ошибки, json — результаты и ошибки.', '',
      '### Все методы','', '| Метод | URL | Вход → выход / назначение |','|---|---|---|']
    for path,methods in schema['paths'].items():
        for method in methods:
            if method in ('get','post','put','delete','patch'):lines.append(f'|{method.upper()}|`{path}`|{descriptions[path]}|')
    lines+=['','### Разметка, аугментация и дообучение','',
      'Annotation: обязательны region и полная geometry; reviewed — список проверенных модулей spine/artifact/hip/hip_points; targets — словарь флагов; spacing_mm — (Y,X); expected_version — текущая версия, по умолчанию 0. mask_result_id позволяет взять полную маску из результата вместо сокращённого JSON. Перечень проверенных модулей определяет, какие задачи можно дообучать.', '',
      'Enqueue: image_id, annotation_version≥1, необязательные request_id и config. config содержит positive_count (0–500, по умолчанию 5; качественные варианты, флаги 0), negative_count_by_target (квоты искусственных нарушений, флаги 1), negative_source_count (0–500, по умолчанию 5; варианты уже некачественного источника), seed=42 и max_attempts_per_sample (1–1000, по умолчанию 100). Целевые синтетические нарушения: spine_position/spine_axis для позвоночника, hip_position/hip_roi для бедра. Артефакты, ротация и сколиоз отдельной деформацией здесь не синтезируются. Для queue аугментации попадут в очередь обучения; augmentation только создаёт их. Невыполнимые квоты перечисляются в отчёте задания.', '',
      'PartialFit: base_model_version (иначе активная), queue_ids (иначе готовая очередь), replay_per_task=32 (1–10 000), learning_rate=0.00002 (>0, ≤0.01), request_id. Test и validation в очередь обучения не допускаются; кандидат проходит модульную и общую validation. Основные ошибки: 404 — объект не найден; 422 — неверные данные/путь/геометрия или конфликт expected_version разметки; 413 — upload больше лимита; 409 — неподходящий статус модели/очереди/задания; 500 — фатальный сбой предсказания при синхронном ожидании.', '']
    actual=[]
    for path in sorted((OUT/'api_smoke/results').glob('*/result.json')):
        actual.append(json.loads(path.read_text(encoding='utf-8')))
    if actual:
        lines+=['### Реальные ответы API из проверки итоговой сборки','',
                'Ниже реальные ответы для позвоночника и бедра, сокращённые до основных полей. Полная геометрия доступна по geometry_url, маска — по ссылке в geometry.hip. Идентификаторы относятся к изолированному smoke-сервису, а не к действующему серверу.', '']
        for spine in (True,False):
            result=next(r for r in actual if (r['region']=='SPINE')==spine)
            keys=('result_id','model_version','region','router_probabilities','quality_flags','quality_scores','spine_axis_angle_deg','scoliosis','spacing_mm','spacing_basis','lesser_trochanter_area_px2','lesser_trochanter_area_mm2','overlay_url','geometry_url')
            lines+=['```json',json.dumps({k:result[k] for k in keys if k in result},ensure_ascii=False,indent=2),'```','']
    return lines
