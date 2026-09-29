# Итоговый API DXA v2

В текущем сеансе сервис уже запущен на `http://127.0.0.1:8765`; повторно запускать команду, пока он работает, не нужно. При обычном запуске из терминала остановка — Ctrl+C, вместе с worker-процессами.

Сервис использует существующий пакет `dxa_project/outputs/final_20260929/bundle`. На запуске обучение не происходит. Текущая версия принимает однокадровые серые DICOM; JPEG/PNG как вход не поддерживаются. PNG используются для наложения разметки на выходе. Качество самих моделей и ограничения описаны в `FINAL_MODEL_20260929.md` и `ROTATION_REVIEW_20260929.md`.

## Запуск в текущем проекте

Запускать из корня `Хакатон`:

```powershell
$env:PYTHONPATH="$PWD\analysis_20260929\runtime;$PWD"
.venv\Scripts\python.exe -X utf8 -m dxa_project.service --device cuda --data-dir dxa_project/outputs/final_api_20260929
```

CPU: заменить `cuda` на `cpu`; автоматический выбор: `auto`. По умолчанию `127.0.0.1:8765`, документация `http://127.0.0.1:8765/docs`, схема `http://127.0.0.1:8765/openapi.json`. Сервер запускает два worker-процесса; GPU-worker последовательно выполняет predict и partial_fit. Не добавлять uvicorn workers и второй GPU-worker к тому же хранилищу. `--no-workers` нужен только при отдельно запущенных workers.

При пустом хранилище выбранные веса автоматически копируются и регистрируются в отдельной версии. При существующей активной версии она сохраняется. Другой пакет можно указать `--bootstrap-bundle <путь>`; рядом с bundle должен лежать `protocol.json`. Ручная регистрация — POST `/v1/model/versions` с `checkpoints`, `protocol`, `activate` и необязательным `epochs`. Без epochs используются сохранённые бюджеты `training_config.json`.

Разрешённые локальные пути: проект, хранилище сервиса и `DXA_ALLOWED_ROOTS` (несколько корней через `;`). Для удалённого клиента использовать upload, а не передавать путь с его компьютера. Сервис запускается как локальный вычислительный компонент; внешний backend обращается к нему через свой сервер.

На другой машине создать Python 3.13 environment, установить совместимые torch/torchvision для CPU/CUDA, затем зависимости из `requirements-geometry.txt` и `service/requirements.txt`. Здесь проверены torch 2.12.1+cu130 и torchvision 0.27.1+cu130. Особенность текущей машины: pydicom берётся из `analysis_20260929/runtime`, поэтому здесь этот каталог включён в PYTHONPATH; в чистом окружении установленный pydicom этого не требует. Для сервисных логов используется UTF-8. Worker-логи: `<data-dir>/logs/worker_gpu.log` и `worker_cpu.log`.

## predict

Самый простой запрос удалённого клиента:

```python
import requests, time
base = 'http://127.0.0.1:8765'
with open('scan.dcm', 'rb') as f:
    r = requests.post(base+'/v1/model/predict/upload',
                      files=[('files', ('scan.dcm', f, 'application/dicom'))])
r.raise_for_status()
job_id = r.json()['job_id']
while True:
    job = requests.get(base+f'/v1/jobs/{job_id}').json()
    if job['status'] not in ('queued', 'running'): break
    time.sleep(0.5)
print(job['result']['table'])
zip_response = requests.get(base+job['result']['download_url'])
zip_response.raise_for_status()
open('predictions.zip','wb').write(zip_response.content)
```

Поле multipart называется **files**, допускает несколько DICOM. Один файл — не более 128 MiB. Upload требует StudyInstanceUID. Для повторяемых/idempotent-запросов сначала загрузить DICOM в POST `/v1/data/uploads` (поле **file**), получить image_id, затем POST `/v1/model/predict`:

```json
{
  "image_ids": ["<image_id>"],
  "paths": [],
  "mode": "single",
  "model_version": null,
  "request_id": "predict-001"
}
```

Локальный набор: `paths: ["C:/server/path/folder"]`, `mode: "batch"`; папка обходится рекурсивно по .dcm/.dicom. Предел — 10 000 файлов. model_version=null означает активную версию. `/predict` — короткий алиас JSON-метода. `wait_seconds=0..30`, по умолчанию 10; upload по умолчанию 0. При незавершённой работе возвращается 202 и job_id. При завершении single возвращает подробное предсказание, job_id, table и ссылки на выгрузку; batch возвращает задание с result.table. Одинаковый request_id с другим payload отклоняется.

### Таблица

CSV, JSON и result.table содержат ровно эти колонки:

| Колонка | Тип и значение |
|---|---|
| path_to_study | string, фактический путь входного файла на сервере |
| study_uid | string, исходный StudyInstanceUID |
| image_uid | string, исходный SOPInstanceUID |
| anatomical_region | SPINE / LEG_LEFT / LEG_RIGHT; UNKNOWN при ошибке |
| quality_class | integer: 0 — все определённые флаги нормальные; 1 — нарушение или непригодный/неопределённый результат |
| violation_type | string, коды нарушений через `;`, пусто при норме |
| processing_status | Success / Failure |
| time_of_processing | float, секунды; при batch forward-время распределено поровну по файлам + индивидуальное сохранение |

При технической ошибке либо неизвестных флагах без подтверждённого нарушения: Failure, quality_class=1, violation_type=prediction_unavailable или undetermined:... . Это отказ принять снимок как качественный, а не диагноз нарушения. Если хотя бы один определённый флаг равен 1, итоговое наличие нарушения известно даже при других null-флагах. Подробные quality_flags сохраняют значения 0/1/null. Ошибка одного файла не останавливает остальные.

Коды: spine_position, spine_axis, spine_artifact, spine_scoliosis; hip_position, hip_roi, hip_rotation. Координаты геометрии — пиксели исходного DICOM, (0,0) сверху слева. Score не является клинической вероятностью.

### Две папки и метаданные

GET `/v1/jobs/{job_id}/export?format=zip`:

```text
table.csv
table.json
manifest.json
originals/00001_<name>.dcm
annotated/00001_<name>.png
```

`originals`: исходные DICOM с сохранёнными PixelData и UID; добавлены предсказания и полная геометрия в private block группы 0011, creator **DXA_MODEL_V1**. Offset 01 — JSON `{schema, table, prediction}`, offset 02 — полный JSON геометрии в формате обучения. Номер xx блока определяется по private creator, его нельзя жёстко считать 10. Повреждённый файл копируется с отдельным `.failure.json`; PNG для него отсутствует. `annotated`: изображение исходного размера с линиями, ориентирами, ROI, маской, артефактами и восстановленными осями.

```python
import pydicom, json
ds = pydicom.dcmread('originals/00001_image.dcm')
b = ds.private_block(0x0011, 'DXA_MODEL_V1', create=False)
prediction = json.loads(ds[b.get_tag(0x01)].value)
geometry = json.loads(ds[b.get_tag(0x02)].value)
```

`format=csv` — только таблица; `format=json` — JSON-массив строк таблицы. Подробное предсказание: GET `/v1/results/{result_id}`; полный geometry.json — `/v1/results/{result_id}/files/geometry.json`. GET `/v1/jobs/{job_id}/artifacts` перечисляет артефакты; GET `/v1/jobs/{job_id}/files/{path}` отдаёт файл внутри output.

Логи: GET `/v1/jobs/{job_id}/logs?offset=0&limit=100`; events, next_offset, total. Worker печатает «Фотография 1/N: обработка», затем «готово»/«ошибка». Прогресс и результаты — GET `/v1/jobs/{job_id}`.

## partial_fit

POST `/v1/model/partial_fit` (алиас `/partial_fit`) получает **два непустых набора** human и model. Метки не пересчитываются моделью для человеческого набора. Копии данных фиксируются до обучения, одинаковые пиксели внутри/между наборами отклоняются. Известные validation/test-исследования и пиксельные дубликаты этих снимков запрещены.

```json
{
  "base_model_version": null,
  "human": {
    "images": [{
      "path": "C:/server/human/scan.dcm",
      "geometry_path": "C:/server/human/geometry.json",
      "region": "LEG_LEFT",
      "reviewed": ["hip", "hip_points", "hip_mask"],
      "targets": {"hip_position": 0, "hip_roi": 0, "hip_rotation": 0}
    }]
  },
  "model": {
    "images": [{"path": "C:/server/model/scan_with_metadata.dcm"}]
  },
  "augmentation": {"n_pp": 5, "n_pn": 5, "n_nn": 5, "seed": 42},
  "learning_rate": 0.00002,
  "request_id": "fit-001"
}
```

Вместо path можно передать image_id + annotation_version ранее загруженного и размеченного DICOM. Для разметки PUT `/v1/data/images/{image_id}/annotation`: region, geometry, reviewed, targets, expected_version; для маски из API-ответа дополнительно mask_result_id. Можно дать geometry прямо в JSON вместо geometry_path. Формат geometry — тот же labeler JSON, полный пример `examples/geometry.json`.

manifest_path — необязательный путь к JSON-массиву таких же TrainingSample; добавляется к images. Paths в манифесте должны быть абсолютными серверными путями. Нельзя одновременно указать path и image_id. Для model допускается читать разметку непосредственно из экспортированного DXA_MODEL_V1 DICOM; человеческий DICOM с DXA_MANUAL_LABELER имеет приоритет перед оставшейся модельной разметкой. Явно переданная геометрия имеет приоритет перед обоими блоками. Для SPINE targets включают четыре флага, reviewed: spine, artifact, spine_crests, scoliosis; для ног — три флага и hip, hip_mask, hip_points. Неизвестная метка сколиоза не используется для обучения классификатора.

Последовательность:

1. human_original — дообучение на исходных человеческих примерах.
2. model_original — дообучение получившихся весов на исходных модельных примерах.
3. Каждый набор аугментируется отдельно.
4. human_augmented — дообучение на человеческих аугментациях.
5. model_augmented — дообучение на модельных аугментациях.
6. Проверка кандидатного пайплайна на фиксированной inner validation, сохранение новой версии.

Для каждого этапа используются **фактически выполненные эпохи первоначального обучения** выбранной семьи: router=8, spine=10, spine_crests=16, scoliosis=7, hip=11, hip_mask=11, hip_points=10, artifact=7. Это не номер лучшего checkpoint и не максимальный бюджет до early stopping. Этапы partial_fit выполняют полный сохранённый бюджет без early stopping; внутри этапа сохраняется лучший validation-checkpoint. При отсутствии примеров для модуля записывается skipped_no_labeled_examples. Если квоты аугментации нулевые, соответствующий этап пропускается явно. Логи содержат набор, модуль, эпоха i/N и validation score. GET `/v1/jobs/{job_id}/logs` читает training.jsonl.

Новая версия не активируется автоматически. После completed проверить result.model.status: validated или candidate. Активировать прошедшую проверку — POST `/v1/model/versions/{version}/activate`; candidate получает 409. Исходный bundle и текущая активная версия сохраняются. HTTP 202 подтверждает постановку задания; статусы queued/running/completed/failed/interrupted доступны через GET jobs. Interrupted не возобновляется автоматически: проверить кандидата и создать новый request_id.

### Аугментация и временная папка

Позитив означает **качественный снимок**, негатив — нарушение. n_pp создаёт качественные варианты качественного источника; n_pn — варианты нарушения каждого поддержанного геометрического таргета из качественного источника; n_nn — варианты исходного снимка с нарушением. Из негатива позитивы не создаются. Для SPINE поддержаны spine_position/spine_axis, для ног hip_position/hip_roi. Ротация, сколиоз и существующие артефакты берутся из реальных негативов и наследуются; искусственная новая анатомическая ротация/сколиоз не синтезируется. При потере части сколиотической геометрии метка не используется для обучения сколиоза. Квоты проверяются; при недоборе задание завершается failed с сохранённым отчётом, вместо молчаливого обучения на меньшем наборе.

```text
<data-dir>/jobs/<job_id>/output/
  originals/human/<id>/image.dcm + geometry.json + annotation.json
  originals/model/<id>/image.dcm + geometry.json + annotation.json
  annotated/human/<id>/overlay.png
  annotated/model/<id>/overlay.png
  temporary/augmented/human/<source_id>/<aug_id>/image.dcm + geometry.json + metadata.json
  temporary/augmented/model/<source_id>/<aug_id>/image.dcm + geometry.json + metadata.json
```

Временная папка находится рядом с originals/annotated. Наборы не смешиваются. Артефакты доступны через jobs/artifacts и jobs/files. Автоматического удаления после обучения нет; удалять временные папки можно после завершения задания и проверки отчёта.

Старый queue-based partial_fit оставлен для совместимости, но описанный новый контракт использует human/model, а не queue_ids. Не смешивать эти режимы в одном запросе.

## Что передать backend-разработчику

1. `backend_handoff.zip`: код сервиса и его Python-зависимостей из проекта, OpenAPI, этот гайд, обзор ротации, примеры geometry/predict/partial_fit и Python-клиент. Это архив исходников без весов и медицинских данных.
2. Отдельно **весь** каталог `outputs/final_20260929/bundle` (8 .pt, calibration, landmark bounds, training_config) и соседний `protocol.json`. Файлы весов не менялись; SHA256 исходных весов — в model_manifest.json.
3. Для partial_fit дополнительно исходные `Исследования`, `Размеченные` с labels.csv/geometry и `dxa_project/outputs/manifest.csv`: эти данные нужны для fixed validation, меток и проверки утечки. Старые 15 000 аугментаций передавать не требуется. Ресурсы перечислены в `backend_delivery_manifest.json`.
4. Порядок интеграции: загрузка DICOM → predict → опрос jobs/logs → показать table и annotated PNG → скачать ZIP; редактор разметки → human/model → partial_fit → показать epochs → вручную активировать validated-версию. Учесть Failure/unknown, а не отображать их как норму.

В текущем проекте проведён настоящий HTTP/GPU predict на трёх областях; проверены DICOM пиксели/UID/private JSON, CSV/ZIP и 6 событий логов. Полная регрессия: 106 тестов. Четыре этапа partial_fit, бюджеты эпох, работа optimizer и разделение папок проверены на изолированных тестовых данных; реальное дообучение пользовательских весов в этом этапе не запускалось по вашему указанию.
