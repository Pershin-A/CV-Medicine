# DXA DICOM Labeler v3

Версия исправляет две критичные проблемы предыдущего прототипа:

1. список DICOM больше не кэшируется;
2. Docker-путь к данным задаётся явно через `.env`.

Кроме того:
- `labels.csv` считается основным реестром разметки;
- перед перезаписью создаётся `labels.csv.bak`;
- все сохранения дополнительно попадают в `labels_history.csv`;
- DICOM и CSV записываются атомарно;
- widget-state привязан к `relative_path`, а не к номеру файла;
- приложение проверяет ожидаемое число DICOM и fingerprint набора.

## Рекомендуемая структура

```text
dxa_labeler/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── Исследования/
│   └── ... 499 *.dcm ...
└── Размеченные/
    ├── labels.csv
    ├── labels.csv.bak
    ├── labels_history.csv
    └── ...
```

## ВАЖНО перед обновлением

Если часть данных уже размечена, НЕ удаляйте папку `Размеченные`.

Сделайте её резервную копию, например:

```powershell
Copy-Item ".\Размеченные" ".\Размеченные_backup_2026-09-17" -Recurse
```

После этого можно заменить только:
- `app.py`
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`

и добавить `.env`.

## .env

Скопируйте `.env.example` в `.env`.

По умолчанию:

```env
DXA_DATA_DIR=./Исследования
DXA_OUTPUT_DIR=./Размеченные
EXPECTED_DICOM_COUNT=499
EXPECTED_DATASET_FINGERPRINT=
STRICT_DATASET_CHECK=1
```

Если ваши папки лежат в другом месте, укажите абсолютные пути:

```env
DXA_DATA_DIR=C:/DXA_labeling/dxa_labeler/Исследования
DXA_OUTPUT_DIR=C:/DXA_labeling/dxa_labeler/Размеченные
```

## Перезапуск после обновления

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d --force-recreate
```

Открыть:

```text
http://localhost:8501
```

## Проверка набора

В боковой панели приложение показывает:

- количество DICOM, которое реально видит контейнер;
- fingerprint набора;
- кнопку скачивания `dataset_manifest.csv`.

Для текущего набора должно быть 499 файлов.

Сначала запустите приложение на компьютере, где точно есть полный набор 499 файлов.
Скопируйте fingerprint из боковой панели и впишите его в `.env`:

```env
EXPECTED_DATASET_FINGERPRINT=<fingerprint>
```

Тот же fingerprint передайте всем разметчикам.

Если у человека 493 вместо 499 или fingerprint отличается, приложение остановит новую разметку, но существующий `labels.csv` не изменит.

## Объединение двух разметчиков

У каждого должен быть свой OUTPUT:

```text
Размеченные_1/
Размеченные_2/
```

Объединять нужно ПО `relative_path`, а не по номеру файла в интерфейсе.

Пример:

```powershell
python merge_labels.py `
  ".\Размеченные_1\labels.csv" `
  ".\Размеченные_2\labels.csv" `
  --out-dir ".\merged_labels"
```

Результаты:

```text
merged_labels/
├── labels_merged.csv
├── labels_conflicts.csv
└── labels_all_rows.csv
```

- `labels_merged.csv` — все уникальные пути без конфликта;
- `labels_conflicts.csv` — один и тот же DICOM получил разные классы;
- `labels_all_rows.csv` — полный аудит исходных разметок.

Если два человека размечали один файл одинаково, он попадёт в merged как согласованная разметка.
Если классы отличаются, скрипт не выбирает победителя автоматически.
