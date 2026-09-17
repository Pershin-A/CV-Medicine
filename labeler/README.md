# DXA DICOM Labeler

Streamlit-приложение для ручной разметки DXA/DICOM-изображений.

Приложение запускается через Docker и работает с исходными DICOM в режиме read-only.

## Возможности

Для каждого DICOM можно задать основной класс:

- `LEG` — проксимальный отдел бедра;
- `SPINE` — позвоночник;
- `UNKNOWN` — класс не определён.

Для изображений бедра дополнительно размечаются:

- сторона: `LEFT` / `RIGHT`;
- металл / имплант;
- перелом.

Для позвоночника дополнительно размечается состояние:

- нет выявленной проблемы;
- сколиоз;
- люмбализация;
- сколиоз + люмбализация.

Также можно указать:

- имя разметчика;
- свободный комментарий.

Разметка сохраняется в `labels.csv` и в private DICOM metadata размеченной копии.

Исходные DICOM не изменяются.

## Ожидаемая локальная структура

```text
project/
├── labeler/
│   ├── app.py
│   ├── dataset_manifest.py
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   └── .env
│
├── Исследования/
├── Размеченные/
└── разметка.xlsx
```

## Настройка

Перейдите в папку приложения:

```powershell
cd labeler
```

Создайте `.env`:

```powershell
Copy-Item .env.example .env
```

Стандартная конфигурация:

```env
DXA_DATA_DIR=../Исследования
DXA_OUTPUT_DIR=../Размеченные
DXA_REFERENCE_XLSX=../разметка.xlsx

EXPECTED_DICOM_COUNT=499
EXPECTED_DATASET_FINGERPRINT=
STRICT_DATASET_CHECK=1
```

`.env` является локальным файлом и не должен добавляться в Git.

## Запуск

```powershell
docker compose down
docker compose up -d --build --force-recreate
```

После запуска открыть:

```text
http://localhost:8501
```

## Проверка подключения DICOM

```powershell
docker exec dxa_dicom_labeler python -c "from pathlib import Path; print(sum(1 for p in Path('/data').rglob('*.dcm')))"
```

Для текущего набора ожидается:

```text
499
```

Проверить доступность справочного Excel:

```powershell
docker exec dxa_dicom_labeler python -c "from pathlib import Path; print(Path('/reference/разметка.xlsx').is_file())"
```

Ожидается:

```text
True
```

## Выходные данные

Разметка сохраняется в:

```text
Размеченные/labels.csv
```

Основные колонки:

| Колонка | Значения |
|---|---|
| `relative_path` | относительный путь DICOM |
| `label` | `LEG`, `SPINE`, `UNKNOWN` |
| `side` | `LEFT`, `RIGHT`, пусто |
| `metal` | `0`, `1`, пусто |
| `fracture` | `0`, `1`, пусто |
| `spine_issue` | `NONE`, `SCOLIOSIS`, `LUMBARIZATION`, `BOTH`, пусто |
| `notes` | комментарий |
| `annotator` | разметчик |
| `annotated_at` | дата и время |
| `output_path` | путь размеченной копии |

Для каждого размеченного файла создаётся DICOM-копия с той же относительной структурой каталогов.

## Информация из `разметка.xlsx`

Для текущего `study` приложение дополнительно показывает экспертную информацию из Excel.

В зависимости от выбранной области отображается информация для:

- позвоночника;
- правого бедра;
- левого бедра.

Показываются только признаки со значением `1`.

Значения `0` и пустые ячейки скрываются.

Итог для выбранной области и текстовый комментарий показываются отдельно.

## Навигация

Интерфейс поддерживает:

- выбор DICOM из списка;
- переход к предыдущему и следующему изображению;
- сохранение текущей разметки;
- сохранение и автоматический переход к следующему DICOM;
- отображение прогресса разметки.

Список DICOM и preview кэшируются для уменьшения задержек при навигации.

## DICOM warnings

Некоторые исходные DICOM могут содержать нестандартные значения отдельных metadata-тегов, например длинные `LO` или формально некорректные `UI`.

Приложение не исправляет исходные медицинские metadata.

Такие предупреждения сами по себе не означают повреждение pixel data.

## Пересборка после изменения `app.py`

После изменения исходного кода необходимо пересобрать Docker image:

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d --force-recreate
```

Проверить, какая версия кода находится внутри контейнера:

```powershell
docker exec dxa_dicom_labeler python -c "from pathlib import Path; print(Path('/app/app.py').read_text()[:300])"
```

## Безопасность данных

Не добавляйте в Git:

```text
.env
Исследования/
Размеченные/
разметка.xlsx
*.dcm
```

`Размеченные/labels.csv` также может содержать идентификаторы исследований и должен рассматриваться как файл с данными, а не как исходный код.