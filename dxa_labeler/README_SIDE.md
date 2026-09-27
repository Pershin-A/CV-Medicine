# DXA labeler v4.2 — явная LEFT / RIGHT разметка

В интерфейсе поле стороны больше не спрятано в отдельном radio. Основной блок выбора содержит:

- Не определено
- Позвоночник
- Нога — левая
- Нога — правая
- Нога — сторона не указана

В `labels.csv` сохраняется совместимый формат:

- `label = LEG`, `side = LEFT`
- `label = LEG`, `side = RIGHT`
- `label = SPINE`, `side = ""`

Старые 499 строк не теряются. При повторном сохранении меняется только текущая строка, создаётся `labels.csv.bak`, а история добавляется в `labels_history.csv`.

## Обновление Docker

Распакуйте архив поверх папки `dxa_labeler`, затем из неё выполните:

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d --force-recreate
```

Убедитесь, что `.env` указывает на текущие папки:

```env
DXA_DATA_DIR=C:/Users/Андрей/Desktop/Хакатон/Исследования
DXA_OUTPUT_DIR=C:/Users/Андрей/Desktop/Хакатон/Размеченные
EXPECTED_DICOM_COUNT=499
STRICT_DATASET_CHECK=1
```

Интерфейс: http://localhost:8501