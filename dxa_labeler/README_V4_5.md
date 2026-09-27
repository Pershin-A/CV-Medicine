# DXA Labeler 4.5-fast-clinical

Добавлено:
- вывод информации из `разметка.xlsx` для текущего study;
- показываются только названия полей со значением `1` для текущей области;
- показывается итог текущей области;
- комментарий показывается всегда;
- для ноги: side, metal, fracture;
- для позвоночника: spine_issue = NONE / SCOLIOSIS / LUMBARIZATION / BOTH;
- Excel парсится один раз и кэшируется.

`разметка.xlsx` монтируется read-only в `/reference/разметка.xlsx`.

После замены файлов пересоберите контейнер:

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d --force-recreate
```

Проверка версии:

```powershell
docker exec dxa_dicom_labeler python -c "from pathlib import Path; t=Path('/app/app.py').read_text(); print('4.5-fast-clinical' in t)"
```
