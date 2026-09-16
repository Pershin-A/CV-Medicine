# DXA DICOM Labeler

Локальное приложение для ручной разметки DICOM DXA-файлов.

## Рекомендуемая структура

C:\Users\Андрей\Downloads\Датасет\
├── Исследования\
├── dxa_labeler\
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
└── Размеченные\     # создастся автоматически

## Запуск

Открой PowerShell:

cd "C:\Users\Андрей\Downloads\Датасет\dxa_labeler"
docker compose up --build

После запуска открыть:

http://localhost:8501

## Остановка

docker compose down

## Повторный запуск

docker compose up -d

## Где лежит результат

Размеченные DICOM сохраняются в:

C:\Users\Андрей\Downloads\Датасет\Размеченные\...

Внутренняя структура каталогов повторяет "Исследования".

Кроме DICOM-файлов создаётся:

Размеченные\labels.csv

## DICOM private metadata

Private creator:
DXA_MANUAL_LABELER

Поля:
- 0x01: CS — MANUAL_LABEL (`SPINE`, `LEG`, `UNKNOWN`)
- 0x02: LT — NOTES
- 0x03: LO — ANNOTATOR
- 0x04: DT — ANNOTATED_DATETIME

Конкретный числовой private tag может зависеть от занятого private block в исходном DICOM,
поэтому корректно читать данные через `Dataset.private_block(...)`, а не жёстко ожидать
например `(0011,1001)`.

## Важно

Исходная папка подключается в Docker как read-only (`:ro`).
Поэтому приложение физически не может перезаписать оригиналы.
