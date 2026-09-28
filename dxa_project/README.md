# DXA: запуск в вашей структуре каталогов

## Актуальный результат

Обучение на 15 000 аугментациях выполнено. [Простой отчёт для команды с примерами](team_demo/index.html), [что загружать в Git](GIT_PUBLICATION.md). Текущий код: `geometry_ml/`, ноутбук: `DXA_augmented_pipeline.ipynb`. Ниже сохранены инструкции ранних экспериментов.

Все команды запускаются из корня `Хакатон`. Исходные DICOM, Excel, CSV разметки и сохранённые веса присутствуют. Проверка от 26.09.2026: подготовка манифеста, экспорт геометрии и один проход модели выполнены; полное обучение ещё не запускалось.

| Назначение | Путь относительно корня |
| --- | --- |
| Таблица семи целей по исследованиям | `разметка.xlsx` |
| Оригиналы DICOM (UID/серия/изображения) | `Исследования/` |
| Разметка трёх участников | `Размеченные/labels.csv`, `Размеченные_25_09/labels.csv`, `Размеченные_my/labels.csv` |
| Геометрия | `Размеченные/geometry/` и возможные `geometry/` в остальных папках |
| Сохранённые модели классификации | `prototype_output/resnet18_anatomy3.pt`, `prototype_output/resnet18_metal_binary.pt` |
| Готовый bundle и экспериментальные результаты | `prototype_output/dxa_qc_pipeline_bundle.pt`, `prototype_output/full_dataset_predictions.csv` |
| Экспериментальный скрипт инференса | `dxa_inference_pipeline.py` |
| Внешний код RouterNet | `RouterNet/`; его MICCAI изображения не смешивать с оценкой DXA |
| Три демонстрационных DICOM | `Для теста/`; не включать в валидационную оценку |

## Команды PowerShell

```powershell
Set-Location 'C:\Users\Андрей\Desktop\Хакатон'
python .\dxa_project\audit.py --reference '.\разметка.xlsx' --output '.\dxa_project\outputs\audit.json'
python .\dxa_project\merge_labels.py --root . --output 'dxa_project/outputs/merged_labels.csv'
python .\dxa_project\prepare.py --reference '.\разметка.xlsx' --dicoms '.\Исследования' --output '.\dxa_project\outputs\manifest.csv'
python .\dxa_project\train.py --manifest '.\dxa_project\outputs\manifest.csv' --labels '.\dxa_project\outputs\merged_labels.csv' --dicoms-root '.\Исследования' --output '.\dxa_project\outputs\model'
```

Установите `pydicom`, декодер для сжатых DICOM при необходимости, `pandas`, `openpyxl`, `numpy`, `scikit-learn`, `Pillow` и совместимые `torch`/`torchvision`. Версии для итогового контейнера требуется отдельно зафиксировать после проверки на вашей машине.

`merge_labels.py` сопоставляет точные относительные пути, записывает несовпадающие метки в `dxa_project/outputs/label_conflicts.json` и исключает конфликтные изображения. Копии DICOM в `Размеченные*` повторно не сканируются. Неизвестный класс или сторона не попадают в обучение. `prepare.py` сканирует исходные `.dcm`, соединяет их с Excel по имени папки исследования (`reference_study_uid`): поле `study` в Excel **не равно** DICOM `StudyInstanceUID`. Оба идентификатора сохранены в манифесте. Разбиение выполняется по пациенту, если его ID информативен, иначе по исследованию. Во всех предоставленных DICOM `PatientID = Anonymized`, поэтому невозможно доказать отсутствие одного пациента в разных выборках. Если внутри исследования несколько снимков одной анатомической области, цели качества этой области маскируются: Excel не указывает, какой снимок оценивался.

Экспорт геометрии каждой папки запускается отдельно:

```powershell
python .\labeler\export_geometry.py --output-root '.\Размеченные'
python .\labeler\export_geometry.py --output-root '.\Размеченные_25_09'
python .\labeler\export_geometry.py --output-root '.\Размеченные_my'
python .\dxa_project\audit.py --reference '.\разметка.xlsx' --geometry '.\Размеченные\annotations_geometry.jsonl'
```

Команда экспорта сигнализирует об отсутствующих sidecar-файлах. `audit.py` анализирует один экспорт геометрии за запуск. Координаты геометрии соответствуют исходным пикселям DICOM.

Для проверки уже имеющихся весов на вашем компьютере:

```powershell
python .\dxa_inference_pipeline.py --bundle '.\prototype_output\dxa_qc_pipeline_bundle.pt' --input '.\Для теста' --output '.\dxa_project\outputs\example_predictions.csv'
```

Результаты существующих ноутбуков нельзя считать независимой валидацией до проверки пересечения обучающих и оценочных исследований. Новый скрипт ResNet18 — базовый эксперимент со случайной инициализацией, а не замена готовым моделям или будущим моделям геометрии. Цели позиционирования и ротации бедра в Excel объединены. Положительных примеров для ROI всего 3 справа и 4 слева, поэтому в отдельных частях выборки они могут отсутствовать. При текущем разбиении после исключения неоднозначных изображений ROI справа имеет один положительный обучающий пример и ни одного на валидации и тесте; ROI слева — по одному на обучении и валидации, ни одного на тесте. Метрики для одноклассовых частей выборки выводятся как `null`. Модели линий/точек, визуализация и сервис требуют завершённой разметки. Подробный текущий статус и порядок дальнейших работ — в `READINESS.md`.

Пока геометрическая разметка накапливается, существующие признаки из ноутбуков можно исследовать командой `python dxa_project/rule_experiments.py`. Сравнение правил, результаты групповой проверки и ограничения изложены в `RULE_RESULTS.md`.

Интерактивный вариант: из корня `Хакатон` запустите `python -m jupyter lab .\dxa_project\DXA_rule_experiments.ipynb`, выберите ядро Python 3 и нажмите **Run → Run All Cells**. Ноутбук пересчитывает те же CSV, показывает таблицы, график, пороги по фолдам и снимки с ошибочными решениями. Он работает с сохранёнными признаками; DICOM на этом шаге не перечитываются.

Базовый ML эксперимент с семью независимыми классификаторами поверх замороженного ImageNet ResNet18 запускается из корня проекта командой `.\.venv\Scripts\python.exe .\dxa_project\train_naive_resnet.py`. Его таблицы, OOF прогнозы и обученные головы находятся в `dxa_project/outputs/naive_resnet18/`. Для просмотра результатов откройте `dxa_project/DXA_naive_resnet18.ipynb`; подробное описание и ограничения — в `dxa_project/NAIVE_RESNET_RESULTS.md`.

Парное сравнение правил и ResNet18 на одинаковых исследованиях и фолдах: `python dxa_project/compare_rules_resnet.py`. Единая таблица и выводы — в `dxa_project/PAIRED_COMPARISON.md`; подробные прогнозы и параметры — в `dxa_project/outputs/paired_comparison/`.
