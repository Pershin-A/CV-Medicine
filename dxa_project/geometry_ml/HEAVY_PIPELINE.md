# Более тяжёлый вариант для сервера

Реализован тем же конвейером, переключатель `--architecture heavy`.
Загрузка чекпойнтов при инференсе автоматически учитывает архитектуру.

| Компонент | Текущий | Более тяжёлый |
|---|---|---|
| Маршрутизация | Замороженный ResNet18 + голова | Та же модель |
| Линии и точки позвоночника | ResNet18 + U-Net | ResNet50 + U-Net |
| Общая модель бедра | ResNet18 + U-Net, heatmaps, ROI | ResNet50 + U-Net с теми же выходами |
| Артефакты | Faster R-CNN MobileNetV3 FPN | Faster R-CNN ResNet50 FPN v2 |
| Геометрические условия | Совместные оси и правила | Те же правила |

ResNet50 и Faster R-CNN ResNet50 FPN v2 доступны в torchvision:
[ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[Faster R-CNN FPN v2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html).
Более ёмкая модель может оказаться лучше, но это требует сравнения на том же fold;
результаты COCO/ImageNet не являются оценкой качества на DXA.

Таргеты и функции потерь сохранены: Dice+BCE для линий/маски, heatmap и координатные
потери для точек, BCE для видимости, регрессия ROI; стандартные потери детектора.
Бедро справа зеркально приводится к левому для геометрических моделей,
но маршрутизатор получает исходное направление.

## Запуск на Linux/GPU из корня проекта

```bash
python -m dxa_project.geometry_ml.train \
  --architecture heavy --augmented-root dxa_project/outputs/augmented_15000_final \
  --output dxa_project/outputs/geometry_ml_heavy \
  --epochs 20 --size 384 --batch-size 2 --loader-workers 4 --fold 0

python -m dxa_project.geometry_ml.evaluate \
  --checkpoints dxa_project/outputs/geometry_ml_heavy \
  --output dxa_project/outputs/geometry_ml_heavy_evaluation --fold 0
```

На сервер нужно перенести проект, `Исследования/`, `Размеченные/`, оригинальный
манифест и финальный аугментированный набор. Загрузчик строит пути от корня проекта;
сохранённые Windows-пути исходного манифеста в обучении не используются.
Первый запуск с pretrained весами скачивает их в `outputs/torch_hub`.
Установить совместимые CUDA-сборки torch и torchvision для GPU сервера.
Точный batch size определяется доступной памятью; детектор по умолчанию использует
своё масштабирование с короткой стороной 800 пикселей.

## Воспроизводимое сравнение после доразметки

Сохранить текущий набор неизменным. Новую разметку разместить отдельной версией
с теми же путями изображений и source_study_uid. Повторить тот же fold, seed,
архитектуру и число эпох; сравнить на фиксированных исходных исследованиях
вне обучения. Не перераспределять версии одного источника между train/validation.
Автоматически исчезнувшая точка и реально ещё видимая структура — разные случаи:
доразметка особенно важна для подвздошных костей после crop/поворота.

Локальная проверка работоспособности тяжёлых моделей:
`python -m dxa_project.geometry_ml.verify_heavy`.
Она использует случайные веса и маленькие тестовые размеры; не является обучением
или оценкой точности тяжёлого варианта на реальных данных.
