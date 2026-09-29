# Изменения второго этапа

## Внесено

- `decoding.py`: максимум только внутри реального кадра; выход за допустимую
  координатную область возвращает неопределённость вместо clipping к краю.
  Доступны legacy, masked_argmax, local_softargmax, masked_logit_argmax,
  local_logit_softargmax. Рабочий максимум выбирается по логитам: насыщение
  sigmoid до 1 не должно создавать ложные равные максимумы.
- `landmarks.py`: отдельная сеть трёх точек или три независимые сети, heatmap /
  координатный loss, разрешения 256/384/512, ROI с контекстом и обратным
  преобразованием координат. ROI исходной разметки не изменяется.
- `protocol.py`: внутренние train/validation с близкими долями артефактов,
  проверка Study overlap и точных дублей. Прежний fold 0 сохранён для сравнения,
  но не объявляется новым независимым test. Выбор исходного снимка при sampling
  имеет равный вес независимо от числа его аугментаций.
- `train.py`: разные LR encoder/heads, cosine/plateau, best checkpoint, early
  stopping и source balancing для всех прежних модулей. Артефакты балансируются
  одновременно по классу и исходному снимку.
- `experiments.py`: запуск и измерение вариантов, PCK 5/10 мм, ошибки каждой
  точки, видимость и бинарное позиционирование с групповыми CI.
- `predict.py`: автоматически подключает `hip_points.pt`, если он лежит рядом
  с четырьмя прежними весами. Для crop-модели используется предсказанная ROI,
  при пропусках точек — возврат к полному кадру. Старые модели остаются совместимы.
- `landmark_geometry.py`: физические расстояния и нормированные отношения
  между точками; диагностика совпавших точек. Без калибровки диапазонов этот
  признак не меняет бинарную метку и не выдаёт анатомическую вариативность за ошибку.
- `artifact_review.py`: восемь расхождений вынесены в страницу и CSV проверки;
  метки и оригиналы автоматически не перезаписываются.

## Отдельный ноутбук

`dxa_project/DXA_improvement_variants.ipynb` содержит шесть вариантов архитектуры,
loss, размера и crop; пять декодеров и три режима LR. Короткие проверки выполняют
два обучающих батча и validation. Это проверки исполняемости, не сравнительная
оценка точности. Длительные запуски включаются явно и сохраняются отдельно.

## Запуск

```powershell
.venv\Scripts\python.exe -m dxa_project.geometry_ml.experiments --phase decode
.venv\Scripts\python.exe -m dxa_project.geometry_ml.experiments --phase smoke --variants shared256,coordinate256,shared384,shared512,roi384,independent256
.venv\Scripts\python.exe -m dxa_project.geometry_ml.experiments --phase train --variants shared256,coordinate256 --epochs 20
.venv\Scripts\python.exe -m dxa_project.geometry_ml.experiments --phase test --variants shared256,coordinate256
```

Включить warmup для отдельного нового запуска: `--scheduler cosine --warmup-epochs 2`.
Варианты LR запускайте в отдельную `--output` папку. Чтобы изолировать влияние
длительности, задайте одинаковые LR/scheduler и меняйте только epochs.

Для остальных моделей доступны те же улучшенные параметры:

```powershell
.venv\Scripts\python.exe -m dxa_project.geometry_ml.train --task spine --epochs 20 --encoder-lr 0.00002 --head-lr 0.0001 --scheduler cosine --patience 7 --balance-sources --selection-protocol dxa_project/outputs/improvements_v2/protocol.json --augmented-root dxa_project/outputs/augmented_15000_final --output dxa_project/outputs/spine_v2
```

ROI crop в автономной оценке experiments.py использует истинную ROI и помечен
как oracle. Его качество нельзя выдавать за качество полного инференса.
На всех исходных тестовых бедрах точки видны: sensitivity и ROC AUC
позиционирования там не определены; отрицательные случаи оцениваются отдельно
на аугментациях от test-источников.

Тяжёлый вариант уже поддерживается для общего пайплайна; `PointsNet` также
поддерживает ResNet50 через architecture. Длительные абляции выполняются только
для указанных в отчёте вариантов. Остальные прошли проверку исполнения, но
из этого нельзя заключить, что они лучше или хуже.

## Полный улучшенный пайплайн

Веса экспериментального запуска сохраняются в
`outputs/improvements_v2/pipeline/`: прежний `router.pt`, переобученные
`spine.pt`, `hip.pt`, `artifact.pt`, отдельный `hip_points.pt` и диапазоны
геометрии `landmark_geometry_bounds.json`. Модель точек выбирается по ошибке
на внутренней validation среди сохранённых best/last двух вариантов.
Порог площади малого вертела калибруется на внутренней validation, а не test.

```powershell
.venv\Scripts\python.exe -m dxa_project.geometry_ml.predict "путь\к\снимку.dcm" --checkpoints dxa_project/outputs/improvements_v2/pipeline --output dxa_project/outputs/example_prediction.json
```

Дополнительная геометрическая проверка: `--position-rule framing_and_geometry`.
При неправдоподобной геометрии она возвращает неопределённый ответ, если
снимок прошёл проверку видимости и отступов; основной режим — `framing`.
Геометрические отношения не заменяют требования к наличию трёх структур.

Сравнение: `team_demo/improvements.html`, отдельный ноутбук вариантов выше.
Прежние веса и первоначальный отчёт сохранены для сравнения.

## Успехи, ошибки и ручная проверка

`team_demo/model_examples.html`: отдельные примеры для маршрутизатора,
линий позвоночника, подвздошных точек, артефактов, обеих сетей точек бедра,
ROI и маски малого вертела. Пороги отбора примеров указаны на странице;
это иллюстративные допуски, а не клинические критерии. До пяти разных
пиксельных массивов каждого типа; малое число успехов/ошибок не дополняется
повторами. Источник всех примеров — прежняя контрольная часть, геометрическим
модулям задаётся верная область тела.

Номера соответствуют строкам `Размеченные/labels.csv` и галерее всех
исходных сканов, начиная с 1. Кандидаты на проверку флагов артефактов:
**52, 55, 58, 102, 121, 122, 287, 288**. Это расхождения таблицы и визуальных
рамок, а не доказанные ошибки эталона. Повторная генерация:

```powershell
.venv\Scripts\python.exe -m dxa_project.geometry_ml.model_examples
```
