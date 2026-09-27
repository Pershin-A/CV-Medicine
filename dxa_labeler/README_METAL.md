# LEFT/RIGHT + METAL

В labels.csv сохраняются независимые поля:
- label: LEG / SPINE / UNKNOWN
- side: LEFT / RIGHT / пусто
- metal: 0 / 1 / пусто

metal=0 — металла/импланта нет.
metal=1 — металл/имплант есть.
Пусто — признак ещё не размечен.

Пути Docker по умолчанию:
- ../Исследования -> /data
- ../Размеченные -> /output