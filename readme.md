4️⃣ Обновление README

Можно переписать readme.txt или создать README.md. Например, в формате Markdown:

# DURKA - MobileNetV3 Wagon Classifier

## Модель

- MobileNetV3-Small
- Классы:
  1. `empty` — кадр без вагона
  2. `one_wagon` — цельный вагон
  3. `transition` — кадр с переходом (окончание вагонов)

## Использование

### Обучение
```bash
python train.py

Тест

Поместить изображения в test/input_image

Запустить:

python test.py


Обработанные кадры будут в test/output_image

JSON отчёт с количеством вагонов и кадрами на каждый вагон: test/wagon_report.json

Логика

Каждая вторая фотка каждого вагона отправляется в output

Первая и последняя фотка каждого вагона всегда сохраняются

Пустые кадры (empty) не отправляются
