4️⃣ Обновление README

# DURKA - MobileNetV3 Wagon Classifier - 2 класса

## Модель

- MobileNetV3-Small
- Классы:
  1. `one_wagon` — цельный вагон
  2. `transition` — кадр с переходом (окончание вагонов)

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