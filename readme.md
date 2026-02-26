4️⃣ Обновление README

# DURKA - MobileNetV3 Wagon Classifier - 2 класса (доработанная логика)

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

Кадры (`transition`) не отправляются


Если нужно добавить 3 класс. 

В заходим в папку dataset добавляем в каждую из папке новые классы.

--dataset|----test|------one_wagon
         |        |------transition
         |        |------"новый_класс"
         |
         |-----val|------one_wagon
         |        |------transition
         |        |------"новый_класс"
         |
         |
         |---train|------one_wagon
         |        |------transition
         |        |------"новый_класс"


в файле train.py меняем эти строку кода 
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
                          ||
                          \/
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)

в файле test.py меняем строку
CLASS_NAMES = ["one_wagon", "transition"]
                          ||
                          \/
CLASS_NAMES = ["новый_класс","one_wagon", "transition"]



model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)
                          ||
                          \/
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 3)




