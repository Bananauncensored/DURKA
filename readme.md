Архитектура (коротко)

[Кадры] 
   ↓
Base YOLO (детекция)
   ↓
Async Queue (вагон → кадры)
   ↓
Next Model (ID, ориентация, и т.д.)


1️⃣ Контракт данных между моделями (ВАЖНО)

Мы договоримся, что базовая модель отдаёт данные пачками по вагону:

{
    "wagon_id": int,
    "frames": [PIL.Image | np.ndarray | path],
}


Это идеальный формат для следующей модели.
2️⃣ Асинхронная очередь

Используем стандартную библиотеку:

import asyncio


Очередь:

wagon_queue = asyncio.Queue()


3️⃣ Асинхронный базовый процессор (YOLO)
import os
import asyncio
from ultralytics import YOLO

class BaseWagonProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.current_wagon_id = 1
        self.current_wagon_frames = []

    async def process_frame(self, frame_path: str, wagon_queue: asyncio.Queue):
        results = self.model.predict(frame_path, verbose=False)
        boxes = results[0].boxes if results else []
        num_boxes = len(boxes)

        if num_boxes == 0:
            # пусто — игнор
            return

        if num_boxes == 1:
            # один вагон — копим кадры
            self.current_wagon_frames.append(frame_path)

        else:
            # 2+ бокса → вагон заканчивается
            if self.current_wagon_frames:
                await wagon_queue.put({
                    "wagon_id": self.current_wagon_id,
                    "frames": self.current_wagon_frames.copy()
                })

                self.current_wagon_id += 1
                self.current_wagon_frames.clear()

4️⃣ Асинхронный producer (чтение кадров)
async def frame_producer(
    images_dir: str,
    processor: BaseWagonProcessor,
    wagon_queue: asyncio.Queue
):
    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )

    for img in image_files:
        await processor.process_frame(
            os.path.join(images_dir, img),
            wagon_queue
        )

    # отправляем последний вагон
    if processor.current_wagon_frames:
        await wagon_queue.put({
            "wagon_id": processor.current_wagon_id,
            "frames": processor.current_wagon_frames
        })

    # сигнал завершения
    await wagon_queue.put(None)

5️⃣ Consumer — следующая модель (заглушка)

Вот точка подключения второй нейронки (ID + ориентация):

async def next_model_consumer(wagon_queue: asyncio.Queue):
    while True:
        wagon_data = await wagon_queue.get()

        if wagon_data is None:
            break

        wagon_id = wagon_data["wagon_id"]
        frames = wagon_data["frames"]

        # 👇 ТУТ ВТОРАЯ МОДЕЛЬ
        print(f"[NEXT MODEL] Вагон {wagon_id}, кадров: {len(frames)}")

        # пример:
        # id_result = id_model.predict(frames)
        # orientation = orient_model.predict(frames)

6️⃣ Точка входа (main)
async def main():
    MODEL_PATH = r"runs/train/base_wagon_model/weights/best.pt"
    IMAGES_DIR = r"dataset/images/test"

    wagon_queue = asyncio.Queue(maxsize=5)

    processor = BaseWagonProcessor(MODEL_PATH)

    producer_task = asyncio.create_task(
        frame_producer(IMAGES_DIR, processor, wagon_queue)
    )

    consumer_task = asyncio.create_task(
        next_model_consumer(wagon_queue)
    )

    await asyncio.gather(producer_task, consumer_task)


if __name__ == "__main__":
    asyncio.run(main())

7️⃣ Что ты в итоге получил

✅ Ничего не пишется на диск
✅ Данные живут в оперативке
✅ Асинхронная обработка
✅ Чёткий контракт между моделями
✅ Готово к масштабированию (очередь → multiprocessing → Kafka)


Wagon_seqiencer

DURKA/
 ├── core/
 │    ├── wagon_sequencer/
 │    │    ├── __init__.py
 │    │    ├── model.py          # загрузка YOLO
 │    │    ├── processor.py      # логика 0 / 1 / 2+ боксов
 │    │    ├── pipeline.py       # связка: папка → вагоны
 │    │    └── types.py          # структуры данных


Структура данных вагона
WagonBatch = {
    "wagon_id": int,
    "frames": list[str],   # пути к изображениям
}


Пример реального использования:

from core.wagon_sequencer.model import WagonDetectionModel
from core.wagon_sequencer.processor import WagonStateProcessor
from core.wagon_sequencer.pipeline import WagonSequencingPipeline

detector = WagonDetectionModel("weights/base_wagon.pt")
processor = WagonStateProcessor()
pipeline = WagonSequencingPipeline(detector, processor)

for wagon_batch in pipeline.run("IMAGE"):
    print(f"Вагон {wagon_batch['wagon_id']}")
    next_model.process(wagon_batch)


    core/wagon_sequencer/model.py
Назначение

Модуль-обёртка над нейросетевой моделью детекции вагонов.
Отвечает только за запуск инференса и возврат количества найденных вагонов (bounding boxes) на изображении.

Модуль не содержит бизнес-логики, не принимает решений и не хранит состояние.

Ответственность

загрузка весов модели

инференс одного изображения

возврат количества bounding box’ов

Публичный интерфейс
class WagonDetectionModel:
    def __init__(weights_path: str)
    def count_boxes(image_path: str) -> int

count_boxes
Параметр	Тип	Описание
image_path	str	путь к изображению

Возвращает

0 — вагонов нет

1 — один вагон

>1 — пересечение вагонов

Ограничения

не определяет ID вагонов

не фильтрует кадры

не знает про очередь или pipeline

Использование
detector = WagonDetectionModel("weights/base_wagon.pt")
count = detector.count_boxes("frame_001.jpg")

core/wagon_sequencer/processor.py
Назначение

Модуль управления состоянием текущего вагона.
Принимает результат детекции и решает:

относится ли кадр к текущему вагону

завершён ли текущий вагон

можно ли передавать данные дальше

Ответственность

хранение current_wagon_id

накопление кадров одного вагона

формирование WagonBatch

управление сменой вагона

Логика принятия решений
Кол-во боксов	Действие
0	кадр игнорируется
1	кадр добавляется в текущий вагон
>1	текущий вагон завершается
Публичный интерфейс
class WagonStateProcessor:
    def process_frame(frame_path: str, box_count: int) -> dict | None
    def flush() -> dict | None

Формат выходных данных (WagonBatch)
{
    "wagon_id": int,
    "frames": list[str]
}

Важно

состояние хранится только здесь

wagon_id увеличивается только при пересечении вагонов

consumer не должен изменять frames

core/wagon_sequencer/pipeline.py
Назначение

Связывает модель детекции и процессор состояния в единый pipeline.
Обрабатывает последовательность изображений и генерирует готовые вагоны.

Ответственность

последовательная обработка изображений

вызов детектора

передача результатов в процессор

генерация WagonBatch через yield

Архитектура
images → detector → processor → WagonBatch → next model

Публичный интерфейс
class WagonSequencingPipeline:
    def run(image_dir: str) -> Generator[WagonBatch]

Поведение

изображения сортируются по имени

pipeline ленивый (generator)

последний вагон автоматически сбрасывается

Использование
pipeline = WagonSequencingPipeline(detector, processor)

for wagon in pipeline.run("IMAGE"):
    next_model.process(wagon)

Преимущества

легко заменить детектор

не зависит от формата следующей модели

подходит для async / queue архитектуры
