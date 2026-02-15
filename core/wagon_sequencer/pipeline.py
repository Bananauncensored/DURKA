import os

class WagonSequencingPipeline:
    def __init__(self, detector, processor):
        self.detector = detector
        self.processor = processor

    def run(self, image_dir):
        images = sorted(os.listdir(image_dir))

        for img in images:
            path = os.path.join(image_dir, img)
            box_count = self.detector.count_boxes(path)

            batch = self.processor.process_frame(path, box_count)
            if batch:
                yield batch

        final = self.processor.flush()
        if final:
            yield final
