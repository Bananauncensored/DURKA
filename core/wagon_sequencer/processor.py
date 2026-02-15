class WagonStateProcessor:
    def __init__(self):
        self.current_wagon_id = 1
        self.current_frames = []

    def process_frame(self, frame_path: str, box_count: int):
        if box_count == 0:
            return None

        if box_count > 1:
            batch = self.flush()
            self.current_wagon_id += 1
            return batch

        self.current_frames.append(frame_path)
        return None

    def flush(self):
        if not self.current_frames:
            return None

        batch = {
            "wagon_id": self.current_wagon_id,
            "frames": self.current_frames.copy()
        }
        self.current_frames.clear()
        return batch
