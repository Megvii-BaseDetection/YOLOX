from Fusion.save.fusion_save_init import fusion_save_init


class Saver(object):
    def __init__(
            self,
            current_time,
            mode,
    ):
        self.current_time = current_time
        self.mode = mode
        self.fusion_csv_f, self.fusion_csv_writer = fusion_save_init(
            self.current_time,
            self.mode
        )

    def fusion_save(self, fusion_frame, fusion_class, cnt):
        for i in range(len(fusion_frame)):
            x = fusion_frame[i, 0]
            y = fusion_frame[i, 1]
            z = fusion_frame[i, 2]
            v = fusion_frame[i, 3]
            e = fusion_frame[i, 4]
            class_name = fusion_class[i]
            self.fusion_csv_writer.writerow([cnt, x, y, z, v, e, class_name])
