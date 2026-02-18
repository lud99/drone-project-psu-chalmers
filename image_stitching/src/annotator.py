import supervision as sv


class Annotator:
    def __init__(self) -> None:
        self.boxAnnotator = sv.BoxAnnotator(thickness=5)
        self.labelAnnotator = sv.LabelAnnotator(
            text_scale=1, text_position=sv.Position.TOP_LEFT
        )
        self.positionAnnotator = sv.LabelAnnotator(
            text_scale=0.5, text_position=sv.Position.BOTTOM_LEFT
        )

    def annotate_frame(self, frame, detections, labels, position_labels):
        frame = self.boxAnnotator.annotate(scene=frame, detections=detections)

        frame = self.labelAnnotator.annotate(
            scene=frame, detections=detections, labels=labels
        )

        frame = self.positionAnnotator.annotate(
            scene=frame, detections=detections, labels=position_labels
        )

        return frame
