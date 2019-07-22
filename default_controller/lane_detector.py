import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        pass

    def detect_lane(self, image: np.ndarray):
        self.image_array = image
        self._get_roi_image()
        self._blur_image()
        self._detect_edges()
        self._hough_line()
        self._get_lane_candidates()
        self._lines_linear_regression()
        self._get_bird_eye_view()
        self._get_angle_and_intersept

        angle = None
        intersept = None
        return True, intersept, angle

    def _get_roi_image(self):
        pass

    def _blur_image(self):
        pass

    def _detect_edges(self):
        pass

    def _hough_line(self):
        pass

    def _get_lane_candidates(self):
        pass

    def _lines_linear_regression(self):
        pass

    def _get_bird_eye_view(self):
        pass

    def _get_angle_and_intersept(self):
        pass


if __name__ == "__main__":
    pass
