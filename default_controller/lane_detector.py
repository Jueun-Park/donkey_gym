import cv2
import numpy as np

RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


class LaneDetector:
    def __init__(self):
        pass

    def detect_lane(self, image: np.ndarray):
        self.original_image_array = image.copy()
        self.image_array = image.copy()

        self._blur_image()
        self._get_grayscale_image()
        self._detect_edges()
        self._get_roi_image()
        self._hough_line()
        if self.hough_lines is None:
           return False, None, None  # not done
        self._get_lane_candidates()
        self._intersection_of_lanes()
        self._get_angle_error()

        angle = None
        intersept = None
        return True, intersept, angle

    def _blur_image(self):
        kernel_size = 3
        self.image_array = cv2.GaussianBlur(
            self.image_array, (kernel_size, kernel_size), 0)

    def _get_grayscale_image(self):
        self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)

    def _detect_edges(self):
        self.image_array = cv2.Canny(self.image_array,
                                     50,
                                     150,
                                     )

    def _get_roi_image(self):
        xsize = self.image_array.shape[1]
        ysize = self.image_array.shape[0]
        mask = np.zeros_like(self.image_array)
        dx1 = int(1 * xsize)
        dx2 = int(0.675 * xsize)
        dy = int(0.475 * ysize)
        dy2 = int(0.2 * ysize)
        vertices = np.array([[(dx1, ysize - dy2),
                              (dx2, dy),
                              (xsize - dx2, dy),
                              (xsize - dx1, ysize - dy2)]],
                            dtype=np.int32)

        if len(self.image_array.shape) > 2:
            channel_count = self.image_array.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.image_array = cv2.bitwise_and(self.image_array, mask)

    def _hough_line(self):
        self.hough_lines = cv2.HoughLinesP(self.image_array,
                                           rho=1,
                                           theta=np.pi/180,
                                           threshold=30,
                                           minLineLength=10,
                                           maxLineGap=20,
                                           )
        if self.hough_lines is None:
            return

    def __draw_lines(self, lines, color):
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2 or y1 == y2:
                    continue
                cv2.line(self.original_image_array,
                            (x1, y1), (x2, y2), color, 2)

    def _get_lane_candidates(self):
        self.right = []
        self.left = []
        self.r_x_points, self.r_y_points = [], []
        self.l_x_points, self.l_y_points = [], []
        for x1, y1, x2, y2 in self.hough_lines[:, 0]:
            m = self.__slope(x1, y1, x2, y2)
            if m >= 0:
                self.right.append([[x1, y1, x2, y2]])
                self.r_x_points.append(x1)
                self.r_x_points.append(x2)
                self.r_y_points.append(y1)
                self.r_y_points.append(y2)

            else:
                self.left.append([[x1, y1, x2, y2]])
                self.l_x_points.append(x1)
                self.l_x_points.append(x2)
                self.l_y_points.append(y1)
                self.l_y_points.append(y2)

        self.__draw_lines(self.right, RED)
        self.__draw_lines(self.left, BLUE)

    def __slope(self, x1, y1, x2, y2):
        return (y1 - y2) / (x1 - x2)

    def _intersection_of_lanes(self):
        r_m, r_c = self.__one_line_linear_regression(self.r_x_points, self.r_y_points)
        l_m, l_c = self.__one_line_linear_regression(self.l_x_points, self.l_y_points)
        a = np.array([[-r_m, 1], [-l_m, 1]])
        b = np.array([r_c, l_c])
        self.intersection_point = np.linalg.solve(a, b)
        self.intersection_point = tuple(int(i) for i in self.intersection_point)
        print(self.intersection_point)
        cv2.circle(img=self.original_image_array,
                    center=self.intersection_point,
                    radius=1,
                    color=GREEN,
                    thickness=3,
                    )

    def __one_line_linear_regression(self, x_points, y_points):
        A = np.vstack([x_points, np.ones(len(x_points))]).T
        m, c = np.linalg.lstsq(A, y_points, rcond=None)[0]
        return m, c

    def _get_angle_error(self):
        pass


if __name__ == "__main__":
    image = cv2.imread("default_controller/sample0_0.png", cv2.IMREAD_COLOR)
    detector = LaneDetector()
    detector.detect_lane(image)
    # conda install -c conda-forge opencv=4.1.0
    cv2.imshow('processed', detector.image_array)
    cv2.imshow('original', detector.original_image_array)
    cv2.waitKey(0)
