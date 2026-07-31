'''import cv2
import numpy as np


class AnnotationDetector:

    def __init__(
        self,
        min_width=6,
        min_height=6,
        padding=10,
        merge_dist=15,
        min_confident_area=1000,
        min_confident_dim=40,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.padding = padding
        self.merge_dist = merge_dist

        self.min_ink_pixels = 300    
        self.min_fill_ratio = 0.08 

        # post-merge confidence thresholds — tune against your smallest
        # real mark (e.g. the filled dot) and your worst speckle case
        self.min_confident_area = min_confident_area
        self.min_confident_dim = min_confident_dim

    def remove_lines(self, binary):
        h, w = binary.shape

        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
        vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 15))

        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
        vert_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel, iterations=1)

        lines_mask = cv2.bitwise_or(horiz_lines, vert_lines)
        lines_mask = cv2.dilate(lines_mask, np.ones((3, 3), np.uint8), iterations=1)

        return cv2.bitwise_and(binary, cv2.bitwise_not(lines_mask))

    def _boxes_close(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        # expand a by merge_dist and test overlap with b
        ax1 -= self.merge_dist; ay1 -= self.merge_dist
        ax2 += self.merge_dist; ay2 += self.merge_dist
        return not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2)

    def _merge_boxes(self, boxes):
        # boxes: list of (x1,y1,x2,y2). Repeated pairwise merge until stable.
        changed = True
        while changed:
            changed = False
            out = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                cur = list(boxes[i])
                used[i] = True
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    if self._boxes_close(tuple(cur), boxes[j]):
                        cur[0] = min(cur[0], boxes[j][0])
                        cur[1] = min(cur[1], boxes[j][1])
                        cur[2] = max(cur[2], boxes[j][2])
                        cur[3] = max(cur[3], boxes[j][3])
                        used[j] = True
                        changed = True
                out.append(tuple(cur))
            boxes = out
        return boxes

    def detect(self, image):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.medianBlur(gray, 5)

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        binary = self.remove_lines(binary)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        raw_boxes = []
        for i in range(1, num_labels):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]

            if w < self.min_width or h < self.min_height:
                continue

            raw_boxes.append((x, y, x + w, y + h))

        merged = self._merge_boxes(raw_boxes)

        # ---- confidence decision AFTER merging, on actual ink density ----
        confident = []
        for (x1, y1, x2, y2) in merged:
            w = x2 - x1
            h = y2 - y1
            bbox_area = w * h

            if bbox_area < self.min_confident_area:
                continue
            if w < self.min_confident_dim or h < self.min_confident_dim:
                continue

            # count real ink pixels inside this box, not just bbox geometry —
            # a merged bbox spanning two noise specks is mostly empty
            ink_pixels = cv2.countNonZero(binary[y1:y2, x1:x2])
            fill_ratio = ink_pixels / float(bbox_area)

            if ink_pixels < self.min_ink_pixels:
                continue
            if fill_ratio < self.min_fill_ratio:
                continue

            confident.append((x1, y1, x2, y2))

        detected = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        rois = []

        for (x1, y1, x2, y2) in confident:
            x1p = max(0, x1 - self.padding)
            y1p = max(0, y1 - self.padding)
            x2p = min(gray.shape[1], x2 + self.padding)
            y2p = min(gray.shape[0], y2 + self.padding)

            rois.append({
                "bbox": (x1p, y1p, x2p, y2p),
                "roi": gray[y1p:y2p, x1p:x2p],
            })

            cv2.rectangle(detected, (x1p, y1p), (x2p, y2p), (0, 0, 255), 2)

        return rois, {"binary": binary, "detected": detected}'''

"""
Annotation Detector

Detects handwritten annotation regions by removing printed table
lines, extracting connected components, merging nearby fragments and
filtering candidates based on size and ink density. Returns cropped
ROIs for OCR.
"""


import cv2
import numpy as np


class AnnotationDetector:

    def __init__(
        self,
        min_width=6,
        min_height=6,
        padding=10,
        merge_dist=25,
        min_confident_area=800,
        min_confident_dim=20,
        min_ink_pixels=150,
        min_fill_ratio=0.08,
    ):

        self.min_width = min_width
        self.min_height = min_height
        self.padding = padding
        self.merge_dist = merge_dist

        self.min_confident_area = min_confident_area
        self.min_confident_dim = min_confident_dim
        self.min_ink_pixels = min_ink_pixels
        self.min_fill_ratio = min_fill_ratio

    # --------------------------------------------------

    def remove_lines(self, binary):

        h, w = binary.shape

        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(20, w // 15), 1)
        )

        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(20, h // 15))
        )

        horiz_lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            horiz_kernel
        )

        vert_lines = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            vert_kernel
        )

        lines = cv2.bitwise_or(
            horiz_lines,
            vert_lines
        )

        lines = cv2.dilate(
            lines,
            np.ones((3, 3), np.uint8),
            iterations=1
        )

        return cv2.bitwise_and(
            binary,
            cv2.bitwise_not(lines)
        )

    # --------------------------------------------------

    def _boxes_close(self, a, b):

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ax1 -= self.merge_dist
        ay1 -= self.merge_dist
        ax2 += self.merge_dist
        ay2 += self.merge_dist

        return not (
            bx2 < ax1 or
            bx1 > ax2 or
            by2 < ay1 or
            by1 > ay2
        )

    # --------------------------------------------------

    def _merge_boxes(self, boxes):

        changed = True

        while changed:

            changed = False
            merged = []
            used = [False] * len(boxes)

            for i in range(len(boxes)):

                if used[i]:
                    continue

                current = list(boxes[i])
                used[i] = True

                for j in range(i + 1, len(boxes)):

                    if used[j]:
                        continue

                    if self._boxes_close(
                        tuple(current),
                        boxes[j]
                    ):

                        current[0] = min(current[0], boxes[j][0])
                        current[1] = min(current[1], boxes[j][1])
                        current[2] = max(current[2], boxes[j][2])
                        current[3] = max(current[3], boxes[j][3])

                        used[j] = True
                        changed = True

                merged.append(tuple(current))

            boxes = merged

        return boxes

    # --------------------------------------------------

    def detect(
        self,
        gray,
        binary
    ):

        # Remove printed table lines only
        binary = self.remove_lines(binary)

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8
        )

        boxes = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            if w < self.min_width:
                continue

            if h < self.min_height:
                continue

            boxes.append(
                (
                    x,
                    y,
                    x + w,
                    y + h
                )
            )

        # Merge nearby fragments
        merged = self._merge_boxes(boxes)

        confident = []

        for x1, y1, x2, y2 in merged:

            w = x2 - x1
            h = y2 - y1

            bbox_area = w * h

            if bbox_area < self.min_confident_area:
                continue

            if w < self.min_confident_dim:
                continue

            if h < self.min_confident_dim:
                continue

            ink_pixels = cv2.countNonZero(
                binary[y1:y2, x1:x2]
            )

            fill_ratio = ink_pixels / float(bbox_area)

            if ink_pixels < self.min_ink_pixels:
                continue

            if fill_ratio < self.min_fill_ratio:
                continue

            confident.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

        detected = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        rois = []

        for x1, y1, x2, y2 in confident:

            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(gray.shape[1], x2 + self.padding)
            y2 = min(gray.shape[0], y2 + self.padding)

            roi = gray[
                y1:y2,
                x1:x2
            ]

            rois.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "roi": roi
                }
            )

            cv2.rectangle(
                detected,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

        return rois, {
            "binary": binary,
            "detected": detected
        }