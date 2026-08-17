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
        min_width=8, #6
        min_height=6, #6
        padding=10, #10
        merge_dist=25, #25
        min_confident_area=350, #800
        min_confident_dim=8, #10
        min_ink_pixels=80, #150
        min_fill_ratio=0.08, #0.08
        row_y_tolerance=18, #18
    ):

        self.min_width = min_width
        self.min_height = min_height
        self.padding = padding
        self.merge_dist = merge_dist

        self.min_confident_area = min_confident_area
        self.min_confident_dim = min_confident_dim
        self.min_ink_pixels = min_ink_pixels
        self.min_fill_ratio = min_fill_ratio

        self.refine_height = 140          # Only refine tall ROIs
        self.valley_threshold = 0.15      # 15% of max projection
        self.min_valley_width = 5         # Consecutive empty rows
        self.min_segment_height = 20      # Prevent tiny fragments
        self.row_y_tolerance = row_y_tolerance
    

    def _horizontal_projection(self, roi):

        return np.count_nonzero(
            roi,
            axis=1
        )
    
    def _smooth_projection(self, proj):

        kernel = np.ones(5) / 5

        return np.convolve(
            proj,
            kernel,
            mode="same"
        )
    

    def _find_split_rows(
        self,
        projection
    ):

        threshold = 0.15 * projection.max()

        valleys = []

        start = None

        for i, v in enumerate(projection):

            if v < threshold:

                if start is None:
                    start = i

            else:

                if start is not None:

                    if i - start >= 4:

                        valleys.append(
                            (start + i)//2
                        )

                    start = None

        return valleys
    

    def _refine_rois(self, merged, binary):
        """
        Split vertically merged ROIs using horizontal projection.

        Runs ONLY on unusually tall ROIs.
        """

        refined = []

        for (x1, y1, x2, y2) in merged:

            roi_h = y2 - y1

            # Small ROIs are assumed correct
            if roi_h < self.refine_height:
                refined.append((x1, y1, x2, y2))
                continue

            roi = binary[y1:y2, x1:x2]

            # Horizontal projection
            projection = np.count_nonzero(roi, axis=1).astype(np.float32)

            # Smooth projection
            projection = cv2.GaussianBlur(
                projection.reshape(-1, 1),
                (1, 9),
                0
            ).flatten()

            threshold = projection.max() * self.valley_threshold

            valleys = []

            start = None

            for i, value in enumerate(projection):

                if value < threshold:

                    if start is None:
                        start = i

                else:

                    if start is not None:

                        if (i - start) >= self.min_valley_width:
                            valleys.append((start + i) // 2)

                        start = None

            # No valid valleys
            if len(valleys) == 0:
                refined.append((x1, y1, x2, y2))
                continue

            prev = 0
            pieces = []

            for split in valleys:

                if split - prev >= self.min_segment_height:
                    pieces.append(
                        (
                            x1,
                            y1 + prev,
                            x2,
                            y1 + split
                        )
                    )

                prev = split

            if roi_h - prev >= self.min_segment_height:
                pieces.append(
                    (
                        x1,
                        y1 + prev,
                        x2,
                        y2
                    )
                )

            # Safety check
            if len(pieces) <= 1:
                refined.append((x1, y1, x2, y2))
            else:
                refined.extend(pieces)

        return refined

    # --------------------------------------------------

    def remove_lines(self, binary):

        h, w = binary.shape


#        horiz_kernel = cv2.getStructuringElement(
#            cv2.MORPH_RECT,
#            (max(20, w // 15), 1)
#        )

#        vert_kernel = cv2.getStructuringElement(
#            cv2.MORPH_RECT,
#            (1, max(20, h // 15))
#        ) 

        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(30, w // 4), 1)
        )

        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(30, h // 4))
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

    """def _boxes_close(self, a, b):

        #Decide whether two connected components belong to the same
        #handwritten text instance.

        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        aw = ax2 - ax1
        ah = ay2 - ay1

        bw = bx2 - bx1
        bh = by2 - by1

        # ----------------------------
        # 1. Horizontal gap
        # ----------------------------
        if ax2 < bx1:
            hgap = bx1 - ax2
        elif bx2 < ax1:
            hgap = ax1 - bx2
        else:
            hgap = 0

        # ----------------------------
        # 2. Vertical overlap
        # ----------------------------
        overlap = min(ay2, by2) - max(ay1, by1)

        if overlap <= 0:
            return False

        overlap_ratio = overlap / float(min(ah, bh))

        # ----------------------------
        # 3. Adaptive horizontal gap
        # ----------------------------
        max_gap = max(
            self.merge_dist,
            int(0.5 * max(aw, bw))
        )

        # ----------------------------
        # 4. Merge decision
        # ----------------------------
        return (
            overlap_ratio >= 0.40 and
            hgap <= max_gap
        ) """

    # The above function was perfect for numbers but broke on roman numerals 
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


    def _group_boxes_by_y(self, boxes):
        """
        Group boxes that belong to the same horizontal writing row.

        Each box is represented as:
            (x1, y1, x2, y2)

        Boxes are grouped using their vertical center.
        """

        rows = []

        # Process from top to bottom
        boxes = sorted(
            boxes,
            key=lambda b: (b[1] + b[3]) / 2
        )

        for box in boxes:

            x1, y1, x2, y2 = box

            cy = (y1 + y2) / 2

            assigned = False

            for row in rows:

                # Current average Y position of this row
                row_cy = row["center_y"]

                if abs(cy - row_cy) <= self.row_y_tolerance:

                    row["boxes"].append(box)

                    # Recalculate row center
                    centers = [
                        (b[1] + b[3]) / 2
                        for b in row["boxes"]
                    ]

                    row["center_y"] = float(
                        np.mean(centers)
                    )

                    assigned = True
                    break

            if not assigned:

                rows.append({
                    "center_y": cy,
                    "boxes": [box]
                })

        # ---------------------------------------------
        # Convert each row into one bounding box
        # ---------------------------------------------

        row_boxes = []

        for row in rows:

            boxes_in_row = row["boxes"]

            x1 = min(
                b[0]
                for b in boxes_in_row
            )

            y1 = min(
                b[1]
                for b in boxes_in_row
            )

            x2 = max(
                b[2]
                for b in boxes_in_row
            )

            y2 = max(
                b[3]
                for b in boxes_in_row
            )

            row_boxes.append(
                (x1, y1, x2, y2)
            )

        return row_boxes


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

        merged = self._refine_rois(
            merged,
            binary
        )

        confident = []

        for x1, y1, x2, y2 in merged:
            w = x2 - x1
            h = y2 - y1

            bbox_area = w * h

            ink_pixels = cv2.countNonZero(
                binary[y1:y2, x1:x2]
            )

            fill_ratio = ink_pixels / float(bbox_area)

            print(
                f"\nCandidate:"
                f" bbox={w}x{h}"
                f" | bbox_area={bbox_area}"
                f" | ink_pixels={ink_pixels}"
                f" | fill_ratio={fill_ratio:.3f}"
            )

            if bbox_area < self.min_confident_area:
                print(
                    f"  REJECT → bbox_area "
                    f"{bbox_area} < {self.min_confident_area}"
                )
                continue

            if w < self.min_confident_dim:
                print(
                    f"  REJECT → width "
                    f"{w} < {self.min_confident_dim}"
                )
                continue

            if h < self.min_confident_dim:
                print(
                    f"  REJECT → height "
                    f"{h} < {self.min_confident_dim}"
                )
                continue

            if ink_pixels < self.min_ink_pixels:
                print(
                    f"  REJECT → ink_pixels "
                    f"{ink_pixels} < {self.min_ink_pixels}"
                )
                continue

            if fill_ratio < self.min_fill_ratio:
                print(
                    f"  REJECT → fill_ratio "
                    f"{fill_ratio:.3f} < {self.min_fill_ratio}"
                )
                continue

            print("  ACCEPT ✓")

            confident.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

            confident.append(
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
            )

        # ==================================================
        # GROUP CONFIDENT ROIs BY HORIZONTAL ROW
        # ==================================================

        grouped = self._group_boxes_by_y(confident)

        # ==================================================
        # FINAL ROI CREATION
        # ==================================================

        detected = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        rois = []

        for x1, y1, x2, y2 in grouped:

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