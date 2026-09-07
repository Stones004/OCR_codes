import cv2
import numpy as np


"""
Separator Detector

Identifies the vertical separator line from the extracted vertical
structures using connected component analysis and geometric filtering.
The detected separator is later used to isolate the annotation margin.
"""

class LineDetector:

    def __init__(
        self,
        left_ratio=0.03,
        right_ratio=0.30,
        expected_x=None,
        tolerance=None
    ):
        """
        Parameters
        ----------
        left_ratio : float
            Ignore detections too close to the left page border.

        right_ratio : float
            Ignore detections beyond this percentage of the page width.

        expected_x : int | None
            Expected x-coordinate of the separator.

        tolerance : int | None
            Allowed deviation from expected_x.
        """

        self.left_ratio = left_ratio
        self.right_ratio = right_ratio

        self.expected_x = expected_x
        self.tolerance = tolerance

    def detect(self, vertical):

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            vertical,
            connectivity=8
        )

        h, w = vertical.shape

        LEFT_LIMIT = int(w * self.left_ratio)
        RIGHT_LIMIT = int(w * self.right_ratio)

        output = cv2.cvtColor(vertical, cv2.COLOR_GRAY2BGR)

        candidates = []

        for i in range(1, num_labels):

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # -----------------------------
            # Basic geometric filtering
            # -----------------------------

            reason = None

            if height < h * 0.50:
                reason = f"Too short ({height}px)"

            elif ( height / width) < 40:
                reason = f"Poor aspect ratio ({(height / width):.1f})"

            elif x < LEFT_LIMIT:
                reason = f"Too close to left border (x={x})"

            elif x > RIGHT_LIMIT:
                reason = f"Outside left half (x={x})"

            if reason is not None:
                print(
                    f"Rejected Component {i} | "
                    f"x={x}, y={y}, w={width}, h={height} | "
                    f"Reason: {reason}"
                )

                continue

            component_mask = np.zeros_like(vertical)
            component_mask[labels == i] = 255

            points = np.column_stack(np.where(component_mask > 0))

            candidate = {

                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "area": area,

                "mask": component_mask,
                "points": points
            }

            candidates.append(candidate)

            # Draw every valid candidate
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            print("\nValid Candidates")

            for c in candidates:
                print(
                    f"x={c['x']}, "
                    f"h={c['h']}, "
                    f"w={c['w']}, "
                    f"area={c['area']}"
                )

        # ------------------------------------------------
        # No candidates
        # ------------------------------------------------

        if len(candidates) == 0:
            return output, None

        # ------------------------------------------------
        # If we know approximately where the separator is,
        # choose the closest candidate.
        # ------------------------------------------------

        if self.expected_x is not None:

            if self.tolerance is not None:

                filtered = []

                for c in candidates:

                    if abs(c["x"] - self.expected_x) <= self.tolerance:
                        filtered.append(c)

                if len(filtered) > 0:
                    candidates = filtered

            best = min(
                candidates,
                key=lambda c: abs(c["x"] - self.expected_x)
            )

        else:

            # First run:
            # choose the left-most candidate.

            best = min(
                candidates,
                key=lambda c: c["x"]
            )



        # Draw best candidate in RED
        cv2.rectangle(
            output,
            (best["x"], best["y"]),
            (best["x"] + best["w"], best["y"] + best["h"]),
            (0, 0, 255),
            4
        )

        print("\nSelected Candidate")
        print(best)

        return output, best