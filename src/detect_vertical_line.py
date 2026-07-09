import cv2
import numpy as np

def detect_vertical_lines(image):

    output = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Slight blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(
        gray,
        50,
        150,
        apertureSize=3
    )

    # Detect line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=image.shape[0] // 2,
        maxLineGap=20
    )

    vertical_lines = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

            # Near vertical
            if 85 <= angle <= 95:

                vertical_lines.append((x1, y1, x2, y2))

                cv2.line(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    3
                )

    return output, vertical_lines