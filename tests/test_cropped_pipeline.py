from pathlib import Path
import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

# Change this to the cropped folder you want to inspect.
INPUT_DIR = Path(
    r"G:\OCR_codes\outputs\Doc0512\cropped"
)

# All diagnostic images will be written here.
OUTPUT_DIR = Path(
    r"test_outputs/Doc0512/cropped_diagnostics"
)

# Test multiple min-area values so we can see
# exactly how aggressive 200 is.
MIN_AREAS = [0, 20, 50, 100, 150, 200]

# Detector-like minimum dimensions.
MIN_WIDTH = 6
MIN_HEIGHT = 6

# Final detector settings from the current implementation.
FINAL_MIN_AREA = 800
FINAL_MIN_DIM = 20
FINAL_MIN_INK = 150
FINAL_MIN_FILL = 0.08


# ============================================================
# UTILITY
# ============================================================

def save(name, image):
    path = OUTPUT_DIR / name
    cv2.imwrite(str(path), image)
    print(f"  Saved: {path}")


def count_ink(image):
    return cv2.countNonZero(image)


def make_overlay(gray, boxes, color=(0, 0, 255), thickness=2):
    """
    Draw bounding boxes on grayscale image.
    """
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for x, y, w, h in boxes:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )

    return output


# ============================================================
# 1. LOAD IMAGE
# ============================================================

def process_image(image_path):

    print("\n" + "=" * 80)
    print(f"PROCESSING: {image_path.name}")
    print("=" * 80)

    img = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR
    )

    if img is None:
        print("ERROR: Could not read image.")
        return

    # --------------------------------------------------------
    # Create output folder for this page
    # --------------------------------------------------------

    page_name = image_path.stem

    page_output = OUTPUT_DIR / page_name
    page_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # STEP 1 — GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    save(
        f"{page_name}/01_gray.png",
        gray
    )

    print("\n[1] GRAY")
    print(f"    Shape      : {gray.shape}")
    print(f"    Min pixel  : {gray.min()}")
    print(f"    Max pixel  : {gray.max()}")
    print(f"    Mean pixel : {gray.mean():.2f}")

    # ========================================================
    # STEP 2 — OTSU
    # ========================================================

    otsu_threshold, raw_otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    save(
        f"{page_name}/02_raw_otsu.png",
        raw_otsu
    )

    print("\n[2] RAW OTSU")
    print(
        f"    Otsu threshold : {otsu_threshold:.2f}"
    )

    raw_ink = count_ink(raw_otsu)

    print(
        f"    Foreground pixels : {raw_ink}"
    )

    # ========================================================
    # STEP 3 — CONNECTED COMPONENT ANALYSIS
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            raw_otsu,
            connectivity=8
        )
    )

    print("\n[3] RAW CONNECTED COMPONENTS")

    print(
        f"    Total components : {num_labels - 1}"
    )

    components = []

    for i in range(1, num_labels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]

        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]

        components.append({
            "label": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })

    components.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    # --------------------------------------------------------
    # Print largest components
    # --------------------------------------------------------

    print("\n    Largest components:")

    for c in components[:20]:

        print(
            f"      label={c['label']:4d} "
            f"area={c['area']:6d} "
            f"bbox={c['w']:4d}x{c['h']:<4d}"
        )

    # ========================================================
    # STEP 4 — VISUALIZE ALL COMPONENTS
    # ========================================================

    all_components = np.zeros_like(
        raw_otsu
    )

    for c in components:

        label = c["label"]

        all_components[
            labels == label
        ] = 255

    save(
        f"{page_name}/03_all_components.png",
        all_components
    )

    # ========================================================
    # STEP 5 — TEST MIN_AREA
    # ========================================================

    print("\n[4] MIN_AREA EXPERIMENT")

    for min_area in MIN_AREAS:

        clean = np.zeros_like(
            raw_otsu
        )

        kept = []
        removed = []

        for c in components:

            if min_area == 0:
                keep = True
            else:
                keep = (
                    c["area"] >= min_area
                )

            if keep:

                clean[
                    labels == c["label"]
                ] = 255

                kept.append(c)

            else:
                removed.append(c)

        filename = (
            f"{page_name}/"
            f"04_min_area_{min_area:03d}.png"
        )

        save(
            filename,
            clean
        )

        print(
            f"    min_area={min_area:3d} | "
            f"kept={len(kept):4d} | "
            f"removed={len(removed):4d} | "
            f"ink={count_ink(clean):8d}"
        )

    # ========================================================
    # STEP 6 — SHOW EXACTLY WHAT min_area=200 REMOVES
    # ========================================================

    min_area = 200

    kept = []
    removed = []

    for c in components:

        if c["area"] >= min_area:
            kept.append(c)
        else:
            removed.append(c)

    removed_image = np.zeros_like(
        raw_otsu
    )

    for c in removed:

        removed_image[
            labels == c["label"]
        ] = 255

    save(
        f"{page_name}/05_removed_by_min_area_200.png",
        removed_image
    )

    print("\n[5] WHAT min_area=200 REMOVES")

    print(
        f"    Removed components : {len(removed)}"
    )

    print(
        f"    Removed ink pixels  : "
        f"{count_ink(removed_image)}"
    )

    print("\n    Small components removed by 200:")

    for c in removed[:50]:

        print(
            f"      label={c['label']:4d} "
            f"area={c['area']:6d} "
            f"bbox={c['w']:4d}x{c['h']:<4d}"
        )

    # ========================================================
    # STEP 7 — VISUALIZE COMPONENT BOUNDING BOXES
    # ========================================================

    raw_boxes = [
        (
            c["x"],
            c["y"],
            c["w"],
            c["h"]
        )
        for c in components
    ]

    overlay = make_overlay(
        gray,
        raw_boxes
    )

    save(
        f"{page_name}/06_all_component_boxes.png",
        overlay
    )

    # ========================================================
    # STEP 8 — LINE REMOVAL
    # ========================================================

    print("\n[6] LINE REMOVAL")

    # Use the 200-filtered binary as the input,
    # matching the current detector architecture.

    binary_200 = np.zeros_like(
        raw_otsu
    )

    for c in kept:

        binary_200[
            labels == c["label"]
        ] = 255

    h, w = binary_200.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(20, w // 15),
            1
        )
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            1,
            max(20, h // 15)
        )
    )

    horizontal_lines = cv2.morphologyEx(
        binary_200,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )

    vertical_lines = cv2.morphologyEx(
        binary_200,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    line_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines
    )

    line_mask = cv2.dilate(
        line_mask,
        np.ones(
            (3, 3),
            np.uint8
        ),
        iterations=1
    )

    after_line_removal = cv2.bitwise_and(
        binary_200,
        cv2.bitwise_not(line_mask)
    )

    save(
        f"{page_name}/07_line_mask.png",
        line_mask
    )

    save(
        f"{page_name}/08_after_line_removal.png",
        after_line_removal
    )

    print(
        f"    Before line removal : "
        f"{count_ink(binary_200)} ink pixels"
    )

    print(
        f"    Line mask           : "
        f"{count_ink(line_mask)} pixels"
    )

    print(
        f"    After line removal  : "
        f"{count_ink(after_line_removal)} ink pixels"
    )

    # ========================================================
    # STEP 9 — COMPONENTS AFTER LINE REMOVAL
    # ========================================================

    num_after, labels_after, stats_after, _ = (
        cv2.connectedComponentsWithStats(
            after_line_removal,
            connectivity=8
        )
    )

    boxes_after = []

    for i in range(1, num_after):

        x = stats_after[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats_after[
            i,
            cv2.CC_STAT_TOP
        ]

        ww = stats_after[
            i,
            cv2.CC_STAT_WIDTH
        ]

        hh = stats_after[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats_after[
            i,
            cv2.CC_STAT_AREA
        ]

        if ww < MIN_WIDTH:
            continue

        if hh < MIN_HEIGHT:
            continue

        boxes_after.append(
            (
                x,
                y,
                ww,
                hh,
                area
            )
        )

    print("\n[7] COMPONENTS AFTER LINE REMOVAL")

    print(
        f"    Total components : "
        f"{num_after - 1}"
    )

    print(
        f"    Candidate boxes  : "
        f"{len(boxes_after)}"
    )

    overlay_after_lines = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    for x, y, ww, hh, area in boxes_after:

        cv2.rectangle(
            overlay_after_lines,
            (x, y),
            (x + ww, y + hh),
            (0, 255, 255),
            2
        )

    save(
        f"{page_name}/09_after_line_removal_boxes.png",
        overlay_after_lines
    )

    # ========================================================
    # STEP 10 — TEST FINAL DETECTOR FILTERS
    # ========================================================

    print("\n[8] FINAL FILTER TEST")

    final_boxes = []

    rejected_area = 0
    rejected_dim = 0
    rejected_ink = 0
    rejected_fill = 0

    for x, y, ww, hh, area in boxes_after:

        bbox_area = ww * hh

        if bbox_area < FINAL_MIN_AREA:
            rejected_area += 1
            continue

        if ww < FINAL_MIN_DIM or hh < FINAL_MIN_DIM:
            rejected_dim += 1
            continue

        ink_pixels = cv2.countNonZero(
            after_line_removal[
                y:y + hh,
                x:x + ww
            ]
        )

        if ink_pixels < FINAL_MIN_INK:
            rejected_ink += 1
            continue

        fill_ratio = (
            ink_pixels /
            float(bbox_area)
        )

        if fill_ratio < FINAL_MIN_FILL:
            rejected_fill += 1
            continue

        final_boxes.append(
            (
                x,
                y,
                ww,
                hh
            )
        )

    print(
        f"    Rejected by bbox area : "
        f"{rejected_area}"
    )

    print(
        f"    Rejected by dimension : "
        f"{rejected_dim}"
    )

    print(
        f"    Rejected by ink       : "
        f"{rejected_ink}"
    )

    print(
        f"    Rejected by fill      : "
        f"{rejected_fill}"
    )

    print(
        f"    FINAL BOXES           : "
        f"{len(final_boxes)}"
    )

    # ========================================================
    # STEP 11 — FINAL DETECTION VISUALIZATION
    # ========================================================

    final_overlay = make_overlay(
        gray,
        final_boxes
    )

    save(
        f"{page_name}/10_final_detector.png",
        final_overlay
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    print(
        f"Gray foreground information : "
        f"{np.sum(gray < 200)}"
    )

    print(
        f"Raw Otsu ink                : "
        f"{count_ink(raw_otsu)}"
    )

    print(
        f"Components after Otsu       : "
        f"{len(components)}"
    )

    print(
        f"Components surviving 200    : "
        f"{len(kept)}"
    )

    print(
        f"Components after lines      : "
        f"{len(boxes_after)}"
    )

    print(
        f"Final detector boxes        : "
        f"{len(final_boxes)}"
    )

    print("-" * 80)

    print(
        "\nIMPORTANT FILES TO INSPECT:"
    )

    print(
        f"  01_gray.png"
    )

    print(
        f"  02_raw_otsu.png"
    )

    print(
        f"  05_removed_by_min_area_200.png"
    )

    print(
        f"  07_line_mask.png"
    )

    print(
        f"  08_after_line_removal.png"
    )

    print(
        f"  09_after_line_removal_boxes.png"
    )

    print(
        f"  10_final_detector.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted(
        [
            p
            for p in INPUT_DIR.iterdir()
            if p.suffix.lower()
            in {
                ".png",
                ".jpg",
                ".jpeg",
                ".tif",
                ".tiff"
            }
        ]
    )

    if not images:

        print(
            f"No images found in:\n"
            f"{INPUT_DIR}"
        )

        return

    print("=" * 80)
    print("CROPPED IMAGE OCR PIPELINE DIAGNOSTIC")
    print("=" * 80)

    print(
        f"Input : {INPUT_DIR}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        f"Images: {len(images)}"
    )

    for image_path in images:

        process_image(
            image_path
        )

    print("\n")
    print("=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()