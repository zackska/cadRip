import matplotlib.pyplot as plt
import easyocr
import cv2
import re
from pdf2image import convert_from_path
from collections import defaultdict
from sklearn.cluster import DBSCAN
import pdfplumber
import numpy as np
import os

def ripSymbolDims(
    file_path,
    template_path,
    img_end=1.0,
    font_size=0.1,
    plot=False,
    ocr_langs=['en'],
    dpi_pdf=800,
    pause=5,
    match_angle_tol=10,
    n_matches=200,
    min_conf=0.5,
    max_distance=200,
    numeric_ratio=0.6,
    results_dir="results"
):
    """
    Generalized function to extract numerical values from an engineering drawing using a drawing symbol template matching and OCR.

    Args:
        file_path (str): Path to the input PDF file.
        template_path (str): Path to the symbol template image.
        img_end (float): Fraction of image height to keep (0-1).
        font_size (float): Relative font size (in inches, for scaling template).
        plot (bool): Whether to plot and save match visualizations.
        ocr_langs (list): Languages for OCR.
        dpi_pdf (int): DPI for PDF to image conversion.
        pause (int): Pause duration for plots (if shown).
        match_angle_tol (float): Angle tolerance for SIFT match filtering.
        n_matches (int): Number of matches to draw in visualization.
        min_conf (float): Minimum OCR confidence to accept.
        max_distance (float): Maximum SIFT match distance to accept.
        numeric_ratio (float): Minimum ratio of numeric chars in OCR result.
        results_dir (str): Directory to save results.

    Returns:
        list: Extracted dimensional strings.
    """
    diameters = []
    filename = os.path.splitext(os.path.basename(file_path))[0]
    images = convert_from_path(file_path, dpi=dpi_pdf)
    img = np.array(images[0])[:, :, ::-1]
    img_end = int(img_end * img.shape[0])
    img = img[:img_end, :, :]

    symbol_template = cv2.imread(template_path, 0)
    orig_h, orig_w = symbol_template.shape
    desired_h = int(dpi_pdf * font_size)
    scale_factor = desired_h / orig_h
    new_w = int(orig_w * scale_factor)
    symbol_template = cv2.resize(
        symbol_template,
        (new_w, desired_h),
        interpolation=cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC,
    )

    sift = cv2.SIFT_create(contrastThreshold=0.001)
    kp1, des1 = sift.detectAndCompute(symbol_template, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    filtered_matches = [
        m
        for m in matches
        if abs(kp2[m.trainIdx].angle - kp1[m.queryIdx].angle) < match_angle_tol
    ]
    matches = filtered_matches

    if plot:
        img_matches = cv2.drawMatches(
            symbol_template,
            kp1,
            img,
            kp2,
            matches[:n_matches],
            None,
            matchColor=(255, 0, 0),
            singlePointColor=None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        os.makedirs(results_dir, exist_ok=True)
        plt.figure(figsize=(15, 10))
        plt.imshow(img_matches)
        plt.title(f"SIFT Matches (Top {n_matches})")
        plt.axis("off")
        plt.savefig(os.path.join(results_dir, f"Diams_{filename}.png"), bbox_inches="tight")
        plt.close()

    reader = easyocr.Reader(ocr_langs, gpu=True)
    found_diams = []
    h, w = symbol_template.shape
    for m in matches:
        x, y = map(int, kp2[m.trainIdx].pt)
        tx, ty = kp1[m.queryIdx].pt
        symbol_size = dpi_pdf * font_size
        roi_x = x - tx + symbol_size * (w / h)
        roi_y = y - ty + symbol_size / 2
        roi = img[
            int(roi_y - symbol_size) : int(roi_y + symbol_size),
            int(roi_x) : int(roi_x + 4 * symbol_size * (w / h)),
        ]
        result = reader.readtext(roi, allowlist="0123456789./")
        if result:
            text, conf = result[0][1], result[0][2]
            if plot:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(symbol_template, cmap="gray")
                axs[0].scatter([tx], [ty], c="red", s=80, label="Template (tx, ty)")
                axs[0].set_title("Template")
                axs[0].legend()
                axs[0].axis("off")
                axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[1].scatter([x], [y], c="blue", s=80, label="Image (x, y)")
                axs[1].scatter([roi_x], [roi_y], c="green", s=80, label="ROI (roi_x, roi_y)")
                rect = plt.Rectangle(
                    (roi_x, roi_y - symbol_size / 2),
                    int(4 * symbol_size * (w / h)),
                    int(symbol_size),
                    linewidth=2,
                    edgecolor="lime",
                    facecolor="none",
                    label="ROI Box",
                )
                axs[1].add_patch(rect)
                axs[1].set_title("Image")
                axs[1].legend()
                axs[1].axis("off")
                pad = symbol_size * 2
                x_min = roi_x - pad
                x_max = roi_x + int(4 * symbol_size * (w / h)) + pad
                y_min = roi_y - pad
                y_max = roi_y + pad
                axs[1].set_xlim(x_min, x_max)
                axs[1].set_ylim(y_max, y_min)
                axs[0].set_box_aspect(axs[1].get_position().height / axs[0].get_position().height)
                plt.title(f'Diameter: "{text.strip()}" | Confidence: {conf:.2f}')
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        results_dir,
                        f"match_{filename}_{text.strip().replace('/', '_').replace('.', '_')}.png",
                    ),
                    bbox_inches="tight",
                )
                plt.close()
            for (bbox, text, conf) in result:
                x0, y0 = x, y
                found = False
                for entry in found_diams:
                    if np.hypot(entry["x"] - x, entry["y"] - y) < symbol_size:
                        found = True
                        if conf > entry["conf"]:
                            entry.update(
                                {
                                    "text": text.strip(),
                                    "conf": conf,
                                    "roi": roi,
                                    "x": x0,
                                    "y": y0,
                                    "distance": m.distance,
                                }
                            )
                        break
                if not found:
                    numeric_count = sum(1 for c in text if c.isdigit() or c in ". /")
                    if (
                        len(text) > 0
                        and numeric_count / len(text) > numeric_ratio
                        and m.distance < max_distance
                        and conf > min_conf
                    ):
                        found_diams.append(
                            {
                                "text": text.strip(),
                                "conf": conf,
                                "roi": roi,
                                "x": x0,
                                "y": y0,
                                "distance": m.distance,
                            }
                        )
    diameters = [entry["text"] for entry in found_diams]
    return diameters

def add_frac_to_prev(match):
    prev_num = match.group(1)
    frac = match.group(2)
    num, denom = frac.split("/")
    decimal = round(float(num) / float(denom), 4)
    if prev_num:
        total = round(float(prev_num) + decimal, 4)
        return str(total)
    else:
        return str(decimal)

def group_words_by_line(words, y_tol=1, x_gap_tol=15):
    """
    Groups words into lines based on vertical and horizontal proximity.

    Args:
        words (list): List of words from pdfplumber.extract_words().
        y_tol (float): Tolerance for grouping words into the same line (by vertical alignment).
        x_gap_tol (float): Maximum horizontal gap between words to consider them part of the same phrase.

    Returns:
        List of grouped line strings.
    """
    lines = defaultdict(list)
    for word in words:
        y_key = round(word["top"] / y_tol)
        lines[y_key].append(word)
    lines = {k: v for k, v in sorted(lines.items(), key=lambda item: item[0])}
    grouped_lines = []
    for _, line_words in sorted(lines.items()):
        line_words.sort(key=lambda w: w["x0"])
        current_line = []
        last_x1 = None
        temp_group = []
        for word in line_words:
            x0 = word["x0"]
            if last_x1 is not None and x0 - last_x1 > x_gap_tol:
                if temp_group:
                    current_line.append(" ".join(w["text"] for w in temp_group))
                temp_group = []
            temp_group.append(word)
            last_x1 = word["x1"]
        if temp_group:
            current_line.append(" ".join(w["text"] for w in temp_group))
        grouped_lines.append(" | ".join(current_line))
    return grouped_lines

def cluster_words(words, eps=10, y_tol=3, x_gap_tol=15):
    """
    Cluster words using both x and y coordinates (DBSCAN or custom clustering).

    Args:
        words (list): List of word dicts.
        eps (float): DBSCAN epsilon.
        y_tol (float): y tolerance for grouping.
        x_gap_tol (float): x gap tolerance for grouping.

    Returns:
        list: Grouped line strings.
    """
    if not words:
        return []
    coords = np.array([[w["x0"], w["top"]] for w in words])
    db = DBSCAN(eps=eps, min_samples=1).fit(coords)
    labels = db.labels_
    clusters = defaultdict(list)
    for label, word in zip(labels, words):
        clusters[label].append(word)
    grouped_lines = []
    for cluster_words_ in clusters.values():
        cluster_words_.sort(key=lambda w: w["x0"])
        cluster_words_ = group_words_by_line(cluster_words_, y_tol=y_tol, x_gap_tol=x_gap_tol)
        grouped_lines.append(" | ".join(cluster_words_))
    return grouped_lines

def ripNumericDims(
    pdf_path,
    description_keyword="description",
    cluster_eps=20,
    y_tol=3,
    x_gap_tol=15,
    font_sample_chars=15,
):
    """
    Extracts dimension lines from a PDF using pdfplumber.

    Args:
        pdf_path (str): Path to PDF.
        description_keyword (str): Keyword to find max_y.
        cluster_eps (float): DBSCAN epsilon for clustering.
        y_tol (float): y tolerance for grouping.
        x_gap_tol (float): x gap tolerance for grouping.
        font_sample_chars (int): Number of chars to sample for font size.

    Returns:
        tuple: (lines, max_y/page.height, font_size/72)
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        text = page.extract_words()
        page.chars.sort(key=lambda c: c["y0"])
        font_size = np.mean([char["size"] for char in page.chars[:font_sample_chars]])
        max_y = None
        for word in text:
            if description_keyword.lower() in word["text"].lower():
                max_y = word["top"] - font_size
                break
        text = [word for word in text if max_y is None or word["top"] <= max_y]
        lines = cluster_words(text, eps=cluster_eps, y_tol=y_tol, x_gap_tol=x_gap_tol)
        return lines, (max_y / page.height if max_y else 1.0), font_size / 72

def cleanDims(lines, numeric_threshold=0.8):
    """
    Extracts numeric dimension lines and converts fractions to decimals.

    Args:
        lines (list): List of line strings.
        numeric_threshold (float): Minimum ratio of numeric chars.

    Returns:
        list: List of numeric values.
    """
    lines = [line.replace('"', "") for line in lines]
    numeric_lines = [
        line
        for line in lines
        if line and all(c.isdigit() or c in ". /" for c in line)
    ]
    numeric_strings = []
    for line in numeric_lines:
        if "X" not in line:
            split = re.split(r"\s*[\+\-±]\s*", line)
            line = split[0]
            if "/" in line and " " in line:
                pattern = r"(\d+)\s+(\d+/\d+)"
                line = re.sub(pattern, add_frac_to_prev, line)
            elif "/" in line:
                num, denom = line.strip().split("/")
                line = float(num) / float(denom)
            numeric_strings.append(float(line))
    return numeric_strings
