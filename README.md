# CadRip.py

## Overview

`CadRip.py` provides a set of functions for extracting and processing dimensional information from engineering drawings, particularly from PDF files and scanned images. The code combines computer vision (SIFT feature matching), OCR (using EasyOCR), and PDF text extraction (using pdfplumber) to automate the retrieval of diameter and numeric dimension values from technical drawings.

## Features

- **Symbol-based Dimension Extraction:**  
  Uses SIFT feature matching to locate symbol templates (e.g., diameter symbols) in drawings and OCR to extract associated numeric values.

- **PDF Text Extraction:**  
  Extracts and clusters text from PDF drawings to identify and group dimension lines.

- **Fraction Handling:**  
  Converts fractional dimensions (e.g., `1 1/2`) to decimal values.

- **Clustering and Grouping:**  
  Groups words into lines and clusters using spatial proximity for robust parsing of engineering drawing layouts.

- **Visualization:**  
  Optionally saves annotated images showing detected symbols, matches, and OCR results.

## Main Functions

### `ripSymbolDims(...)`

Extracts diameter (or other symbol-based) dimensions from a PDF drawing using a symbol template and OCR.

**Arguments:**
- `file_path`: Path to the input PDF file.
- `template_path`: Path to the symbol template image.
- `img_end`: Fraction of image height to keep.
- `font_size`: Relative font size for scaling template.
- `plot`: Whether to plot/save match visualizations.
- `ocr_langs`: Languages for OCR.
- `dpi_pdf`: DPI for PDF to image conversion.
- `pause`: Pause duration for plots.
- `match_angle_tol`: Angle tolerance for SIFT match filtering.
- `n_matches`: Number of matches to draw in visualization.
- `min_conf`: Minimum OCR confidence to accept.
- `max_distance`: Maximum SIFT match distance to accept.
- `numeric_ratio`: Minimum ratio of numeric chars in OCR result.
- `results_dir`: Directory to save results.

**Returns:**  
List of extracted dimensional strings.

---

### `ripNumericDims(...)`

Extracts dimension lines from a PDF using pdfplumber and clusters/grouping.

**Arguments:**
- `pdf_path`: Path to PDF.
- `description_keyword`: Keyword to find the end of the dimension block.
- `cluster_eps`: DBSCAN epsilon for clustering.
- `y_tol`, `x_gap_tol`: Tolerances for grouping.
- `font_sample_chars`: Number of chars to sample for font size.

**Returns:**  
Tuple of (lines, max_y/page.height, font_size/72).

---

### `cleanDims(...)`

Cleans and converts extracted lines to numeric values, handling fractions.

**Arguments:**
- `lines`: List of line strings.
- `numeric_threshold`: Minimum ratio of numeric chars.

**Returns:**  
List of numeric values.

---

### Utility Functions

- `add_frac_to_prev(match)`: Converts mixed fractions to decimals.
- `group_words_by_line(words, ...)`: Groups words into lines.
- `cluster_words(words, ...)`: Clusters words spatially.

## Dependencies

- `opencv-python`
- `matplotlib`
- `easyocr`
- `pdf2image`
- `pdfplumber`
- `numpy`
- `scikit-learn`

## Example Usage

```python
from CadRip import ripSymbolDims, ripNumericDims, cleanDims

# Extract symbol-based dimensions
diams = ripSymbolDims("drawing.pdf", "diameter_template.png", plot=True)

# Extract numeric dimensions from PDF text
lines, _, _ = ripNumericDims("drawing.pdf")
numeric_dims = cleanDims(lines)
