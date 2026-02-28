"""
Handwriting OCR Module
Primary: EasyOCR with bounding-box line grouping — correctly separates each line
Fallback: TrOCR on individual detected lines
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class HandwritingOCR:
    def __init__(self, use_trocr=False, use_easyocr_fallback=True):
        # EasyOCR is primary — it returns bounding boxes so we can group lines properly
        self.use_trocr = use_trocr
        self.easyocr_reader = None
        self.trocr_processor = None
        self.trocr_model = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return

        print("[OCR] Loading models...")

        # Always load EasyOCR as primary
        try:
            import easyocr
            print("[OCR] Loading EasyOCR (primary)...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("[OCR] EasyOCR loaded ✅")
        except Exception as e:
            print(f"[OCR] EasyOCR failed: {e}")

        # Optionally load TrOCR for line-level refinement
        if self.use_trocr:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                print("[OCR] Loading TrOCR for line refinement...")
                self.trocr_processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-large-handwritten"
                )
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-large-handwritten"
                )
                print("[OCR] TrOCR loaded ✅")
            except Exception as e:
                print(f"[OCR] TrOCR skipped: {e}")
                self.use_trocr = False

        self._loaded = True

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Resize if too small — OCR needs decent resolution
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_CUBIC)

        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive threshold — handles uneven lighting on handwritten papers
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )

        # Deskew if needed
        thresh = self._deskew(thresh)

        return thresh

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Correct slight rotation in scanned papers"""
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 10:  # Don't correct large rotations — likely wrong
            return img
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _group_words_into_lines(self, ocr_results: list, y_tolerance: int = 15) -> list:
        """
        Group EasyOCR word detections into lines based on Y position.
        Each result is: (bbox, text, confidence)
        bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        Returns list of lines sorted top to bottom, words sorted left to right.
        """
        if not ocr_results:
            return []

        # Extract word info: (y_center, x_center, text, confidence)
        words = []
        for bbox, text, conf in ocr_results:
            if conf < 0.1 or not text.strip():
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x_center = (min(xs) + max(xs)) / 2
            y_center = (min(ys) + max(ys)) / 2
            words.append((y_center, x_center, text.strip(), conf))

        if not words:
            return []

        # Sort by Y position
        words.sort(key=lambda w: w[0])

        # Group into lines: words within y_tolerance of each other = same line
        lines = []
        current_line = [words[0]]

        for word in words[1:]:
            if abs(word[0] - current_line[-1][0]) <= y_tolerance:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        lines.append(current_line)

        # Sort each line left to right by X, join into string
        sorted_lines = []
        for line in lines:
            line.sort(key=lambda w: w[1])
            line_text = ' '.join(w[2] for w in line)
            sorted_lines.append(line_text)

        return sorted_lines

    def recognize_image(self, image_path: str) -> str:
        """Main entry point — OCR an image and return structured text"""
        self.load_models()
        print(f"[OCR] Processing: {image_path}")

        # Preprocess
        preprocessed = self.preprocess_image(image_path)

        if self.easyocr_reader is None:
            raise RuntimeError("No OCR engine available!")

        # EasyOCR on preprocessed image
        results = self.easyocr_reader.readtext(preprocessed)

        if not results:
            print("[OCR] Warning: No text detected in image")
            return ""

        # Group detections into proper lines
        lines = self._group_words_into_lines(results, y_tolerance=20)

        full_text = '\n'.join(lines)
        print(f"[OCR] Extracted {len(lines)} lines, {len(full_text)} characters")

        return full_text
