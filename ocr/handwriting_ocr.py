"""
Handwriting OCR Module
Architecture:
  1. EasyOCR detects word bounding boxes
  2. Words are grouped into lines by Y position (left-to-right order preserved)
  3. TrOCR reads each full line crop for better handwriting accuracy
  4. Noise cleaner fixes common OCR misreads before grading
"""

import os
import cv2
import numpy as np
from PIL import Image
import warnings
import logging

# Suppress ALL noisy library output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for _name in [
    "transformers", "transformers.modeling_utils",
    "transformers.configuration_utils",
    "accelerate", "accelerate.utils", "easyocr", "torch", "PIL",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)


NOISE_PATTERNS = [
    (r'[#~_\|]+', ' '),
    (r'\b0F\b', 'OF'),
    (r'\b1S\b', 'IS'),
    (r'\b1T\b', 'IT'),
    (r'\bdiox=\s*id[oa]\b', 'dioxide'),
    (r'\bWaken\b', 'water'),
    (r'\bPhotosy[a-z]+is\b', 'Photosynthesis'),
    (r'\bPHOTOSY[A-Z]+IS\b', 'PHOTOSYNTHESIS'),
    (r'\bchlorophy[l\s)]+\b', 'chlorophyll'),
    (r'\bgluco[a-z]+\b', 'glucose'),
    (r'\s{2,}', ' '),
]


def clean_ocr_text(text: str) -> str:
    import re
    for pattern, replacement in NOISE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)[!@$%^&*=+<>{}[\]\'\"\\](?!\w)', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


class HandwritingOCR:
    def __init__(self):
        self.easyocr_reader = None
        self.trocr_processor = None
        self.trocr_model = None
        self._loaded = False

    def load_models(self):
        if self._loaded:
            return
        print("[OCR] Loading models...")

        try:
            import easyocr
            print("[OCR] Loading EasyOCR (word detector)...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[OCR] EasyOCR loaded ✅")
        except Exception as e:
            print(f"[OCR] EasyOCR failed: {e}")

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            print("[OCR] Loading TrOCR (handwriting reader)...")
            self.trocr_processor = TrOCRProcessor.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-large-handwritten"
            )
            print("[OCR] TrOCR loaded ✅")
        except Exception as e:
            print(f"[OCR] TrOCR failed: {e}")
            self.trocr_model = None

        self._loaded = True

    def preprocess_image(self, image_path: str):
        """Returns (preprocessed_gray, original_pil)"""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")

        h, w = img_bgr.shape[:2]
        if w < 1400:
            scale = 1400 / w
            img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)

        original_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        thresh = self._deskew(thresh)
        return thresh, original_pil

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 10:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _group_into_lines(self, easyocr_results: list) -> list:
        """
        Groups word detections into text lines sorted top-to-bottom.
        Within each line, words are sorted LEFT-TO-RIGHT.
        This ensures question numbers like '1.' always appear
        before the answer text on the same line.
        """
        if not easyocr_results:
            return []

        words = []
        for bbox, text, conf in easyocr_results:
            if conf < 0.15 or not text.strip():
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            words.append({
                'x1': min(xs), 'x2': max(xs),
                'y1': min(ys), 'y2': max(ys),
                'xc': (min(xs) + max(xs)) / 2,
                'yc': (min(ys) + max(ys)) / 2,
                'h':  max(ys) - min(ys),
                'text': text.strip()
            })

        if not words:
            return []

        words.sort(key=lambda w: w['yc'])

        # Adaptive tolerance based on average word height
        avg_h = sum(w['h'] for w in words) / len(words)
        y_tol = max(12, avg_h * 0.65)

        lines = [[words[0]]]
        for w in words[1:]:
            if abs(w['yc'] - lines[-1][-1]['yc']) <= y_tol:
                lines[-1].append(w)
            else:
                lines.append([w])

        # Build line objects sorted left-to-right within each line
        result = []
        for line_words in lines:
            line_words.sort(key=lambda w: w['x1'])   # ← LEFT TO RIGHT
            result.append({
                'x1': min(w['x1'] for w in line_words),
                'y1': min(w['y1'] for w in line_words),
                'x2': max(w['x2'] for w in line_words),
                'y2': max(w['y2'] for w in line_words),
                'fallback': ' '.join(w['text'] for w in line_words)
            })

        result.sort(key=lambda l: l['y1'])
        return result

    def _trocr_read_line(self, crop_pil: Image.Image) -> str:
        """Run TrOCR on a single line crop"""
        import torch
        w, h = crop_pil.size
        # TrOCR needs minimum height
        if h < 32:
            crop_pil = crop_pil.resize((int(w * 32 / h), 32), Image.LANCZOS)

        pixel_values = self.trocr_processor(
            crop_pil.convert("RGB"), return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            ids = self.trocr_model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=4,
                early_stopping=True,
            )
        return self.trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def recognize_image(self, image_path: str) -> str:
        """Main entry point. Returns cleaned text ready for answer parsing."""
        self.load_models()
        print(f"[OCR] Processing: {image_path}")

        preprocessed, original_pil = self.preprocess_image(image_path)
        orig_w, orig_h = original_pil.size
        proc_h, proc_w = preprocessed.shape
        sx, sy = orig_w / proc_w, orig_h / proc_h

        raw_results = self.easyocr_reader.readtext(preprocessed)
        lines = self._group_into_lines(raw_results)

        if not lines:
            print("[OCR] No text detected")
            return ""

        pad = 10
        final_lines = []
        for line in lines:
            cx1 = max(0, int(line['x1'] * sx) - pad)
            cy1 = max(0, int(line['y1'] * sy) - pad)
            cx2 = min(orig_w, int(line['x2'] * sx) + pad)
            cy2 = min(orig_h, int(line['y2'] * sy) + pad)
            crop = original_pil.crop((cx1, cy1, cx2, cy2))

            if self.trocr_model is not None:
                try:
                    text = self._trocr_read_line(crop)
                    if not text:
                        text = line['fallback']
                except Exception:
                    text = line['fallback']
            else:
                text = line['fallback']

            if text.strip():
                final_lines.append(text.strip())

        raw_text = '\n'.join(final_lines)
        cleaned = clean_ocr_text(raw_text)

        print(f"[OCR] Detected {len(final_lines)} lines")
        print(f"\n[OCR] Raw text:\n{raw_text}")
        print(f"\n[OCR] Cleaned text:\n{cleaned}\n")

        return cleaned
