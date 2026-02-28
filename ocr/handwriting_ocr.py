"""
Handwriting OCR Module - Column-Aware Layout Detection

Key insight: exam papers have a LEFT COLUMN for question numbers
and a RIGHT COLUMN for answer text. We detect the split point
and process them separately to get correct Q→Answer mapping.

Pipeline:
  1. Detect left-margin question numbers separately
  2. Detect answer text blocks
  3. Map each answer block to its nearest question number by Y position
  4. TrOCR reads each line crop for handwriting accuracy
"""

import os
import re
import cv2
import numpy as np
from PIL import Image
import warnings
import logging

# Must set env vars BEFORE any imports to suppress accelerate noise
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"

warnings.filterwarnings("ignore")
for _name in [
    "transformers", "transformers.modeling_utils",
    "transformers.configuration_utils",
    "accelerate", "accelerate.utils", "accelerate.logging",
    "accelerate.state", "accelerate.checkpointing",
    "easyocr", "torch", "PIL",
]:
    logging.getLogger(_name).setLevel(logging.ERROR)


def clean_ocr_text(text: str) -> str:
    """Fix common TrOCR/EasyOCR misreads on handwritten text"""
    fixes = [
        (r'\bPhotosy[a-z]+is\b', 'Photosynthesis'),
        (r'\bPHOTOSY[A-Z]+IS\b', 'PHOTOSYNTHESIS'),
        (r'\bchlorophy[l\s)1]+\b', 'chlorophyll'),
        (r'\bgluco[a-z]+\b', 'glucose'),
        (r'\bdiox[=\-]?\s*id[oa]\b', 'dioxide'),
        (r'\b([Cc])arb[o0]n\b', r'\1arbon'),
        (r'\b[Oo]xig[ae]n\b', 'oxygen'),
        (r'\b[Ss]unligh[t+]+\b', 'sunlight'),
        (r'\bw[ae]t[ae]r\b', 'water'),
        (r'[#~_\|]+', ' '),
        (r'\s{2,}', ' '),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'(?<!\w)[!@$%^&*=+<>{}[\]\'\"\\](?!\w)', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


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
            print("[OCR] Loading EasyOCR (layout detector)...")
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
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )
        thresh = self._deskew(thresh)
        return thresh, original_pil

    def _deskew(self, img):
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 100:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 10:
            return img
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    def _trocr_read(self, crop_pil: Image.Image) -> str:
        """Run TrOCR on a single image crop"""
        import torch
        w, h = crop_pil.size
        if h < 32:
            crop_pil = crop_pil.resize((max(100, int(w * 32 / h)), 32), Image.LANCZOS)
        if w < 50:
            return ""

        pv = self.trocr_processor(crop_pil.convert("RGB"), return_tensors="pt").pixel_values
        with torch.no_grad():
            ids = self.trocr_model.generate(
                pv, max_new_tokens=150, num_beams=4, early_stopping=True
            )
        return self.trocr_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def _detect_q_number(self, text: str):
        """
        Returns question number string if text looks like a question marker.
        Examples: '1', '1.', '2)', 'Q1', '(a)' → '1', '1', '2', '1', 'a'
        """
        t = text.strip()
        patterns = [
            r'^(?:Q\.?\s*)?(\d+)\s*[.):\-]?$',          # 1  1.  2)  Q1
            r'^(?:Q\.?\s*)?(\d+)\s*\(\s*([a-z])\s*\)$', # 1(a)
            r'^\(\s*([a-zA-Z])\s*\)$',                   # (a)
        ]
        for p in patterns:
            m = re.match(p, t, re.IGNORECASE)
            if m:
                groups = [g for g in m.groups() if g]
                return ''.join(groups).lower()
        return None

    def _find_column_split(self, easyocr_results, img_width: int) -> int:
        """
        Find the X coordinate that separates the question-number column
        from the answer column.

        Strategy: question numbers are short words near the left margin.
        We look for a natural gap in X positions of short detections.
        """
        # Collect x2 positions of short (likely numbered) detections near left
        left_boundary = img_width * 0.25  # numbers should be in left 25%
        short_word_x2s = []

        for bbox, text, conf in easyocr_results:
            if conf < 0.2:
                continue
            xs = [p[0] for p in bbox]
            x1, x2 = min(xs), max(xs)
            word_width = x2 - x1
            # Short word near left margin
            if x1 < left_boundary and word_width < img_width * 0.15:
                short_word_x2s.append(x2)

        if not short_word_x2s:
            return int(img_width * 0.12)  # fallback: 12% of width

        # Split point = a bit past the rightmost short-left-word
        split = int(np.percentile(short_word_x2s, 85) * 1.15)
        return min(split, int(img_width * 0.25))

    def _group_words_into_lines(self, words: list, y_tol: float) -> list:
        """Group word dicts into lines, sorted top-to-bottom, left-to-right within line"""
        if not words:
            return []
        words = sorted(words, key=lambda w: w['yc'])
        lines = [[words[0]]]
        for w in words[1:]:
            if abs(w['yc'] - lines[-1][-1]['yc']) <= y_tol:
                lines[-1].append(w)
            else:
                lines.append([w])
        result = []
        for line in lines:
            line.sort(key=lambda w: w['x1'])
            result.append({
                'x1': min(w['x1'] for w in line),
                'y1': min(w['y1'] for w in line),
                'x2': max(w['x2'] for w in line),
                'y2': max(w['y2'] for w in line),
                'yc': sum(w['yc'] for w in line) / len(line),
                'text': ' '.join(w['text'] for w in line)
            })
        return sorted(result, key=lambda l: l['y1'])

    def recognize_image(self, image_path: str) -> str:
        """
        Main entry point.

        Uses column-aware layout:
          LEFT  column → question numbers
          RIGHT column → answer text

        Maps each answer block to the closest question number above it.
        """
        self.load_models()
        print(f"[OCR] Processing: {image_path}")

        preprocessed, original_pil = self.preprocess_image(image_path)
        orig_w, orig_h = original_pil.size
        proc_h, proc_w = preprocessed.shape
        sx, sy = orig_w / proc_w, orig_h / proc_h

        # Run EasyOCR to get all word positions
        raw = self.easyocr_reader.readtext(preprocessed)

        # Find the column split X
        split_x = self._find_column_split(raw, proc_w)
        print(f"[OCR] Column split at x={split_x} (proc coords), "
              f"{int(split_x*sx)} (orig coords)")

        # Separate into left (numbers) and right (answers) words
        avg_h = 20.0
        heights = []
        left_words, right_words = [], []

        for bbox, text, conf in raw:
            if conf < 0.15 or not text.strip():
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            w = {
                'x1': min(xs), 'x2': max(xs),
                'y1': min(ys), 'y2': max(ys),
                'xc': (min(xs)+max(xs))/2,
                'yc': (min(ys)+max(ys))/2,
                'h':  max(ys)-min(ys),
                'text': text.strip()
            }
            heights.append(w['h'])
            xc = w['xc']
            if xc < split_x:
                left_words.append(w)
            else:
                right_words.append(w)

        if heights:
            avg_h = sum(heights) / len(heights)

        y_tol = max(12, avg_h * 0.65)

        # Group into lines
        left_lines  = self._group_words_into_lines(left_words,  y_tol)
        right_lines = self._group_words_into_lines(right_words, y_tol)

        # ── Identify question numbers from left column ──
        # Each entry: {'q_id': '1', 'yc': float, 'y1': float}
        q_markers = []
        for line in left_lines:
            q_id = self._detect_q_number(line['text'])
            if q_id:
                q_markers.append({
                    'q_id': q_id,
                    'yc': line['yc'],
                    'y1': line['y1'],
                    'y2': line['y2'],
                })

        print(f"[OCR] Found question markers: {[m['q_id'] for m in q_markers]}")

        # ── Read each right-column line with TrOCR ──
        answer_lines = []  # list of {'yc', 'text'}
        pad = 10
        for line in right_lines:
            cx1 = max(0, int(line['x1'] * sx) - pad)
            cy1 = max(0, int(line['y1'] * sy) - pad)
            cx2 = min(orig_w, int(line['x2'] * sx) + pad)
            cy2 = min(orig_h, int(line['y2'] * sy) + pad)
            crop = original_pil.crop((cx1, cy1, cx2, cy2))

            if self.trocr_model is not None:
                try:
                    text = self._trocr_read(crop)
                    if not text:
                        text = line['text']
                except Exception:
                    text = line['text']
            else:
                text = line['text']

            if text.strip():
                answer_lines.append({'yc': line['yc'], 'text': clean_ocr_text(text)})

        # ── Map answer lines to question numbers ──
        # Each answer line belongs to the question marker immediately ABOVE it
        if not q_markers:
            # No markers detected → fall back to plain text output
            print("[OCR] ⚠️  No question markers found in left column")
            print("[OCR] Falling back to full-page line reading")
            return self._fallback_full_page(preprocessed, original_pil, sx, sy)

        # Sort markers by Y
        q_markers.sort(key=lambda m: m['y1'])

        # Build question buckets
        buckets = {m['q_id']: [] for m in q_markers}

        for al in answer_lines:
            # Find the last marker whose Y is above this answer line
            assigned = None
            for m in q_markers:
                if m['yc'] <= al['yc'] + avg_h * 0.5:
                    assigned = m['q_id']
            if assigned:
                buckets[assigned].append(al['text'])
            else:
                # Above all markers → attach to first question
                if q_markers:
                    buckets[q_markers[0]['q_id']].append(al['text'])

        # ── Build final structured text ──
        output_lines = []
        for m in q_markers:
            q_id = m['q_id']
            answer_text = ' '.join(buckets[q_id]).strip()
            if answer_text:
                output_lines.append(f"{q_id}. {answer_text}")
                print(f"[OCR] Q{q_id} → {answer_text[:80]}")
            else:
                print(f"[OCR] Q{q_id} → (no answer text found)")

        final_text = '\n'.join(output_lines)
        print(f"\n[OCR] Final structured text:\n{final_text}\n")
        return final_text

    def _fallback_full_page(self, preprocessed, original_pil, sx, sy) -> str:
        """Fallback: read all lines top-to-bottom without column separation"""
        raw = self.easyocr_reader.readtext(preprocessed)
        avg_h = 20.0
        words = []
        for bbox, text, conf in raw:
            if conf < 0.15 or not text.strip():
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            words.append({
                'x1': min(xs), 'x2': max(xs),
                'y1': min(ys), 'y2': max(ys),
                'xc': (min(xs)+max(xs))/2,
                'yc': (min(ys)+max(ys))/2,
                'h':  max(ys)-min(ys),
                'text': text.strip()
            })

        y_tol = max(12, avg_h * 0.65)
        lines = self._group_words_into_lines(words, y_tol)
        orig_w, orig_h = original_pil.size
        pad = 10
        result_lines = []
        for line in lines:
            cx1 = max(0, int(line['x1'] * sx) - pad)
            cy1 = max(0, int(line['y1'] * sy) - pad)
            cx2 = min(orig_w, int(line['x2'] * sx) + pad)
            cy2 = min(orig_h, int(line['y2'] * sy) + pad)
            crop = original_pil.crop((cx1, cy1, cx2, cy2))
            if self.trocr_model:
                try:
                    text = self._trocr_read(crop) or line['text']
                except Exception:
                    text = line['text']
            else:
                text = line['text']
            if text.strip():
                result_lines.append(clean_ocr_text(text))

        return '\n'.join(result_lines)
