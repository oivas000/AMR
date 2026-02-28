"""
Handwriting OCR Module
Primary: TrOCR (Microsoft) - best for handwriting
Fallback: EasyOCR
"""

import cv2
import numpy as np
from PIL import Image
import re
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class HandwritingOCR:
    def __init__(self, use_trocr=True, use_easyocr_fallback=True):
        self.use_trocr = use_trocr
        self.use_easyocr_fallback = use_easyocr_fallback
        self.trocr_processor = None
        self.trocr_model = None
        self.easyocr_reader = None
        self._loaded = False

    def load_models(self):
        """Lazy-load models to avoid slow startup"""
        if self._loaded:
            return

        print("[OCR] Loading models... (first run downloads ~1GB, cached after)")

        if self.use_trocr:
            try:
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                print("[OCR] Loading TrOCR handwriting model...")
                self.trocr_processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-large-handwritten"  # large = better accuracy
                )
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-large-handwritten"
                )
                print("[OCR] TrOCR loaded ✅")
            except Exception as e:
                print(f"[OCR] TrOCR failed to load: {e}. Falling back to EasyOCR.")
                self.use_trocr = False

        if self.use_easyocr_fallback or not self.use_trocr:
            try:
                import easyocr
                print("[OCR] Loading EasyOCR...")
                self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
                print("[OCR] EasyOCR loaded ✅")
            except Exception as e:
                print(f"[OCR] EasyOCR failed: {e}")

        self._loaded = True

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # Adaptive thresholding - great for uneven lighting on handwritten papers
        thresh = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Deskew
        coords = np.column_stack(np.where(thresh < 128))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 10:  # Only correct small skews
                (h, w) = thresh.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                thresh = cv2.warpAffine(thresh, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)

        return thresh

    def split_into_regions(self, preprocessed_img: np.ndarray) -> list:
        """
        Split paper into question-answer regions.
        Returns list of (region_image, y_position) tuples sorted by position.
        """
        # Find horizontal text lines using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        dilated = cv2.dilate(preprocessed_img, kernel, iterations=2)
        inverted = cv2.bitwise_not(dilated)

        # Find contours of text blocks
        contours, _ = cv2.findContours(
            inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        h_total, w_total = preprocessed_img.shape

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise - only meaningful text blocks
            if h > 15 and w > 50 and h < h_total * 0.4:
                # Add padding
                pad = 5
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(w_total, x + w + pad)
                y2 = min(h_total, y + h + pad)
                region = preprocessed_img[y1:y2, x1:x2]
                regions.append((region, y1, x1))

        # Sort by vertical position (top to bottom)
        regions.sort(key=lambda r: r[1])
        return regions

    def trocr_recognize(self, pil_image: Image.Image) -> str:
        """Run TrOCR on a PIL image"""
        import torch
        pixel_values = self.trocr_processor(
            pil_image.convert("RGB"), return_tensors="pt"
        ).pixel_values

        with torch.no_grad():
            generated_ids = self.trocr_model.generate(pixel_values)

        text = self.trocr_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def easyocr_recognize(self, image_array: np.ndarray) -> str:
        """Run EasyOCR on numpy array"""
        results = self.easyocr_reader.readtext(image_array)
        # Sort by position and join
        results.sort(key=lambda r: r[0][0][1])  # sort by y coordinate
        text = " ".join([r[1] for r in results])
        return text.strip()

    def recognize_full_page(self, image_path: str) -> str:
        """OCR an entire page and return full text"""
        self.load_models()

        preprocessed = self.preprocess_image(image_path)

        if self.use_trocr:
            # TrOCR works best on smaller regions; process page in chunks
            h, w = preprocessed.shape
            chunk_height = 150  # pixels per chunk
            chunks = []

            for y in range(0, h, chunk_height):
                chunk = preprocessed[y:y + chunk_height, 0:w]
                if chunk.shape[0] < 20:
                    continue
                pil_chunk = Image.fromarray(chunk)
                try:
                    text = self.trocr_recognize(pil_chunk)
                    if text:
                        chunks.append(text)
                except:
                    pass

            full_text = "\n".join(chunks)
        elif self.easyocr_reader:
            full_text = self.easyocr_recognize(preprocessed)
        else:
            raise RuntimeError("No OCR engine available!")

        return full_text

    def recognize_image(self, image_path: str) -> str:
        """Main entry point - OCR an image file"""
        self.load_models()
        print(f"[OCR] Processing: {image_path}")
        text = self.recognize_full_page(image_path)
        print(f"[OCR] Extracted {len(text)} characters")
        return text
