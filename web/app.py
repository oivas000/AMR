"""
Web GUI Backend — FastAPI
Run with: uvicorn app:app --reload --port 7860
"""

import os, sys, warnings, logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")
for _l in ["transformers","accelerate","sentence_transformers","easyocr","torch","PIL"]:
    logging.getLogger(_l).setLevel(logging.ERROR)

import shutil, tempfile, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent))
import yaml
from utils.answer_parser import parse_answers_from_text
from grader.grading_engine import PaperGrader

app = FastAPI(title="AI Paper Grader")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── cache models after first load ──────────────────────────────────────────
_ocr   = None
_grader_cache = {}

def get_ocr():
    global _ocr
    if _ocr is None:
        from ocr.handwriting_ocr import HandwritingOCR
        _ocr = HandwritingOCR()
        _ocr.load_models()
    return _ocr

def get_grader(use_llm: bool):
    key = str(use_llm)
    if key not in _grader_cache:
        _grader_cache[key] = PaperGrader(
            use_llm_feedback=use_llm,
            llama_server_url="http://localhost:8080"
        )
        _grader_cache[key].semantic_grader.load()
    return _grader_cache[key]

# ── list available answer keys ─────────────────────────────────────────────
@app.get("/api/answer-keys")
def list_answer_keys():
    keys_dir = Path(__file__).parents[1] / "answer_keys"
    keys = [f.name for f in keys_dir.glob("*.yaml") if f.is_file()]
    print("Available answer keys:", keys, keys_dir)
    return {"keys": sorted(keys)}

# ── grade endpoint ─────────────────────────────────────────────────────────
@app.post("/api/grade")
async def grade_paper(
    file: UploadFile = File(...),
    answer_key: str  = Form(...),
    student_id: str  = Form(default="Student"),
    use_llm: bool    = Form(default=False),
    input_type: str  = Form(default="image"),  # "image" or "text"
):
    keys_dir = Path(__file__).resolve().parent.parent / "answer_keys"
    key_path = keys_dir / answer_key
    if not key_path.exists():
        raise HTTPException(404, f"Answer key not found: {answer_key}")

    with open(key_path) as f:
        raw_key = yaml.safe_load(f)
    answer_key_data = {str(k): v for k, v in raw_key.items()}

    # Save uploaded file to temp
    suffix = Path(file.filename).suffix or (".jpg" if input_type == "image" else ".txt")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if input_type == "image":
            ocr = get_ocr()
            extracted_text = ocr.recognize_image(tmp_path)
            raw_lines = extracted_text
        else:
            with open(tmp_path, "r") as f:
                extracted_text = f.read()
            raw_lines = extracted_text

        student_answers = parse_answers_from_text(extracted_text)

        if not student_answers:
            return JSONResponse({
                "error": "No answers could be parsed from the uploaded file.",
                "raw_text": extracted_text
            }, status_code=422)

        grader = get_grader(use_llm)
        result = grader.grade_paper(student_answers, answer_key_data, student_id)

        # Serialize result
        questions = []
        for qid in sorted(result.question_results.keys(), key=lambda x: (len(x), x)):
            qr = result.question_results[qid]
            questions.append({
                "id":          qid,
                "student":     qr.student_answer,
                "correct":     qr.correct_answer,
                "marks":       qr.marks_awarded,
                "max_marks":   qr.max_marks,
                "similarity":  qr.similarity_score,
                "grade":       qr.grade,
                "feedback":    qr.feedback,
            })

        return {
            "student_id":   result.student_id,
            "total_marks":  result.total_marks,
            "max_marks":    result.max_marks,
            "percentage":   result.percentage,
            "grade":        result.grade,
            "questions":    questions,
            "raw_text":     raw_lines,
        }
    finally:
        os.unlink(tmp_path)

# ── serve frontend ─────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
