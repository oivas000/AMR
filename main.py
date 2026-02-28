#!/usr/bin/env python3
"""
AI Paper Grading System - CLI
Usage:
  python main.py grade --image paper.jpg --key answer_keys/science.yaml
  python main.py grade --image paper.jpg --key answer_keys/science.yaml --llm
  python main.py grade --text student.txt --key answer_keys/science.yaml
  python main.py batch --folder papers/ --key answer_keys/science.yaml
  python main.py create-key
  python main.py demo
"""

# ── Must set env vars BEFORE any other imports ──────────────────────────────
import os
import warnings
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["ACCELERATE_LOG_LEVEL"] = "error"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")
for _log in [
    "transformers", "transformers.modeling_utils",
    "transformers.configuration_utils",
    "accelerate", "accelerate.utils", "accelerate.logging", "accelerate.state",
    "sentence_transformers", "easyocr", "torch", "PIL",
]:
    logging.getLogger(_log).setLevel(logging.ERROR)
# ────────────────────────────────────────────────────────────────────────────

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from utils.answer_parser import parse_answers_from_text, display_parsed_answers
from grader.grading_engine import PaperGrader
from utils.report_generator import (
    print_report, save_report_json, save_report_csv, print_batch_summary
)


def load_answer_key(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return {str(k): v for k, v in data.items()}


def grade_from_image(image_path: str, answer_key: dict, student_id: str,
                     use_llm: bool = False, save_output: bool = True):
    from ocr.handwriting_ocr import HandwritingOCR

    print(f"\n[System] Grading paper : {image_path}")
    print(f"[System] Student ID    : {student_id}")
    print(f"[System] LLM feedback  : {'✅ enabled (llama-server)' if use_llm else '❌ disabled (use --llm to enable)'}")

    ocr = HandwritingOCR()
    extracted_text = ocr.recognize_image(image_path)

    print(f"\n[OCR] Final text sent to parser:\n{'─'*50}")
    print(extracted_text)
    print('─' * 50)

    student_answers = parse_answers_from_text(extracted_text)
    display_parsed_answers(student_answers, f"Student {student_id} — Parsed Answers")

    if not student_answers:
        print("[Warning] No answers parsed. Try --text mode with corrected text.")
        return None

    grader = PaperGrader(use_llm_feedback=use_llm, llama_server_url="http://localhost:8080")
    result = grader.grade_paper(student_answers, answer_key, student_id)
    print_report(result)

    if save_output:
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        save_report_json(result, str(out / f"{student_id}_result.json"))

    return result


def grade_from_text(text_path: str, answer_key: dict, student_id: str,
                    use_llm: bool = False, save_output: bool = True):
    print(f"\n[System] Grading text  : {text_path}")
    print(f"[System] Student ID    : {student_id}")
    print(f"[System] LLM feedback  : {'✅ enabled (llama-server)' if use_llm else '❌ disabled (use --llm to enable)'}")

    with open(text_path, 'r') as f:
        text = f.read()

    student_answers = parse_answers_from_text(text)
    display_parsed_answers(student_answers, f"Student {student_id} — Parsed Answers")

    grader = PaperGrader(use_llm_feedback=use_llm, llama_server_url="http://localhost:8080")
    result = grader.grade_paper(student_answers, answer_key, student_id)
    print_report(result)

    if save_output:
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        save_report_json(result, str(out / f"{student_id}_result.json"))

    return result


def batch_grade(folder: str, answer_key: dict, use_llm: bool = False):
    folder_path = Path(folder)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    text_exts  = {'.txt'}
    all_results = []

    files = list(folder_path.iterdir())
    print(f"\n[Batch] {len(files)} files in {folder}")

    for fp in sorted(files):
        if not fp.is_file():
            continue
        student_id = fp.stem
        if fp.suffix.lower() in image_exts:
            r = grade_from_image(str(fp), answer_key, student_id, use_llm=use_llm)
        elif fp.suffix.lower() in text_exts:
            r = grade_from_text(str(fp), answer_key, student_id, use_llm=use_llm)
        else:
            continue
        if r:
            all_results.append(r)

    if all_results:
        print_batch_summary(all_results)
        out = Path("outputs")
        out.mkdir(exist_ok=True)
        save_report_csv(all_results, str(out / "batch_results.csv"))

    return all_results


def run_demo():
    print("\n" + "="*60)
    print("  🎓 AI PAPER GRADING SYSTEM - DEMO")
    print("="*60)
    key_path  = Path(__file__).parent / "answer_keys" / "example_science.yaml"
    text_path = Path(__file__).parent / "sample_papers" / "sample_student_text.txt"
    if not key_path.exists() or not text_path.exists():
        print("[Error] Sample files not found")
        return
    grade_from_text(str(text_path), load_answer_key(str(key_path)),
                    "DEMO_STUDENT_101", use_llm=False)


def create_answer_key_interactive():
    print("\n📝 Answer Key Creator\n" + "="*40)
    subject = input("Subject name: ").strip()
    n = int(input("Number of questions: ").strip())
    key_data = {}

    for i in range(1, n + 1):
        print(f"\n--- Question {i} ---")
        has_parts = input(f"Q{i} has sub-parts? (y/n): ").lower() == 'y'
        if has_parts:
            np_ = int(input("How many parts? "))
            for part in ['a','b','c','d','e'][:np_]:
                qid = f"{i}{part}"
                ans  = input(f"  Answer for Q{i}({part}): ").strip()
                marks = int(input(f"  Marks: "))
                kw   = input(f"  Keywords (comma-sep or blank): ").strip()
                entry = {"answer": ans, "marks": marks}
                if kw:
                    entry["keywords"] = [k.strip() for k in kw.split(',')]
                key_data[qid] = entry
        else:
            ans  = input(f"Answer for Q{i}: ").strip()
            marks = int(input(f"Marks: "))
            kw   = input(f"Keywords (comma-sep or blank): ").strip()
            entry = {"answer": ans, "marks": marks}
            if kw:
                entry["keywords"] = [k.strip() for k in kw.split(',')]
            key_data[str(i)] = entry

    out = Path("answer_keys") / f"{subject.lower().replace(' ','_')}.yaml"
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as f:
        yaml.dump(key_data, f, default_flow_style=False, allow_unicode=True)
    print(f"\n✅ Saved: {out}")
    return str(out)


def main():
    parser = argparse.ArgumentParser(
        description="🎓 AI Paper Grading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo
  python main.py grade --image paper.jpg --key answer_keys/science.yaml
  python main.py grade --image paper.jpg --key answer_keys/science.yaml --llm
  python main.py batch --folder papers/ --key answer_keys/science.yaml --llm
  python main.py create-key

LLM feedback (--llm flag):
  Requires llama-server running:
    ./llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
        """
    )

    sub = parser.add_subparsers(dest='command')
    sub.add_parser('demo')

    gp = sub.add_parser('grade')
    grp = gp.add_mutually_exclusive_group(required=True)
    grp.add_argument('--image', help='Scanned paper image')
    grp.add_argument('--text',  help='Pre-extracted text file')
    gp.add_argument('--key',        required=True)
    gp.add_argument('--student-id', default='Student')
    gp.add_argument('--llm',        action='store_true',
                    help='Enable LLM feedback via llama-server (Qwen)')
    gp.add_argument('--no-save',    action='store_true')

    bp = sub.add_parser('batch')
    bp.add_argument('--folder', required=True)
    bp.add_argument('--key',    required=True)
    bp.add_argument('--llm',    action='store_true')

    sub.add_parser('create-key')

    args = parser.parse_args()

    if args.command == 'demo':
        run_demo()
    elif args.command == 'grade':
        key = load_answer_key(args.key)
        if args.image:
            grade_from_image(args.image, key, args.student_id,
                             use_llm=args.llm, save_output=not args.no_save)
        else:
            grade_from_text(args.text, key, args.student_id,
                            use_llm=args.llm, save_output=not args.no_save)
    elif args.command == 'batch':
        batch_grade(args.folder, load_answer_key(args.key), use_llm=args.llm)
    elif args.command == 'create-key':
        create_answer_key_interactive()
    else:
        parser.print_help()
        print("\n💡 Quick start: python main.py demo")


if __name__ == "__main__":
    main()
