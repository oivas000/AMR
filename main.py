#!/usr/bin/env python3
"""
AI Paper Grading System - CLI Interface
========================================
Usage:
  python main.py grade --image paper.jpg --key answer_keys/science.yaml
  python main.py grade --text sample_papers/student.txt --key answer_keys/science.yaml
  python main.py batch --folder papers/ --key answer_keys/science.yaml
  python main.py create-key
  python main.py demo
"""

import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from utils.answer_parser import parse_answers_from_text, display_parsed_answers
from grader.grading_engine import PaperGrader
from utils.report_generator import print_report, save_report_json, save_report_csv, print_batch_summary


def load_answer_key(yaml_path: str) -> dict:
    """Load answer key from YAML file"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Normalize keys to strings
    normalized = {}
    for k, v in data.items():
        normalized[str(k)] = v

    return normalized


def grade_from_image(image_path: str, answer_key: dict, student_id: str,
                     use_llm: bool = False, save_output: bool = True):
    """Grade a paper from scanned image"""
    from ocr.handwriting_ocr import HandwritingOCR

    print(f"\n[System] Grading paper: {image_path}")
    print(f"[System] Student ID: {student_id}")

    # Step 1: OCR
    ocr = HandwritingOCR()
    extracted_text = ocr.recognize_image(image_path)

    print(f"\n[OCR] Extracted Text:\n{'─'*50}")
    print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))
    print("─" * 50)

    # Step 2: Parse answers
    student_answers = parse_answers_from_text(extracted_text)
    display_parsed_answers(student_answers, f"Student {student_id} Answers")

    if not student_answers:
        print("[Warning] No answers could be parsed from the image!")
        print("[Hint] Try --text mode with a pre-extracted text file")
        return None

    # Step 3: Grade
    grader = PaperGrader(use_llm_feedback=use_llm, llama_server_url="http://localhost:8080")

    if save_output:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        save_report_json(result, str(output_dir / f"{student_id}_result.json"))

    return result


def grade_from_text(text_path: str, answer_key: dict, student_id: str,
                    use_llm: bool = False, save_output: bool = True):
    """Grade a paper from pre-extracted text file"""
    print(f"\n[System] Grading from text: {text_path}")
    print(f"[System] Student ID: {student_id}")

    with open(text_path, 'r') as f:
        text = f.read()

    # Parse answers
    student_answers = parse_answers_from_text(text)
    display_parsed_answers(student_answers, f"Student {student_id} Answers")

    # Grade
    grader = PaperGrader(use_llm_feedback=use_llm, llama_server_url="http://localhost:8080")
    result = grader.grade_paper(student_answers, answer_key, student_id)

    # Report
    print_report(result)

    if save_output:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        save_report_json(result, str(output_dir / f"{student_id}_result.json"))

    return result


def batch_grade(folder: str, answer_key: dict, use_llm: bool = False):
    """Grade all papers in a folder"""
    folder_path = Path(folder)

    # Find all image files and text files
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    text_exts = {'.txt'}

    all_results = []

    files = list(folder_path.iterdir())
    print(f"\n[Batch] Found {len(files)} files in {folder}")

    for file_path in sorted(files):
        if not file_path.is_file():
            continue

        student_id = file_path.stem  # filename without extension

        if file_path.suffix.lower() in image_exts:
            result = grade_from_image(
                str(file_path), answer_key, student_id,
                use_llm=use_llm, save_output=True
            )
        elif file_path.suffix.lower() in text_exts:
            result = grade_from_text(
                str(file_path), answer_key, student_id,
                use_llm=use_llm, save_output=True
            )
        else:
            continue

        if result:
            all_results.append(result)

    if all_results:
        print_batch_summary(all_results)

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        save_report_csv(all_results, str(output_dir / "batch_results.csv"))
        print(f"[Batch] CSV saved to outputs/batch_results.csv")

    return all_results


def run_demo():
    """Run demo with sample data (no image needed)"""
    print("\n" + "="*60)
    print("  🎓 AI PAPER GRADING SYSTEM - DEMO")
    print("="*60)
    print("  Using sample student answers + sample answer key")
    print("  No image/OCR needed for this demo\n")

    # Load sample answer key
    key_path = Path(__file__).parent / "answer_keys" / "example_science.yaml"
    if not key_path.exists():
        print(f"[Error] Sample answer key not found: {key_path}")
        return

    answer_key = load_answer_key(str(key_path))

    # Load sample student text
    text_path = Path(__file__).parent / "sample_papers" / "sample_student_text.txt"
    if not text_path.exists():
        print(f"[Error] Sample paper not found: {text_path}")
        return

    grade_from_text(str(text_path), answer_key, "DEMO_STUDENT_101", use_llm=False)


def create_answer_key_interactive():
    """Interactive wizard to create an answer key"""
    print("\n📝 Answer Key Creator")
    print("="*40)

    subject = input("Subject name: ").strip()
    num_questions = int(input("Number of questions: ").strip())

    key_data = {}

    for i in range(1, num_questions + 1):
        print(f"\n--- Question {i} ---")

        has_parts = input(f"Does Q{i} have sub-parts? (y/n): ").strip().lower() == 'y'

        if has_parts:
            num_parts = int(input("How many parts? ").strip())
            for part in ['a', 'b', 'c', 'd', 'e'][:num_parts]:
                q_id = f"{i}{part}"
                answer = input(f"  Answer for Q{i}({part}): ").strip()
                marks = int(input(f"  Marks for Q{i}({part}): ").strip())
                keywords_str = input(f"  Keywords (comma-separated, or enter to skip): ").strip()

                entry = {"answer": answer, "marks": marks}
                if keywords_str:
                    entry["keywords"] = [k.strip() for k in keywords_str.split(',')]

                key_data[q_id] = entry
        else:
            answer = input(f"Answer for Q{i}: ").strip()
            marks = int(input(f"Marks for Q{i}: ").strip())
            keywords_str = input(f"Keywords (comma-separated, or enter to skip): ").strip()

            entry = {"answer": answer, "marks": marks}
            if keywords_str:
                entry["keywords"] = [k.strip() for k in keywords_str.split(',')]

            key_data[str(i)] = entry

    # Save
    output_path = Path("answer_keys") / f"{subject.lower().replace(' ', '_')}.yaml"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(key_data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ Answer key saved: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="🎓 AI Paper Grading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py demo
  python main.py grade --image paper.jpg --key answer_keys/science.yaml
  python main.py grade --text student.txt --key answer_keys/science.yaml --student-id "Roll101" --llm
  python main.py batch --folder papers/ --key answer_keys/science.yaml
  python main.py create-key
        """
    )

    subparsers = parser.add_subparsers(dest='command')

    # Demo command
    subparsers.add_parser('demo', help='Run demo with sample data')

    # Grade command
    grade_parser = subparsers.add_parser('grade', help='Grade a single paper')
    grade_group = grade_parser.add_mutually_exclusive_group(required=True)
    grade_group.add_argument('--image', help='Path to scanned paper image')
    grade_group.add_argument('--text', help='Path to pre-extracted text file')
    grade_parser.add_argument('--key', required=True, help='Path to answer key YAML file')
    grade_parser.add_argument('--student-id', default='Student', help='Student ID/name')
    grade_parser.add_argument('--llm', action='store_true', help='Use Ollama LLM for feedback')
    grade_parser.add_argument('--no-save', action='store_true', help='Do not save output files')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Grade all papers in a folder')
    batch_parser.add_argument('--folder', required=True, help='Folder containing papers')
    batch_parser.add_argument('--key', required=True, help='Path to answer key YAML file')
    batch_parser.add_argument('--llm', action='store_true', help='Use Ollama LLM for feedback')

    # Create key command
    subparsers.add_parser('create-key', help='Interactively create an answer key')

    args = parser.parse_args()

    if args.command == 'demo':
        run_demo()

    elif args.command == 'grade':
        answer_key = load_answer_key(args.key)

        if args.image:
            grade_from_image(
                args.image, answer_key, args.student_id,
                use_llm=args.llm, save_output=not args.no_save
            )
        else:
            grade_from_text(
                args.text, answer_key, args.student_id,
                use_llm=args.llm, save_output=not args.no_save
            )

    elif args.command == 'batch':
        answer_key = load_answer_key(args.key)
        batch_grade(args.folder, answer_key, use_llm=args.llm)

    elif args.command == 'create-key':
        create_answer_key_interactive()

    else:
        parser.print_help()
        print("\n💡 Quick start: python main.py demo")


if __name__ == "__main__":
    main()
