"""
Report Generator
Produces formatted grading reports for CLI display and file saving.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List
from grader.grading_engine import GradingResult, QuestionResult


GRADE_COLORS = {
    "A+": "\033[92m",  # Bright green
    "A":  "\033[92m",
    "B":  "\033[94m",  # Blue
    "C":  "\033[93m",  # Yellow
    "D":  "\033[33m",  # Orange
    "F":  "\033[91m",  # Red
}
RESET = "\033[0m"
BOLD = "\033[1m"


def color_grade(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "")
    return f"{BOLD}{color}{grade}{RESET}"


def print_report(result: GradingResult, show_answers: bool = True):
    """Print a formatted grading report to terminal"""
    
    width = 70
    print("\n" + "=" * width)
    print(f"{BOLD}  📋 GRADING REPORT{RESET}")
    print("=" * width)
    print(f"  Student ID : {result.student_id}")
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Total      : {result.total_marks:.1f} / {result.max_marks:.1f}")
    print(f"  Percentage : {result.percentage:.1f}%")
    print(f"  Grade      : {color_grade(result.grade)}")
    print("=" * width)
    
    print(f"\n{BOLD}  Question-wise Breakdown:{RESET}\n")
    
    for q_id in sorted(result.question_results.keys(), key=lambda x: (len(x), x)):
        qr = result.question_results[q_id]
        
        # Question header
        marks_color = _marks_color(qr.marks_awarded, qr.max_marks)
        print(f"  ┌─ Question {q_id} " + "─" * (width - 18 - len(q_id)))
        print(f"  │  Marks   : {marks_color}{qr.marks_awarded:.1f} / {qr.max_marks:.1f}{RESET}  "
              f"[{color_grade(qr.grade)}]  (similarity: {qr.similarity_score:.2f})")
        
        if show_answers:
            # Student answer
            student_preview = qr.student_answer[:100]
            if len(qr.student_answer) > 100:
                student_preview += "..."
            print(f"  │  Student : {student_preview}")
            
            # Correct answer (truncated)
            correct_preview = qr.correct_answer[:100]
            if len(qr.correct_answer) > 100:
                correct_preview += "..."
            print(f"  │  Correct : {correct_preview}")
        
        # Feedback
        if qr.feedback:
            print(f"  │  💬 {qr.feedback}")
        
        print(f"  └" + "─" * (width - 4))
        print()
    
    # Summary bar
    print("=" * width)
    filled = int(result.percentage / 100 * 40)
    bar = "█" * filled + "░" * (40 - filled)
    grade_color = GRADE_COLORS.get(result.grade, "")
    print(f"  [{grade_color}{bar}{RESET}] {result.percentage:.1f}%")
    print("=" * width + "\n")


def _marks_color(marks: float, max_marks: float) -> str:
    ratio = marks / max_marks if max_marks > 0 else 0
    if ratio >= 0.8: return "\033[92m"
    if ratio >= 0.6: return "\033[93m"
    return "\033[91m"


def save_report_json(result: GradingResult, output_path: str):
    """Save grading result as JSON"""
    data = {
        "student_id": result.student_id,
        "timestamp": datetime.now().isoformat(),
        "total_marks": result.total_marks,
        "max_marks": result.max_marks,
        "percentage": result.percentage,
        "grade": result.grade,
        "questions": {}
    }
    
    for q_id, qr in result.question_results.items():
        data["questions"][q_id] = {
            "student_answer": qr.student_answer,
            "correct_answer": qr.correct_answer,
            "marks_awarded": qr.marks_awarded,
            "max_marks": qr.max_marks,
            "similarity_score": qr.similarity_score,
            "grade": qr.grade,
            "feedback": qr.feedback
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"[Report] Saved JSON: {output_path}")


def save_report_csv(results: List[GradingResult], output_path: str):
    """Save multiple results as CSV for batch processing"""
    rows = []
    for result in results:
        row = {
            "student_id": result.student_id,
            "total_marks": result.total_marks,
            "max_marks": result.max_marks,
            "percentage": result.percentage,
            "grade": result.grade,
        }
        for q_id, qr in result.question_results.items():
            row[f"q{q_id}_marks"] = qr.marks_awarded
            row[f"q{q_id}_sim"] = qr.similarity_score
        rows.append(row)
    
    if not rows:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"[Report] Saved CSV: {output_path}")


def print_batch_summary(results: List[GradingResult]):
    """Print summary table for batch grading"""
    print(f"\n{'='*60}")
    print(f"{BOLD}  BATCH GRADING SUMMARY{RESET}")
    print(f"{'='*60}")
    print(f"  {'Student':<15} {'Marks':<12} {'%':<8} {'Grade'}")
    print(f"  {'-'*50}")
    
    total_pct = 0
    for r in results:
        grade_str = color_grade(r.grade)
        print(f"  {r.student_id:<15} {r.total_marks:.1f}/{r.max_marks:<6.1f} "
              f"{r.percentage:<8.1f} {grade_str}")
        total_pct += r.percentage
    
    if results:
        avg = total_pct / len(results)
        print(f"  {'-'*50}")
        print(f"  {'Class Average':<15} {'':12} {avg:<8.1f}")
    
    print(f"{'='*60}\n")
