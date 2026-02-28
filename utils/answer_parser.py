"""
Question-Answer Parser
Maps recognized text to question numbers and answers
Handles formats like: 1. / Q1: / (a) / Answer 1: etc.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParsedAnswer:
    question_id: str          # "1", "2a", "Q3", etc.
    raw_text: str             # Raw OCR text for this answer
    cleaned_text: str         # Cleaned version
    confidence: float = 1.0   # OCR confidence if available


@dataclass
class ParsedPaper:
    student_id: str
    answers: Dict[str, ParsedAnswer] = field(default_factory=dict)
    raw_text: str = ""


# All supported question number patterns
QUESTION_PATTERNS = [
    # "1." or "1)" or "1:"
    r"(?:^|\n)\s*(?:Q\.?|Question\s*)?(\d+)\s*[.):\-]\s*",
    # "Q1" or "Q.1" or "Question 1"
    r"(?:^|\n)\s*(?:Q\.?\s*|Question\s+)(\d+)\s*[.):\-]?\s*",
    # "(a)" or "(i)" or "(A)"
    r"(?:^|\n)\s*\(([a-zA-Z])\)\s*",
    # "Part a:" or "Part A:"
    r"(?:^|\n)\s*Part\s+([a-zA-Z\d]+)\s*[.):\-]\s*",
    # "Answer 1:" or "Ans 1:"
    r"(?:^|\n)\s*(?:Answer|Ans)\.?\s*(\d+)\s*[.):\-]\s*",
    # Sub-questions: "1a" or "1(a)" or "2.b"
    r"(?:^|\n)\s*(\d+)\s*[.(]?\s*([a-z])\s*[.):\-]\s*",
]

# Combined compiled pattern
COMBINED_PATTERN = re.compile(
    r"(?:^|\n)\s*"
    r"(?:"
    r"(?:Q\.?\s*|Question\s+)?(\d+)\s*(?:\(\s*([a-z])\s*\))?\s*[.):\-]"  # 1. or 1(a). or Q1.
    r"|"
    r"\(([a-zA-Z\d]+)\)\s*"                                                 # (a) or (i)
    r"|"
    r"(?:Answer|Ans)\.?\s*(\d+)\s*[.):\-]"                                  # Answer 1:
    r")",
    re.IGNORECASE | re.MULTILINE
)


def normalize_question_id(raw_id: str) -> str:
    """Normalize question ID to consistent format"""
    # Remove spaces, lowercase
    q = raw_id.strip().lower()
    # Remove leading 'q' if it's just a number prefix
    q = re.sub(r'^q\.?', '', q)
    q = q.strip('.):-')
    return q


def parse_answers_from_text(text: str) -> Dict[str, str]:
    """
    Parse text and extract question_id -> answer_text mapping.
    Returns dict like {"1": "Paris", "2": "Newton's law...", "3a": "..."}
    """
    answers = {}

    # Split text into lines for processing
    lines = text.split('\n')
    
    # Strategy: find question markers and collect text until next marker
    question_markers = []
    
    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Try to detect if this line starts with a question number
        match = _detect_question_marker(stripped)
        if match:
            q_id, remaining_text = match
            question_markers.append({
                'q_id': q_id,
                'line_idx': line_idx,
                'inline_text': remaining_text
            })
    
    # Now collect answer text for each question
    for i, marker in enumerate(question_markers):
        q_id = marker['q_id']
        start_line = marker['line_idx']
        end_line = question_markers[i+1]['line_idx'] if i+1 < len(question_markers) else len(lines)
        
        # Collect all text from this question to next
        answer_lines = [marker['inline_text']] if marker['inline_text'] else []
        for line in lines[start_line+1:end_line]:
            stripped = line.strip()
            if stripped:
                answer_lines.append(stripped)
        
        answer_text = ' '.join(answer_lines).strip()
        if answer_text:
            answers[q_id] = answer_text
    
    # If no markers found, try a simpler split approach
    if not answers:
        answers = _fallback_parse(text)
    
    return answers


def _detect_question_marker(line: str):
    """
    Returns (question_id, remaining_answer_text) if line starts with a question marker.
    Returns None if not a question marker.
    """
    patterns = [
        # "1." or "1)" or "1:" with optional sub-letter: "1a." "1(a)"
        (r'^(?:Q\.?\s*|Question\s+)?(\d+)\s*(?:\(\s*([a-z])\s*\)|([a-z]))?\s*[.):\-]\s*(.*)',
         lambda m: (m.group(1) + (m.group(2) or m.group(3) or ''), m.group(4) or '')),
        
        # "(a)" "(b)" standalone
        (r'^\(\s*([a-zA-Z\d]+)\s*\)\s*(.*)',
         lambda m: (m.group(1).lower(), m.group(2) or '')),
        
        # "Answer 1:" or "Ans. 2:"
        (r'^(?:Answer|Ans)\.?\s*(\d+)\s*[.):\-]\s*(.*)',
         lambda m: (m.group(1), m.group(2) or '')),
    ]
    
    for pattern, extractor in patterns:
        m = re.match(pattern, line, re.IGNORECASE)
        if m:
            try:
                q_id, remaining = extractor(m)
                q_id = q_id.strip().lower()
                if q_id:
                    return q_id, remaining.strip()
            except:
                pass
    
    return None


def _fallback_parse(text: str) -> Dict[str, str]:
    """
    Fallback: try to split by common patterns when sophisticated parsing fails.
    """
    answers = {}
    
    # Try splitting by number patterns
    parts = re.split(r'(?:^|\n)\s*(\d+)\s*[.):\-]', text, flags=re.MULTILINE)
    
    if len(parts) > 2:
        # parts[0] = before first question, then alternating: q_num, answer_text
        i = 1
        while i < len(parts) - 1:
            q_num = parts[i].strip()
            answer = parts[i+1].strip() if i+1 < len(parts) else ''
            if q_num.isdigit() and answer:
                answers[q_num] = answer
            i += 2
    
    return answers


def parse_answer_key(answer_key_text: str) -> Dict[str, str]:
    """
    Parse teacher's answer key from text file or string.
    Same format as student answers.
    """
    return parse_answers_from_text(answer_key_text)


def display_parsed_answers(answers: Dict[str, str], title: str = "Parsed Answers"):
    """Pretty print parsed answers"""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    if not answers:
        print("  No answers found!")
    for q_id in sorted(answers.keys(), key=lambda x: (len(x), x)):
        answer = answers[q_id]
        preview = answer[:80] + "..." if len(answer) > 80 else answer
        print(f"  Q{q_id}: {preview}")
    print(f"{'='*50}\n")
