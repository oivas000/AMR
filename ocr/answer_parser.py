"""
Answer Parser
Converts structured OCR text into {question_id: answer_text} dict.

Since the new OCR already outputs clean "1. answer" format,
this parser is simpler and more reliable.
"""

import re
from typing import Dict, Optional


_Q_START = re.compile(
    r"""
    ^\s*
    (?:
        (?:Q\.?\s*|Question\s+)?        # optional Q/Question prefix
        (\d+)                           # question number  → group 1
        \s*(?:\(\s*([a-z])\s*\))?       # optional (a) sub-part → group 2
        \s*[.):\-]\s*                   # separator
    |
        \(\s*([a-zA-Z\d]+)\s*\)\s*      # standalone (a) → group 3
    |
        (?:Answer|Ans)\.?\s*(\d+)\s*[.):\-]\s*  # Answer 1: → group 4
    )
    """,
    re.VERBOSE | re.IGNORECASE
)


def _extract_q_id(line: str) -> Optional[tuple]:
    """
    Returns (q_id, remaining_text) if line starts with a question marker.
    Returns None otherwise.
    """
    m = _Q_START.match(line)
    if not m:
        return None

    if m.group(1):
        num = m.group(1)
        sub = m.group(2) or ''
        q_id = (num + sub).lower()
    elif m.group(3):
        q_id = m.group(3).lower()
    elif m.group(4):
        q_id = m.group(4)
    else:
        return None

    remaining = line[m.end():].strip()
    remaining = re.sub(r'^[.):\-\s]+', '', remaining).strip()
    return q_id, remaining


def parse_answers_from_text(text: str) -> Dict[str, str]:
    """
    Parse structured text into {q_id: answer} dict.

    Handles:
    - Clean "1. answer text" (output from our column-aware OCR)
    - Multi-line answers
    - Orphan question numbers (number alone, answer on next line)
    - Mixed formats
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    segments = []   # list of {q_id, lines: []}
    i = 0

    while i < len(lines):
        parsed = _extract_q_id(lines[i])

        if parsed is not None:
            q_id, inline = parsed
            answer_lines = [inline] if inline else []

            # Handle orphan number: "1\n" then answer on next line
            if not inline and i + 1 < len(lines):
                if _extract_q_id(lines[i + 1]) is None:
                    answer_lines.append(lines[i + 1])
                    i += 1

            segments.append({'q_id': q_id, 'lines': answer_lines})
        else:
            # Continuation of previous answer
            if segments:
                segments[-1]['lines'].append(lines[i])

        i += 1

    answers = {}
    for seg in segments:
        q_id = seg['q_id']
        answer = ' '.join(seg['lines']).strip()
        answer = re.sub(r'^[.):\-\s]+', '', answer).strip()
        if answer:
            if q_id in answers:
                answers[q_id] = answers[q_id] + ' ' + answer
            else:
                answers[q_id] = answer

    if not answers:
        answers = _fallback_parse(text)

    return answers


def _fallback_parse(text: str) -> Dict[str, str]:
    parts = re.split(r'(?m)^\s*(\d+)\s*[.):\-]', text)
    answers = {}
    i = 1
    while i < len(parts) - 1:
        q_num = parts[i].strip()
        answer = parts[i + 1].strip()
        if q_num.isdigit() and answer:
            answers[q_num] = answer
        i += 2
    return answers


def display_parsed_answers(answers: Dict[str, str], title: str = "Parsed Answers"):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    if not answers:
        print("  ⚠️  No answers parsed — check OCR output above")
    else:
        for q_id in sorted(answers.keys(), key=lambda x: (len(x), x)):
            preview = answers[q_id][:80] + ("..." if len(answers[q_id]) > 80 else "")
            print(f"  Q{q_id}: {preview}")
    print(f"{'='*55}\n")
