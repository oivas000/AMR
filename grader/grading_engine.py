"""
Grading Engine
Uses SentenceTransformers for semantic similarity scoring.
Optional: Ollama LLM for detailed feedback.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class QuestionResult:
    question_id: str
    student_answer: str
    correct_answer: str
    marks_awarded: float
    max_marks: float
    similarity_score: float
    grade: str
    feedback: str = ""


@dataclass
class GradingResult:
    student_id: str
    question_results: Dict[str, QuestionResult] = field(default_factory=dict)
    total_marks: float = 0
    max_marks: float = 0
    percentage: float = 0
    grade: str = ""
    overall_feedback: str = ""


class SemanticGrader:
    """
    Grades answers using semantic similarity via SentenceTransformers.
    No GPU needed - runs on CPU, very fast.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, good accuracy

    def __init__(self):
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"[Grader] Loading SentenceTransformer: {self.MODEL_NAME}")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(self.MODEL_NAME)
        print("[Grader] Grading model loaded ✅")
        self._loaded = True

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts (0.0 - 1.0)"""
        from sklearn.metrics.pairwise import cosine_similarity
        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])
        score = cosine_similarity(emb1, emb2)[0][0]
        return float(np.clip(score, 0, 1))

    def score_answer(
        self,
        student_answer: str,
        correct_answer: str,
        max_marks: float = 10.0,
        keywords: Optional[list] = None,
    ) -> Tuple[float, float, str]:
        """
        Score a single answer.
        Returns: (marks_awarded, similarity_score, grade_letter)
        """
        self.load()

        if not student_answer.strip():
            return 0.0, 0.0, "F"

        # Base similarity score
        sim = self.similarity(student_answer, correct_answer)

        # Keyword bonus (if keywords specified in answer key)
        keyword_bonus = 0.0
        if keywords:
            student_lower = student_answer.lower()
            matched = sum(1 for kw in keywords if kw.lower() in student_lower)
            keyword_bonus = 0.1 * (matched / len(keywords))  # up to 10% bonus
            sim = min(1.0, sim + keyword_bonus)

        # Convert similarity to marks
        marks = self._sim_to_marks(sim, max_marks)
        grade = self._marks_to_grade(marks / max_marks)

        return marks, sim, grade

    def _sim_to_marks(self, similarity: float, max_marks: float) -> float:
        """
        Convert similarity score to marks.
        Uses a curve to avoid being too harsh on partial answers.
        """
        # Scoring thresholds
        if similarity >= 0.85:
            ratio = 1.0
        elif similarity >= 0.70:
            ratio = 0.8 + (similarity - 0.70) * (0.2 / 0.15)
        elif similarity >= 0.55:
            ratio = 0.6 + (similarity - 0.55) * (0.2 / 0.15)
        elif similarity >= 0.40:
            ratio = 0.4 + (similarity - 0.40) * (0.2 / 0.15)
        elif similarity >= 0.25:
            ratio = 0.2 + (similarity - 0.25) * (0.2 / 0.15)
        else:
            ratio = similarity * (0.2 / 0.25)

        marks = round(ratio * max_marks, 1)
        return marks

    def _marks_to_grade(self, ratio: float) -> str:
        if ratio >= 0.90: return "A+"
        if ratio >= 0.80: return "A"
        if ratio >= 0.70: return "B"
        if ratio >= 0.60: return "C"
        if ratio >= 0.50: return "D"
        return "F"


class LLMFeedbackGenerator:
    """
    Uses local llama-server (llama.cpp) with Qwen model to generate feedback.
    llama-server exposes an OpenAI-compatible /v1/chat/completions endpoint.
    Falls back gracefully if server not running.

    Start server with:
      ./llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --host 0.0.0.0 --port 8080 -c 4096
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")
        self.available = None

    def check_available(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            import requests
            # llama-server health endpoint
            r = requests.get(f"{self.base_url}/health", timeout=3)
            self.available = r.status_code == 200
        except:
            self.available = False
        if not self.available:
            print("[LLM] llama-server not reachable at", self.base_url,
                  "— using simple feedback instead.")
        return self.available

    def generate_feedback(
        self,
        question_id: str,
        student_answer: str,
        correct_answer: str,
        marks: float,
        max_marks: float,
    ) -> str:
        """Generate feedback using Qwen via llama-server OpenAI-compatible API"""
        if not self.check_available():
            return self._simple_feedback(marks, max_marks)

        try:
            import requests

            system_prompt = (
                "You are a helpful, encouraging exam grader. "
                "Give concise, constructive feedback in 2-3 sentences."
            )
            user_prompt = (
                f"Question {question_id}:\n"
                f"Correct Answer: {correct_answer}\n"
                f"Student Answer: {student_answer}\n"
                f"Marks Awarded: {marks}/{max_marks}\n\n"
                "Briefly state: what was correct, what was missing, and one improvement tip."
            )

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "max_tokens": 150,
                    "temperature": 0.3,
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code == 200:
                return (
                    response.json()["choices"][0]["message"]["content"].strip()
                )
        except Exception as e:
            print(f"[LLM] Feedback generation error: {e}")

        return self._simple_feedback(marks, max_marks)

    def _simple_feedback(self, marks: float, max_marks: float) -> str:
        ratio = marks / max_marks if max_marks > 0 else 0
        if ratio >= 0.85:
            return "Excellent answer! Very well covered."
        elif ratio >= 0.70:
            return "Good answer. Minor details missing."
        elif ratio >= 0.50:
            return "Partial answer. Key concepts present but incomplete."
        elif ratio >= 0.25:
            return "Incomplete answer. Review the topic more carefully."
        else:
            return "Answer needs significant improvement. Please revisit this topic."


class PaperGrader:
    """
    Main grading orchestrator.
    Combines semantic grading + optional LLM feedback.
    """

    def __init__(self, use_llm_feedback: bool = False, llama_server_url: str = "http://localhost:8080"):
        self.semantic_grader = SemanticGrader()
        self.llm = LLMFeedbackGenerator(base_url=llama_server_url)
        self.use_llm_feedback = use_llm_feedback

    def grade_paper(
        self,
        student_answers: Dict[str, str],
        answer_key: Dict,   # {q_id: {"answer": str, "marks": int, "keywords": list}}
        student_id: str = "Unknown",
    ) -> GradingResult:
        """
        Grade all answers in a paper.
        answer_key format: {
            "1": {"answer": "Paris", "marks": 5, "keywords": ["capital", "france"]},
            "2": {"answer": "...", "marks": 10},
        }
        """
        self.semantic_grader.load()

        result = GradingResult(student_id=student_id)

        if self.use_llm_feedback and self.llm.check_available():
            print("[Grader] LLM feedback enabled ✅")
        elif self.use_llm_feedback:
            print("[Grader] Ollama not running - using simple feedback")

        for q_id, key_data in answer_key.items():
            correct_answer = key_data["answer"]
            max_marks = key_data.get("marks", 10)
            keywords = key_data.get("keywords", [])

            # Get student's answer (try exact match, then fuzzy match)
            student_ans = self._find_student_answer(q_id, student_answers)

            if student_ans is None:
                # Question not answered
                q_result = QuestionResult(
                    question_id=q_id,
                    student_answer="[NOT ANSWERED]",
                    correct_answer=correct_answer,
                    marks_awarded=0.0,
                    max_marks=max_marks,
                    similarity_score=0.0,
                    grade="F",
                    feedback="Question not attempted."
                )
            else:
                marks, sim, grade = self.semantic_grader.score_answer(
                    student_ans, correct_answer, max_marks, keywords
                )

                feedback = ""
                if self.use_llm_feedback:
                    feedback = self.llm.generate_feedback(
                        q_id, student_ans, correct_answer, marks, max_marks
                    )
                else:
                    feedback = self.llm._simple_feedback(marks, max_marks)

                q_result = QuestionResult(
                    question_id=q_id,
                    student_answer=student_ans,
                    correct_answer=correct_answer,
                    marks_awarded=marks,
                    max_marks=max_marks,
                    similarity_score=round(sim, 3),
                    grade=grade,
                    feedback=feedback
                )

            result.question_results[q_id] = q_result
            result.max_marks += max_marks
            result.total_marks += q_result.marks_awarded

        result.percentage = round(
            (result.total_marks / result.max_marks * 100) if result.max_marks > 0 else 0, 1
        )
        result.grade = self._overall_grade(result.percentage)

        return result

    def _find_student_answer(self, q_id: str, student_answers: Dict[str, str]) -> Optional[str]:
        """Try to match question ID with some flexibility"""
        # Exact match
        if q_id in student_answers:
            return student_answers[q_id]

        # Normalize both sides
        q_norm = q_id.lower().strip()
        for k, v in student_answers.items():
            if k.lower().strip() == q_norm:
                return v

        # Try numeric equivalence (e.g., "01" == "1")
        try:
            q_int = str(int(q_id))
            for k, v in student_answers.items():
                try:
                    if str(int(k)) == q_int:
                        return v
                except:
                    pass
        except:
            pass

        return None

    def _overall_grade(self, percentage: float) -> str:
        if percentage >= 90: return "A+"
        if percentage >= 80: return "A"
        if percentage >= 70: return "B"
        if percentage >= 60: return "C"
        if percentage >= 50: return "D"
        return "F"
