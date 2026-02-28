# 🎓 AI Paper Grading System (Local, No Cloud)

A fully offline, CLI-based AI paper valuation system.  
Uses TrOCR for handwriting recognition + SentenceTransformers for semantic grading.

---

## 📦 What You Need to Install

### 1. System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr python3-pip python3-venv
```

**Windows:**
- Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
- Add it to PATH

**Mac:**
```bash
brew install tesseract
```

### 2. Python Environment

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/Mac
# OR
venv\Scripts\activate      # Windows

# Install all Python dependencies
pip install -r requirements.txt
```

> ⚠️ First run will download models (~1.5GB total). After that, works offline.

### 3. (Optional) Ollama for LLM Feedback

If you want AI-written detailed feedback on each answer:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model (good for 16GB RAM, no GPU)
ollama pull llama3.2       # ~2GB, fast
# OR
ollama pull phi3            # ~2.3GB, very good

# Start Ollama server
ollama serve
```

---

## 📁 Project Structure

```
paper_grader/
├── main.py                          # CLI entry point
├── requirements.txt
│
├── ocr/
│   └── handwriting_ocr.py           # TrOCR + EasyOCR
│
├── grader/
│   └── grading_engine.py            # Semantic similarity + LLM
│
├── utils/
│   ├── answer_parser.py             # Q&A text extraction
│   └── report_generator.py          # Output formatting
│
├── answer_keys/
│   └── example_science.yaml         # Example answer key
│
├── sample_papers/
│   └── sample_student_text.txt      # Test without scanner
│
└── outputs/                         # Generated results (auto-created)
```

---

## 🚀 Quick Start

### Run Demo (No camera/scanner needed)
```bash
python main.py demo
```

### Grade a Scanned Image
```bash
python main.py grade --image student_paper.jpg --key answer_keys/example_science.yaml --student-id "Roll101"
```

### Grade a Text File (if OCR already done)
```bash
python main.py grade --text my_student.txt --key answer_keys/example_science.yaml --student-id "Roll101"
```

### Grade with LLM Feedback (needs Ollama running)
```bash
python main.py grade --text my_student.txt --key answer_keys/example_science.yaml --llm
```

### Batch Grade (entire folder of papers)
```bash
python main.py batch --folder papers/ --key answer_keys/example_science.yaml
```

### Create a New Answer Key
```bash
python main.py create-key
```

---

## 📝 Answer Key Format (YAML)

```yaml
"1":
  answer: "Full correct answer text here"
  marks: 10
  keywords:
    - important_word1
    - important_word2

"2":
  answer: "Another correct answer"
  marks: 5

"3a":
  answer: "Sub-question answer"
  marks: 3
```

---

## 📄 Student Paper Format

The system detects these question number formats automatically:
- `1.`  `1)` `1:`
- `Q1.` `Q1:` `Question 1:`
- `(a)` `(b)` `(i)`
- `1a.` `1(a):`
- `Answer 1:` `Ans 1.`

Example student paper:
```
1. Photosynthesis is when plants make food from sunlight...

2. Newton's first law states that objects in motion...

3a. ATP means Adenosine Triphosphate.

3b. Respiration occurs in the mitochondria.
```

---

## 🔧 Grading Logic

1. **OCR**: TrOCR extracts handwritten text from scanned images
2. **Parsing**: Regex maps text to question numbers
3. **Grading**: SentenceTransformer computes semantic similarity
4. **Scoring curve**:
   - ≥ 85% similarity → 100% marks
   - 70-85% → 80-100% marks
   - 55-70% → 60-80% marks
   - 40-55% → 40-60% marks
   - < 25% → 0-20% marks
5. **Feedback**: Simple rule-based (or LLM if Ollama running)

---

## 💻 Hardware Notes

| Mode | RAM | GPU | Speed |
|------|-----|-----|-------|
| Text grading only | 4GB | None | ~1s/paper |
| OCR (EasyOCR) | 4GB | None | ~10s/page |
| OCR (TrOCR) | 8GB | Optional | ~15s/page |
| LLM Feedback | 8GB+ | Optional | ~5s/answer |

Your 16GB RAM is more than enough for all modes.

---

## 🐛 Troubleshooting

**"No answers could be parsed"**  
→ Check that your paper uses supported question numbering formats  
→ Try running with `--text` mode and pre-format the text

**TrOCR download fails**  
→ Check internet connection for first run  
→ Models are cached at `~/.cache/huggingface/` after first download

**Ollama connection refused**  
→ Make sure you ran `ollama serve` in a separate terminal  
→ System will fall back to simple feedback automatically

**Low accuracy on handwriting**  
→ Make sure image is well-lit and high resolution (300+ DPI recommended)  
→ TrOCR large model gives better results than base model

---

## 🔮 Future Enhancements (Web GUI)
- Upload papers via browser
- Live grading dashboard
- Student portal for viewing results
- Plagiarism detection
- Analytics and grade distributions
