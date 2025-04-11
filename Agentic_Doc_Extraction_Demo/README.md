
# 🧾 Document Extraction Demo with LandingAI

This demo showcases an **agentic document processing pipeline** that extracts structured fields and supports question-answering (Q&A) from institutional KYC documents using the following components:

- [LandingAI](https://landing.ai): Document extraction (Python SDK + REST API)
- [OpenAI GPT-4](https://platform.openai.com): Field extraction, Q&A, and evaluation
- [LangChain](https://www.langchain.com/): Evaluation framework
- 🖼️ Visual Grounding with overlaid bounding boxes
- 📊 Embedded charts and QA results

---

## 📁 File Structure

```
.
├── Doc_Extraction_Demo_vf.ipynb      # Main notebook with REST + SDK pipeline
├── config.py                         # API keys and environment variables
├── KYC_EXAMPLE_DOC.pdf               # Sample institutional KYC document
├── LandingAI_Logo.svg                # Company logo for branding
├── visualizations/
│   ├── KYC_EXAMPLE_DOC_viz_page_0.png
│   └── KYC_EXAMPLE_DOC_viz_page_1.png
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/your-repo-name.git
cd your-repo-name
```

### 2. Set up your environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file or update `config.py` with your credentials:

```python
# config.py
import os
from dotenv import load_dotenv
load_dotenv()

LANDINGAI_API_KEY = os.getenv("LANDINGAI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 🧠 Pipeline Overview

### 1. Load and visualize the KYC PDF

- Uses `PIL.Image` and `matplotlib` for displaying document pages

### 2. Extract structured fields

- LandingAI's document model extracts text + bounding boxes
- Results are visualized and overlaid on the original PDF

### 3. Perform Q&A with OpenAI

- User-defined prompts for field-level Q&A
- Structured output as JSON for downstream use

### 4. Evaluate Results with LangChain

- Ground truth questions vs. predicted answers
- Automated grading using `QAGradeChain`

---

## 🧪 Example Output

- Extracted Fields: `Client Name`, `Registration Number`, `Industry Sector`, etc.
- Visualizations: Bounding boxes showing where values were pulled from
- Q&A Response:
  ```json
  {
    "query": "What is the client's LEI?",
    "predicted_answer": "5493001KJTIIGC8Y1R12",
    "evaluation": "Correct"
  }
  ```

---

## 🧩 Optional: SDK vs REST API

The notebook supports both LandingAI's Python SDK and direct REST API calls to demonstrate flexibility in deployment.

---

## 🛡️ Disclaimer

This demo uses **fake data** and is intended for educational and testing purposes only.

---

## 📬 Contact

For internal demo purposes only. Reach out to LandingAI for questions.
