
# 🧾 Document Extraction Demo with LandingAI and OpenAI

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
├── groundings/                       # Saved Visual Groundings from each page
│   └── KYC_EXAMPLE_DOC_20250411_121653
│       ├── page_0
│           ├──0c0c8571-d356-41ab-841b-84f0e04a4651_0.png
│           ├──75d1f906-7b73-4fb4-985d-52fddb7b5e71_0.png
│           └──0519b915-80a4-4a1f-bbc6-e2fc59af348c_0.png
│       └── page_1
│           ├──6a6d1924-d57c-4a25-860c-1249562be730_0.png
│           ├──0353f668-a4f5-4b0e-8413-fc6d913c7ba0_0.png
│           ├──979c6048-7c7a-45da-99a7-b208a9b1c714_0.png
│           ├──26939ba0-220a-4cbb-8bff-be13265c6364_0.png
│           ├──aa2878ff-30e9-414b-932a-96d1d6cc8f86_0.png
│           └──e4ad3629-4341-4f08-8a20-1cc05ef1519f_0.png
├── visualizations/                  # Saved pages with Visual Groundings overlayed
│   ├── KYC_EXAMPLE_DOC_viz_page_0.png
│   └── KYC_EXAMPLE_DOC_viz_page_1.png
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/landing-ai/helper-scripts.git
cd Agentic_Doc_Extraction_Demo
```

### 2. Set up your environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file or update `config.py` with your credentials:
```python
# Example config.py file
API_KEYS = {
    # Add your Agentic Document Extraction API Key
    "ade": "your_ade_api_key_here", 
    # Add your VisionAgent API Key
    "va": "your_va_api_key_here",
    # Add your OpenAI Key
    "openai": "your_openai_key_here"
}
```
OR

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

For internal demo purposes only. Reach out to `yoursupport@yourdomain.com` for questions.
