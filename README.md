# SatQuery — Satellite Image Question Answering

Ask a question about a satellite image in plain English and get an answer, a
confidence score, and a map that shows where the answer came from.

SatQuery is a **Visual Question Answering (VQA)** system for satellite imagery.
It brings together **computer vision** (understanding the image) and **NLP**
(understanding the question) to answer questions about what an image shows and
how a place has changed over time.

---

## What you can ask

You give SatQuery a satellite image, or two images of the same place taken on
different dates, and ask questions such as:

- "Is there water in this image?"
- "What type of land cover is this — forest, water, urban or farmland?"
- "Is there a built-up area in this image?"
- "What changed between these two dates, and where?"

For every question it returns:

- a short, clear answer,
- a confidence level (high / medium / low),
- the areas it looked at, highlighted on the image,
- a step-by-step record of how it reached the answer,
- a downloadable report (PDF, map file or JSON).

No images of your own? The app ships with ready-made demo scenes, including a
real Sentinel-2 before/after pair of an area under construction.

---

## How it works

1. You upload an image (or pick a demo) and type a question.
2. The system reads the question and works out what is being asked.
3. It chooses the right analysis and runs trained AI models on the image.
4. It checks the results and returns the answer with a confidence score and a map.

Everything runs on your own computer. No internet connection or API key is
needed, and the same image always gives the same answer.

---

## The AI models

SatQuery uses three models trained on public satellite-image datasets:

| Model | What it does | Accuracy |
|---|---|---|
| Scene classifier | Identifies the land type (forest, water, urban, farmland, ...) | 95.7% |
| Question answering | Answers questions about a single image | 84.6% |
| Change detection | Answers questions about what changed between two dates | 65% |

Accuracy was measured on test images the models never saw during training.

---

## Getting started

You need **Python 3.10 or newer**.

1. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
2. Start the app:
   ```
   python run.py
   ```
3. Open the link it prints (usually http://localhost:8000), choose a demo
   scene, type a question, and press **Analyse**.

To train the models on your own machine (one time):
```
python run.py --setup-models
```

To run the automated tests:
```
python run.py --test
```

---

## Project layout

| Folder | What's inside |
|---|---|
| `satquery/` | The application: question handling, models, tools, web server |
| `frontend/` | The web interface (React) |
| `training/` | Scripts that train the AI models |
| `eval/` | Scripts that measure accuracy |
| `demo_data/` | The demo scenes |
| `tests/` | Automated tests |

---

## Built with

Python and FastAPI (backend), React (frontend), and PyTorch (models).
