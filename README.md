# Analyzing-Political-Bias-in-News-Using-NLP-and-LLMs


## 🔍 Overview
This project analyzes political bias in news articles using advanced NLP models. It goes beyond simple left/right labeling to assess emotional tone, sensationalism, and ideological framing. We built a multi-task pipeline that performs:

- 🔴🟢🔵 Political Bias Classification (Left, Center, Right)
- 😠😢😊 Emotion Detection using BERT and GPT-3.5 Turbo
- 🚨 Sensationalism Scoring
- 🔁 Style Transfer (e.g., rewriting a Right-leaning article in a Left tone)

A user-friendly Streamlit interface lets users input a URL and receive an instant bias analysis.

---

## 🧠 Key Features

- **Bias Classification**: Fine-tuned `bert-base-cased` with an ANN head
- **Emotion Detection**: Compared results using `distilbert-base-uncased-emotion` and GPT-3.5 Turbo
- **Style Transfer**: GPT-3.5 rephrases articles to reflect opposite political tone
- **Semantic Similarity**: Cosine similarity measures the preservation of core facts
- **Streamlit UI**: Paste a news article URL and get an automated analysis

---

## 🧰 Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- OpenAI GPT-3.5 API
- Streamlit
- newspaper3k

---


