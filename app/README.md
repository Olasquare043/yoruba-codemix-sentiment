---
title: Yoruba Tweet Sentiment
emoji: 🗣️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Sentiment of Yoruba and code-mixed Yoruba-English tweets
---

# Yorùbá and code-mixed tweet sentiment

Demo for the CSC790 Deep Learning project *Deep Neural Networks for Semantic Sentiment Analysis in Code-Mixed English-Yorùbá Social Media Conversations* (Saheed Olayemi Olayinka and Samson Akinyele Fagbenjo, University of Ibadan).

Type a tweet and the app shows:

- the predicted sentiment (negative, neutral or positive) and the probability of each class, from [Olasquare/afriberta-large-yoruba-sentiment](https://huggingface.co/Olasquare/afriberta-large-yoruba-sentiment);
- a code-mixing check: the tweet's Code-Mixing Index and each word tagged Yorùbá, English or neutral, using the word-list method from the project's notebook 01.

Nothing you type is logged or stored.

Code, annotations and the full results: https://github.com/Olasquare043/yoruba-codemix-sentiment

## Run locally

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py                 # then open http://127.0.0.1:7860
```

`yor_stopwords.csv` is the Yorùbá stopword list from [NaijaSenti](https://github.com/hausanlp/NaijaSenti), included so the app needs no download for it.
