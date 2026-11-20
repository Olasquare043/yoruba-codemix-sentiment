"""Demo app: sentiment of Yoruba and code-mixed Yoruba-English tweets.

Runs on CPU (Hugging Face Space, Gradio SDK). Nothing the user types is logged, stored or sent anywhere:
the text goes only through the model and the code-mixing functions in this process.
"""
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"   # no Gradio usage analytics
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import json
import re
import unicodedata
from pathlib import Path

import gradio as gr
import pandas as pd
from english_words import get_english_words_set
from huggingface_hub import hf_hub_download
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "Olasquare/afriberta-large-yoruba-sentiment")
GITHUB = "https://github.com/Olasquare043/yoruba-codemix-sentiment"
LABELS = ["negative", "neutral", "positive"]
HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Code-Mixing Index: copied from notebooks/01_data_cmi_annotation_baseline.ipynb
# (the stopword list is the NaijaSenti Yoruba list, vendored in yor_stopwords.csv)
# ---------------------------------------------------------------------------

def strip_marks(w):
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")

ENG = {w for w in get_english_words_set(["web2"], lower=True, alpha=True) if len(w) >= 3}
ENG |= {"lol", "omg", "pls", "abeg", "sha", "wey", "dey", "abi", "guy", "guys", "bro", "okay", "ok", "thanks", "u", "ur", "im", "dont", "cant", "wont"}

yo_stop = pd.read_csv(HERE / "yor_stopwords.csv")["word"].astype(str).str.lower().tolist()
YOR = set(yo_stop) | {strip_marks(w) for w in yo_stop}
# Common Yoruba words that collide with English dictionary entries
YOR |= {"ni", "si", "o", "e", "a", "mi", "wa", "fun", "ko", "lo", "ba", "ti", "to", "se", "je", "re", "won", "omo", "oba", "owo", "ile", "ise", "oro", "ogun", "ojo", "awo", "ewe", "obi", "ore", "ara"}
YO_MARKS = re.compile(r"[ẹọṣàáèéìíòóùúńǹ]")

def tag_tokens(text):
    tags = []
    for tok in text.lower().split():
        if tok.startswith(("@", "#", "http")) or tok == "rt":
            tags.append("univ"); continue
        w = re.sub(r"[^\w']", "", tok)
        if not w or w.isdigit():
            tags.append("univ"); continue
        if YO_MARKS.search(w) or w in YOR:
            tags.append("yo")
        elif w in ENG:
            tags.append("en")
        else:
            tags.append("yo")   # unknown ASCII words in this corpus are mostly unaccented Yoruba
    return tags

def cmi(text):
    t = tag_tokens(text)
    n = len(t); u = t.count("univ")
    if n - u == 0:
        return 0.0, 0, 0
    en, yo = t.count("en"), t.count("yo")
    return round(100 * (1 - max(en, yo) / (n - u)), 1), en, yo

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

clf = pipeline("text-classification", model=MODEL_ID, top_k=None, device=-1)


def load_test_scores():
    """The model's own test scores, published next to it by notebook 04 (test_scores.json)."""
    try:
        with open(hf_hub_download(MODEL_ID, "test_scores.json")) as f:
            s = json.load(f)["test"]
        return (f"On the full AfriSenti Yorùbá test set (4,515 tweets) this model scores **{100*s['weighted_f1']:.1f} weighted F1**, "
                f"{100*s['macro_f1']:.1f} macro F1 and {100*s['accuracy']:.1f}% accuracy. "
                "It is a single training run, so it differs slightly from the 5-run average in the report (72.5).")
    except Exception:
        return "This model's own test score is shown on its [model page](https://huggingface.co/" + MODEL_ID + ")."


TAG_NAMES = {"yo": "Yoruba", "en": "English", "univ": "neutral"}
TAG_COLORS = {"Yoruba": "#2e9e6b", "English": "#3b6fd8", "neutral": "#9aa0a6"}


def analyse(text):
    text = (text or "").strip()
    if not text:
        return "Type or paste a tweet first.", None, "", []

    scores = {r["label"]: float(r["score"]) for r in clf(text, truncation=True, max_length=128)[0]}
    top = max(scores, key=scores.get)
    verdict = f"## {top.capitalize()}\nConfidence: {100*scores[top]:.0f}%"

    index, n_en, n_yo = cmi(text)
    words = text.split()
    tagged = [(w + " ", TAG_NAMES[t]) for w, t in zip(words, tag_tokens(text))]
    if index == 0:
        level = "one language only"
    elif index < 20:
        level = "lightly mixed"
    elif index < 35:
        level = "mixed"
    else:
        level = "heavily mixed"
    mix = (f"**Code-Mixing Index: {index:g}** ({level}; 0 means a single language, the maximum is 50). "
           f"Yoruba words: {n_yo}, English words: {n_en}.\n\n"
           "This is the rough word-list method used to find candidate tweets in the project. "
           "It flags too much mixing (only about 1 in 7 flagged tweets was truly mixed when we checked by hand). "
           "Short English words (one or two letters, such as *is*, *in*, *me*) and unknown words are tagged Yoruba, "
           "so English-only tweets can still get a score above 0.")
    return verdict, scores, mix, tagged


EXAMPLES = [
    ["Ọlọ́run á bá wa ṣe é, a ò ní kú ní kékeré"],
    ["Inú mi kò dùn rárá, wọ́n ti jí fóònù mi"],
    ["yay he did it jẹ́ ń lọ gbàlù kí n jó"],
    ["ó yá follow me back ẹ ṣeun"],
    ["The traffic in Lagos today is terrible"],
    ["Àgbà kì í wà lọ́jà kí orí ọmọ tuntun wó"],
]

ABOUT = f"""
## About this project

**Research question.** Many Nigerians mix Yorùbá and English in the same tweet. Does this code-mixing make sentiment
models less accurate, which kind of model works best on Yorùbá tweets, and do the stronger models understand local
expressions such as proverbs and idioms better?

We added human code-mixing labels to the Yorùbá test set of AfriSenti-SemEval 2023 (two annotators, Cohen's kappa 0.71)
and compared six models (LSTM, BiLSTM with attention, mBERT, XLM-R base, AfriBERTa-large, AfroXLMR-large), each trained five times.

**What we found**

1. **Real code-mixing is less common than it looks, and it does not hurt.** Only about 6% of test tweets are truly mixed.
   No model scores significantly lower on them than on the other tweets.
2. **The two African-language models are best, and tied.** AfroXLMR-large (73.6 weighted F1) and AfriBERTa-large (72.5)
   are statistically tied and clearly ahead of the general multilingual models and the LSTMs.
3. **Contextual models cope better with proverbs and idioms.** Every model does worse on tweets about proverbs, idioms and
   sayings, but the fine-tuned transformers lose about 5 to 10 points, while the LSTMs lose 13 to 15.

**This demo's model.** AfriBERTa-large fine-tuned with the same recipe as in the project (seed 42).
{{SCORES}}

**Limitations.** The model was trained only on tweets from one platform. Its main errors are on proverbs, sarcasm and
slang. It has only three classes, so mixed feelings are forced into one label. It is for research and demonstration,
not for decisions about people.

**Privacy.** What you type is not logged, stored or shared. It is only passed through the model to produce the result.

**Authors.** Saheed Olayemi Olayinka and Samson Akinyele Fagbenjo, Department of Computer Science, University of Ibadan.
Data: AfriSenti-SemEval 2023 (Muhammad et al., 2023, CC BY 4.0). Code and annotations: [GitHub repository]({GITHUB}).
"""

with gr.Blocks(title="Yorùbá Tweet Sentiment", analytics_enabled=False) as demo:
    gr.Markdown("# Yorùbá and code-mixed tweet sentiment\n"
                "Type a tweet in Yorùbá, English, or a mix of both. The model says whether it is negative, neutral or positive, "
                "and the code-mixing check shows which words look Yorùbá and which look English.")
    with gr.Tab("Analyse"):
        with gr.Row():
            with gr.Column():
                inp = gr.Textbox(label="Tweet", lines=3, max_length=500,
                                 placeholder="e.g. ó yá follow me back ẹ ṣeun")
                btn = gr.Button("Analyse", variant="primary")
                gr.Examples(EXAMPLES, inputs=inp, cache_examples=False,
                            label="Examples (2 Yorùbá, 2 code-mixed, 1 English, 1 proverb)")
            with gr.Column():
                verdict = gr.Markdown(label="Predicted sentiment")
                probs = gr.Label(label="Probability of each class", num_top_classes=3)
        gr.Markdown("### Code-mixing check")
        mix = gr.Markdown()
        tagged = gr.HighlightedText(label="Each word tagged (neutral = names, hashtags, links, numbers, emoji)",
                                    color_map=TAG_COLORS, show_legend=True, combine_adjacent=False)
        btn.click(analyse, inputs=inp, outputs=[verdict, probs, mix, tagged])
        inp.submit(analyse, inputs=inp, outputs=[verdict, probs, mix, tagged])
    with gr.Tab("About this project"):
        gr.Markdown(ABOUT.replace("{SCORES}", load_test_scores()))

if __name__ == "__main__":
    # SHARE=1 gives a temporary public gradio.live link (up to 72 hours) while this runs
    demo.launch(share=os.environ.get("SHARE") == "1")
