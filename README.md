# Does Code-Mixing Hurt? English-Yorùbá Code-Mixing in AfriSenti

Code and data for the CSC790 Deep Learning project *Deep Neural Networks for Semantic Sentiment Analysis in Code-Mixed English-Yorùbá Social Media Conversations* (Saheed Olayemi Olayinka and Samson Akinyele Fagbenjo, Department of Computer Science, University of Ibadan, 2026).

We add a human code-mixing annotation layer to the Yorùbá test set of [AfriSenti-SemEval 2023](https://github.com/afrisenti-semeval/afrisent-semeval-2023) and compare six sentiment models (LSTM, BiLSTM with attention, mBERT, XLM-R base, AfriBERTa-large, AfroXLMR-large), five seeds each, on mixed and non-mixed tweets.

## Main results

| | |
|---|---|
| Test tweets annotated | 2,378 of 4,515 (all 1,978 heuristic candidates, plus a random audit of 400) |
| Agreement before discussion (292 shared tweets) | 92.8%, Cohen's kappa 0.71 (95% CI 0.59 to 0.82) |
| Mixed tweets | 269, i.e. 6.0% of the test set (95% CI 6.0% to 6.5%) |
| Yorùbá line followed by its English translation | 312 tweets, 6.9% |
| Lexicon heuristic (CMI above 0) | precision 0.14, recall 1.00 |

| Model | Weighted F1, full test (mean ± SD, 5 seeds) | Mixed | Other | Gap (95% CI) |
|---|---|---|---|---|
| AfroXLMR-large | 73.6 ± 0.8 | 73.2 | 73.6 | -0.4 (-5.3 to 5.0) |
| AfriBERTa-large | 72.5 ± 0.5 | 71.3 | 72.5 | -1.3 (-5.6 to 4.8) |
| LSTM | 67.8 ± 0.4 | 63.6 | 68.1 | -4.5 (-9.7 to 1.9) |
| BiLSTM + attention | 67.2 ± 1.4 | 62.5 | 67.5 | -4.9 (-9.2 to 2.4) |
| mBERT | 63.6 ± 1.7 | 59.2 | 63.9 | -4.7 (-10.8 to 2.0) |
| XLM-R base | 61.7 ± 2.9 | 64.8 | 61.5 | +3.3 (-3.5 to 9.2) |

No model scores significantly lower on mixed tweets.

## Repository layout

```
data/
  annotations/
    afrisenti_yor_test_codemixing_v1.csv     final code-mixing labels (one row per annotated test tweet)
    double_annotated_before_discussion.csv  both annotators' labels on the shared tweets, before discussion
    error_categories.csv                    categories for the best model's 67 errors on mixed tweets
    GUIDELINES.md                           annotation rules
  download_afrisenti.py                     fetches the AfriSenti Yorùbá files into data/raw/
notebooks/
  01_data_cmi_annotation_baseline.ipynb     data loading, duplicate audit, Code-Mixing Index, first annotation sheets
  02_train_all_models.ipynb                 trains all six models with five seeds (GPU; Colab or Kaggle)
  02b_kaggle_remaining_runs.ipynb           Kaggle copy used for the last eight runs
  03_results_and_figures.ipynb              all tables and figures of the report (no GPU; outputs included)
src/
  analysis.py                               agreement, prevalence, mixed vs non-mixed, regression, errors
scripts/
  build_annotation_sheets_v2.py             record of how the second annotation sheets were built
report/
  CSC790_Project_Report.pdf / .docx          the course project report
results/
  all_30_runs.csv                           test scores of every run
  runs/                                     one JSON per run (seed, versions, GPU, best epoch, scores)
  preds/                                    test predictions and class probabilities of every run
  analysis/                                 output of src/analysis.py and notebook 03
  figures/                                  figures used in the report
```

## Annotation file

`afrisenti_yor_test_codemixing_v1.csv` has no tweet text. Rows are keyed by `id`, which follows the row order of the AfriSenti files (train, then dev, then test, so the test set starts at `yor_10612`), and by `text_md5`, the MD5 hash of the tweet after removing double quotes and surrounding spaces. Columns: `cmi` (lexicon Code-Mixing Index), `sampling_group` (`S1_candidate` for CMI above zero, `S0_audit` for the random audit of zero-score tweets), `is_codemixed`, `dominance` (YO, EN or BAL), `translation_pair`, `quoted`, `double_annotated`, `adjudicated`. See `GUIDELINES.md`.

Test tweets that are not in the file have a CMI of zero and were not sampled for the audit.

## Reproducing the analysis

```bash
pip install -r requirements.txt
python data/download_afrisenti.py
python src/analysis.py
```

`src/analysis.py` checks every tweet against its hash, then writes `results/analysis/results.json`. It runs on a laptop in under a minute and needs no GPU. Retraining the models needs the notebooks and a GPU (the 30 runs took about 6 hours of training in total; per-run times are in `results/runs/`).

## Licence and credit

Code: MIT. Annotations: CC BY 4.0. The tweets and sentiment labels belong to AfriSenti-SemEval 2023 (Muhammad et al., 2023, CC BY 4.0), which builds on NaijaSenti (Muhammad et al., 2022). Please cite them as well as this work.

## Citation

```
[to be added after acceptance]
```
