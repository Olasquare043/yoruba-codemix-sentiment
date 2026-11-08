# Code-mixing annotation guidelines

Unit: one tweet. Annotators did not see the sentiment label or the heuristic score.
Both annotators read and write Yorùbá fluently.

## Columns

| Column | Values | Meaning |
|---|---|---|
| `is_codemixed` | 1 / 0 | 1 = the writer naturally mixes Yorùbá and English in the tweet |
| `dominance` | YO / EN / BAL | Only when mixed. YO or EN when one language clearly dominates (about two thirds of the words or more); BAL otherwise. Hashtags, links, names and user names are not counted |
| `translation_pair` | 1 / 0 | 1 = a Yorùbá line followed by its English translation (always labelled `is_codemixed = 0`) |
| `quoted` | 1 / 0 | 1 = mixed because an English sentence is quoted inside Yorùbá text (for example "translate this into Yorùbá: life is vanity") |

## Rules

Label **1** for natural mixing within or across sentences, including an English sentence quoted inside Yorùbá text.

Label **0** for:
- text in one language only;
- a Yorùbá line followed by its English translation (`translation_pair = 1`);
- English only in hashtags, links, names or user names;
- a single ordinary borrowed word ("phone", "okay", "daddy").

If English is added that is not a translation of the Yorùbá (a comment, a reaction, a title such as "today's Yorùbá motivation"), the tweet is mixed.

Judge the language only, never the sentiment.

## Agreement

293 tweets were labelled by both annotators (292 completed). Agreement before discussion:
92.8% raw, Cohen's kappa 0.71 (95% bootstrap CI 0.59 to 0.82). Disagreements were settled by discussion;
those tweets are flagged `adjudicated = 1` in the release file, and the labels before discussion are in
`double_annotated_before_discussion.csv`.
