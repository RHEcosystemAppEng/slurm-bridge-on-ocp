# Training data — 20 Newsgroups

`train.csv` (10,952 rows) and `test.csv` (7,269 rows) are the
[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) dataset, a classic
benchmark for text classification. Posts shorter than 20 characters and
quoting/signature headers are stripped.

- **Columns:** `text`, `label`
- **Labels (20 categories):**
  - 0: alt.atheism, 1: comp.graphics, 2: comp.os.ms-windows.misc,
    3: comp.sys.ibm.pc.hardware, 4: comp.sys.mac.hardware,
    5: comp.windows.x, 6: misc.forsale, 7: rec.autos,
    8: rec.motorcycles, 9: rec.sport.baseball, 10: rec.sport.hockey,
    11: sci.crypt, 12: sci.electronics, 13: sci.med, 14: sci.space,
    15: soc.religion.christian, 16: talk.politics.guns,
    17: talk.politics.mideast, 18: talk.politics.misc,
    19: talk.religion.misc
- **Size:** ~22 MB total (14 MB train, 8 MB test)
- **Balance:** ~480-590 train rows per category
- **Use case:** Multi-topic text classification across 20 distinct Usenet
  newsgroups. Pass `--dataset news20` to the demo script, or
  `--data-dir /app/data-news` to `train.py` directly.
- **License:** Free for educational/research use (see
  [original source](http://qwone.com/~jason/20Newsgroups/)).

A `labels.json` file is included so `train.py` auto-detects the 20 category
names instead of using the default AG News labels.
