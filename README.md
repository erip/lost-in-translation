# What’s Lost in Translation? Characterizing the Impact of Machine Translation as Cross-lingual Normalization on Text Classification

This repo contains the code for testing the impact of machine translation artifacts on downstream text classification.

## Translating a file

Assuming `text.es` contains the newline-delimited segments to be translated from `es_XX` (mBART's language token for Spanish) to `en_XX` (mBART's language token for English), issue:

```sh
python translate.py --in-file text.es --out-file text.en --src es_XX --tgt en_XX
```

To see a full list of options when translating, issue

```sh
python translate.py -h
```

## Training a translate-train model

Assuming your training text originally lives in `train.es` and the corresponding labels live in `train.txt` and the validation file lives in `dev.es` and corresponding labels live in `dev.txt`

```sh
# Translates the training data into English
python translate.py --in-file train.es --out-file train.en --src es_XX --tgt en_XX

# Note that the training file is the newly created train.en
python train.py --training-text-file train.en --training-label-file train.txt --develop-text-file dev.es --develop-label-file dev.txt --output-dir ./translate-train_es_en
```

