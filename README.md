# CRAC 2026 Empty Nodes Baseline

This repository contains the source code of the CRAC 2026 Empty Nodes Baseline
system for predicting empty nodes in the input CoNLL-U files. The source is
available under the MPL-2.0 license, and the pre-trained model under the
CC BY-NC-SA license.

Compared to the [last year CRAC 2025 Empty Nodes Baseline](https://github.com/ufal/crac2025_empty_nodes_baseline),
this year's baseline predicts **all available information for the empty nodes**,
i.e., including forms, lemmas, UPOS, XPOS, and FEATS columns, in addition to
previously predicted word order and dependency relations of the empty nodes.

---

## Content of this Repository

- `crac2026_empty_nodes_baseline.py` is the source code of the whole system,
  implemented in PyTorch and [Minnt](https://minnt.org).

- `crac2026_empty_nodes_eval.py` provides evaluation of predicted empty nodes,
  both as a module (used by the `crac2026_empty_nodes_baseline.py`) and also
  as a command-line tool.

## The Released `crac2026_empty_nodes_baseline` Model

The [crac2026_empty_nodes_baseline](https://huggingface.co/ufal/crac2026_empty_nodes_baseline)
is a `XLM-RoBERTa-large`-based multilingual model for predicting empty nodes, trained on CorefUD 1.4 data.
It is released on [LINDAT/CLARIAH-CZ](https://hdl.handle.net/11234/1-6081) and on
[HuggingFace](https://huggingface.co/ufal/crac2026_empty_nodes_baseline) under the CC BY-NC-SA
4.0 license, and it is downloaded automatically by `crac2026_empty_nodes_baseline.py`
when running prediction with the `--load ufal/crac2026_empty_nodes_baseline` argument.

The model was used to generate baseline empty nodes prediction in the
[CRAC 2026 Shared Task on Multilingual Coreference Resolution](https://ufal.mff.cuni.cz/corefud/crac26).

The model is language agnostic, so in theory it can be used to
predict coreference in any `XLM-RoBERTa` language.

## Training a Single Multilingual `XLM-RoBERTa-large`-based Model

To train a single multilingual model on all the data using `XLM-RoBERTa-large`, you should
1. download the CorefUD 1.4 data by running `get.sh` from the `data` directory,
2. create a Python environments with the packages listed in `requirements.txt`,
3. train the model itself using the `crac2026_empty_nodes_baseline.py` script.

The released model has been trained using the following command:
```sh
tbs="ca_ancora cs_pcedt cs_pdt cs_pdtsc cu_proiel es_ancora grc_proiel hu_korkor hu_szegedkoref pl_pcc tr_itcc"
python3 crac2026_empty_nodes_baseline.py $(for mode in train minidev; do echo --${mode#mini}; for tb in $tbs; do echo data/$tb-corefud-$mode.conllu; done; done) --batch_size=96 --max_train_sentence_len=120 --lazy_adam --seed=7 --save_model
```
It assumes the training files are available in `data/{treebank}-corefud-{train/minidev}.conllu`,
with `train` and `minidev` files containing the gold empty nodes.

## Predicting with a Trained Model.

To predict with the released `crac2026_empty_nodes_baseline` model, use the following arguments:
```sh
python3 crac2026_empty_nodes_baseline.py --load ufal/crac2026_empty_nodes_baseline --exp target_directory --test input1.conllu input2.conllu
```
- instead of a HuggingFace identifier, you can use directory name – if the given
  path name exists, model is loaded from it;
- the outputs are generated in the target directory, with `.predicted.conllu` suffix;
- if you want to also evaluate the predicted files, you can use `--dev` option instead of `--test`;
  that way, another file with `.predicted.conllu.eval` suffix will be created by `crac2026_empty_nodes_eval.py`.

## Evaluation of Empty Nodes Prediction Performance

The `crac2026_empty_nodes_eval.py` performs intrinsic evaluation of empty nodes
prediction. It computes F1-score, precision, and recall of several metrics:
- `ARC`: a predicted empty node is considered correct if it has correct
  parent in the `DEPS` column (but not necessarily a correct DEPREL).

For all other metrics, both the parent and one or more other attributes must match:
- `DEP`: also the dependency relation must be correct, i.e., the whole `DEPS` column;
- `WO`: also the word order (the value of the CoNLL-U first column before a dot) must be correct;
- `DEP_WO`: both the dependency relation and the word order must be correct (this
  metric can be used to compare with last year's CRAC 2025 Empty Nodes Baseline);
- `FORM`: also the `FORM` column must be correct;
- `LEMMA`: also the `LEMMA` column must be correct;
- `UPOS`: also the `UPOS` column must be correct;
- `XPOS`: also the `XPOS` column must be correct;
- `FEATS`: also the `FEATS` column must be correct;
- `ALL`: all of the above attributes must be correct.

### Evaluation of the Released `crac2026_empty_nodes_baseline` Model

The following table contains the F1-scores of the released
`crac26_empty_nodes_baseline` model on the CorefUD 1.4 minidev data.

| Treebank       |  ARC   |  DEP   |   WO   | DEP_WO |  FORM  | LEMMA  |  UPOS  |  XPOS  | FEATS  |  ALL   |
|:---------------|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| ca_ancora      | 95.55% | 95.55% | 92.74% | 92.74% | —      | —      | —      | —      | —      | 92.74% |
| cs_pcedt       | 70.91% | 69.36% | 70.77% | 69.21% | 69.07% | 70.77% | 70.91% | 70.91% | 68.08% | 67.80% |
| cs_pdt         | 79.21% | 78.34% | 78.52% | 78.00% | 77.83% | 78.00% | 79.03% | 79.21% | 77.14% | 76.79% |
| cs_pdtsc       | 85.88% | 85.11% | 85.16% | 84.39% | 84.59% | 85.73% | 85.88% | 85.88% | 82.79% | 81.91% |
| cu_proiel      | 80.55% | 79.50% | 79.77% | 78.85% | —      | —      | —      | —      | —      | 78.85% |
| es_ancora      | 95.74% | 95.74% | 93.48% | 93.48% | —      | —      | —      | —      | —      | 93.48% |
| grc_proiel     | 89.85% | 87.90% | 89.85% | 87.90% | —      | —      | —      | —      | —      | 87.90% |
| hu_korkor      | 85.44% | 79.61% | 83.50% | 77.67% | 85.44% | —      | —      | —      | —      | 77.67% |
| hu_szegedkoref | 92.48% | 89.86% | 91.82% | 89.20% | —      | —      | —      | —      | —      | 89.20% |
| pl_pcc         | 90.99% | 90.88% | 90.88% | 90.76% | —      | —      | —      | —      | —      | 90.65% |
| tr_itcc        | 84.82% | 84.82% | 84.72% | 84.72% | 82.26% | —      | —      | —      | —      | 82.17% |
