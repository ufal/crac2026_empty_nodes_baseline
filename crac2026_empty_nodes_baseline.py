#!/usr/bin/env python3

# This file is part of CRAC26 Zero Nodes Baseline
# <https://github.com/ufal/crac2026_empty_nodes_baseline>.
#
# Copyright 2026 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import argparse
import collections
import dataclasses
import datetime
import io
import json
import os
import re

import huggingface_hub
import minnt
import numpy as np
import torch
import transformers

import crac2026_empty_nodes_eval

minnt.require_version("1.0")

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--dev", default=[], nargs="+", type=str, help="Dev CoNLL-U files.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout")
parser.add_argument("--enodes_per_head", default=2, type=int, help="Max empty nodes per head.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--epochs_frozen", default=2, type=int, help="Number of epochs with frozen transformer.")
parser.add_argument("--exp", default=None, type=str, help="Experiment name.")
parser.add_argument("--label_smoothing", default=0.0, type=float, help="Label smoothing.")
parser.add_argument("--lazy_adam", default=False, action="store_true", help="Use LazyAdam optimizer.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default="cos", choices=["cos", "none"], type=str, help="Learning rate decay.")
parser.add_argument("--learning_rate_warmup", default=5_000, type=int, help="Number of warmup steps.")
parser.add_argument("--load", default=None, type=str, help="Path to load the model from.")
parser.add_argument("--max_train_sentence_len", default=512, type=int, help="Max sentence subwords in training.")
parser.add_argument("--prediction_threshold", default=0.5, type=float, help="Prediction threshold.")
parser.add_argument("--save_model", default=False, action="store_true", help="Save the model.")
parser.add_argument("--seed", default=42, type=int, help="Initial random seed.")
parser.add_argument("--steps_per_epoch", default=5_000, type=int, help="Steps per epoch.")
parser.add_argument("--tags_weight", default=0.3, type=float, help="Weight of the tags loss.")
parser.add_argument("--tags_min_occurrences", default=2, type=int, help="Minimum tag occurrences to include in the vocab.")
parser.add_argument("--task_dim", default=512, type=int, help="Task dimension size.")
parser.add_argument("--task_hidden_layer", default=2_048, type=int, help="Task hidden layer size.")
parser.add_argument("--test", default=[], nargs="+", type=str, help="Test CoNLL-U files.")
parser.add_argument("--train", default=[], nargs="+", type=str, help="Train CoNLL-U files.")
parser.add_argument("--train_sampling_exponent", default=0.5, type=float, help="Train sampling exponent.")
parser.add_argument("--transformer", default="xlm-roberta-large", type=str, help="XLM-RoBERTA model to use.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")


class CorefDataset:
    TAG_NAMES = ["DEPREL", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS"]  # What tags are we predicting

    @dataclasses.dataclass
    class EmptyNode:
        word_order: int
        head: int
        tags: list[int]  # Remapped to numerical ids already

    @dataclasses.dataclass
    class Sentence:
        conllu_lines: list[str]
        forms: list[str]
        empty_nodes: list["CorefDataset.EmptyNode"]
        new_document: bool

    def __init__(self, path: str, args: argparse.Namespace, train_dataset: "CorefDataset" = None):
        self.path = path

        if train_dataset is None:
            self.tag_mappings = [minnt.Vocabulary([], add_unk=True) for _ in self.TAG_NAMES]
        else:
            self.tag_mappings = train_dataset.tag_mappings
        self.sentences = []
        self.conllu_for_eval = None if train_dataset is None else []

        # Load the CoNLL-U file
        with open(path, "r", encoding="utf-8") as file:
            in_sentence = False
            for line in file:
                if self.conllu_for_eval is not None:
                    self.conllu_for_eval.append(line)

                line = line.rstrip("\r\n")
                if not line:
                    in_sentence = False
                else:
                    if not in_sentence:
                        self.sentences.append(self.Sentence([], [], [], False))
                        in_sentence = True

                    if not re.match(r"^[0-9]*[.]", line):
                        self.sentences[-1].conllu_lines.append(line)
                        if match := re.match(r"^([0-9]+)\t([^\t]*)\t", line):
                            word_id, form = int(match.group(1)), match.group(2)
                            assert len(self.sentences[-1].forms) == word_id - 1, "Bad IDs in the CoNLL-U file"
                            self.sentences[-1].forms.append(form)
                        continue

                    columns = line.split("\t")
                    word_order = columns[0].split(".", maxsplit=1)[0]
                    head, deprel = columns[8].split("|", maxsplit=1)[0].split(":", maxsplit=1)
                    tags = [deprel] + columns[1:6]
                    for i in range(len(tags)):
                        tags[i] = self.tag_mappings[i].index(tags[i], add_missing=train_dataset is None)

                    self.sentences[-1].empty_nodes.append(self.EmptyNode(int(word_order), int(head), tags))

        if self.conllu_for_eval is not None:
            self.conllu_for_eval = "".join(self.conllu_for_eval)

        # Fill new_document
        for i, sentence in enumerate(self.sentences):
            sentence.new_document = i == 0 or any(re.match(r"^\s*#\s*newdoc", line) for line in sentence.conllu_lines)

        # The dataset consists of a single treebank
        self.treebank_ranges = [(0, len(self.sentences))]

    def save_mappings(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as mappings_file:
            json.dump([list(tag_mapping) for tag_mapping in self.tag_mappings], mappings_file, ensure_ascii=False, indent=2)

    @staticmethod
    def from_mappings(path: str) -> "CorefDataset":
        with open(path, "r", encoding="utf-8") as mappings_file:
            data = json.load(mappings_file)
        mappings = CorefDataset.__new__(CorefDataset)
        mappings.tag_mappings = [minnt.Vocabulary(tags) for tags in data]
        return mappings

    def write_sentence(self, output: io.TextIOBase, index: int, empty_nodes: list[EmptyNode]) -> None:
        assert index < len(self.sentences), f"Sentence index {index} out of range"

        empty_nodes_lines = {}
        for empty_node in empty_nodes:
            wo_enodes = empty_nodes_lines.setdefault(empty_node.word_order, [])
            wo_enodes.append("{}.{}\t{}\t{}\t{}\t{}\t{}\t_\t_\t{}:{}\t_".format(
                empty_node.word_order, len(wo_enodes) + 1,
                *[self.tag_mappings[i].string(empty_node.tags[i]) for i in range(1, len(empty_node.tags))],
                empty_node.head, self.tag_mappings[0].string(empty_node.tags[0])))

        in_initial_comments = True
        for line in self.sentences[index].conllu_lines:
            if not line.startswith("#") and in_initial_comments:
                for empty_node in empty_nodes_lines.pop(0, []):
                    print(empty_node, file=output)
                in_initial_comments = False
            print(line, file=output)
            if match := re.match(r"^([0-9]+)\t", line):
                for empty_node in empty_nodes_lines.pop(int(match.group(1)), []):
                    print(empty_node, file=output)
        print(file=output)
        assert not empty_nodes_lines, "Got empty nodes with incorrect word orders"


class CorefDatasetMerged(CorefDataset):
    def __init__(self, datasets: list[CorefDataset], args: argparse.Namespace):
        self.path = "merged"

        # Create mappings
        tag_mappings = [collections.Counter() for _ in self.TAG_NAMES]

        for dataset in datasets:
            for sentence in dataset.sentences:
                for empty_node in sentence.empty_nodes:
                    for i in range(len(empty_node.tags)):
                        tag = dataset.tag_mappings[i].string(empty_node.tags[i])
                        assert tag != minnt.Vocabulary.UNK_TOKEN
                        tag_mappings[i][tag] += 1

        self.tag_mappings = []
        for i in range(len(self.TAG_NAMES)):
            self.tag_mappings.append(minnt.Vocabulary(
                sorted(tag for tag, count in tag_mappings[i].items() if count >= args.tags_min_occurrences),
                add_unk=True,
            ))
        print("Number of predicted tags:", *[f"{name}={len(tag_mapping)}" for name, tag_mapping in zip(self.TAG_NAMES, self.tag_mappings)])

        # Merge sentences
        self.sentences = []
        self.treebank_ranges = []
        for dataset in datasets:
            assert len(dataset.treebank_ranges) == 1
            self.treebank_ranges.append((len(self.sentences), len(self.sentences) + len(dataset.sentences)))
            for sentence in dataset.sentences:
                empty_nodes = []
                for empty_node in sentence.empty_nodes:
                    tags = [
                        self.tag_mappings[i].index(dataset.tag_mappings[i].string(empty_node.tags[i])) or self.tag_mappings[i].index("_")
                        for i in range(len(empty_node.tags))]
                    empty_nodes.append(self.EmptyNode(empty_node.word_order, empty_node.head, tags))
                self.sentences.append(self.Sentence(sentence.conllu_lines, sentence.forms, empty_nodes, sentence.new_document))


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, coref: CorefDataset, tokenizer: transformers.PreTrainedTokenizer, args: argparse.Namespace, training: bool):
        self.coref = coref
        self.training = training

        # Tokenize the sentences
        tokenized = tokenizer([sentence.forms for sentence in coref.sentences], add_special_tokens=False, is_split_into_words=True)

        tokens, word_indices = [], []
        for i, sentence in enumerate(tokenized.input_ids):
            tokens.append(sentence)
            word_indices.append([-1])  # The future SEP token in front of the sentence
            for j in range(len(coref.sentences[i].forms)):
                span = tokenized.word_to_tokens(i, j)
                word_indices[-1].append(span.start)
            word_indices[-1] = np.array(word_indices[-1], dtype=np.int32)

        # Generate sentences and gold data
        trimmed_sentences = 0
        self._inputs = []
        self._outputs = []
        for i in range(len(tokens)):
            sentence = tokens[i]
            indices = word_indices[i]

            # Trim if needed
            if training and len(sentence) > args.max_train_sentence_len - 2:
                trimmed_sentences += 1
                sentence = sentence[:args.max_train_sentence_len - 2]
                while indices[-1] >= len(sentence):
                    indices = indices[:-1]

            self._inputs.append([
                np.array([tokenizer.cls_token_id] + sentence + [tokenizer.sep_token_id], dtype=np.int32),
                indices + 1,
            ])

            # Generate outputs in the correct format, trimming if necessary
            empty_nodes = [[] for _ in range(len(indices))]
            for empty_node in coref.sentences[i].empty_nodes:
                if max(empty_node.word_order, empty_node.head) < len(indices):
                    empty_nodes[empty_node.head].append([1, empty_node.word_order, *empty_node.tags])
            for i in range(len(empty_nodes)):
                empty_nodes[i].append([0, -1] + [-1] * len(coref.TAG_NAMES))
                while len(empty_nodes[i]) < args.enodes_per_head:
                    empty_nodes[i].append([-1, -1] + [-1] * len(coref.TAG_NAMES))
                empty_nodes[i] = empty_nodes[i][:args.enodes_per_head]
            self._outputs.append(np.array(empty_nodes, dtype=np.int32))

        if trimmed_sentences:
            print("Trimmed {} out of {} sentences from {}".format(trimmed_sentences, len(coref.sentences), os.path.basename(coref.path)))

    def __len__(self):
        return len(self._inputs)

    def __getitem__(self, index: int):
        inputs = [torch.from_numpy(input_) for input_ in self._inputs[index]]
        outputs = torch.from_numpy(self._outputs[index])
        return inputs, outputs


class TorchDataLoader(torch.utils.data.DataLoader):
    class MergedDatasetSampler(torch.utils.data.Sampler):
        def __init__(self, coref: CorefDataset, args: argparse.Namespace):
            self._treebank_ranges = coref.treebank_ranges
            self._sentences_per_epoch = args.steps_per_epoch * args.batch_size
            self._generator = torch.Generator().manual_seed(args.seed)

            treebank_weights = np.array([r[1] - r[0] for r in self._treebank_ranges], np.float32)
            treebank_weights = treebank_weights ** args.train_sampling_exponent
            treebank_weights /= np.sum(treebank_weights)
            self._treebank_sizes = np.array(treebank_weights * self._sentences_per_epoch, np.int32)
            self._treebank_sizes[:self._sentences_per_epoch - np.sum(self._treebank_sizes)] += 1
            self._treebank_indices = [[] for _ in self._treebank_ranges]

        def __len__(self):
            return self._sentences_per_epoch

        def __iter__(self):
            indices = []
            for i in range(len(self._treebank_ranges)):
                required = self._treebank_sizes[i]
                while required:
                    if not len(self._treebank_indices[i]):
                        self._treebank_indices[i] = self._treebank_ranges[i][0] + torch.randperm(
                            self._treebank_ranges[i][1] - self._treebank_ranges[i][0], generator=self._generator)
                    indices.append(self._treebank_indices[i][:required])
                    required -= min(len(self._treebank_indices[i]), required)
            indices = torch.concatenate(indices, axis=0)
            return iter(indices[torch.randperm(len(indices), generator=self._generator)])

    def _collate_fn(self, batch):
        inputs, output = zip(*batch)
        batch_inputs = [
            torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=-1)
            for sequences in zip(*inputs)
        ]
        batch_output = torch.nn.utils.rnn.pad_sequence(output, batch_first=True, padding_value=-1)
        return tuple(batch_inputs), batch_output

    def __init__(self, dataset: TorchDataset, args: argparse.Namespace, **kwargs):
        sampler = None
        if dataset.training:
            if len(dataset.coref.treebank_ranges) == 1 and not args.steps_per_epoch:
                sampler = torch.utils.data.RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed))
            else:
                sampler = self.MergedDatasetSampler(dataset.coref, args)
        super().__init__(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=self._collate_fn, **kwargs)


class Model(minnt.TrainableModule):
    def __init__(self, coref: CorefDataset, args: argparse.Namespace):
        super().__init__()

        # Create the transformer layer
        if not args.load:
            self._transformer = transformers.AutoModel.from_pretrained(args.transformer)
        else:
            config = transformers.AutoConfig.from_pretrained(args.transformer)
            self._transformer = transformers.AutoModel.from_config(config)

        # Create the model layers
        self._relu = torch.nn.ReLU()
        self._dropout = torch.nn.Dropout(args.dropout)

        def ffn_module(output_dim: int) -> torch.nn.Module:
            return torch.nn.Sequential(
                torch.nn.LazyLinear(args.task_hidden_layer),
                self._relu,
                self._dropout,
                torch.nn.LazyLinear(output_dim),
            )

        self._candidate_layers = torch.nn.ModuleList(ffn_module(args.task_dim) for _ in range(args.enodes_per_head))
        self._classification_layers = ffn_module(2)
        self._query_layers = ffn_module(args.task_dim)
        self._key_layers = ffn_module(args.task_dim)
        self._tag_layers = torch.nn.ModuleList(ffn_module(len(tag_mapping)) for tag_mapping in coref.tag_mappings)

        self._args = args

        if args.load:
            self.load_weights(os.path.join(args.load, "model.weights.pt"))

    def forward(self, tokens, indices):
        # Get word embeddings from the backbone.
        embeddings = self._transformer(torch.relu(tokens), attention_mask=tokens >= 0).last_hidden_state
        words = torch.gather(embeddings, 1, torch.relu(indices).unsqueeze(-1).expand(-1, -1, embeddings.shape[-1]))
        words = self._dropout(words)

        # Generate args.enodes_per_head hidden states
        candidates = []
        for candidate_layers in self._candidate_layers:
            candidates.append(candidate_layers(words if not candidates else torch.cat([words] + candidates, dim=-1)))
        candidates = torch.stack(candidates, dim=2)

        # Run the classification head
        empty_nodes = self._classification_layers(candidates)

        # Run the word_order selection head
        queries = self._query_layers(candidates).permute(0, 2, 1, 3)
        keys = self._key_layers(words).unsqueeze(2).permute(0, 2, 3, 1)
        arc_scores = torch.matmul(queries, keys) / (queries.shape[-1] ** 0.5)
        mask = (indices[:, torch.newaxis, torch.newaxis, :] >= 0).type_as(arc_scores)
        arc_scores = arc_scores * mask - 1e9 * (1 - mask)
        arc_scores = arc_scores.permute(0, 2, 1, 3)

        # Run the tag prediction heads
        tags = [tag_layers(candidates) for tag_layers in self._tag_layers]

        return empty_nodes, arc_scores, *tags

    def compute_loss(self, outputs, targets, tokens, indices):
        return {
            "candidate_loss": self.loss(outputs[0], targets[..., 0]),
            "word_order_loss": self.loss_no_ls(outputs[1], targets[..., 1]),
            "deprel_loss": self.loss(outputs[2], targets[..., 2]),
            "tags_loss": self._args.tags_weight * sum(self.loss(outputs[i], targets[..., i]) for i in range(3, len(outputs))),
        }

    def configure(self, epoch_batches: int, frozen: bool):
        args = self._args

        self._transformer.requires_grad_(not frozen)

        lr = 1e-3 if frozen else args.learning_rate
        optimizer = minnt.optimizers.LazyAdam(self, lr) if args.lazy_adam else torch.optim.Adam(self.parameters(), lr)

        if not frozen and args.learning_rate_decay == "cos":
            scheduler = minnt.schedulers.CosineDecay(optimizer, args.epochs * epoch_batches, warmup=args.learning_rate_warmup)
        else:
            scheduler = None

        super().configure(
            optimizer=optimizer,
            scheduler=scheduler,
            logdir=self._args.logdir,
        )
        self.loss = minnt.losses.CategoricalCrossEntropy(dim=-1, ignore_index=-1, label_smoothing=args.label_smoothing)
        self.loss_no_ls = minnt.losses.CategoricalCrossEntropy(dim=-1, ignore_index=-1)

    def predict(self, dataloader: TorchDataLoader, save_as: str | None = None) -> str:
        coref = dataloader.dataset.coref
        threshold = np.log(self._args.prediction_threshold / (1 - self._args.prediction_threshold))

        conllu, sentence = io.StringIO(), 0
        for batch in minnt.ProgressLogger(dataloader, f"Predicting {os.path.basename(save_as)}" if save_as else "Predicting"):
            predictions = self.predict_batch(batch[0], as_numpy=True)
            for b in range(len(predictions[0])):
                sentence_len = len(coref.sentences[sentence].forms)
                is_empty_node = predictions[0][b, :sentence_len + 1, :, 1] - predictions[0][b, :sentence_len + 1, :, 0] >= threshold
                word_order = np.argmax(predictions[1][b, :sentence_len + 1, :sentence_len + 1], axis=-1)
                # Avoid predicting an UNK_TOKEN, which is at index 0.
                tags = [1 + np.argmax(predictions[2 + i][b, :sentence_len + 1, :, 1:], axis=-1) for i in range(len(coref.TAG_NAMES))]

                empty_nodes = []
                for i in range(sentence_len + 1):
                    j = 0
                    while j < len(is_empty_node[i]) and is_empty_node[i][j]:
                        empty_nodes.append(coref.EmptyNode(word_order[i][j], i, [tags[k][i, j] for k in range(len(coref.TAG_NAMES))]))
                        j += 1

                coref.write_sentence(conllu, sentence, empty_nodes)
                sentence += 1

        conllu = conllu.getvalue()
        if save_as is not None:
            if os.path.dirname(save_as):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as, "w", encoding="utf-8") as conllu_file:
                conllu_file.write(conllu)
        return conllu

    def evaluate(self, dataloader: TorchDataLoader, save_as: str | None = None) -> tuple[str, dict[str, float]]:
        conllu = self.predict(dataloader, save_as=save_as)
        evaluation = crac2026_empty_nodes_eval.evaluate(conllu, dataloader.dataset.coref.conllu_for_eval)
        if save_as is not None:
            if os.path.dirname(save_as):
                os.makedirs(os.path.dirname(save_as), exist_ok=True)
            with open(save_as + ".eval", "w", encoding="utf-8") as eval_file:
                for metric, score in evaluation.items():
                    print(f"{metric}: f1={100 * score.f1:.2f}%, p={100 * score.p:.2f}%, r={100 * score.r:.2f}%", file=eval_file)
        return conllu, evaluation


def main(params: list[str] | None = None) -> None:
    args = parser.parse_args(params)

    # Set the random seed and the number of threads
    minnt.startup(args.seed, args.threads)
    minnt.global_keras_initializers()

    # If supplied, load configuration from a trained model
    if args.load:
        resolved_load_path = args.load if os.path.exists(args.load) else huggingface_hub.snapshot_download(args.load)
        with open(os.path.join(resolved_load_path, "options.json"), mode="r") as options_file:
            args = argparse.Namespace(**{k: v for k, v in json.load(options_file).items() if k not in [
                "dev", "exp", "load", "seed", "test", "threads"]})
        args = parser.parse_args(params, namespace=args)
        args.load = resolved_load_path
    else:
        assert args.train, "Either --load or --train must be set."

        # Create logdir
        args.logdir = os.path.join("logs", "{}{}-{}".format(
            args.exp + "-" if args.exp else "",
            os.path.splitext(os.path.basename(globals().get("__file__", "notebook")))[0],
            datetime.datetime.now().strftime("%y%m%d_%H%M%S"),
        ))
        os.makedirs(args.logdir, exist_ok=True)
        with open(os.path.join(args.logdir, "options.json"), mode="w") as options_file:
            json.dump(vars(args), options_file, sort_keys=True, ensure_ascii=False, indent=2)

    if args.load:
        train = CorefDataset.from_mappings(os.path.join(args.load, "mappings.json"))
    else:
        train = CorefDatasetMerged([CorefDataset(path, args) for i, path in enumerate(args.train)], args)
        train.save_mappings(os.path.join(args.logdir, "mappings.json"))
    devs = [CorefDataset(path, args, train_dataset=train) for i, path in enumerate(args.dev)]
    tests = [CorefDataset(path, args, train_dataset=train) for i, path in enumerate(args.test)]

    # Create the model
    model = Model(train, args)

    # Create the datasets
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.transformer)
    if not args.load:
        train_dataloader = TorchDataLoader(TorchDataset(train, tokenizer, args, training=True), args)
    dev_dataloaders = [TorchDataLoader(TorchDataset(dataset, tokenizer, args, training=False), args) for dataset in devs]
    test_dataloaders = [TorchDataLoader(TorchDataset(dataset, tokenizer, args, training=False), args) for dataset in tests]

    # Perform prediction if requested
    if args.load:
        for dataloader in dev_dataloaders:
            model.evaluate(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.coref.path)) if args.exp else dataloader.dataset.coref.path
            )[0] + ".predicted.conllu")
        for dataloader in test_dataloaders:
            model.predict(dataloader, save_as=os.path.splitext(
                os.path.join(args.exp, os.path.basename(dataloader.dataset.coref.path)) if args.exp else dataloader.dataset.coref.path
            )[0] + ".predicted.conllu")
        return

    # Train the model
    def evaluation(model, epoch, logs):
        dev_scores = {"DEP": {}, "DEP_WO": {}, "ALL": {}}
        for dataloader in dev_dataloaders:
            _, metrics = model.evaluate(dataloader, save_as=os.path.splitext(
                os.path.join(args.logdir, os.path.basename(dataloader.dataset.coref.path)))[0] + f".{epoch:02d}.conllu")
            for metric, scores in dev_scores.items():
                scores[f"{os.path.splitext(os.path.basename(dataloader.dataset.coref.path))[0]}:{metric}"] = 100 * metrics[metric].f1
        logs |= {metric: score for scores in dev_scores.values() for metric, score in scores.items()}
        logs |= {f"dev:{metric}": np.mean(list(score.values())) for metric, score in dev_scores.items()}
        if epoch == args.epochs + args.epochs_frozen:
            for dataloader in test_dataloaders:
                model.predict(dataloader, save_as=os.path.splitext(
                    os.path.join(args.logdir, os.path.basename(dataloader.dataset.coref.path)))[0] + f".{epoch:02d}.conllu")
        if args.save_model and epoch + 5 > args.epochs + args.epochs_frozen:
            model.save_weights("{logdir}/model.weights.{epoch:02d}.pt")
        if epoch >= args.epochs_frozen + 2 and logs["dev:DEP"] < 10:
            return minnt.STOP_TRAINING

    if args.epochs_frozen:
        model.configure(len(train_dataloader), frozen=True)
        model.fit(train_dataloader, args.epochs_frozen, callbacks=[evaluation])
    if args.epochs:
        model.configure(len(train_dataloader), frozen=False)
        model.fit(train_dataloader, args.epochs, callbacks=[evaluation])
    if args.save_model:
        model.save_weights("{logdir}/model.weights.pt")


if __name__ == "__main__":
    main([] if "__file__" not in globals() else None)
