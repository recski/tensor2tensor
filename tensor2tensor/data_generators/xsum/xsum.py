# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Summarization based on extractive output"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re
import string

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics
from tensor2tensor.utils import registry
import tensorflow as tf


def _normalize_text(text):
    text = text.lower()
    # Space around punctuation
    text = re.sub("[%s]" % re.escape(string.punctuation), r" \g<0> ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


@registry.register_problem
class XSum(problem.Problem):
    """Base class for Wikisum problems."""

    def example_reading_spec(self):
        data_fields = {
                "inputs": tf.VarLenFeature(tf.int64),
                "targets": tf.VarLenFeature(tf.int64),
                "section_boundaries": tf.VarLenFeature(tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    @property
    def target_vocab_size(self):
        return 2**15

    @property
    def vocab_filename(self):
        return "vocab"

    def feature_encoders(self, data_dir):
        vocab_filename = os.path.join(data_dir, self.vocab_filename)
        encoder = text_encoder.SubwordTextEncoder(vocab_filename)
        # Shared encoder for inputs and targets
        return {"inputs": encoder, "targets": encoder}

    def hparams(self, defaults, unused_model_hparams):
        p = defaults
        p.stop_at_eos = True

        p.vocab_size = {
                "inputs": self._encoders["inputs"].vocab_size,
                "targets": self._encoders["targets"].vocab_size,
        }
        p.modality = {
                "inputs": modalities.ModalityType.SYMBOL,
                "targets": modalities.ModalityType.SYMBOL,
        }

    def eval_metrics(self):
        return super(XSum, self).eval_metrics() + [
                metrics.Metrics.ROUGE_2_F, metrics.Metrics.ROUGE_L_F
        ]

    def generate_lines_for_vocab(self, input_file):
        total_chars = 0
        with open(input_file) as f:
            for line in f:
                data = json.loads(line)
                for sen in data['sentences'] + data['summary']:
                    total_chars += len(sen)
                    yield _normalize_text(sen)

        tf.logging.info("Built vocabulary using %d chars", total_chars)

    def generate_vocab(self, input_file, output_dir):
        # Produce a SubwordTextEncoder from a subset of the data
        return generator_utils.get_or_generate_vocab_inner(
                output_dir, 'vocab', self.target_vocab_size,
                self.generate_lines_for_vocab(input_file))

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        tf.logging.warn(
            "See xsum/README.md for instructions to generate data.")

    def out_filepaths(self, data_dir):
        train_shards = 800
        dev_shards = 100
        test_shards = 100
        train_filepaths = self.training_filepaths(
                data_dir, train_shards, shuffled=True)
        dev_filepaths = self.dev_filepaths(data_dir, dev_shards, shuffled=True)
        test_filepaths = self.test_filepaths(
            data_dir, test_shards, shuffled=True)
        out_filepaths = train_filepaths + dev_filepaths + test_filepaths
        out_filepaths.sort()
        return out_filepaths


def produce_examples(input_file, data_dir, vocab_path, out_filepaths):

    vocab = text_encoder.SubwordTextEncoder(vocab_path)

    def example_generator():
        with open(input_file) as f:
            for line in f:
                data = json.loads(line)
                inputs = []
                for sen in data['textrank']:
                    if len(inputs) >= 1e6:
                        break
                    inputs.extend(vocab.encode(_normalize_text(sen) + " "))

                targets = []
                for sen in data['summary']:
                    targets.extend(vocab.encode(_normalize_text(sen) + " "))

                inputs.append(text_encoder.EOS_ID)
                targets.append(text_encoder.EOS_ID)

                yield {"inputs": inputs, "targets": targets}

    generator_utils.generate_files(example_generator(), out_filepaths)
