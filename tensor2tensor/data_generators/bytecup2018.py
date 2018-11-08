# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generator for the 2018 ByteCup challenge"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os

# from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry


def example_generator(sum_token):
    split_token = u" <summary> " if sum_token else " "
    data_dir = os.getenv("BYTECUP_DATA_DIR")
    if data_dir is None:
        raise ValueError("BYTECUP_DATA_DIR not set")
    for fn in os.listdir(data_dir):
        with open(os.path.join(data_dir, fn)) as f:
            for line in f:
                data = json.loads(line)
                yield data['content'] + split_token + data['title']


def _story_summary_split(story):
    split_str = u" <summary> "
    split_str_len = len(split_str)
    split_pos = story.find(split_str)
    return story[:split_pos], story[split_pos + split_str_len:]


@registry.register_problem
class Bytecup2018(text_problems.Text2TextProblem):
    """Summarize CNN and Daily Mail articles to their summary highlights."""

    def generate_text_for_vocab(self, data_dir, tmp_dir):
        return example_generator(sum_token=False)

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{"split": problem.DatasetSplit.TRAIN, "shards": 100}]
        # {"split": problem.DatasetSplit.EVAL,  "shards": 10},
        # {"split": problem.DatasetSplit.TEST,  "shards": 10}]

    def is_generate_per_split(self):
        return False

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        for example in example_generator(sum_token=True):
            story, summary = _story_summary_split(example)
            yield {"inputs": story, "targets": summary}
