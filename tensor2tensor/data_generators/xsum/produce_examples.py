# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
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

"""Produce examples given a vocab, wikis, references, and dataset URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators.wikisum import utils
from tensor2tensor.data_generators.xsum import xsum

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Directory to write to.")
flags.DEFINE_string("input_file", None, "summarization data")


def main(_):
    output_dir = FLAGS.output_dir
    problem = xsum.XSum()
    vocab_path = os.path.join(output_dir, problem.vocab_filename)
    out_filepaths = problem.out_filepaths(output_dir)

    with utils.timing("produce_examples"):
        xsum.produce_examples(
            FLAGS.input_file, output_dir, vocab_path, out_filepaths)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
