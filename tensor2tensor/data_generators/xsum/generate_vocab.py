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

"""Generate vocab from references and wikis."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.xsum import xsum

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None, "Directory to write to.")
flags.DEFINE_string("input_file", None, "summarization data")


def main(_):
    problem = xsum.XSum()
    problem.generate_vocab(FLAGS.input_file, FLAGS.output_dir)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
