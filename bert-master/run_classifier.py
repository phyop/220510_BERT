# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

# module 'tensorflow' has no attribute 'gfile'
tf.gfile = tf.io.gfile

# 原本是flags = tf.flags
flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

# 原本是tf.flags.DEFINE_string(
tf.compat.v1.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

# 數據處理基礎部分的類，也就是這邊要作爲父類使用
# 所以下面有一些子類，會繼承自這個數據處理，去更加完善
class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    # with tf.gfile.Open(input_file, "r") as f:
    with tf.gfile.GFile(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

# 繼承自數據處理基礎的父類DataProcessor
# 換句話說，如果要應用到自己的數據集，就像這裏這樣，仿造着去寫一個子類，繼承自DataProcessor
class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    # 可以修改自己需要幾個分類，比如可能想要3個等級，那就["0", "1", "2"]
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      # 比如mrpc一筆data有2句text，但我們的只需要1句，就可以以這邊的code爲基礎去修改
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    # 設置斷點，然後步進，就可以去布局設置中的變量，查找到label_map={'0': 0, '1': 1}
    # 所以知道就只是設置個標籤而已，沒什麼特別
    label_map[label] = i

  # 跳進tokenize()查看源碼
  tokens_a = tokenizer.tokenize(example.text_a) # 對一筆數據中的第1句話做分詞
  tokens_b = None # 默認是沒有第二句話
  if example.text_b: # 這邊會做判斷，看數據是不是不止有1幾句話組成
    tokens_b = tokenizer.tokenize(example.text_b) # 對一筆數據中的第2句話做分詞

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    # 一筆數據2句話：[CLS], 序列窗口1, [SEP], 序列窗口2, [SEP]
    # 有b的時候，保留3個特殊字符
    # 一筆資料中的第一句話，開頭會有[CLS]特殊字符，是待分類特徵，看是0還是1
    # 第一句話的結束、第2句話的結束，各會插入一個特殊字符[SEP]

    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3) # 如果這兩個太長就截斷
  else:
    # Account for [CLS] and [SEP] with "- 2"
    # 一筆數據1句話：[CLS], 序列窗口1, [SEP]
    # 沒有b的時候，特殊字符保留2個就夠了
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  # 一筆數據2句話：[CLS], 序列窗口1, [SEP], 序列窗口2, [SEP]
  # 第一句話：[CLS] is this jack ##son ##ville ? [SEP]
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  # type_ids只有0和1兩種值，一堆0的代表描述前面那句話，一堆1的代表描述代表後面那句
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = [] # 詞
  segment_ids = [] # 編碼這個詞是來自於第幾句
  tokens.append("[CLS]") # 一筆數據不管是1句話還是2句話，都是[CLS]開始, [SEP]結尾
  segment_ids.append(0) # [CLS]的編碼就是0
  for token in tokens_a: # 遍歷a，也就是第一句話，當中的每個詞
    tokens.append(token) # 把第一句話的每個詞加進來list
    segment_ids.append(0) # 因爲是第一句話，所以ids都是0；如果是第2句話的話，那ids就都是1了
  tokens.append("[SEP]")
  segment_ids.append(0) # 第一句話的結束符，當然也是第一句話的一部分，所以ids也是0

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  # BERT开源项目及数据/GLUE/BERT_BASE_DIR/uncased_L-12_H-768_A-12/vocab.txt
  # 語料表中的第幾行，就代表詞索引ID，所以語料表也就是詞ID表
  # convert_tokens_to_ids()就是把詞轉成ID的映射函數
  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  # 時序窗格大小必須固定，就如同conv的大小是固定的，因爲這樣特徵的shape才會固定，才不會因爲不匹配而無法運算
  # 也就是說，如果2個句子加起來還不到max_seq_length的長度，那就要做pad補齊
  # 但那就有一個問題啦，後面補齊的那些，根本不需要attention
  # 所以才做一個mask來，把所有實際上有的詞（input_ids），全部都用元素1來做標記
  # 後續執行到self-attention的時候，只去對mask值爲1的地方做計算
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length: # 如果2個句子加起來還不到max_seq_length的長度，那就要做pad補齊
    # input_ids -> {list:128}[101, 2572, ...,1012, 102, 0, 0,..., 0, 0]
    input_ids.append(0) # pad的詞id就是0
    input_mask.append(0) # pad不需要被計算self-attention，所以mask值爲0
    # pad被算在第1或2句話都不對，所以就乾脆也給0的值就好
    # 此外，在第2句話結束，要轉成pad的時候，有一個從一堆1轉成一堆0的轉折，print查錯也比較容易看出來
    # segment_ids -> {list:128}[0, 0,...0, 0, 1, 1, ...,1, 1, 0, 0, ..., 0, 0]
    segment_ids.append(0)


  assert len(input_ids) == max_seq_length # 用assert來寫一些簡單判斷，比如等於時序窗格長度，確定程式執行結果沒有基本上的出錯
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.compat.v1.logging.info("*** Example ***")
    tf.compat.v1.logging.info("guid: %s" % (example.guid))
    # I0515 00:08:47.119734 140607478482752 run_classifier.py:500] tokens: [CLS] am ##ro ##zi accused his brother , whom he called " the witness " , of deliberately di ##stor ##ting his evidence . [SEP] referring to him as only " the witness " , am ##ro ##zi accused his brother of deliberately di ##stor ##ting his evidence . [SEP]
    # 5/15 凌晨0點8分47秒119734微秒 時間戳 檔名.py:第500行程式碼] tokens: [CLS] 第1句話的分詞 [SEP] 第2句話的分詞 [SEP]
    tf.compat.v1.logging.info("tokens: %s" % " ".join( # 分詞
        [tokenization.printable_text(x) for x in tokens]))
    # 時序窗格包含了特殊符號以及2句話，特殊符號也是有自己的詞id，而input_ids是把時序窗格轉成詞id的結果
    # input_ids: 101 2572 ... 1012 102 0 ... 0
    tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids])) # 詞id
    tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask])) # pad是0,非pad是1
    tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids])) # 第1句是0,第2句是1
    tf.compat.v1.logging.info("label: %s (id = %d)" % (example.label, label_id)) # 就是[cls]，不是0就是1

  # 進去InputFeatures()源碼看看是什麼作用
  # 反正就是把一堆上面會用到的值，建立成屬性：self.input_ids、self.input_mask、self.segment_ids、self.label_id、self.is_real_example
  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature



def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # writer = tf.python_io.TFRecordWriter(output_file)
  # 這邊也打下斷點，看看人家io是怎麼做的
  # 對數據處理好特徵之後，轉換成TFRecord的格式，讀數據快，處理也方便
  # 等下會遍歷每一個樣本，用TFRecordWriter去寫入到TFRecord當中
  # 接下來就不用到原始數據中去讀了，直接去TFRecord讀取處理好的特徵就行了
  writer = tf.compat.v1.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples): # 遍歷每一個樣本，而mrpc有3700個訓練樣本
    # 每隔1萬次，打印一次結果。因爲mrpc只有3700筆，所以這邊是打印不出來了
    if ex_index % 10000 == 0:
      # tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    # 一樣，convert_single_example看不懂那就跳進去源碼看看
    # 傳入for中的3700個樣本們examples，得到其中的一個樣本example的feature
    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    # 雖然GPU訓練快，但是讀取數據這一步慢了，为了可以高效地读取数据，我們先將data整理成可訓練的TFRecord序列狀態
    # 首先將各項特征feature，都包進去tf.Example，数据會被表示为{"string": value}形式，再去使用tf.string做成序列化來儲存
    # 《TFRecord步驟》 tf.train.Feature() -> tf.train.Example() -> tf.string序列化
    # 《Feature》先爲自己需要的資料，創建對應的feature型別，指派給一個變數名稱
    # 例如int_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[1, 2, 3, 4]))
    # 變數名稱就是指 "tf.train.Example(features=tf.train.Features(feature={}))" 中的value
    # 創建一個存儲訓練數據的標準原型 tf.train.Example()
    # Example()裏面放的是tf.train.Features()，用features當參數給值（有s）
    # Features()裏面放的是{feature={'string':value, ...}，用feature當參數給值（沒s）
    # 例如：
    # example = tf.train.Example(
    #     features=tf.train.Features(feature={
    #         'my_ints': int_feature,
    #         'my_floats': float_feature,
    #         'my_bytes': bytes_feature,
    #     }))
    # Example.features儲存鍵值對，其中每個鍵（字符串）映射到tf.train.Feature，包含固定的列表訊息
    # 這種靈活緊湊的格式，允許存儲大量的類型化數據（詳見 tf.io.parse_example）
    # https://www.tensorflow.org/tutorials/load_data/tfrecord
    # _int64_feature() 之類的方法，是爲了給內部使用的，比如把資料轉型成 tf.train.Example 需要的格式
    # 反正就是TF定義的標準API，幫忙user構建數據集，不是給外部呼叫，所以用底線開頭的私有方法
    # 所以user有數據，往裏面套用就對了，這種底線開頭的，不用研究的太過深入。比如：
    # 呼叫函數_int64_feature，會返回一個類似字典的結構
    # >>> print(_int64_feature(1) # 輸出 int64_list {value: 1}
    # 告訴我們裏面裝的類型是int64_list，值是1
    # 特徵就不會至於單一種類型而已，轉什麼類型還是要自己決定。例如
    # feature = {
    #         'feature0': _int64_feature(feature0), # 把feature0轉成int64的格式
    #         'feature2': _bytes_feature(feature1), # 把feature1轉成bytes的格式
    #         'feature3': _float_feature(feature2), # 把feature2轉成float的格式
    #       }
    # 第1個feature是Features的參數，第2個feature是引數（也就是轉換完的結果）
    # example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    # 上面code代表，有了各項特征之後，就可以包在一起，存成Example
    # Example -> features{feature字典1, ..., feature字典n}
    # example_proto.SerializeToString() # 將Example转换为二进制字符串

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    # 建立一個OrderedDict()物件，來把各種feature裝進去
    # 因爲OrderedDict()是一個可以裝東西進去的物件，所以被放在collections類裏面
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    # 最後面()的第1個feature是Features的參數，第2個feature是引數（也就是轉換完的結果）
    # 把各種feature用OrderedDict()打包起來，然後傳給Example()，再去使用tf.string做成序列化來儲存
    # Feature()就是一個單純的dict()
    # Example()是多了一層dict()的複合型字典
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    # example = tf.parse_single_example(record, name_to_features)
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        # t = tf.to_int32(t)
        t = tf.compat.v1.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        # tf.compat.v1.estimator.data.map_and_batch(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  # hidden_size = output_layer.shape[-1].value
  hidden_size = output_layer.shape[-1]

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.compat.v1.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      # output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
      output_layer = tf.compat.v1.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint (init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        # tf.compat.v1.train.init_from_checkpoint (init_checkpoint, assignment_map)
        tf.compat.v1.train.init_from_checkpoint (init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
      # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  # 設置展示性的logging而已
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  # 一開始會設置一些if條件式，爲的是去檢查，輸入的參數寫得有沒有問題
  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  # tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.makedirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  # 設置if條件，看有沒有TPU或是GPU之類的，有的話才會起作用的code
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.compat.v1.estimator.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  # 要開始train，那就是要開始讀取數據，也就是要開始做數據的處理了
  if FLAGS.do_train:
    # 利用get_train_examples，把數據一行行的讀進來
    # 函數快速定位
    # 點擊get_train_examples，然後壓着ctrl+shift+滑鼠移到get_train_examples字串上
    # 會出現好幾個選項可以跳轉，包括DataProcessor中的def、還有其子類中也有各自的def，
    # 選在class MrpcProcessor(DataProcessor)下的def get_train_example去看。
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    # 3700筆數據,假設batch_size=100,那就是370次iteration,才能完成一個epoch
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    # 比如總共迭代1375次,那一開始就用比較小的學習率來warmup，直到過了1375*0.1次迭代，才恢復到正常的學習率。
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  # 先把一些需要用到的物件實例化出來，實際要怎麼用、引數的值爲什麼是這樣，要往後面的code繼續看
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # 在這邊打下斷點，也就是「掛起」
    # 然後用「debug調試」模式，如果前面也有別的斷點，那就點擊「步進/恢復程序」直到code執行到這裏
    # 在「布局設置」中，把「變量」給打勾顯示出來，就可以查找到Name-Value，比如「num_train_steps={int}1375」
    # ctrl+shift跳進去file_based_convert_examples_to_features()查看源碼
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.compat.v1.logging.info("***** Running prediction*****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  # 原本是 tf.app.run()
  tf.compat.v1.app.run()