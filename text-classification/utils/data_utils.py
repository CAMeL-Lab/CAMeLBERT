# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2021 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Text classification utils
"""

import logging
import os
import re
import camel_tools.utils.normalize as normalize
import camel_tools.utils.dediac as dediac
from transformers.data.processors.utils import (
    DataProcessor,
    InputExample,
    InputFeatures
)

logger = logging.getLogger(__name__)

SENTIMENT_LABELS = ["positive", "negative", "neutral"]

POETRY_LABELS = ["شعر حر", "شعر التفعيلة", "عامي", "موشح", "الرجز",
                 "الرمل", "الهزج", "البسيط", "الخفيف", "السريع",
                 "الطويل", "الكامل", "المجتث", "المديد", "الوافر",
                 "الدوبيت", "السلسلة", "المضارع", "المقتضب", "المنسرح",
                 "المتدارك", "المتقارب", "المواليا"]

MADAR_26_LABELS = ["KHA", "TUN", "MOS", "CAI", "BAG", "ALE", "DOH", "ALX",
                   "SAN", "BAS", "TRI", "ALG", "MSA", "FES", "BEN", "SAL",
                   "JER", "BEI", "SFX", "MUS", "JED", "RIY", "RAB", "DAM",
                   "ASW", "AMM"]

MADAR_6_LABELS = ["TUN", "CAI", "DOH", "MSA", "BEI", "RAB"]

MADAR_TWITTER_LABELS = ["Algeria", "Bahrain", "Djibouti", "Egypt", "Iraq",
                        "Jordan", "Kuwait", "Lebanon", "Libya", "Mauritania",
                        "Morocco", "Oman", "Palestine", "Qatar",
                        "Saudi_Arabia", "Somalia", "Sudan", "Syria",
                        "Tunisia", "United_Arab_Emirates", "Yemen"]

NADI_COUNTRY_LABELS = ["Algeria", "Bahrain", "Djibouti", "Egypt",
                       "Iraq", "Jordan", "Kuwait", "Lebanon",
                       "Libya", "Mauritania", "Morocco", "Oman",
                       "Palestine", "Qatar", "Saudi_Arabia",
                       "Somalia", "Sudan", "Syria", "Tunisia",
                       "United_Arab_Emirates", "Yemen"]


def convert_examples_to_features(examples, tokenizer,
                                max_length=512,
                                task=None,
                                label_list=None,
                                output_mode=None,
                                pad_on_left=False,
                                pad_token=0,
                                pad_token_segment_id=0,
                                mask_padding_with_zero=True):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: Arabic Sentiment Analysis task
        label_list: List of labels.
                    Can be obtained from the processor using the
                    ``processor.get_labels()`` method
        output_mode: String indicating the output mode.
                     Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``,
                     the examples will be padded on the left rather
                     than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token
                              (It is usually 0, but can vary such as for
                               XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``,
                                the attention mask will be filled by ``1``
                                for actual values
                                and by ``0`` for padded values.
                                If set to ``False``, inverts it (``1`` for
                                padded values, ``0`` for actual values)

    Returns:
        list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" %
                        (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" %
                        (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    logger.info('**LABEL MAP**')
    logger.info(label_map)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + attention_mask
            token_type_ids = (([pad_token_segment_id] * padding_length)
                              + token_type_ids)
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero
                                                  else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id]
                                               * padding_length)

        assert len(input_ids) == max_length, (
                "Error with input length {} vs {}".format(len(input_ids),
                                                          max_length))

        assert len(attention_mask) == max_length, (
                "Error with input length {} vs {}".format(len(attention_mask),
                                                          max_length))

        assert len(token_type_ids) == max_length, (
                "Error with input length {} vs {}".format(len(token_type_ids),
                                                          max_length))

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" %
                        " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" %
                        " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    return features


class ArabicSentimentProcessor(DataProcessor):
    """Processor for Arabic Sentiment Analysis"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['tweet'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_a = self.process_tweet(line[1])
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "ArSAS.test.tsv")),
            "test")

    def get_labels(self):
        return SENTIMENT_LABELS

    def process_tweet(self, tweet):
        """
        processes a tweet by normalizing letters, removing urls,
        and removing diacritics

        Args:
            input tweet
        Returns:
            processed tweet
        """

        # URL regex
        URL_REG = re.compile(r'[https|http|@|RT]([^\s]+)')
        # space regex
        SPACE_REG = re.compile(r'[\s]+')

        tweet = normalize.normalize_unicode(tweet)
        tweet = dediac.dediac_ar(tweet)
        tweet = URL_REG.sub(' ', tweet)
        tweet = SPACE_REG.sub(' ', tweet)
        return tweet

class ArabicPoetryProcessor(DataProcessor):
    """Processor for Arabic Poetry Classification"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['verse_1'].numpy().decode('utf-8'),
                            tensor_dict['verse_2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # check if line contains 2 verses or not
            if len(line) == 3:
                text_a = self.process_verse(line[0])
                text_b = self.process_verse(line[1])
                label = line[2]
            elif len(line) == 2:
                text_a = self.process_verse(line[0])
                text_b = None
                label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         text_b=text_b, label=label))

        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")),
            "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
            "test")

    def get_labels(self):
        return POETRY_LABELS

    def process_verse(self, verse):
        """
        processes a verse by removing diacritics

        Args:
            input verse
        Returns:
            processed verse
        """

        verse = dediac.dediac_ar(verse)
        return verse

class ArabicDIDProcessor_MADAR_26(DataProcessor):
    """Processor for Arabic Dialect ID Classification"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['text'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = self.process_text(line[0])
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a,label=label))

        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "MADAR-Corpus-26-train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MADAR-Corpus-26-train.tsv")),
                                        "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "MADAR-Corpus-26-dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MADAR-Corpus-26-dev.tsv")),
                           "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "MADAR-Corpus-26-test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MADAR-Corpus-26-test.tsv")),
                           "test")

    def get_labels(self):
        return MADAR_26_LABELS

    def process_text(self, text):
        """
        processes the input text by removing diacritics

        Args:
            input text
        Returns:
            processed text
        """

        text = dediac.dediac_ar(text)
        return text

class ArabicDIDProcessor_MADAR_6(DataProcessor):
    """Processor for Arabic Dialect ID Classification on
       MADAR Corpus 6"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['text'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = self.process_text(line[0])
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a,label=label))

        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                    "MADAR-Corpus-6-train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MADAR-Corpus-6-train.tsv")),
                                        "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                    "MADAR-Corpus-6-dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "MADAR-Corpus-6-dev.tsv")),
                                        "dev")

    def get_labels(self):
        return MADAR_6_LABELS

    def process_text(self, text):
        """
        processes the input text by removing diacritics

        Args:
            input text
        Returns:
            processed text
        """

        text = dediac.dediac_ar(text)
        return text

class ArabicDIDProcessor_MADAR_Twitter(DataProcessor):
    """Processor for Arabic Dialect ID Classification on
       NADI Shared Task 1"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['user_id'].numpy().decode('utf-8'),
                            tensor_dict['tweet'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            guid = "%s-%s" % (set_type, i)
            user_id = line[0]
            text_a = self.process_tweet(line[1])
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "grouped_tweets.train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "grouped_tweets.train.tsv")),
                                        "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "grouped_tweets.dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "grouped_tweets.dev.tsv")),
                                        "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "grouped_tweets.test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "grouped_tweets.test.tsv")),
                                        "test")

    def get_labels(self):
        return MADAR_TWITTER_LABELS

    def process_tweet(self, tweet):
        """
        processes a tweet by normalizing letters, removing urls,
        and removing diacritics

        Args:
            input tweet
        Returns:
            processed tweet
        """

        # URL regex
        URL_REG = re.compile(r'[https|http|@|RT]([^\s]+)')
        # space regex
        SPACE_REG = re.compile(r'[\s]+')

        tweet = normalize.normalize_unicode(tweet)
        tweet = dediac.dediac_ar(tweet)
        tweet = URL_REG.sub(' ', tweet)
        tweet = SPACE_REG.sub(' ', tweet)
        return tweet.strip()

class ArabicDIDProcessor_NADI_COUNTRY(DataProcessor):
    """Processor for Arabic Dialect ID Classification on
       NADI Shared Task 1"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['tweet'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines[1:]):
            guid = "%s-%s" % (set_type, i)
            text_a = self.process_tweet(line[1])
            label = line[2]
            examples.append(InputExample(guid=guid, text_a=text_a,
                                         label=label))
        return examples

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "train_labeled.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_labeled.tsv")),
                                        "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "dev_labeled.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_labeled.tsv")),
                                        "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,
                                           "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")),
                                        "test")

    def get_labels(self):
        return NADI_COUNTRY_LABELS

    def process_tweet(self, tweet):
        """
        processes a tweet by normalizing letters, removing urls,
        and removing diacritics

        Args:
            input tweet
        Returns:
            processed tweet
        """

        # URL regex
        URL_REG = re.compile(r'[https|http|@|RT]([^\s]+)')
        # space regex
        SPACE_REG = re.compile(r'[\s]+')

        tweet = normalize.normalize_unicode(tweet)
        tweet = dediac.dediac_ar(tweet)
        tweet = URL_REG.sub(' ', tweet)
        tweet = SPACE_REG.sub(' ', tweet)
        return tweet

processors = {
    "arabic_sentiment": ArabicSentimentProcessor,
    "arabic_poetry": ArabicPoetryProcessor,
    "arabic_did_madar_26": ArabicDIDProcessor_MADAR_26,
    "arabic_did_madar_6": ArabicDIDProcessor_MADAR_6,
    "arabic_did_madar_twitter": ArabicDIDProcessor_MADAR_Twitter,
    "arabic_did_nadi_country": ArabicDIDProcessor_NADI_COUNTRY
}

output_modes = {
    "arabic_sentiment": "classification",
    "arabic_poetry": "classification",
    "arabic_did_madar_26": "classification",
    "arabic_did_madar_6": "classification",
    "arabic_did_madar_twitter": "classification",
    "arabic_did_nadi_country": "classification"

}
