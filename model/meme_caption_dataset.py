import numpy as np
import tensorflow as tf
from tqdm import tqdm
from typing import List

from model.config import MemeGeneratorConfig

TF_RECORD_FILE_PATH = "../data/tokenized_captions.tfrecord"
MAX_SEQ_LENGTH = MemeGeneratorConfig().max_seq_length
DATASET_SIZE = 60214

FEATURE_DESCRIPTION = {
    "input_ids": tf.io.FixedLenSequenceFeature([MAX_SEQ_LENGTH],
                                               tf.int64,
                                               allow_missing=True),
    "output_ids": tf.io.FixedLenSequenceFeature([1],
                                                tf.int64,
                                                allow_missing=True),
}


def _int64_feature(values: List[int]) -> tf.train.Feature:
    """Returns an int64_list from a bool / enum / int / uint."""
    values_np = np.array(values, dtype="int64")
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values_np))


def _create_data_record(tokenized_captions: List[List[int]],
                        labels: List[List[int]],
                        output_file_path: str = TF_RECORD_FILE_PATH):
    """
    Save training data as a TFRecord for faster training
    """
    with tf.io.TFRecordWriter(output_file_path) as record_writer:
        for input_ids, output_ids in tqdm(zip(tokenized_captions, labels),
                                          desc="Converting to TFRecord"):
            feature = {
                "input_ids": _int64_feature(input_ids),
                "output_ids": _int64_feature(output_ids)
            }

            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            record_writer.write(example.SerializeToString())


def _parse_comment_record(serialized_example):
    features = tf.io.parse_single_example(serialized_example,
                                          features=FEATURE_DESCRIPTION)
    input_ids = features["input_ids"]
    labels = features["output_ids"]
    output_ids = tf.one_hot(labels, 50304)
    return input_ids, output_ids


def load_caption_dataset(tf_record_file_path: str = TF_RECORD_FILE_PATH):
    raw_dataset = tf.data.TFRecordDataset(tf_record_file_path)
    return raw_dataset.map(_parse_comment_record)
