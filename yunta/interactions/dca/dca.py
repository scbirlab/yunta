"""Code from Humpreys, Science, 2021 (DOI: 10.1126/science.abm4805), reimplemented for Tensorflow 2.

According to the paper's supplementary material:
"We reimplemented DCA so that it can be computed using GPUs to speed up the calculation.
The code is in the box below. In addition, we applied average product correction (APC).[...]"

```python
import sys
import numpy as np
import string
import tensorflow as tf

def parse_a3m(a3mlines):
    seqs = []
    labels = []
    for line in a3mlines:
        if line[0] == '>':
            labels.append(line.rstrip())
        else:
            seqs.append(line[:-1])
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    seq_num = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        seq_num[seq_num == alphabet[i]] = i
    seq_num[seq_num > 20] = 20
    return {'seqs' : seq_num, 'labels' : labels }

def tf_cov(x,w=None):
    if w is None:
        num_points = tf.cast(tf.shape(x)[0],tf.float32) - 1
        x_mean = tf.reduce_mean(x, axis=0, keep_dims=True)
        x = (x - x_mean)
    else:
        num_points = tf.reduce_sum(w) - tf.sqrt(tf.reduce_mean(w))
        x_mean = tf.reduce_sum(x * w[:,None], axis=0, keepdims=True) / num_points
        x = (x - x_mean) * tf.sqrt(w[:,None])
    return tf.matmul(tf.transpose(x),x)/num_points

fp = open(sys.argv[1] + ".alignments", "r")
pair2a3m = {}
pair = ""
pairs = []
a3m = []
for line in fp:
    if line[:2] == ">>":
        if pair and a3m:
            pair2a3m[pair] = a3m
            pairs.append(pair)
        pair = line[2:-1]
        a3m = []
    else:
        a3m.append(line)
fp.close()
if pair and a3m:
    pair2a3m[pair] = a3m
    pairs.append(pair)

config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))

with tf.Graph().as_default():
    x = tf.placeholder(tf.uint8,shape=(None,None),name="x")
    x_shape = tf.shape(x)
    x_nr = x_shape[0]
    x_nc = x_shape[1]
    x_ns = 21
    x_msa = tf.one_hot(x,x_ns)
    x_cutoff = tf.cast(x_nc,tf.float32) * 0.8
    x_pw = tf.tensordot(x_msa, x_msa, [[1,2], [1,2]])
    x_cut = x_pw > x_cutoff
    x_weights = 1.0/tf.reduce_sum(tf.cast(x_cut, dtype=tf.float32),-1)
    x_feat = tf.reshape(x_msa,(x_nr,x_nc*x_ns))
    x_c = tf_cov(x_feat,x_weights) + tf.eye(x_nc*x_ns) *
    4.5/tf.sqrt(tf.reduce_sum(x_weights))
    x_c_inv = tf.linalg.inv(x_c)
    x_w = tf.reshape(x_c_inv,(x_nc,x_ns,x_nc,x_ns))
    x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(x_nc))
    # APC
    #x_ap = tf.reduce_sum(x_wi,0,keepdims=True) * tf.reduce_sum(x_wi,1,keepdims=True) /
    tf.reduce_sum(x_wi)
    #x_wip = (x_wi - x_ap) * (1-tf.eye(x_nc))

    with tf.Session(config=config) as sess:
        results = []
        get_pairs = []
        for pair in pairs:
            print (pair)
            msa = parse_a3m(pair2a3m[pair])
            try:
                wip = sess.run(x_wi,{x:msa['seqs']})
                results.append(wip.astype(np.float16))
                get_pairs.append(pair)
            except tf.errors.ResourceExhaustedError as e:
                pass
        np.savez_compressed(sys.argv[1], *results, names=get_pairs)
        rp = open(sys.argv[1] + ".log","w")
        for pair in get_pairs:
            rp.write(pair + "\n")
        rp.close()
```
"""

from typing import Optional, Union
from io import TextIOWrapper
import sys

from carabiner import print_err
import numpy as np
import tensorflow as tf

from .structs.msa import MSA, _A3M_ALPHABET

@tf.function
def _tf_cov(x: tf.Tensor, w: Optional[tf.Tensor] = None) -> tf.float32:
    if w is None:
        num_points = tf.cast(tf.shape(x)[0], tf.float32) - 1.
        x_mean = tf.math.reduce_mean(x, axis=0, keep_dims=True)
        x = (x - x_mean)
    else:
        num_points = tf.math.reduce_sum(w) - tf.math.sqrt(tf.math.reduce_mean(w))
        x_mean = tf.math.reduce_sum(x * w[:,tf.newaxis], axis=0, keepdims=True) / num_points
        x = (x - x_mean) * tf.math.sqrt(w[:,tf.newaxis])
    return tf.linalg.matmul(x, x, transpose_a=True) / num_points


@tf.function
def _get_wip(x: tf.Tensor, apc: bool = False) -> tf.float32:
    alphabet_size = len(_A3M_ALPHABET)
    n_row, n_col = tf.shape(x)[-2], tf.shape(x)[-1]
    x_cutoff = tf.cast(n_col, dtype=tf.float32) * .8
    msa_one_hot = tf.one_hot(x, alphabet_size)
    x_pw = tf.tensordot(msa_one_hot, msa_one_hot, [[1,2], [1,2]])
    x_cut = tf.cast(x_pw > x_cutoff, 
                    dtype=tf.float32)

    weights = 1. / tf.math.reduce_sum(x_cut, axis=-1)
    x_feat = tf.reshape(msa_one_hot, (n_row, n_col * alphabet_size))

    x_c = (_tf_cov(x_feat, weights) + tf.eye(n_col * alphabet_size) * 4.5 / 
           tf.sqrt(tf.math.reduce_sum(weights)))
    x_c_inv = tf.linalg.inv(x_c)

    x_w = tf.reshape(x_c_inv, (n_col, alphabet_size, n_col, alphabet_size))
    x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[:,:-1,:,:-1]), (1, 3))) * (1. - tf.eye(n_col))

    if apc:
        x_ap = (tf.math.reduce_sum(x_wi, axis=0, keepdims=True) * tf.math.reduce_sum(x_wi, axis=1, keepdims=True) 
                / tf.math.reduce_sum(x_wi))
        x_wi = (x_wi - x_ap) * (1. - tf.eye(n_col))

    return x_wi


def calculate_dca(msa: MSA, 
                  apc: bool = False,
                  gpu: bool = True) -> np.ndarray:

    """
    
    """
    print_err(f"Devices available:\n{tf.config.list_physical_devices()}")
    if gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) == 0:
            print_err("WARNING! GPU requested but none available. Falling back to CPU.")
    msa_token_ids = np.asarray(msa.sequence_token_ids)
    try:
        wip = _get_wip(msa_token_ids, apc=apc)
    except tf.errors.ResourceExhaustedError as e:
        with tf.device('/cpu:0'):
            wip = _get_wip(msa_token_ids, apc=apc)

    return wip.numpy()