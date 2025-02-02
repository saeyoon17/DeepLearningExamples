#!/usr/bin/env python
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import concurrent
import math
import os
import queue
import warnings
from collections import deque
from functools import reduce
from itertools import combinations_with_replacement
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset


def setup_distributed_print(enable):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if enable or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_distributed() -> bool:
    return get_world_size() > 1


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(backend="nccl", use_gpu=True):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_RANK" in os.environ and "OMPI_COMM_WORLD_SIZE" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        gpu = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Not using distributed mode")
        return 0, 1, 0

    if use_gpu:
        torch.cuda.set_device(gpu)

    if rank != 0:
        warnings.filterwarnings("ignore")

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, init_method="env://"
    )

    return rank, world_size, gpu


def get_gpu_batch_sizes(
    global_batch_size: int,
    num_gpus: int = 4,
    batch_std: int = 64,
    divisible_by: int = 64,
):
    batch_avg = global_batch_size // num_gpus
    start, end = batch_avg - batch_std, batch_avg + batch_std
    sizes_range = (x for x in range(start, end + 1) if x % divisible_by == 0)
    solutions = [
        sizes
        for sizes in combinations_with_replacement(sizes_range, num_gpus)
        if sum(sizes) == global_batch_size
    ]

    if not solutions:
        raise RuntimeError(
            "Could not find GPU batch sizes for a given configuration. "
            "Please adjust global batch size or number of used GPUs."
        )

    return max(solutions, key=lambda sizes: reduce(lambda x, y: x * y, sizes))


def argsort(sequence, reverse: bool = False):
    idx_pairs = [(x, i) for i, x in enumerate(sequence)]
    sorted_pairs = sorted(idx_pairs, key=lambda pair: pair[0], reverse=reverse)
    return [i for _, i in sorted_pairs]


def distribute_to_buckets(sizes: Sequence[int], buckets_num: int):
    def sum_sizes(indices):
        return sum(sizes[i] for i in indices)

    max_bucket_size = math.ceil(len(sizes) / buckets_num)
    idx_sorted = deque(argsort(sizes, reverse=True))
    buckets = [[] for _ in range(buckets_num)]
    final_buckets = []

    while idx_sorted:
        bucket = buckets[0]
        bucket.append(idx_sorted.popleft())

        if len(bucket) == max_bucket_size:
            final_buckets.append(buckets.pop(0))

        buckets.sort(key=sum_sizes)

    final_buckets += buckets

    return final_buckets


def get_device_mapping(embedding_sizes: Sequence[int], num_gpus: int = 8):
    """Get device mappings for hybrid parallelism

    Bottom MLP running on device 0. Embeddings will be distributed across among all the devices.

    Optimal solution for partitioning set of N embedding tables into K devices to minimize maximal subset sum
    is an NP-hard problem. Additionally, embedding tables distribution should be nearly uniform due to the performance
    constraints. Therefore, suboptimal greedy approach with max bucket size is used.

    Args:
        embedding_sizes (Sequence[int]): embedding tables sizes
        num_gpus (int): Default 8.

    Returns:
        device_mapping (dict):
    """
    if num_gpus > 4:
        # for higher no. of GPUs, make sure the one with bottom mlp has no embeddings
        gpu_buckets = distribute_to_buckets(
            embedding_sizes, num_gpus - 1
        )  # leave one device out for the bottom MLP
        gpu_buckets.insert(0, [])
    else:
        gpu_buckets = distribute_to_buckets(embedding_sizes, num_gpus)

    vectors_per_gpu = [len(bucket) for bucket in gpu_buckets]
    vectors_per_gpu[0] += 1  # count bottom mlp

    return {
        "bottom_mlp": 0,
        "embedding": gpu_buckets,
        "vectors_per_gpu": vectors_per_gpu,
    }


class SyntheticDataset(Dataset):
    """Synthetic dataset version of criteo dataset."""

    def __init__(
        self,
        num_entries: int,
        device: str = "cuda",
        batch_size: int = 32768,
        numerical_features: Optional[int] = None,
        categorical_feature_sizes: Optional[
            Sequence[int]
        ] = None,  # features are returned in this order
    ):
        cat_features_count = (
            len(categorical_feature_sizes)
            if categorical_feature_sizes is not None
            else 0
        )
        num_features_count = numerical_features if numerical_features is not None else 0

        self._batches_per_epoch = math.ceil(num_entries / batch_size)
        self._num_tensor = (
            torch.rand(
                size=(batch_size, num_features_count),
                device=device,
                dtype=torch.float32,
            )
            if num_features_count > 0
            else None
        )
        self._label_tensor = torch.randint(
            low=0, high=2, size=(batch_size,), device=device, dtype=torch.float32
        )
        self._cat_tensor = (
            torch.cat(
                [
                    torch.randint(
                        low=0,
                        high=cardinality,
                        size=(batch_size, 1),
                        device=device,
                        dtype=torch.long,
                    )
                    for cardinality in categorical_feature_sizes
                ],
                dim=1,
            )
            if cat_features_count > 0
            else None
        )

    def __len__(self):
        return self._batches_per_epoch

    def __getitem__(self, idx: int):
        if idx >= self._batches_per_epoch:
            raise IndexError()

        return self._num_tensor, self._cat_tensor, self._label_tensor


import argparse
import json
import sys

import numpy as np
import torch
import tritonclient.http as http_client
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# from dlrm.data.datasets import SyntheticDataset
# from dlrm.utils.distributed import get_device_mapping


def get_data_loader(batch_size, *, data_path, model_config):
    with open(model_config.dataset_config) as f:
        categorical_sizes = list(json.load(f).values())
    categorical_sizes = [s - 1 for s in categorical_sizes]

    device_mapping = get_device_mapping(categorical_sizes, num_gpus=1)

    if data_path:
        data = SplitCriteoDataset(
            data_path=data_path,
            batch_size=batch_size,
            numerical_features=True,
            categorical_features=device_mapping["embedding"][0],
            categorical_feature_sizes=categorical_sizes,
            prefetch_depth=1,
            drop_last_batch=model_config.drop_last_batch,
        )
    else:
        data = SyntheticDataset(
            num_entries=batch_size * 1024,
            batch_size=batch_size,
            numerical_features=model_config.num_numerical_features,
            categorical_feature_sizes=categorical_sizes,
            device="cpu",
        )

    if model_config.test_batches > 0:
        data = torch.utils.data.Subset(data, list(range(model_config.test_batches)))

    return torch.utils.data.DataLoader(
        data, batch_size=None, num_workers=0, pin_memory=False
    )


def run_infer(
    model_name, model_version, numerical_features, categorical_features, headers=None
):
    inputs = []
    outputs = []
    num_type = "FP16" if numerical_features.dtype == np.float16 else "FP32"
    inputs.append(
        http_client.InferInput("input__0", numerical_features.shape, num_type)
    )
    inputs.append(
        http_client.InferInput("input__1", categorical_features.shape, "INT64")
    )

    # Initialize the data
    inputs[0].set_data_from_numpy(numerical_features, binary_data=True)
    inputs[1].set_data_from_numpy(categorical_features, binary_data=False)

    outputs.append(http_client.InferRequestedOutput("output__0", binary_data=True))
    results = triton_client.infer(
        model_name,
        inputs,
        model_version=str(model_version) if model_version != -1 else "",
        outputs=outputs,
        headers=headers,
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--triton-server-url",
        type=str,
        required=True,
        help="URL adress of triton server (with port)",
    )
    parser.add_argument(
        "--triton-model-name",
        type=str,
        required=True,
        help="Triton deployed model name",
    )
    parser.add_argument(
        "--triton-model-version", type=int, default=-1, help="Triton model version"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-H",
        dest="http_headers",
        metavar="HTTP_HEADER",
        required=False,
        action="append",
        help="HTTP headers to add to inference server requests. "
        + 'Format is -H"Header:Value".',
    )

    parser.add_argument("--dataset_config", type=str, required=True)
    parser.add_argument(
        "--inference_data", type=str, help="Path to file with inference data."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Inference request batch size"
    )
    parser.add_argument(
        "--drop_last_batch",
        type=bool,
        default=True,
        help="Drops the last batch size if it's not full",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use 16bit for numerical input",
    )
    parser.add_argument(
        "--test_batches",
        type=int,
        default=0,
        help="Specifies number of batches used in the inference",
    )

    FLAGS = parser.parse_args()
    try:
        url = FLAGS.triton_server_url
        ssl = False
        if url.startswith("https://"):
            url = url[len("https://") :]
            ssl = True
        triton_client = http_client.InferenceServerClient(
            url=url, verbose=FLAGS.verbose, ssl=ssl
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit(1)

    if FLAGS.http_headers is not None:
        headers_dict = {l.split(":")[0]: l.split(":")[1] for l in FLAGS.http_headers}
    else:
        headers_dict = None

    triton_client.load_model(FLAGS.triton_model_name)
    if not triton_client.is_model_ready(FLAGS.triton_model_name):
        sys.exit(1)

    dataloader = get_data_loader(
        FLAGS.batch_size, data_path=FLAGS.inference_data, model_config=FLAGS
    )
    results = []
    tgt_list = []

    for numerical_features, categorical_features, target in tqdm(dataloader):
        numerical_features = numerical_features.cpu().numpy()
        numerical_features = numerical_features.astype(
            np.float16 if FLAGS.fp16 else np.float32
        )
        categorical_features = categorical_features.long().cpu().numpy()

        output = run_infer(
            FLAGS.triton_model_name,
            FLAGS.triton_model_version,
            numerical_features,
            categorical_features,
            headers_dict,
        )

        results.append(output.as_numpy("output__0"))
        tgt_list.append(target.cpu().numpy())

    results = np.concatenate(results).squeeze()
    tgt_list = np.concatenate(tgt_list)

    score = roc_auc_score(tgt_list, results)
    print(f"Model score: {score}")

    statistics = triton_client.get_inference_statistics(
        model_name=FLAGS.triton_model_name, headers=headers_dict
    )
    print(statistics)
    if len(statistics["model_stats"]) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
