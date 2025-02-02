# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, Optional

import torch
from dlrm.utils.checkpointing.model import (DlrmCheckpointLoader,
                                            DlrmCheckpointWriter)


class DistributedCheckpointWriter:
    def __init__(
        self,
        writer: DlrmCheckpointWriter,
        device_mapping: Dict[str, Any],
        rank: int,
        main_process: bool,
    ):
        self._device_mapping = device_mapping
        self._main_process = main_process
        self._has_bottom_mlp = rank == device_mapping["bottom_mlp"]
        self._writer = writer
        self._distributed = len(device_mapping["embedding"]) > 1

    def save_checkpoint(
        self,
        model,
        checkpoint_path: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        self._writer.save_embeddings(checkpoint_path, model)

        if self._has_bottom_mlp:
            self._writer.save_bottom_mlp(checkpoint_path, model)

        if self._main_process:
            self._writer.save_top_model(checkpoint_path, model)
            self._save_metadata(checkpoint_path, epoch, step)

        if self._distributed:
            torch.distributed.barrier()

    def _save_metadata(self, checkpoint_path, epoch, step):
        self._writer.save_metadata(
            checkpoint_path,
            {"device_mapping": self._device_mapping, "epoch": epoch, "step": step},
        )


class DistributedCheckpointLoader:
    def __init__(
        self, loader: DlrmCheckpointLoader, device_mapping: Dict[str, Any], rank: int
    ):
        self._has_bottom_mlp = rank == device_mapping["bottom_mlp"]
        self._loader = loader
        self.distributed = len(device_mapping["embedding"]) > 1

    def load_checkpoint(self, model, checkpoint_path: str):
        self._loader.load_top_model(checkpoint_path, model)

        if self._has_bottom_mlp:
            self._loader.load_bottom_mlp(checkpoint_path, model)

        self._loader.load_embeddings(checkpoint_path, model)

        if self.distributed:
            torch.distributed.barrier()


def make_distributed_checkpoint_loader(
    device_mapping, rank: int, device: str = "cpu"
) -> DistributedCheckpointLoader:
    embedding_indices = device_mapping["embedding"][rank]
    return DistributedCheckpointLoader(
        loader=DlrmCheckpointLoader(
            embedding_indices=embedding_indices,
            device=device,
        ),
        device_mapping=device_mapping,
        rank=rank,
    )


def make_distributed_checkpoint_writer(
    device_mapping,
    rank: int,
    is_main_process: bool,
    config: Dict[str, Any],
) -> DistributedCheckpointWriter:
    embedding_indices = device_mapping["embedding"][rank]
    return DistributedCheckpointWriter(
        writer=DlrmCheckpointWriter(embedding_indices=embedding_indices, config=config),
        device_mapping=device_mapping,
        rank=rank,
        main_process=is_main_process,
    )
