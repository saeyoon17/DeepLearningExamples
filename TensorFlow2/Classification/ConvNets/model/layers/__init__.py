# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from model.layers.activations import (gelu, get_activation, hard_swish,
                                      identity, simple_swish)
from model.layers.normalization import get_batch_norm

__all__ = [
    "simple_swish",
    "hard_swish",
    "identity",
    "gelu",
    "get_activation",
    "get_batch_norm",
]
