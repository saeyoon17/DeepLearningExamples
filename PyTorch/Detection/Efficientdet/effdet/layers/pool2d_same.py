""" AvgPool2d w/ Same Padding
Hacked together by / Copyright 2020 Ross Wightman
"""
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

# Copyright 2019-2022 Ross Wightman
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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .helpers import tup_pair
from .padding import get_padding_value, pad_same


def avg_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    ceil_mode: bool = False,
    count_include_pad: bool = True,
):
    # FIXME how to deal with count_include_pad vs not for external padding?
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'SAME' wrapper for 2D average pooling"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        super(AvgPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), ceil_mode, count_include_pad
        )

    def forward(self, x):
        return avg_pool2d_same(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )


def max_pool2d_same(
    x,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int] = (0, 0),
    dilation: List[int] = (1, 1),
    ceil_mode: bool = False,
):
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation, ceil_mode)


class MaxPool2dSame(nn.MaxPool2d):
    """Tensorflow like 'SAME' wrapper for 2D max pooling"""

    def __init__(
        self,
        kernel_size: int,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
        count_include_pad=True,
    ):
        kernel_size = tup_pair(kernel_size)
        stride = tup_pair(stride)
        dilation = tup_pair(dilation)
        super(MaxPool2dSame, self).__init__(
            kernel_size, stride, (0, 0), dilation, ceil_mode, count_include_pad
        )

    def forward(self, x):
        return max_pool2d_same(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )


def create_pool2d(pool_type, kernel_size, stride=None, **kwargs):
    stride = stride or kernel_size
    padding = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
        padding, kernel_size, stride=stride, **kwargs
    )
    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride=stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride=stride, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
    else:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride=stride, padding=padding, **kwargs)
        else:
            assert False, f"Unsupported pool type {pool_type}"
