""" Create Conv2d Factory Method
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

from .cond_conv2d import CondConv2d
from .conv2d_same import create_conv2d_pad
from .mixed_conv2d import MixedConv2d


def create_conv2d(in_channels, out_channels, kernel_size, **kwargs):
    """Select a 2d convolution implementation based on arguments
    Creates and returns one of torch.nn.Conv2d, Conv2dSame, MixedConv2d, or CondConv2d.
    Used extensively by EfficientNet, MobileNetv3 and related networks.
    """
    if isinstance(kernel_size, list):
        assert (
            "num_experts" not in kwargs
        )  # MixNet + CondConv combo not supported currently
        assert "groups" not in kwargs  # MixedConv groups are defined by kernel list
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        m = MixedConv2d(in_channels, out_channels, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop("depthwise", False)
        groups = out_channels if depthwise else kwargs.pop("groups", 1)
        if "num_experts" in kwargs and kwargs["num_experts"] > 0:
            m = CondConv2d(
                in_channels, out_channels, kernel_size, groups=groups, **kwargs
            )
        else:
            m = create_conv2d_pad(
                in_channels, out_channels, kernel_size, groups=groups, **kwargs
            )
    return m
