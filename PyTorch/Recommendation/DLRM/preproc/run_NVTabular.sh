#!/bin/bash

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

#########################################################################
# File Name: run_NVTabular.sh

set -e

# the data path including 1TB criteo data, day_0, day_1, ...
export INPUT_PATH=${1:-'/data/dlrm/criteo'}

# the output path, use for generating the dictionary and the final dataset
# the output folder should have more than 300GB
export OUTPUT_PATH=${2:-'/data/dlrm/output'}

export FREQUENCY_LIMIT=${3:-'15'}

export CRITEO_PARQUET=${4:-'/data/dlrm/criteo_parquet'}

if [ "$DGX_VERSION" = "DGX-2" ]; then
    export DEVICES=0
else
    export DEVICES=0
fi

echo "Preprocessing data"
python preproc_NVTabular.py $INPUT_PATH $OUTPUT_PATH --devices $DEVICES --intermediate_dir $CRITEO_PARQUET --freq_threshold $FREQUENCY_LIMIT
