import os

import numpy as np
import torch
from data_utils import TFTDataset
from torch.utils.data import DataLoader


def update_argparser(parser):
    parser.add_argument(
        "--dataset", type=str, help="Path to dataset to be used", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint to be used", required=True
    )
    parser.add_argument(
        "--batch-size", type=int, help="Path to dataset to be used", default=64
    )


def get_dataloader_fn(dataset, checkpoint, batch_size=64):
    state_dict = torch.load(os.path.join(checkpoint, "checkpoint.pt"))
    config = state_dict["config"]
    test_split = TFTDataset(os.path.join(dataset, "test.csv"), config)
    data_loader = DataLoader(test_split, batch_size=int(batch_size), num_workers=2)
    input_names_dict = {
        "s_cat": "s_cat__0",
        "s_cont": "s_cont__1",
        "k_cat": "k_cat__2",
        "k_cont": "k_cont__3",
        "o_cat": "o_cat__4",
        "o_cont": "o_cont__5",
        "target": "target__6",
        "id": "id__7",
    }
    reshaper = [-1] + [1]

    def _get_dataloader():
        for step, batch in enumerate(data_loader):
            bs = batch["target"].shape[0]
            x = {
                input_names_dict[key]: tensor.numpy()
                if tensor.numel()
                else np.ones([bs]).reshape(reshaper)
                for key, tensor in batch.items()
            }
            ids = batch["id"][:, 0, :].numpy()
            # ids = np.arange(step * batch_size, (step + 1) * batch_size)
            y_real = {
                "target__0": np.tile(
                    batch["target"][:, config.encoder_length :, :].numpy(),
                    (1, 1, len(config.quantiles)),
                )
            }
            yield (ids, x, y_real)

    return _get_dataloader
