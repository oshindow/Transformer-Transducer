import json
import torch
import numpy as np
from tt.utils.io_utils import LoadInputsAndTargets
from tt.batchfy import make_batchset
from tt.utils.net_utils import pad_list
from pytorch_lightning import LightningDataModule
from tt.utils.dataset_utils import TransformDataset


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32, left_context_width=3, right_context_width=0):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype
        self.left_context_width = left_context_width
        self.right_context_width = right_context_width

    def concat_frame(self, features, left_context_width, right_context_width):
        bsz = len(features)

        stack = []
        for b in range(bsz):
            time_steps, features_dim = features[b].shape
            concated_features = np.zeros(
                shape=[time_steps, features_dim *
                    (1 + left_context_width + right_context_width)],
                dtype=features[b].dtype)
            # middle part is just the uttarnce
            concated_features[:, left_context_width * features_dim:
                             (left_context_width + 1) * features_dim] = features[b]

            for i in range(left_context_width):
                # add left context
                concated_features[i + 1:time_steps,
                              (left_context_width - i - 1) * features_dim:
                              (left_context_width - i) * features_dim] = features[b][0:time_steps - i - 1, :]

            for i in range(right_context_width):
                # add right context
                concated_features[0:time_steps - i - 1,
                              (right_context_width + i + 1) * features_dim:
                              (right_context_width + i + 2) * features_dim] = features[b][i + 1:time_steps, :]

            concated_features = np.delete(concated_features, range(left_context_width), axis=0)
            concated_features = np.delete(concated_features, [(x + concated_features.shape[0] - right_context_width) for x in range(right_context_width)], axis=0)
            stack.append(concated_features)
        return stack

    def __call__(self, batch, device=torch.device("cpu")):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]
        # logging.info("xs:{}".format(xs[0]))
        # logging.info("xs:{}".format(xs[0].shape))
        # xs = self.concat_frame(xs, 3, 0)
        # logging.info("xs_:{}".format(xs[0]))
        # logging.info("xs_:{}".format(xs[0].shape))
        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[:: self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == "c":
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0
            ).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {"real": xs_pad_real, "imag": xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(
                device, dtype=self.dtype
            )

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list(
            [
                torch.from_numpy(
                    np.array(y[0][:]) if isinstance(y, tuple) else y
                ).long()
                for y in ys
            ],
            self.ignore_id,
        ).to(device)

        return xs_pad, ilens, ys_pad


def training_data_process(args):
    # Setup a converter

    converter = CustomConverter(subsampling_factor=1)

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
    )
    valid = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
    )

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )
    return train, valid, load_tr, load_cv, converter


class Dataloader(LightningDataModule):

    def __init__(self, args):
        super(Dataloader, self).__init__()
        self.args = args

    def prepare_data(self):
        self.train, self.valid, self.load_tr, self.load_cv, self.converter = training_data_process(self.args)

    def train_dataloader(self):
        train_iter = torch.utils.data.dataloader.DataLoader(
            dataset=TransformDataset(self.train, lambda data: self.converter([self.load_tr(data)])),
            batch_size=1,
            num_workers=self.args.n_iter_processes,
            shuffle=True,
            collate_fn=lambda x: x[0],
        )
        return train_iter

    def val_dataloader(self):
        valid_iter = torch.utils.data.dataloader.DataLoader(
            dataset=TransformDataset(self.valid, lambda data: self.converter([self.load_cv(data)])),
            batch_size=1,
            num_workers=self.args.n_iter_processes,
            shuffle=False,
            collate_fn=lambda x: x[0],
        )
        return valid_iter

