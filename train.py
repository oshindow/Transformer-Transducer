import os
import shutil
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import json
import configargparse
from tt.optim import Optimizer
from tt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer
from tt.batchfy import make_batchset
from tt.io_utils import LoadInputsAndTargets
from tt.chainer_dataset import ChainerDataLoader
from tt.chainer_dataset import TransformDataset
from tt.net_utils import pad_list
import test


def get_parser(parser=None, required=False):

    if parser is None:
        parser = configargparse.ArgumentParser()

    # general configuration
    parser.add_argument('-config', type=str, default='config/timit.yaml')
    parser.add_argument('-log', type=str, default='train.log')

    return parser


def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

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


def train(epoch, config, model, train_iter, optimizer, logger, device):

    # begin training
    model.train()
    total_loss = 0

    # for save model
    optimizer.epoch()

    batch_steps = train_iter.len

    for step in range(batch_steps):

        batch = train_iter.next()
        batch = _recursive_to(batch, device)
        optimizer.zero_grad()
        start = time.process_time()

        loss = model(*batch)

        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(
           model.parameters(), config.training.max_grad_norm)

        # refresh learning rate
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)

        if step % config.training.show_interval == 0:

            process = step / batch_steps * 100
            end = time.process_time()

            if hasattr(optimizer, '_rate'):
                logger.info('-Training-Epoch:%d, process:%.2f %%, avg_loss:%.2f, grad norm:%.2f, lr:%.6f, time:%2f'
                     % (epoch, process, avg_loss, grad_norm, optimizer._rate, (end-start)))
            else:
                logger.info('-Training-Epoch:%d, process:%.2f %%, avg_loss:%.2f, grad norm:%.2f, lr:%.6f, time:%2f'
                     % (epoch, process, avg_loss, grad_norm, optimizer.lr, (end-start)))


def main():
    # set program parameters.
    # args include:
    # args.config for the configfile address
    # args.log for train.log save address
    # args.mode for retrain or continue
    parser = get_parser()
    args = parser.parse_args()

    # load yaml config file
    configfile = open(args.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    # set device
    device = torch.device("cuda" if config.ngpu > 0 else "cpu")

    # set exp dir and copy the config file to exp dir
    exp_name = os.path.join('egs', config.data.name, 'exp', config.training.save_model)
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, args.log))
    shutil.copyfile(args.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    # read json data to form dict inputs and target data
    with open(config.data.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(config.data.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    # read idim and odim from json data
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1])
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])

    # minibatch is a list of batches
    # make minibatch list
    # len(minibatch) = len(data) / batchsize

    train_set = make_batchset(
        train_json,
        config.data.batch_size,
        config.data.max_input_length,
        config.data.max_target_length,
        num_batches=config.minibatch,
        min_batch_size=config.ngpu if config.ngpu > 1 else 1,
    )

    # load inputs and targets
    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )

    converter = CustomConverter()

    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    # next, __iter__, epoch_detail, serialize, start_shuffle, finalize

    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train_set, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=1,
        shuffle=config.data.shuffle,
        collate_fn=lambda x: x[0],
    )

    # set random seed
    if config.ngpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    # model

    from tt.model import Transducer
    model = Transducer(idim, odim, args)
    if config.ngpu != 0:
        model = model.cuda()

    # write model config
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    model_conf = exp_name + "/model.json"
    with open(model_conf, "wb") as f:
        logger.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim, odim, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logger.info("ARGS: " + key + ": " + str(vars(args)[key]))

    # load model param from exist model
    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        # model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    # count model param and initialize optimizer
    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    # logger.info('# the number of parameters in the JointNet: %d' % (n_params - dec - enc))
    # logger.info('Created a %s optimizer.' % config.optim.type)

    # Setup an optimizer
    if config.optim.type == 'noam':
        from tt.noam import get_std_opt
        optimizer = get_std_opt(
            model.parameters(), config.d_model, config.transformer_warmup_steps, config.optim.lr
            )
    elif config.optim.type == 'adam':
        optimizer = Optimizer(model.parameters(), config.optim)

    for epoch in range(config.training.epochs):

        train(epoch, config, model, train_iter, optimizer, logger, device)

        save_name = os.path.join(exp_name, '%s.epoch%d.ckpt' % (config.training.save_model, epoch))
        save_model(model, optimizer, config, save_name)
        logger.info('Epoch %d model has been saved.' % epoch)

        if config.training.eval_or_not:
            test.eval()


if __name__ == '__main__':
    main()
