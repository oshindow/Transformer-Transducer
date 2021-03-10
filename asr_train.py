#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Tomoki Hayashi (Nagoya University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Automatic speech recognition model training script."""

import logging
import os
import random
import sys
import json
import numpy as np
from argument import get_parser
from data import Dataloader
from model import Transducer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main(cmd_args):
    """Run the main training function."""

    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # set random seed
    logging.info("random seed = %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dictionary for debug log
    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None

    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim = int((valid_json[utts[0]]["input"][0]["shape"][-1]))
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])

    logging.info("input dims: " + str(idim))
    logging.info("#output dims: " + str(odim))

    # data
    Data = Dataloader(args)

    # model
    Model = Transducer(idim, odim, args)

    # update saved model
    call_back = ModelCheckpoint(monitor='val_loss', dirpath=args.outdir)

    # train model
    trainer = Trainer(gpus=args.ngpu, callbacks=[call_back], max_epochs=args.epochs, resume_from_checkpoint=args.resume)
    trainer.fit(Model, Data)


if __name__ == "__main__":
    main(sys.argv[1:])
