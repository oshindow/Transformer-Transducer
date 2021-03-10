#!/usr/bin/env python3
# encoding: utf-8

import logging
import torch
import sys
import json
from tt.utils.asr_utils import add_results_to_json
from tt.utils.io_utils import LoadInputsAndTargets
from argument import get_parser
from model import Transducer


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

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

    with open(args.recog_json, "rb") as f:
        test_json = json.load(f)["utts"]
    utts = list(test_json.keys())
    idim = int((test_json[utts[0]]["input"][0]["shape"][-1]))
    odim = int(test_json[utts[0]]["output"][0]["shape"][-1])

    load_test = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=None,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )

    if args.dict is not None:
        with open(args.dict, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list
    else:
        args.char_list = None

    Model = Transducer.load_from_checkpoint(args.model, idim=idim, odim=odim, args=args)

    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(test_json.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(test_json.keys()))
            batch = [(name, test_json[name])]
            feat = load_test(batch)[0][0]

            nbest_hyps = Model.recog(feat)

            new_js[name] = add_results_to_json(
                    test_json[name], nbest_hyps, args.char_list
                )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


if __name__ == "__main__":
    main(sys.argv[1:])
