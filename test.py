import yaml
import torch
import torch.utils.data
import json
import logging
import configargparse
import editdistance
import numpy as np
from numpy import mean
from tt.utils import AttrDict
from tt.asr_utils import add_results_to_json
from tt.io_utils import LoadInputsAndTargets
logging.basicConfig(level=logging.INFO)


def get_parser(parser=None, required=False):
    if parser is None:
        parser = configargparse.ArgumentParser()
    # general configuration
    parser.add_argument('-config', type=str, default='config/timit.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-model', type=str, default='egs/timit/exp/test/test.epoch0.ckpt')

    return parser


def eval():
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.

    """
    parser = get_parser()
    args = parser.parse_args()

    # load yaml config file
    configfile = open(args.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    # read json data
    with open(config.data.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    # read idim and odim from json data
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]["input"][0]["shape"][-1])
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    # model, train_args = load_trained_model(args.model)
    from tt.model import Transducer
    model = Transducer(idim, odim, args)
    checkpoint = torch.load(args.model)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.recog_args = args

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    rnnlm = None

    if config.data.vocab is not None:
        with open(config.data.vocab, "rb") as f:
            dictionary = f.readlines()
        char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        char_list.insert(0, "<blank>")
        char_list.append("<eos>")
        args.char_list = char_list

    # gpu
    if config.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info("gpu id: " + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

    new_js = {}

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_args={"train": False},
    )
    # model.eval()  ## training: false
    avg_cer = []
    with torch.no_grad():  ## training: false
        for idx, name in enumerate(valid_json.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(valid_json.keys()))
            batch = [(name, valid_json[name])]
            feat = load_inputs_and_targets(batch)
            feat = (
                feat[0][0]
            )
            nbest_hyps = model.recognize(
                    feat, args, args.char_list, rnnlm
                )
            new_js[name] = add_results_to_json(
                    valid_json[name], nbest_hyps, args.char_list
                )

            hyp_chars = new_js[name]['output'][0]['rec_text'].replace(" ", "")
            ref_chars = valid_json[name]['output'][0]['text'].replace(" ", "")
            char_eds = editdistance.eval(hyp_chars, ref_chars)

            cer = float(char_eds / len(ref_chars)) * 100
            avg_cer.append(cer)
            logging.info("{} cer: {}".format(name, cer))

    logging.info('avg_cer:{}'.format(mean(np.array(avg_cer))))

    with open('result.txt', "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


if __name__ == '__main__':
    eval()
