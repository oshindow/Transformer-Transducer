#!/usr/bin/env python3
# encoding: utf-8

import configargparse
from distutils.util import strtobool


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--train_config", is_config_file=True, help="Train config file path")
    parser.add("--recog_config", is_config_file=True, help="Recog config file path")

    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument("--debugdir", type=str, help="Output directory for debugging")
    parser.add_argument("--outdir", type=str, help="Output directory")
    parser.add_argument("--dict", help="Dictionary")

    parser.add_argument(
        "--ngpu",
        default=None,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        type=str,
        nargs="?",
        help="Tensorboard log dir path",
    )

    # train related
    parser.add_argument(
        "--resume",
        "-r",
        default="",
        nargs="?",
        help="Resume the training from snapshot",
    )
    parser.add_argument(
        "--minibatches",
        "-N",
        type=int,
        default="-1",
        help="Process only N minibatches (for debug)",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for beam search (0: means no batch processing)",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default=None,
        help="Filename of validation label data (json)",
    )

    # recog related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--result-label",
        type=str,
        help="Filename of result label data (json)",
    )
    parser.add_argument(
        "--model", type=str, help="Model file parameters to read"
    )
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="default",
        choices=["default", "nsc", "tsd", "alsd"],
        help="""Type of beam search implementation to use during inference.
        Can be either: default beam search, n-step constrained beam search ("nsc"),
        time-synchronous decoding ("tsd") or alignment-length synchronous decoding
        ("alsd").
        Additional associated parameters: "nstep" + "prefix-alpha" (for nsc),
        "max-sym-exp" (for tsd) and "u-max" (for alsd)""",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=1,
        help="Number of expansion steps allowed in NSC beam search.",
    )
    parser.add_argument(
        "--prefix-alpha",
        type=int,
        default=2,
        help="Length prefix difference allowed in NSC beam search.",
    )
    parser.add_argument(
        "--max-sym-exp",
        type=int,
        default=2,
        help="Number of symbol expansions allowed in TSD decoding.",
    )
    parser.add_argument(
        "--u-max",
        type=int,
        default=400,
        help="Length prefix difference allowed in ALSD beam search.",
    )

    # rnnlm related
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument(
        "--word-rnnlm", type=str, default=None, help="Word RNNLM model file to read"
    )
    parser.add_argument(
        "--word-rnnlm-conf",
        type=str,
        default=None,
        help="Word RNNLM model config file to read",
    )
    parser.add_argument("--word-dict", type=str, default=None, help="Word list to read")

    # ngram related
    parser.add_argument(
        "--ngram-model", type=str, default=None, help="ngram model file to read"
    )
    parser.add_argument("--ngram-weight", type=float, default=0.1, help="ngram weight")
    parser.add_argument(
        "--ngram-scorer",
        type=str,
        default="part",
        choices=("full", "part"),
        help="""if the ngram is set as a part scorer, similar with CTC scorer,
                ngram scorer only scores topK hypethesis.
                if the ngram is set as full scorer, ngram scorer scores all hypthesis
                the decoding speed of part scorer is musch faster than full one""",
    )

    # loss related
    parser.add_argument(
        "--ctc_type",
        default="warpctc",
        type=str,
        choices=["builtin", "warpctc"],
        help="Type of CTC implementation to calculate loss.",
    )
    parser.add_argument(
        "--mtlalpha",
        default=0.5,
        type=float,
        help="Multitask learning coefficient, "
        "alpha: alpha*ctc_loss + (1-alpha)*att_loss ",
    )
    parser.add_argument(
        "--lsm-weight", default=0.0, type=float, help="Label smoothing weight"
    )

    parser.add_argument("--lm-weight", default=0.0, type=float, help="RNNLM weight.")
    parser.add_argument("--sym-space", default="<space>", type=str, help="Space symbol")
    parser.add_argument("--sym-blank", default="<blank>", type=str, help="Blank symbol")

    # minibatch related
    parser.add_argument(
        "--sortagrad",
        default=0,
        type=int,
        nargs="?",
        help="How many epochs to use sortagrad for. 0 = deactivated, -1 = all epochs",
    )
    parser.add_argument(
        "--batch-size",
        "--batch-seqs",
        "-b",
        default=0,
        type=int,
        help="Maximum seqs in a minibatch (0 to disable)",
    )
    parser.add_argument(
        "--maxlen-in",
        "--batch-seq-maxlen-in",
        default=800,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the input sequence length > ML.",
    )
    parser.add_argument(
        "--maxlen-out",
        "--batch-seq-maxlen-out",
        default=150,
        type=int,
        metavar="ML",
        help="When --batch-count=seq, "
        "batch size is reduced if the output sequence length > ML",
    )
    parser.add_argument(
        "--n-iter-processes",
        default=0,
        type=int,
        help="Number of processes of iterator",
    )

    # optimization related
    parser.add_argument(
        "--opt",
        default="adadelta",
        type=str,
        choices=["adadelta", "adam", "noam"],
        help="Optimizer",
    )
    parser.add_argument(
        "--accum-grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--eps", default=1e-8, type=float, help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay", default=0.01, type=float, help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--weight-decay", default=0.0, type=float, help="Weight decay ratio"
    )
    parser.add_argument(
        "--criterion",
        default="acc",
        type=str,
        choices=["loss", "loss_eps_decay_only", "acc"],
        help="Criterion to perform epsilon decay",
    )
    parser.add_argument(
        "--epochs", "-e", default=30, type=int, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        nargs="?",
        help="Number of epochs to wait without improvement "
        "before stopping the training",
    )
    parser.add_argument(
        "--grad-clip", default=5, type=float, help="Gradient norm threshold to clip"
    )

    # finetuning related
    parser.add_argument(
        "--enc-init",
        default=None,
        type=str,
        help="Pre-trained ASR model to initialize encoder.",
    )
    parser.add_argument(
        "--enc-init-mods",
        default="enc.enc.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of encoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--dec-init",
        default=None,
        type=str,
        help="Pre-trained ASR, MT or LM model to initialize decoder.",
    )
    parser.add_argument(
        "--dec-init-mods",
        default="att., dec.",
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of decoder modules to initialize, separated by a comma.",
    )
    parser.add_argument(
        "--freeze-mods",
        default=None,
        type=lambda s: [str(mod) for mod in s.split(",") if s != ""],
        help="List of modules to freeze, separated by a comma.",
    )

    # transducer related
    parser.add_argument(
        "--right-mask",
        default=-1,
        type=int,
        help="Compute CER on development set",
    )
    parser.add_argument(
        "--etype",
        default="blstmp",
        type=str,
        choices=[
                "transformer",
                "lstm",
                "blstm",
                "lstmp",
                "blstmp",
                "vgglstmp",
                "vggblstmp",
                "vgglstm",
                "vggblstm",
                "gru",
                "bgru",
                "grup",
                "bgrup",
                "vgggrup",
                "vggbgrup",
                "vgggru",
                "vggbgru",
            ],
        help="Type of encoder network architecture",
    )

    parser.add_argument(
        "--dropout-rate",
        default=0.1,
        type=float,
        help="Dropout rate for the encoder",
    )
    # Encoder - RNN
    parser.add_argument(
        "--elayers",
        default=4,
        type=int,
        help="Number of encoder layers (for shared recognition part "
        "in multi-speaker asr mode)",
    )
    parser.add_argument(
        "--eunits",
        "-u",
        default=300,
        type=int,
        help="Number of encoder hidden units",
    )
    parser.add_argument(
        "--eprojs", default=320, type=int, help="Number of encoder projection units"
    )
    parser.add_argument(
        "--subsample",
        default="1",
        type=str,
        help="Subsample input frames x_y_z means subsample every x frame "
        "at 1st layer, every y frame at 2nd layer etc.",
    )
    # Encoder - Transformer
    parser.add_argument(
        "--enc-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Encoder architecture definition by blocks",
    )

    parser.add_argument(
        "--enc-block-repeat",
        default=0,
        type=int,
        help="Repeat N times the provided encoder blocks if N > 1",
    )
    parser.add_argument(
        "--transformer-enc-input-layer",
        type=str,
        default="conv2d",
        choices=["conv2d", "vgg2l", "linear", "embed"],
        help="Transformer encoder input layer type",
    )
    parser.add_argument(
        "--transformer-enc-positional-encoding-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="Transformer encoder positional encoding layer type",
    )
    parser.add_argument(
        "--transformer-enc-self-attn-type",
        type=str,
        default="self_attn",
        choices=["self_attn", "rel_self_attn"],
        help="Transformer encoder self-attention type",
    )
    parser.add_argument(
        "--transformer-enc-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Transformer encoder pointwise activation type",
    )
    parser.add_argument(
        "--transformer-enc-conv-mod-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Transformer encoder convolutional module activation type",
    )
    # Attention - RNN
    parser.add_argument(
        "--adim",
        default=320,
        type=int,
        help="Number of attention transformation dimensions",
    )
    parser.add_argument(
        "--aheads",
        default=4,
        type=int,
        help="Number of heads for multi head attention",
    )
    parser.add_argument(
        "--atype",
        default="location",
        type=str,
        choices=[
                "noatt",
                "dot",
                "add",
                "location",
                "coverage",
                "coverage_location",
                "location2d",
                "location_recurrent",
                "multi_head_dot",
                "multi_head_add",
                "multi_head_loc",
                "multi_head_multi_res_loc",
            ],
        help="Type of attention architecture",
    )
    parser.add_argument(
        "--awin", default=5, type=int, help="Window size for location2d attention"
    )
    parser.add_argument(
        "--aconv-chans",
        default=10,
        type=int,
        help="Number of attention convolution channels "
            "(negative value indicates no location-aware attention)",
    )
    parser.add_argument(
        "--aconv-filts",
        default=100,
        type=int,
        help="Number of attention convolution filters "
            "(negative value indicates no location-aware attention)",
    )
    # Decoder - general
    parser.add_argument(
        "--dec_type",
        default="lstm",
        type=str,
        choices=["lstm", "gru", "transformer"],
        help="Type of decoder to use",
    )
    parser.add_argument(
        "--dropout-rate-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder",
    )
    parser.add_argument(
        "--dropout-rate-embed-decoder",
        default=0.0,
        type=float,
        help="Dropout rate for the decoder embedding layer",
    )
    # Decoder - RNN
    parser.add_argument(
        "--dec-embed-dim",
        default=320,
        type=int,
        help="Number of decoder embeddings dimensions",
    )
    parser.add_argument(
        "--dlayers", default=1, type=int, help="Number of decoder layers"
    )
    parser.add_argument(
        "--dunits", default=320, type=int, help="Number of decoder hidden units"
    )
    # Decoder - Transformer
    parser.add_argument(
        "--dec-block-arch",
        type=eval,
        action="append",
        default=None,
        help="Decoder architecture definition by blocks",
    )
    parser.add_argument(
        "--dec-block-repeat",
        default=1,
        type=int,
        help="Repeat N times the provided decoder blocks if N > 1",
    )
    parser.add_argument(
        "--transformer-dec-input-layer",
        type=str,
        default="embed",
        choices=["linear", "embed"],
        help="Transformer decoder input layer type",
    )
    parser.add_argument(
        "--transformer-dec-pw-activation-type",
        type=str,
        default="relu",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="Transformer decoder pointwise activation type",
    )
    # Transformer - General
    parser.add_argument(
        "--transformer-warmup-steps",
        default=25000,
        type=int,
        help="Optimizer warmup steps",
    )
    parser.add_argument(
        "--transformer-init",
        type=str,
        default="pytorch",
        choices=[
            "pytorch",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            ],
        help="How to initialize transformer parameters",
    )
    parser.add_argument(
        "--transformer-lr",
        default=10.0,
        type=float,
        help="Initial value of learning rate",
    )
    # Transducer
    parser.add_argument(
        "--trans-type",
        default="warp-rnnt",
        type=str,
        choices=["warp-transducer", "warp-rnnt"],
        help="Type of transducer implementation to calculate loss.",
    )
    parser.add_argument(
        "--rnnt-mode",
        default="rnnt",
        type=str,
        choices=["rnnt", "rnnt-att"],
        help="Transducer mode for RNN decoder",
    )
    parser.add_argument(
        "--joint-dim",
        default=320,
        type=int,
        help="Number of dimensions in joint space",
    )
    parser.add_argument(
        "--joint-activation-type",
        type=str,
        default="tanh",
        choices=["relu", "tanh", "swish"],
        help="Joint network activation type",
    )
    parser.add_argument(
        "--score-norm",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize transducer scores by length",
    )
    parser.add_argument(
        "--attype",
        type=str,
        default="transformer",
        choices=["transformer", "tanh", "sigmoid"],
        help="Normalize transducer scores by length",
    )

    return parser
