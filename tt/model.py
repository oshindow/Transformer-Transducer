import yaml
import torch
import torch.nn as nn

from tt.encoder import Encoder
from tt.decoder import Decoder
from tt.mask import target_mask
from tt.loss_utils import prepare_loss_inputs
from tt.initializer import initializer
from tt.loss import TransLoss
from tt.net_utils import make_non_pad_mask
from tt.utils import AttrDict


class Transducer(nn.Module):

    def __init__(self, idim, odim, args, ignore_id=-1, blank_id=0):
        """
        ignore_id for padding,
        blank_id is the start of sequence.
        """
        torch.nn.Module.__init__(self)

        configfile = open(args.config)
        config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
        config = config.model

        self.idim = idim
        self.odim = odim
        self.ignore_id = ignore_id
        self.blank_id = blank_id

        self.encoder = Encoder(
                idim=idim,
                n_layer=config.enc.n_layer,
                n_layer_1=config.enc.n_layer_1,
                n_layer_2=config.enc.n_layer_2,
                n_layer_3=config.enc.n_layer_3,
                n_head=config.enc.n_head,
                d_model=config.enc.d_model,
                d_inner=config.enc.d_inner,
                dropout=config.dropout,
                etype=config.enc.etype
            )
        self.decoder = Decoder(
                odim=odim,
                n_layer=config.dec.n_layer,
                n_head=config.dec.n_head,
                d_model=config.dec.d_model,
                d_inner=config.dec.d_inner,
                d_joint=config.dec.d_joint,
                dropout=config.dropout
            )

        self.default_parameters(args)
        self.beam_size = 1
        self.criterion = TransLoss(self.blank_id)
        self.loss = None
        self.rnnlm = None

    def default_parameters(self, args):
        """Initialize/reset parameters for transducer."""
        initializer(self, args)

    def forward(self, xs_pad, ilens, ys_pad):

        # 1. encoder
        # xs_pad = [batch, max_input_length, idim]
        # ilens = [length] * batch
        # ys_pad = [batch * max_target_length]
        # ys_pad is a tensor padded with -1 with a shape of [batch * max_target_length]
        # src_mask is a tensor with a shape of [batch, 1, max_input_length]
        # src_mask considers the variable length of input batch
        # src_mask is a length mask
        xs_pad = xs_pad[:, : max(ilens)]
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, ilens = self.encoder(xs_pad, ilens, src_mask)
        self.hs_pad = hs_pad

        # 1.5. transducer preparation related
        # ys_in_pad add a 0 symbol at the first of the sequence for <sos>
        ys_in_pad, target, target_len = prepare_loss_inputs(ys_pad)

        # 2. decoder
        # ys_mask is a length mask
        ys_mask = target_mask(ys_in_pad, self.blank_id)
        pred_pad = self.decoder(ys_in_pad, ys_mask, hs_pad)
        self.pred_pad = pred_pad

        # 3. loss computation
        loss = self.criterion(pred_pad, target.int(), ilens.int(), target_len.int())
        self.loss = loss

        return self.loss

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        x = torch.as_tensor(x).unsqueeze(0)
        length = torch.as_tensor(x.size(1))
        enc_output, _ = self.encoder(x, length)

        return enc_output.squeeze(0)

    def recognize(self, x, recog_args=None, char_list=None, rnnlm=None):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            recog_args (namespace): argument Namespace containing options
            char_list (list): list of characters
            rnnlm (torch.nn.Module): language model module

        Returns:
            y (list): n-best decoding results

        """

        h = self.encode_transformer(x)
        params = [h, recog_args]

        if self.beam_size == 1:
            nbest_hyps = self.decoder.recognize(*params)
        else:
            params.append(rnnlm)
            nbest_hyps = self.decoder.recognize_beam(*params)

        return nbest_hyps
