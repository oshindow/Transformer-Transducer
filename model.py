#!/usr/bin/env python3
# encoding: utf-8

import torch

from tt.utils.net_utils import get_subsample, make_non_pad_mask
from tt.rnn.encoders import RNNEncoder
from tt.transducer.loss import TransLoss
from tt.transducer.rnn_decoder import RNNDecoder
from tt.transducer.transformer_decoder import TDecoder
from tt.transducer.transformer_encoder import TEncoder
from tt.transducer.utils import prepare_loss_inputs
from tt.transformer.mask import target_mask
from pytorch_lightning import LightningModule
from dataclasses import asdict

from tt.beam_search_transducer import BeamSearchTransducer


class Transducer(LightningModule):
    def __init__(self, idim, odim, args, blank_id=0, training=True):
        super(Transducer, self).__init__()

        if 'lstm' in args.etype:
            self.subsample = get_subsample(args, mode="asr", arch="rnn-t")
            self.Encoder = RNNEncoder(
                args.etype,
                args.elayers,
                args.eunits,
                args.eprojs,
                idim,
                self.subsample,
                args.dropout_rate
            )
        else:
            self.Encoder = TEncoder(
                idim,
                args.enc_block_arch,
                input_layer=args.transformer_enc_input_layer,
                repeat_block=args.enc_block_repeat,
                self_attn_type=args.transformer_enc_self_attn_type,
                positional_encoding_type=args.transformer_enc_positional_encoding_type,
                positionwise_activation_type=args.transformer_enc_pw_activation_type,
                conv_mod_activation_type=args.transformer_enc_conv_mod_activation_type,
            )
            encoder_out = self.Encoder.enc_out

        if 'lstm' in args.dec_type:
            self.Decoder = RNNDecoder(
                args.eprojs,
                odim,
                args.dec_type,
                args.dlayers,
                args.dunits,
                blank_id,
                args.dec_embed_dim,
                args.joint_dim,
                args.joint_activation_type,
                args.dropout_rate_decoder,
                args.dropout_rate_embed_decoder,
            )
        else:
            self.Decoder = TDecoder(
                odim,
                encoder_out,
                args.joint_dim,
                args.dec_block_arch,
                input_layer=args.transformer_dec_input_layer,
                repeat_block=args.dec_block_repeat,
                joint_activation_type=args.joint_activation_type,
                positionwise_activation_type=args.transformer_dec_pw_activation_type,
                dropout_rate_embed=args.dropout_rate_embed_decoder,
            )

        self.etype = args.etype
        self.dec_type = args.dec_type

        self.blank_id = blank_id
        self.blank = args.sym_blank

        if training:
            self.criterion = TransLoss(args.trans_type, self.blank_id)

        self.error_calculator = None
        self.rnnlm = None
        self.args = args
        self.idim = idim
        self.odim = odim

        self.beamsearch = BeamSearchTransducer(
            decoder=self.Decoder,
            beam_size=args.beam_size,
            lm=self.rnnlm,
            lm_weight=args.lm_weight,
            search_type="default",
            score_norm=args.score_norm,
        )



    def forward(self, xs_pad, ilens, ys_pad):
        xs_pad = xs_pad[:, : max(ilens)]
        masks = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if self.args.right_mask > -1:
            masks = target_mask(masks, self.blank_id, self.args.right_mask)

        # Encoder
        if "lstm" in self.etype:
            hs_pad, hs_mask, _ = self.Encoder(xs_pad, ilens)
        else:
            hs_pad, hs_mask = self.Encoder(xs_pad, masks)

        # Loss input
        ys_in_pad, target, pred_len, target_len = prepare_loss_inputs(ys_pad, hs_mask)

        # Decoder
        if "lstm" in self.dec_type:
            pred_pad = self.Decoder(hs_pad, ys_in_pad)
        else:
            ys_mask = target_mask(ys_in_pad, self.blank_id)
            pred_pad, _ = self.Decoder(ys_in_pad, ys_mask, hs_pad)

        return pred_pad, target, pred_len, target_len

    def configure_optimizers(self):
        if self.args.opt == "adadelta":
            optimizer = torch.optim.Adadelta(
            self.parameters(), rho=0.95, eps=self.args.eps, weight_decay=self.args.weight_decay
            )
        elif self.args.opt == "adam":
            optimizer = torch.optim.Adam(self.parameters(), weight_decay=self.args.weight_decay)
        elif self.args.opt == "noam":
            from tt.noam import get_std_opt
            # For transformer-transducer, adim declaration is within the block definition.
            # Thus, we need retrieve the most dominant value (d_hidden) for Noam scheduler.
            adim = self.args.adim

            optimizer = get_std_opt(
                self.parameters(), adim, self.args.transformer_warmup_steps, self.args.transformer_lr
            )
        return optimizer

    def training_step(self, batch, batch_idx):
        xs_pad, ilens, ys_pad = batch
        pred_pad, target, pred_len, target_len = self.forward(xs_pad, ilens, ys_pad)
        loss = self.criterion(pred_pad, target, pred_len, target_len)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        xs_pad, ilens, ys_pad = batch
        pred_pad, target, pred_len, target_len = self.forward(xs_pad, ilens, ys_pad)
        val_loss = self.criterion(pred_pad, target, pred_len, target_len)
        return self.log('val_loss', val_loss)

    def encode_transformer(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        x = torch.as_tensor(x).unsqueeze(0)

        ret = torch.ones(x.size(1), x.size(1), dtype=torch.uint8)
        mask = torch.tril(ret, diagonal=self.args.right_mask).unsqueeze(0)
        enc_output, _ = self.Encoder(x, mask)

        return enc_output.squeeze(0)

    def encode_rnn(self, x):
        """Encode acoustic features.

        Args:
            x (ndarray): input acoustic feature (T, D)

        Returns:
            x (torch.Tensor): encoded features (T, attention_dim)

        """
        self.eval()

        ilens = [x.shape[0]]

        x = x[:: self.subsample[0], :]
        p = next(self.parameters())
        h = torch.as_tensor(x, device=p.device, dtype=p.dtype)

        hs = h.contiguous().unsqueeze(0)

        hs, _, _ = self.Encoder(hs, ilens)

        return hs.squeeze(0)

    def recog(self, x):
        """Recognize input features.

        Args:
            x (ndarray): input acoustic feature (T, D)
            beam_search (class): beam search class

        Returns:
            nbest_hyps (list): n-best decoding results
        """

        if "transformer" in self.etype:
            h = self.encode_transformer(x)
        else:
            h = self.encode_rnn(x)

        nbest_hyps = self.beamsearch(h)

        if isinstance(nbest_hyps, list):
            return [asdict(n) for n in nbest_hyps]
        else:
            return asdict(nbest_hyps)


