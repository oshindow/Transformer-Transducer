import six
import torch
import torch.nn as nn
from tt.transformer import DecoderLayer
from tt.repeat import repeat
from tt.positionwise_feed_forward import PositionwiseFeedForward
from tt.embedding import PositionalEncoding
from tt.decoder_layer import DecoderLayer
from tt.attention import MultiHeadedAttention
from tt.mask import subsequent_mask
from tt.layer_norm import LayerNorm


class Decoder(nn.Module):

    def __init__(self, odim, n_layer, n_head, d_model, d_inner, d_joint, dropout,
                 normalize_before=True,
                 concat_after=False
                 ):
        super(Decoder, self).__init__()

        # one hot
        # self.embedding = nn.Embedding.from_pretrained(self.one_hot(vocab_size, d_model), padding_idx=0)
        # self.embedding = self.one_hot(vocab_size, d_model)

        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(odim, d_model),
            PositionalEncoding(d_model, dropout),
        )
        self.decoders = repeat(
            n_layer,
            lambda lnum: DecoderLayer(
                d_model,
                MultiHeadedAttention(n_head, d_model, dropout),
                PositionwiseFeedForward(d_model, d_inner, dropout),
                dropout,
                normalize_before,
                concat_after,
            ),
        )

        self.lin_enc = torch.nn.Linear(d_model, d_joint)
        self.lin_dec = torch.nn.Linear(d_model, d_joint, bias=False)
        self.lin_out = torch.nn.Linear(d_joint, odim)
        self.after_norm = LayerNorm(d_model)
        self.blank = 0

    def one_hot(self, vocab_size, d_model):

        idx_list = [vocab_size - 1]

        for x in range(0, vocab_size - 1):
            idx_list.append(x)

        embedding = torch.eye(vocab_size, d_model)[idx_list].cuda()

        return embedding

    def pretrain(self):

        embedding = []
        with open('thchs30_train_char_embedding.txt', 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split()
                emb = [float(i) for i in parts[1:]]
                embedding.append(emb)

        embedding = torch.tensor(embedding).cuda()

        return embedding

    def joint(self, h_enc, h_dec):
        """Joint computation of z.

        Args:
            h_enc (torch.Tensor):
                batch of expanded hidden state (batch, maxlen_in, 1, Henc)
            h_dec (torch.Tensor):
                batch of expanded hidden state (batch, 1, maxlen_out, Hdec)

        Returns:
            z (torch.Tensor): output (batch, maxlen_in, maxlen_out, odim)

        """
        z = torch.tanh(self.lin_enc(h_enc) + self.lin_dec(h_dec))
        z = self.lin_out(z)

        return z

    def forward(self, targets_in_pad, targets_mask, enc_state):

        # one-hot
        # embed_inputs = self.embedding[inputs]

        # nn.Embedding
        embed_inputs = self.embed(targets_in_pad)

        embed_inputs, _ = self.decoders(embed_inputs, targets_mask)

        h_enc = enc_state.unsqueeze(2)
        h_dec = embed_inputs.unsqueeze(1)

        z = self.joint(h_enc, h_dec)

        return z

    def forward_one_step(self, tgt, tgt_mask, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): input token ids, int64 (batch, maxlen_out)
                                if input_layer == "embed"
                                input tensor (batch, maxlen_out, #mels)
                                in the other cases
            tgt_mask (torch.Tensor): input token mask,  (batch, Tmax)
                                     dtype=torch.uint8 in PyTorch 1.2-
                                     dtype=torch.bool in PyTorch 1.2+ (include 1.2)

        """
        tgt = self.embed(tgt)

        if cache is None:
            cache = self.init_state()
        new_cache = []

        for c, decoder in zip(cache, self.decoders):
            tgt, tgt_mask = decoder(tgt, tgt_mask, c)
            new_cache.append(tgt)

        tgt = self.after_norm(tgt[:, -1])

        return tgt, new_cache

    def init_state(self, x=None):
        """Get an initial state for decoding."""
        return [None for i in range(len(self.decoders))]

    def recognize(self, h, recog_args):
        """Greedy search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options

        Returns:
            hyp (list of dicts): 1-best decoding results

        """
        hyp = {"score": 0.0, "yseq": [self.blank]}

        ys = torch.tensor(hyp["yseq"], dtype=torch.long).unsqueeze(0)
        ys_mask = subsequent_mask(1).unsqueeze(0)
        y, c = self.forward_one_step(ys, ys_mask, None)

        for i, hi in enumerate(h):
            ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)
            logp, pred = torch.max(ytu, dim=0)

            if pred != self.blank:
                hyp["yseq"].append(int(pred))
                hyp["score"] += float(logp)

                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                ys_mask = subsequent_mask(len(hyp["yseq"])).unsqueeze(0)

                y, c = self.forward_one_step(ys, ys_mask, c)

        return [hyp]

    def recognize_beam(self, h, recog_args, rnnlm=None):
        """Beam search implementation for transformer-transducer.

        Args:
            h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
            recog_args (Namespace): argument Namespace containing options
            rnnlm (torch.nn.Module): language model module

        Returns:
            nbest_hyps (list of dicts): n-best decoding results

        """
        beam = recog_args.beam_size
        k_range = min(beam, self.odim)
        nbest = recog_args.nbest
        normscore = recog_args.score_norm_transducer

        if rnnlm:
            kept_hyps = [
                {"score": 0.0, "yseq": [self.blank], "cache": None, "lm_state": None}
            ]
        else:
            kept_hyps = [{"score": 0.0, "yseq": [self.blank], "cache": None}]

        for i, hi in enumerate(h):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                new_hyp = max(hyps, key=lambda x: x["score"])
                hyps.remove(new_hyp)

                ys = torch.tensor(new_hyp["yseq"]).unsqueeze(0)
                ys_mask = subsequent_mask(len(new_hyp["yseq"])).unsqueeze(0)
                y, c = self.forward_one_step(ys, ys_mask, new_hyp["cache"])

                ytu = torch.log_softmax(self.joint(hi, y[0]), dim=0)

                if rnnlm:
                    rnnlm_state, rnnlm_scores = rnnlm.predict(
                        new_hyp["lm_state"], ys[:, -1]
                    )

                for k in six.moves.range(self.odim):
                    beam_hyp = {
                        "score": new_hyp["score"] + float(ytu[k]),
                        "yseq": new_hyp["yseq"][:],
                        "cache": new_hyp["cache"],
                    }

                    if rnnlm:
                        beam_hyp["lm_state"] = new_hyp["lm_state"]

                    if k == self.blank:
                        kept_hyps.append(beam_hyp)
                    else:
                        beam_hyp["yseq"].append(int(k))
                        beam_hyp["cache"] = c

                        if rnnlm:
                            beam_hyp["lm_state"] = rnnlm_state
                            beam_hyp["score"] += (
                                recog_args.lm_weight * rnnlm_scores[0][k]
                            )

                        hyps.append(beam_hyp)

                if len(kept_hyps) >= k_range:
                    break

        if normscore:
            nbest_hyps = sorted(
                kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
            )[:nbest]
        else:
            nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[
                :nbest
            ]

        return nbest_hyps


