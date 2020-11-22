import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tt.repeat import repeat
from tt.positionwise_feed_forward import PositionwiseFeedForward
from tt.embedding import PositionalEncoding
from tt.encoder_layer import EncoderLayer
from tt.attention import MultiHeadedAttention
from tt.net_utils import make_non_pad_mask


class Encoder(nn.Module):
    def __init__(self, idim,
                 n_layer, n_layer_1, n_layer_2, n_layer_3,
                 n_head, d_model, d_inner, dropout, etype,
                 bidirectional=True,
                 normalize_before=True,
                 concat_after=False,
                 ):

        super(Encoder, self).__init__()

        self.etype = etype
        self.d_model = d_model

        # encoder input layer
        self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, d_model),
                torch.nn.LayerNorm(d_model),
                torch.nn.Dropout(dropout),
                torch.nn.ReLU(),
                PositionalEncoding(d_model, dropout),
            )

        if self.etype == 'hs-conv-transformer':
            # set conv layers
            # nn.Conv2d(input channels, output channels, kernel=(width, height), stride=(width, height))
            # nn.Conv1d(input channels, output channels, kernel=(width, height), stride=(width, height))
            # conv block 1

            self.conv = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(3, 9), stride=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(8, 8, kernel_size=(3, 8), stride=(2, 2)),
                nn.ReLU()
            )

            self.conv_group_2 = nn.Sequential(
                nn.Conv2d(2, 2, kernel_size=1, stride=1),
                nn.ReLU()
            )

            self.conv_group_3 = nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=1, stride=1),
                nn.ReLU()
            )

            self.conv_group_4 = nn.Sequential(
                nn.Conv2d(4, 4, kernel_size=1, stride=1),
                nn.ReLU()
            )

            self.tlayers = repeat(
                n_layer,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )

        elif self.etype == 'conv-transformer':
            # set conv layers
            # nn.Conv2d(input channels, output channels, kernel=(width, height), stride=(width, height))
            # nn.Conv1d(input channels, output channels, kernel=(width, height), stride=(width, height))
            # conv block 1
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 7), stride=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 5), stride=(2, 2)),
                nn.ReLU()
            )

            self.Conv1d_1_3 = nn.Conv1d(16, 320, kernel_size=18, stride=1)
            self.relu = nn.ReLU()

            # conv block 2
            self.conv2 = nn.Sequential(
                nn.Conv1d(320, 512, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(512, 640, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(640, 320, kernel_size=1, stride=1),
                nn.ReLU()
            )
            # conv block 3
            self.conv3 = nn.Sequential(
                nn.Conv1d(320, 512, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Conv1d(512, 640, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Conv1d(640, 320, kernel_size=1, stride=1),
                nn.ReLU()
            )
            self.normalize_before = normalize_before

            self.tlayers_1 = repeat(
                n_layer_1,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )
            self.tlayers_2 = repeat(
                n_layer_2,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )
            self.tlayers_3 = repeat(
                n_layer_3,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )

        elif self.etype == 'channel-transformer':

            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 8), stride=(1, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 7), stride=(2, 2)),
                nn.ReLU()
            )
            self.cd_model = d_model // 16
            self.ctlayers = repeat(
                n_layer,
                lambda lnum: EncoderLayer(
                    self.cd_model,
                    MultiHeadedAttention(n_head, self.cd_model, dropout),
                    PositionwiseFeedForward(self.cd_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )
            self.pe = PositionalEncoding(d_model, dropout)

            self.tlayers = repeat(
                n_layer,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )

        elif self.etype == 'transformer':
            # transformer-encoder
            self.tlayers = repeat(
                n_layer,
                lambda lnum: EncoderLayer(
                    d_model,
                    MultiHeadedAttention(n_head, d_model, dropout),
                    PositionwiseFeedForward(d_model, d_inner, dropout),
                    dropout,
                    normalize_before,
                    concat_after,
                ),
            )

        elif self.etype == 'lstm':
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_inner,
                num_layers=n_layer,
                batch_first=True,
                dropout=dropout if n_layer > 1 else 0,
                bidirectional=bidirectional
            )
            self.output_proj = nn.Linear(2 * d_inner if bidirectional else d_inner, d_model, bias=True)

    def forward(self, inputs, input_lengths=None, mask=None):

        assert inputs.dim() == 3

        if self.etype == 'transformer':
            # input format: batch * length * dim (batch first)
            # core_out format: length * batch * dim (length first)
            inputs = self.embed(inputs)
            outputs = self.tlayers(inputs, mask)[0]

        elif self.etype == 'hs-conv-transformer':
            # compute the sampled inputs_length
            '''
            fig = plt.figure(figsize=(20, 5))
            heatmap = plt.pcolor(inputs[0].T)
            fig.colorbar(mappable=heatmap)
            plt.xlabel('Time(frame)')
            plt.ylabel('Frequency(83-dim)')
            plt.tight_layout()
            plt.savefig('raw-Spectrogram.png')
            '''

            batchsize = inputs.size(0)
            inputs = inputs.unsqueeze(1)

            if input_lengths is not None:
                for i in range(batchsize):
                    if input_lengths.dim() == 0:
                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths, :]
                        core_out = self.conv(inputs_s)
                        input_lengths = torch.as_tensor(core_out.size(2))
                    else:

                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths[i], :]

                        core_out = self.conv(inputs_s)
                        '''
                        for j in range(core_out[0].size(0)):
                            fig = plt.figure(figsize=(20, 5))
                            heatmap = plt.pcolor(core_out[0][j].detach().T)
                            fig.colorbar(mappable=heatmap)
                            plt.xlabel('Time(frame)')
                            plt.ylabel('Frequency(16-dim)')
                            plt.tight_layout()
                            plt.savefig('Spectrogram_8c_'+str(j)+'.png')
                        '''
                        input_lengths[i] = core_out.size(2)

            # block 1
            # the inputs shape of Conv2d is 4-dim of (bsz * c * l * w)
            # the inputs shape of Conv1d is 3-dim of (bsz * c * l)
            # the inputs shape of transformer is 3-dim of (l * bsz * c)
            # core_out format: length * batch * dim (length first)
            inputs = self.conv(inputs)

            # group 1 stay                   4 channel
            # group 2 conv + split           2 channel
            # group 3 concate + conv + split 3 channel
            # group 4 concate + conv         7 channel
            group = [item for item in enumerate(torch.chunk(inputs, 4, 1))]

            group[0] = group[0][1]

            group[1] = self.conv_group_2(group[1][1])
            group[1] = [item[1] for item in enumerate(torch.chunk(group[1], 2, 1))]

            group[2] = torch.cat((group[1][1], group[2][1]), 1)
            group[2] = self.conv_group_3(group[2])
            group[2] = [item[1] for item in enumerate(torch.chunk(group[2], 2, 1))]

            group[3] = torch.cat((group[2][1], group[3][1]), 1)
            group[3] = self.conv_group_4(group[3])

            core_out_12 = torch.cat((group[0], group[1][0]), 1)
            core_out_34 = torch.cat((group[2][0], group[3]), 1)
            core_out = torch.cat((core_out_12, core_out_34), 1)

            for i in range(core_out[0].size(0)):
                fig = plt.figure(figsize=(20, 5))
                heatmap = plt.pcolor(core_out[0][i].detach().T)
                fig.colorbar(mappable=heatmap)
                plt.xlabel('Time(frame)')
                plt.ylabel('Frequency(16-dim)')
                plt.tight_layout()
                plt.savefig('hs-coreout_8c_'+str(i)+'.png')

            # ResNet
            core_out = core_out + inputs

            core_out = core_out.view(core_out.size(0), core_out.size(2), -1)
            fig = plt.figure(figsize=(20, 5))
            heatmap = plt.pcolor(core_out[0].detach().T)
            fig.colorbar(mappable=heatmap)
            plt.xlabel('Time(frame)')
            plt.ylabel('Frequency(256-dim)')
            plt.tight_layout()
            plt.savefig('transformer_input.png')
            if input_lengths.dim() == 0:
                mask = make_non_pad_mask([input_lengths]).unsqueeze(-2)
            else:
                mask = make_non_pad_mask(input_lengths.tolist()).unsqueeze(-2)

            core_out = self.tlayers(core_out, mask)
            fig = plt.figure(figsize=(20, 5))
            heatmap = plt.pcolor(core_out[0][0].detach().T)
            fig.colorbar(mappable=heatmap)
            plt.xlabel('Time(frame)')
            plt.ylabel('Frequency(256-dim)')
            plt.tight_layout()
            plt.savefig('transformer_output.png')
            outputs = core_out[0]

        elif self.etype == "channel-transformer":
            batchsize = inputs.size(0)
            inputs = inputs.unsqueeze(1)

            if input_lengths is not None:
                for i in range(batchsize):
                    if input_lengths.dim() == 0:
                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths, :]
                        core_out = self.conv(inputs_s)
                        input_lengths = torch.as_tensor(core_out.size(2))
                    else:

                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths[i], :]
                        core_out = self.conv(inputs_s)
                        '''
                        for j in range(core_out[0].size(0)):
                            fig = plt.figure(figsize=(20, 5))
                            heatmap = plt.pcolor(core_out[0][j].detach().T)
                            fig.colorbar(mappable=heatmap)
                            plt.xlabel('Time(frame)')
                            plt.ylabel('Frequency(16-dim)')
                            plt.tight_layout()
                            plt.savefig('Spectrogram_8c_'+str(j)+'.png')
                        '''
                        input_lengths[i] = core_out.size(2)

            # block 1
            # the inputs shape of Conv2d is 4-dim of (bsz * c * l * w)
            # the inputs shape of Conv1d is 3-dim of (bsz * c * l)
            # the inputs shape of transformer is 3-dim of (l * bsz * c)
            # conv output format: (bsz * c * t * d)
            inputs = self.conv(inputs)

            # we can get a batch of 16 channels feature maps in all time steps
            # merge 16 channels of one timestep to create one self-attention input (batch, 16, dim)
            inputs = inputs.permute(2, 0, 1, 3)
            merge = torch.zeros(inputs.size(0), batchsize, self.d_model)

            if input_lengths.dim() == 0:
                mask = make_non_pad_mask([input_lengths]).unsqueeze(-2)
            else:
                mask = make_non_pad_mask(input_lengths.tolist()).unsqueeze(-2)

            for t in range(inputs.size(0)):
                merge[t] = self.ctlayers(inputs[t], None)[0].reshape(batchsize, self.d_model)

            merge = merge.permute(1, 0, 2)
            merge = merge + self.pe(merge)
            outputs = self.tlayers(merge, mask)[0]

        elif self.etype == 'conv-transformer':
            # conv-transformer encoder
            # compute the sampled inputs_length
            batchsize = inputs.size(0)
            inputs = inputs.unsqueeze(1)

            block_1_length = torch.zeros(batchsize)
            block_2_length = torch.zeros(batchsize)
            if input_lengths is not None:
                for i in range(batchsize):
                    if input_lengths.dim() == 0:
                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths, :]
                    else:
                        inputs_s = inputs[i].unsqueeze(0)[:, :, 0:input_lengths[i], :]

                    core_out = self.conv1(inputs_s)
                    block_1_length[i] = core_out.size(2)
                    core_out = torch.ones(core_out.size(0), 320, core_out.size(2))

                    core_out = self.conv2(core_out)
                    block_2_length[i] = core_out.size(2)

                    core_out = self.conv3(core_out)
                    if input_lengths.dim() == 0:
                        input_lengths = torch.as_tensor(core_out.size(2))
                    else:
                        input_lengths[i] = core_out.size(2)

            # block 1
            # the inputs shape of Conv2d is 4-dim of (bsz * c * l * w)
            # the inputs shape of Conv1d is 3-dim of (bsz * c * l)
            # the inputs shape of transformer is 3-dim of (l * bsz * c)
            # core_out format: length * batch * dim (length first)
            core_out = self.conv1(inputs)

            concat_0 = self.Conv1d_1_3(core_out[:,:,0,:])
            concat_1 = self.Conv1d_1_3(core_out[:,:,1,:])
            concat = torch.cat((concat_0, concat_1), dim=2)
            for t in range(2, core_out.size(2)):
                next = self.Conv1d_1_3(core_out[:,:,t,:])
                concat = torch.cat((concat, next), dim=2)
            core_out = self.relu(concat)

            block_1_mask = make_non_pad_mask(block_1_length.tolist()).unsqueeze(-2)
            core_out = core_out.permute(0, 2, 1)
            core_out = self.tlayers_1(core_out, block_1_mask)
            core_out = core_out[0].permute(0, 2, 1)

            # block 2
            # the inputs shape of Conv1d is 3-dim of (bsz * c * l)
            # the inputs shape of transformer is 3-dim of (l * bsz * c)
            core_out = self.conv2(core_out)

            block_2_mask = make_non_pad_mask(block_2_length.tolist()).unsqueeze(-2)
            core_out = core_out.permute(0, 2, 1)
            core_out = self.tlayers_2(core_out, block_2_mask)
            core_out = core_out[0].permute(0, 2, 1)

            # block 3
            core_out = self.conv3(core_out)
            if input_lengths.dim() == 0:
                block_3_mask = make_non_pad_mask([input_lengths]).unsqueeze(-2)
            else:
                block_3_mask = make_non_pad_mask(input_lengths.tolist()).unsqueeze(-2)
            core_out = core_out.permute(0, 2, 1)
            core_out = self.tlayers_3(core_out, block_3_mask)
            outputs = core_out[0]

        elif self.etype == 'lstm':
            # core_out format: length * batch * dim (length first)
            if input_lengths is not None:
                sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
                inputs = inputs[indices]
                inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)

            self.lstm.flatten_parameters()
            outputs, hidden = self.lstm(inputs)

            if input_lengths is not None:
                _, desorted_indices = torch.sort(indices, descending=False)
                outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
                outputs = outputs[desorted_indices]

            outputs = self.output_proj(outputs)

        return outputs, input_lengths

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """

        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache

