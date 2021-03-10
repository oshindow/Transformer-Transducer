#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parameter initialization for transducer RNN/Transformer parts."""

import six

# from tt.initialization import lecun_normal_init_parameters
# from tt.initialization import set_forget_bias_to_one

from tt.transformer_initializer import initialize


def initializer(model, args):
    """Initialize transducer model.

    Args:
        model (torch.nn.Module): transducer instance
        args (Namespace): argument Namespace containing options

    """
    initialize(model)
