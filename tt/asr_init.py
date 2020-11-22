from tt.asr_utils import get_model_conf
from tt.asr_utils import torch_load
import os
import logging
from tt.dynamic_import import dynamic_import


def load_trained_model(model_path):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

    logging.warning("reading model parameters from " + model_path)

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "tt.model:Transducer"
    model_class = dynamic_import(model_module)
    model = model_class(idim, odim, train_args)

    torch_load(model_path, model)
