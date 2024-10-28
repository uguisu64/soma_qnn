import data_reader
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from onnx.external_data_helper import load_external_data_for_model
from transformers import AutoTokenizer

TEXT = [
    "Hello, how are you?",
    "This is an example sentence for calibration."
]

if __name__ == "__main__":
    input_model_path = "onnx/phi3.5mini.onnx"  # TODO: Replace with your actual model
    external_data_path = "onnx/phi3.5mini.onnx.data"
    output_model_path = "phi3.5mini.qdq.onnx"  # Name of final quantized model
    model_checkpoint = "microsoft/Phi-3.5-mini-instruct"
    my_data_reader = data_reader.TokenizedDataReader(input_model_path, model_checkpoint, TEXT)

    # Pre-process the original float32 model.
    preproc_model_path = "model.preproc.onnx"
    model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
    model_to_quantize = preproc_model_path if model_changed else input_model_path

    # Generate a suitable quantization configuration for this model.
    # Note that we're choosing to use uint16 activations and uint8 weights.
    qnn_config = get_qnn_qdq_config(model_to_quantize,
                                    my_data_reader,
                                    activation_type=QuantType.QUInt16,  # uint16 activations
                                    weight_type=QuantType.QUInt8)       # uint8 weights

    # Quantize the model.
    quantize(model_to_quantize, output_model_path, qnn_config)