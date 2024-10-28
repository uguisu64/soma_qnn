from data_reader import TokenizedDataReader
import onnx
from onnxruntime.quantization import quantize_static, QuantType

model_path = "onnx/elyzallama3.onnx"
tokenizer_name = "elyza/Llama-3-ELYZA-JP-8B"  # 任意のトークナイザ名
texts = [
    "Hello, how are you?",
    "This is an example sentence for calibration."
]

data_reader = TokenizedDataReader(model_path, tokenizer_name, texts)

model_fp32 = onnx.load_model("onnx/elyzallama3.onnx", load_external_data=True)
model_quant = 'quant_model/elyzallama3.quant.onnx'
quantized_model = quantize_static(model_fp32, model_quant, use_external_data_format=False, calibration_data_reader=data_reader)