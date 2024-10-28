import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = onnx.load_model("onnx/elyzallama3.onnx", load_external_data=True)
model_quant = 'quant_model/elyzallama3.quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, use_external_data_format=True)