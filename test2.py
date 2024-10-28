import onnx
import onnxruntime

model = onnx.load_model("quant_model/elyzallama3.quant.onnx", load_external_data=True)
model = model.SerializeToString()
sess = onnxruntime.InferenceSession(model)