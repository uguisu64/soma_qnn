import numpy as np
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader


class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()
        print(inputs)

        self.data_list = []

        # Generate 10 random float32 inputs
        # TODO: Load valid calibration input data for your model
        for _ in range(10):
            input_data = {inp.name : np.random.random(inp.shape).astype(np.int64) for inp in inputs}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

from transformers import AutoTokenizer  # Hugging Faceのトークン化ライブラリを使用

class TokenizedDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, tokenizer_name: str, texts: list):
        self.enum_data = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # モデルの入力形状を取得するためのセッション作成
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()
        self.input_names = [inp.name for inp in inputs]

        # 入力テキストをトークン化してデータリストに変換
        self.data_list = []
        for text in texts:
            # トークン化し、トークンIDを取得
            tokenized_inputs = self.tokenizer(text, return_tensors="np", padding="max_length", truncation=True)
            print(tokenized_inputs)
            input_data = {}
            # モデルの期待する形状に合わせてデータを作成
            for input_name in self.input_names:
                if input_name in tokenized_inputs:
                    input_data[input_name] = tokenized_inputs[input_name].astype(np.int64)
                else:
                    # オプションで、必要に応じてダミーのデータを生成（例：全て1のattention_maskなど）
                    input_data[input_name] = np.ones_like(tokenized_inputs["input_ids"]).astype(np.int64)
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.data_list)
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
