import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
text = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"

model_checkpoint = "microsoft/Phi-3.5-mini-instruct"
save_directory = "onnx/"

# モデルとトークナイザを読み込み
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float16)
model.eval()
print(model)
print(model.forward.__annotations__)

# 保存ディレクトリを確認・作成
os.makedirs(save_directory, exist_ok=True)

# ダミー入力データ（バッチサイズ1、トークン数8）を準備
messages = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
    {"role": "user", "content": text},
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
token_ids = tokenizer.encode(
    prompt, add_special_tokens=False, return_tensors="pt"
)
dummy_input = token_ids

inputs = {'input_ids':      torch.ones(1,128, dtype=torch.int64),
            'attention_mask': torch.ones(1,128, dtype=torch.int64)}
outputs = model(**inputs)
symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}

# ONNX形式でエクスポート
torch.onnx.export(
    model,
    (inputs['input_ids'], inputs['attention_mask']),
    save_directory + "phi3.5mini.onnx",
    opset_version=17,  # ONNXのopsetバージョン
    do_constant_folding=True,
    input_names=['input_ids', 'input_mask'],
    output_names=['output'],
    dynamic_axes={'input_ids': symbolic_names, 'input_mask' : symbolic_names, 'output' : symbolic_names}
)
