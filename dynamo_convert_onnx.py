import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
text = "仕事の熱意を取り戻すためのアイデアを5つ挙げてください。"

model_checkpoint = "elyza/Llama-3-ELYZA-JP-8B"
save_directory = "onnx"

# モデルとトークナイザを読み込み
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, torch_dtype=torch.float16)

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
kwargs = {"max_new_tokens": 1200,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9}

# ONNX形式でエクスポート
onnx_model = torch.onnx.dynamo_export(
    model,
    *token_ids,
    *kwargs
)

onnx_model.save("onnx/elyzallama.onnx")