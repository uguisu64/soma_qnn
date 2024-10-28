import onnx

model = onnx.load_model("quant_model/elyzallama3.quant.onnx", load_external_data=True)
print("====== Nodes ======")
for i, node in enumerate(model.graph.node):
    print("[Node #{}]".format(i))
    print(node)

# モデルの入力データ一覧を出力する
print("====== Inputs ======")
for i, input in enumerate(model.graph.input):
    print("[Input #{}]".format(i))
    print(input)

# モデルの出力データ一覧を出力する
print("====== Outputs ======")
for i, output in enumerate(model.graph.output):
    print("[Output #{}]".format(i))
    print(output)