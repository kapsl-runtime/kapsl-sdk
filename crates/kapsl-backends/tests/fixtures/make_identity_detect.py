"""Regenerate identity_detect.onnx: Identity, pred_in[1,anchors,7] -> pred[1,anchors,7].

7 channels = 4 box (cx, cy, w, h) + 1 objectness + 2 classes, matching the
`metadata.detect` config used by onnx_detect_e2e.rs. Same shape of fixture as
identity_logits.onnx / identity_embed.onnx (ir_version 9, opset 13).
Requires the `onnx` Python package: python make_identity_detect.py
"""
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("pred_in", TensorProto.FLOAT, [1, "anchors", 7])
y = helper.make_tensor_value_info("pred", TensorProto.FLOAT, [1, "anchors", 7])
graph = helper.make_graph([helper.make_node("Identity", ["pred_in"], ["pred"])], "identity_detect", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, "identity_detect.onnx")
