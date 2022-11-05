from winmltools.utils import convert_float_to_float16
from winmltools.utils import load_model, save_model

modelpath = 'best_fall.onnx'

onnx_model = load_model(modelpath)
new_onnx_model = convert_float_to_float16(onnx_model)
save_model(new_onnx_model, modelpath[:-5] + "_fp16.onnx")
