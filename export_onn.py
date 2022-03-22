import torch
import torch.onnx #Export to ONNX
from config import MODEL_DICT_FILE
from model import create_model
# set the computation device
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=5).to(device)
model.load_state_dict(torch.load(
     MODEL_DICT_FILE, map_location=device
))
model.eval().to(device)

dummpy_input = torch.randn(1, 3, 640, 480).cuda()
dummpy_input = dummpy_input.to(device)
#model.eval()
#model(dummpy_input)
model.eval().to(device)
# Export the model
input_names = ['input_0']
output_names = ['scores', 'boxes']

torch.onnx.export(model, dummpy_input, "model.onnx", verbose=True, opset_version=11, input_names=input_names, output_names=output_names)

#torch.onnx.export(model,               # model being run
#                  dummpy_input,                         # model input (or a tuple for multiple inputs)
#                  "model.onnx",   # where to save the model (can be a file or file-like object)
#                  export_params=True,        # store the trained parameter weights inside the model file
#                  opset_version=11,          # the ONNX version to export the model to
#                  do_constant_folding=True,  # whether to execute constant folding for optimization
#                  input_names = ['input'],   # the model's input names
#                  output_names = ['output'],) # the model's output names)
print("=======================")
print("Exported model to ONNX ")
print("=======================")