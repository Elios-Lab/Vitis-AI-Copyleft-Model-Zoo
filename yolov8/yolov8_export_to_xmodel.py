import torch
from pathlib import Path
from pytorch_nndct.apis import torch_quantizer
from ultralytics import YOLO
from pytorch_nndct.apis import Inspector

model =YOLO('yolov8n.pt').model.eval()      
inspector =Inspector('DPUCZDX8G_ISA1_B4096')    
inputSize =torch.randn(1, 3, 640, 640)         
device  =torch.device('cpu')                    
inspector.inspect(model, (inputSize,), device=device)

# Calibration
quantizer = torch_quantizer('calib', model, inputSize, device=torch.device('cpu'))
quantizer.export_quant_config()

# Quantization and Exporting to Xmodel
output_dir = Path('output_dir')
quantizer = torch_quantizer('test', model, inputSize, device=torch.device('cpu'))
quantizer.export_xmodel(output_dir)
quantizer.export_onnx_model(output_dir)
quantizer.export_torch_script(output_dir)
print('XMODEL is exported to',output_dir,'folder.')

# IMPORTANT NOTE: QAT and Pruning steps were skipped in this example because the focus was on the Power Consumption and Inference Time of the model.