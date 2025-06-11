import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import (ToTensor, Resize, Compose, InterpolationMode)
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_nndct.apis import torch_quantizer
from ultralytics import YOLO
from pytorch_nndct.apis import Inspector

# Path to coco validation dataset
data_dir = Path("Path/to/coco/images/val2017")  
transform = Compose([Resize((640, 640), InterpolationMode.BILINEAR), ToTensor()])

# Calibration dataset
calib_set = ImageFolder(data_dir.parent, transform=transform)  
calib_loader = DataLoader(calib_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
model = YOLO('yolov8n.pt').model 

# Check if all layers are supported on the target DPU
inspector = Inspector('DPUCZDX8G_ISA1_B4096')    
inputSize = torch.randn(1, 3, 640, 640)         
device  = torch.device('cpu')                    
inspector.inspect(model, (inputSize,), device=device)

# Apply the calibration on some images to get the bias error file
quantizer = torch_quantizer('calib', model, inputSize, device=torch.device('cpu'))
qmodel = quantizer.quant_model
with torch.no_grad():
	for i, (img, _) in enumerate(calib_loader):
		qmodel(img)
		if i == 3:
			break

quantizer.export_quant_config()

# Quantization and Exporting to Xmodel
output_dir = Path('Path/to/output_dir')
quantizer = torch_quantizer('test', model, inputSize, device=torch.device('cpu'))
quantizer.export_xmodel(output_dir)
quantizer.export_onnx_model(output_dir)
quantizer.export_torch_script(output_dir)
print('XMODEL is exported to',output_dir,'folder.')

# IMPORTANT NOTE: QAT and Pruning steps were skipped in this example because the focus was on the Power Consumption and Inference Time of the model.
