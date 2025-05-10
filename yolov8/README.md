## YOLOv8 

### Changes
- Support for Yolov8 to convert into xmodel

### Prepare

#### Prepare the environment

##### For vitis-ai docker user
```bash
conda activate vitis-ai-pytorch
pip install ultralytics==8.3.128
```

### Important Note : For yolov8 QAT & Pruning were skipped since the goal was to computer power and inference time of the model so if you need these steps, you can extend the calibration step to support it

#### YOLOv8n model installation : [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)

#### Convert the .pt model into .xmodel
```bash
cd yolov8/
python ./yolov8_export_to_xmodel.py --nndct_convert_sigmoid_to_hsigmoid --nndct_convert_silu_to_hswish
```

#### Run the compilation with vai_c_xir
```bash
cd yolov8/
vai_c_xir -x Path/to/Xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/PATH_TO_TARGET_DEVICE/arch.json -o PATH/TO/OUTPUT_FILE -n yolov8n
```

