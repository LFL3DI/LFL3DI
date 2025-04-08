# LFL3DI

##  How to Custom Train the YOLOv8s model

### Load the model

model = YOLO('yolov8s.pt')

### Training the custom model

path = provide absolute path to the data.yaml file. This file is present in the Lidar-6 folder. This folder contains the custom taining, test and validation data

results = model.train(
   data= path,
   epochs=5,
   name='yolov8s_custom')


Results saved to runs\detect\yolov8s_custom

---

### Prepare the model to run on the AI Hat+
> **Note**: This section is to be performed on a host computer with Python 3.10, which doesn't necessarily have to be the Raspberry Pi.

Download required files from https://hailo.ai/developer-zone/software-downloads/. You will require an account to access these downloads.
Download the .whl files for the `Dataflow Compiler, x86, Linux, Python 3.10` and `Model Zoo, x86, Linux, Python 3.10` (Python 3.10 versions are the latest at the time of writing).

Prepare a Python 3.10 virtual environment. The following commands install Python 3.10, but you can run only the final command if you already have it installed.

```
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-venv python3.10-dev
python3.10 -m venv venv310
```
Activate the virtual environment

`source venv310/bin/activate`

1. **Install the Dataflow Compiler**. Filename may not be exact:

   `pip install hailo_dataflow_compiler-3.31.0-py3-none-linux_x86_64.whl`

   - If you run into the "Building wheel for pygraphviz (pyproject.toml) did not run successfully" issue, the Graphviz development libraries are not installed on your system. Install them with the following command:

      `sudo apt-get install graphviz graphviz-dev`

2. **Clone Remote Model Zoo Git Repository**.

   `git clone https://github.com/LJ-Hao/hailo_model_zoo.git`

3. **Install Model Zoo**. Enter hailo_model_zoo directory and install the package.

   `cd hailo_model_zoo && pip install -e .`
   
   - Verify that the installation was successful by running `hailomz info mobilenet_v1`

4. **Convert `.pt` model to `.onnx`**.
   
   `yolo export model=/path/to/best.pt imgsz=640 format=onnx opset=11`
   
   - If you encounter issues related to `No module named 'onnx'`, make sure that you have the correct venv activated (the 3.10 venv created for this section) and that your regular non-venv environment does not have any packages installed.
   - The `best.onnx` file will be saved to the same directory that the `best.pt` file was in.

5. **Convert `.onnx` model to `.hef`**. This model will be compatible with the AI Hat+.

   `hailomz parse --hw-arch hailo8 --ckpt /path/tobest.onnx yolov8s`

---

### Running the custom model

path = provide absolute path to the best.pt present in the yolov8s_custom/weights/best.pt

model = YOLO(path)


### Run the program
source ~/yolov5-venv/bin/activate
python main.py



