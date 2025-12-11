# LeNet-5 MNIST Inference (C + PyTorch)

**Lab 4 – Image and Video Processing**  
**Handwritten Digit Recognition using LeNet-5**  

Accuracy on 10 test images: **100.00%**

## Project Structure
```
.
├── main_lenet_5.c          → C implementation of LeNet-5 inference (from scratch)
├── tensor.h                → Model constants (LABEL_LEN = 10 for demo)
├── lenet5.py               → Original PyTorch training script (Lab 2)
├── weights/                → 10 weight files exported from PyTorch model
│   ├── 1. convolution 1 (w_conv1.txt)
│   ├── 2. bias của convolution 1 (b_conv1.txt)
│   ├── ...
│   └── 10. bias của fully connection 3 (b_fc3.txt)
├── mnist-test-image.txt    → 10 test images (28×28 normalized)
├── mnist-test-target.txt   → Ground truth labels for 10 images
├── lenet.exe               → Executable (Windows)
└── README.md               → This file
```

## Results (Exactly as required)
```
Predicted label: 4
Prediction: 1/1
Predicted label: 2
Prediction: 2/2
Predicted label: 0
Prediction: 3/3
Predicted label: 6
Prediction: 4/4
Predicted label: 3
Prediction: 5/5
Predicted label: 8
Prediction: 6/6
Predicted label: 7
Prediction: 7/7
Predicted label: 8
Prediction: 8/8
Predicted label: 3
Prediction: 9/9
Predicted label: 5
Prediction: 10/10
Accuracy = 1.000000
```

## How to Run

### On Windows (already compiled)
```cmd
lenet.exe
```

Compile from source (MinGW/GCC)
```gcc main_lenet_5.c -o lenet.exe -lm
lenet.exe
```
On Linux/macOS
```gcc main_lenet_5.c -o lenet -lm
./lenet
```
Model Architecture (LeNet-5)

Input: 28×28 grayscale

Conv1: 1×1 kernel → 6 channels → ReLU → AvgPool 2×2
Conv2: 5×5 kernel → 16 channels → ReLU → AvgPool 2×2

Flatten → 400 neurons

FC1: 400 → 120 → ReLU
FC2: 120 → 84 → ReLU
FC3: 84 → 10 → LogSoftmax

End here.
