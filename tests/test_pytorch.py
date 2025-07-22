import torch
# Check if PyTorch is installed
print(f"Torch Version: {torch.__version__}")
# Check if CUDA is available
print(f"CUDA Available: {torch.cuda.is_available()}")
# Check the CUDA version
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("CUDA is not available. Please check your installation.")
# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU available. Please check your GPU installation.")
# Check if cuDNN is available
if torch.backends.cudnn.is_available():
    print("cuDNN is available.")
else:
    print("cuDNN is not available. Please check your cuDNN installation.")