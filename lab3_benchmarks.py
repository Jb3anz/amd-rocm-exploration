import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import time
import gc
import os
import warnings
from pathlib import Path
import tarfile
import subprocess

warnings.filterwarnings("ignore")
gc.collect()

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"HIP Version: {torch.version.hip}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (2**30):.1f} GB")

device = torch.device('cuda')

# --- Exercise 1 Helper: Compile and Run C++ rocBLAS GEMM ---
print("\n--- Running rocBLAS GEMM via C++ ---")
comp = subprocess.run(['hipcc', '-O3', '-o', 'rocblas_gemm', 'rocblas_gemm.cpp', '-lrocblas'], capture_output=True, text=True)
if comp.returncode != 0:
    print(f"Compile error:\n{comp.stderr}")
else:
    result = subprocess.run(['./rocblas_gemm'], capture_output=True, text=True)
    print(result.stdout)


# --- Exercise 2: MIOpen Convolution Autotune Test ---
print("\n--- Running MIOpen Convolution Autotune Test ---")
x = torch.randn(32, 64, 224, 224, device=device)
w = torch.randn(128, 64, 3, 3, device=device)

torch.cuda.synchronize()
t0 = time.perf_counter()
y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
first_run = (time.perf_counter() - t0) * 1000

times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = F.conv2d(x, w, padding=1)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)

avg_cached = sum(times) / len(times)
print(f"MIOpen First Run (Autotune): {first_run:.1f} ms")
print(f"MIOpen Cached Run (Average): {avg_cached:.1f} ms")


# --- Exercise 3: rocFFT Round-Trip Test ---
print("\n--- Running rocFFT Round-Trip Test ---")
N = 8192
signal = torch.randn(N, dtype=torch.float32, device=device)
spectrum = torch.fft.fft(signal)
recovered = torch.fft.ifft(spectrum).real
error = torch.max(torch.abs(signal - recovered)).item()
print(f"rocFFT Max Error: {error:.2e} ({'PASS' if error < 1e-5 else 'FAIL'})")


# --- Exercise 4: ResNet-18 Training Loop on CIFAR-10 ---
print("\n--- Running ResNet-18 Training Loop ---")
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

DATA_ROOT = Path("/opt/project/data")
ARCHIVE = DATA_ROOT / "cifar-10-python.tar.gz"
EXTRACTED = DATA_ROOT / "cifar-10-batches-py"

if not EXTRACTED.exists() and ARCHIVE.exists():
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(DATA_ROOT)

trainset = torchvision.datasets.CIFAR10(root=str(DATA_ROOT), train=True, download=False, transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

model = torchvision.models.resnet18(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

for epoch in range(2):
    model.train()
    running_loss = 0.0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}, Batch {i+1}/{len(loader)}, Loss: {running_loss/100:.3f}")
            running_loss = 0.0
            
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Epoch {epoch+1} complete — {elapsed:.1f}s")

print(f"Peak GPU Memory Used: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
