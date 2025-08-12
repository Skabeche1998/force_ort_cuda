It’s already patched for ONNX Runtime 1.22+

It uses the monkey-patch InferenceSession method instead of set_default_providers

It defaults to TensorRT → CUDA → CPU providers automatically

It’s “Manager safe” (loads without throwing import errors)

# force_ort_cuda

**Purpose:**  
This ComfyUI custom node forces ONNX Runtime to use GPU acceleration providers (TensorRT, CUDA, CPU) in a specific order — without requiring changes to other nodes.

---

## Key Features
- Compatible with **ONNX Runtime 1.22.x and newer**.
- **No `set_default_providers`** — instead, monkey-patches `onnxruntime.InferenceSession` to enforce provider defaults.
- Defaults to:

["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

- Works with **ComfyUI Manager** — no manual edits needed.
- Safe fallback to CPU if GPU providers are unavailable.

---

## Why the Change?
ONNX Runtime 1.22 removed the `set_default_providers` function.  
Older patches that relied on it caused:


module 'onnxruntime' has no attribute 'set_default_providers'

This repo now uses a **safe, forward-compatible** method.

---

## Installation
**Via ComfyUI Manager:**
1. Open ComfyUI Manager → Install Custom Node
2. Paste repo URL:  


https://github.com/Skabeche1998/force_ort_cuda

3. Restart ComfyUI.

**Manual:**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Skabeche1998/force_ort_cuda.git

Credits
Patch logic updated for ORT 1.22+ by Skabeche

