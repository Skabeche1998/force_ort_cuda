import onnxruntime as ort
import sys

def _patch_force_ort_cuda():
    try:
        # Preferred ONNX Runtime providers
        preferred_providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        # Log available providers before patch
        available = ort.get_available_providers()
        print(f"[force_ort_cuda] Providers: {available} default= {ort.get_available_providers()}")

        # Backup the original InferenceSession.__init__ method
        if not hasattr(ort.InferenceSession, "__init__original__"):
            ort.InferenceSession.__init__original__ = ort.InferenceSession.__init__

        # Patched init to force our preferred providers
        def _patched_init(self, path_or_bytes, sess_options=None, providers=None, provider_options=None):
            return self.__init__original__(
                path_or_bytes,
                sess_options=sess_options,
                providers=preferred_providers,
                provider_options=provider_options
            )

        ort.InferenceSession.__init__ = _patched_init

        print(f"[OK] Patched: \"{__file__}\"")
        print(f"ORT: {ort.__version__} | Providers: {preferred_providers}")

    except Exception as e:
        print(f"[force_ort_cuda] ERROR patching ORT providers: {e}", file=sys.stderr)

# Execute the patch at import time
_patch_force_ort_cuda()
