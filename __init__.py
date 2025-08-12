import onnxruntime as ort

def _patch_ort_providers():
    try:
        sess_opts = ort.SessionOptions()

        # Desired providers priority order
        desired_providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        # Show original and intended providers
        print(f"[force_ort_cuda] Providers: {ort.get_available_providers()} default= {desired_providers}")

        # Override default execution providers
        ort.set_default_logger_severity(3)  # 0 = verbose, 3 = warning
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort.set_default_providers(desired_providers)

        print(f"[OK] Patched: \"{__file__}\"")
    except Exception as e:
        print(f"[force_ort_cuda] ERROR applying provider patch: {e}")

# Apply patch immediately when the module loads
_patch_ort_providers()
