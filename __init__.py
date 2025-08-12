try:
    import os
    import onnxruntime as ort
    from onnxruntime import InferenceSession as _OrigSession

    # Detect available providers
    prov = ort.get_available_providers()

    # Default TensorRT cache directory
    cache = os.environ.get("TRT_CACHE", r"D:\ComfyUI\trt_engine_cache")

    # TensorRT-specific options
    TRT_OPTS = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache,
        "trt_timing_cache_enable": True,
        "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,  # 2 GB
    }

    # Decide default providers + options
    if "TensorrtExecutionProvider" in prov:
        _default = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        _prov_options = [TRT_OPTS, {}, {}]
    elif "CUDAExecutionProvider" in prov:
        _default = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _prov_options = [{}, {}]
    else:
        _default = ["CPUExecutionProvider"]
        _prov_options = [{}]

    # Monkey-patch InferenceSession to set providers automatically
    class _PatchedSession(_OrigSession):
        def __init__(
            self,
            path_or_bytes,
            sess_options=None,
            providers=None,
            provider_options=None,
            *args, **kwargs
        ):
            if providers is None:
                providers = _default
                provider_options = _prov_options
            super().__init__(
                path_or_bytes,
                sess_options=sess_options,
                providers=providers,
                provider_options=provider_options,
                *args, **kwargs
            )

    ort.InferenceSession = _PatchedSession

    print(f"[force_ort_cuda] Providers: {prov} default= {_default}")

except Exception as e:
    print(f"[force_ort_cuda] ERROR applying provider patch: {e}")

# Needed so ComfyUI Manager recognizes it as a valid node module
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
