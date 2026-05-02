from pathlib import Path

BASE_URL = (
    "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/"
)
INDEX_DIR = Path.home() / ".triton_docs_mcp_index"
CHROMA_DIR = INDEX_DIR / "chroma_db"
SQLITE_PATH = INDEX_DIR / "triton_docs.db"
COLLECTION_NAME = "triton_docs"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64

CRAWL_MAX_CONCURRENT = 5
CRAWL_DELAY_SECONDS = 0.3
CRAWL_TIMEOUT_SECONDS = 30

GITHUB_SOURCES = {
    "server": {
        "repo": "triton-inference-server/server",
        "branch": "main",
        "description": "Triton Inference Server source — model config, backends, protocol",
        "paths": [
            "README.md",
            "docs/user_guide/model_configuration.md",
            "docs/user_guide/model_repository.md",
            "docs/user_guide/batcher.md",
            "docs/user_guide/scheduler.md",
            "docs/user_guide/rate_limiter.md",
            "docs/user_guide/ensemble_models.md",
            "docs/user_guide/decoupled_models.md",
            "docs/user_guide/response_cache.md",
            "docs/user_guide/metrics.md",
            "docs/user_guide/performance_tuning.md",
            "docs/user_guide/architecture.md",
            "docs/customization_guide/inference_protocols.md",
        ],
    },
    "client": {
        "repo": "triton-inference-server/client",
        "branch": "main",
        "description": "Triton Python client library and examples",
        "paths": [
            "README.md",
            "src/python/library/tritonclient/http/__init__.py",
            "src/python/library/tritonclient/grpc/__init__.py",
            "src/python/library/tritonclient/utils/__init__.py",
        ],
        "extra_glob_patterns": [
            "src/python/examples/*_client.py",
            "src/python/examples/*_perf_client.py",
        ],
    },
    "perf_analyzer": {
        "repo": "triton-inference-server/perf_analyzer",
        "branch": "main",
        "description": "Performance analyzer and GenAI-Perf for benchmarking Triton",
        "paths": [
            "README.md",
            "docs/cli.md",
            "docs/benchmarking.md",
            "docs/quick_start.md",
            "docs/input_data.md",
            "docs/measurements_metrics.md",
            "docs/inference_load_modes.md",
            "genai-perf/README.md",
        ],
    },
    "model_analyzer": {
        "repo": "triton-inference-server/model_analyzer",
        "branch": "main",
        "description": "Model Analyzer for profiling and optimizing Triton model configurations",
        "paths": [
            "README.md",
            "docs/cli.md",
            "docs/config.md",
            "docs/config_search.md",
            "docs/quick_start.md",
            "docs/install.md",
            "docs/metrics.md",
            "docs/model_types.md",
            "docs/launch_modes.md",
        ],
    },
}

GITHUB_RAW_URL = "https://raw.githubusercontent.com"
GITHUB_API_URL = "https://api.github.com/repos"

TOPICS = {
    "architecture": "architecture,scheduler,model execution,concurrent",
    "backends": "TensorRT,PyTorch,ONNX,Python backend,vLLM,TensorRT-LLM,FIL,DALI,custom backend",
    "model_config": "model configuration,config.pbtxt,model repository,model management",
    "deployment": "Docker,Kubernetes,AWS,deployment,scaling,multi-node",
    "performance": "performance,batching,dynamic batching,optimization,benchmarking,perf analyzer,model analyzer",
    "client": "client libraries,HTTP,REST,gRPC,Python client,C API,protocol",
    "llm": "LLM,TensorRT-LLM,vLLM,speculative decoding,constrained decoding,function calling",
    "protocol": "KServe,HTTP/REST,gRPC,protocol,extensions",
}

BACKEND_INFO = {
    "tensorrt_llm": {
        "name": "TensorRT-LLM",
        "description": "High-performance LLM inference backend using NVIDIA TensorRT. Supports tensor parallelism, in-flight batching, and KV cache management.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
            "gpu_device",
            "tensor_parallel_size",
            "max_queue_delay_microseconds",
        ],
    },
    "vllm": {
        "name": "vLLM",
        "description": "Backend for serving LLMs using vLLM engine. Supports PagedAttention, continuous batching, and multi-LoRA.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
            "gpu_device",
        ],
    },
    "python": {
        "name": "Python Backend",
        "description": "Write custom model logic in Python. Supports async execution, shared memory, and BLS (Business Logic Scripting).",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
        ],
    },
    "pytorch": {
        "name": "PyTorch",
        "description": "Serve PyTorch models (TorchScript or Eager mode). Supports GPU and CPU inference with dynamic batching.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
        ],
    },
    "onnxruntime": {
        "name": "ONNX Runtime",
        "description": "Serve ONNX models using ONNX Runtime. Supports GPU and CPU with multiple execution providers.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
        ],
    },
    "tensorrt": {
        "name": "TensorRT",
        "description": "High-performance inference for NVIDIA GPUs using TensorRT. Optimizes models for maximum throughput and minimum latency.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
            "acceleration",
        ],
    },
    "fil": {
        "name": "FIL (Forest Inference Library)",
        "description": "Accelerated inference for decision tree models (XGBoost, LightGBM, Scikit-learn forests).",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
            "max_batch_size",
        ],
    },
    "dali": {
        "name": "DALI (Data Loading Library)",
        "description": "Pre/posprocessing backend using NVIDIA DALI for image, video, and audio pipelines.",
        "config_fields": [
            "model_instance_kind",
            "model_instance_device",
        ],
    },
    "custom": {
        "name": "Custom Backend",
        "description": "Implement your own backend using the Triton Backend C API for custom inference logic.",
        "config_fields": [],
    },
}
