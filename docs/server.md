./build-cuda/bin/llama-server -hf sahilchachra/Unlimited-OCR-GGUF:BF16 --host 127.0.0.1 --port 8001 -sm layer -np 48


Build is CUDA 12.5 + NCCL

Mason: V100 (32GB) *5 
