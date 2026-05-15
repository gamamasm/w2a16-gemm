import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import SimpleFNN

os.environ['LD_LIBRARY_PATH'] = '/root/anaconda3/envs/pytorch_2.5.1/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64' + (':' + os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else '')

try:
    import w2a16_gemm
    HAS_CUDA_KERNEL = True
except ImportError:
    HAS_CUDA_KERNEL = False
    print("Warning: CUDA kernel not compiled, using PyTorch implementation")


class W2A16Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.padded_in_features = (in_features + 3) // 4 * 4

        self.register_buffer('weight_int2', torch.zeros(out_features, self.padded_in_features // 4, dtype=torch.uint8))
        self.register_buffer('scale', torch.ones(out_features, self.num_groups, dtype=torch.float16))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        return w2a16_gemm.w2a16_gemm(
                self.weight_int2,
                self.scale,
                x,
                self.bias if self.bias is not None else torch.tensor([]),
                self.in_features
            )


def replace_linear_with_quantized(model, group_size=128):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.split('.')[-1] if '.' in name else name

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            quantized_linear = W2A16Linear(
                module.in_features,
                module.out_features,
                group_size=group_size,
                bias=module.bias is not None
            )
            setattr(parent, child_name, quantized_linear)
        else:
            replace_linear_with_quantized(module, group_size)
    return model


def test_quantized_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"CUDA kernel: {'Enabled' if HAS_CUDA_KERNEL else 'Disabled'}")

    input_dim = 1024
    hidden_dims = [16384, 16384, 16384, 16384, 16384, 16384]
    output_dim = 1

    model = SimpleFNN(input_dim, hidden_dims, output_dim)
    model = replace_linear_with_quantized(model, group_size=128)

    quantized_model_path = "/mnt/d/python/GEMM/model/quantized_model_w2a16.pth"
    model.load_state_dict(torch.load(quantized_model_path, weights_only=True))
    print(f"Loaded quantized model from {quantized_model_path}")

    model = model.to(device)
    model.eval()

    test_input = torch.ones(1, input_dim).to(device).half()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    import time
    start_time = time.perf_counter()

    with torch.no_grad():
        predictions = model(test_input)

    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"\n=== Quantized Model Inference Results ===")
        print(f"Inference time: {inference_time:.2f} ms")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
    else:
        print(f"\n=== Quantized Model Inference Results ===")
        print(f"Inference time: {inference_time:.2f} ms")

    print(f"Output: {predictions.item():.6f}")

    model_size = os.path.getsize(quantized_model_path) / (1024 ** 2)
    print(f"Model size: {model_size:.2f} MB")


if __name__ == "__main__":
    test_quantized_model()
