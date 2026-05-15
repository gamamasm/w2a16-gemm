import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model import SimpleFNN

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
        dev = x.device

        w = self.weight_int2.to(dev)
        s = self.scale.to(dev)

        # Unpack 4 INT2 values per byte
        w0 = (w & 0x03).float()
        w1 = ((w >> 2) & 0x03).float()
        w2 = ((w >> 4) & 0x03).float()
        w3 = ((w >> 6) & 0x03).float()

        w_interleaved = torch.stack([w0, w1, w2, w3], dim=2)
        w_expanded = w_interleaved.view(self.out_features, self.padded_in_features)

        if self.in_features < self.padded_in_features:
            w_expanded = w_expanded[:, :self.in_features]

        w_groups = w_expanded.view(self.out_features, self.num_groups, self.group_size)
        w_fp = (w_groups - 2.0) * s.unsqueeze(2)
        w_fp = w_fp.view(self.out_features, self.in_features).half()

        b = self.bias
        if b is not None:
            b = b.to(dev)

        return nn.functional.linear(x, w_fp, b)


def quantize_weight_to_int2(weight_fp16, group_size=128):
    out_features, in_features = weight_fp16.shape
    num_groups = in_features // group_size

    weight_groups = weight_fp16.view(out_features, num_groups, group_size)
    weight_max = weight_groups.abs().max(dim=2)[0]
    scale = weight_max / 1.0

    weight_int2 = torch.round(weight_groups / scale.unsqueeze(2))
    weight_int2 = (weight_int2 + 2).clamp(0, 3).to(torch.uint8)

    padded_in_features = (in_features + 3) // 4 * 4
    if in_features < padded_in_features:
        padding = torch.zeros(out_features, padded_in_features - in_features, dtype=torch.uint8)
        weight_int2 = torch.cat([weight_int2.view(out_features, -1), padding], dim=1)

    weight_flat = weight_int2.view(out_features, -1)

    # Pack 4 INT2 values per byte: val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)
    val0 = weight_flat[:, 0::4]
    val1 = weight_flat[:, 1::4]
    val2 = weight_flat[:, 2::4]
    val3 = weight_flat[:, 3::4]

    weight_packed = val0 | (val1 << 2) | (val2 << 4) | (val3 << 6)

    return weight_packed, scale


def quantize_model(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.split('.')[-1] if '.' in name else name

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            weight_int2, scale = quantize_weight_to_int2(module.weight.data, group_size=128)

            quantized_linear = W2A16Linear(
                module.in_features,
                module.out_features,
                group_size=128,
                bias=module.bias is not None
            )
            quantized_linear.weight_int2 = weight_int2
            quantized_linear.scale = scale
            if module.bias is not None:
                quantized_linear.bias = module.bias.data.clone()

            setattr(parent, child_name, quantized_linear)
            print(f"Quantized: {name}")
        else:
            quantize_model(module)

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    input_dim = 1024
    hidden_dims = [16384, 16384, 16384, 16384, 16384, 16384]
    output_dim = 1

    model = SimpleFNN(input_dim, hidden_dims, output_dim).half()

    model_path = "/mnt/d/python/GEMM/model/model_weights.pth"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Loaded FP16 model weights from {model_path}")

    model = model.to(device)
    model = quantize_model(model)
    model.eval()

    quantized_model_path = "/mnt/d/python/GEMM/model/quantized_model_w2a16.pth"
    torch.save(model.state_dict(), quantized_model_path)
    print(f"\nQuantized model saved to {quantized_model_path}")

    original_size = os.path.getsize(model_path) / (1024 ** 2)
    quantized_size = os.path.getsize(quantized_model_path) / (1024 ** 2)
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")

    test_input = torch.ones(1, input_dim).to(device).half()
    with torch.no_grad():
        output = model(test_input)
    print(f"\nTest output: {output.item():.6f}")


if __name__ == "__main__":
    main()
