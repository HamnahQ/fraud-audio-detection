import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Brevitas
from brevitas.nn import QuantIdentity, QuantLinear
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Int16Bias
from brevitas.export import export_qonnx


class AttentionPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        return (weights * x).sum(dim=1)    # [B, D]


class QuantAttentionPool(nn.Module):
    """
    Quantized version of AttentionPool's Linear.
    Softmax stays float (that's fine).
    """
    def __init__(self, in_dim):
        super().__init__()
        self.attn = QuantLinear(
            in_dim, 1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias,
            input_quant=Int8ActPerTensorFloat,
            output_quant=None  # keep scores float for softmax stability
        )

    def forward(self, x):
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1) # [B, T, 1]
        return (weights * x).sum(dim=1)    # [B, D]


class CRNNWithAttn_QONNX(nn.Module):
    """
    - ResNet18 backbone kept float (torchvision).
    - GRU kept float (PyTorch).
    - Attention + classifier use Brevitas quant modules to embed QONNX Quant ops.
    """
    def __init__(self, pretrained=True, hidden_size=128, num_layers=1, dropout=0.2):
        super().__init__()

        # 0) Input quant stub (this creates a Quant node at the model input in QONNX)
        self.inp_quant = QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False
        )

        # 1) Pretrained ResNet18 (float)
        if pretrained:
            resnet = models.resnet18(weights="DEFAULT")
        else:
            resnet = models.resnet18(weights=None)

        # --- IMPORTANT FIX ---
        # Your forward expects 2 channels (you set conv1 to 2), so x should be [B, 2, F, T].
        # Initialize conv1 from pretrained 3-channel weights by averaging across RGB and copying into 2 chans.
        old_w = resnet.conv1.weight.data.clone()  # [64, 3, 7, 7]
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        avg_w = old_w.mean(dim=1)  # [64, 7, 7]
        resnet.conv1.weight.data[:, 0] = avg_w
        resnet.conv1.weight.data[:, 1] = avg_w

        # Remove avgpool & fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 512, F', T']

        # 2) Bi-GRU (float)
        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 3) Quantized attention pooling (quant Linear inside)
        self.attn_pool = QuantAttentionPool(hidden_size * 2)

        # 4) Quantized classification head
        self.fc1 = QuantLinear(
            hidden_size * 2, hidden_size, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias,
            input_quant=Int8ActPerTensorFloat,
            output_quant=Int8ActPerTensorFloat
        )
        self.drop = nn.Dropout(dropout)
        self.fc2 = QuantLinear(
            hidden_size, 1, bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int16Bias,
            input_quant=Int8ActPerTensorFloat,
            output_quant=None  # logits float
        )

    def forward(self, x):
        # x: [B, 2, F, T]
        x = self.inp_quant(x)

        feat = self.backbone(x)         # [B, 512, F', T']
        feat = feat.mean(dim=2)         # collapse freq -> [B, 512, T']
        feat = feat.permute(0, 2, 1)    # -> [B, T', 512]

        out, _ = self.gru(feat)         # -> [B, T', 2*hidden]
        pooled = self.attn_pool(out)    # -> [B, 2*hidden]

        h = self.fc1(pooled)
        h = F.relu(h)
        h = self.drop(h)
        logits = self.fc2(h)            # -> [B, 1]
        return logits


def export_model_to_qonnx(
    export_path: str = "crnn_attn_qonnx.onnx",
    pretrained: bool = True,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.2,
    F_bins: int = 224,   # choose the input height you actually use
    T_frames: int = 224  # choose the input width you actually use
):
    torch.manual_seed(0)

    model = CRNNWithAttn_QONNX(
        pretrained=pretrained,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).eval()

    # Dummy input must match forward: [B, 2, F, T]
    dummy = torch.randn(1, 2, F_bins, T_frames)

    exported_model = export_qonnx(
        model,
        input_t=dummy,
        export_path=export_path
    )
    return exported_model


if __name__ == "__main__":
    export_model_to_qonnx("crnn_attn_qonnx.onnx", pretrained=True, F_bins=224, T_frames=224)
    print("Exported QONNX to crnn_attn_qonnx.onnx")