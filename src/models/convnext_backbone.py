import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.ops import StochasticDepth

CONVNEXT_TINY_22K_URL = (
    "https://dl.fbaipublicfiles.com/convnext/"
    "convnext_tiny_22k_224.pth"
)

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-6,
        data_format="channels_last",
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in [
            "channels_last",
            "channels_first",
        ]:
            raise NotImplementedError

        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = (
            self.weight[:, None, None] * x
            + self.bias[:, None, None]
        )

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = LayerNorm(
            dim,
            eps=1e-6,
        )

        self.pwconv1 = nn.Linear(
            dim,
            4 * dim,
        )

        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(
            4 * dim,
            dim,
        )

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value
                * torch.ones((dim)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )

        self.stochastic_depth = StochasticDepth(
            drop_path,
            "row",
        )

    def forward(self, x):
        residual = x

        x = self.dwconv(x)

        x = x.permute(
            0,
            2,
            3,
            1,
        )

        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(
            0,
            3,
            1,
            2,
        )
        x = residual + self.stochastic_depth(x)
        return x


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=None,
        dims=None,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
    ):
        super().__init__()

        if depths is None:
            depths = [3, 3, 9, 3]

        if dims is None:
            dims = [96, 192, 384, 768]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_chans,
                dims[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                dims[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )

        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(
                    dims[i],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    dims[i],
                    dims[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(
                downsample_layer
            )

        self.stages = nn.ModuleList()

        dp_rates = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(depths),
            )
        ]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=(
                            layer_scale_init_value
                        ),
                    )
                    for j in range(depths[i])
                ]
            )

            self.stages.append(stage)

            cur += depths[i]

        self.norm = nn.LayerNorm(
            dims[-1],
            eps=1e-6,
        )

        self.head = nn.Linear(
            dims[-1],
            num_classes,
        )

        for m in self.modules():
            if isinstance(
                m,
                (nn.Conv2d, nn.Linear),
            ):
                nn.init.trunc_normal_(
                    m.weight,
                    std=0.02,
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1])

        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.head(x)


def build_convnext_tiny(
    num_classes=7,
    pretrained=True,
    drop_path_rate=0.1,
):
    # ImageNet-22K ConvNeXt checkpoint has 21,841 classes.
    pretrained_num_classes = (
        21841 if pretrained else num_classes
    )

    model = ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        num_classes=pretrained_num_classes,
        drop_path_rate=drop_path_rate,
    )

    if pretrained:
        checkpoint = load_state_dict_from_url(
            url=CONVNEXT_TINY_22K_URL,
            map_location="cpu",
        )
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
        model.head = nn.Linear(
            768,
            num_classes,
        )

    return model