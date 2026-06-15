import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitEdgeExtractor(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        return self.head(y)


class GatedSemanticFusion(nn.Module):
    """Gated semantic fusion module for text-guided feature modulation."""

    def __init__(self, in_channel):
        super().__init__()
        self.conv_bn_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
        )
        self.conv_bn_2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
        )
        self.conv_bn_3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
        )
        self.conv_bn_4 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1),
            nn.BatchNorm2d(in_channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, mixed_feature, pure_visual_feature):
        q = self.conv_bn_1(mixed_feature)
        k = self.conv_bn_2(pure_visual_feature)
        v = self.conv_bn_3(pure_visual_feature)

        gate_map = self.sigmoid(q * k)
        weighted_feat = self.conv_bn_4(gate_map * v)
        return weighted_feat + mixed_feature


class TextLearnedSpatialGenerator(nn.Module):
    def __init__(self, in_dim, height, width):
        super().__init__()
        self.h = height
        self.w = width
        out_pixels = height * width
        hidden_dim = max(in_dim, out_pixels // 2)
        self.generator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_pixels),
            nn.BatchNorm1d(out_pixels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        spatial_flat = self.generator(x)
        return spatial_flat.view(batch_size, 1, self.h, self.w)


class ResNet52FusionBlock(nn.Module):
    def __init__(self, visual_dim, text_dim, spatial_size):
        super().__init__()
        height, width = spatial_size
        self.spatial_gen = TextLearnedSpatialGenerator(text_dim, height, width)
        self.smooth = nn.Sequential(
            nn.Conv2d(visual_dim + 1, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True),
        )
        self.gated_semantic_fusion = GatedSemanticFusion(in_channel=visual_dim)

    def forward(self, x_visual, x_text):
        text_spatial = self.spatial_gen(x_text)
        mixed_feat = self.smooth(torch.cat([x_visual, text_spatial], dim=1))
        return self.gated_semantic_fusion(
            mixed_feature=mixed_feat,
            pure_visual_feature=x_visual,
        )


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class ParallelExpertNet(nn.Module):
    def __init__(
        self,
        backbone_name="mambaout_tiny",
        num_classes=5,
        text_dim=768,
        img_size=224,
        drop_path_rate=0.1,
        iee_checkpoint=None,
        freeze_iee=True,
    ):
        super().__init__()

        self.edge_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(3 + 16, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        print(f"Initializing backbone: {backbone_name}")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            out_indices=(0, 2),
            in_chans=3,
            drop_path_rate=drop_path_rate,
        )

        feature_info = self.backbone.feature_info
        self.c_stride4 = feature_info[0]["num_chs"]
        self.c_stride16 = feature_info[2]["num_chs"]

        s16_size = img_size // 16

        self.iee_module = ImplicitEdgeExtractor(
            in_channels=self.c_stride4,
            mid_channels=self.c_stride4 // 2,
        )
        if iee_checkpoint and os.path.exists(iee_checkpoint):
            self.load_iee_weights(iee_checkpoint)
        else:
            print(
                "IEE checkpoint was not found. Provide pretrained IEE weights "
                "for paper-consistent finetuning."
            )

        if freeze_iee:
            self.freeze_iee()

        self.text_expert_s16 = ResNet52FusionBlock(
            visual_dim=self.c_stride16,
            text_dim=text_dim,
            spatial_size=(s16_size, s16_size),
        )

        total_dim = self.c_stride4 + self.c_stride16
        self.se_block = SEBlock(channel=total_dim, reduction=16)
        self.head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def load_iee_weights(self, path):
        print(f"Loading IEE weights from: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module.iee."):
                new_key = key.replace("module.iee.", "")
            elif key.startswith("iee."):
                new_key = key.replace("iee.", "")
            else:
                new_key = key
            new_state_dict[new_key] = value
        missing, unexpected = self.iee_module.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"Missing IEE keys: {missing}")
        if unexpected:
            print(f"Unexpected IEE keys: {unexpected}")

    def freeze_iee(self):
        for param in self.iee_module.parameters():
            param.requires_grad = False
        self.iee_module.eval()
        print("IEE module is frozen.")

    @staticmethod
    def _to_nchw(feat, expected_channels):
        if feat.ndim == 4 and feat.shape[1] == expected_channels:
            return feat
        if feat.ndim == 4 and feat.shape[-1] == expected_channels:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    @staticmethod
    def structure_pooling(feat, mask):
        batch_size, channels, _, _ = feat.shape
        feat_flat = feat.view(batch_size, channels, -1)
        mask_flat = mask.view(batch_size, 1, -1)
        numerator = (feat_flat * mask_flat).sum(dim=2)
        denominator = mask_flat.sum(dim=2) + 1e-6
        return numerator / denominator

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "iee_module") and not any(
            p.requires_grad for p in self.iee_module.parameters()
        ):
            self.iee_module.eval()
        return self

    def forward(self, x, text_feat):
        edge_feat = self.edge_extractor(x)
        x_fused = self.fusion_conv(torch.cat([x, edge_feat], dim=1))

        feats = self.backbone(x_fused)
        feat_s4 = self._to_nchw(feats[0], self.c_stride4)
        feat_s16 = self._to_nchw(feats[1], self.c_stride16)

        vis_logits = self.iee_module(feat_s4)
        vis_mask_s4 = torch.sigmoid(vis_logits)
        vec_vis_s4 = self.structure_pooling(feat_s4, vis_mask_s4)

        feat_text_fused_s16 = self.text_expert_s16(feat_s16, text_feat)
        vec_text_s16 = F.adaptive_avg_pool2d(
            feat_text_fused_s16,
            output_size=(1, 1),
        ).flatten(1)

        final_vec = torch.cat([vec_vis_s4, vec_text_s16], dim=1)
        final_vec = self.se_block(final_vec)
        return self.head(final_vec)
