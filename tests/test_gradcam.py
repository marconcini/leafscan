import pytest
import torch
from torch import nn

from leafscan.gradcam import GradCAM, for_model, leaf_focus, pointing_hit, tokens_to_map
from leafscan.model import create_model


class LeftHalfDetector(nn.Module):
    """Class 1 score = mean brightness of the left half of the image."""

    def __init__(self):
        super().__init__()
        self.features = nn.Identity()

    def forward(self, x):
        f = self.features(x.mean(1, keepdim=True))  # (B, 1, H, W)
        w = f.shape[-1]
        left = f[..., : w // 2].mean((1, 2, 3))
        return torch.stack([-left, left], dim=1)


def test_cam_highlights_the_region_that_drives_the_score():
    model = LeftHalfDetector()
    x = torch.zeros(1, 3, 8, 8)
    x[..., 2:6, 1:3] = 1.0  # bright patch in the left half
    cam, logits = GradCAM(model, model.features)(x, torch.tensor([1]))
    assert cam.shape == (1, 8, 8) and 0 <= cam.min() and cam.max() == 1
    assert cam[0, 2:6, 1:3].mean() > cam[0, :, 4:].mean()  # right half doesn't matter


def test_tokens_to_map_drops_prefix_tokens():
    t = torch.arange(2 * 5 * 3, dtype=torch.float).reshape(2, 5, 3)  # 1 cls + 2x2 patches
    m = tokens_to_map(t, num_prefix_tokens=1)
    assert m.shape == (2, 3, 2, 2)
    assert torch.equal(m[0, :, 0, 0], t[0, 1])
    with pytest.raises(ValueError):
        tokens_to_map(t, num_prefix_tokens=0)  # 5 tokens is not a square


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0", "convnext_atto",
                                  "vit_tiny_patch16_224"])
def test_gradcam_runs_on_every_architecture(name):
    model = create_model(name, num_classes=4, pretrained=False).eval()
    size = 224 if name.startswith("vit") else 64
    cam, logits = for_model(model, name)(torch.randn(2, 3, size, size))
    assert cam.shape == (2, size, size) and logits.shape == (2, 4)
    assert torch.isfinite(cam).all()


def test_leaf_metrics():
    mask = torch.zeros(4, 4)
    mask[:, :2] = 1  # leaf = left half
    uniform = torch.ones(4, 4)
    assert leaf_focus(uniform, mask) == 0.5  # uniform map -> leaf area
    on_leaf = torch.zeros(4, 4)
    on_leaf[1, 0] = 1
    assert leaf_focus(on_leaf, mask) == 1.0 and pointing_hit(on_leaf, mask)
    assert not pointing_hit(on_leaf.flip(1), mask)
