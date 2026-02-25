import pytest
import torch
from perception.model import SegmentationNet

def test_segmentation_forward():
    model = SegmentationNet(num_classes=2)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 2, 64, 64)
