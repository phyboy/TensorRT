import torch
import torch.fx
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestUnsqueeze(DispatchTestCase):
    @parameterized.expand(
        [
            ("negative_dim", -2),
            ("positive_dim", 2),
        ]
    )
    def test_unsqueeze(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.unsqueeze(x, self.dim)

        inputs = [torch.randn(1, 2, 3)]
        self.run_test(
            Unsqueeze(dim), inputs, expected_ops={torch.ops.aten.unsqueeze.default}
        )

    # Testing with more than one dynamic dims results in following error:
    # AssertionError: Currently we don't support unsqueeze with more than one dynamic dims.

    @parameterized.expand(
        [
            ("negative_dim_dynamic", -4),
            ("positive_dim_dynamic", 1),
        ]
    )
    def test_unsqueeze_with_dynamic_shape(self, _, dim):
        class Unsqueeze(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                return torch.unsqueeze(x, self.dim)

        input_specs = [
            Input(
                shape=(-1, 2, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 2, 3), (2, 2, 3), (3, 2, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Unsqueeze(dim), input_specs, expected_ops={torch.ops.aten.unsqueeze.default}
        )


if __name__ == "__main__":
    run_tests()
