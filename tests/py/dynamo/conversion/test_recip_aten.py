import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestRecipConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.float),
            ((1, 20), torch.float),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_recip_float(self, input_shape, dtype):
        class recip(nn.Module):
            def forward(self, input):
                return torch.reciprocal(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            recip(),
            inputs,
            expected_ops={torch.ops.aten.reciprocal.default},
        )

    @parameterized.expand(
        [
            ((10,), torch.int, 0, 5),
            ((1, 20), torch.int32, -10, 10),
            ((2, 3, 4), torch.int, -5, 5),
        ]
    )
    def test_recip_int(self, input_shape, dtype, low, high):
        class recip(nn.Module):
            def forward(self, input):
                return torch.reciprocal(input)

        inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
        self.run_test(
            recip(),
            inputs,
            expected_ops={torch.ops.aten.reciprocal.default},
        )


if __name__ == "__main__":
    run_tests()
