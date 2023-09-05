import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from harness import DispatchTestCase

class TestArgmaxConverter(DispatchTestCase):
    @parameterized.expand(
            [
                ("dim_0_keep_dim_false", (3, 4), 0, False)
            ]
    )

    def test_argmax(self, _, input_shape, dim, keep_dim):
        class ArgMax(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input): 
                return torch.argmax(input, dim, keep_dim)
            

        input = [torch.randn(*input_shape)]

        self.run_test(
            ArgMax(),
            input, 
            expected_ops={torch.ops.aten.argmax.default}
        )

if __name__ == "__main__":
    run_tests()  


