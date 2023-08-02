import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.test_utils import DispatchTestCase
from torch_tensorrt import Input


class TestErfConverter(DispatchTestCase):
    def test_erf(self):
        class erf(nn.Module):
            def forward(self, input):
                return torch.erf(input)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            erf(),
            inputs,
            expected_ops={torch.ops.aten.erf.default},
        )


if __name__ == "__main__":
    run_tests()