import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestMatMulConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "2_2",
                (2, 3),
                (3, 2),
            ),
            (
                "4_6",
                (4, 5),
                (5, 6),
            ),
            (
                "2_1",
                (2, 3),
                (3, 1),
            ),
            (
                "4_1",
                (4, 1),
                (1, 1),
            ),
            (
                "1_2",
                (1, 3),
                (3, 2),
            ),
            (
                "1_3",
                (1, 2),
                (2, 3),
            ),
            # Following cases use torch.ops.aten.bmm.default
            # ("4_3", (3,1,3,2), (2,2,3)),
            # ("3_4", (3,1,3,2), (2,2,3)),
            # ("3_4", (2, 2, 3), (3, 1, 3, 3)),
            # ("4_2", (1, 2, 2, 3), (3, 2)),
        ]
    )
    def test_matmul_mm(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.matmul(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )

    @parameterized.expand(
        [
            (
                "1_1",
                (1, 1),
                (1,),
            ),
            (
                "1_1",
                (1, 2),
                (2,),
            ),
            (
                "2_1",
                (2, 1),
                (1,),
            ),
            (
                "3_1",
                (3, 4),
                (4,),
            ),
        ]
    )
    def test_matmul_mv(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def __init__(self):
                super().__init__()
                self.other = nn.Parameter(torch.randn(*other_shape))

            def forward(self, input):
                return torch.matmul(input, self.other)

        inputs = [torch.randn(*input_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mv.default},
        )

    @parameterized.expand(
        [
            ("2_2", (2, 3), (3, 2)),
            # ("2_3", (2, 3), (2, 3, 4)),
            # ("4_4", (2, 2, 2, 3), (2, 1, 3, 2)),
            # ("4_2", (2, 1, 2, 3), (3, 2)),
            # ("2_1", (2, 3), (3,)),
            # ("1_2", (3,), (3, 2)),
            # ("1_1", (3,), (3,)),
        ]
    )
    def test_matmul(self, _, input_shape, other_shape):
        class MatMul(nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        inputs = [torch.randn(*input_shape), torch.randn(*other_shape)]

        self.run_test(
            MatMul(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )

    # FIXME: dynamic shape is giving bmm


if __name__ == "__main__":
    run_tests()
