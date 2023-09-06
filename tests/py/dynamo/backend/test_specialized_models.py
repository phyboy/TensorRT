import torch
import torch_tensorrt
from torch.testing._internal.common_utils import TestCase, run_tests

from ..testing_utilities import lower_graph_testing


class TestFakeTensors(TestCase):
    def test_lowering_mul_int(self):
        class MulInt(torch.nn.Module):
            def forward(self, x):
                return x * 7

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.mul.Tensor,
        }

        inputs = [
            torch.rand(
                3,
                5,
                7,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(MulInt())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"MulInt TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()

    def test_lowering_add_float(self):
        class AddFloat(torch.nn.Module):
            def forward(self, x):
                return x + 84.0

        # Operations expected to be included in the traced graph after decompositions
        expected_ops = {
            torch.ops.aten.add.Tensor,
        }

        inputs = [
            torch.rand(
                1,
                5,
                7,
                9,
            ).cuda(),
        ]

        fx_graph = torch.fx.symbolic_trace(AddFloat())
        _, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )

        torch._dynamo.reset()

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"AddFloat TRT outputs don't match with the original model.",
        )

        torch._dynamo.reset()


class Test0DTensors(TestCase):
    def test_0D_input(self):
        class Tensor0DInput(torch.nn.Module):
            def forward(self, x):
                return x * 7

        inputs = [
            torch.tensor(
                3,
            )
            .cuda()
            .int(),
        ]

        fx_graph = torch.fx.symbolic_trace(Tensor0DInput())

        # Validate that the results between Torch and Torch-TRT are similar
        optimized_model = torch_tensorrt.compile(
            fx_graph,
            "torch_compile",
            inputs,
            min_block_size=1,
            pass_through_build_failures=True,
        )
        optimized_model_results = optimized_model(*inputs).detach().cpu()
        torch_model_results = fx_graph(*inputs).detach().cpu()

        max_diff = float(
            torch.max(torch.abs(optimized_model_results - torch_model_results))
        )
        self.assertAlmostEqual(
            max_diff,
            0,
            msg=f"0D-Tensor TRT outputs don't match with the original model.",
        )
        torch._dynamo.reset()


class TestPacketOperator(TestCase):
    def test_packet_operator(self):
        class PacketAsOperator(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.tanh(x)

        # Operations expected to be removed in the traced graph
        expected_ops = {torch.ops.aten.tanh.default}
        unexpected_ops = {
            torch.ops.aten.tanh,
        }

        inputs = [torch.rand((3, 5)).cuda()]

        fx_graph = torch.fx.symbolic_trace(PacketAsOperator())
        unexpected_ops_seen, expected_ops_unseen = lower_graph_testing(
            fx_graph,
            inputs,
            expected_ops=expected_ops,
            unexpected_ops=unexpected_ops,
            min_block_size=1,
        )

        self.assertEquals(
            len(unexpected_ops_seen),
            0,
            f"The following unexpected ops were encountered: {unexpected_ops_seen}",
        )

        self.assertEquals(
            len(expected_ops_unseen),
            0,
            f"The following expected ops were not encountered: {expected_ops_unseen}",
        )


if __name__ == "__main__":
    run_tests()
