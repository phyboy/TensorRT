import logging

import torch
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


# TODO: Delete this lowering pass once aot_export_joint_simple is patched
def remove_input_alias_fixing_clones(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove the auxiliary clone nodes inserted to fix input aliasing

    See: https://github.com/pytorch/pytorch/issues/108079
    """
    modified_graph = False

    for node in gm.graph.nodes:
        # If the node is a placeholder and its only user is a clone node
        # it was modified by the input alias-fixing pass, and the change
        # needs to be undone
        if (
            node.op == "placeholder"
            and len(node.users) == 1
            and list(node.users)[0].target == torch.ops.aten.clone.default
        ):
            modified_graph = True

            # Replace all uses of the clone with the placholder, delete the clone
            clone_node = list(node.users)[0]
            clone_node.replace_all_uses_with(node)
            gm.graph.erase_node(clone_node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Removed auxiliary clone nodes for placeholders:\n{gm.graph}")

    return gm
