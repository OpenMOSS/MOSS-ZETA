# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Lumine project
"""Rule proposer for the compact six-chunk gamepad action protocol.

This is intentionally a *proposal* mechanism, not constrained decoding.  It
may suggest a delimiter after any compact action atom, including an atom in the
middle of a multi-action chunk.  The target model's existing speculative
verification then accepts the delimiter only when it is the target token, and
otherwise emits the target correction token as usual.

The proposer is conservative by design.  It handles only the exact token-level
compact-v4 contract configured by the deployment: one start token, four atomic
coordinate tokens, bare action atom tokens, a single bare ';' token, and one
end token.  BPE-compressed semicolon runs or malformed output return no draft
instead of attempting character-level recovery.
"""

from collections.abc import Sequence

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch


class CompactActionProposer:
    """Propose one configured delimiter after a compact action atom."""

    def __init__(self, vllm_config: VllmConfig):
        config = vllm_config.speculative_config
        assert config is not None
        assert config.method == "compact_action"
        assert config.num_speculative_tokens == 1
        assert config.compact_action_start_token_id is not None
        assert config.compact_action_end_token_id is not None
        assert config.compact_action_separator_token_id is not None
        assert config.compact_action_coordinate_token_ids is not None
        assert config.compact_action_atom_token_ids is not None

        self.action_start_token_id = config.compact_action_start_token_id
        self.action_end_token_id = config.compact_action_end_token_id
        self.separator_token_id = config.compact_action_separator_token_id
        self.coordinate_token_ids = frozenset(config.compact_action_coordinate_token_ids)
        self.action_atom_token_ids = frozenset(config.compact_action_atom_token_ids)

    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,  # unused; matches other CPU-side proposers
    ) -> list[list[int]]:
        """Return at most one candidate token per active request.

        ``sampled_token_ids`` gates proposals for partial-prefill requests.
        This proposer is called after V1 bookkeeping, so for ordinary
        synchronous decoding ``input_batch.token_ids_cpu`` already includes
        the just-verified output tokens when this method is called.
        """
        del slot_mappings
        drafts: list[list[int]] = []
        for request_index, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                drafts.append([])
                continue
            # V1 bookkeeping has already appended ``sampled_ids`` to this
            # buffer and advanced ``num_tokens_no_spec``.  Do not append the
            # tokens again: doing so would parse a fictitious duplicate of the
            # most recent model output.
            prompt_length = int(input_batch.num_prompt_tokens[request_index])
            token_count = int(input_batch.num_tokens_no_spec[request_index])
            output_token_ids = input_batch.token_ids_cpu[
                request_index, prompt_length:token_count
            ]
            drafts.append(self.propose_from_output_token_ids(output_token_ids))
        return drafts

    def propose_from_output_token_ids(
        self, output_token_ids: Sequence[int]
    ) -> list[int]:
        """Return a draft for one complete output suffix, or ``[]`` safely.

        A compact action body is valid only when it has four coordinate tokens
        after its last ``<a>`` and all later tokens are configured atoms or the
        *bare* separator.  The rule can then draft ';' until five separators
        have been generated, and ``</a>`` for the final chunk.  It never emits
        a delimiter after a separator, coordinate, end tag, BPE semicolon run,
        or an unknown token.
        """
        start_index = -1
        for index in range(len(output_token_ids) - 1, -1, -1):
            if int(output_token_ids[index]) == self.action_start_token_id:
                start_index = index
                break
        if start_index < 0:
            return []

        suffix = output_token_ids[start_index + 1 :]
        if len(suffix) < 5:  # four coordinates and at least one action atom
            return []
        coordinate_ids = suffix[:4]
        if any(int(token_id) not in self.coordinate_token_ids for token_id in coordinate_ids):
            return []

        body = suffix[4:]
        separators = 0
        for token_id in body:
            token_id = int(token_id)
            if token_id == self.separator_token_id:
                separators += 1
            elif token_id not in self.action_atom_token_ids:
                return []

        # Draft only at an action-atom boundary.  A proposed separator after a
        # non-final atom is intentionally allowed here; target verification
        # rejects it when the true next token is another action atom.
        if len(body) == 0 or int(body[-1]) not in self.action_atom_token_ids:
            return []
        if separators < 5:
            return [self.separator_token_id]
        if separators == 5:
            return [self.action_end_token_id]
        return []

    def load_model(self, *args, **kwargs) -> None:
        """No draft-model weights are used by this rule proposer."""
