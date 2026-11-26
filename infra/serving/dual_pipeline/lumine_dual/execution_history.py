"""Completed-action history, independent of speculative inference and OCR delivery.

Only the workstation can acknowledge that the entire action packet completed.
Raw OCR delivery remains causal even when its associated action never executes.
"""
from collections import OrderedDict
from copy import copy, deepcopy
import math

from pretrain import training_ocr_input as ocr
from .protocol import digest, image_ids, validate_block

MODE = 'completed_ack_v1'


def encode_ocr(state, previous):
    if previous is None:
        return ocr.render('S', {key: state[key] for key in ocr.FIELDS if state[key]})
    changes = {key: state[key] for key in ocr.FIELDS if state[key] != previous[key]}
    return ocr.render('D', changes) if changes else ocr.EMPTY_BLOCK


class ExecutionHistory:
    def __init__(self):
        self.pending = OrderedDict()
        self.resolved = {}
        self.delivered = None
        self.observation_count = 0
        self.committed_count = 0
        self.committed_ids = []
        self.last_completed_action_seq = 0
        self.last_completed_generation = 0

    def pinned_images(self):
        return [key for candidate in self.pending.values() for key in candidate['images']]

    def apply(self, context, receipts):
        """Validate a whole receipt batch before changing any committed state."""
        if not isinstance(receipts, list) or len(receipts) > 128:
            raise ValueError('execution receipts must be a list of at most 128 records')
        state = copy(self)
        state.pending = OrderedDict(self.pending)
        state.resolved = dict(self.resolved)
        state.committed_ids = list(self.committed_ids)
        updated = deepcopy(context)
        applied = []
        for receipt in receipts:
            if not isinstance(receipt, dict):
                raise ValueError('Invalid execution receipt')
            rid, status = receipt.get('request_id'), receipt.get('status')
            if not isinstance(rid, str) or not rid or status not in ('completed', 'discarded'):
                raise ValueError('Execution receipt requires request_id and completed/discarded status')
            fingerprint = digest(receipt)
            if rid in state.resolved:
                if state.resolved[rid] != fingerprint:
                    raise ValueError('Conflicting execution receipt for ' + rid)
                applied.append(deepcopy(receipt))
                continue
            candidate = state.pending.get(rid)
            if candidate is None:
                raise ValueError('Execution receipt references an unknown candidate: ' + rid)
            if status == 'completed':
                seq = receipt.get('action_seq')
                completed_at = receipt.get('completed_at')
                if type(seq) is not int or seq <= state.last_completed_action_seq:
                    raise ValueError('Completed action_seq must strictly increase')
                if (not isinstance(completed_at, (str, int, float)) or isinstance(completed_at, bool)
                        or isinstance(completed_at, str) and not completed_at.strip()
                        or isinstance(completed_at, (int, float)) and not math.isfinite(completed_at)):
                    raise ValueError('Completed receipt requires completed_at')
                if candidate['generation'] <= state.last_completed_generation:
                    raise ValueError('Completed candidates must follow causal generation order')
                block = encode_ocr(candidate['delivered'], updated.delivered)
                before = len(updated.turns)
                staged, user, _, _ = updated.stage(candidate['images'], block)
                dropped = before - len(staged.turns)
                if dropped:
                    del state.committed_ids[:dropped]
                staged.turns.append((user, candidate['content']))
                updated = staged
                state.committed_ids.append(rid)
                state.committed_count += 1
                state.last_completed_action_seq = seq
                state.last_completed_generation = candidate['generation']
            state.resolved[rid] = fingerprint
            del state.pending[rid]
            applied.append(deepcopy(receipt))
        self.__dict__.update(state.__dict__)
        return updated, applied

    def stage(self, context, images, wire_block):
        if len(self.pending) >= 128:
            raise ValueError('Too many unacknowledged candidates; execution receipts are required')
        validate_block(wire_block)
        delivered = ocr.replay(wire_block, self.delivered)
        model_block = encode_ocr(delivered, context.delivered)
        staged, observation, messages, _ = context.stage(images, model_block)
        # This optimization contains only confirmed completed actions. Future
        # receipts may extend it, but exact token/media checks remain mandatory.
        prefix = staged.messages()
        self.delivered = delivered
        self.observation_count += 1
        dropped = len(context.turns) - len(staged.turns)
        visible_ids = self.committed_ids[dropped:]
        audit = dict(sequence=self.observation_count, wire_block=wire_block,
                     model_block=model_block, delivered_snapshot=deepcopy(delivered),
                     rebased=wire_block != model_block)
        return staged, observation, messages, prefix, visible_ids, audit

    def remember(self, rid, images, content, generation):
        if len(self.pending) >= 128:
            raise ValueError('Too many unacknowledged candidates; execution receipts are required')
        self.pending[rid] = dict(images=list(images), content=content,
                                 delivered=deepcopy(self.delivered), generation=generation)

    def summary(self, applied, visible_ids, observation):
        return dict(mode=MODE, applied_receipts=deepcopy(applied),
                    committed_request_ids=list(visible_ids),
                    committed_request_ids_scope='current_prefill_window',
                    pending_request_ids=list(self.pending),
                    committed_count=self.committed_count,
                    last_completed_action_seq=self.last_completed_action_seq,
                    observation_ocr=deepcopy(observation))
