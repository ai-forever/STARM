"""
HRM ACT V2: Transformer Baseline for Architecture Ablation

This is an architecture ablation of the Hierarchical Reasoning Model (HRM).
Key changes from V1:
1. REMOVED hierarchical split (no separate H and L levels)
2. REMOVED inner cycles (no H_cycles/L_cycles loops within reasoning)
3. KEPT ACT outer loop structure intact
4. KEPT all data preprocessing, embeddings, and evaluation infrastructure

Architecture: Single-level transformer that processes the full 30x30 grid as a
900-token sequence, with the same positional encodings and sparse embeddings as V1.

"""

import math
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLUD, AttentionD, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear, \
    _find_multiple
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV2DropoutInnerCarry:
    z_H: torch.Tensor

    embed_dropout_mask: Optional[torch.Tensor] = None  # [B, S, hidden]

    H_qkv_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)
    H_attn_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)  # separate mask for attn
    H_mlp_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)  # separate mask for mlp
    H_ffn_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)

    def with_updated_states(self, z_H: torch.Tensor) -> 'HierarchicalReasoningModel_ACTV2DropoutInnerCarry':
        return HierarchicalReasoningModel_ACTV2DropoutInnerCarry(
            z_H=z_H,
            embed_dropout_mask=self.embed_dropout_mask,
            H_qkv_dropout_masks=self.H_qkv_dropout_masks,
            H_attn_residual_dropout_masks=self.H_attn_residual_dropout_masks,
            H_mlp_residual_dropout_masks=self.H_mlp_residual_dropout_masks,
            H_ffn_dropout_masks=self.H_ffn_dropout_masks,
        )


@dataclass
class HierarchicalReasoningModel_ACTV2DropoutCarry:
    inner_carry: HierarchicalReasoningModel_ACTV2DropoutInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV2DropoutConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int

    H_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float
    act_enabled: bool = True  # If False, always run halt_max_steps (no early stopping during training)
    act_inference: bool = False  # If True, use adaptive computation during inference

    forward_dtype: str = "bfloat16"

    embed_dropout_p: float = 0.1
    qkv_dropout_p: float = 0.1
    residual_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1

    @property
    def ffn_inter_size(self) -> int:
        return _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @property
    def qkv_proj_size(self) -> int:
        return (self.num_heads + 2 * self.num_heads) * (self.hidden_size // self.num_heads)


class HierarchicalReasoningModel_ACTV2DropoutBlock(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2DropoutConfig) -> None:
        super().__init__()

        self.self_attn = AttentionD(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLUD(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor,
                qkv_dropout_mask: Optional[torch.Tensor] = None,
                attn_residual_dropout_mask: Optional[torch.Tensor] = None,
                mlp_residual_dropout_mask: Optional[torch.Tensor] = None,
                ffn_dropout_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Post Norm
        # Self Attention
        attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states,
                                  qkv_dropout_mask=qkv_dropout_mask)
        # Residual Dropout for Attention part
        if attn_residual_dropout_mask is not None:
            attn_out = attn_out * attn_residual_dropout_mask

        hidden_states = rms_norm(hidden_states + attn_out, variance_epsilon=self.norm_eps)

        # Fully Connected
        mlp_out = self.mlp(hidden_states, ffn_dropout_mask=ffn_dropout_mask)

        # Residual Dropout for MLP part
        if mlp_residual_dropout_mask is not None:
            mlp_out = mlp_out * mlp_residual_dropout_mask

        hidden_states = rms_norm(hidden_states + mlp_out, variance_epsilon=self.norm_eps)

        return hidden_states


class HierarchicalReasoningModel_ACTV2DropoutReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV2DropoutBlock]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor,
                qkv_dropout_masks: List[Optional[torch.Tensor]],
                attn_residual_dropout_masks: List[Optional[torch.Tensor]],
                mlp_residual_dropout_masks: List[Optional[torch.Tensor]],
                ffn_dropout_masks: List[Optional[torch.Tensor]],
                **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        # Layers
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                qkv_dropout_mask=qkv_dropout_masks[i],
                attn_residual_dropout_mask=attn_residual_dropout_masks[i],
                mlp_residual_dropout_mask=mlp_residual_dropout_masks[i],
                ffn_dropout_mask=ffn_dropout_masks[i],
                **kwargs
            )

        return hidden_states


class HierarchicalReasoningModel_ACTV2Dropout_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2DropoutConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype,
            )

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                init_std=embed_init_std,
                cast_to=self.forward_dtype,
            )
        else:
            raise NotImplementedError()

        # Reasoning Layers
        self.H_level = HierarchicalReasoningModel_ACTV2DropoutReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV2DropoutBlock(self.config) for _i in range(self.config.H_layers)]
        )

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor,
                          embed_dropout_mask: Optional[torch.Tensor]):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2
            )

        if embed_dropout_mask is not None:
            embedding = embedding * embed_dropout_mask

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 1 / math.sqrt(2) * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def _maybe_generate_mask(self, p: float, shape: Tuple[int, ...], enabled: bool) -> Optional[torch.Tensor]:
        if not enabled or p <= 1e-6:
            return None

        mask = torch.bernoulli(
            torch.full(shape, 1 - p, device=self.H_init.device, dtype=torch.float32)
        )

        return mask.to(self.forward_dtype) / (1 - p)

    def empty_carry(self, batch_size: int):
        seq_len = self.config.seq_len + self.puzzle_emb_len

        def _empty_mask_list(num_layers: int) -> List[Optional[torch.Tensor]]:
            return [None] * num_layers

        return HierarchicalReasoningModel_ACTV2DropoutInnerCarry(
            z_H=torch.empty(
                batch_size,
                seq_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
            embed_dropout_mask=None,
            H_qkv_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_attn_residual_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_mlp_residual_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_ffn_dropout_masks=_empty_mask_list(self.config.H_layers),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV2DropoutInnerCarry):
        batch_size = reset_flag.shape[0]
        seq_len = self.config.seq_len + self.puzzle_emb_len

        def _update_mask(new_mask: Optional[torch.Tensor], old_mask: Optional[torch.Tensor], ndim: int) -> Optional[
            torch.Tensor]:
            if new_mask is None and old_mask is None:
                return None
            if new_mask is None:
                return None
            if old_mask is None:
                return new_mask
            view_shape = tuple([batch_size] + [1] * (ndim - 1))
            return torch.where(reset_flag.view(*view_shape), new_mask, old_mask)

        def _generate_layer_masks(num_layers: int, shape: Tuple[int, ...], p: float) -> List[Optional[torch.Tensor]]:
            if not self.training or p <= 1e-6:
                return [None] * num_layers
            return [self._maybe_generate_mask(p, shape, self.training) for _ in range(num_layers)]

        def _update_mask_list(new_masks: List[Optional[torch.Tensor]],
                              old_masks: List[Optional[torch.Tensor]],
                              ndim: int) -> List[Optional[torch.Tensor]]:
            return [_update_mask(n, o, ndim) for n, o in zip(new_masks, old_masks)]

        # Embed
        cand_embed = self._maybe_generate_mask(self.config.embed_dropout_p,
                                               (batch_size, seq_len, self.config.hidden_size),
                                               self.training)
        embed_mask = _update_mask(cand_embed, carry.embed_dropout_mask, ndim=3) if (
                cand_embed is not None or carry.embed_dropout_mask is not None) else None

        # H level
        H_qkv = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.qkv_proj_size),
                                      self.config.qkv_dropout_p)
        H_attn_res = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.hidden_size),
                                           self.config.residual_dropout_p)
        H_mlp_res = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.hidden_size),
                                          self.config.residual_dropout_p)
        H_ffn = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.ffn_inter_size),
                                      self.config.ffn_dropout_p)

        return HierarchicalReasoningModel_ACTV2DropoutInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            embed_dropout_mask=embed_mask,
            H_qkv_dropout_masks=_update_mask_list(H_qkv, carry.H_qkv_dropout_masks, ndim=3),
            H_attn_residual_dropout_masks=_update_mask_list(H_attn_res, carry.H_attn_residual_dropout_masks, ndim=3),
            H_mlp_residual_dropout_masks=_update_mask_list(H_mlp_res, carry.H_mlp_residual_dropout_masks, ndim=3),
            H_ffn_dropout_masks=_update_mask_list(H_ffn, carry.H_ffn_dropout_masks, ndim=3),
        )

    def forward(
            self, carry: HierarchicalReasoningModel_ACTV2DropoutInnerCarry, batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV2DropoutInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"],
                                                  carry.embed_dropout_mask)

        # 1-step grad
        z_H = self.H_level(carry.z_H, input_embeddings,
                           qkv_dropout_masks=carry.H_qkv_dropout_masks,
                           attn_residual_dropout_masks=carry.H_attn_residual_dropout_masks,
                           mlp_residual_dropout_masks=carry.H_mlp_residual_dropout_masks,
                           ffn_dropout_masks=carry.H_ffn_dropout_masks,
                           **seq_info)

        # LM Outputs
        new_carry = carry.with_updated_states(z_H.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV2Dropout(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV2DropoutConfig(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV2Dropout_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV2DropoutCarry(
            inner_carry=self.inner.empty_carry(
                batch_size
            ),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
            self,
            carry: HierarchicalReasoningModel_ACTV2DropoutCarry,
            batch: Dict[str, torch.Tensor],
            compute_target_q: bool = True,
    ) -> Tuple[HierarchicalReasoningModel_ACTV2DropoutCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps

            halted = is_last_step

            # Check if adaptive computation should be used
            use_adaptive = (self.config.halt_max_steps > 1) and (
                    (self.training and self.config.act_enabled)
                    or (not self.training and self.config.act_inference)
            )

            if use_adaptive:
                # Halt signal based on Q-values (but always halt at max steps)
                q_halt_signal = q_halt_logits > q_continue_logits
                halted = halted | q_halt_signal

                # Store actual steps used for logging (only during inference)
                if not self.training:
                    outputs["actual_steps"] = new_steps.float()

                # Exploration (only during training)
                if self.training:
                    min_halt_steps = (
                                             torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob
                                     ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                    halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q (only during training)
                # NOTE: No replay buffer and target networks for computing target Q-value.
                # As batch_size is large, there're many parallel envs.
                # Similar concept as PQN https://arxiv.org/abs/2407.04811
                if self.training and compute_target_q:
                    next_q_halt_logits, next_q_continue_logits = self.inner(
                        new_inner_carry, new_current_data
                    )[-1]

                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return HierarchicalReasoningModel_ACTV2DropoutCarry(
            new_inner_carry, new_steps, halted, new_current_data
        ), outputs
