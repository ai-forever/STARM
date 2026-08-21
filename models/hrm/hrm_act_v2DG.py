import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field
from torch import nn

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLUD, ConvSwiGLUD, AttentionD, RotaryEmbedding, CosSin, CastedEmbedding, \
    CastedLinear, _find_multiple
from models.sparse_embedding import CastedSparseEmbedding


@dataclass
class HierarchicalReasoningModel_ACTV2DGInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

    embed_dropout_mask: Optional[torch.Tensor] = None  # [B, S, hidden]
    H_dropout_mask: Optional[torch.Tensor] = None  # [B, 1, hidden]
    L_dropout_mask: Optional[torch.Tensor] = None  # [B, 1, hidden]

    H_qkv_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)
    H_attn_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)  # separate mask for attn
    H_mlp_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)  # separate mask for mlp
    H_ffn_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)

    L_qkv_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)
    L_attn_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)
    L_mlp_residual_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)
    L_ffn_dropout_masks: List[Optional[torch.Tensor]] = field(default_factory=list)

    def with_updated_states(self, z_H: torch.Tensor,
                            z_L: torch.Tensor) -> 'HierarchicalReasoningModel_ACTV2DGInnerCarry':
        return HierarchicalReasoningModel_ACTV2DGInnerCarry(
            z_H=z_H,
            z_L=z_L,
            embed_dropout_mask=self.embed_dropout_mask,
            H_dropout_mask=self.H_dropout_mask,
            L_dropout_mask=self.L_dropout_mask,

            H_qkv_dropout_masks=self.H_qkv_dropout_masks,
            H_attn_residual_dropout_masks=self.H_attn_residual_dropout_masks,
            H_mlp_residual_dropout_masks=self.H_mlp_residual_dropout_masks,
            H_ffn_dropout_masks=self.H_ffn_dropout_masks,
            L_qkv_dropout_masks=self.L_qkv_dropout_masks,
            L_attn_residual_dropout_masks=self.L_attn_residual_dropout_masks,
            L_mlp_residual_dropout_masks=self.L_mlp_residual_dropout_masks,
            L_ffn_dropout_masks=self.L_ffn_dropout_masks,
        )


@dataclass
class HierarchicalReasoningModel_ACTV2DGCarry:
    inner_carry: HierarchicalReasoningModel_ACTV2DGInnerCarry

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV2DGConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int
    L_layers: int

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

    forward_dtype: str = "bfloat16"

    embed_dropout_p: float = 0.0
    qkv_dropout_p: float = 0.0
    residual_dropout_p: float = 0.0
    ffn_dropout_p: float = 0.0

    H_dropout_p: float = 0.0
    L_dropout_p: float = 0.0

    grad_through_last_H: int = 1
    grad_through_last_L: int = 1

    use_conv: bool = False

    drop_H_transformer: bool = False

    noise_intensity: float = 0.0

    embed_dim: Optional[int] = None

    tied_embeddings: bool = False

    uplift: int = 0

    no_ACT_continue: bool = False

    L_gate: bool = False

    L_trust_region: float = 0.0

    norm_after_L_step: bool = False

    randomize_H_depth: bool = False

    use_spectral_encoding: bool = False
    token_groups_file: Optional[str] = None
    spectral_frequencies: Optional[List[float]] = None
    token_groups: Optional[List[List[int]]] = Field(default=None, exclude=True)

    @property
    def effective_embed_dim(self) -> int:
        return self.embed_dim if self.embed_dim is not None else self.hidden_size

    @property
    def ffn_inter_size(self) -> int:
        return _find_multiple(round(self.expansion * self.hidden_size * 2 / 3), 256)

    @property
    def qkv_proj_size(self) -> int:
        return (self.num_heads + 2 * self.num_heads) * (self.hidden_size // self.num_heads)


class HierarchicalReasoningModel_ACTV2DGBlock(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2DGConfig) -> None:
        super().__init__()

        self.self_attn = AttentionD(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        if not config.use_conv:
            self.mlp = SwiGLUD(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        else:
            self.mlp = ConvSwiGLUD(
                hidden_size=config.hidden_size,
                expansion=config.expansion,
            )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor,
                qkv_dropout_mask: Optional[torch.Tensor] = None,
                attn_residual_dropout_mask: Optional[torch.Tensor] = None,  # separate mask
                mlp_residual_dropout_mask: Optional[torch.Tensor] = None,
                ffn_dropout_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

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


class HierarchicalReasoningModel_ACTV2DReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV2DGBlock]):
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


class TiedLmHead(nn.Module):
    """
    Linear layer for language modelling that uses tied weights from embedding
    matrix and optional projection. No trainable parameters of its own.
    """

    def __init__(self, embed_tokens: CastedEmbedding, embed_proj: Optional[nn.Module],
                 uplift: int = 0,
                 eff_dim: int = None,
                 hidden_size: int = None):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.embed_proj = embed_proj

        self.uplift = uplift

        if uplift > 0:
            self.uplift_emb = nn.Parameter(
                trunc_normal_init_(torch.empty((embed_tokens.embedding_weight.shape[0], uplift)),
                                   std=1.0 / (eff_dim ** 0.5))
            )
            self.uplift_proj = nn.Parameter(
                trunc_normal_init_(torch.empty((hidden_size, uplift)),
                                   std=1.0 / (hidden_size ** 0.5))
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb_weight = self.embed_tokens.embedding_weight
        if self.embed_proj is not None:
            proj_weight = self.embed_proj.weight
            base_weight = emb_weight @ proj_weight.T
        else:
            base_weight = emb_weight

        if self.uplift > 0:
            uplift_contrib = self.uplift_emb @ self.uplift_proj.T
            weight = base_weight + uplift_contrib
        else:
            weight = base_weight

        return F.linear(x, weight.to(x.dtype), bias=None)


class HierarchicalReasoningModel_ACTV2DG_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV2DGConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O
        self.embed_scale = math.sqrt(self.config.effective_embed_dim)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.effective_embed_dim,
                                            init_std=embed_init_std, cast_to=self.forward_dtype)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0,
                                                    cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size,
                                             init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            raise NotImplementedError()

        if self.config.effective_embed_dim != self.config.hidden_size:
            self.embed_proj = CastedLinear(
                self.config.effective_embed_dim,
                self.config.hidden_size,
                bias=False,
            )
        else:
            self.embed_proj = None

        if self.config.tied_embeddings:
            self.lm_head = TiedLmHead(self.embed_tokens, self.embed_proj, uplift=self.config.uplift,
                                      eff_dim=self.config.effective_embed_dim,
                                      hidden_size=self.config.hidden_size)
        else:
            self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # Reasoning Layers
        self.L_level = HierarchicalReasoningModel_ACTV2DReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV2DGBlock(self.config) for _i in range(self.config.L_layers)])

        if config.drop_H_transformer:
            self.H_level = self.L_level
        else:
            self.H_level = HierarchicalReasoningModel_ACTV2DReasoningModule(
                layers=[HierarchicalReasoningModel_ACTV2DGBlock(self.config) for _i in range(self.config.H_layers)])

        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

        if config.use_spectral_encoding:
            assert config.token_groups_file is not None, \
                "token_groups_file must be specified when use_spectral_encoding=True"

            groups_path = Path(config.token_groups_file)
            if not groups_path.is_absolute():
                groups_path = Path("config") / groups_path

            with open(groups_path, "r") as f:
                groups_data = yaml.safe_load(f)

            self.token_groups = groups_data["token_groups"]

            # Validation
            self._validate_token_groups()

            # Build lookup-tables
            self._vocab_to_groupt = torch.full((config.vocab_size,), -1, dtype=torch.long)
            self._vocab_to_elemt = torch.full((config.vocab_size,), -1, dtype=torch.long)

            self.register_buffer('_vocab_to_group', self._vocab_to_groupt)
            self.register_buffer('_vocab_to_elem', self._vocab_to_elemt)

            for gid, tokens in enumerate(self.token_groups):
                for elem_idx, tid in enumerate(tokens):
                    self._vocab_to_group[tid] = gid
                    self._vocab_to_elem[tid] = elem_idx

            # Group embeddings
            embed_init_std = 1.0 / math.sqrt(config.effective_embed_dim)
            self.group_embeddings = CastedEmbedding(
                len(self.token_groups),
                config.effective_embed_dim,
                init_std=embed_init_std,
                cast_to=self.forward_dtype
            )

            # Frequencies
            if config.spectral_frequencies and len(config.spectral_frequencies) > 0:
                self._spectral_freqs_cachedt = torch.tensor(
                    config.spectral_frequencies,
                    device=self.H_init.device,
                    dtype=self.forward_dtype
                )
            else:
                dim = config.hidden_size // 2
                inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 1,
                                                           device=self.H_init.device,
                                                           dtype=torch.float32) / dim))
                self._spectral_freqs_cachedt = inv_freq.to(self.forward_dtype)
            self.register_buffer('_spectral_freqs_cached', self._spectral_freqs_cachedt)

        else:
            self.token_groups = None
            self._vocab_to_group = None
            self._vocab_to_elem = None
            self.group_embeddings = None
            self._spectral_freqs_cached = None

        if config.L_gate:

            self.L_gate_proj = CastedLinear(
                config.hidden_size, 1, bias=True
            )

            with torch.no_grad():
                self.L_gate_proj.weight.mul_(0.1)
                if self.L_gate_proj.bias is not None:
                    self.L_gate_proj.bias.zero_()

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor,
                          embed_dropout_mask: Optional[torch.Tensor]):

        # Token embedding
        if self.config.use_spectral_encoding:
            group_ids = self._vocab_to_group[input]
            embedding = self.group_embeddings(group_ids.to(torch.int32))
        else:
            embedding = self.embed_tokens(input.to(torch.int32))

        if self.embed_proj is not None:
            embedding = self.embed_proj(embedding)

        if self.config.use_spectral_encoding:
            embedding = self._apply_spectral_encoding(embedding, input)

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)

            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                                  dim=-2)

        if embed_dropout_mask is not None:
            embedding = embedding * embed_dropout_mask

        # Position embeddings
        if self.config.pos_encodings == "learned":
            embedding = 1 / math.sqrt(2) * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def _validate_token_groups(self):
        all_tokens = set()
        for group in self.token_groups:
            all_tokens.update(group)

        total_in_groups = sum(len(g) for g in self.token_groups)
        assert len(all_tokens) == total_in_groups, \
            f"Token groups have duplicates! {total_in_groups} vs {len(all_tokens)}"

        expected_tokens = set(range(self.config.vocab_size))
        assert all_tokens == expected_tokens, \
            f"Token groups don't cover vocab! Missing: {expected_tokens - all_tokens}"

    def _apply_spectral_encoding(self, embeddings: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        B, S, H = embeddings.shape
        num_pairs = H // 2

        elem_indices = self._vocab_to_elem[token_ids].unsqueeze(-1).to(self.forward_dtype)

        pairs = embeddings.view(B, S, num_pairs, 2)

        base_freqs = self._spectral_freqs_cached
        if len(base_freqs) < num_pairs:
            repeats = (num_pairs + len(base_freqs) - 1) // len(base_freqs)
            freqs = base_freqs.repeat(repeats)[:num_pairs]
        else:
            freqs = base_freqs[:num_pairs]

        theta = elem_indices * freqs.unsqueeze(0).unsqueeze(0)

        # Cos/sin
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        x1, x2 = pairs[..., 0].clone(), pairs[..., 1].clone()
        pairs[..., 0] = x1 * cos_t - x2 * sin_t
        pairs[..., 1] = x1 * sin_t + x2 * cos_t

        return pairs.view(B, S, H)

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

        return HierarchicalReasoningModel_ACTV2DGInnerCarry(
            z_H=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            embed_dropout_mask=None,
            H_dropout_mask=None,
            L_dropout_mask=None,

            H_qkv_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_attn_residual_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_mlp_residual_dropout_masks=_empty_mask_list(self.config.H_layers),
            H_ffn_dropout_masks=_empty_mask_list(self.config.H_layers),
            L_qkv_dropout_masks=_empty_mask_list(self.config.L_layers),
            L_attn_residual_dropout_masks=_empty_mask_list(self.config.L_layers),
            L_mlp_residual_dropout_masks=_empty_mask_list(self.config.L_layers),
            L_ffn_dropout_masks=_empty_mask_list(self.config.L_layers),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV2DGInnerCarry):
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

        cand_H = self._maybe_generate_mask(self.config.H_dropout_p,
                                           (batch_size, 1, self.config.hidden_size),
                                           self.training)

        cand_L = self._maybe_generate_mask(self.config.L_dropout_p,
                                           (batch_size, 1, self.config.hidden_size),
                                           self.training)

        embed_mask = _update_mask(cand_embed, carry.embed_dropout_mask, ndim=3) if (
                cand_embed is not None or carry.embed_dropout_mask is not None) else None
        H_mask = _update_mask(cand_H, carry.H_dropout_mask, ndim=3) if (
                cand_H is not None or carry.H_dropout_mask is not None) else None
        L_mask = _update_mask(cand_L, carry.L_dropout_mask, ndim=3) if (
                cand_L is not None or carry.L_dropout_mask is not None) else None

        # H-level
        H_qkv = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.qkv_proj_size),
                                      self.config.qkv_dropout_p)
        H_attn_res = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.hidden_size),
                                           self.config.residual_dropout_p)
        H_mlp_res = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.hidden_size),
                                          self.config.residual_dropout_p)
        H_ffn = _generate_layer_masks(self.config.H_layers, (batch_size, seq_len, self.config.ffn_inter_size),
                                      self.config.ffn_dropout_p)

        # L-level
        L_qkv = _generate_layer_masks(self.config.L_layers, (batch_size, seq_len, self.config.qkv_proj_size),
                                      self.config.qkv_dropout_p)
        L_attn_res = _generate_layer_masks(self.config.L_layers, (batch_size, seq_len, self.config.hidden_size),
                                           self.config.residual_dropout_p)
        L_mlp_res = _generate_layer_masks(self.config.L_layers, (batch_size, seq_len, self.config.hidden_size),
                                          self.config.residual_dropout_p)
        L_ffn = _generate_layer_masks(self.config.L_layers, (batch_size, seq_len, self.config.ffn_inter_size),
                                      self.config.ffn_dropout_p)

        return HierarchicalReasoningModel_ACTV2DGInnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
            embed_dropout_mask=embed_mask,
            H_dropout_mask=H_mask,
            L_dropout_mask=L_mask,
            H_qkv_dropout_masks=_update_mask_list(H_qkv, carry.H_qkv_dropout_masks, ndim=3),
            H_attn_residual_dropout_masks=_update_mask_list(H_attn_res, carry.H_attn_residual_dropout_masks, ndim=3),
            H_mlp_residual_dropout_masks=_update_mask_list(H_mlp_res, carry.H_mlp_residual_dropout_masks, ndim=3),
            H_ffn_dropout_masks=_update_mask_list(H_ffn, carry.H_ffn_dropout_masks, ndim=3),
            L_qkv_dropout_masks=_update_mask_list(L_qkv, carry.L_qkv_dropout_masks, ndim=3),
            L_attn_residual_dropout_masks=_update_mask_list(L_attn_res, carry.L_attn_residual_dropout_masks, ndim=3),
            L_mlp_residual_dropout_masks=_update_mask_list(L_mlp_res, carry.L_mlp_residual_dropout_masks, ndim=3),
            L_ffn_dropout_masks=_update_mask_list(L_ffn, carry.L_ffn_dropout_masks, ndim=3),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2DGInnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[
        HierarchicalReasoningModel_ACTV2DGInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        def _noise_like(tensor):
            noise = torch.randn_like(tensor, dtype=self.forward_dtype)

            norm = (tensor.norm(dim=-1, keepdim=True, p=2)).detach()

            noise = noise * (self.config.noise_intensity * norm)
            return noise

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_identifiers"],
            carry.embed_dropout_mask
        )

        if self.config.noise_intensity > 0 and self.training:
            z_H = carry.z_H + _noise_like(carry.z_H)
            z_L = carry.z_L + _noise_like(carry.z_L)
            input_embeddings = input_embeddings + _noise_like(input_embeddings)
        else:
            z_H, z_L = carry.z_H, carry.z_L

        if self.config.randomize_H_depth and self.training:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    H_depth = torch.randint(1, self.config.H_cycles + 1, (1,)).item()
                else:
                    H_depth = None

                H_depth_list = [H_depth]
                dist.broadcast_object_list(H_depth_list, src=0)
                H_depth = H_depth_list[0]
            else:

                H_depth = torch.randint(1, self.config.H_cycles + 1, (1,)).item()
        else:
            H_depth = self.config.H_cycles

        # Unified loop with conditional gradient flow + dropout masks
        for _H_step in range(H_depth):
            h_enable_grad = _H_step >= self.config.H_cycles - self.config.grad_through_last_H

            for _L_step in range(self.config.L_cycles):
                enable_grad = h_enable_grad and (_L_step >= self.config.L_cycles - self.config.grad_through_last_L)

                l_kwargs = dict(
                    qkv_dropout_masks=carry.L_qkv_dropout_masks,
                    attn_residual_dropout_masks=carry.L_attn_residual_dropout_masks,
                    mlp_residual_dropout_masks=carry.L_mlp_residual_dropout_masks,
                    ffn_dropout_masks=carry.L_ffn_dropout_masks,
                    **seq_info
                )

                if enable_grad:
                    z_L_candidate = self.L_level(z_L, z_H + input_embeddings, **l_kwargs)
                    if carry.L_dropout_mask is not None:
                        z_L_candidate = z_L_candidate * carry.L_dropout_mask
                else:
                    with torch.no_grad():
                        z_L_candidate = self.L_level(z_L, z_H + input_embeddings, **l_kwargs)
                        if carry.L_dropout_mask is not None:
                            z_L_candidate = z_L_candidate * carry.L_dropout_mask

                if self.config.L_trust_region > 0:
                    delta = z_L_candidate - z_L

                    norm_z_L = z_L.norm(dim=-1, keepdim=True, p=2)
                    norm_delta = delta.norm(dim=-1, keepdim=True, p=2)

                    r = norm_delta / (norm_z_L + 1e-8)

                    scale = torch.clamp(r / self.config.L_trust_region, min=1.0)
                    scale = scale.detach()

                    delta = delta / scale
                    z_L_candidate = z_L + delta
                else:
                    pass

                if self.config.L_gate:
                    gate_input = z_H + z_L + input_embeddings  # [B, S, H]
                    alpha = torch.sigmoid(self.L_gate_proj(gate_input))  # [B, S, 1]
                    z_L = z_L + alpha * (z_L_candidate - z_L)
                else:
                    z_L = z_L_candidate

                if self.config.norm_after_L_step:
                    z_L = F.layer_norm(z_L, (self.config.hidden_size,))

            h_kwargs = dict(
                qkv_dropout_masks=carry.H_qkv_dropout_masks,
                attn_residual_dropout_masks=carry.H_attn_residual_dropout_masks,
                mlp_residual_dropout_masks=carry.H_mlp_residual_dropout_masks,
                ffn_dropout_masks=carry.H_ffn_dropout_masks,
                **seq_info
            )

            if h_enable_grad:
                z_H = self.H_level(z_H, z_L, **h_kwargs)
                if carry.H_dropout_mask is not None:
                    z_H = z_H * carry.H_dropout_mask
            else:
                with torch.no_grad():
                    z_H = self.H_level(z_H, z_L, **h_kwargs)
                    if carry.H_dropout_mask is not None:
                        z_H = z_H * carry.H_dropout_mask

        # LM Outputs
        new_carry = carry.with_updated_states(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]

        # Q head
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)

        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class HierarchicalReasoningModel_ACTV2DG(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV2DGConfig(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV2DG_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV2DGCarry(
            inner_carry=self.inner.empty_carry(batch_size),
            # Empty is expected, it will be reseted in first pass as all sequences are halted.

            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),  # Default to halted

            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV2DGCarry, batch: Dict[str, torch.Tensor],
                wo_q: bool = False, custom_max_steps: Optional[torch.Tensor] = None) -> Tuple[
        HierarchicalReasoningModel_ACTV2DGCarry, Dict[str, torch.Tensor]]:
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v
                            in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            threshold = (
                torch.tensor(self.config.halt_max_steps, device=new_steps.device)
                if custom_max_steps is None
                else custom_max_steps
            )

            is_last_step = new_steps >= threshold

            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1) and (not wo_q):
                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(
                    q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2,
                                                                                             high=self.config.halt_max_steps + 1)

                halted = halted & (new_steps >= min_halt_steps)
                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]

                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits,
                                                                             torch.maximum(next_q_halt_logits,
                                                                                           next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV2DGCarry(new_inner_carry, new_steps, halted, new_current_data), outputs
