"""
DeepSpeed-MoE end-to-end training entry point for FluidMoE paper §5.6.

Architecture: full DeepSpeed stack (DeepSpeed Ulysses attention + DeepSpeed MoE,
no Transformer Engine, no RoPE, no RMSNorm). The transformer body is the same
DeepSpeedBlockBaselineTransformerModel used by the block test, so block and
E2E share the same model definition.

Infrastructure: shared with pretrain_megatron.py / pretrain_fluidmoe.py via
Megatron utilities (data loader, LR scheduler, log format), not because
they are "DeepSpeed modules" but because they are framework-agnostic
plumbing. The actual training loop is hand-rolled (DeepSpeed engine's
forward/backward/step is incompatible with Megatron's train_step contract,
which expects optimizer.step() → (success, grad_norm, num_zeros)).
"""

import math
import os
import sys
import time

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, "tools"))
sys.path.insert(0, os.path.join(proj_root, "examples"))

import torch
import torch._dynamo
torch._dynamo.config.disable = True
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import deepspeed
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer

from deepspeed_ulysses_baseline import DeepSpeedBlockBaselineTransformerModel, DeepSpeedBlockBaselineLayer


# ---------------------------------------------------------------------------
# CLI extension (mirrors pretrain_fluidmoe.py)
# ---------------------------------------------------------------------------

def add_dataset_args(parser):
    group = parser.add_argument_group(title="fluidmoe dataset")
    group.add_argument(
        "--dataset-source",
        choices=("mock", "wikitext"),
        default=os.environ.get("FLUIDMOE_DATASET_SOURCE", "mock"),
        help="Choose mock tokens or cached WikiText-103 samples for training.",
    )
    return parser


# ---------------------------------------------------------------------------
# DeepSpeed-stack GPT outer wrapper
# ---------------------------------------------------------------------------

class DeepSpeedGPTModel(nn.Module):
    """Outer GPT wrapper around the DeepSpeed-Ulysses + DeepSpeed-MoE block.

    Architecture (matches block test + structurally aligned with Megatron GPTModel):
      - learned token + position embeddings (matches Megatron default
        position_embedding_type='learned_absolute')
      - DeepSpeedBlockBaselineTransformerModel
          - LayerNorm + QKV proj + DeepSpeed DistributedAttention (Ulysses) + out proj
          - LayerNorm + DeepSpeed MoE (top-k routed)
      - final LayerNorm
      - **untied LM head** (separate nn.Linear; matches Megatron default
        share_embeddings_and_output_weights=False — Mixtral / Llama / DBRX style)
    """

    def __init__(self, vocab_size, max_seq_len, num_layers, hidden_size,
                 num_heads, num_kv_heads, ffn_hidden_size,
                 num_experts, top_k, capacity_factor,
                 cp_group, ep_group, dtype, device):
        super().__init__()
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, hidden_size, dtype=dtype, device=device)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size, dtype=dtype, device=device)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

        self.transformer = DeepSpeedBlockBaselineTransformerModel(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            ffn_hidden_size=ffn_hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            cp_group=cp_group,
            ep_group=ep_group,
            capacity_factor=capacity_factor,
            dtype=dtype,
            device=device,
            wire_ep_group=False,  # let deepspeed.initialize() set ep_group
        )

        self.ln_f = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype, device=device)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, tokens, position_ids):
        # tokens, position_ids: [B, S_local]
        x = self.tok_emb(tokens) + self.pos_emb(position_ids)         # [B, S, H]
        x = x.permute(1, 0, 2).contiguous()                           # [S, B, H]
        x = self.transformer(x)                                       # [S, B, H]
        x = x.permute(1, 0, 2).contiguous()                           # [B, S, H]
        x = self.ln_f(x)
        return self.lm_head(x)                                         # [B, S, V] untied


# ---------------------------------------------------------------------------
# Dataset provider — copied verbatim from pretrain_fluidmoe.py
# ---------------------------------------------------------------------------

def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()

    if args.dataset_source == "mock":
        config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            tokenizer=build_tokenizer(args),
            mid_level_dataset_surplus=0.005,
        )
        return BlendedMegatronDatasetBuilder(
            MockGPTDataset, train_val_test_num_samples, lambda: True, config
        ).build()

    if args.dataset_source == "wikitext":
        from wikitext_dataset import WikiTextDataset
        if args.vocab_size != 50257:
            raise ValueError(
                "--dataset-source wikitext expects --vocab-size 50257 (GPT-2 tokenizer)."
            )
        train_ds = WikiTextDataset(seq_len=args.seq_length, split="train")
        if args.eval_iters == 0:
            return train_ds, None, None
        return (
            train_ds,
            WikiTextDataset(seq_len=args.seq_length, split="validation"),
            WikiTextDataset(seq_len=args.seq_length, split="test"),
        )

    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")


train_valid_test_datasets_provider.is_distributed = True


# ---------------------------------------------------------------------------
# Batch utility — copied from pretrain_fluidmoe.py
# ---------------------------------------------------------------------------

def _get_batch_on_this_cp_rank_contiguous(batch):
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return batch
    cp_rank = parallel_state.get_context_parallel_rank()
    for key, val in batch.items():
        if val is None:
            continue
        seq_dim = 1 if key != 'attention_mask' else 2
        if val.ndim <= seq_dim:
            continue
        seq = val.shape[seq_dim]
        if seq % cp_size != 0:
            continue
        chunk = seq // cp_size
        slc = [slice(None)] * val.ndim
        slc[seq_dim] = slice(cp_rank * chunk, (cp_rank + 1) * chunk)
        batch[key] = val[tuple(slc)].contiguous()
    return batch


def _get_batch(data_iterator):
    from megatron.training.utils import get_batch_on_this_tp_rank
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = _get_batch_on_this_cp_rank_contiguous(batch)
    batch.pop('packed_seq_params', None)
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    initialize_megatron(
        extra_args_provider=add_dataset_args,
        args_defaults={'tokenizer_type': 'NullTokenizer', 'vocab_size': 50257},
    )
    args = get_args()
    # Megatron's pretrain() normally seeds these before the data iterator is
    # built (training.py:1233/1269). Since we bypass pretrain(), set them here.
    if not hasattr(args, 'iteration') or args.iteration is None:
        args.iteration = 0
    if not hasattr(args, 'consumed_train_samples') or args.consumed_train_samples is None:
        args.consumed_train_samples = 0
    if not hasattr(args, 'consumed_valid_samples') or args.consumed_valid_samples is None:
        args.consumed_valid_samples = 0

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    cp_group = parallel_state.get_context_parallel_group()
    ep_group = parallel_state.get_expert_model_parallel_group()
    cp_size = dist.get_world_size(group=cp_group)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # 1. Build the DeepSpeed-stack GPT model (DS Ulysses + DS MoE).
    model = DeepSpeedGPTModel(
        vocab_size=args.padded_vocab_size,
        max_seq_len=args.max_position_embeddings,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_heads=args.num_attention_heads,
        num_kv_heads=getattr(args, 'num_query_groups', args.num_attention_heads),
        ffn_hidden_size=args.ffn_hidden_size,
        num_experts=args.num_experts,
        top_k=getattr(args, 'moe_router_topk', 2),
        capacity_factor=1.0,
        cp_group=cp_group,
        ep_group=ep_group,
        dtype=dtype,
        device=device,
    )

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[pretrain_deepspeed] world={world_size} cp={cp_size} "
              f"n_params={n_params/1e6:.1f}M  arch=DS-Ulysses+DS-MoE", flush=True)

    # 2. MoE-aware param grouping (DS-MoE expects expert/non-expert split).
    base_groups = [{"name": "default",
                    "params": [p for p in model.parameters() if p.requires_grad]}]
    grouped = split_params_into_different_moe_groups_for_optimizer(base_groups)
    weight_decay = getattr(args, 'weight_decay', 0.1)
    optimizer = torch.optim.AdamW(
        grouped, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        weight_decay=weight_decay,
    )

    # 3. DeepSpeed engine — ZeRO-0 (aligned with Megatron / FluidMoE; no
    # distributed-optimizer in any of the three). LR scheduler also lives
    # inside the engine via DS-native WarmupCosineLR; engine.step() advances
    # it automatically. Mapping from Megatron flags:
    #   total_num_steps   = args.lr_decay_iters       (cos decay endpoint)
    #   warmup_num_steps  = args.lr_warmup_iters
    #   warmup_min_ratio  = 0.0                        (matches Megatron lr_warmup_init=0)
    #   cos_min_ratio     = args.min_lr / args.lr      (final LR floor)
    #   warmup_type       = 'linear'                   (Megatron uses linear warmup)
    ds_train_batch_size = args.micro_batch_size * world_size
    ds_config = {
        "train_batch_size": ds_train_batch_size,
        "train_micro_batch_size_per_gpu": args.micro_batch_size,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": bool(args.bf16)},
        "fp16": {"enabled": bool(args.fp16)},
        "zero_optimization": {"stage": 0},
        # Force fp32 AR to match Megatron's default behavior. Megatron auto-sets
        # accumulate_allreduce_grads_in_fp32=True when main_grads_dtype='fp32'
        # (default), so its DDP grad buffer + AR run in fp32. FluidMoE matches
        # this explicitly. DeepSpeed defaults `communication_data_type` to bf16
        # when bf16 is enabled (engine.py:1138-1145 + 2965-2990); without this
        # override it would AR in bf16, halving the AR comm volume vs Megatron
        # / FluidMoE and biasing §5.6 step time in DS's favor.
        "communication_data_type": "fp32",
        "gradient_clipping": getattr(args, 'clip_grad', 1.0),
        "wall_clock_breakdown": False,
        "steps_per_print": 10**9,
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.lr_decay_iters,
                "warmup_num_steps": max(1, args.lr_warmup_iters),
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": (args.min_lr / args.lr) if args.lr > 0 else 0.0,
                "warmup_type": "linear",
            },
        },
    }
    engine, _, _, lr_scheduler = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config,
    )

    # 5. Data iterator — Megatron's pipeline (mock GPTDataset or WikiText).
    train_iter, _, _ = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )

    aux_loss_coeff = getattr(args, 'moe_aux_loss_coeff', 0.0) or 0.0

    # 6. Training loop — manual; mirrors Megatron train_step structure but
    # routes through engine.backward / engine.step.
    for step in range(args.train_iters):
        batch = _get_batch(train_iter)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        position_ids = batch['position_ids']

        torch.cuda.synchronize()
        t0 = time.time()

        logits = engine(tokens, position_ids)                          # [B, S_local, V]

        flat_logits = logits.float().view(-1, args.padded_vocab_size)
        flat_labels = labels.view(-1)
        flat_mask = loss_mask.view(-1).float()
        per_tok = F.cross_entropy(flat_logits, flat_labels, reduction='none')
        loss = (per_tok * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)

        # Aggregate DeepSpeed-MoE per-layer aux loss (mirrors Megatron's aux_loss
        # behavior; coefficient from args.moe_aux_loss_coeff).
        aux_terms = []
        for module in engine.module.modules():
            if isinstance(module, DeepSpeedBlockBaselineLayer) \
                    and getattr(module, '_last_l_aux', None) is not None:
                aux_terms.append(module._last_l_aux)
        if aux_terms and aux_loss_coeff > 0:
            loss = loss + aux_loss_coeff * torch.stack(aux_terms).sum()

        # Reported loss: average across CP for display. Backward already
        # CP-correct via Ulysses.
        loss_report = loss.detach().clone()
        if cp_size > 1:
            dist.all_reduce(loss_report, op=dist.ReduceOp.AVG, group=cp_group)

        engine.backward(loss)
        engine.step()
        # NB: engine.step() already advances lr_scheduler internally.

        torch.cuda.synchronize()
        elapsed_ms = (time.time() - t0) * 1000.0

        gn = engine.get_global_grad_norm()
        if gn is None or (isinstance(gn, float) and (math.isnan(gn) or math.isinf(gn))):
            gn = 0.0

        if rank == 0 and (step + 1) % args.log_interval == 0:
            print(
                f" iteration {step + 1:>6d}/{args.train_iters:>6d} | "
                f"consumed samples: {(step + 1) * args.global_batch_size:>10d} | "
                f"elapsed time per iteration (ms): {elapsed_ms:.1f} | "
                f"learning rate: {optimizer.param_groups[0]['lr']:.3E} | "
                f"global batch size: {args.global_batch_size:>5d} | "
                f"lm loss: {loss_report.item():.6E} | "
                f"grad norm: {float(gn):.3f}",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
