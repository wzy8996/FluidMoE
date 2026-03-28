"""
FluidMoE training via Megatron's pretrain() entry point.

Uses Megatron's full training infrastructure (distributed optimizer, mixed precision,
gradient clipping, lr scheduling, checkpointing) with FluidMoE's P2P-based MoE scheduling.

Usage:
    torchrun --nproc_per_node=2 examples/pretrain_fluidmoe.py \
        --num-layers 2 --hidden-size 4096 --ffn-hidden-size 14336 \
        --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
        --num-experts 8 --moe-router-topk 2 \
        --seq-length 2048 --micro-batch-size 2 --global-batch-size 4 \
        --train-iters 100 --lr 1e-4 --min-lr 1e-5 \
        --context-parallel-size 2 --expert-model-parallel-size 2 \
        --bf16 --no-bias-linear \
        --mock-data --tokenizer-type NullTokenizer --vocab-size 50257
"""

import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
megatron_path = os.environ.get('MEGATRON_PATH', '/home/zju/wzy/Megatron-LM')
if megatron_path not in sys.path:
    sys.path.insert(0, megatron_path)

import torch
from megatron.training import get_args, pretrain
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer

from fluid.model import FluidMoEModel
from fluid.setup import setup_model_and_optimizer as fluid_setup_model_and_optimizer
from wikitext_dataset import WikiTextDataset


# ── Model Provider ────────────────────────────────────────────────────

def add_dataset_args(parser):
    group = parser.add_argument_group(title="fluidmoe dataset")
    group.add_argument(
        "--dataset-source",
        choices=("mock", "wikitext"),
        default=os.environ.get("FLUIDMOE_DATASET_SOURCE", "mock"),
        help="Choose mock tokens or cached WikiText-103 samples for training.",
    )
    return parser

def model_provider(pre_process=True, post_process=True, **kwargs):
    """Build FluidMoEModel for Megatron's pretrain()."""
    args = get_args()

    from megatron.core.transformer.transformer_config import TransformerConfig
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=getattr(args, 'num_query_groups', args.num_attention_heads),
        ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=args.num_experts,
        moe_router_topk=getattr(args, 'moe_router_topk', 2),
        use_cpu_initialization=args.use_cpu_initialization,
        pipeline_dtype=args.params_dtype,
        params_dtype=args.params_dtype,
        hidden_dropout=args.hidden_dropout,
        attention_dropout=args.attention_dropout,
        add_bias_linear=getattr(args, 'add_bias_linear', True),
        bf16=args.bf16,
        fp16=args.fp16,
    )

    model = FluidMoEModel(
        config=config,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    return model


# ── Forward Step ──────────────────────────────────────────────────────

def loss_func(loss_mask, output_tensor, model=None):
    """Standard cross-entropy loss (same as Megatron's pretrain_gpt.py)."""
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
    return loss, num_tokens, report


def get_batch(data_iterator):
    """Get a batch from the data iterator, handling CP and TP slicing."""
    from megatron.training.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)
    return batch.values()


def forward_step(data_iterator, model, return_schedule_plan=False):
    """Forward step for FluidMoE model."""
    from megatron.training import get_timers
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


# ── Dataset Provider ──────────────────────────────────────────────────

def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    args = get_args()

    if args.dataset_source == "mock":
        config = GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            tokenizer=_NullTokenizer(vocab_size=args.vocab_size),
            mid_level_dataset_surplus=0.005,
        )
        return BlendedMegatronDatasetBuilder(
            MockGPTDataset, train_val_test_num_samples, lambda: True, config
        ).build()

    if args.dataset_source == "wikitext":
        if args.vocab_size != 50257:
            raise ValueError(
                "--dataset-source wikitext expects --vocab-size 50257 because it uses the GPT-2 tokenizer."
            )
        train_ds = WikiTextDataset(seq_len=args.seq_length, split="train")
        # Skip validation/test cache build when evaluation is disabled.
        if args.eval_iters == 0:
            return train_ds, None, None
        return (
            train_ds,
            WikiTextDataset(seq_len=args.seq_length, split="validation"),
            WikiTextDataset(seq_len=args.seq_length, split="test"),
        )

    raise ValueError(f"Unsupported dataset source: {args.dataset_source}")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    # Replace Megatron's setup with FluidMoE's (adds scheduler wrapping)
    import megatron.training.training as _mt
    _mt.setup_model_and_optimizer = fluid_setup_model_and_optimizer

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_dataset_args,
        args_defaults={'tokenizer_type': 'NullTokenizer', 'vocab_size': 50257},
    )
