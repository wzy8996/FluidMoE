"""
FluidMoE training via Megatron's pretrain() entry point.

Uses Megatron's full training infrastructure (distributed optimizer, mixed precision,
gradient clipping, lr scheduling, checkpointing) with FluidMoE's P2P-based MoE scheduling.
"""

import os
import sys
from functools import partial

proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)

import torch
from megatron.training import get_args, pretrain
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig

from fluid.layer.fluid_spec import get_fluid_gpt_layer_spec
from fluid.setup import setup_model_and_optimizer as fluid_setup_model_and_optimizer
from wikitext_dataset import WikiTextDataset


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
    args = get_args()

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_query_groups=getattr(args, 'num_query_groups', args.num_attention_heads),
        ffn_hidden_size=args.ffn_hidden_size,
        num_moe_experts=args.num_experts,
        moe_router_topk=getattr(args, 'moe_router_topk', 2),
        moe_token_dispatcher_type="alltoall",
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_router_dtype='fp32',
        moe_grouped_gemm=True,
        moe_permute_fusion=True,
        moe_expert_capacity_factor=1.0,
        moe_pad_expert_input_to_capacity=True,
        use_cpu_initialization=args.use_cpu_initialization,
        pipeline_dtype=args.params_dtype,
        params_dtype=args.params_dtype,
        hidden_dropout=args.hidden_dropout,
        attention_dropout=args.attention_dropout,
        add_bias_linear=getattr(args, 'add_bias_linear', True),
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        transformer_impl="transformer_engine",
        bf16=args.bf16,
        fp16=args.fp16,
        cp_comm_type="a2a",
    )

    transformer_layer_spec = get_fluid_gpt_layer_spec()

    return GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )


def loss_func(loss_mask, output_tensor, model=None):
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
    return loss, num_tokens, report


def _get_batch_on_this_cp_rank_contiguous(batch):
    """Contiguous CP split: rank r takes seq[r*chunk : (r+1)*chunk]."""
    from megatron.core import parallel_state
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


def get_batch(data_iterator):
    from megatron.training.utils import get_batch_on_this_tp_rank
    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = _get_batch_on_this_cp_rank_contiguous(batch)
    packed_seq_params = batch.pop('packed_seq_params', None)

    tokens = batch['tokens']
    labels = batch['labels']
    loss_mask = batch['loss_mask']
    attention_mask = batch.get('attention_mask', None)
    position_ids = batch['position_ids']

    return tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params


def forward_step(data_iterator, model, return_schedule_plan=False):
    from megatron.training import get_timers
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )

    return output_tensor, partial(loss_func, loss_mask)


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
        if args.vocab_size != 50257:
            raise ValueError(
                "--dataset-source wikitext expects --vocab-size 50257 because it uses the GPT-2 tokenizer."
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


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    import megatron.training.training as _mt
    _mt.setup_model_and_optimizer = fluid_setup_model_and_optimizer

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_dataset_args,
        args_defaults={
            'tokenizer_type': 'NullTokenizer',
            'vocab_size': 50257,
        },
    )
