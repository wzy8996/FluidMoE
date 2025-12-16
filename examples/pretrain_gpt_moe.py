#!/usr/bin/env python3
"""
FluidMoE GPT Pretraining Example

This example shows how to pretrain a GPT-style MoE model with FluidMoE.

Key Features:
- Uses FluidMoE custom layers (FluidSelfAttention + FluidMoELayer)
- Automatic optimizer wrapping via fluid.pretrain()
- Attention layers use Ulysses Sequence Parallel (SP)
- Expert layers use Expert Parallel (EP)
- dW computation overlaps with AllToAll communication

Usage:
    python examples/pretrain_gpt_moe.py \\
        --num-layers 2 \\
        --hidden-size 512 \\
        --num-attention-heads 8 \\
        --num-experts 4 \\
        --moe-router-topk 2 \\
        --context-parallel-size 2 \\
        --expert-model-parallel-size 2 \\
        --tensor-model-parallel-size 1 \\
        --micro-batch-size 1 \\
        --global-batch-size 2 \\
        --train-iters 10
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def model_provider(pre_process=True, post_process=True, **kwargs):
    """
    Build MoE model using Fluid layer spec.

    This is the key function where we specify to use FluidMoE optimization.

    Note: **kwargs is used to handle any additional arguments from Megatron.
    """
    from megatron.training import get_args
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.transformer import TransformerConfig
    from fluid import get_fluid_custom_layers

    args = get_args()

    print("\n[FluidMoE] Building model with Layer Spec...")
    print(f"  - Layers: {args.num_layers}")
    print(f"  - Hidden: {args.hidden_size}")
    print(f"  - Experts: {getattr(args, 'num_experts', 'N/A')}")
    print(f"  - SP (context_parallel): {args.context_parallel_size}")
    print(f"  - EP (expert_parallel): {args.expert_model_parallel_size}")
    print(f"  - TP (tensor_parallel): {args.tensor_model_parallel_size}")

    # Create transformer config
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,

        # MoE specific
        num_moe_experts=getattr(args, 'num_experts', None),
        moe_router_topk=getattr(args, 'moe_router_topk', 2),
        moe_grouped_gemm=getattr(args, 'moe_grouped_gemm', False),

        # Parallelism
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        context_parallel_size=args.context_parallel_size,  # SP for attention
        expert_model_parallel_size=args.expert_model_parallel_size,  # EP for MoE
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,

        # Sequence parallel (only enable if TP > 1, as required by Megatron)
        sequence_parallel=(args.tensor_model_parallel_size > 1),

        # Other configs
        use_cpu_initialization=args.use_cpu_initialization,
        perform_initialization=True,
        add_bias_linear=args.add_bias_linear,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    # ============================================================
    # Key: Get Fluid custom layers (complete custom implementation)
    # ============================================================
    layer_spec = get_fluid_custom_layers()

    # Build model with Fluid spec
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,  # 👈 Use Fluid spec
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
    )

    print("\n[FluidMoE] ✅ Model created with Fluid optimization!")

    # Print layer info
    from fluid import print_fluid_layer_info
    print_fluid_layer_info(model)

    return model


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    # Import Megatron's dataset builder
    try:
        from pretrain_gpt import train_valid_test_datasets_provider as default_provider
        return default_provider(train_val_test_num_samples)
    except ImportError:
        # Fallback: return dummy datasets
        print("[Warning] Using dummy datasets (pretrain_gpt not found)")
        return None, None, None


def forward_step_func(data_iterator, model):
    """Forward step function for GPT model"""
    import torch
    from megatron.training import get_args
    from megatron.core import mpu

    args = get_args()

    # Get batch and move to GPU
    batch = next(data_iterator)
    tokens = batch['tokens'].cuda(non_blocking=True)
    labels = batch['labels'].cuda(non_blocking=True)
    loss_mask = batch['loss_mask'].cuda(non_blocking=True)
    attention_mask = batch.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.cuda(non_blocking=True)
    position_ids = batch.get('position_ids', None)
    if position_ids is not None:
        position_ids = position_ids.cuda(non_blocking=True)

    # Forward pass
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    # Loss function
    def loss_func(output_tensor):
        losses = output_tensor.float()
        loss_mask_flat = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask_flat) / loss_mask_flat.sum()

        # Return loss (Megatron's training loop will handle averaging)
        return loss, {'lm loss': loss}

    return output_tensor, loss_func


if __name__ == "__main__":
    from megatron.core.enums import ModelType
    from fluid.pretrain import pretrain

    print("\n" + "="*60)
    print("FluidMoE GPT Pretraining")
    print("="*60 + "\n")

    # Run FluidMoE pretraining (automatically wraps optimizer)
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step_func=forward_step_func,
        # args_defaults={
        #     'tokenizer_type': 'GPT2BPETokenizer',
        # }
    )

    # Print final statistics
    print("\n" + "="*60)
    print("Training completed! FluidMoE statistics:")
    print("="*60)

    import fluid
    fluid.print_status()

    print("\n[FluidMoE] Training session finished successfully! 🚀\n")
