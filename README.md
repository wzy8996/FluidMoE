# FluidMoE

Computation-Communication Overlap for MoE and Attention Layers.

## Features

- **Forward P2P Overlap**: Round-Robin Tournament scheduling for Dispatch/Combine (MoE) and SP2HP/HP2SP (Attention)
- **Backward dW Scheduling**: Weight gradient computation overlaps with AllToAll communication
- **Chunked dX Backward**: Pipelined input gradient computation with AllToAll
- **Standalone Implementation**: No Megatron dependency for core functionality
- **Megatron Integration**: Optional Layer Spec for seamless Megatron-LM integration

## Benchmark Results (2 GPUs)

| Mode | Baseline | Overlap | Speedup |
|------|----------|---------|---------|
| Inference | 136.55 ms | 106.31 ms | **1.28x (+28%)** |
| Forward | 136.45 ms | 113.21 ms | **1.21x (+21%)** |
| Training | 345.36 ms | 304.34 ms | **1.13x (+13%)** |

## Quick Start

```bash
# Install
pip install torch  # requires CUDA support
git clone https://github.com/your-repo/FluidMoE.git
cd FluidMoE
export PYTHONPATH=$PWD:$PYTHONPATH

# Run tests
torchrun --nproc_per_node=2 tests/test_correctness.py
torchrun --nproc_per_node=2 tests/test_transformer_speedup.py
```

## Usage

### Standalone (No Megatron)

```python
import torch
import torch.distributed as dist
from fluid import (
    MoEBaseline,                      # Baseline MoE with AllToAll
    moe_multicard_p2p_overlap_forward, # P2P overlap forward
    MultiCardOverlapContext,
    get_backward_scheduler,
)

# Initialize
dist.init_process_group('nccl')
device = torch.device(f'cuda:{dist.get_rank()}')
ep_group = dist.group.WORLD

# Create overlap context
overlap_ctx = MultiCardOverlapContext(device, ep_size=dist.get_world_size())

# Forward with P2P overlap
output = moe_multicard_p2p_overlap_forward(
    tokens, input_splits, output_splits,
    weight1, weight2, ep_group, torch.nn.functional.gelu,
    overlap_ctx, layer_id=0,
    num_local_experts, tokens_per_expert, num_global_tokens_per_local_expert
)

# Training: enable scheduler for dW overlap during backward
scheduler = get_backward_scheduler()
scheduler.enable()
loss.backward()
scheduler.finish_batch()
```

### With Megatron-LM

```python
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel
from fluid import get_fluid_custom_layers

config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_moe_experts=8,
    moe_router_topk=2,
    expert_model_parallel_size=2,
)

layer_spec = get_fluid_custom_layers()
model = GPTModel(config, transformer_layer_spec=layer_spec)
```

## Architecture

```
FluidMoE/
├── fluid/
│   ├── __init__.py              # Public API
│   ├── core/
│   │   ├── alltoall.py          # AllToAll primitives
│   │   ├── forward_comm.py      # P2P scheduling & overlap context
│   │   ├── scheduler.py         # Backward dW scheduler
│   │   └── utils.py             # Activation gradients, utilities
│   ├── moe/
│   │   ├── baseline.py          # MoE with standard AllToAll
│   │   ├── p2p_overlap.py       # MoE with P2P overlap
│   │   ├── chunked_backward.py  # Chunked dX backward
│   │   └── router.py            # Top-K routing with dW scheduling
│   ├── attention/
│   │   ├── baseline.py          # Attention with AllToAll (Ulysses SP)
│   │   ├── p2p_overlap.py       # Attention with P2P overlap
│   │   └── chunked_backward.py  # Chunked output projection backward
│   └── layers/
│       ├── transformer.py       # TransformerLayer (standalone)
│       └── megatron_integration.py  # Megatron Layer Spec
├── tests/
│   ├── test_correctness.py      # Correctness verification
│   └── test_transformer_speedup.py  # Speedup benchmark
└── examples/
    └── pretrain_gpt_moe.py      # Megatron pretraining example
```

## How It Works

### Forward P2P Overlap (Round-Robin Tournament)

```
Standard AllToAll:
  GPU0: [Compute] → [AllToAll] → [Compute]
  GPU1: [Compute] → [AllToAll] → [Compute]
              ↑ All GPUs blocked

FluidMoE P2P Overlap (4 GPUs example):
  Round 0: P2P(0↔3), P2P(1↔2) || Local computation
  Round 1: P2P(0↔2), P2P(1↔3) || Compute recv from Round 0
  Round 2: P2P(0↔1), P2P(2↔3) || Compute recv from Round 1
  Final:   Wait Round 2       || Compute recv from Round 2

Benefit: Communication overlaps with computation
```

### Backward dW Overlap

```
Standard (sequential):
  GPU: [dX + dW] → [AllToAll]
  Time: ├────────┤  ├────────┤

FluidMoE (overlap):
  GPU: [dX] → [AllToAll || dW]
  Time: ├───┤  ├──────────────┤

Benefit: dW computation hidden behind AllToAll
```

## Multi-GPU Testing

```bash
# 2 GPUs
torchrun --nproc_per_node=2 tests/test_transformer_speedup.py

# 4 GPUs
torchrun --nproc_per_node=4 tests/test_transformer_speedup.py

# 8 GPUs
torchrun --nproc_per_node=8 tests/test_transformer_speedup.py

# Multi-node (Node 0)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
  --master_addr=<MASTER_IP> --master_port=29500 \
  tests/test_transformer_speedup.py
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8
- NCCL >= 2.18
- (Optional) Megatron-Core >= 0.5.0

## License

Apache 2.0
