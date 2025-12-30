# FluidMoE

Computation-Communication Overlap for Megatron-LM MoE Training.

## Features

- **Custom MoE Layer** - FluidSelfAttention and FluidMoELayer with full control over forward/backward
- **dW/dX Separation** - Backward dW computation overlaps with AllToAll communication
- **Fused Kernels** - CUTLASS GroupedGEMM with AllToAll overlap (experimental)
- **No Global Patching** - Direct Fluid AllToAll calls, no monkey patching
- **Megatron Compatible** - Uses Layer Spec mechanism for seamless integration

## Quick Start

```python
from megatron.core.transformer import TransformerConfig
from megatron.core.models.gpt import GPTModel
from fluid import get_fluid_custom_layers

# 1. Create config
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_moe_experts=8,
    moe_router_topk=2,
    expert_model_parallel_size=2,  # EP
)

# 2. Get Fluid layer spec
layer_spec = get_fluid_custom_layers()

# 3. Create model with Fluid optimization
model = GPTModel(config, transformer_layer_spec=layer_spec)
```

## Directory Structure

```
FluidMoE/
├── fluid/                      # Core implementation
│   ├── __init__.py            # Module entry (v0.8.0)
│   ├── scheduler.py           # dW task scheduler
│   ├── communication.py       # Fluid AllToAll primitives
│   ├── attention_module.py    # FluidSelfAttention
│   ├── moe_module.py          # FluidMoELayer
│   ├── moe_layers.py          # FluidGroupedMLP + Fused kernels
│   ├── attention_layers.py    # Fluid linear layers
│   ├── megatron_layers.py     # Layer Spec integration
│   ├── kernels/               # CUDA kernels (CUTLASS GroupedGEMM)
│   └── ops/                   # Compiled .so binaries
│
├── examples/                   # Usage examples
│   ├── pretrain_gpt_moe.py    # MoE pretraining script
│   └── run_test_2gpu.sh       # 2-GPU test script
│
├── 3rdparty/
│   └── cutlass/               # NVIDIA CUTLASS (submodule)
│
└── README.md
```

## How It Works

### Backward Overlap

```
Standard Megatron (sequential):
GPU:  [dX + dW] → [AllToAll] ← GPU idle
Time: ├────────┤  ├────────┤

FluidMoE (overlap):
GPU:  [dX] → [AllToAll + dW parallel]
Time: ├───┤  ├────────────────────┤

Speedup: dW computation hidden behind AllToAll
```

### Key Components

| Component | Description |
|-----------|-------------|
| `BackwardScheduler` | Queues dW tasks, executes during AllToAll |
| `FluidGroupedMLP` | Expert MLP with dW scheduling |
| `FluidMoELayer` | Custom MoE layer with direct AllToAll |
| `fluid_kernels` | CUTLASS GroupedGEMM + fused operations |

## Build Kernels

```bash
cd fluid/kernels
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Copy to ops directory
cp fluid_kernels.so ../../ops/
```

## Run Example

```bash
# Set paths
export PYTHONPATH=/path/to/FluidMoE:/path/to/Megatron-LM:$PYTHONPATH

# Run 2-GPU test
cd examples
CUDA_VISIBLE_DEVICES=0,1 bash run_test_2gpu.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FLUID_FORWARD_MODE` | Forward mode: `baseline` or `fused` | `baseline` |
| `FLUID_DX_NUM_CHUNKS` | Chunks for pipelined dX | `1` |
| `FLUID_DEBUG_TIMING` | Enable timing debug | `0` |

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- Megatron-Core >= 0.5.0
- CUDA >= 11.8
- NCCL >= 2.18

## License

Apache 2.0
