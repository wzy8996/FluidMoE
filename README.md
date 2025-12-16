# FluidMoE - Complete Custom Layer for Megatron-LM

## 简介

**FluidMoE** 是 Megatron-LM 的完全自定义层实现，提供计算通信重叠优化，支持前向和反向优化。

**核心特性**:
- 🚀 **完全自定义层** - FluidSelfAttention 和 FluidMoELayer
- ⚡ **计算通信重叠** - dW 计算与 AllToAll 通信自动重叠
- 🎯 **无需 Patch** - 直接调用 Fluid AllToAll，无全局函数污染
- 🔧 **完全控制** - 可自定义前向计算逻辑，支持前向优化
- 📦 **兼容 Megatron** - 使用 Layer Spec 机制，与 Megatron-LM 无缝集成

## 🚀 5 分钟快速开始

```python
from megatron.core.transformer import TransformerConfig
from megatron.core import GPTModel
from fluid import get_fluid_moe_layer_spec

# 1. 创建配置（MoE + SP + EP）
config = TransformerConfig(
    num_layers=32,
    hidden_size=4096,
    num_moe_experts=8,
    moe_router_topk=2,
    context_parallel_size=4,           # SP for attention
    expert_model_parallel_size=2,      # EP for MoE
    sequence_parallel=True,
)

# 2. 获取 Fluid layer spec (无需 patch!)
layer_spec = get_fluid_moe_layer_spec()

# 3. 创建模型（Fluid 优化自动启用！）
model = GPTModel(config, transformer_layer_spec=layer_spec)
```

就这么简单！查看 [QUICKSTART.md](QUICKSTART.md) 了解更多。

## 架构设计

```
FluidMoE = 完全自定义层 + 无 Patch + 计算通信重叠

┌─────────────────────────────────────────────────────┐
│  您的 Megatron 训练脚本                              │
│  - 标准模型定义                                      │
│  - 标准训练循环                                      │
└──────────────┬──────────────────────────────────────┘
               │ get_fluid_moe_layer_spec()
               ▼
┌─────────────────────────────────────────────────────┐
│  FluidMoE 自定义层 (无需 Patch)                      │
│  ├─ FluidSelfAttention                              │
│  │   - 直接调用 fluid_all_to_all_sp2hp/hp2sp        │
│  │   - 支持自定义前向计算                            │
│  │   - FluidColumnParallelLinear (dW scheduling)    │
│  │   - FluidRowParallelLinear (dW scheduling)       │
│  ├─ FluidMoELayer                                   │
│  │   - FluidTokenDispatcher (直接调用 fluid_all_to_all) │
│  │   - 支持自定义前向计算                            │
│  │   - FluidGroupedMLP (dW scheduling)              │
│  └─ BackwardScheduler                               │
│      - dW 任务队列                                   │
│      - AllToAll 反向时执行 dW                        │
└──────────────┬──────────────────────────────────────┘
               │ 使用基础设施
               ▼
┌─────────────────────────────────────────────────────┐
│  Megatron-LM                                        │
│  - TransformerLayer                                 │
│  - Router, DotProductAttention                      │
└─────────────────────────────────────────────────────┘
```

## 目录结构

```
Fluid/
├── fluid/                          # 核心代码
│   ├── __init__.py                # 模块入口 (v0.5.0)
│   ├── scheduler.py               # dW 调度器
│   ├── communication.py           # Fluid AllToAll 原语
│   ├── attention_module.py        # FluidSelfAttention (完全自定义)
│   ├── moe_module.py              # FluidMoELayer + FluidTokenDispatcher
│   ├── attention_layers.py        # Fluid 线性层 (dW scheduling)
│   ├── moe_layers.py              # FluidGroupedMLP (dW scheduling)
│   └── megatron_layers.py         # Layer Spec 集成
│
├── examples/                       # 使用示例
│   ├── pretrain_moe.py            # 通用 MoE 训练
│   └── run_mixtral_8x7b.sh       # Mixtral-8x7B 训练脚本
│
├── QUICKSTART.md                  # 快速入门 (推荐!)
├── MAINTENANCE_GUIDE.md           # 维护指南
├── CORE_CONCEPT.md                # 核心概念
└── README.md                      # 本文档
```

## 核心组件

### 1. FluidSelfAttention (完全自定义注意力层)

```python
class FluidSelfAttention(MegatronModule):
    """
    完全自定义的 Self-Attention 层
    - 内部直接调用 fluid_all_to_all_sp2hp/hp2sp (无需 patch)
    - 支持自定义前向计算逻辑
    - 使用 FluidColumnParallelLinear/FluidRowParallelLinear
    """

    def forward(self, hidden_states, attention_mask):
        # 1. QKV 投影
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        query, key, value = self._split_qkv(mixed_qkv)

        # 2. AllToAll sp2hp (直接调用,无需 patch!)
        if self.cp_size > 1:
            query = fluid_all_to_all_sp2hp(query, group=self.cp_group)
            key = fluid_all_to_all_sp2hp(key, group=self.cp_group)
            value = fluid_all_to_all_sp2hp(value, group=self.cp_group)

        # 3. 注意力计算
        context = self.core_attention(query, key, value, attention_mask)

        # 4. AllToAll hp2sp (直接调用)
        if self.cp_size > 1:
            context = fluid_all_to_all_hp2sp(context, group=self.cp_group)

        # 5. 输出投影
        output, bias = self.linear_proj(context)
        return output, bias
```

### 2. FluidMoELayer (完全自定义 MoE 层)

```python
class FluidMoELayer(MegatronModule):
    """
    完全自定义的 MoE 层
    - 内部使用 FluidTokenDispatcher
    - FluidTokenDispatcher 直接调用 fluid_all_to_all (无需 patch)
    - 支持自定义前向计算逻辑
    """

    def forward(self, hidden_states):
        # 1. Router
        scores, indices = self.router(hidden_states)

        # 2. Token Dispatch (FluidTokenDispatcher 直接调用 fluid_all_to_all)
        dispatched_input, tokens_per_expert, probs = \
            self.token_dispatcher.dispatch(hidden_states, indices, scores)

        # 3. Expert 计算 (FluidGroupedMLP with dW scheduling)
        expert_output, _ = self.experts(dispatched_input, tokens_per_expert, probs)

        # 4. Token Combine (FluidTokenDispatcher 直接调用 fluid_all_to_all)
        output = self.token_dispatcher.combine(expert_output)

        return output, None
```

### 3. FluidTokenDispatcher (自定义 Token 路由)

```python
class FluidTokenDispatcher:
    """
    自定义 Token Dispatcher
    - 直接调用 fluid_all_to_all (无需 patch Megatron 函数)
    - 支持前向优化 (router-dispatch overlap 等)
    """

    def dispatch(self, hidden_states, routing_map, probs):
        # Permute tokens
        permuted_tokens, self.permutation_map = permute(hidden_states, routing_map)

        # AllToAll Dispatch (直接调用,无需 patch!)
        if self.ep_size > 1:
            global_tokens = fluid_all_to_all(
                self.ep_group,
                permuted_tokens,
                output_splits,
                input_splits,
                comm_type="moe_dispatch",
            )

        return global_tokens, tokens_per_expert, probs
```

## 支持的模型

FluidMoE 支持所有 Megatron-LM 的 MoE 模型：

- ✅ **Mixtral-8x7B** (Mistral AI)
- ✅ **DeepSeekMoE** (DeepSeek)
- ✅ **自定义 MoE 模型** (基于 Megatron GPTModel)

## 使用示例

### 示例 1: 基本用法

```python
from fluid import get_fluid_moe_layer_spec

# 获取 layer spec
layer_spec = get_fluid_moe_layer_spec()

# 使用 layer spec 创建模型
model = GPTModel(config, transformer_layer_spec=layer_spec)
```

### 示例 2: 自定义前向计算

```python
from fluid import FluidSelfAttention, FluidMoELayer

# 继承 FluidSelfAttention 并自定义 forward
class MyCustomAttention(FluidSelfAttention):
    def forward(self, hidden_states, attention_mask):
        # 添加你的自定义逻辑
        # 例如: Ring Attention, 计算-通信重叠等
        ...
        return super().forward(hidden_states, attention_mask)

# 使用自定义层
from megatron.core.transformer.spec_utils import ModuleSpec
layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules={
        'self_attention': ModuleSpec(module=MyCustomAttention),
        'moe': ModuleSpec(module=FluidMoELayer),
    }
)
```

### 示例 3: 查看 Fluid 层信息

```python
from fluid import print_fluid_layer_info, print_status

# 打印模型中的 Fluid 层
print_fluid_layer_info(model)

# 打印 FluidMoE 状态
print_status()
```

输出:
```
============================================================
FluidMoE Layer Information
============================================================
Attention layers: 32 FluidSelfAttention
  - decoder.layers.0.self_attention
  - decoder.layers.1.self_attention
  ...

MoE layers: 32 FluidMoELayer
  - decoder.layers.0.moe
  - decoder.layers.1.moe
  ...

Scheduler status: ✅ Enabled
============================================================
```

## 优化原理

### Backward dW 延迟与重叠

```
标准 Megatron (顺序执行):
GPU:  [dX + dW] → [AllToAll] ← GPU 空闲 ❌
Time: ├────────┤ ├────────┤
      T_dX+dW    T_AllToAll

FluidMoE (计算通信重叠):
GPU:  [dX] → [AllToAll] ← 同时执行 dW ✅
           ⬇
      [dW 并行执行]
Time: ├───┤ ├────────┤
      T_dX  max(T_dW, T_AllToAll)

加速比: (T_dX + T_dW + T_AllToAll) / (T_dX + max(T_dW, T_AllToAll))
```

**节省时间**: 如果 `T_dW ≈ T_AllToAll`，理论加速 ~1.3-1.5x

### 关键设计

1. ✅ **dX 立即计算** - 保证梯度传播不阻塞
2. ✅ **dW 延迟注册** - 注册到调度器队列
3. ✅ **AllToAll 触发** - 反向传播时执行队列中的 dW
4. ✅ **GPU 不空闲** - dW 计算填补 AllToAll 等待时间

## 性能统计

```python
from fluid import get_backward_scheduler

scheduler = get_backward_scheduler()
stats = scheduler.get_stats()

print(f"Total dW tasks: {stats['total_dw_tasks']}")
print(f"Completed dW tasks: {stats['completed_dw_tasks']}")
print(f"Overlap efficiency: {stats['completed_dw_tasks']/stats['total_dw_tasks']*100:.1f}%")
```

**理想重叠率**:
- **90%+** - 优秀！dW 几乎完全与 AllToAll 重叠
- **70-90%** - 良好,部分 dW 在通信期间完成
- **<70%** - 需要调优,可能 dW 太快或通信太慢

## 文档

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目概览 (本文档) |
| [QUICKSTART.md](QUICKSTART.md) | 快速入门和使用示例 |
| [CORE_CONCEPT.md](CORE_CONCEPT.md) | 核心概念和设计思想 |
| [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md) | 维护指南和 Megatron API 跟随策略 |
| [WORKFLOW.md](WORKFLOW.md) | 详细的代码工作流程 |

## 要求

- Python >= 3.8
- PyTorch >= 2.0
- Megatron-Core >= 0.5.0
- CUDA >= 11.8

## 安装

```bash
# 克隆仓库
cd /path/to/your/project

# 将 Fluid 添加到 PYTHONPATH
export PYTHONPATH="/path/to/Fluid:$PYTHONPATH"

# 确保 Megatron-LM 也在 PYTHONPATH 中
export PYTHONPATH="/path/to/Megatron-LM:$PYTHONPATH"
```

## 完整示例

查看 [examples/pretrain_moe.py](examples/pretrain_moe.py) 获取完整的训练示例。

运行 Mixtral-8x7B:
```bash
bash examples/run_mixtral_8x7b.sh
```

## 与 Megatron 的关系

| 组件 | Megatron | FluidMoE |
|------|----------|---------|
| **TransformerLayer** | ✅ 使用原版 | - |
| **SelfAttention** | ❌ 替换 | ✅ FluidSelfAttention |
| **MoELayer** | ❌ 替换 | ✅ FluidMoELayer |
| **Router** | ✅ 使用原版 | - |
| **DotProductAttention** | ✅ 使用原版 | - |
| **TokenDispatcher** | ❌ 替换 | ✅ FluidTokenDispatcher |
| **GroupedMLP** | ❌ 替换 | ✅ FluidGroupedMLP |
| **Linear 层** | ❌ 替换 | ✅ Fluid*ParallelLinear |
| **AllToAll 函数** | ✅ 不 patch | ✅ 直接调用 Fluid 版本 |

**设计原则**:
- 替换最少的模块 (只替换需要优化的层)
- 直接调用 Fluid AllToAll (不污染全局命名空间)
- 尽可能复用 Megatron 组件 (Router, CoreAttention 等)

## 维护

FluidMoE 使用完全自定义层，需要跟随 Megatron API 变化：

```bash
# Megatron 更新后
cd Megatron-LM
git pull origin main

# 检查 SelfAttention 和 MoELayer API 变化
git diff <old_commit> <new_commit> -- megatron/core/transformer/attention.py
git diff <old_commit> <new_commit> -- megatron/core/transformer/moe/

# 同步到 FluidSelfAttention 和 FluidMoELayer
cd Fluid
vim fluid/attention_module.py
vim fluid/moe_module.py

# 运行测试
python -m pytest tests/
```

详见 [MAINTENANCE_GUIDE.md](MAINTENANCE_GUIDE.md)。

## 版本历史

- **v0.5.0** (当前): 完全自定义层实现，无需 patch
- **v0.4.0**: Layer Spec 模式，需要 patch AllToAll
- **v0.3.0**: Monkey Patching + Layer Spec 双模式
- **v0.2.0**: 初始 Monkey Patching 实现

## License

Apache 2.0

## Citation

如果 FluidMoE 对你的研究有帮助，请引用:

```bibtex
@software{fluidmoe2024,
  title={FluidMoE: Complete Custom Layer Implementation for Megatron-LM MoE},
  author={FluidMoE Team},
  year={2024},
  url={https://github.com/your-org/FluidMoE}
}
```
