# MagiAttention API & Architecture

## Overview

MagiAttention provides a distributed attention mechanism designed for ultra-long contexts and heterogeneous mask types. The API allows users to define attention patterns (masks), dispatch data across a distributed process group (Context Parallelism), compute attention efficiently, and then gather the results.

## Key Concepts

*   **DistAttnRuntimeKey**: A lightweight hashable key that uniquely identifies a distributed attention configuration (sequence lengths, mask patterns, parallelism settings). It serves as a handle to retrieve the heavy-weight runtime manager.
*   **DistAttnRuntimeMgr**: The core manager class (hidden behind the key) that holds the pre-calculated metadata for dispatching data, scheduling communication, and executing kernels.
*   **Context Parallelism (CP)**: The strategy of splitting the sequence dimension across multiple GPUs to handle long contexts.
*   **Flexible Flash Attention (FFA)**: The underlying kernel that supports irregular mask patterns.

## Public API Interface (`magi_attention.api`)

The public API is designed to be functional and stateless from the user's perspective, using the `key` to maintain state.

### 1. Key Generation & Setup

These functions prepare the distributed environment and calculate the necessary metadata (load balancing, communication schedules).

*   **`magi_attn_varlen_key(...) -> DistAttnRuntimeKey`**
    *   **Purpose**: Creates a runtime key for variable sequence length (varlen) scenarios, similar to `flash_attn_varlen`.
    *   **Inputs**: Cumulative sequence lengths (`cu_seqlens_q/k`), padding/chunk sizes, process group, causal flag, window size.
    *   **Logic**: Infers attention ranges and mask types from the sequence lengths, then initializes the runtime manager.

*   **`magi_attn_flex_key(...) -> DistAttnRuntimeKey`**
    *   **Purpose**: The most flexible interface for arbitrary mask patterns.
    *   **Inputs**: Explicit `AttnRanges` for Q and K, `attn_mask_type` (e.g., CAUSAL, FULL, or mixed), global sequence lengths, parallelism config.
    *   **Logic**: Validates inputs, handles padding, and initializes the runtime manager with custom ranges.

*   **`make_varlen_key_for_new_mask_after_dispatch(...)`** / **`make_flex_key_for_new_mask_after_dispatch(...)`**
    *   **Purpose**: Creates a new key that reuses the *dispatch solution* (data distribution) of an existing key but applies a *different mask pattern* (computation/communication schedule).
    *   **Use Case**: Hybrid attention models where different layers use different masks (e.g., sliding window vs. full) but must share the same data layout to avoid costly re-shuffling.

### 2. Data Movement (Dispatch/Undispatch)

*   **`magi_attn_varlen_dispatch(...) -> (local_x, key)`** / **`magi_attn_flex_dispatch(...) -> (local_x, key)`**
    *   **Purpose**: Convenience wrappers that generate the key and immediately dispatch the input tensor.

*   **`dispatch(x, key, pad_value=0.0) -> local_x`**
    *   **Purpose**: Distributes a global tensor (shape `[B*S, H, D]`) to the local rank according to the dispatch plan stored in `key`.
    *   **Logic**: Pads the input and scatters chunks to the appropriate GPUs to ensure load balance based on the attention mask.

*   **`undispatch(x, key) -> global_x`**
    *   **Purpose**: Gathers the local results back to the global tensor layout.
    *   **Logic**: Reverses the dispatch operation and removes padding.

### 3. Computation

*   **`calc_attn(q, k, v, key, ...) -> (out, lse)`**
    *   **Purpose**: Computes the distributed attention.
    *   **Inputs**: Local `q, k, v` tensors (already dispatched), the `key`.
    *   **Logic**: Executes the pre-planned communication (ring/all-to-all) and computation (FFA) steps to produce the attention output.

### 4. Utilities

*   **`get_position_ids(key) -> Tensor`**: Returns the global position IDs corresponding to the local tensor chunks.
*   **`get_most_recent_key() -> Key`**: Helper to retrieve the last created key.

---

## Internal Calling Process

The system uses a "Plan-Execute" model.

### Phase 1: Planning (Key Generation)
When `magi_attn_varlen_key` or `magi_attn_flex_key` is called:
1.  **Validation & preprocessing**: Input ranges and masks are validated. Padding is calculated.
2.  **`init_dist_attn_runtime_key`**: A lightweight key is created.
3.  **`init_dist_attn_runtime_mgr`**: (If key is new)
    *   **Dispatch Planning**: `make_dispatch_meta_from_qk_ranges` determines how to split and permute global sequences into chunks and assigns them to ranks to balance the workload (considering the sparsity of the mask).
    *   **Execution Planning**: `make_attn_meta_from_dispatch_meta` generates `CommMeta` (communication peers/steps) and `CalcMeta` (kernel arguments).
    *   **Solver**: A `DistAttnSolver` is selected to optimize the strategy.
    *   **Storage**: The resulting `DistAttnRuntimeMgr` is cached in the global `dist_attn_runtime_dict`.

### Phase 2: Runtime Execution

**A. Dispatching (`dispatch`)**
1.  User calls `dispatch(x, key)`.
2.  Look up `DistAttnRuntimeMgr` using `key`.
3.  Manager calls `dispatch_func`.
4.  Data is padded and communicated (All-to-All or scatter) to the target ranks based on `dispatch_meta`.

**B. Attention (`calc_attn`)**
1.  User calls `calc_attn(local_q, local_k, local_v, key)`.
2.  Look up `DistAttnRuntimeMgr`.
3.  Manager calls `dist_attn_func`.
4.  The `DistAttnRuntime` executes the loop:
    *   **Comm**: Asynchronously fetch K/V blocks from other ranks (`GroupCast` / `P2P`).
    *   **Comp**: Compute attention on available blocks using `flex_flash_attn`.
    *   **Overlap**: Computation and communication are overlapped.

**C. Undispatching (`undispatch`)**
1.  User calls `undispatch(local_out, key)`.
2.  Look up `DistAttnRuntimeMgr`.
3.  Manager calls `undispatch_func` to gather results back to the global layout.
