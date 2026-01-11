# MagiAttention API & Architecture

## Overview

MagiAttention provides a distributed attention mechanism designed for ultra-long contexts and heterogeneous mask types. The API allows users to define attention patterns (masks), dispatch data across a distributed process group (Context Parallelism), compute attention efficiently, and then gather the results.

## Key Concepts

*   **DistAttnRuntimeKey**: A lightweight hashable key that uniquely identifies a distributed attention configuration (sequence lengths, mask patterns, parallelism settings). It serves as a handle to retrieve the heavy-weight runtime manager.
*   **DistAttnRuntimeMgr**: The core manager class (hidden behind the key) that holds the pre-calculated metadata for dispatching data, scheduling communication, and executing kernels.
*   **Context Parallelism (CP)**: The strategy of splitting the sequence dimension across multiple GPUs to handle long contexts.
*   **Flexible Flash Attention (FFA)**: The underlying kernel that supports irregular mask patterns.

## Context Parallelism Strategy

MagiAttention implements a advanced and flexible strategy often referred to as **Unified Context Parallelism** or a **Generalized All-to-All** approach.

### Comparison with Other Strategies

| Feature | DeepSpeed-Ulysses | Ring Attention | MagiAttention |
| :--- | :--- | :--- | :--- |
| **Sharding Dim** | Head Dimension | Sequence Dimension | Sequence Dimension |
| **Communication** | All-to-All (Synchronous) | P2P / Circular (Sequential) | GroupCast / All-to-All-v (Async) |
| **Scalability** | Limited by # of Heads | Linear with Context Length | Linear with Context Length |
| **Load Balancing** | Uniform | Uniform | **Non-uniform / Mask-aware** |
| **Mask Support** | Dense | Simple Causal | **Heterogeneous / Complex** |

### Key Advantages of MagiAttention CP

1.  **Beyond Ulysses**: By sharding the sequence dimension instead of heads, MagiAttention can scale to a number of GPUs that exceeds the head count of the model.
2.  **Beyond Ring Attention**: Instead of circular P2P passes which can be slow and create "bubbles," MagiAttention uses a custom **`GroupCast`** primitive based on **All-to-All-v**. This allows for more direct, efficient data exchanges.
3.  **Mask-aware Load Balancing**: The `DistAttnSolver` analyzes the mask pattern (e.g., Causal, Block-Sparse) and shards the sequence *non-uniformly*. This ensures every GPU performs an equal amount of "active" computation, preventing idle time on ranks handling heavily masked regions.
4.  **Asynchronous Overlap**: It leverages asynchronous communication to fetch K/V blocks in the background while the GPU is busy calculating attention on the current block, maximizing hardware utilization.

## Public API Interface (`magi_attention.api`)

The public API is designed to be functional and stateless from the user's perspective, using the `key` to maintain state.

### 1. Key Generation & Setup

These functions prepare the distributed environment and calculate the necessary metadata (load balancing, communication schedules).

*   **`magi_attn_varlen_dispatch(...) -> (local_x, key)`**
    *   **Purpose**: Serves as the initialization and preparation phase for running distributed attention with variable-length sequences. It is designed to be called once per forward pass to set up the "execution plan" that all subsequent attention layers will follow.
    *   **Core Responsibilities**:
        1.  **Generating the Execution Plan**: It analyzes `cu_seqlens` and the attention mask (e.g., Causal) to determine an optimal way to shard data across GPUs, aiming to balance computational load while minimizing communication overhead.
        2.  **Padding the Input**: It pads the input tensor along the sequence dimension so that its total length is perfectly divisible by `chunk_size * cp_size`.
        3.  **Dispatching (Sharding)**: It physically splits the padded global input tensor and sends the appropriate "shard" (`local_x`) to the local GPU.
    *   **Why it returns a padded input?**: 
        *   **Kernel Requirements**: High-performance GPU kernels (like FFA) and distributed communication primitives work most efficiently on fixed-size blocks. Padding ensures that data can be evenly divided into chunks without complex boundary checks.
        *   **Uniform Sharding**: It simplifies the logic of splitting the global sequence into local shards, ensuring every rank receives a consistently sized workload that fits the pre-calculated execution plan.

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

## Data Flow and Tensor Shapes

In MagiAttention's Context Parallel (CP) workflow, it is important to distinguish between **global** and **local** tensors.

### Local Shards
The attention function (e.g., `magi_attention_func`) receives `q`, `k`, and `v` tensors that are **already sliced** along the sequence dimension.

*   **Pre-Model Dispatch**: The global input is sharded via `magi_attn_varlen_dispatch` before entering the model.
*   **Layer-wise Projection**: The model's layers operate on these local shards. Therefore, the `q`, `k`, and `v` generated by projections are themselves local shards.
*   **Sequence Dimension (`s`)**: In internal notation like `"1 nh s hd"`, the `s` refers to the **local sequence length** (e.g., `Total_Seq / CP_Size`), not the global length.

### The Role of `calc_attn`
While `calc_attn` receives local shards, it is responsible for fetching the required **remote** K/V slices from other ranks (as planned by the `DistAttnSolver`) to ensure that every query in the local shard can attend to its full context across the entire distributed system.

## Detailed Analysis of DistAttnSolver

The `DistAttnSolver` is a sophisticated component responsible for partitioning attention computation and communication across multiple ranks to maximize overlap. It transforms a logical mask into a physical execution plan.

### 1. State Initialization
*   **`__init__`**: Initializes the distributed environment (rank, size, group) and the `OverlapSolver`. It sets up the `OverlapConfig`, which defines cost factors and constraints on chunking (e.g., `min_chunk_size`, `max_num_chunks`).
*   **`solve`**: The main execution flow. It identifies the local attention bucket, determines global host/remote K ranges, and then proceeds through a multi-step process: chunking remote K ranges, solving the overlap problem, and constructing a detailed communication plan (`TransferTable`).

### 2. Core Algorithms
*   **Chunking (`_chunk_remote_k_ranges_global`)**: Determines how to split the remote K sequence. It attempts to use `min_chunk_size` but caps the total number of chunks at `max_num_chunks`. If the cap is hit, it increases the chunk size dynamically.
*   **Cost Estimation (`_calc_cost_pairs_per_chunk`)**: Estimates execution costs for the host chunk and each remote chunk:
    *   `calc_cost = calc_cost_factor * attention_area` (where area is Q_len * K_len).
    *   `comm_cost = comm_cost_factor * sequence_length`.
    *   The host chunk has a communication cost of 0.
*   **Overlap Solving (`_solve_multi_stage_overlap`)**: Uses the `OverlapSolver` to group chunks into stages to minimize total execution time. In `DYNAMIC` mode, it synchronizes the `overlap_degree` across all ranks using a global maximum to ensure consistent scheduling.
*   **Rank Entry Construction (`_calc_remote_rank_entry_for_one_stage`)**: Merges the chunks assigned to a specific stage into a single `RemoteRankEntry`. It handles complex logic for slice splitting and mask merging (e.g., merging FULL and CAUSAL masks when chunks are combined).
*   **Communication Planning (`_init_transfer_table_per_stage`)**: Constructs the `TransferTable` for each stage. It identifies precisely which global K ranges need to be sent to or received from other ranks and calculates their local offsets in the communication buffers.

### 3. Data Flow
The transformation of data follows this path:
1.  **`HostRankEntry`**: Captures the initial global state (local Q/K ranges) and the fine-grained list of all remote chunks needed by the rank.
2.  **`RemoteRankEntry`**: Represents the aggregated work (computation + required remote data) for a specific *overlap stage*.
3.  **`TransferTable`**: A detailed 2D plan of rank-to-rank data transfers for a stage, mapping global ranges to local send/recv buffer offsets.
4.  **`CommMeta` / `CalcMeta`**: The final runtime arguments. `CommMeta` provides the `GroupCollectiveArg` for communication primitives, and `CalcMeta` provides the `AttnArg` for the GPU kernels.

### 4. Key Classes
*   **`HostRankEntry`**: Container for global ranges and the list of all remote chunks.
*   **`RemoteRankEntry`**: Container for a single stage's remote K ranges and corresponding attention slices.
*   **`TransferTable`**: A `[CP_SIZE x CP_SIZE]` table mapping global ranges to local send/recv buffer offsets.
*   **`GroupCastRanges`**: A specialized range collection that handles overlapping requirements from multiple ranks, splitting them into non-overlapping segments for efficient `GroupCast` communication.

## Detailed Analysis of OverlapSolver

The `OverlapSolver` is a specialized component responsible for minimizing total execution time by optimizing the scheduling of communication and computation chunks.

### 1. Objective
Its primary goal is to solve the **Multi-Stage Overlapping Problem**: Given a set of work "chunks" (each with a communication cost and a calculation cost), how should they be grouped into "stages" and ordered to maximize the overlap between:
*   **Communication**: Fetching data for the *current* stage (`Stage i`).
*   **Calculation**: Processing data for the *previous* stage (`Stage i-1`).

### 2. Algorithms
Currently, the solver implements a **Uniform** strategy (with a placeholder for a future **Greedy** strategy).

*   **Uniform Overlap (`_solve_with_uniform`)**:
    *   **Logic**: It partitions the available chunks into `overlap_degree` stages as evenly as possible.
    *   **Host Pinning**: The host chunk (index 0) is always pinned to the first stage (`Stage 0`) because it is immediately available for computation.
    *   **Randomization**: If enabled, it shuffles the remote chunks to potentially find a better random distribution, which can be useful for avoiding "hotspots" in communication patterns.
    *   **Idle Stages**: If the requested degree is higher than the number of chunks, it appends empty "idle" stages to satisfy the architectural requirement.

### 3. Cost Model
The solver uses a simplified pipeline simulation to estimate the `overall_cost` (total time).

**Formula**:
`Total Time = Stage_0_Cost + Σ (Stage_i_Cost) + Final_Calc`

Where the cost of each overlapped step is:
`Stage_i_Cost = max(Comm_i, Calc_{i-1})`

*   **`Comm_i`**: Time to fetch all chunks in Stage `i`.
*   **`Calc_{i-1}`**: Time to compute attention for all chunks in Stage `i-1`.
*   **Assumption**: It assumes perfect concurrency—computation and communication happen simultaneously. The pipeline is limited by whichever operation takes longer (the "bottleneck").

### 4. Dynamic vs. Static Mode
*   **Static Mode**: Uses a fixed `overlap_degree` defined by the user (e.g., 2 or 3). The solver simply partitions chunks into that many stages.
*   **Dynamic Mode**: The solver performs a search to find the optimal degree.
    1.  It iterates from `degree=1` to `dynamic_max_degree`.
    2.  For each degree, it generates a solution and calculates the `overall_cost`.
    3.  It selects the **Best Solution** by minimizing total cost (primary) and overlap degree (secondary).
    4.  **Global Sync**: In a distributed setting, all ranks must agree on the number of stages. The `DistAttnSolver` performs an `all_reduce(MAX)` to synchronize the final `overlap_degree` across the cluster.

## Computation-Communication Overlap

MagiAttention achieves high performance by interleaving communication and computation in a pipelined manner, managed by the `DistAttnRuntime`.

### 1. The Pipeline Structure
The execution is divided into a **Host Stage** and multiple **Remote Stages**.

*   **Host Stage (Local)**: The GPU immediately starts computing attention for the local Q and K shards (which are already in memory).
*   **Remote Stages (Pipelined)**: While the Host Stage is running, the system asynchronously fetches the first batch of remote K/V blocks.

### 2. Execution Loop (`DistAttnFunc.forward`)
The loop inside `DistAttnFunc.forward` orchestrates this pipeline:

1.  **Start Host Computation**: The attention kernel (`_flex_flash_attn_forward`) is launched for the local data.
2.  **Prefetch Next Stage**: *Simultaneously*, the runtime issues asynchronous `GroupCast` operations to fetch the K/V blocks needed for the *next* stage.
3.  **Process Remote Stages**: For each subsequent stage `i`:
    *   **Wait**: The system waits for the data transfer for Stage `i` to complete (using `wait_post_process`).
    *   **Compute**: It launches the attention kernel for Stage `i`, accumulating results directly into the local output buffer (`out_acc`).
    *   **Prefetch**: It triggers the data fetch for Stage `i+1`.
    *   **Reduce**: In parallel, it may launch asynchronous `GroupReduce` operations to merge partial results (if needed).

### 3. Key Optimization Techniques
*   **Asynchronous I/O**: Operations like `group_cast` and `group_reduce` are non-blocking, allowing the CPU to proceed and launch GPU kernels while data is moving.
*   **Direct Accumulation**: The Flexible Flash Attention (FFA) kernel supports accumulating results directly into the `out_acc` tensor, avoiding the need to store and manually sum partial intermediate tensors.
*   **SM Margin**: The kernels are designed to leave a small margin of Streaming Multiprocessors (SMs) free, ensuring that communication kernels (like NCCL operations) don't get starved of resources while a heavy attention computation is running.

## Backward Pass and Gradient Computation

The backward pass mirrors the forward pass's pipelined structure but involves more complex data dependencies and reduction steps.

### 1. Objective
Compute gradients for `q`, `k`, `v`, and `sink` (`dq`, `dk`, `dv`, `dsink`).

### 2. The Backward Pipeline
Similar to the forward pass, the backward pass is divided into a **Host Stage** and multiple **Remote Stages**. However, the data flow is different:

*   **Host Stage**:
    *   Loads saved tensors from the forward pass (`q`, `kv`, `out`, `lse`, `sink`).
    *   Computes partial gradients using local data.
    *   Starts pre-fetching remote data for the first remote stage.

*   **Remote Stages**:
    *   **Data Fetching**: The backward pass requires fetching not just `k` and `v`, but also `q`, `out`, `do` (gradient of output), and `lse` from remote ranks.
    *   **Computation**: `apply_bwd_partial_attn` calculates partial gradients for the current stage. `dq` gradients are often accumulated directly into a local buffer.
    *   **Reduction**:
        *   `dq` is accumulated locally.
        *   `dk` and `dv` are partial results that must be summed across ranks. The system uses asynchronous `GroupReduce` to merge these partial gradients.
        *   `dsink` is synchronized using `all_reduce`.

### 3. Synchronization
The system uses `prepare_reduced_local_dqkv_global_dsink` as the final synchronization point. It waits for all asynchronous `GroupReduce` and `all_reduce` operations to complete before returning the final gradients, ensuring correctness while maximizing overlap during the computation phase.

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
