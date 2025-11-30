DeepEP provides two distinct modes optimized for different stages of the MoE lifecycle: **Normal Mode** (for training/prefilling) and **Low Latency Mode** (for inference decoding).

### 1. High-Level Comparison

| Feature | Normal Mode | Low Latency Mode |
| :--- | :--- | :--- |
| **Primary Goal** | **Throughput** (Bandwidth saturation) | **Latency** (Speed of delivery) |
| **Use Case** | Training, Inference Prefilling (Large Batch) | Inference Decoding (Small Batch) |
| **Protocol** | NVLink Forwarding + RDMA (Store-and-Forward) | Pure RDMA (Direct Point-to-Point) |
| **SM Usage** | Occupies dedicated SMs for data moving | Minimal / "SM-free" (via async overlap) |
| **Buffer Strategy** | Dynamic (negotiated sizes) | Static (pre-allocated max size) |
| **Requirement** | Standard NVSHMEM | NVSHMEM with IBGDA (Async GPUDirect) |

---

### 2. Implementation Differences

#### A. Communication Topology: Forwarding vs. Direct
*   **Normal Mode (`csrc/kernels/internode.cu`):**
    *   **Mechanism:** Uses a **"Store-and-Forward"** architecture. Data is not sent directly from the source rank to the destination rank if they are on different nodes.
    *   **Workflow:**
        1.  **Gather:** Local tokens are sent via NVLink to specific "Forwarder" SMs on the same node.
        2.  **Transfer:** These Forwarder SMs aggregate data and send it via RDMA to Forwarder SMs on the destination node.
        3.  **Scatter:** Destination Forwarders distribute data via NVLink to the final target ranks.
    *   **Code Evidence:** The `dispatch` kernel in `internode.cu` defines specific `WarpRole`s like `kRDMAAndNVLForwarder` and `kForwarderCoordinator` to manage this multi-hop pipeline.

*   **Low Latency Mode (`csrc/kernels/internode_ll.cu`):**
    *   **Mechanism:** Uses **"Pure RDMA"** (Point-to-Point). Every rank establishes a direct connection to every other rank.
    *   **Workflow:** The kernel directly calculates the destination address on the remote GPU and issues an RDMA write operation.
    *   **Code Evidence:** The `dispatch` kernel in `internode_ll.cu` directly calls `nvshmemi_ibgda_put_nbi_warp` to write data to the destination buffer, bypassing any local aggregation or forwarding steps to minimize hops.

#### B. Synchronization & Buffer Management
*   **Normal Mode:**
    *   **Dynamic Sizing:** The CPU and GPU synchronize to negotiate the exact number of tokens to be sent/received (`moe_recv_counter`). This allows for variable batch sizes but adds synchronization overhead.
    *   **Logic:** It uses a complex "channel" system where SMs lock and release buffer slots (`rdma_send_channel_lock`) to manage flow control.

*   **Low Latency Mode:**
    *   **Static Sizing:** Requires a pre-defined `num_max_dispatch_tokens_per_rank`. This eliminates the need for a CPU-GPU handshake to agree on message sizes.
    *   **Busy-Wait:** It uses atomic flags in GPU memory (`atomic_clean_flag`, `rdma_recv_flag`). The receiving kernel simply busy-waits (`ld_acquire_sys_global`) on a specific memory address until the flag indicates data has arrived.

#### C. Execution & Overlap (The "Hook")
*   **Normal Mode:**
    *   The `dispatch` and `combine` operations are monolithic CUDA kernels. When you call them, the allocated SMs are fully occupied moving data until the operation completes.

*   **Low Latency Mode:**
    *   **Split Execution:** Supports a **Hook-based Overlap**. The Python API `low_latency_dispatch(..., return_recv_hook=True)` allows splitting the kernel into two phases:
        1.  **Send Phase:** Issues the async RDMA requests (`put_nbi`) and returns immediately.
        2.  **Computation:** The GPU is free to perform other tasks (like Attention or MLP) while data travels over the network.
        3.  **Recv Phase (Hook):** A second kernel (or callback) is launched later to verify data arrival (busy-wait on flags).
    *   **Code Evidence:** The C++ `dispatch` function accepts a `phases` argument (checking `LOW_LATENCY_SEND_PHASE` vs `LOW_LATENCY_RECV_PHASE`) to execute only part of the logic.

### 3. Deep Dive: Normal Mode Implementation

In **Normal Mode** (optimized for high-throughput training/prefilling), DeepEP uses a "Store-and-Forward" architecture to maximize bandwidth utilization. This involves complex coordination between Python, C++, and multiple CUDA kernel launches to manage NVLink and RDMA traffic efficiently.

#### A. Dispatch Process (Forward Pass)

**1. Function Calling Stack**

*   **Phase 1: Python Entry (`deep_ep/buffer.py`)**
    *   User calls `Buffer.dispatch(...)`.
    *   Checks `self.runtime.get_num_rdma_ranks()`. If > 1, calls `self.runtime.internode_dispatch(...)`.

*   **Phase 2: C++ Binding & Host Logic (`csrc/deep_ep.cpp`)**
    *   `Buffer::internode_dispatch` is invoked.
    *   **Step 1: Notification (Size Negotiation):** Calls `internode::notify_dispatch` (host function) to calculate per-rank token counts.
    *   **Step 2: CPU Busy-Wait:** Enters a `while(true)` loop reading a pinned memory address (`moe_recv_counter`) mapped to the GPU, waiting for the GPU to write back the total token count.
    *   **Step 3: Allocation:** Allocates the output tensor `recv_x` via `torch::empty`.
    *   **Step 4: Execution:** Calls the main `internode::dispatch` (host function) to move data.

*   **Phase 3: CUDA Kernels (`csrc/kernels/internode.cu`)**
    *   **Kernel 1: `notify_dispatch`**
        *   **Purpose:** All-to-all exchange of integer counts.
        *   **Action:** Ranks send `num_tokens_per_rank` to peers via RDMA (`nvshmemi_ibgda_put_nbi_warp`) and calculate prefix sums to determine write offsets.
    *   **Kernel 2: `dispatch` (The "Forwarding" Engine)**
        *   **Purpose:** Move actual token data (BF16/FP8).
        *   **Architecture:** The thread block is split into **Warp Roles**:
            1.  **`kRDMASender` Warps:** Read local GPU memory -> Push via RDMA to target node's buffer.
            2.  **`kRDMAAndNVLForwarder` Warps:** Read data arrived via RDMA -> Forward via NVLink to the specific local target rank.
            3.  **`kNVLReceivers` Warps:** Final consumers. Read from NVLink buffers -> Write to `recv_x`.
            4.  **`kCoordinator` Warps:** Manage lock-free queues/buffer slots.

#### B. Combine Process (Backward/Aggregation Pass)

**1. Function Calling Stack**

*   **Phase 1: Python Entry (`deep_ep/buffer.py`)**
    *   User calls `Buffer.combine(...)` with `x`, `topk_weights`, and a `handle` (from the previous `dispatch`).
    *   Calls `self.runtime.internode_combine(...)`.

*   **Phase 2: C++ Binding & Host Logic (`csrc/deep_ep.cpp`)**
    *   `Buffer::internode_combine` is invoked.
    *   **Step 1: Reuse Metadata:** Reuses the `handle` (prefix matrices) since the return path matches the forward path.
    *   **Step 2: Barrier & Clean:** Calls `internode::cached_notify` (host function) to perform a global barrier and reset queue head counters.
    *   **Step 3: Execution:** Calls `internode::combine` (host function).

*   **Phase 3: CUDA Kernels (`csrc/kernels/internode.cu`)**
    *   **Kernel 1: `cached_notify`**
        *   **Purpose:** Prepare buffers, reset `rdma_channel_head`/`nvl_channel_head`.
    *   **Kernel 2: `combine` (The "Aggregation" Engine)**
        *   **Purpose:** Move partial results, forward, and sum up at destination.
        *   **Architecture:** Reversed flow compared to dispatch:
            1.  **`kNVLSender` Warps:** Read local data -> Send via NVLink to local "Forwarder" SMs.
            2.  **`kNVLAndRDMAForwarder` Warps:** Read data from local peers via NVLink -> Push via RDMA to destination node.
            3.  **`kRDMAReceiver` Warps:** Read data arriving via RDMA -> **Sum/Reduce** with `topk_weights` -> Write to `combined_x`.
            4.  **`kCoordinator` Warps:** Flow control.

#### C. Key Data Structures

*   **`SourceMeta`**: Metadata packed with tokens (contains `src_rdma_rank` and bitmask `is_token_in_nvl_rank_bits`) to help forwarders route data.
*   **`SymBuffer<T>` / `AsymBuffer<T>`**: Helper templates for managing Symmetric (RDMA) and Asymmetric (NVLink) buffer layouts.
*   **Prefix Matrices (`rdma_channel_prefix_matrix`, `gbl_channel_prefix_matrix`)**: Store token counts per channel/rank, ensuring deterministic write offsets without atomic adds.
*   **Handles (`combined_rdma_head`, `combined_nvl_head`)**: Track buffer offsets from `dispatch` to allow `combine` to "replay" the data path efficiently.