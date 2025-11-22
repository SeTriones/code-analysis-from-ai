# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Torch Memory Saver is a PyTorch library that enables temporary release and resumption of tensor memory. It works by creating pauseable memory regions where tensors can be created, paused to free GPU memory, and later resumed. The library supports both CUDA and ROCm/HIP platforms.

## Development Commands

### Quick Development Cycle
- `make reinstall` - Clean build and reinstall the package (removes .so files, build directory, uninstalls, and reinstalls)

### Testing
- `pytest /path/to/torch_memory_saver/test` - Run all tests
- `pytest /path/to/torch_memory_saver/test/test_examples.py::test_simple -s` - Run a specific test case
- Tests are parametrized to run with both "preload" and "torch" hook modes

### Building for Distribution
- `make clean` - Remove distribution files
- `make build-wheel` - Build wheel using Docker (requires PYTHON_VERSION=3.9 CUDA_VERSION=12.4)
- `make build-sdist` - Build source distribution
- `make upload` - Upload to PyPI using Docker

## Architecture

### Hook Modes
The library supports two hook modes for intercepting CUDA memory allocations:
1. **preload** (`hooks/mode_preload.py`) - Uses LD_PRELOAD to hook CUDA malloc/free APIs
2. **torch** (`hooks/mode_torch.py`) - Uses PyTorch's custom allocator API

### Core Components

#### Python Layer
- `entrypoint.py` - Main `TorchMemorySaver` class providing the public API
- `binary_wrapper.py` - Python wrapper for the C++ binary interface
- `__init__.py` - Exports the global singleton instance

#### C++ Layer (csrc/)
- `core.cpp/core.h` - Core memory management logic with allocation metadata tracking
- `api_forwarder.cpp/api_forwarder.h` - API forwarding between Python and C++
- `entrypoint.cpp` - C++ entry points for Python bindings

### Key Concepts

#### Memory Regions
Memory regions are created using context managers that mark tensors as pauseable:
```python
with torch_memory_saver.region():
    tensor = torch.full(...)  # This tensor becomes pauseable
```

#### Tagging System
Tensors can be tagged for selective pause/resume operations:
```python
with torch_memory_saver.region(tag="type1"):
    tensor1 = torch.full(...)

torch_memory_saver.pause("type1")  # Only pause tensors with "type1" tag
```

#### CPU Backup
Optional CPU backup preserves tensor contents during pause/resume cycles:
```python
with torch_memory_saver.region(enable_cpu_backup=True):
    tensor = torch.full(...)
```

### Platform Support
The build system detects and supports both CUDA and ROCm/HIP platforms automatically. Extension modules are built for both hook modes during installation.

### Testing Structure
Tests are located in `test/examples/` with corresponding test runners in `test_examples.py`. Each test runs in a subprocess to ensure proper isolation of memory management state.

## CUDA Memory Interception Mechanism

The library intercepts CUDA memory allocations through two different mechanisms, both routing to the same core virtual memory management system.

### Hook Mode Implementations

#### 1. Preload Mode (LD_PRELOAD Interception)

**How it works:**
- Uses `LD_PRELOAD` to load the library before program startup
- **Literally replaces** the system's `cudaMalloc` and `cudaFree` functions at runtime

**Key implementation in `csrc/entrypoint.cpp:44-60`:**
```cpp
#ifdef TMS_HOOK_MODE_PRELOAD
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        return TorchMemorySaver::instance().malloc(ptr, ...);
    } else {
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}
#endif
```

**Function replacement mechanism:**
1. Library is preloaded via `LD_PRELOAD` environment variable
2. Custom `cudaMalloc`/`cudaFree` functions override system versions
3. Conditional routing based on whether we're inside a pauseable region
4. Falls back to original CUDA functions via `dlsym(RTLD_NEXT, "cudaMalloc")`

#### 2. Torch Mode (PyTorch Custom Allocator)

**How it works:**
- Uses PyTorch's `CUDAPluggableAllocator` API
- Registers custom allocation functions that PyTorch calls instead of `cudaMalloc`

**Setup in `hooks/mode_torch.py:10-19`:**
```python
self.allocator = CUDAPluggableAllocator(
    self.get_path_binary(), 
    "tms_torch_malloc", 
    "tms_torch_free"
)
```

**Custom allocator functions in `csrc/entrypoint.cpp:64-86`:**
```cpp
extern "C" {
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
    void *ptr;
    TorchMemorySaver::instance().malloc(&ptr, ...);
    return ptr;
}
}
```

### Core Virtual Memory Management

Both hook modes route to `TorchMemorySaver::malloc()` in `csrc/core.cpp`, which uses **CUDA Virtual Memory Management** instead of regular `cudaMalloc`:

```cpp
// Instead of: cudaMalloc() -> single allocation
// TMS uses: Virtual memory with separate allocation and mapping steps

CUmemGenericAllocationHandle allocHandle;
CUDAUtils::cu_mem_create(&allocHandle, size, device);              // 1. Create memory handle
cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0);            // 2. Reserve virtual address
cuMemMap((CUdeviceptr)*ptr, size, 0, allocHandle, 0);             // 3. Map handle to address
CUDAUtils::cu_mem_set_access(ptr, size, device);                  // 4. Set access permissions
```

**Why virtual memory approach:**
- **Virtual addresses remain constant** during pause/resume cycles
- **Physical memory can be released** while preserving virtual addresses
- **PyTorch tensors are unaware** their memory was paused/resumed

### Pause/Resume Implementation

**Pause operation (`csrc/core.cpp:242-244`):**
```cpp
cuMemUnmap((CUdeviceptr)ptr, metadata.size);      // Unmap physical memory
cuMemRelease(metadata.allocHandle);               // Release physical memory
// Virtual address stays reserved but becomes inaccessible
```

**Resume operation (`csrc/core.cpp:303-308`):**
```cpp
CUmemGenericAllocationHandle newAllocHandle;
CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device); // New physical memory
cuMemMap((CUdeviceptr)ptr, metadata.size, 0, newAllocHandle, 0);          // Map to same virtual address
CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
```

### Thread-Local Configuration

Memory interception is controlled by thread-local state in `csrc/entrypoint.cpp:9-39`:
- `is_interesting_region()` - Whether we're inside a pauseable region
- `current_tag_` - Tag for selective pause/resume operations  
- `enable_cpu_backup()` - Whether to backup tensor contents to CPU

This allows fine-grained control over which allocations get intercepted and how they're handled.

## CUDA Virtual Memory Management Explained

### Traditional CUDA Memory vs Virtual Memory Management

**Traditional CUDA Memory (cudaMalloc):**
```cpp
// Single-step allocation - physical memory directly tied to virtual address
cudaMalloc(&ptr, size);  // Gets both virtual address AND physical memory
cudaFree(ptr);           // Releases both virtual address AND physical memory
```

**CUDA Virtual Memory Management (VMM):**
```cpp
// Multi-step process - separate virtual addresses from physical memory
CUmemGenericAllocationHandle handle;
cuMemCreate(&handle, size, &props);           // 1. Create physical memory handle
cuMemAddressReserve(&virt_ptr, size, 0,0,0);  // 2. Reserve virtual address space
cuMemMap(virt_ptr, size, 0, handle, 0);       // 3. Map handle to virtual address  
cuMemSetAccess(virt_ptr, size, &access, 1);   // 4. Set access permissions
```

**Key VMM Benefits:**
- **Persistent virtual addresses**: Virtual address can persist while physical memory changes
- **Flexible mapping**: Same physical memory can be mapped to different virtual addresses
- **Memory sharing**: Multiple virtual addresses can map to the same physical memory
- **Fine-grained control**: Pause/resume by unmapping/remapping without changing virtual addresses

### How Torch Memory Saver Uses VMM

#### 1. Allocation Process (csrc/core.cpp:95-108)

**CUDA Implementation:**
```cpp
CUmemGenericAllocationHandle allocHandle;
CUDAUtils::cu_mem_create(&allocHandle, size, device);              // Create physical memory
CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0)); // Reserve virtual space
CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));   // Map physical to virtual
CUDAUtils::cu_mem_set_access(ptr, size, device);                  // Set read/write permissions

// Store metadata for later pause/resume
allocation_metadata_.emplace(*ptr, AllocationMetadata{
    size, device, tag, AllocationState::ACTIVE, enable_cpu_backup, nullptr, allocHandle
});
```

**ROCm/HIP Implementation (Chunked Approach):**
ROCm has limitations with `hipMemCreate`, so the project uses a chunked approach:
```cpp
// Split large allocation into chunks due to ROCm limitations
size_t num_chunks = (aligned_size + MEMCREATE_CHUNK_SIZE - 1) / MEMCREATE_CHUNK_SIZE;

// Reserve single virtual address space
hipMemAddressReserve(&d_mem, aligned_size, granularity, 0, node_id);

// Create and map each chunk separately
for (size_t i = 0; i < num_chunks; ++i) {
    hipMemCreate(&allocHandles[i], chunk_sizes[i], &prop, 0);    // Create chunk handle
    void* map_addr = (void*)((uintptr_t)d_mem + allocated_size);
    hipMemMap((hipDeviceptr_t)map_addr, chunk_sizes[i], 0, allocHandles[i], 0); // Map chunk
}
```

#### 2. Pause Operation - Unmapping (csrc/core.cpp:242-244)

```cpp
// Backup to CPU if enabled (optional)
if (metadata.enable_cpu_backup) {
    cudaMemcpy(metadata.cpu_backup, ptr, metadata.size, cudaMemcpyDeviceToHost);
}

// Unmap physical memory from virtual address
CURESULT_CHECK(cuMemUnmap((CUdeviceptr) ptr, metadata.size));
CURESULT_CHECK(cuMemRelease(metadata.allocHandle));  // Release physical memory

// Virtual address 'ptr' remains reserved but becomes inaccessible
// PyTorch tensors still have the same pointer value, but accessing it would segfault
metadata.state = AllocationState::PAUSED;
```

**What happens during pause:**
- **Virtual address stays the same**: `ptr` value unchanged
- **Physical memory released**: GPU memory freed, available for other allocations  
- **Virtual address becomes inaccessible**: Accessing `ptr` causes segmentation fault
- **Metadata preserved**: All information needed for resume is stored

#### 3. Resume Operation - Remapping (csrc/core.cpp:303-327)

```cpp
// Create NEW physical memory handle
CUmemGenericAllocationHandle newAllocHandle;
CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

// Map new physical memory to the SAME virtual address
CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));
CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);

// Restore from CPU backup if enabled
if (metadata.enable_cpu_backup) {
    cudaMemcpy(ptr, metadata.cpu_backup, metadata.size, cudaMemcpyHostToDevice);
}

// Update metadata with new handle
metadata.state = AllocationState::ACTIVE;
metadata.allocHandle = newAllocHandle;
```

**What happens during resume:**
- **Same virtual address**: `ptr` value identical to before pause
- **New physical memory**: Fresh GPU memory allocated (may be different physical location)
- **Seamless to PyTorch**: Tensors work normally, unaware of pause/resume cycle
- **Content restored**: If CPU backup enabled, original tensor values restored

## When Hooks Fall Back to Naive cudaMalloc

The hooks fall back to standard `cudaMalloc` in several specific scenarios:

### 1. Outside Pauseable Regions (Primary Fallback)

**Preload Mode Fallback (`csrc/entrypoint.cpp:44-51`):**
```cpp
cudaError_t cudaMalloc(void **ptr, size_t size) {
    if (thread_local_config.is_interesting_region()) {
        // Use VMM - inside torch_memory_saver.region()
        return TorchMemorySaver::instance().malloc(...);
    } else {
        // FALLBACK: Use normal cudaMalloc - outside region()  
        return APIForwarder::call_real_cuda_malloc(ptr, size);
    }
}
```

**When `is_interesting_region()` returns false:**
- **Outside `region()` context**: Not inside `with torch_memory_saver.region():`
- **Before initialization**: Before any region is created
- **Inside `disable()` context**: Explicitly disabled via `torch_memory_saver.disable()`

### 2. Torch Mode - Only Inside Regions

**Torch Mode Requirement (`csrc/entrypoint.cpp:70`):**
```cpp
void *tms_torch_malloc(ssize_t size, int device, cudaStream_t stream) {
    SIMPLE_CHECK(thread_local_config.is_interesting_region(), "only support interesting region");
    // Torch mode REQUIRES being inside region - no fallback, will error if outside
}
```

**Torch Mode Behavior:**
- **No fallback**: Torch mode doesn't fall back to cudaMalloc
- **Error if outside region**: Will crash with assertion error
- **Dedicated allocator**: PyTorch uses this allocator only for tensors in regions

### 3. Free Operations for Unknown Pointers

**Free Fallback (`csrc/core.cpp:151-153`):**
```cpp
cudaError_t TorchMemorySaver::free(void *ptr) {
    const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
    if (allocation_metadata_.count(ptr) == 0) {
        // FALLBACK: Pointer not allocated by us, use normal cudaFree
        return APIForwarder::call_real_cuda_free(ptr);
    }
    // ... handle our allocated pointer with VMM
}
```

**When this happens:**
- **Mixed allocations**: Some tensors allocated inside regions, others outside
- **Library interop**: Other CUDA libraries allocating memory
- **PyTorch internals**: PyTorch's own memory allocations outside regions

### 4. Environment-Based Initialization Fallback

**Thread-local initialization (`csrc/entrypoint.cpp:13-18`):**
```cpp
bool is_interesting_region() {
    if (!is_interesting_region_.has_value()) {
        is_interesting_region_ = get_bool_env_var("TMS_INIT_ENABLE"); // Check env var
    }
    return is_interesting_region_.value();
}
```

**Environment variable control:**
- **`TMS_INIT_ENABLE=1`**: Start with regions enabled (for testing)
- **`TMS_INIT_ENABLE=0`** or unset: Start with normal cudaMalloc until first region
- **`TMS_INIT_ENABLE_CPU_BACKUP=1`**: Enable CPU backup by default

### Summary of Fallback Scenarios

| Scenario | Preload Mode | Torch Mode | Reason |
|----------|-------------|------------|---------|
| Outside `region()` | ✅ Falls back to `cudaMalloc` | ❌ Errors out | Torch mode requires regions |
| Unknown pointer in `free()` | ✅ Falls back to `cudaFree` | ✅ Falls back to `cudaFree` | Not our allocation |
| Inside `disable()` context | ✅ Falls back to `cudaMalloc` | ❌ Not supported | Temporary disable |
| Before first region | ✅ Falls back to `cudaMalloc` | N/A | No allocator registered yet |

The fallback mechanism ensures **seamless interoperability** - only tensors explicitly created inside regions use VMM, while everything else works normally with standard CUDA memory allocation.