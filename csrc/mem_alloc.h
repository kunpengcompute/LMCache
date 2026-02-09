#include <cstdint>

uintptr_t alloc_pinned_ptr(size_t size, unsigned int flags);
uintptr_t alloc_numa_ptr(size_t size, int node);
uintptr_t alloc_pinned_numa_ptr(size_t size, int node);

void free_pinned_ptr(uintptr_t ptr);
void free_numa_ptr(uintptr_t ptr, size_t size);
void free_pinned_numa_ptr(uintptr_t ptr, size_t size);

typedef void *(*alloc_func_t)(size_t size);
typedef void (*free_func_t)(void *ptr);

void register_cm_memory_func(uintptr_t alloc_func_addr, uintptr_t free_func_addr);
uintptr_t cm_alloc_ptr(size_t size);
void cm_free_ptr(uintptr_t ptr);