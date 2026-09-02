#include "kapsl_backend_v1.h"

#include <stddef.h>
#include <stdint.h>

_Static_assert(KAPSL_BACKEND_ABI_VERSION == 1u, "unexpected backend ABI version");
_Static_assert(KAPSL_BACKEND_ENTRYPOINT_MAGIC == 0x4b424e44u, "unexpected magic");
_Static_assert(KAPSL_BACKEND_DESCRIPTOR_SCHEMA_V1 == 1u,
               "unexpected descriptor schema");
_Static_assert(KAPSL_DTYPE_UTF8 == 14u, "unexpected UTF-8 dtype value");
_Static_assert(offsetof(kapsl_backend_api_v1, describe) == 24u,
               "function table prefix changed");

#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(kapsl_slice) == 16u, "kapsl_slice layout changed");
_Static_assert(sizeof(kapsl_owned_buffer) == 24u, "owned buffer layout changed");
_Static_assert(sizeof(kapsl_device_allocation_request_v1) == 48u,
               "allocation request layout changed");
_Static_assert(sizeof(kapsl_device_allocation_v1) == 32u,
               "allocation identity layout changed");
_Static_assert(sizeof(kapsl_backend_host_v1) == 48u, "host table layout changed");
_Static_assert(sizeof(kapsl_device_allocation_scope_v1) == 40u,
               "allocation scope layout changed");
_Static_assert(sizeof(kapsl_scoped_device_allocation_request_v1) == 80u,
               "scoped allocation request layout changed");
_Static_assert(offsetof(kapsl_backend_host_scoped_allocator_v1, base) == 0u,
               "scoped host prefix moved");
_Static_assert(sizeof(kapsl_backend_host_scoped_allocator_v1) == 64u,
               "scoped host extension layout changed");
_Static_assert(sizeof(kapsl_backend_config_v1) == 80u, "config layout changed");
_Static_assert(sizeof(kapsl_tensor_view_v1) == 56u, "tensor view layout changed");
_Static_assert(sizeof(kapsl_named_tensor_view_v1) == 80u,
               "named tensor view layout changed");
_Static_assert(sizeof(kapsl_inference_request_v1) == 64u, "request layout changed");
_Static_assert(sizeof(kapsl_inference_result_v1) == 40u, "result layout changed");
_Static_assert(sizeof(kapsl_inference_batch_v1) == 16u, "batch layout changed");
_Static_assert(sizeof(kapsl_inference_batch_result_v1) == 24u,
               "batch result layout changed");
_Static_assert(sizeof(kapsl_backend_api_v1) == 192u, "API table layout changed");
#endif

static void consume_api(const kapsl_backend_api_v1 *api) {
    if (api != NULL && api->magic == KAPSL_BACKEND_ENTRYPOINT_MAGIC) {
        (void)api->capabilities;
    }
}

int main(void) {
    kapsl_tensor_view_v1 tensor = {0};
    kapsl_inference_result_v1 result = {0};
    kapsl_backend_host_scoped_allocator_v1 host = {0};
    tensor.struct_size = (uint32_t)sizeof(tensor);
    result.struct_size = (uint32_t)sizeof(result);
    host.base.struct_size = (uint32_t)sizeof(host);
    host.base.abi_version = KAPSL_BACKEND_ABI_VERSION;
    host.scoped_allocator_version = KAPSL_SCOPED_DEVICE_ALLOCATOR_VERSION;
    consume_api(NULL);
    return (tensor.struct_size == 0u || result.struct_size == 0u ||
            host.base.struct_size == 0u)
               ? 1
               : 0;
}
