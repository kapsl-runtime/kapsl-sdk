#include "kapsl_kv_native_v1.h"

#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(kapsl_kv_geometry_v1) == 64u, "geometry layout changed");
_Static_assert(sizeof(kapsl_kv_pool_v1) == 56u, "pool layout changed");
_Static_assert(sizeof(kapsl_kv_reserve_request_v1) == 40u, "request layout changed");
_Static_assert(sizeof(kapsl_kv_reservation_v1) == 16u, "reservation layout changed");
_Static_assert(sizeof(kapsl_native_kv_host_v1) == 88u, "host layout changed");
_Static_assert(offsetof(kapsl_native_kv_host_v1, create_pool) == 32u, "host callback moved");
#endif

static int32_t create_pool(void *context, const kapsl_kv_geometry_v1 *geometry,
                           kapsl_kv_pool_v1 *pool) {
    (void)context;
    (void)geometry;
    (void)pool;
    return KAPSL_KV_STATUS_UNSUPPORTED;
}

int main(void) {
    kapsl_native_kv_host_v1 host = {0};
    host.struct_size = (uint32_t)sizeof(host);
    host.abi_version = KAPSL_NATIVE_KV_VERSION;
    host.create_pool = create_pool;
    return host.create_pool(NULL, NULL, NULL) == KAPSL_KV_STATUS_UNSUPPORTED ? 0 : 1;
}
