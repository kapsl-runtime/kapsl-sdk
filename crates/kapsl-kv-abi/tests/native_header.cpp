#include "kapsl_kv_native_v1.h"

#if UINTPTR_MAX == UINT64_MAX
static_assert(sizeof(kapsl_kv_geometry_v1) == 64u, "geometry layout changed");
static_assert(sizeof(kapsl_kv_pool_v1) == 56u, "pool layout changed");
static_assert(sizeof(kapsl_kv_reserve_request_v1) == 40u, "request layout changed");
static_assert(sizeof(kapsl_kv_reservation_v1) == 16u, "reservation layout changed");
static_assert(sizeof(kapsl_native_kv_host_v1) == 88u, "host layout changed");
static_assert(offsetof(kapsl_native_kv_host_v1, create_pool) == 32u, "host callback moved");
#endif

int main() {
    kapsl_native_kv_host_v1 host{};
    host.struct_size = static_cast<uint32_t>(sizeof(host));
    host.abi_version = KAPSL_NATIVE_KV_VERSION;
    return host.abi_version == 1u ? 0 : 1;
}
