#ifndef KAPSL_KV_NATIVE_V1_H
#define KAPSL_KV_NATIVE_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KAPSL_NATIVE_KV_EXTENSION_NAME "kapsl-kv-native"
#define KAPSL_NATIVE_KV_VERSION 1u
#define KAPSL_KV_STATUS_OK 0
#define KAPSL_KV_STATUS_INVALID_ARGUMENT 1
#define KAPSL_KV_STATUS_INCOMPATIBLE_ABI 2
#define KAPSL_KV_STATUS_UNSUPPORTED 3
#define KAPSL_KV_STATUS_BACKEND_ERROR 4
#define KAPSL_KV_STATUS_CANCELLED 5
#define KAPSL_KV_STATUS_PANIC 6
#define KAPSL_KV_RESERVE_UPLOAD 1u

typedef struct kapsl_kv_geometry_v1 {
    uint32_t struct_size;
    uint32_t device_id;
    uint64_t requested_blocks;
    uint32_t block_size_tokens;
    uint32_t num_layers;
    uint32_t num_kv_heads;
    uint32_t key_head_dim;
    uint32_t value_head_dim;
    uint32_t element_bytes;
    uint32_t max_sequences;
    uint32_t max_blocks_per_sequence;
    uint64_t model_fingerprint;
    uint32_t flags;
    uint32_t reserved;
} kapsl_kv_geometry_v1;

typedef struct kapsl_kv_pool_v1 {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t pool_id;
    void *device_base;
    uint64_t addressable_blocks;
    uint32_t *block_table_device;
    uint32_t block_table_layer_stride;
    uint32_t block_table_sequence_stride;
    uint32_t sequence_slots;
    uint32_t reserved_tail;
} kapsl_kv_pool_v1;

/* request_id is engine-issued and nonzero. sequence_id is never reused in a
 * pool; sequence_slot may be reused only after successful sequence release. */
typedef struct kapsl_kv_reserve_request_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t model_id;
    uint32_t replica_id;
    uint64_t request_id;
    uint64_t sequence_id;
    uint32_t sequence_slot;
    uint32_t tokens_needed;
} kapsl_kv_reserve_request_v1;

typedef struct kapsl_kv_reservation_v1 {
    uint32_t struct_size;
    uint32_t logical_blocks;
    uint32_t *block_table_device;
} kapsl_kv_reservation_v1;

typedef int32_t (*kapsl_kv_create_pool_fn)(
    void *, const kapsl_kv_geometry_v1 *, kapsl_kv_pool_v1 *);
typedef int32_t (*kapsl_kv_reserve_fn)(
    void *, uint64_t, const kapsl_kv_reserve_request_v1 *, kapsl_kv_reservation_v1 *);
typedef int32_t (*kapsl_kv_commit_fn)(void *, uint64_t, uint32_t **);
typedef int32_t (*kapsl_kv_sequence_fn)(void *, uint64_t, uint64_t);
typedef int32_t (*kapsl_kv_pool_bytes_fn)(void *, uint64_t, uint64_t *);
typedef int32_t (*kapsl_kv_destroy_pool_fn)(void *, uint64_t);

/* Query this immutable table through kapsl_backend_host_extensions_v1. It and
 * its context remain live through adapter shutdown. All callbacks return a
 * status; failures preserve ownership/accounting and leave outputs untouched.
 * A successful release/destroy includes synchronization before storage reuse.
 */
typedef struct kapsl_native_kv_host_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t device_id;
    uint32_t model_id;
    uint32_t replica_id;
    uint32_t reserved;
    void *user_data;
    kapsl_kv_create_pool_fn create_pool;
    kapsl_kv_reserve_fn reserve;
    kapsl_kv_commit_fn commit;
    kapsl_kv_sequence_fn release;
    kapsl_kv_sequence_fn touch;
    kapsl_kv_pool_bytes_fn pool_bytes;
    kapsl_kv_destroy_pool_fn destroy_pool;
} kapsl_native_kv_host_v1;

#ifdef __cplusplus
}
#endif

#endif
