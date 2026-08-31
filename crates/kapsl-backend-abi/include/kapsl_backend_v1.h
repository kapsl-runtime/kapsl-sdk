#ifndef KAPSL_BACKEND_V1_H
#define KAPSL_BACKEND_V1_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KAPSL_BACKEND_ABI_VERSION 1u
#define KAPSL_BACKEND_ENTRYPOINT_MAGIC 0x4b424e44u
#define KAPSL_BACKEND_ENTRYPOINT_SYMBOL "kapsl_backend_v1"
#define KAPSL_BACKEND_WIRE_FORMAT_TENSORS_V1 1u

#define KAPSL_STATUS_OK 0
#define KAPSL_STATUS_INVALID_ARGUMENT 1
#define KAPSL_STATUS_INCOMPATIBLE_ABI 2
#define KAPSL_STATUS_UNSUPPORTED 3
#define KAPSL_STATUS_BACKEND_ERROR 4
#define KAPSL_STATUS_CANCELLED 5
#define KAPSL_STATUS_PANIC 6

#define KAPSL_BACKEND_CAP_CPU (UINT64_C(1) << 0)
#define KAPSL_BACKEND_CAP_CUDA (UINT64_C(1) << 1)
#define KAPSL_BACKEND_CAP_TENSORRT (UINT64_C(1) << 2)
#define KAPSL_BACKEND_CAP_BATCHING (UINT64_C(1) << 3)
#define KAPSL_BACKEND_CAP_STREAMING (UINT64_C(1) << 4)
#define KAPSL_BACKEND_CAP_CANCELLATION (UINT64_C(1) << 5)
#define KAPSL_BACKEND_CAP_MEMORY_REPORTING (UINT64_C(1) << 6)
#define KAPSL_BACKEND_CAP_GOVERNED_DEVICE_ALLOCATOR (UINT64_C(1) << 7)
#define KAPSL_BACKEND_CAP_KV_PARTICIPANT (UINT64_C(1) << 8)
#define KAPSL_BACKEND_CAP_CONCURRENT_INFERENCE (UINT64_C(1) << 9)

#define KAPSL_MEMORY_HOST 1u
#define KAPSL_MEMORY_HOST_PINNED 2u
#define KAPSL_MEMORY_CUDA 3u
#define KAPSL_MEMORY_PROVIDER 4u

#define KAPSL_DTYPE_BOOL 1u
#define KAPSL_DTYPE_U8 2u
#define KAPSL_DTYPE_I8 3u
#define KAPSL_DTYPE_U16 4u
#define KAPSL_DTYPE_I16 5u
#define KAPSL_DTYPE_U32 6u
#define KAPSL_DTYPE_I32 7u
#define KAPSL_DTYPE_U64 8u
#define KAPSL_DTYPE_I64 9u
#define KAPSL_DTYPE_F16 10u
#define KAPSL_DTYPE_BF16 11u
#define KAPSL_DTYPE_F32 12u
#define KAPSL_DTYPE_F64 13u
#define KAPSL_DTYPE_UTF8 14u

#define KAPSL_TENSOR_FLAG_CONTIGUOUS (UINT32_C(1) << 0)
#define KAPSL_TENSOR_FLAG_READ_ONLY (UINT32_C(1) << 1)

#define KAPSL_ALLOCATION_CLASS_WEIGHTS 1u
#define KAPSL_ALLOCATION_CLASS_WORKSPACE 2u
#define KAPSL_ALLOCATION_CLASS_KV 3u
#define KAPSL_ALLOCATION_CLASS_REQUEST 4u
#define KAPSL_ALLOCATION_CLASS_OTHER 5u

#define KAPSL_LOG_ERROR 1u
#define KAPSL_LOG_WARN 2u
#define KAPSL_LOG_INFO 3u
#define KAPSL_LOG_DEBUG 4u
#define KAPSL_LOG_TRACE 5u

typedef struct kapsl_slice {
    const uint8_t *ptr;
    size_t len;
} kapsl_slice;

typedef struct kapsl_owned_buffer {
    uint8_t *ptr;
    size_t len;
    size_t capacity;
} kapsl_owned_buffer;

typedef void (*kapsl_log_fn)(void *user_data, uint32_t level, kapsl_slice message);
typedef uint32_t (*kapsl_request_cancelled_fn)(void *user_data, uint64_t request_id);
typedef void (*kapsl_free_buffer_fn)(kapsl_owned_buffer buffer);

typedef struct kapsl_device_allocation_request_v1 {
    uint32_t struct_size;
    uint32_t device_id;
    uint32_t memory_kind;
    uint32_t allocation_class;
    uint32_t model_id;
    uint32_t replica_id;
    uint32_t flags;
    uint32_t reserved;
    uint64_t bytes;
    uint64_t alignment;
} kapsl_device_allocation_request_v1;

typedef struct kapsl_device_allocation_v1 {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t allocation_id;
    void *device_ptr;
    uint64_t granted_bytes;
} kapsl_device_allocation_v1;

typedef int32_t (*kapsl_device_allocate_fn)(
    void *user_data,
    const kapsl_device_allocation_request_v1 *request,
    kapsl_device_allocation_v1 *allocation_out);
typedef int32_t (*kapsl_device_free_fn)(
    void *user_data,
    const kapsl_device_allocation_v1 *allocation);
typedef int32_t (*kapsl_device_synchronize_fn)(void *user_data, uint32_t device_id);

typedef struct kapsl_backend_host_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    void *user_data;
    kapsl_log_fn log;
    kapsl_device_allocate_fn allocate_device;
    kapsl_device_free_fn free_device;
    kapsl_device_synchronize_fn synchronize_device;
} kapsl_backend_host_v1;

typedef struct kapsl_backend_config_v1 {
    uint32_t struct_size;
    uint32_t device_id;
    uint32_t model_id;
    uint32_t replica_id;
    uint32_t require_governed_device_memory;
    uint32_t reserved;
    kapsl_slice profile;
    kapsl_slice manifest_json;
    kapsl_slice options_json;
    const kapsl_backend_host_v1 *host;
} kapsl_backend_config_v1;

typedef struct kapsl_tensor_view_v1 {
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t memory_kind;
    uint32_t flags;
    int32_t device_id;
    uint32_t rank;
    const int64_t *shape;
    const int64_t *strides;
    const void *data;
    uint64_t byte_len;
} kapsl_tensor_view_v1;

typedef struct kapsl_named_tensor_view_v1 {
    uint32_t struct_size;
    uint32_t reserved;
    kapsl_slice name;
    kapsl_tensor_view_v1 tensor;
} kapsl_named_tensor_view_v1;

typedef struct kapsl_inference_request_v1 {
    uint32_t struct_size;
    uint32_t wire_format;
    uint64_t request_id;
    const kapsl_named_tensor_view_v1 *inputs;
    uint32_t input_count;
    uint32_t reserved;
    kapsl_slice metadata_json;
    void *cancellation_context;
    kapsl_request_cancelled_fn is_cancelled;
} kapsl_inference_request_v1;

typedef struct kapsl_inference_result_v1 {
    uint32_t struct_size;
    uint32_t output_count;
    const kapsl_named_tensor_view_v1 *outputs;
    kapsl_slice metadata_json;
    void *owner_context;
} kapsl_inference_result_v1;

typedef struct kapsl_inference_batch_v1 {
    uint32_t struct_size;
    uint32_t request_count;
    const kapsl_inference_request_v1 *requests;
} kapsl_inference_batch_v1;

typedef struct kapsl_inference_batch_result_v1 {
    uint32_t struct_size;
    uint32_t result_count;
    const kapsl_inference_result_v1 *results;
    void *owner_context;
} kapsl_inference_batch_result_v1;

typedef int32_t (*kapsl_backend_describe_fn)(
    kapsl_owned_buffer *descriptor_json_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_initialize_fn)(
    const kapsl_backend_config_v1 *config,
    void **handle_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_path_report_fn)(
    void *handle,
    kapsl_slice model_path,
    kapsl_owned_buffer *report_json_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_load_model_fn)(
    void *handle,
    kapsl_slice model_path,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_request_report_fn)(
    void *handle,
    const kapsl_inference_request_v1 *request,
    kapsl_owned_buffer *report_json_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_infer_fn)(
    void *handle,
    const kapsl_inference_request_v1 *request,
    kapsl_inference_result_v1 *result_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_infer_batch_fn)(
    void *handle,
    const kapsl_inference_batch_v1 *batch,
    kapsl_inference_batch_result_v1 *result_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_stream_chunk_fn)(
    void *user_data,
    uint64_t request_id,
    const kapsl_inference_result_v1 *result);
typedef int32_t (*kapsl_backend_infer_stream_fn)(
    void *handle,
    const kapsl_inference_request_v1 *request,
    void *user_data,
    kapsl_backend_stream_chunk_fn on_chunk,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_cancel_fn)(void *handle, uint64_t request_id);
typedef int32_t (*kapsl_backend_json_report_fn)(
    void *handle,
    kapsl_owned_buffer *report_json_out,
    kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_health_fn)(void *handle, kapsl_owned_buffer *error_out);
typedef int32_t (*kapsl_backend_unload_fn)(void *handle, kapsl_owned_buffer *error_out);
typedef void (*kapsl_backend_shutdown_fn)(void *handle);
typedef void (*kapsl_backend_release_result_fn)(
    void *handle,
    kapsl_inference_result_v1 *result);
typedef void (*kapsl_backend_release_batch_result_fn)(
    void *handle,
    kapsl_inference_batch_result_v1 *result);

typedef struct kapsl_backend_api_v1 {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t wire_format;
    uint64_t capabilities;
    kapsl_backend_describe_fn describe;
    kapsl_backend_initialize_fn initialize;
    kapsl_backend_path_report_fn planned_memory;
    kapsl_backend_load_model_fn load_model;
    kapsl_backend_request_report_fn planned_request_memory;
    kapsl_backend_infer_fn infer;
    kapsl_backend_infer_batch_fn infer_batch;
    kapsl_backend_infer_stream_fn infer_stream;
    kapsl_backend_cancel_fn cancel;
    kapsl_backend_json_report_fn actual_memory;
    kapsl_backend_json_report_fn metrics;
    kapsl_backend_json_report_fn model_info;
    kapsl_backend_json_report_fn kv_capabilities;
    kapsl_backend_json_report_fn kv_topology;
    kapsl_backend_json_report_fn batching_policy;
    kapsl_backend_health_fn health_check;
    kapsl_backend_unload_fn unload;
    kapsl_backend_shutdown_fn shutdown;
    kapsl_backend_release_result_fn release_result;
    kapsl_backend_release_batch_result_fn release_batch_result;
    kapsl_free_buffer_fn free_buffer;
} kapsl_backend_api_v1;

typedef const kapsl_backend_api_v1 *(*kapsl_backend_entrypoint_v1_fn)(void);

/* Every native backend pack exports this symbol with default visibility. */
const kapsl_backend_api_v1 *kapsl_backend_v1(void);

#ifdef __cplusplus
}
#endif

#endif
