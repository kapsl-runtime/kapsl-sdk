"""Keep Kapsl's OIP descriptors separate from other clients such as Triton."""

from google.protobuf.descriptor_pool import DescriptorPool

POOL = DescriptorPool()
