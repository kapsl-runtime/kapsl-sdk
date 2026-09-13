# CPU binding fixture

`profile-matmul.onnx` is the 194-byte synthetic fixture from
`kapsl-integrations` commit `819e8c713a00de36506016b8ca65b1c10e79d35b`,
`integrations/ort/tests/fixtures/profile-matmul.onnx`.
It multiplies a `[1, 4]` float input by a constant diagonal matrix with entries
`2`, producing `[2, 4, 6, 8]` for `[1, 2, 3, 4]`. It contains no model weights
from an external model or accelerator operators.

The binding regression intercepts the real CPU runtime's synchronization calls
to model a delayed input copy. It checks execution ordering and error handling;
it does not reproduce or qualify CUDA's asynchronous execution on a CPU.
