"""Exercise an SDK echo server with generated Python or NVIDIA Triton clients."""
import argparse
from pathlib import Path
import queue
import struct
import subprocess
import sys
import tempfile
import threading


def generated(target, root):
    import grpc
    from grpc_tools import protoc

    with tempfile.TemporaryDirectory(prefix="kapsl-grpc-proto-") as output:
        proto = root / "crates/kapsl-grpc/proto"
        result = protoc.main([
            "protoc", f"-I{proto}", f"--python_out={output}",
            f"--grpc_python_out={output}", str(proto / "open_inference_grpc.proto"),
            str(proto / "kapsl_inference.proto"),
        ])
        assert result == 0
        sys.path.insert(0, output)
        import open_inference_grpc_pb2 as inference
        import open_inference_grpc_pb2_grpc as standard
        import kapsl_inference_pb2_grpc as streaming

        with grpc.insecure_channel(target) as channel:
            unary = standard.GRPCInferenceServiceStub(channel)
            assert unary.ServerLive(inference.ServerLiveRequest(), timeout=5).live
            text = "Hello 🧠"
            request = inference.ModelInferRequest(
                model_name="text", id="python-request",
                inputs=[inference.ModelInferRequest.InferInputTensor(
                    name="input", datatype="BYTES", shape=[1],
                    contents=inference.InferTensorContents(bytes_contents=[text.encode()]),
                )],
            )
            packets = list(streaming.KapslInferenceStub(channel).InferStream(request, timeout=5))
            assert len(packets) == 2
            for packet in packets:
                raw = packet.raw_output_contents[0]
                assert struct.unpack_from("<I", raw)[0] == len(raw) - 4
                assert raw[4:].decode() == text
                assert packet.id == "python-request"
                assert list(packet.outputs[0].shape) == [1]
            output = unary.ModelInfer(request, timeout=5)
            assert output.raw_output_contents[0] == packets[0].raw_output_contents[0]


def triton(target):
    import numpy as np
    import tritonclient.grpc as client

    with client.InferenceServerClient(url=target) as server:
        assert server.is_server_live(client_timeout=5)
        assert server.is_server_ready(client_timeout=5)
        assert server.is_model_ready("tensor", model_version="1", client_timeout=5)
        metadata = server.get_model_metadata("tensor", client_timeout=5)
        assert metadata.inputs[0].datatype == "UINT8"
        values = np.array([0, 127, 128, 255], dtype=np.uint8)
        tensor = client.InferInput("input", values.shape, "UINT8")
        tensor.set_data_from_numpy(values)
        result = server.infer("tensor", [tensor], model_version="1",
                              request_id="triton-request", client_timeout=5)
        np.testing.assert_array_equal(result.as_numpy("output"), values)
        assert result.get_response().id == "triton-request"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--client", choices=["generated", "triton"], required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    process = subprocess.Popen([str(args.server.resolve())], stdout=subprocess.PIPE, text=True)
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True).start()
        target = ready.get(timeout=15).strip()
        assert target.startswith("127.0.0.1:")
        if args.client == "generated":
            generated(target, root)
        else:
            triton(target)
        print(f"{args.client} gRPC interoperability passed")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


if __name__ == "__main__":
    main()
