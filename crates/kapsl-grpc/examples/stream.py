"""Generate the protobuf modules as described in docs/grpc.md before running."""
import argparse
import os
import struct

import grpc
import kapsl_inference_pb2_grpc
import open_inference_grpc_pb2 as inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="host:port")
    parser.add_argument("model", help="model name or numeric ID")
    parser.add_argument("prompt")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--tls", action="store_true")
    args = parser.parse_args()
    token = os.environ.get("KAPSL_API_TOKEN")
    metadata = [("authorization", f"Bearer {token}")] if token else []
    channel = (
        grpc.secure_channel(args.target, grpc.ssl_channel_credentials())
        if args.tls else grpc.insecure_channel(args.target)
    )
    with channel:
        request = inference.ModelInferRequest(
            model_name=args.model,
            inputs=[inference.ModelInferRequest.InferInputTensor(
                name=args.input_name, datatype="BYTES", shape=[1],
                contents=inference.InferTensorContents(
                    bytes_contents=[args.prompt.encode("utf-8")]
                ),
            )],
        )
        stub = kapsl_inference_pb2_grpc.KapslInferenceStub(channel)
        call = stub.InferStream(request, metadata=metadata, timeout=60)
        try:
            for response in call:
                raw = response.raw_output_contents[0]
                if response.outputs[0].datatype == "BYTES":
                    length, = struct.unpack_from("<I", raw)
                    if length != len(raw) - 4:
                        raise ValueError("Invalid BYTES output length")
                    print(raw[4:].decode("utf-8"), end="", flush=True)
                else:
                    print(response.outputs[0], raw)
            print()
        finally:
            call.cancel()


if __name__ == "__main__":
    main()
