import asyncio
from concurrent.futures import ThreadPoolExecutor
import shutil
import subprocess

import grpc
import pytest

from kapsl_sdk import KapslGrpcClient, AsyncKapslGrpcClient
from kapsl_sdk.grpc_protocol import inference, inference_grpc


@pytest.fixture
def tls_endpoint(tmp_path):
    openssl = shutil.which("openssl")
    if not openssl:
        pytest.skip("OpenSSL is required to generate temporary test certificates")
    config = tmp_path / "openssl.cnf"
    config.write_text("""
[req]
prompt = no
distinguished_name = dn
x509_extensions = extensions
[dn]
CN = localhost
[extensions]
basicConstraints = critical,CA:TRUE
subjectAltName = DNS:localhost,IP:127.0.0.1
extendedKeyUsage = serverAuth,clientAuth
""")
    key_path, cert_path = tmp_path / "key.pem", tmp_path / "cert.pem"
    subprocess.run([
        openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-config", str(config), "-keyout", str(key_path), "-out", str(cert_path),
    ], check=True, capture_output=True)
    key, cert = key_path.read_bytes(), cert_path.read_bytes()
    admitted = []

    class Health(inference_grpc.GRPCInferenceServiceServicer):
        def ServerLive(self, request, context):
            admitted.append(True)
            assert dict(context.invocation_metadata())["authorization"] == "Bearer tls-test-token"
            return inference.ServerLiveResponse(live=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        inference_grpc.add_GRPCInferenceServiceServicer_to_server(Health(), server)
        port = server.add_secure_port("127.0.0.1:0", grpc.ssl_server_credentials(
            [(key, cert)], root_certificates=cert, require_client_auth=True,
        ))
        server.start()
        try:
            yield f"localhost:{port}", key, cert, admitted
        finally:
            server.stop(0).wait()


def test_tls_and_mutual_tls_credentials(tls_endpoint):
    target, key, cert, admitted = tls_endpoint
    options = dict(
        api_token="Bearer tls-test-token", tls=True, root_certificates=cert,
        private_key=key, certificate_chain=cert, timeout_ms=2000,
    )
    with KapslGrpcClient(target, **options) as client:
        assert client.server_live()

    async def check_async():
        async with AsyncKapslGrpcClient(target, **options) as client:
            assert await client.server_live()
    asyncio.run(check_async())
    assert len(admitted) == 2

    with KapslGrpcClient(target, tls=True, root_certificates=cert, timeout_ms=1000) as client:
        with pytest.raises(grpc.RpcError) as error:
            client.server_live()
        # Windows may wait for the RPC deadline after a rejected TLS handshake.
        assert error.value.code() in (
            grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED,
        )
    assert len(admitted) == 2  # No certificate-less RPC reached the handler.
    with KapslGrpcClient(target, **options) as client:
        assert client.server_live()
    assert len(admitted) == 3


def test_invalid_tls_configuration_is_rejected():
    with pytest.raises(ValueError, match="requires tls=True"):
        KapslGrpcClient(root_certificates=b"certificate")
    with pytest.raises(ValueError, match="Both private_key and certificate_chain"):
        KapslGrpcClient(tls=True, private_key=b"private key")
