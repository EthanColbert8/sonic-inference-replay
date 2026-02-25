import uuid
import numpy as np
import tritonclient.grpc as grpcclient

batch_size = 10

with grpcclient.InferenceServerClient("localhost:8001") as client:
    inputs = [
        grpcclient.InferInput("pf_points__0", [batch_size, 2, 100], "FP32"),
        grpcclient.InferInput("pf_features__1", [batch_size, 20, 100], "FP32"),
        grpcclient.InferInput("pf_mask__2", [batch_size, 1, 100], "FP32"),
        grpcclient.InferInput("sv_points__3", [batch_size, 2, 10], "FP32"),
        grpcclient.InferInput("sv_features__4", [batch_size, 11, 10], "FP32"),
        grpcclient.InferInput("sv_mask__5", [batch_size, 1, 10], "FP32"),
    ]

    outputs = [grpcclient.InferRequestedOutput("softmax__0")]

    pf_points = np.random.randn(batch_size, 2, 100).astype(np.float32)
    pf_features = np.random.randn(batch_size, 20, 100).astype(np.float32)
    pf_mask = np.random.randn(batch_size, 1, 100).astype(np.float32)
    sv_points = np.random.randn(batch_size, 2, 10).astype(np.float32)
    sv_features = np.random.randn(batch_size, 11, 10).astype(np.float32)
    sv_mask = np.random.randn(batch_size, 1, 10).astype(np.float32)

    inputs[0].set_data_from_numpy(pf_points)
    inputs[1].set_data_from_numpy(pf_features)
    inputs[2].set_data_from_numpy(pf_mask)
    inputs[3].set_data_from_numpy(sv_points)
    inputs[4].set_data_from_numpy(sv_features)
    inputs[5].set_data_from_numpy(sv_mask)

    req_id = str(uuid.uuid4())
    print(f"Sending request with id: {req_id}")

    result = client.infer("particlenet_AK4_PT", inputs, outputs=outputs, request_id=req_id)

    output_data = result.as_numpy("softmax__0")

print("Got outputs:")
print(output_data)
