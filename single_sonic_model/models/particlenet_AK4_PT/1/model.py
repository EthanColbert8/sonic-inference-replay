import json
import numpy as np
import pickle
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        # Location to dump replay data on failed requests
        self.replay_dump_dir = "/dumps"
        self.device = torch.device("cuda:0")

        self.model = torch.jit.load("/models/particlenet_AK4_PT/1/model.pt", map_location=self.device).to(self.device)
        self.model.eval()

        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "softmax__0")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        # debug print
        print("Reached execute method")

        responses = []

        with torch.no_grad():
            for request in requests:
                request_inputs = {
                    "pf_points__0": pb_utils.get_input_tensor_by_name(request, "pf_points__0").as_numpy(),
                    "pf_features__1": pb_utils.get_input_tensor_by_name(request, "pf_features__1").as_numpy(),
                    "pf_mask__2": pb_utils.get_input_tensor_by_name(request, "pf_mask__2").as_numpy(),
                    "sv_points__3": pb_utils.get_input_tensor_by_name(request, "sv_points__3").as_numpy(),
                    "sv_features__4": pb_utils.get_input_tensor_by_name(request, "sv_features__4").as_numpy(),
                    "sv_mask__5": pb_utils.get_input_tensor_by_name(request, "sv_mask__5").as_numpy()
                }
                
                try:
                    # Get inputs as tensors
                    pf_points = torch.from_numpy(request_inputs["pf_points__0"]).to(self.device)
                    pf_features = torch.from_numpy(request_inputs["pf_features__1"]).to(self.device)
                    pf_mask = torch.from_numpy(request_inputs["pf_mask__2"]).to(self.device)
                    sv_points = torch.from_numpy(request_inputs["sv_points__3"]).to(self.device)
                    sv_features = torch.from_numpy(request_inputs["sv_features__4"]).to(self.device)
                    sv_mask = torch.from_numpy(request_inputs["sv_mask__5"]).to(self.device)

                    # Pass through the model
                    output_tensor = self.model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

                    # Format output properly
                    output = pb_utils.Tensor("softmax__0", output_tensor.cpu().numpy().astype(self.output_dtype))
                    response = pb_utils.InferenceResponse(output_tensors=[output])
                
                except Exception as ex:
                    err_msg = f"Error during inference: {str(ex)}"
                    response = pb_utils.InferenceResponse(error=pb_utils.TritonError(err_msg))

                    req_id = request.request_id() # Unknown whether this method exists
                    err_dict = {
                        "id": req_id,
                        "model": "particlenet_AK4_PT",
                        "message": err_msg,
                        "inputs": request_inputs
                    }

                    print("Hit error. Dumping replay file.")
                    with open(f"{self.replay_dump_dir}/{req_id}.pkl", "wb") as f:
                        pickle.dump(err_dict, f)
                
                responses.append(response)
        
        return responses
