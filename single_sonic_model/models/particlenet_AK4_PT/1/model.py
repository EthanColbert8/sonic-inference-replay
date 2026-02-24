import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda:0")

        self.model = torch.jit.load("/models/particlenet_AK4_PT/1/model.pt", map_location=self.device).to(self.device)
        self.model.eval()

        model_config = json.loads(args["model_config"])
        output_config = pb_utils.get_output_config_by_name(model_config, "softmax__0")

        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])

    def execute(self, requests):
        responses = []

        with torch.no_grad():
            for request in requests:
                # Get input tensors
                pf_points = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "pf_points__0").as_numpy()).to(self.device)
                pf_features = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "pf_features__1").as_numpy()).to(self.device)
                pf_mask = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "pf_mask__2").as_numpy()).to(self.device)
                sv_points = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "sv_points__3").as_numpy()).to(self.device)
                sv_features = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "sv_features__4").as_numpy()).to(self.device)
                sv_mask = torch.from_numpy(pb_utils.get_input_tensor_by_name(request, "sv_mask__5").as_numpy()).to(self.device)

                # Pass through the model
                output_tensor = self.model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)
                
                # Format output properly
                output = pb_utils.Tensor("softmax__0", output_tensor.cpu().numpy().astype(self.output_dtype))
                response = pb_utils.InferenceResponse(output_tensors=[output])
                responses.append(response)

        return responses