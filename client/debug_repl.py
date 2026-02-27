import os
import cmd
import pickle
import uuid
import traceback
import numpy as np
from tritonclient import grpc as grpcclient

class TritonReplayREPL(cmd.Cmd):
    """REPL for replaying Triton inference requests from pickled dumps."""
    
    intro = "Triton Replay REPL - Type 'help' for available commands."
    prompt = "triton> "
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dump_dir = os.path.join(os.path.dirname(__file__), "..", "replay_dumps")
        self.client = None
        self._connect_client()
    
    def do_set_dump_dir(self, arg):
        """Set the dumps directory. Usage: set_dump_dir <path>"""
        if not arg:
            print("Usage: set_dump_dir <path>")
            return
        
        arg = arg.strip()
        if not os.path.isdir(arg):
            print(f"✗ Directory does not exist: {arg}")
            return
        
        self.dump_dir = arg
        print(f"✓ Dumps directory set to: {self.dump_dir}")
    
    def do_replay(self, arg):
        """Replay an inference request from a pickled dump. Usage: replay <filename>"""
        if not arg:
            print("Usage: replay <filename>")
            return
        
        if self.client is None:
            print("✗ Not connected to Triton server. Cannot replay.")
            return
        
        filename = arg.strip()
        filepath = os.path.join(self.dump_dir, filename)
        
        # Add .pkl extension if not present
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        
        if not os.path.isfile(filepath):
            print(f"✗ File not found: {filepath}")
            print(f"  Looked in: {self.dump_dir}")
            return
        
        try:
            # Load the pickled data
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✓ Loaded dump from: {filepath}")
            
            # Send inference request
            self._send_inference(data["inputs"], data["model"])
            
        except Exception as e:
            print(f"✗ Error loading or replaying dump: {e}")
            traceback.print_exc()
    
    def do_list_dumps(self, arg):
        """List available dump files in the current dumps directory."""
        if not os.path.isdir(self.dump_dir):
            print(f"✗ Dumps directory not found: {self.dump_dir}")
            return
        
        files = [f for f in os.listdir(self.dump_dir) if f.endswith('.pkl')]
        
        if not files:
            print(f"No dump files found in {self.dump_dir}")
            return
        
        print(f"✓ Dump files in {self.dump_dir}:")
        for f in sorted(files):
            filepath = os.path.join(self.dump_dir, f)
            size = os.path.getsize(filepath)
            print(f"  {f} ({size} bytes)")
    
    def do_status(self, arg):
        """Show current status and configuration."""
        print("Current Configuration:")
        print(f"  Dumps directory: {self.dump_dir}")
        if self.client:
            try:
                server_live = self.client.is_server_live()
                server_ready = self.client.is_server_ready()
                print(f"  Triton server: Connected (live={server_live}, ready={server_ready})")
            except Exception as e:
                print(f"  Triton server: Connected but {e}")
        else:
            print(f"  Triton server: Not connected")
    
    def do_reconnect(self, arg):
        """Reconnect to the Triton server."""
        self._connect_client()
    
    def do_quit(self, arg):
        """Quit the REPL."""
        if self.client:
            self.client.close()
            self.client = None
        print("Goodbye!")
        return True
    
    def do_exit(self, arg):
        """Exit the REPL."""
        return self.do_quit(arg)
    
    def do_EOF(self, arg):
        """Handle EOF (Ctrl+D)."""
        print()
        return self.do_quit(arg)
    
    def emptyline(self):
        """Do nothing on empty input."""
        pass

    def _connect_client(self):
        """Connect to the Triton server."""
        if self.client:
            self.client.close()
            self.client = None

        try:
            self.client = grpcclient.InferenceServerClient("localhost:8001")
            print(f"✓ Connected to Triton server at localhost:8001")
        except Exception as e:
            print(f"✗ Failed to connect to Triton server: {e}")
            self.client = None
    
    def _send_inference(self, data, model_name, req_id=None):
        """Send inference request to Triton with the loaded data."""

        input_names, output_names = self._get_model_io(self.client.get_model_config(model_name))

        # Create Triton inputs for given data
        inputs = []
        for name in input_names:
            tensor_data = data[name]
            triton_dtype = self._get_triton_dtype(tensor_data.dtype)
            inputs.append(grpcclient.InferInput(name, tensor_data.shape, triton_dtype))
            inputs[-1].set_data_from_numpy(tensor_data)

        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]
        
        if req_id is None:
            req_id = str(uuid.uuid4())
        
        print(f"✓ Sending inference request with id: {req_id}")
        try:
            result = self.client.infer(model_name, inputs, outputs=outputs, request_id=req_id)
            print(f"✓ Inference successful for request id: {req_id}")
            for out in output_names:
                output_data = result.as_numpy(out)
                print(f"Output '{out}': shape={output_data.shape}, dtype={output_data.dtype}")
        
        except Exception as e:
            print(f"✗ Inference failed for request id: {req_id} with error: {e}")
            traceback.print_exc()

    def _get_model_io(self, model_config):
        """Extract input and output names and dtypes from model config."""
        input_names = [inp.name for inp in model_config.config.input]
        output_names = [out.name for out in model_config.config.output]
        return input_names, output_names

    def _get_triton_dtype(self, numpy_dtype):
        """Convert numpy dtype to Triton dtype string."""
        dtype_map = {
            np.float32: "FP32",
            np.float64: "FP64",
            np.int32: "INT32",
            np.int64: "INT64",
            np.uint32: "UINT32",
            np.uint64: "UINT64",
            np.int8: "INT8",
            np.uint8: "UINT8",
        }
        
        # Convert numpy dtype to numpy type
        if hasattr(numpy_dtype, 'type'):
            numpy_type = numpy_dtype.type
        else:
            numpy_type = numpy_dtype
        
        for np_type, triton_type in dtype_map.items():
            if numpy_type == np_type:
                return triton_type
        
        # Default to FP32
        return "FP32"

if (__name__ == "__main__"):
    repl = TritonReplayREPL()
    repl.cmdloop()
