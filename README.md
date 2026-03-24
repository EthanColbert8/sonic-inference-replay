# sonic-inference-replay
An implementation of serializing failed inference requests in a Triton server, so they can be re-run in a debugging environment.

## The Client
The main user-facing utility in this repo is [`client/debug_repl.py`](https://github.com/EthanColbert8/sonic-inference-replay/blob/main/client/debug_repl.py), which starts a REPL that connects to a _local_ Triton server (looks for gRPC service on `localhost:8001`) and allows one to inspect dumps of previous requests, resend them to the server, or send random inputs in a new request.

The interactive commands are:

- `replay` (Usage: `replay <filename_or_id>`): rerun a specified dump, via file name or ID.
- `request_random` (Usage explained below): Generate random inputs and send as an inference request. Usage is documented below (slightly more complicated as input shapes must be given).
- `list_dumps`: List all dumps visible to the program.
- `inspect_dump` (usage: `inspect_dump <filename_or_id>`): Print the request ID, model name, error message, and input shapes and dtypes from a dump of a previous request.
- `get_models`: Print the model repository index from Triton server (all models Triton sees, and their state).
- `get_model_info` (Usage: `get_model_info <model_name>`): Print the config Triton is currently using for the given model.
- `last_request`: Get the ID of the last request sent to the model in this session.
- `status`: Print the status of the REPL program (state of connection to Triton)
- `reconnect`: Attempt to reconnect to Triton server.
- `set_dump_dir` (Usage: `set_dump_dir <directory>`): Set the directory where the REPL will look for dumps. Note that this will NOT affect where the Triton server will save the dumps. This must be set at server startup time.

### Generating random inputs for requests
It is possible to use the debug REPL to send a request with randomly-generated inputs of specified shape.
The command in the repl is:
```
request_random <model_name> <input_name> <input_shape> [<input_name> <input_shape> ...]
```
The input names and shapes must be paired, and all must be specified and match what the model expects.
Shapes should be specified in Python's tuple syntax, enclosed in quotes.

A concrete example for the `particlenet_AK4_PT` model, with batch size 4, is the following:
```
triton> request_random particlenet_AK4_PT pf_points__0 "(4, 2, 100)" pf_features__1 "(4, 20, 100)" pf_mask__2 "(4, 1, 100)" sv_points__3 "(4, 2, 10)" sv_features__4 "(4, 11, 10)" sv_mask__5 "(4, 1, 10)"
```
