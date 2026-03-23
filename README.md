# sonic-inference-replay
An implementation of serializing failed inference requests in a Triton server, so they can be re-run in a debugging environment.

### Generating random inputs for requests
It is possible to use the debug REPL to send a request with randomly-generated inputs of specified shape.
The command in the repl is:
```
request_random <model_name> <input_name> <input_shape> [<input_name> <input_shape> ...]
```
The input names and shapes must be paired, and all must be specified and match what the model expects.
Shapes should be specified in Python's tuple syntax, enclosed in quotes.

A concrete example for the `particlenet_AK4_PT` model is the following:
```
triton> request_random particlenet_AK4_PT pf_points__0 "(4, 2, 100)" pf_features__1 "(4, 20, 100)" pf_mask__2 "(4, 1, 100)" sv_points__3 "(4, 2, 10)" sv_features__4 "(4, 11, 10)" sv_mask__5 "(4, 1, 10)"
```
