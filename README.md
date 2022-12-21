# Swap Engine

## Routing Frequency Tests

### Data Format

Saved with numpy, that create max portability. Contains `routes` (one hot matrix [batch_size, seq_len, num_experts]), `hidden_states` (float32 or float16 [batch_size, seq_len, hidden_size]), `route_prob_max` (float32 or float16 [batch_size, seq_len]).
```python
# @param routes: list of routes
# @param layer_name: name of the layer, encoder or decoder
# @param layer_idx: index of the layer, always odd number
# @param req_id_md5: md5 of the request id, use save id for data in difference layers
np.save("{data_path}/routes_{layer_name}_{layer_idx}_{req_id_md5}", routes, allow_pickle=False)
```

