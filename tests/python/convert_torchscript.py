import torch
import transformers
from transformers import AutoTokenizer
from transformers import SwitchTransformersForConditionalGeneration

MODEL_NAME = "switch-base-256"

model = SwitchTransformersForConditionalGeneration.from_pretrained(
    f"google/{MODEL_NAME}", cache_dir="/mnt/data/xly/.cache"
)

input_ids = torch.ones((1, 128), dtype=torch.long)
attention_mask = torch.ones((1, 128), dtype=torch.long)
decoder_input_ids = input_ids.clone()
decoder_input_ids = decoder_input_ids[:, :32]
decoder_attention_mask = torch.ones_like(decoder_input_ids)

# input_ids: Optional[torch.LongTensor] = None,
# attention_mask: Optional[torch.FloatTensor] = None,
# decoder_input_ids: Optional[torch.LongTensor] = None,
# decoder_attention_mask: Optional[torch.BoolTensor] = None,
# head_mask: Optional[torch.FloatTensor] = None,
# decoder_head_mask: Optional[torch.FloatTensor] = None,
# cross_attn_head_mask: Optional[torch.Tensor] = None,
# encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
# past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
# inputs_embeds: Optional[torch.FloatTensor] = None,
# decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
# labels: Optional[torch.LongTensor] = None,
# use_cache: Optional[bool] = None,
# output_attentions: Optional[bool] = None,
# output_hidden_states: Optional[bool] = None,
# output_router_logits: Optional[bool] = True,
# return_dict: Optional[bool] = None,

module = torch.jit.trace(
    model, (
        input_ids, # input_ids
        attention_mask,  # attention_maskq
        decoder_input_ids ,  # decoder_input_ids
        decoder_attention_mask,  # decoder_attention_mask
    )
)
module.save(f"/mnt/data/xly/.cache/{MODEL_NAME}.pt")
