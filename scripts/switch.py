# Copyright 2022 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=line-too-long
# pyformat: disable
r"""This script runs inference-evaluation on a T5X-compatible model.

"""
# pyformat: enable
# pylint:enable=line-too-long

import functools
import os
import re
import string
import sys
import tensorflow as tf
import numpy as np
import torch
import gc
from typing import Callable, Mapping, Optional, Sequence, Tuple, Type

# pylint:disable=g-import-not-at-top
# TODO(adarob): Re-enable once users are notified and tests are updated.
os.environ["FLAX_LAZY_RNG"] = "no"
from absl import logging
from clu import metric_writers
import jax
import seqio
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import train_state as train_state_lib
from t5x import utils
from typing_extensions import Protocol

from pyutils.ckpt_load import copy_t5x_weights
from pysrc.transformer.switch import SwitchConfig

# Automatically search for gin files relative to the T5X package.
home = os.path.expanduser("~")
_DEFAULT_GIN_SEARCH_PATHS = [home, f"{home}/flaxformer/"]


def tqa_open_preprocessor(
    dataset: tf.data.Dataset, prefix: str = "trivia_qa question: "
) -> tf.data.Dataset:
    @seqio.map_over_dataset
    def tqa_map(ex: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
        """Map TriviaQA example to text-to-text example."""
        return {
            "inputs": prefix + ex["question"],
            "targets": ex["answer"]["value"],
            "answers": ex["answer"]["aliases"],
        }

    return tqa_map(dataset)


def tqa_open_postprocessor(output_or_target, example=None, is_target=False):
    """Returns output as answer, or all answers if the full example is provided."""
    if is_target:
        return [a.decode("utf-8") for a in example["answers"]]
    else:
        return output_or_target.decode("utf-8")


def tqa_metric(
    targets: Sequence[Sequence[str]], predictions: Sequence[str]
) -> Mapping[str, seqio.metrics.MetricValue]:
    """Computes official TriviaQA metrics.

    Args:
      targets: list of lists of strings
      predictions: list of strings

    Returns:
      dict with score_key: squad score across all targets and predictions
    """

    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")

    def _normalize_answer(text):
        """Lower text and remove punctuation, articles and extra whitespace."""
        # Remove articles.
        text = re.sub(r"\b(a|an|the)\b", " ", s)
        # Remove punctuation.
        for punc in string.punctuation:
            text = text.replace(punc, "")
        # Normalize white space
        text = " ".join(s.split())
        return text

    # Normalize answers before comparing.
    targets = [[_normalize_answer(t) for t in u] for u in targets]
    predictions = [_normalize_answer(p) for p in predictions]

    em = np.mean(
        [
            max(pred == gt for gt in ground_truths)
            for pred, ground_truths in zip(predictions, targets)
        ]
    )
    return {
        "exact_match": seqio.metrics.Scalar(em),
    }


vocabulary = seqio.SentencePieceVocabulary(
    "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model", extra_ids=100
)

seqio.TaskRegistry.add(
    "trivia_qa_open",
    source=seqio.TfdsDataSource(
        tfds_name="trivia_qa/unfiltered.nocontext:1.1.0",
        splits={
            "train": "train[:90%]",
            "validation": "train[90%:]",
            "test": "validation",
        },
    ),
    preprocessors=[
        tqa_open_preprocessor,
        seqio.preprocessors.tokenize,
        seqio.preprocessors.append_eos,
    ],
    output_features={
        "inputs": seqio.Feature(vocabulary=vocabulary, add_eos=False, dtype=tf.int32),
        "targets": seqio.Feature(vocabulary=vocabulary, add_eos=True, dtype=tf.int32),
    },
    postprocess_fn=tqa_open_postprocessor,
    metric_fns=[tqa_metric],
)


class SummarizeConfigFn(Protocol):
    def __call__(
        self,
        model_dir: str,
        summary_writer: Optional[metric_writers.SummaryWriter],
        step: int,
    ) -> None:
        ...


class InferenceEvaluator:
    """Runs evaluation of the model against a given SeqIo task."""

    def __init__(
        self,
        infer_eval_dataset_cfg: utils.DatasetConfig,
        inference_evaluator_cls: utils.EvaluatorConstructor,
        model: models.BaseModel,
        partitioner: partitioning.BasePartitioner,
        log_dir: Optional[str] = None,
        verify_matching_vocabs_fn: Optional[
            Callable[[utils.DatasetConfig, models.BaseModel], None]
        ] = utils.verify_matching_vocabs,
    ):
        """Constructs inference evaluator.

        Args:
          infer_eval_dataset_cfg: Specification for the dataset to evaluate with
            using the inference metrics (e.g., uses sampled decoding). If None,
            inference eval is disabled.
          inference_evaluator_cls: seqio.Evaluator class to use for inference
            evaluation, potentially with bound configuration args.
          model: Model to be evaluated.
          partitioner: the partitioner to use.
          log_dir: Parent directory to log evaluation results.
          verify_matching_vocabs_fn: Function to validate whether the task
            vocabulary matches the model vocabulary. Should raise an exception on
            error.
        """
        if verify_matching_vocabs_fn is not None:
            verify_matching_vocabs_fn(infer_eval_dataset_cfg, model)

        self._model = model
        self._partitioner = partitioner
        self._infer_eval_dataset_cfg = infer_eval_dataset_cfg
        kwargs = {}
        if log_dir:
            kwargs["log_dir"] = os.path.join(log_dir, "inference_eval")
        else:
            # Disable loggers if log dir is not provided.
            kwargs["logger_cls"] = ()
        self._seqio_evaluator = inference_evaluator_cls(
            mixture_or_task_name=infer_eval_dataset_cfg.mixture_or_task_name,
            feature_converter=model.FEATURE_CONVERTER_CLS(pack=False),
            eval_split=infer_eval_dataset_cfg.split,
            use_cached=infer_eval_dataset_cfg.use_cached,
            seed=infer_eval_dataset_cfg.seed,
            sequence_length=infer_eval_dataset_cfg.task_feature_lengths,
            use_memory_cache=infer_eval_dataset_cfg.use_memory_cache,
            **kwargs,
        )
        # Lazily initialized upon the first `evaluate` call.
        self._predict_fn = None
        self._predict_with_aux_fn = None
        self._score_fn = None

    @property
    def model_feature_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return self._seqio_evaluator.model_feature_shapes

    @property
    def eval_tasks(self) -> Sequence[seqio.Task]:
        return self._seqio_evaluator.eval_tasks

    def close(self):
        self._seqio_evaluator.close()

    def evaluate(
        self,
        train_state: train_state_lib.TrainState,
        train_state_axes: train_state_lib.TrainState,
    ) -> seqio.evaluation.AllMetricsFuture:
        """Runs the prediction based inference eval.

        Args:
          train_state: Training state to run evaluation of.
          train_state_axes: partitioning info for the train state to be used.

        Returns:
          A dictionary of training eval metrics.
        """
        if not self._predict_fn:
            self._predict_fn = utils.get_infer_fn(
                infer_step=self._model.predict_batch,
                batch_size=self._infer_eval_dataset_cfg.batch_size,
                train_state_axes=train_state_axes,
                partitioner=self._partitioner,
            )

            self._predict_with_aux_fn = utils.get_infer_fn(
                infer_step=self._model.predict_batch_with_aux,
                batch_size=self._infer_eval_dataset_cfg.batch_size,
                train_state_axes=train_state_axes,
                partitioner=self._partitioner,
            )

            self._score_fn = utils.get_infer_fn(
                infer_step=self._model.score_batch,
                batch_size=self._infer_eval_dataset_cfg.batch_size,
                train_state_axes=train_state_axes,
                partitioner=self._partitioner,
            )

        all_metrics, _ = self._seqio_evaluator.evaluate(
            compute_metrics=jax.process_index() == 0,
            step=int(utils.get_local_data(train_state.step)),
            predict_fn=functools.partial(
                self._predict_fn, train_state=train_state, rng=jax.random.PRNGKey(0)
            ),
            score_fn=functools.partial(self._score_fn, train_state=train_state),
            predict_with_aux_fn=functools.partial(
                self._predict_with_aux_fn,
                train_state=train_state,
                rng=jax.random.PRNGKey(0),
            ),
        )
        return all_metrics


# def jax2tensor(arr):
#     """Converts a JAX array to a PyTorch tensor."""
#     return torch.from_numpy(jax.device_get(arr))


# CKPT_PATH = "/mnt/raid0nvme1/xly/checkpoints/t5x-torchscript/moe/base/e128"

# def copy_ds_weights(source_state_dict):
#     from pysrc.transformer.switch import SwitchModelDeepSpeed, SwitchConfig

#     config = SwitchConfig.from_pretrained("config/t5x/base")
#     print("config", config)
#     model = SwitchModelDeepSpeed(config)

#     token_embedder = {
#         "embedding.weight": jax2tensor(
#             source_state_dict["token_embedder"].pop("embedding")
#         )
#     }
#     model.encoder_layers[0].load_state_dict(token_embedder)
#     model.decoder_layers[0].load_state_dict(token_embedder)

#     gc.collect()

#     k = 1
#     for i in range(config.num_layers):
#         gc.collect()
#         # attention layers
#         jax_attention_layer = source_state_dict["encoder"][f"layers_{i}"].pop(
#             "attention"
#         )
#         torch_attention_layer = {
#             "SelfAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "SelfAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
#             "SelfAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "SelfAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["encoder"][f"layers_{i}"]["pre_attention_layer_norm"][
#                     "scale"
#                 ]
#             ),
#         }
#         if (i==0):
#             torch_attention_layer["SelfAttention.relative_attention_bias.weight"] = jax2tensor(
#                 source_state_dict["encoder"]["relpos_bias"]["rel_embedding"]
#             ).transpose(0, 1)
#         print(type(model.encoder_layers[i + k]))
#         model.encoder_layers[i + k].attention.load_state_dict(torch_attention_layer)

#         # feed forward layers
#         if i % 2 == 1:
#             k += 1
#             jax_router_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"][
#                 "router"
#             ]
#             jax_mlp_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"].pop(
#                 "expert"
#             )
#             wi = jax2tensor(jax_mlp_layer["wi"]["kernel"])
#             wo = jax2tensor(jax_mlp_layer["wo"]["kernel"])

#             torch_router_layer = {
#                 "deepspeed_moe.gate.wg.weight": jax2tensor(
#                     jax_router_layer["router_weights"]["w"]["kernel"]
#                 ).transpose(0, 1),
#                 **{
#                     f"deepspeed_moe.experts.deepspeed_experts.{j}.wi.weight": wi[j].transpose(0, 1) for j in range(config.num_experts)
#                 },
#                 **{ f"deepspeed_moe.experts.deepspeed_experts.{j}.wo.weight": wo[j].transpose(0, 1) for j in range(config.num_experts)
#                 }
#             }
#             # print(torch_router_layer.keys())
#             print(type(model.encoder_layers[i + k]))
#             model.encoder_layers[i + k].load_state_dict(torch_router_layer)
#         else:
#             # normal MLP
#             k += 1
#             jax_mlp_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"]
#             torch_mlp_layer = {
#                 "DenseReluDense.wi.weight": jax2tensor(
#                     jax_mlp_layer["wi"]["kernel"]
#                 ).transpose(0, 1),
#                 "DenseReluDense.wo.weight": jax2tensor(
#                     jax_mlp_layer["wo"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["encoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             print(type(model.encoder_layers[i + k]))
#             model.encoder_layers[i + k].load_state_dict(torch_mlp_layer)

#     k += 1  # final layer norm
#     torch_final_layer = {
#         "layer_norm.scale": jax2tensor(
#             source_state_dict["encoder"]["encoder_norm"]["scale"]
#         ),
#     }
#     model.encoder_layers[-1].load_state_dict(torch_final_layer)

#     # decoder layers
#     k = 1
#     for i in range(config.num_layers):
#         gc.collect()
#         # attention layers
#         jax_attention_layer = source_state_dict["decoder"][f"layers_{i}"].pop(
#             "self_attention"
#         )
#         torch_attention_layer = {
#             "SelfAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "SelfAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
#             "SelfAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "SelfAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["decoder"][f"layers_{i}"][
#                     "pre_self_attention_layer_norm"
#                 ]["scale"]
#             ),
#         }
#         if (i==0):
#             torch_attention_layer["SelfAttention.relative_attention_bias.weight"] = jax2tensor(
#                 source_state_dict["encoder"]["relpos_bias"]["rel_embedding"]
#             ).transpose(0, 1)
#         model.decoder_layers[i + k].attention.load_state_dict(torch_attention_layer)

#         jax_attention_layer = source_state_dict["decoder"][f"layers_{i}"].pop(
#             "encoder_decoder_attention"
#         )
#         torch_attention_layer = {
#             "EncDecAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "EncDecAttention.k.weight": jax2tensor(
#                 jax_attention_layer["key"]["kernel"]
#             ),
#             "EncDecAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "EncDecAttention.o.weight": jax2tensor(
#                 jax_attention_layer["out"]["kernel"]
#             ),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["decoder"][f"layers_{i}"][
#                     "pre_cross_attention_layer_norm"
#                 ]["scale"]
#             ),
#         }
#         model.decoder_layers[i + k].cross_attention.load_state_dict(torch_attention_layer)

#         # feed forward layers
#         if i % 2 == 1:
#             k += 1
#             jax_router_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"][
#                 "router"
#             ]
#             jax_mlp_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"].pop(
#                 "expert"
#             )
#             wi = jax2tensor(jax_mlp_layer["wi"]["kernel"])
#             wo = jax2tensor(jax_mlp_layer["wo"]["kernel"])

#             torch_router_layer = {
#                 "deepspeed_moe.gate.wg.weight": jax2tensor(
#                     jax_router_layer["router_weights"]["w"]["kernel"]
#                 ).transpose(0, 1),
#                 **{
#                     f"deepspeed_moe.experts.deepspeed_experts.{j}.wi.weight": wi[j].transpose(0, 1) for j in range(config.num_experts)
#                 },
#                 **{ f"deepspeed_moe.experts.deepspeed_experts.{j}.wo.weight": wo[j].transpose(0, 1) for j in range(config.num_experts)
#                 }
#             }
#             # print(torch_router_layer.keys())
#             model.decoder_layers[i + k].load_state_dict(torch_router_layer)
#         else:
#             # normal MLP
#             k += 1
#             jax_mlp_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"]
#             torch_mlp_layer = {
#                 "DenseReluDense.wi.weight": jax2tensor(
#                     jax_mlp_layer["wi"]["kernel"]
#                 ).transpose(0, 1),
#                 "DenseReluDense.wo.weight": jax2tensor(
#                     jax_mlp_layer["wo"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["decoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             model.decoder_layers[i + k].load_state_dict(torch_mlp_layer)

#     k += 1  # final layer norm
#     torch_final_layer = {
#         "layer_norm.scale": jax2tensor(
#             source_state_dict["decoder"]["decoder_norm"]["scale"]
#         ),
#     }
#     model.decoder_layers[-1].load_state_dict(torch_final_layer)


#     torch.save(model.state_dict(), os.path.join(CKPT_PATH, "model.pth"))

# def copy_model_weights(source_state_dict):
#     # append workspace folder to system path
#     # sys.path.append(os.getcwd())
#     from pysrc.transformer.switch import SwitchModel, SwitchConfig

#     config = SwitchConfig.from_pretrained("config/t5x/base")
#     print("config", config)
#     model = SwitchModel(config)

#     # shared embedding layers

#     # convert ShardedDeviceArray to torch Tensor
#     # source_state_dict["token_embedder"] = jax.device_get(source_state_dict["token_embedder"])
#     # print(type(source_state_dict["token_embedder"]))
#     token_embedder = {
#         "embedding.weight": jax2tensor(
#             source_state_dict["token_embedder"].pop("embedding")
#         )
#     }
#     model.encoder_layers[0].load_state_dict(token_embedder)
#     model.decoder_layers[0].load_state_dict(token_embedder)

#     torch.jit.save(
#         torch.jit.script(model.encoder_layers[0]),
#         os.path.join(CKPT_PATH, "encoder_token_embedder.pt"),
#     )
#     torch.jit.save(
#         torch.jit.script(model.decoder_layers[0]),
#         os.path.join(CKPT_PATH, "decoder_token_embedder.pt"),
#     )

#     # encoder layers
#     k = 1
#     for i in range(config.num_layers):
#         # attention layers
#         jax_attention_layer = source_state_dict["encoder"][f"layers_{i}"].pop(
#             "attention"
#         )
#         torch_attention_layer = {
#             "SelfAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "SelfAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
#             "SelfAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "SelfAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["encoder"][f"layers_{i}"]["pre_attention_layer_norm"][
#                     "scale"
#                 ]
#             ),
#         }
#         if (i==0):
#             torch_attention_layer["SelfAttention.relative_attention_bias.weight"] = jax2tensor(
#                 source_state_dict["encoder"]["relpos_bias"]["rel_embedding"]
#             ).transpose(0, 1)
#         model.encoder_layers[i + k].attention.load_state_dict(torch_attention_layer)
#         torch.jit.save(
#             torch.jit.script(model.encoder_layers[i + k]),
#             os.path.join(CKPT_PATH, f"encoder_layer{i}_attention.pt"),
#         )

#         # feed forward layers
#         if i % 2 == 1:
#             # Router
#             k += 1
#             jax_router_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"][
#                 "router"
#             ]
#             torch_router_layer = {
#                 "router.weight": jax2tensor(
#                     jax_router_layer["router_weights"]["w"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["encoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             model.encoder_layers[i + k].load_state_dict(torch_router_layer)
#             torch.jit.save(
#                 torch.jit.script(model.encoder_layers[i + k]),
#                 os.path.join(CKPT_PATH, f"encoder_layer{i}_router.pt"),
#             )

#             # Experts
#             jax_mlp_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"].pop(
#                 "expert"
#             )
#             print(
#                 jax_mlp_layer["wi"]["kernel"].shape
#             )  # (num_experts, hidden_size, expert_dim)
#             wi = jax2tensor(jax_mlp_layer["wi"]["kernel"])
#             wo = jax2tensor(jax_mlp_layer["wo"]["kernel"])
#             for j in range(config.num_experts):
#                 k += 1
#                 torch_mlp_layer = {
#                     "wi.weight": wi[j].transpose(0, 1),
#                     "wo.weight": wo[j].transpose(0, 1),
#                 }
#                 model.encoder_layers[i + k].load_state_dict(torch_mlp_layer)
#                 torch.jit.save(
#                     torch.jit.script(model.encoder_layers[i + k]),
#                     os.path.join(CKPT_PATH, f"encoder_layer{i}_expert{j}.pt"),
#                 )

#             k += 1  # skip the aggregator layer
#         else:
#             # normal MLP
#             k += 1
#             jax_mlp_layer = source_state_dict["encoder"][f"layers_{i}"]["mlp"]
#             torch_mlp_layer = {
#                 "DenseReluDense.wi.weight": jax2tensor(
#                     jax_mlp_layer["wi"]["kernel"]
#                 ).transpose(0, 1),
#                 "DenseReluDense.wo.weight": jax2tensor(
#                     jax_mlp_layer["wo"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["encoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             model.encoder_layers[i + k].load_state_dict(torch_mlp_layer)
#             torch.jit.save(
#                 torch.jit.script(model.encoder_layers[i + k]),
#                 os.path.join(CKPT_PATH, f"encoder_layer{i}_ff.pt"),
#             )

#     k += 1  # final layer norm
#     torch_final_layer = {
#         "layer_norm.scale": jax2tensor(
#             source_state_dict["encoder"]["encoder_norm"]["scale"]
#         ),
#     }
#     model.encoder_layers[-1].load_state_dict(torch_final_layer)

#     # decoder layers
#     k = 1
#     for i in range(config.num_layers):
#         # attention layers
#         jax_attention_layer = source_state_dict["decoder"][f"layers_{i}"].pop(
#             "self_attention"
#         )
#         torch_attention_layer = {
#             "SelfAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "SelfAttention.k.weight": jax2tensor(jax_attention_layer["key"]["kernel"]),
#             "SelfAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "SelfAttention.o.weight": jax2tensor(jax_attention_layer["out"]["kernel"]),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["decoder"][f"layers_{i}"][
#                     "pre_self_attention_layer_norm"
#                 ]["scale"]
#             ),
#         }
#         if (i==0):
#             torch_attention_layer["SelfAttention.relative_attention_bias.weight"] = jax2tensor(
#                 source_state_dict["encoder"]["relpos_bias"]["rel_embedding"]
#             ).transpose(0, 1)
#         model.decoder_layers[i + k].attention.load_state_dict(torch_attention_layer)

#         jax_attention_layer = source_state_dict["decoder"][f"layers_{i}"].pop(
#             "encoder_decoder_attention"
#         )
#         torch_attention_layer = {
#             "EncDecAttention.q.weight": jax2tensor(
#                 jax_attention_layer["query"]["kernel"]
#             ),
#             "EncDecAttention.k.weight": jax2tensor(
#                 jax_attention_layer["key"]["kernel"]
#             ),
#             "EncDecAttention.v.weight": jax2tensor(
#                 jax_attention_layer["value"]["kernel"]
#             ),
#             "EncDecAttention.o.weight": jax2tensor(
#                 jax_attention_layer["out"]["kernel"]
#             ),
#             "layer_norm.scale": jax2tensor(
#                 source_state_dict["decoder"][f"layers_{i}"][
#                     "pre_cross_attention_layer_norm"
#                 ]["scale"]
#             ),
#         }
#         model.decoder_layers[i + k].cross_attention.load_state_dict(torch_attention_layer)
#         torch.jit.save(
#             torch.jit.script(model.decoder_layers[i + k]),
#             os.path.join(CKPT_PATH, f"decoder_layer{i}_cross_attention.pt"),
#         )

#         # feed forward layers
#         if i % 2 == 1:
#             # Router
#             k += 1
#             jax_router_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"][
#                 "router"
#             ]
#             torch_router_layer = {
#                 "router.weight": jax2tensor(
#                     jax_router_layer["router_weights"]["w"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["decoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             model.decoder_layers[i + k].load_state_dict(torch_router_layer)

#             # Experts
#             jax_mlp_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"].pop(
#                 "expert"
#             )
#             print(
#                 jax_mlp_layer["wi"]["kernel"].shape
#             )  # (num_experts, hidden_size, expert_dim)
#             wi = jax2tensor(jax_mlp_layer["wi"]["kernel"])
#             wo = jax2tensor(jax_mlp_layer["wo"]["kernel"])
#             for j in range(config.num_experts):
#                 k += 1
#                 torch_mlp_layer = {
#                     "wi.weight": wi[j].transpose(0, 1),
#                     "wo.weight": wo[j].transpose(0, 1),
#                 }
#                 model.decoder_layers[i + k].load_state_dict(torch_mlp_layer)

#             k += 1  # skip the aggregator layer
#         else:
#             # normal MLP
#             k += 1
#             jax_mlp_layer = source_state_dict["decoder"][f"layers_{i}"]["mlp"]
#             torch_mlp_layer = {
#                 "DenseReluDense.wi.weight": jax2tensor(
#                     jax_mlp_layer["wi"]["kernel"]
#                 ).transpose(0, 1),
#                 "DenseReluDense.wo.weight": jax2tensor(
#                     jax_mlp_layer["wo"]["kernel"]
#                 ).transpose(0, 1),
#                 "layer_norm.scale": jax2tensor(
#                     source_state_dict["decoder"][f"layers_{i}"]["pre_mlp_layer_norm"][
#                         "scale"
#                     ]
#                 ),
#             }
#             model.decoder_layers[i + k].load_state_dict(torch_mlp_layer)

#     k += 1  # final layer norm
#     torch_final_layer = {
#         "layer_norm.scale": jax2tensor(
#             source_state_dict["decoder"]["decoder_norm"]["scale"]
#         ),
#     }
#     model.decoder_layers[-1].load_state_dict(torch_final_layer)

    

#     # # save model in torchscript format
#     # torch.jit.save(torch.jit.script(model), os.path.join(CKPT_PATH, "model.pt"))
#     # save model as pytorch state dict
#     torch.save(model.state_dict(), os.path.join(CKPT_PATH, "model.pth"))

    

def evaluate(
    *,
    model: models.BaseTransformerModel,
    dataset_cfg: utils.DatasetConfig,
    restore_checkpoint_cfg: utils.RestoreCheckpointConfig,
    partitioner: partitioning.BasePartitioner,
    output_dir: str,
    inference_evaluator_cls: utils.EvaluatorConstructor = seqio.Evaluator,
    summarize_config_fn: SummarizeConfigFn = gin_utils.summarize_gin_config,
    train_state_initializer_cls: Type[
        utils.TrainStateInitializer
    ] = utils.TrainStateInitializer,
    fallback_init_rng: Optional[int] = None,
):
    """Evaluation function.

    Args:
      model: The model object to use for inference.
      dataset_cfg: Specification for the dataset to infer based on.
      restore_checkpoint_cfg: Specification for the model parameter checkpoint to
        load.
      partitioner: Partitioner for the model parameters and data across devices.
      output_dir: Path to directory to write temporary files and final results.
      inference_evaluator_cls: seqio.Evaluator class to use for inference
        evaluation, potentially with bound configuration args.
      summarize_config_fn: A function that takes in the model directory, an
        optional SummaryWriter, and the step number, and writes a summary of the
        configuration. SummaryWriter will be None in most cases.
      train_state_initializer_cls: t5x.utils.TrainStateInitializer class
        for initializing partitioned TrainState from checkpoints or scratch.
      fallback_init_rng: A random seed used for parameter initialization during
        model re-loading when utils.RestoreCheckpointConfig.fallback_to_scratch is
        set to True. If None, parameter initialization is not allowed during model
        loading and having fallback_to_scratch enabled will result in an error.
    """
    logging.info("Process ID: %d", jax.process_index())
    if dataset_cfg.module:
        utils.import_module(dataset_cfg.module)
    batch_size = dataset_cfg.batch_size

    # TODO(b/234480674): GDA not supported for eval.
    restore_checkpoint_cfg.use_gda = False

    summarize_config_fn(model_dir=output_dir, summary_writer=None, step=0)

    evaluator = InferenceEvaluator(
        dataset_cfg, inference_evaluator_cls, model, partitioner, log_dir=output_dir
    )
    if not evaluator.eval_tasks:
        raise ValueError(
            f"'{dataset_cfg.mixture_or_task_name}' has no metrics for evaluation."
        )

    # ----------------------------------------------------------------------------
    # T5X model loading.
    # ----------------------------------------------------------------------------

    # Initialize optimizer from the existing checkpoint.
    input_shapes = {
        k: (batch_size,) + s for k, s in evaluator.model_feature_shapes.items()
    }

    # input_shapes['encoder_input_tokens'] = (batch_size, 256)
    # input_shapes['decoder_input_tokens'] = (batch_size, 128)

    train_state_initializer = train_state_initializer_cls(
        optimizer_def=None,  # Do not load optimizer state.
        init_fn=model.get_initial_variables,
        input_shapes=input_shapes,
        partitioner=partitioner,
    )
    train_state_axes = train_state_initializer.train_state_axes
    # Log the variable shapes information and write to a file.
    log_file = os.path.join(output_dir, "model-info.txt")
    utils.log_model_info(
        log_file, train_state_initializer.global_train_state_shape, partitioner
    )

    # Disable strictness since we are dropping the optimizer state.
    restore_checkpoint_cfg.strict = False

    if fallback_init_rng is not None:
        fallback_init_rng = jax.random.PRNGKey(fallback_init_rng)
    for train_state in train_state_initializer.from_checkpoints(
        [restore_checkpoint_cfg], init_rng=fallback_init_rng
    ):
        # print("train_state", train_state.state_dict()["target"])
        flax_state = train_state.state_dict()["target"]

        config = SwitchConfig.from_pretrained("config/t5x/large/e128")
        copy_t5x_weights(config, "model_repo_t5x_large_e128", "t5x_large_e128", flax_state)
        # copy_model_weights(flax_state)
        # copy_ds_weights(flax_state)
        exit()
        # ----------------------------------------------------------------------------
        # Main evaluation loop
        # ----------------------------------------------------------------------------

        # Run final evaluation (with decoding) on the full eval dataset.
        host_step = int(utils.get_local_data(train_state.step))
        all_metrics = evaluator.evaluate(train_state, train_state_axes)
        all_metrics.result()  # Ensure metrics are finished being computed.
        # Wait until computations are done before continuing.
        utils.sync_global_devices(f"step_{host_step}:complete")

    logging.info("Finished.")


if __name__ == "__main__":
    from absl import app
    from absl import flags
    import gin

    FLAGS = flags.FLAGS

    jax.config.parse_flags_with_absl()

    flags.DEFINE_multi_string(
        "gin_file",
        default=None,
        help="Path to gin configuration file. Multiple paths may be passed and "
        "will be imported in the given order, with later configurations  "
        "overriding earlier ones.",
    )

    flags.DEFINE_multi_string(
        "gin_bindings", default=[], help="Individual gin bindings."
    )

    flags.DEFINE_list(
        "gin_search_paths",
        default=["."],
        help="Comma-separated list of gin config path prefixes to be prepended "
        "to suffixes given via `--gin_file`. If a file appears in. Only the "
        "first prefix that produces a valid path for each suffix will be "
        "used.",
    )

    flags.DEFINE_string(
        "tfds_data_dir",
        None,
        "If set, this directory will be used to store datasets prepared by "
        "TensorFlow Datasets that are not available in the public TFDS GCS "
        "bucket. Note that this flag overrides the `tfds_data_dir` attribute of "
        "all `Task`s.",
    )

    def main(argv: Sequence[str]):
        """Wrapper for pdb post mortems."""
        _main(argv)

    def _main(argv: Sequence[str]):
        """True main function."""
        if len(argv) > 1:
            raise app.UsageError("Too many command-line arguments.")

        if FLAGS.tfds_data_dir:
            seqio.set_tfds_data_dir_override(FLAGS.tfds_data_dir)

        # Create gin-configurable version of `eval`.
        evaluate_using_gin = gin.configurable(evaluate)

        gin_utils.parse_gin_flags(
            # User-provided gin paths take precedence if relative paths conflict.
            FLAGS.gin_search_paths + _DEFAULT_GIN_SEARCH_PATHS,
            FLAGS.gin_file,
            FLAGS.gin_bindings,
        )
        evaluate_using_gin()

    gin_utils.run(main)
    # logging.info("Finished.")
