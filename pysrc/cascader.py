import copy
from ctypes import Union
from functools import wraps
import time
from typing import Dict
import uuid
import numpy as np
from pysrc.connector import BaseConnector, DeepspeedLocalConnector, TritonLocalConnector
from scipy.special import softmax
from asgiref.sync import async_to_sync
from pyutils.timer import timeit


class CascadeHandler:
    def __init__(self, config, connector: BaseConnector):

        self.ensembles = config["ensembles"]
        self.num_ensembles = len(self.ensembles)
        self.ensemble_weight = config["ensemble_weight"]

        self.model_meta = []
        for model_name in self.ensembles:
            self.model_meta.append(config[model_name])

        self.connector = connector

    def _prepare_outputs(self, output_name):
        if "vit" in self.ensembles[0]:
            logits = np.zeros((1, 1000), dtype=np.float32)

        return {output_name: logits}

    @timeit
    def __call__(self, inputs: Dict[str, np.ndarray]):
        session_id = uuid.uuid4().hex

        batch_size = inputs[list(inputs.keys())[0]].shape[0]
        ensemble_outputs = None

        batch_mask = np.zeros((self.num_ensembles, batch_size))
        batch_mask[0, :] = 1  # WHERE TO ENTER
        batch_mask = batch_mask.astype(bool)

        for idx, model_name in enumerate(self.ensembles):
            local_mask = batch_mask[idx]

            if np.any(local_mask):

                # if (
                #     isinstance(self.connector, TritonLocalConnector)
                #     and "ensemble" not in self.ensembles[idx]
                # ):
                #     for i in range(self.model_meta[idx]["npart"]):
                #         cur_model_
                # 
                # name = model_name + f"_{i}"
                #         # outputs = self.connector.get_model_outputs_as_numpy(cur_model_name)
                #         # print(outputs)
                #         outputs = {"output": None}

                #         start_time = time.perf_counter()
                #         outputs = self.connector.infer(
                #             cur_model_name, inputs, outputs, session_id
                #         )
                #         # print(f"Time taken for connector inference: {time.perf_counter() - start_time}")
                #         # print(outputs)
                #         inputs = copy.deepcopy(outputs)
                #         inputs["input"] = inputs["output"]
                #         del inputs["output"]
                if isinstance(self.connector, TritonLocalConnector):
                    outputs = self._prepare_outputs("ouput")
                    outputs = async_to_sync(
                        self.connector.infer(model_name, inputs, outputs, session_id)
                    )
                else:
                    outputs = self._prepare_outputs("logits")
                    outputs = async_to_sync(
                        self.connector.infer(model_name, inputs, outputs, session_id)
                    )

                # print(outputs)
                outputs = outputs["output"]

                extended_mask, max_prob = self.offload_mask(outputs, local_mask, idx)
                ensemble_outputs = self.model_ensemble(
                    ensemble_outputs, outputs, local_mask, idx
                )

                num_next_models = self.num_ensembles - idx - 1
                if np.any(extended_mask) and num_next_models > 0:
                    batch_mask[idx] &= ~extended_mask
                    batch_mask[idx + 1] |= extended_mask
                    # batch_mask = self.update_batch_mask(
                    #     max_prob, batch_mask.copy(), extended_mask, idx
                    # )
                    # self.logger.trace(
                    #     "%s batch_mask updated %s", options.name, batch_mask
                    # )
            # print(options.name, num_next_models, batch_mask)
            assert np.sum(batch_mask) == batch_size

        return ensemble_outputs

    def offload_mask(self, logits, mask, idx):
        probabilities = np.power(softmax(logits), 2)
        max_prob = np.max(probabilities, axis=-1)
        prob_mask = max_prob < self.model_meta[idx]["threshold"]
        extended_mask = mask & prob_mask
        return extended_mask, max_prob

    def model_ensemble(self, ensemble_outputs, local_outputs, mask, idx):
        # start_time = time.perf_counter()
        if ensemble_outputs is not None:
            ensemble_outputs[mask] = (
                ensemble_outputs[mask] * (1 - self.ensemble_weight)
                + local_outputs * self.ensemble_weight
            )
        return (
            ensemble_outputs if ensemble_outputs is not None else local_outputs.copy()
        )  # MEMCOPY

    # def update_batch_mask(self, max_prob, mask, local_mask, idx):
    #     num_next_models = self.num_ensembles - idx - 1

    #     if num_next_models <= 0:
    #         return mask

    #     if self.ensembles[idx].skip_connection:
    #         base_step = (self.ensembles[idx].threshold - 0.25) / num_next_models
    #         for skip in range(num_next_models):
    #             skip_th_lower = base_step * (num_next_models - 1 - skip) + 0.25
    #             skip_th_upper = base_step * (num_next_models - skip) + 0.25
    #             skip_mask = (
    #                 (max_prob >= skip_th_lower)
    #                 & (max_prob < skip_th_upper)
    #                 & local_mask
    #             )
    #             self.logger.trace(
    #                 "%s skip_th_lower %s, skip_th_upper %s, skip_mask %s",
    #                 self.ensembles[idx].name,
    #                 skip_th_lower,
    #                 skip_th_upper,
    #                 skip_mask,
    #             )
    #             mask[skip + 1 + idx] |= skip_mask
    #     else:
    #         mask[1 + idx] |= (max_prob < self.ensembles[idx].threshold) & local_mask

    #     mask[idx] &= ~local_mask
    #     return mask
