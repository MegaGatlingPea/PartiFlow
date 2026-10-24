from functools import partial
from typing import Any
import torch
from loguru import logger
from .batching import (
    batch_size_from_any,
    chunk_batch,
    concat_tensor_dicts,
    replicate_batch,
)


def resolve_bestofn_replicas(search_cfg):
    value = search_cfg.get("best_of_n", {}).get("replicas", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("replicas must be a positive integer")
    return value


class BestOfNSearch:
    """Complete-trajectory replica sampling: expand each input sample into ``replicas`` copies."""

    def __init__(self, model, search_cfg):
        self.model = model
        self.search_cfg = search_cfg

    def search(self, batch: Any) -> dict[str, Any]:
        replicas = resolve_bestofn_replicas(self.search_cfg)
        if replicas < 1:
            raise ValueError(f"best-of-n replicas must be >= 1, got {replicas}")

        max_batch_size = self.search_cfg.get("max_batch_size", None)
        max_batch_size = int(max_batch_size) if max_batch_size is not None else None
        nsamples = batch_size_from_any(batch)
        total = nsamples * replicas

        logger.info(
            "[BestOfN] nsamples={}, replicas={}, total={}, max_batch_size={}",
            nsamples,
            replicas,
            total,
            max_batch_size,
        )

        all_tags = [
            f"bon_orig{s}_r{r}" for s in range(nsamples) for r in range(replicas)
        ]
        expanded_batch = replicate_batch(batch, replicas)
        chunks = (
            chunk_batch(expanded_batch, max_batch_size)
            if max_batch_size is not None
            else [expanded_batch]
        )

        predict_fn = partial(
            self.model.predict_for_sampling,
            n_recycle=self.model.inf_cfg.get("n_recycle", 0),
        )
        nsteps = int(self.model.inf_cfg.args.nsteps)
        self_cond = bool(self.model.inf_cfg.args.self_cond)
        guidance_w = float(self.model.inf_cfg.args.get("guidance_w", 1.0))
        ag_ratio = float(self.model.inf_cfg.args.get("ag_ratio", 0.0))

        final_parts: list[dict[str, Any]] = []
        tag_offset = 0
        src_offset = 0
        for chunk in chunks:
            gen_samples, extra_info = self.model.fm.full_simulation(
                batch=chunk,
                predict_for_sampling=predict_fn,
                nsteps=nsteps,
                self_cond=self_cond,
                sampling_model_args=self.model.inf_cfg.model,
                device=self.model.device,
                save_trajectory_every=0,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
                motif_constraint_mode=self.model.motif_constraint_mode,
            )
            chunk_final = self.model.sample_formatting(
                x=gen_samples,
                extra_info=extra_info,
                ret_mode="coors37_n_aatype",
            )
            chunk_size = int(chunk_final["coors"].shape[0])
            chunk_final["metadata_tag"] = all_tags[tag_offset : tag_offset + chunk_size]
            chunk_final["source_index"] = torch.tensor(
                [idx // replicas for idx in range(src_offset, src_offset + chunk_size)],
                device=chunk_final["coors"].device,
                dtype=torch.long,
            )
            final_parts.append(chunk_final)
            tag_offset += chunk_size
            src_offset += chunk_size

        final = (
            final_parts[0]
            if len(final_parts) == 1
            else concat_tensor_dicts(final_parts)
        )
        if len(final["metadata_tag"]) != total:
            raise ValueError(
                f"[BestOfN] metadata tags ({len(final['metadata_tag'])}) do not match total outputs ({total})"
            )
        return {
            "final": final,
            "streams": [
                {
                    "sample_prots": final,
                    "sample_type": "final",
                    "rankable": True,
                    "stream_subdir": None,
                    "step_start": None,
                    "step_end": None,
                }
            ],
        }
