"""Let a released vLLM (through v0.26.0) serve a TEXT-ONLY Qwen3.5 checkpoint.

Those releases only register the multimodal Qwen3.5 classes, so a text-only
checkpoint is loaded through Qwen3_5ForConditionalGeneration + the Qwen3-VL
code path. With `--language-model-only` the language model loads fine, but two
spots in that path unconditionally read vision-only config fields that a
Qwen3_5TextConfig does not have (vLLM #39231):

  1) Qwen3_5ForConditionalGeneration.__init__ builds the vision tower
     (config.vision_config) -> AttributeError.
  2) Qwen3VL._get_mrope_input_positions reads config.video_token_id /
     vision_config.spatial_merge_size while computing mrope positions, even for
     pure-text input -> AttributeError.

This patch:
  1) Skips the vision tower when there is no vision_config.
  2) Short-circuits mrope position computation for text-only requests
     (no multimodal features) to the exact sequential 3D positions the function
     would otherwise produce, without touching any vision config field.

Both are idempotent and abort loudly if an anchor is missing.

Upstream registered Qwen3_5ForCausalLM as a text-generation architecture in
vllm-project/vllm#50210, merged to main just after v0.26.0. This script is only
needed until that reaches a release.
"""

import pathlib
import sys

MODELS = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models")


def patch(path: pathlib.Path, old: str, new: str, marker: str, label: str) -> None:
    src = path.read_text()
    if marker in src:
        print(f"[patch] {label}: already applied")
        return
    count = src.count(old)
    if count == 0:
        sys.exit(f"[patch] {label}: anchor not found (file layout changed?)")
    path.write_text(src.replace(old, new))
    print(f"[patch] {label}: patched {count} occurrence(s)")


# 1) Skip the vision tower for text-only configs ----------------------------
patch(
    MODELS / "qwen3_5.py",
    old="""        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )""",
    new="""        if getattr(config, "vision_config", None) is None:
            self.visual = None
        else:
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )""",
    marker='vision_config", None) is None',
    label="qwen3_5: skip vision tower",
)

# 2) Short-circuit mrope position computation for text-only requests ---------
patch(
    MODELS / "qwen3_vl.py",
    old="""    @staticmethod
    def _get_mrope_input_positions(
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
        config: Qwen3VLConfig,
    ):
        llm_pos_ids_list = []""",
    new="""    @staticmethod
    def _get_mrope_input_positions(
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
        config: Qwen3VLConfig,
    ):
        if not mm_features:
            llm_positions = np.broadcast_to(
                np.arange(len(input_tokens)), (3, len(input_tokens))
            ).copy()
            return torch.from_numpy(llm_positions), 0
        llm_pos_ids_list = []""",
    marker="if not mm_features:",
    label="qwen3_vl: text-only mrope short-circuit",
)
