# Third-party notices

This extension contains code adapted from the projects below. Their license
texts are in [licenses/](licenses/) and are installed with the extension.

| Project | Used for | License | Revision | Source |
| --- | --- | --- | --- | --- |
| Gufo | Safetensors reader (`safetensors.cpp`), adapted from Qwen3-TTS by way of the VoxCPM extension | `MIT` | `5f9612c3b19ab6676b842095bc88bb9b32da5cf0` | [gufo-org/gufo](https://github.com/gufo-org/gufo) |
| transformers | `model.cpp` and `kernels/kernels.hip` port `Nemotron3_5AsrForRNNT` and `NemotronAsrStreamingEncoder` (cache-aware FastConformer encoder, language prompt projector, RNN-T prediction and joint networks, greedy decoding); `features.cpp` ports `NemotronAsrStreamingFeatureExtractor`; `tokenizer.cpp` follows its Metaspace decoding | `Apache-2.0` | `6d195c6262a0ac2ef44f7b9a308617562286ff45` (v5.22.2) | [huggingface/transformers](https://github.com/huggingface/transformers) |
| NeMo | Through transformers, which ports NVIDIA's cache-aware FastConformer, its causal convolutions, the RNN-T decoder and the greedy decoding rules from NeMo | `Apache-2.0` | `079e48770098a4cffcf542a4bc974cafc84502a3` | [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) |
| ESPnet | Through NeMo's relative-position multi-head attention, which NeMo adopts from ESPnet (`kernels/kernels.hip`, `model.cpp`) | `Apache-2.0` | (as credited in NeMo's `multi_head_attention.py`) | [espnet/espnet](https://github.com/espnet/espnet) |
| librosa | The Slaney mel filter bank (`features.cpp`), which the reference's feature extractor builds with librosa | `ISC` | `9caf46a4ecaeb8ea24bb541231065c9f761174ae` | [librosa/librosa](https://github.com/librosa/librosa) |

Copyright notices of the adapted code:

- transformers: Copyright 2026 The HuggingFace Inc. team.
- NeMo: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.
- ESPnet (in NeMo): Copyright 2017 Johns Hopkins University (Shinji Watanabe).
- librosa: Copyright (c) 2013--2026, librosa development team.

| License text | Projects |
| --- | --- |
| [licenses/gufo.txt](licenses/gufo.txt) | Gufo |
| [licenses/transformers.txt](licenses/transformers.txt) | transformers |
| [licenses/NeMo.txt](licenses/NeMo.txt) | NeMo, ESPnet |
| [licenses/librosa.txt](licenses/librosa.txt) | librosa |

The test fixtures in `tests/fixtures/` are two short speech clips that this
repository's author generated with the VoxCPM2 text-to-speech model (voice
design, no real person's voice) and the token IDs and float32 tensors
(features, encoder activations, logits) that `tools/reference/golden.py`
computed from them with the transformers implementation and the released
checkpoint. None of them contains weights.
