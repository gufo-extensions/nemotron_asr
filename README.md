# Nemotron ASR Gufo extension

A community [Gufo](https://github.com/gufo-org/gufo) extension for
[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b),
NVIDIA's multilingual streaming speech recognition model, on AMD Strix Halo
(`gfx1151`).

> **Status:** 0.1.0 builds against Gufo 0.34.0 (extension API 0.3).
> Transcription runs through `gufo serve asr`, for uploads and live over
> the realtime WebSocket, and matches the reference implementation's
> offline transcription token for token on the test clips. A 10 s upload
> takes 27 ms, five minutes 0.64 s, four to six times faster than
> audio.cpp with its Q8_0 model. See
> [quality](docs/models/nemotron-3.5-asr/QUALITY.md),
> [benchmarks](docs/models/nemotron-3.5-asr/BENCHMARKS.md) and the known
> limits below.

## Model

| Provider | Model | Files |
| --- | --- | --- |
| `nemotron-asr` | Nemotron 3.5 ASR Streaming 0.6B (cache-aware FastConformer, RNN-T, 40 languages) | The Hugging Face directory [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) |

The encoder turns 16 kHz audio into 128-band log-mel features, subsamples
them eight times to one **frame** per 80 ms and runs 24 FastConformer layers
whose attention sees 56 frames back and a fixed **lookahead** of 0, 3, 6 or
13 frames ahead. The audio is encoded in **chunks** of lookahead + 1 frames;
each chunk reuses the cached attention keys and convolution states of the
ones before it, so a chunk is never computed twice. A one-hot language
prompt (a locale or automatic detection) is projected into the encoder
output, and an RNN-T prediction network (two LSTM layers) and joint network
emit the text with punctuation and capitalization, greedily, up to ten tokens
per frame.

The provider loads the Hugging Face directory: `config.json`,
`processor_config.json`, `tokenizer.json` and `model.safetensors` (float32,
2.6 GB). On the GPU the matrices are float16, multiplied on the matrix cores
with float32 sums; activations, norms, caches and the LSTM state stay
float32 (1.5 GB of GPU memory in all). The `.nemo` archive and the GGUF
file in the same repository are not used.

The weights aren't part of this repository. NVIDIA publishes them under the
OpenMDW-1.1 license.

## Use

```sh
MODEL=/models/nemotron-3.5-asr-streaming-0.6b   # the Hugging Face directory
gufo serve asr --model "$MODEL" --port 8080
curl -s localhost:8080/v1/audio/transcriptions -F file=@talk.wav -F language=de
```

- `language` takes a code (`de`), a locale (`de-DE`), an English name
  (`German`) or nothing for automatic detection. Other values are rejected.
  `verbose_json` reports the detected language.
- Uploads may be WAV, FLAC, MP3 or Ogg at any sample rate and are
  resampled to 16 kHz. They are transcribed at the widest lookahead
  (1.12 s), the most accurate, in passes of up to 16 chunks. Gufo accepts
  request bodies up to 8 MiB by default, about 4.4 minutes of 16 kHz mono
  WAV; raise `--max-request-bytes` for longer recordings.
- `stream=true` sends the text pass by pass (about 4.5 s of audio each).
- Live transcription (`GET /v1/realtime?intent=transcription`) sends text
  every 0.56 s (lookahead 6) while audio arrives.
- `prompt` is ignored: the model has no text context.
- The end of an utterance is transcribed as the reference's offline
  transcription ends it: the last chunk holds only the frames that exist,
  so the last sentence keeps its punctuation.

## Known limits

- Gufo starts live transcription only when a session's language is unset
  or English; other languages are transcribed after the commit instead.
  Without a language the model detects it, live as well.
- The lookahead of live sessions is fixed: Gufo passes no options to speech
  recognition providers.
- Words have no timestamps.
- Encoder passes (up to 18 s of audio for uploads, 4.5 s for live sessions
  and streamed uploads) take turns on the GPU in arrival order.
- Tested on Strix Halo (`gfx1151`) only.

## Build with Gufo

The extension needs extension API 0.3 (Gufo 0.34.0 or a later 0.3 release);
`flake.nix` pins the revision it was built and tested with.

```sh
git clone https://github.com/gufo-org/gufo
cd gufo
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/nemotron_asr
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

With Nix, `nix build` in this repository builds the pinned Gufo revision with
this extension.

## Tests

```sh
cmake --preset gpu-test -B build/gpu-test -DGUFO_EXTENSION_DIRS=/path/to/nemotron_asr
cmake --build build/gpu-test --parallel 4 --target nemotron_asr_host_test \
  nemotron_asr_features_test nemotron_asr_tokenizer_test nemotron_asr_model_test
GUFO_NEMOTRON_ASR_MODEL=/models/nemotron-3.5-asr-streaming-0.6b \
  ctest --test-dir build/gpu-test -R '^nemotron_asr\.'
```

The tests compare the features, the encoder, the prediction and joint
networks and the token IDs of 16 offline transcriptions (two clips, two
language settings, four lookaheads) with outputs that
`tools/reference/golden.py` recorded from the transformers implementation
in float32. The float16 matrices change activations by up to about 0.1 %
of their range; every token matches. Tests labeled
`external-model` skip unless `GUFO_NEMOTRON_ASR_MODEL` names the model
directory.

`nemotron_asr_transcribe MODEL AUDIO [--language L] [--lookahead N]
[--repeat N] [--push MS]` is the development driver.

## License

The extension source is MIT licensed ([LICENSE](LICENSE)). Code adapted from
other projects keeps their licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
