# Nemotron 3.5 ASR benchmarks

Measured November 13, 2026, on Strix Halo (`gfx1151`, 128 GB) with
extension 0.1.0 in a release build of Gufo 0.34.0
(`2845939b`, ROCm 10.1). The reference is the audio.cpp server v0.9.4
(`f7ac446f`) on Vulkan, once with its Q8_0 GGUF and once with the official
float32 Hugging Face directory. Each server ran alone under a machine
reservation, device power profile balanced (85 W sustained limit); package
power peaked at 92 W for both.

Gufo loads the float32 checkpoint and computes with float16 matrices and
float32 sums (see [quality](QUALITY.md)); audio.cpp's Q8_0 GGUF quantizes
the linear layers to 8 bits. Both transcribe uploads at lookahead 13
(1.12 s), audio.cpp in its offline mode.

All times are client wall times of whole HTTP requests: one warmup, then the
median of three. The clips are the test fixtures `english.wav` (10.08 s) and
`german.wav` (6.08 s) and Gufo's functional-test recording
`clear-english-voice.wav` (10.0 s, "reference"); the long recording is the
two fixture clips with 1 s pauses, repeated to 308.7 s.

## Loading

One cold-file-cache launch per server until its readiness route answers
(`/ready`, audio.cpp `/health`), then the first request (English clip), and
memory after the buffered requests:

| Server | Ready | First request | Resident memory | GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Gufo | 1.06 s | 0.039 s | 240 MiB | 1,490 MiB |
| audio.cpp, Q8_0 | 2.38 s | 0.161 s | 128 MiB | 1,958 MiB |
| audio.cpp, float32 | 2.68 s | 0.182 s | 133 MiB | 5,068 MiB |

audio.cpp loads the model twice, as an offline and a streaming model; its
GPU memory includes both.

## Uploads

| Audio | Gufo | audio.cpp Q8_0 | audio.cpp float32 | Gain over Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| English, 10.08 s | 0.027 s | 0.113 s | 0.156 s | 4.2× |
| German, 6.08 s | 0.022 s | 0.083 s | 0.125 s | 3.8× |
| Reference, 10.0 s | 0.025 s | 0.100 s | 0.139 s | 4.1× |
| Long recording, 308.7 s | 0.640 s | 4.018 s | 4.512 s | 6.3× |

Gufo transcribes the long recording at 482 times real time. Its server
reports where a request's time goes: for the English clip 3 ms of features
(CPU), 16 ms of encoder and 7 ms of greedy decoding; for the long recording
70 ms, 383 ms and 170 ms. Word error rates are on the [quality](QUALITY.md)
page.

## Concurrent uploads

The English clip, C requests at a time, three rounds:

| C | Gufo median | Gufo p95 | Gufo throughput | audio.cpp Q8_0 median | Q8_0 throughput | audio.cpp float32 median | float32 throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.027 s | 0.027 s | 371× | 0.113 s | 87× | 0.170 s | 59× |
| 2 | 0.041 s | 0.056 s | 394× | 0.168 s | 90× | 0.226 s | 66× |
| 4 | 0.064 s | 0.101 s | 422× | 0.273 s | 91× | 0.388 s | 65× |

Throughput is audio seconds transcribed per second. Gufo's encoder passes
take turns on the GPU in arrival order, while other requests compute their
features. Every response repeated the single request's text.

## Streamed text

One `stream=true` request per clip; first text and complete transcript:

| Audio | Gufo first | Gufo complete | audio.cpp Q8_0 first | Q8_0 complete |
| --- | ---: | ---: | ---: | ---: |
| English | 0.020 s (4 events) | 0.058 s | 0.031 s (9 events) | 0.406 s |
| German | 0.020 s (3 events) | 0.042 s | 0.032 s (6 events) | 0.287 s |
| Reference | 0.019 s (3 events) | 0.046 s | 0.070 s (8 events) | 0.386 s |

Gufo streams an upload in passes of about 4.5 s of audio and sends the text
after each; a buffered upload uses passes of up to 18 s and is faster.
audio.cpp streams with its streaming model. In both, the streamed events add
up to the buffered transcript.

## Reproduce

Run from a Gufo checkout with a release build that includes this extension:

```sh
NEMOTRON_ASR_MODEL=/models/nemotron-3.5-asr-streaming-0.6b GUFO=build/release/gufo \
REFERENCE=tests/functional/fixtures/audio/clear-english-voice.wav \
  python3 /path/to/nemotron_asr/tools/bench/asr_bench.py gufo gufo.json
AUDIOCPP_SERVER=/path/to/audiocpp_server \
AUDIOCPP_MODEL=/models/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf \
REFERENCE=tests/functional/fixtures/audio/clear-english-voice.wav \
  python3 /path/to/nemotron_asr/tools/bench/asr_bench.py audiocpp audiocpp.json
```

For audio.cpp on the float32 directory, set `AUDIOCPP_MODEL` to a copy of
the Hugging Face directory without its GGUF file and `AUDIOCPP_MODEL_SPEC`
to audio.cpp's `model_specs/nemotron_asr.json`. The harness records package
power, temperatures and the GPU clock with every run. The results above are
in [artifacts](artifacts/).
