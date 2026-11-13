# Nemotron 3.5 ASR quality

This page records numerical checks against the reference implementation,
word error rates on the benchmark recordings and Gufo's functional ASR
suite. Measured November 13, 2026, with extension 0.1.1 on Gufo 0.34.0
(`2845939b`), on NVIDIA's Hugging Face files at `ea30d66d`.

## Against the reference implementation

The reference is Hugging Face transformers 5.22.2 (`Nemotron3_5AsrForRNNT`)
in float32 on the CPU. [`tools/reference/golden.py`](../../../tools/reference/golden.py)
writes the fixtures in `tests/fixtures/`; the tests compare against them.
The clips are two sentences each in English (10.08 s) and German (6.08 s),
spoken by VoxCPM2 voice design.

| Stage | Check | Result |
| --- | --- | --- |
| Tokenizer | Token IDs of every reference run to raw and clean transcripts | identical |
| Log-mel features | Every frame of both clips, pushed in pieces of 1 to 997 samples | within 6e-4 (of 16.6) |
| Subsampling | First three chunks of both clips at lookahead 13 | within 1.3 (of 1,160) |
| Encoder output | Same chunks | within 6.3e-4 (of 0.90) |
| Encoder projection | Same chunks | within 6.2e-3 (of 1.65) |
| Prediction network | Start state | within 5.5e-4 (of 4.4) |
| Joint | Logits of the first frame | within 0.019 (of 104) |
| Transcription | Token IDs of 16 offline transcriptions: both clips, the clip's language and automatic detection, lookaheads 0, 3, 6 and 13; audio pushed in random pieces | 16/16 identical, same chunk and frame counts |

The reference's offline transcription (the whole clip at once) is what the
extension reproduces, for uploads and for the end of live audio: the
features are those of a centered pass over the clip (samples / 160 frames,
zero padding after the pre-emphasis), and the last chunk holds only the
frames that exist. Ten of the 16 transcripts end with a period, among them
both clips at lookahead 13.

Every matrix product takes float16 operands with float32 sums: the
subsampling's, the encoder layers', the attention's scores and values, the
projections, the prediction network and the joint. The subsampling also
keeps its activations and cached rows in float16 (saturated at 65504);
norms, the softmax, the encoder's residual stream, the attention and
convolution caches and the LSTM state are float32. That explains
the deviations: about 1e-3 of the range after the subsampling, 7e-4 at the
encoder's output, up to 4.5e-3 for single projected values (about 1e-4 on
average) and 2e-4 for the joint's logits; in float32 they were below 5e-6.
On a 320.8 s recording of all three clips, float16 matrices in the encoder
layers and the transducer returned the same tokens as float32 ones, and
moving the subsampling, projections and attention to float16 (0.1.1) left
the text unchanged: 0.1.0 and 0.1.1 both return it in 1,744 tokens.

## Word error rates

Against the texts the clips were synthesized from and the reference
recording's transcript; words lowercased, punctuation except apostrophes
removed. The long recording is the two fixture clips with 1 s pauses,
repeated to 308.7 s.

| Audio | Gufo | audio.cpp Q8_0 | audio.cpp float32 |
| --- | ---: | ---: | ---: |
| English, 10.08 s | 0.0 | 0.0 | 0.0 |
| German, 6.08 s | 0.0 | 0.0 | 0.0 |
| Reference, 10.0 s | 0.045 | 0.045 | 0.045 |
| Long recording, 308.7 s | 0.119 | 0.085 | 0.119 |

All three miss the reference recording's last word ("Extract."). On the long
recording, which the model hears as one utterance, Gufo's transcript has the
float32 reference's error rate; the Q8_0 model happens to make fewer errors
there. The servers and settings are those of the [benchmarks](BENCHMARKS.md).

## Functional API suite

Gufo's `tests/functional/model_services.py asr` passes all checks with a
release build: readiness, English transcription, formats and repeated
requests, streamed text equal to the buffered text, silence, concurrent
speech and silence, ignored unknown fields, rejections and recovery, stream
cancellation and the server observing it.

## Reproduce

```sh
# Fixtures, from a Python environment with transformers 5.22.2 and torch:
python -I tools/reference/golden.py MODEL_DIR

# In a Gufo checkout built with this extension:
GUFO_NEMOTRON_ASR_MODEL=MODEL_DIR ctest --test-dir build/gpu-test -R '^nemotron_asr\.'
python3 tests/functional/model_services.py asr --binary build/release/gufo \
  --model MODEL_DIR --output report
```
