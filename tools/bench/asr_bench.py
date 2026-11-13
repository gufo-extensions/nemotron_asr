#!/usr/bin/env python3
"""Nemotron 3.5 ASR over HTTP: Gufo `serve asr` vs the audio.cpp server.

Usage: asr_bench.py gufo|audiocpp OUTPUT_JSON

Environment:
  NEMOTRON_ASR_MODEL  the Hugging Face directory (what Gufo loads)
  GUFO                the gufo binary (a release build)
  AUDIOCPP_SERVER     the audiocpp_server binary
  AUDIOCPP_MODEL      what audio.cpp loads: the same directory (default) or
                      its GGUF
  AUDIOCPP_MODEL_SPEC audio.cpp's model_specs/nemotron_asr.json, which it
                      needs for a directory (its GGUFs embed the spec)
  AUDIOCPP_BACKEND    audio.cpp's backend, vulkan by default
  REFERENCE           an extra English WAV with its transcript beside it
                      (.txt), such as Gufo's functional-test recording
  PORT                18741 by default
  LOOKAHEAD           audio.cpp's lookahead, 13 by default as Gufo's (Gufo
                      transcribes uploads at the checkpoint's widest
                      lookahead, 13, whatever this says)
  PHASES              any of single,stream,concurrent,long (default all)

Launches the server with a cold file cache and times readiness and the first
transcript. Then, per clip, one warmup and three timed buffered requests
(median); one SSE request per clip (first text, complete); concurrent
requests (C = 1, 2 and 4, three rounds) on the first clip; and one request
on a long recording (the two fixture clips with 1 s pauses, repeated to
five minutes; REFERENCE is not part of it).
Records word error rates against the clips' transcripts, the server's memory
after the buffered requests, and package power, temperatures and the GPU
clock during the run. All times are client wall times of whole requests.
"""
import json, os, re, signal, statistics, struct, subprocess, sys, threading, time
import urllib.request, uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "tests", "fixtures")
MODEL_DIR = os.environ.get("NEMOTRON_ASR_MODEL", "")
GUFO = os.environ.get("GUFO", "gufo")
AUDIOCPP_SERVER = os.environ.get("AUDIOCPP_SERVER", "audiocpp_server")
AUDIOCPP_MODEL = os.environ.get("AUDIOCPP_MODEL", MODEL_DIR)
AUDIOCPP_MODEL_SPEC = os.environ.get("AUDIOCPP_MODEL_SPEC", "")
AUDIOCPP_BACKEND = os.environ.get("AUDIOCPP_BACKEND", "vulkan")
REFERENCE = os.environ.get("REFERENCE", "")
PORT = int(os.environ.get("PORT", "18741"))
LOOKAHEAD = int(os.environ.get("LOOKAHEAD", "13"))
PHASES = set(os.environ.get("PHASES", "single,stream,concurrent,long").split(","))
GPU = "/sys/class/drm/card0/device"
LONG_SECONDS = 300.0

# The texts the fixture clips were synthesized from.
CLIPS = [
    ("english", os.path.join(FIXTURES, "english.wav"), "en-US",
     "The quick brown fox jumps over the lazy dog. Speech recognition turns spoken "
     "words into written text one sentence at a time."),
    ("german", os.path.join(FIXTURES, "german.wav"), "de-DE",
     "Heute ist ein schöner Tag in Berlin. Wir prüfen die Spracherkennung mit zwei "
     "kurzen Sätzen auf Deutsch."),
]


def clips():
    result = list(CLIPS)
    if REFERENCE:
        stem = os.path.splitext(REFERENCE)[0]
        result.append(("reference", REFERENCE, "en-US", open(stem + ".txt").read().strip()))
    return result


# --- audio --------------------------------------------------------------------

def read_wav(path):
    """(rate, mono PCM16 bytes) of a 16-bit mono WAV."""
    data = open(path, "rb").read()
    rate = channels = bits = None
    at = 12
    while at + 8 <= len(data):
        chunk, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        if chunk == b"fmt ":
            channels, rate = struct.unpack("<HI", data[at + 10:at + 16])
            bits = struct.unpack("<H", data[at + 22:at + 24])[0]
        elif chunk == b"data":
            if channels != 1 or bits != 16:
                raise ValueError(f"{path}: not 16-bit mono")
            return rate, data[at + 8:at + 8 + size]
        at += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def write_wav(path, rate, pcm):
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
                         rate, rate * 2, 2, 16, b"data", len(pcm))
    with open(path, "wb") as stream:
        stream.write(header + pcm)


def long_recording(path, items):
    """The clips with 1 s pauses, repeated to LONG_SECONDS; returns (seconds, text)."""
    pcm, texts = bytearray(), []
    while len(pcm) < LONG_SECONDS * 16000 * 2:
        for _, wav, _, text in items:
            rate, data = read_wav(wav)
            if rate != 16000:
                raise ValueError(f"{wav}: long-form input needs 16 kHz")
            pcm += data + bytes(2 * 16000)
            texts.append(text)
    write_wav(path, 16000, bytes(pcm))
    return len(pcm) / 32000, " ".join(texts)


def wav_seconds(path):
    rate, pcm = read_wav(path)
    return len(pcm) / 2 / rate


# --- word error rate ---------------------------------------------------------

def words(text):
    return re.sub(r"[^\w\s']", " ", text.lower()).split()


def wer(hypothesis, reference):
    hyp, ref = words(hypothesis), words(reference)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1] / max(1, len(ref))


# --- machine state -----------------------------------------------------------

def read_number(path):
    try:
        with open(path) as stream:
            return float(stream.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def hwmon(name):
    for entry in sorted(os.listdir("/sys/class/hwmon")):
        if read_text(f"/sys/class/hwmon/{entry}/name") == name:
            return f"/sys/class/hwmon/{entry}"
    return None


def read_text(path):
    try:
        with open(path) as stream:
            return stream.read().strip()
    except OSError:
        return None


class Monitor:
    """Package power, GPU edge and CPU Tctl temperatures, GPU clock and load."""

    def __init__(self):
        self.gpu = hwmon("amdgpu")
        self.cpu = hwmon("k10temp")
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        clock = None
        for line in (read_text(f"{GPU}/pp_dpm_sclk") or "").splitlines():
            if line.endswith("*"):
                clock = float(re.findall(r"(\d+)Mhz", line)[0])
        power = read_number(f"{self.gpu}/power1_average") if self.gpu else None
        return {"t": time.time(),
                "watts": power / 1e6 if power is not None else None,
                "gpu_c": (read_number(f"{self.gpu}/temp1_input") or 0) / 1000 if self.gpu else None,
                "cpu_c": (read_number(f"{self.cpu}/temp1_input") or 0) / 1000 if self.cpu else None,
                "sclk_mhz": clock,
                "gpu_busy": read_number(f"{GPU}/gpu_busy_percent")}

    def run(self):
        while not self.stop.is_set():
            self.samples.append(self.sample())
            self.stop.wait(0.5)

    def summary(self, since=None):
        rows = [s for s in self.samples if since is None or s["t"] >= since]
        result = {}
        for key in ("watts", "gpu_c", "cpu_c", "sclk_mhz", "gpu_busy"):
            values = [s[key] for s in rows if s[key] is not None]
            if values:
                result[key] = {"median": statistics.median(values), "max": max(values)}
        return result


def power_profile():
    for command in (["axb-profile", "get"], ["sg", "axb-profile", "-c", "axb-profile get"]):
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout.strip()
    return None


def gpu_memory():
    total = 0
    for name in ("mem_info_vram_used", "mem_info_gtt_used"):
        total += int(read_number(f"{GPU}/{name}") or 0)
    return total


def resident(pid):
    with open(f"/proc/{pid}/status") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def drop_cache(root):
    files = [root] if os.path.isfile(root) else [
        os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs]
    for path in files:
        with open(path, "rb") as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


# --- HTTP --------------------------------------------------------------------

def multipart(fields, wav):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                 f"{value}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f"filename=\"{os.path.basename(wav)}\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += open(wav, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


class Server:
    def __init__(self, kind, scratch, log):
        self.kind, self.scratch, self.log = kind, scratch, log
        self.process = None

    def fields(self, language, stream=False):
        fields = {"language": language, "response_format": "json"}
        if self.kind == "audiocpp":
            fields["model"] = "nemotron-stream" if stream else "nemotron"
        if stream:
            fields["stream"] = "true"
        return fields

    def launch(self):
        if self.kind == "gufo":
            # The default 8 MiB body limit holds 4.4 minutes of 16 kHz WAV.
            command = [GUFO, "serve", "asr", "--model", MODEL_DIR, "--host", "127.0.0.1",
                       "--port", str(PORT), "--max-request-bytes", str(64 << 20)]
            ready = "/ready"
            drop_cache(MODEL_DIR)
        else:
            options = {"lookahead_tokens": str(LOOKAHEAD)}
            config = {"host": "127.0.0.1", "port": PORT, "backend": AUDIOCPP_BACKEND, "device": 0,
                      "threads": 4, "lazy_load": False,
                      "models": [{"id": "nemotron", "family": "nemotron_asr", "path": AUDIOCPP_MODEL,
                                  "task": "asr", "mode": "offline",
                                  "default_request_options": options}]}
            if "stream" in PHASES:
                config["models"].append({"id": "nemotron-stream", "family": "nemotron_asr",
                                         "path": AUDIOCPP_MODEL, "task": "asr", "mode": "streaming",
                                         "default_request_options": options})
            if AUDIOCPP_MODEL_SPEC:
                for model in config["models"]:
                    model["model_spec_override"] = AUDIOCPP_MODEL_SPEC
            path = os.path.join(self.scratch, "audiocpp_server.json")
            json.dump(config, open(path, "w"), indent=1)
            command = [AUDIOCPP_SERVER, "--config", path]
            ready = "/health"
            drop_cache(AUDIOCPP_MODEL)
        self.command = command
        start = time.perf_counter()
        self.process = subprocess.Popen(["choom", "-n", "1000", "--"] + command, stdout=self.log,
                                        stderr=subprocess.STDOUT)
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("server exited")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{ready}", timeout=2) as response:
                    if response.status == 200:
                        return time.perf_counter() - start
            except Exception:
                time.sleep(0.05)

    def stop(self):
        if self.process is None:
            return
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def transcribe(self, wav, language):
        body, content_type = multipart(self.fields(language), wav)
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/audio/transcriptions",
                                         data=body, headers={"Content-Type": content_type})
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=1200) as response:
            data = json.loads(response.read())
        return time.perf_counter() - start, data

    def stream(self, wav, language):
        body, content_type = multipart(self.fields(language, stream=True), wav)
        request = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/audio/transcriptions",
                                         data=body, headers={"Content-Type": content_type})
        start = time.perf_counter()
        first, text, deltas = None, None, []
        with urllib.request.urlopen(request, timeout=1200) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                try:
                    event = json.loads(line[5:])
                except ValueError:
                    continue  # such as "[DONE]"
                if event.get("type", "").endswith(".delta") and event.get("delta"):
                    if first is None:
                        first = time.perf_counter() - start
                    deltas.append(event["delta"])
                elif event.get("type", "").endswith(".done"):
                    text = event.get("text")
        return {"first_text_seconds": first, "complete_seconds": time.perf_counter() - start,
                "deltas": len(deltas), "text": text, "deltas_equal_text": "".join(deltas).strip() == (text or "").strip()}


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    return value


def main():
    kind, output = sys.argv[1], sys.argv[2]
    if kind not in ("gufo", "audiocpp"):
        sys.exit(__doc__)
    scratch = os.path.dirname(os.path.abspath(output))
    items = clips()
    model = MODEL_DIR if kind == "gufo" else AUDIOCPP_MODEL
    # The model's name only: results are published.
    result = {"server": kind, "lookahead": 13 if kind == "gufo" else LOOKAHEAD,
              "power_profile_before": power_profile(),
              "model": os.path.basename(os.path.normpath(model)), "clips": {}}
    monitor = Monitor()
    result["machine_before"] = monitor.sample()
    monitor.thread.start()
    with open(output + ".server.log", "w") as log:
        server = Server(kind, scratch, log)
        gpu_before = gpu_memory()
        try:
            result["ready_seconds"] = server.launch()
            result["command"] = [os.path.basename(c) if os.path.sep in c else c for c in server.command]
            name, wav, language, text = items[0]
            seconds, data = server.transcribe(wav, language)
            result["first_request"] = {"clip": name, "seconds": seconds, "text": data.get("text")}
            print(kind, "ready %.2f s, first request %.3f s" % (result["ready_seconds"], seconds), flush=True)

            if "single" in PHASES:
                for name, wav, language, text in items:
                    server.transcribe(wav, language)  # warmup
                    since = time.time()
                    runs = [server.transcribe(wav, language) for _ in range(3)]
                    times = [t for t, _ in runs]
                    texts = [d.get("text", "") for _, d in runs]
                    duration = wav_seconds(wav)
                    entry = {"audio_seconds": duration, "language": language,
                             "request_seconds": statistics.median(times), "runs": times,
                             "rtf": statistics.median(times) / duration,
                             "audio_seconds_per_second": duration / statistics.median(times),
                             "text": texts[0], "repeatable": all(t == texts[0] for t in texts),
                             "wer": wer(texts[0], text), "machine": monitor.summary(since)}
                    if "timing" in runs[0][1]:
                        entry["server_timing"] = runs[0][1]["timing"]
                    result["clips"][name] = entry
                    print(kind, name, json.dumps(rounded({k: v for k, v in entry.items()
                                                          if k not in ("runs", "machine")})), flush=True)
            result["resident_bytes"] = resident(server.process.pid)
            result["gpu_bytes"] = gpu_memory() - gpu_before
            print(kind, "resident %.0f MiB, GPU %.0f MiB" % (
                result["resident_bytes"] / 2**20, result["gpu_bytes"] / 2**20), flush=True)

            if "long" in PHASES:
                path = os.path.join(scratch, "long.wav")
                duration, text = long_recording(path, CLIPS)
                since = time.time()
                seconds, data = server.transcribe(path, "auto")
                result["long"] = {"audio_seconds": duration, "request_seconds": seconds,
                                  "rtf": seconds / duration, "wer": wer(data.get("text", ""), text),
                                  "words": len(words(text)), "machine": monitor.summary(since)}
                open(output + ".long.txt", "w").write(data.get("text", ""))
                print(kind, "long", json.dumps(rounded({k: v for k, v in result["long"].items()
                                                         if k != "machine"})), flush=True)

            if "concurrent" in PHASES:
                name, wav, language, text = max(items, key=lambda item: wav_seconds(item[1]))
                duration = wav_seconds(wav)
                result["concurrent"] = {"clip": name}
                for users in (1, 2, 4):
                    latencies, rounds, mismatches = [], [], 0
                    with ThreadPoolExecutor(users) as pool:
                        for _ in range(3):
                            start = time.perf_counter()
                            runs = list(pool.map(lambda _: server.transcribe(wav, language), range(users)))
                            rounds.append(time.perf_counter() - start)
                            latencies += [t for t, _ in runs]
                            mismatches += sum(d.get("text") != result["clips"].get(name, {}).get("text", d.get("text"))
                                              for _, d in runs)
                    latencies.sort()
                    entry = {"latency_median": statistics.median(latencies),
                             "latency_p95": latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))],
                             "audio_seconds_per_second": users * duration * len(rounds) / sum(rounds),
                             "text_mismatches": mismatches}
                    result["concurrent"][str(users)] = entry
                    print(kind, f"C={users}", json.dumps(rounded(entry)), flush=True)

            if "stream" in PHASES:
                result["stream"] = {}
                for name, wav, language, text in items:
                    server.stream(wav, language)  # warmup
                    entry = server.stream(wav, language)
                    entry["wer"] = wer(entry["text"] or "", text)
                    result["stream"][name] = entry
                    print(kind, "stream", name, json.dumps(rounded(entry)), flush=True)
        finally:
            server.stop()
            monitor.stop.set()
            monitor.thread.join()
    result["machine"] = monitor.summary()
    result["power_profile_after"] = power_profile()
    json.dump(rounded(result), open(output, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
