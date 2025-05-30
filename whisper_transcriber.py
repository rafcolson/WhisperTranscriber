import warnings, datetime, tempfile, whisper, torch
from datetime import datetime, timedelta
from pathlib import Path
from pydub import AudioSegment

def log(*args, level="INFO", **kwargs):
    prefix = {
        "INFO": "ℹ️ ",
        "WARNING": "⚠️ ",
        "ERROR": "❌ "
    }.get(level.upper(), "🔹")
    print("[{}] {}".format(datetime.now().strftime("%H:%M:%S"), prefix), *args, **kwargs)

# Explicitly set path to ffmpeg (adjust if necessary)
AudioSegment.converter = "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe"

# Suppress FutureWarnings from torch.load()
warnings.filterwarnings("ignore", category=FutureWarning)

# Detect CUDA availability
if torch.cuda.is_available():
    log("GPU available: Whisper will use CUDA", level="INFO")
else:
    log("No GPU detected: Whisper will use CPU (slower)", level="WARNING")

# Configuration
model_name = "large"  # Set this based on the selected model (tiny, base, small, medium, large)
model = whisper.load_model(model_name)

# Directories for audio files and transcriptions
output_dir = Path(__file__).resolve().parent / "transcriptions"
output_dir.mkdir(exist_ok=True)
audio_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
parent_dir = Path(__file__).resolve().parent

# Determine if chunking is needed based on the model (only for medium and large models)
use_chunks = model_name in ["medium", "large"]

# If using chunks (for larger models), process audio in chunks
if use_chunks:
    chunk_duration_min = 1  # duration of each chunk in minutes
    chunk_overlap_sec = 10  # seconds of overlap between chunks
    chunk_duration_sec = chunk_duration_min * 60

    # Process each audio file in the working directory
    for audio_file in parent_dir.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        log("Processing:", audio_file.name)

        # Dict to collect all segments across chunks
        all_segments = {}

        # Create temporary directory for chunks
        with tempfile.TemporaryDirectory() as temp_chunk_dir:
            temp_chunk_path = Path(temp_chunk_dir)

            # Load and normalize audio to mono at 16kHz
            audio = AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(16000)
            chunk_length_ms = chunk_duration_sec * 1000
            overlap_ms = chunk_overlap_sec * 1000

            total_length = len(audio)
            start_points = list(range(0, total_length, chunk_length_ms))  # Calculate chunk start points

            # Process each chunk, applying overlap to the start and end times
            for idx, start_point in enumerate(start_points):
                if idx == 0:
                    start_ms = 0
                    end_ms = chunk_length_ms + overlap_ms
                else:
                    start_ms = start_point - overlap_ms
                    end_ms = start_point + chunk_length_ms + overlap_ms

                chunk = audio[start_ms:end_ms]  # Extract the chunk

                # Generate filename for each chunk
                chunk_audio_name = "{}_part{:02d}.wav".format(audio_file.stem, idx + 1)
                chunk_text_name = "{}_part{:02d}.txt".format(audio_file.stem, idx + 1)
                chunk_path = temp_chunk_path / chunk_audio_name
                chunk.export(chunk_path, format="wav")

                log("  Transcribing -> {}".format(chunk_text_name))
                try:
                    result = model.transcribe(str(chunk_path))
                except Exception as e:
                    log("Error while processing", chunk_audio_name, ":", e, level="ERROR")
                    continue

                # Save the transcription for each chunk
                transcript_file = output_dir / chunk_text_name
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for segment in result["segments"]:
                        # Adjust start time by adding the chunk's starting point (start_ms)
                        start_sec = int(segment["start"]) + start_ms // 1000
                        text = segment["text"].strip()

                        # Write timestamp and text for each segment
                        timestamp = str(timedelta(seconds=start_sec))
                        f.write("[{}] {}\n".format(timestamp, text))

                        # Collect the segments across chunks for final compilation
                        if idx not in all_segments:
                            all_segments[idx] = []
                        all_segments[idx].append((start_sec, text))

        # Compile all the transcriptions from different chunks, without duplicates
        unique_entries = set()
        audio_file_name = audio_file.stem + ".txt"
        compiled_output = output_dir / audio_file_name
        with open(compiled_output, "w", encoding="utf-8") as f:
            for idx, segments in all_segments.items():
                for sec, text in sorted(segments):
                    key = (sec, text)
                    if key in unique_entries:
                        continue
                    if sec < idx * chunk_duration_sec:
                        continue
                    if sec >= (idx + 1) * chunk_duration_sec:
                        continue
                    unique_entries.add(key)
                    timestamp = str(timedelta(seconds=sec))
                    f.write("[{}] {}\n".format(timestamp, text))

        log("  Compiling ->", audio_file_name, level="INFO")

# If not using chunks (for smaller models), transcribe the entire audio file in one go
else:
    for audio_file in parent_dir.iterdir():
        if audio_file.suffix.lower() not in audio_extensions:
            continue

        log("Processing:", audio_file.name)

        # Load and normalize audio to mono at 16kHz
        audio = AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(16000)

        # Transcribe the entire audio file (no chunking)
        try:
            result = model.transcribe(str(audio_file))
        except Exception as e:
            log("Error while processing", audio_file.name, ":", e, level="ERROR")
            continue

        # Save the raw transcription for the entire audio file
        text_file_name = audio_file.stem + ".txt"
        transcript_file = output_dir / text_file_name
        with open(transcript_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                # Write timestamp and text for each segment
                start_sec = int(segment["start"])
                text = segment["text"].strip()
                timestamp = str(timedelta(seconds=start_sec))
                f.write("[{}] {}\n".format(timestamp, text))

        log("Transcript saved as:", text_file_name, level="INFO")

# Log that all files have been processed
log("All files processed.", level="INFO")

