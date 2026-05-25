import argparse
import json
from pathlib import Path

import librosa
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_jsonl", default="train_raw.jsonl")
    parser.add_argument("--sample_rate", type=int, default=24000)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    audio_paths = sorted({row["audio"] for row in rows} | {row["ref_audio"] for row in rows})

    for audio_path in audio_paths:
        src_path = Path(audio_path)
        audio, _ = librosa.load(src_path, sr=args.sample_rate, mono=True)
        sf.write(output_dir / src_path.name, audio, args.sample_rate, subtype="PCM_16")

    output_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["audio"] = str(output_dir / Path(row["audio"]).name)
        new_row["ref_audio"] = str(output_dir / Path(row["ref_audio"]).name)
        output_rows.append(new_row)

    output_jsonl = output_dir / args.output_jsonl
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"converted_wavs={len(audio_paths)}")
    print(f"output_jsonl={output_jsonl}")


if __name__ == "__main__":
    main()
