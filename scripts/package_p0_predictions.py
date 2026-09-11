"""Package completed P0/seed3 predictions, without original images or GT."""

import argparse
import gzip
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "p0-locked-s20260824-stats120"
WORK = Path("F:/YOLO-Master-A2-P1/closure-evaluation") / NAME
OUT = ROOT / "results/closure-evaluation" / NAME


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=NAME, choices=(NAME, "p1-fixed-s20260826", "p1-adaptive-s20260826", "p1-tal-s20260826"))
    name = parser.parse_args().run
    work = WORK.parent / name
    out = OUT.parent / name
    evaluation = json.loads((out / "evaluation-manifest.json").read_text(encoding="utf-8"))
    if evaluation["completed_epochs"] != 120 or evaluation["images"] != 548:
        raise ValueError("Evaluation must be complete")
    files = sorted((work / "det").glob("*.txt"))
    names = {p.name for p in Path("F:/datasets/VisDrone/VisDrone2019-DET-val/annotations").glob("*.txt")}
    if len(files) != 548 or {p.name for p in files} != names:
        raise ValueError("DET filenames do not exactly cover the 548 validation images")
    predictions = work / "val/predictions.json"
    if sha(predictions) != evaluation["predictions_sha256"]:
        raise ValueError("Raw prediction hash differs from the evaluated input")
    zipped = out / "predictions-det-548.zip"
    hashes = {p.name: sha(p) for p in files}
    if not zipped.exists():
        with zipfile.ZipFile(zipped, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in files:
                info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
    with zipfile.ZipFile(zipped) as archive:
        if set(archive.namelist()) != names or archive.testzip() is not None:
            raise ValueError("DET zip membership/CRC verification failed")
        if any(hashlib.sha256(archive.read(name)).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError("DET zip bytes differ from the evaluated input")
    compressed = out / "predictions.json.gz"
    if not compressed.exists():
        with predictions.open("rb") as source, compressed.open("xb") as destination:
            with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as archive:
                shutil.copyfileobj(source, archive)
    with gzip.open(compressed, "rb") as archive:
        if hashlib.file_digest(archive, "sha256").hexdigest() != evaluation["predictions_sha256"]:
            raise ValueError("Prediction gzip round-trip verification failed")
    manifest = {
        "run": name, "scope": "Predictions only; original images and GT are not included",
        "images": 548, "det_files_sha256": hashes,
        "det_zip_sha256": sha(zipped), "det_zip_bytes": zipped.stat().st_size,
        "predictions_gzip_sha256": sha(compressed), "predictions_gzip_bytes": compressed.stat().st_size,
        "predictions_uncompressed_sha256": evaluation["predictions_sha256"],
        "runtime_parity": "original MATLAB execution still pending",
    }
    (out / "prediction-archive-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "det_files_sha256"}))


if __name__ == "__main__":
    main()
