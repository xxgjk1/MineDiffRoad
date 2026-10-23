"""Package source, documentation, paper figures, and result tables."""
import argparse
import hashlib
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--update-local-manifest",action="store_true",help="Refresh generated root SHA256SUMS to match this package")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    output=Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    allowed_dirs={"minediffroad","configs","docs","examples","tests","third_party","scripts"}
    allowed_root={"README.md","README.zh-CN.md","LICENSE",".gitignore","pyproject.toml","requirements.txt",
                  "requirements-cuda.txt","requirements-tested.txt"}
    paths=[]
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel=path.relative_to(root)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if rel.parts[:2] == ("docs", "legacy") and path.suffix != ".md":
            continue
        is_source=rel.parts[0] in allowed_dirs and path.suffix in {".py",".md",".txt",".yaml",".json",".jsonl"}
        is_paper_asset=rel.parts[0] == "docs" and path.suffix in {".png",".csv"}
        if (len(rel.parts)==1 and rel.name in allowed_root) or is_source or is_paper_asset:
            paths.append(path)
    paths.sort(key=lambda p:p.relative_to(root).as_posix())
    hashes="\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}" for p in paths)+"\n"
    if args.update_local_manifest:
        (root/"SHA256SUMS").write_text(hashes,encoding="utf-8")
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in paths:
            info=zipfile.ZipInfo(f"{root.name}/{path.relative_to(root).as_posix()}",date_time=(2026,10,19,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,path.read_bytes())
        archive.writestr(f"{root.name}/SHA256SUMS",hashes)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    print(f"{output}\nSource files: {len(paths)}\nBytes: {output.stat().st_size}\nSHA-256: {hashlib.sha256(output.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
