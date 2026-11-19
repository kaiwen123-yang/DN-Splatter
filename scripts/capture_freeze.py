"""Freeze time-ordered splits after reviewing the thumbnails; seal the native packet."""
import argparse
from pathlib import Path

from capture_rectify import files, read, save, sha, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet",type=Path)
    p.add_argument("--review",required=True,type=Path,help="Reviewed exclusions keyed by frame ID, and rationale")
    args=p.parse_args()
    root=args.packet
    if (root/"manifest.json").exists():
        raise FileExistsError("Packet is already sealed")
    review=read(args.review)
    rows=read(root/"selection.json")
    eligible=[r for r in rows if not r["drop_reason"]]
    if not set(review["exclude"]) <= {r["id"] for r in eligible}:
        raise ValueError("Unknown excluded ID")
    splits={k:[] for k in ("train","val","test")}
    entries=[]
    for row in sorted(eligible,key=lambda r:int(r["actual_rgb_timestamp_ns"])):
        if row["id"] in review["exclude"]:
            continue
        i=len(entries)
        split="train" if i%10<8 else ("val" if i%10==8 else "test")
        splits[split].append(row["id"])
        entries.append(dict(retained_order_index=i,split=split,id=row["id"],device_ns=row["actual_rgb_timestamp_ns"],
                            rgb=row["rgb"]["file"],depth=row["depth"]["file"],indices=[row[k]["index_zero_based"] for k in ("rgb","depth")],
                            hashes=[row[k]["sha256"] for k in ("rgb","depth")]))
    save(root/"splits.json",dict(selection_sha256=sha(root/"selection.json"),review=review,splits=splits,entries=entries))
    save(root/"manifest.json",dict(run_id=read(root/"identity.json")["run_id"],files=files(root)))
    verify(root)
    print({k:len(v) for k,v in splits.items()},sha(root/"manifest.json"))


if __name__=="__main__":
    main()
