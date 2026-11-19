"""Freeze time-ordered splits after reviewing the thumbnails; seal the native packet."""
import argparse
from pathlib import Path

from capture_rectify import files, read, save, sha, verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("packet",type=Path)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review",type=Path,help="Reviewed exclusions keyed by frame ID, and rationale")
    mode.add_argument("--full",action="store_true")
    args=p.parse_args()
    root=args.packet
    if (root/"manifest.json").exists():
        raise FileExistsError("Packet is already sealed")
    rows=read(root/"selection.json")
    if args.full:
        protocol=read(Path("configs/capture_full_test_protocol.json"))
        prior=Path(protocol["selection"]["prior_selection"])
        old_split=Path(protocol["split"]["prior_split_manifest"])
        assert sha(prior)==protocol["selection"]["prior_selection_sha256"]
        assert sha(old_split)==protocol["split"]["prior_split_manifest_sha256"]
        ref=read(prior)["rows"]
        for actual,old in zip(rows[:len(ref)],ref,strict=True):
            for key in ("id","actual_rgb_timestamp_ns","rgb_zero_based_index","rgb_payload_sha256","depth_pixels_sha256"):
                assert actual[key]==old[key],(old["id"],key)
            for stream in ("rgb","depth"):
                for key in ("sha256","index_zero_based","device_ns"):
                    assert actual[stream][key]==old[stream][key],(old["id"],stream,key)
        for split,ids in read(old_split)["splits"].items():
            assert ids==[r["id"] for r in rows[:len(ref)] if r["requested_split"]==split]
        review=dict(exclude={},policy="Requested roles frozen before SfM by original candidate index.",
                    s01_identities_verified=len(ref),s01_heldout_verified=48)
    else:
        review=read(args.review)
    eligible=[r for r in rows if not r["drop_reason"]]
    if not set(review["exclude"]) <= {r["id"] for r in eligible}:
        raise ValueError("Unknown excluded ID")
    splits={k:[] for k in ("train","val","test")}
    entries=[]
    for row in sorted(eligible,key=lambda r:int(r["actual_rgb_timestamp_ns"])):
        if row["id"] in review["exclude"]:
            continue
        i=row["candidate_index_zero_based"] if args.full else len(entries)
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
