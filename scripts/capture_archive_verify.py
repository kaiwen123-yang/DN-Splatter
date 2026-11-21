"""Read-only offline verification. Run the copy next to archive_complete.json with Python 3.9+."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
    return h.hexdigest()


def verify(root,metadata_only=False):
    root=root.resolve()
    ctrl=root/"verification"
    combined=load(root/"archive_complete.json")
    if combined["status"]!="complete_verified":
        raise ValueError("Missing completed archive receipt")
    def checked(path,ref):
        if path.stat().st_size!=ref["bytes"] or sha(path)!=ref["sha256"]:
            raise ValueError("Checksum mismatch: "+str(path))
    for p,k in (("private_complete.json","private_receipt_sha256"),("public_complete.json","public_receipt_sha256")):
        if sha(ctrl/p)!=combined[k]:raise ValueError(p)
    private=load(ctrl/"private_complete.json")
    public=load(ctrl/"public_complete.json")
    catalog=load(ctrl/"catalog.json")
    for key in ("inventory","plan"):
        checked(ctrl/("inventory.jsonl.gz" if key=="inventory" else "plan.json.gz"),catalog[key])
    sealed=load(ctrl/"server_sealed.json")
    if not (private["result_commit"]==public["result_commit"]==combined["result_commit"]==catalog["result_commit"] and
            sealed["inventory_sha256"]==catalog["inventory"]["sha256"] and private["volumes"]==sealed["volumes"]):
        raise ValueError("Inconsistent archive identities")
    def path(base,name):
        p=PurePosixPath(name.replace("\\","/"))
        if p.is_absolute() or any(q in (".","..") or ":" in q for q in p.parts):raise ValueError("Invalid relative path")
        resolved=base.joinpath(*p.parts).resolve()
        resolved.relative_to(root)
        return resolved
    cache={};size=0;logical=0;unique=set()
    def member(base,r):
        f=path(base,r["path"])
        st=f.stat()
        if st.st_size!=r["bytes"]:raise ValueError("Size mismatch: "+str(f))
        if not metadata_only:
            inode=(st.st_dev,st.st_ino) if st.st_ino else str(f)
            if inode not in cache:cache[inode]=sha(f)
            if cache[inode]!=r["sha256"]:raise ValueError("Member SHA256 mismatch: "+str(f))
    with gzip.open(ctrl/"inventory.jsonl.gz","rt",encoding="utf-8") as f:
        for line in f:
            r=json.loads(line);member(root/"project",r);size+=r["bytes"];logical+=1;unique.add(r["sha256"])
    if (logical,size,len(unique))!=(catalog["logical_files"],catalog["logical_bytes"],catalog["unique_files"]):
        raise ValueError("Inventory cardinality")
    tree=load(ctrl/"git_tree.json")
    for r in tree["files"]:member(root/"project",r)
    original=load(ctrl/"original.json")
    for base,r in ((root,dict(original,path=original["member"])),(root,dict(combined["source_bundle"],path="DN-Splatter.bundle"))):
        member(base,r)
    for r in public["assets"]:member(root/"public-release",dict(r,path=r["name"]))
    return dict(status="metadata_verified" if metadata_only else "all_bytes_verified",result_commit=combined["result_commit"],
                logical_files=logical,unique_contents=len(unique),logical_bytes=size,
                frozen_repository_files=len(tree["files"]),release_assets=len(public["assets"]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",nargs="?",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--metadata-only",action="store_true",help="Check existence, sizes and manifest bindings without reading every payload")
    args=p.parse_args()
    print(json.dumps(verify(args.archive,args.metadata_only),ensure_ascii=False,indent=2))
