"""Step 1: the partitions of the two datasets (five for carclaims, ten for ULB).

    python make_partitions.py      ->  results/partitions/<dataset>_p<k>.npz and <dataset>_manifest.yaml

Each partition assigns 55%, 10%, 15% and 20% of the cases to the training, calibration, validation and
test parts, stratified by class and with the cases that have identical variables kept in the same part
(``fraud.data``). The manifests record the counts, the groups of identical cases and the SHA-256 of
every file; every later step checks them.
"""
from fraud.data import make_partitions

if __name__ == "__main__":
    for d in ("carclaims", "ulb"):
        man = make_partitions(d)
        print(d, man["groups"])
        for p, v in man["partitions"].items():
            print(f"  p{p}", v["rows"], v["frauds"], v["sha256_file"][:12])
