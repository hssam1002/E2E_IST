"""
여러 progressive_mode (training strategy)를 순차적으로 학습시켜
표의 7가지 방법에 대한 성능을 비교하기 위한 헬퍼 스크립트.

예시:
    python compare_progressive_modes.py \\
        --modes full,full_m,full_dual,full_part,full_part_m,mask_only,hybrid_all \\
        --common_args "--trainset DIV2K --testset Kodak --packet_size 16 --mask_prob 0.1"
"""

import argparse
import subprocess
import shlex


def parse_args():
    parser = argparse.ArgumentParser(description="Run training for multiple progressive modes.")
    parser.add_argument(
        "--modes",
        type=str,
        default="full,full_part,full_part_m",
        help="Comma-separated progressive modes to run.",
    )
    parser.add_argument(
        "--common_args",
        type=str,
        default="",
        help="Extra arguments passed to main.py (e.g. \"--trainset DIV2K --testset Kodak\").",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        cmd = (
            f"python main.py --training "
            f"--progressive_mode {mode} "
            f"{args.common_args}"
        )
        print(f"\n[RUN] {cmd}")
        subprocess.run(shlex.split(cmd), check=True)


if __name__ == "__main__":
    main()

