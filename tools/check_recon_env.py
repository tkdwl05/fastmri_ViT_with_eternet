"""환경 점검 유틸리티: SS2D/ETER 실험 실행 전 기본 의존성과 데이터 상태를 점검한다."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Optional


def _status(flag: bool) -> str:
    return "OK" if flag else "MISSING"


def check_python() -> None:
    print(f"Python: {sys.version.split()[0]}")


def check_packages() -> dict:
    required = ("torch", "h5py", "numpy", "einops")
    statuses = {}
    for pkg in required:
        try:
            module = importlib.import_module(pkg)
            version = getattr(module, "__version__", "ok")
            statuses[pkg] = (True, version)
        except Exception as ex:
            statuses[pkg] = (False, str(ex))
    return statuses


def _format_package_report(name: str, ok: bool, detail: str) -> str:
    if ok:
        return f"  - {name}: {_status(True):<7} {detail}"
    return f"  - {name}: {_status(False):<7} {detail}"


def check_torch_and_device(require_cuda: bool = True) -> tuple[bool, Optional["object"]]:
    try:
        torch = importlib.import_module("torch")
    except Exception as ex:
        print(f"[Torch] import 실패: {ex}")
        return False, None

    print(f"Torch: {_status(True):<7} v{torch.__version__}")
    available = torch.cuda.is_available()
    print(f"CUDA 사용 가능: {available}")
    if require_cuda and not available:
        print("  - CUDA: 필수. torch.cuda.is_available()가 False입니다.")
        return False, torch

    if available:
        num = torch.cuda.device_count()
        print(f"CUDA 장치 수: {num}")
        for i in range(num):
            prop = torch.cuda.get_device_properties(i)
            free_mem = torch.cuda.mem_get_info(i)[0] if hasattr(torch.cuda, "mem_get_info") else -1
            print(f"  - [{i}] {torch.cuda.get_device_name(i)}")
            print(f"      총 메모리: {prop.total_memory / 1024 ** 3:.2f} GB")
            if free_mem >= 0:
                print(f"      여유 메모리(추정): {free_mem / 1024 ** 3:.2f} GB")
    return True, torch


def check_ss2d_dependency(require: bool = False) -> bool:
    try:
        mamba = importlib.import_module("mamba_ssm")
        _ = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        print(f"mamba-ssm: {_status(True):<7} v{getattr(mamba, '__version__', 'ok')}")
        return True
    except Exception as ex:
        print(f"mamba-ssm: {_status(False):<7} {ex}")
        if require:
            return False
        return False


def check_dataset() -> bool:
    train = os.path.exists("./fastMRI_data/multicoil_train")
    val = os.path.exists("./fastMRI_data/multicoil_val")
    train_n = len([f for f in os.listdir("./fastMRI_data/multicoil_train")] ) if train else 0
    val_n = len([f for f in os.listdir("./fastMRI_data/multicoil_val")] ) if val else 0
    print(f"fastMRI 데이터셋: train={train}, val={val}")
    print(f"  - train 파일 수: {train_n}")
    print(f"  - val 파일 수: {val_n}")
    return train and val and train_n > 0 and val_n > 0


def check_config_import(config_module: str) -> dict | None:
    try:
        return vars(importlib.import_module(config_module))
    except Exception as ex:
        print(f"Config import 실패({config_module}): {ex}")
        return None


def check_env_for_model(model_type: str, config_module: str, strict: bool = True) -> bool:
    print("=== 실행 환경 점검 ===")
    check_python()
    pkg = check_packages()
    for name, (ok, detail) in pkg.items():
        print(_format_package_report(name, ok, detail))
    missing_pkg = [name for name, (ok, _) in pkg.items() if not ok]
    if missing_pkg:
        print(f"필수 패키지 누락: {', '.join(missing_pkg)}")
        if strict:
            print("필수 패키지 미설치로 학습을 진행할 수 없습니다.")
            return False

    torch_ok, _torch = check_torch_and_device(require_cuda=True)
    if not torch_ok:
        print("필수 조건 미달: torch/CUDA 환경을 다시 확인하세요. 학습을 진행할 수 없습니다.")
        return False

    ss2d_ok = True
    if model_type == "ss2d":
        ss2d_ok = check_ss2d_dependency(require=strict)
        if not ss2d_ok and strict:
            print("SS2D는 mamba-ssm이 필수입니다.")
            return False

    data_ok = check_dataset()
    if not data_ok:
        print("FastMRI 경로/파일이 없어서 학습/평가를 진행할 수 없습니다.")
        return False

    conf = check_config_import(config_module)
    if conf is None:
        if strict:
            return False
    else:
        image_size = conf.get("IMAGE_SIZE", ("?", "?"))
        patch_size = conf.get("PATCH_SIZE", ("?", "?"))
        batch = conf.get("BATCH_SIZE", "?")
        epochs = conf.get("NUM_EPOCHS", "?")
        print(f"Config({config_module})")
        print(f"  - image size: {image_size}, patch size: {patch_size}")
        print(f"  - batch size: {batch}, epochs: {epochs}")
        if model_type == "eter":
            print(f"  - ETER hidden: horiz={conf.get('NUM_ETER_HORI_HIDDEN', '?')}, vert={conf.get('NUM_ETER_VERT_HIDDEN', '?')}")
        if model_type == "ss2d":
            print(f"  - SS2D params: d_inner={conf.get('NUM_SS2D_D_INNER', '?')}, d_state={conf.get('NUM_SS2D_D_STATE', '?')}, out_ch={conf.get('NUM_SS2D_OUT_CH', '?')}")

    print("점검 완료")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MR recon 환경 점검")
    parser.add_argument("--model", choices=["eter", "ss2d"], required=True)
    parser.add_argument("--config", required=True, help="실행할 모델 config 모듈명 예: myConfig_choh_ETER_model")
    parser.add_argument("--strict", action="store_true", help="필수 의존성 누락 시 종료")
    args = parser.parse_args()

    ok = check_env_for_model(args.model, args.config, strict=args.strict)
    raise SystemExit(0 if ok else 1)
