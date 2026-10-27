"""교수님 원본 코드·데이터의 경로와, 원본 파일을 고치지 않고 불러오는 도우미.

- 원본(초기 커밋 7d4e4e0 유래)은 읽기만 한다. 모듈은 파일 경로로 불러와 원래 이름
  (myUNet_DF, mySSIM, myUtils, myDataloader_fastmri_brain_230425)으로 sys.modules 에 등록한다.
  교수님 pickle 체크포인트(torch.save(model))를 나중에 열 때도 같은 이름이 필요하다.
- 교수님 로더(230425)는 옛 기계 경로 '/mnt/sdb/...' 와 상대 경로 목록 파일을 쓴다(230425:489,496).
  파일은 고치지 않고, 불러온 모듈 이름공간의 `sio` 만 경로를 '/mnt/sda/...' 로 바꿔 주는 작은 객체로 바꾼다
  (scipy 전역·디스크 파일은 그대로). 목록 파일은 dataloaders/ 로 잠시 chdir 해서 연다.
- 교수님 학습 스크립트(231030·231022)는 없는 config 를 import 하므로 실행할 수 없다. 모델 클래스 원문을
  AST 로 잘라 내어(파일 무수정) 동등성 시험에만 쓴다(exec_prof_class).
"""
import ast
import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROF_DATALOADER_DIR = REPO / 'dataloaders'
PROF_MODEL_DIR = REPO / 'models' / 'hybrid_eternet'
PROF_SCRIPTS_DIR = REPO / 'scripts_legacy'
PROF_LIST_FILE = PROF_DATALOADER_DIR / 'list_brain_train_320.txt'
PROF_LOADER_FILE = PROF_DATALOADER_DIR / 'myDataloader_fastmri_brain_230425.py'
PROF_UNET_FILE = PROF_MODEL_DIR / 'myUNet_DF.py'
PROF_SSIM_FILE = PROF_MODEL_DIR / 'mySSIM.py'
PROF_UTILS_FILE = PROF_SCRIPTS_DIR / 'myUtils.py'
PROF_DETER_SCRIPT = PROF_SCRIPTS_DIR / 'choh_train_ETER_hybrid_fastmri_brain_acs32R4_231030.py'
PROF_OETER_SCRIPT = PROF_SCRIPTS_DIR / 'choh_train_oETER_fastmri_brain_fixed_acs32R4_231022.py'

# 230425:489 의 옛 경로 접두사 → 이 서버의 같은 폴더(목록 379볼륨 전부 있음, 218.45 GB)
MAT_PREFIX_ORIG = '/mnt/sdb/'
MAT_PREFIX_HERE = '/mnt/sda/'
PROF_MAT_DIR = '/mnt/sda/choh/shared/data/FastMRI_brain/brain_multicoil_train/mat_brain_multi_train_16ch_396ky_ksp/'


def _load_module(name, path):
    """파일 경로로 모듈을 불러와 `name` 으로 등록한다. 이미 같은 파일로 등록돼 있으면 그대로 돌려준다."""
    mod = sys.modules.get(name)
    if mod is not None:
        if os.path.abspath(getattr(mod, '__file__', '') or '') != os.path.abspath(str(path)):
            raise ImportError(f'{name} 이 다른 파일({getattr(mod, "__file__", None)})로 이미 등록돼 있다')
        return mod
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def prof_unet_module():
    """models/hybrid_eternet/myUNet_DF.py (UNet_choh_skip, UNet). 무수정."""
    return _load_module('myUNet_DF', PROF_UNET_FILE)


def prof_ssim_module():
    """models/hybrid_eternet/mySSIM.py. import 하면 torch.backends.cudnn.deterministic = True 가 켜진다(mySSIM.py:6)."""
    return _load_module('mySSIM', PROF_SSIM_FILE)


def prof_utils_module():
    """scripts_legacy/myUtils.py (인자 2개 myOptimizer, myCriterion). 무수정."""
    return _load_module('myUtils', PROF_UTILS_FILE)


def prof_loader_module():
    """dataloaders/myDataloader_fastmri_brain_230425.py 를 불러오고, 그 모듈의 `sio` 만 경로 변환 객체로 바꾼다."""
    os.environ.setdefault('MPLBACKEND', 'Agg')          # 로더가 matplotlib.pyplot 을 import 한다(230425:10)
    mod = _load_module('myDataloader_fastmri_brain_230425', PROF_LOADER_FILE)
    if not getattr(mod, '_repro_sio_remap', False):
        import scipy.io as _sio
        real_loadmat = _sio.loadmat

        def loadmat_remap(path, *args, **kwargs):
            return real_loadmat(str(path).replace(MAT_PREFIX_ORIG, MAT_PREFIX_HERE, 1), *args, **kwargs)

        mod.sio = types.SimpleNamespace(loadmat=loadmat_remap)   # 이 모듈 이름공간에서만 바뀜
        mod._repro_sio_remap = True
    return mod


@contextlib.contextmanager
def in_dir(path):
    cwd = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(cwd)


PROF_DATASET_CLASSES = {
    # 231030:208 학습 묶음, 231030:179 학습 중 검증(목록 300번), 230425:270 시험 59볼륨
    'test_v2': 'choh_fastmri_brain_hybrid_ifft_acs32R4_test_v2',
    'val': 'choh_fastmri_brain_hybrid_ifft_acs32R4_val',
    'test59': 'choh_fastmri_brain_hybrid_ifft_acs32R4_test59',
    'train_v2': 'choh_fastmri_brain_hybrid_ifft_acs32R4_train_v2',
}


def prof_dataset(kind, **kwargs):
    """교수님 로더 클래스를 그대로 만든다(경로만 메모리에서 바꿈). kwargs 는 원 클래스 인자(idx_start, idx_end 등)."""
    mod = prof_loader_module()
    cls = getattr(mod, PROF_DATASET_CLASSES[kind])
    with in_dir(PROF_DATALOADER_DIR):          # open('list_brain_train_320.txt') 가 상대 경로(230425:496)
        return cls(**kwargs)


def read_list():
    with open(PROF_LIST_FILE, 'r') as f:
        return f.readlines()


def mat_path(idx, lines=None):
    """목록 idx 번 줄 → .mat 경로. 교수님 코드와 같은 치환(230425:507 `filename.replace('.npy\\n', '_ksp.mat')`).

    마지막 줄에 줄바꿈이 없으면 원 치환이 실패한다(이 경우 원 코드도 그 볼륨을 열지 못한다). 여기서는 같은 결과를 내도록
    원 치환만 쓰고, 실패하면 예외를 낸다.
    """
    lines = lines if lines is not None else read_list()
    name = lines[idx]
    out = name.replace('.npy\n', '_ksp.mat')
    if out == name:
        raise ValueError(f'목록 {idx}번 줄 {name!r}: 원 코드의 치환(.npy\\n → _ksp.mat)이 적용되지 않는다')
    return PROF_MAT_DIR + out


def extract_class_source(script_path, class_name):
    """교수님 스크립트에서 클래스 정의 원문만 잘라 낸다(파일 무수정, 스크립트의 모듈 수준 코드는 실행하지 않음)."""
    src = Path(script_path).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return ast.get_source_segment(src, node)
    raise KeyError(f'{class_name} not found in {script_path}')


def exec_prof_class(script_path, class_name, consts):
    """원문 클래스를 consts(N_INPUT_VERTICAL 등 config 상수)와 함께 실행해 클래스 객체를 돌려준다(동등성 시험용).

    원문 forward 는 h0 에 .cuda() 를 쓰므로(231030:123-124, 231022:128-129) GPU 에서만 돈다.
    """
    import torch
    import torch.nn as nn
    unet_mod = prof_unet_module()
    g = {'torch': torch, 'nn': nn, 'UNet_choh_skip': unet_mod.UNet_choh_skip, '__name__': 'prof_' + class_name}
    g.update(consts)
    exec(compile(extract_class_source(script_path, class_name), f'<{Path(script_path).name}:{class_name}>', 'exec'), g)
    return g[class_name]
