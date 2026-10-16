"""
v8 sanity: "오직 sequence model(±DC)만 다름" + 차원 계약 + 교수님 파일 무수정 검증.

검사:
  1. 4 variant {gru,ss2d}×{noDC,DC} 생성 + forward (BS=1, 384²) → 출력 (1,1,384,384).
  2. 같은 DC 레벨에서 GRU/SS2D 의 U-Net DFU in_channels / n_hidden / depth / wf 동일.
  3. no-DC GRU 가 교수님 ETER_hybrid_GRU_DFU 와 동일 차원(gru_h/gru_v/unet param shape) 인지.
  4. git: 교수님 원본(초기 커밋 7d4e4e0 의 원본 파일)이 7d4e4e0 과 같은지 — git status 가 아니라 git diff 7d4e4e0 기준이라
     커밋된 수정도 잡는다(2026-10-16). 알려진 수정 3 건은 내용 해시로 고정(KNOWN_MODIFIED). models/mamba_eternet/ss2d.py 는 git status.
실행: conda activate mri_env && python v8_eter_pure/sanity_pure_v8.py
"""

import os
import sys
import subprocess

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))

import myConfig_pure_eter_v8 as C
from u_pure_eternet_gru import PureETER_GRU
from u_pure_eternet_ss2d import PureETER_SS2D

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 교수님 원본 = 초기 커밋 7d4e4e0 에서 온 아래 경로의 파일 (CLAUDE.md §작업 규칙, docs/cleanup_log.md 2026-05-20 §6)
ORIG_COMMIT = '7d4e4e0'
ORIG_PREFIXES = ('choh_train_ViT_ETER_R4regular_240916py', 'scripts_legacy/', 'dataloaders/myDataloader_',
                 'configs/myConfig_choh_ViT_', 'configs/myConfig_temp.py', 'configs/myConfig_choh_model3.py',
                 'models/hybrid_eternet/', 'models/mae/', 'models/vit_pytorch/')
# 이미 알려진 수정 — 모두 우리 쪽 작업이고 2026-10-16 사용자 결정으로 유지·기록(docs/cleanup_log.md 2026-10-16).
# 값 = 지금 내용의 git blob 해시 (내용이 더 바뀌면 실패) 또는 'D'(삭제).
KNOWN_MODIFIED = {
    'models/hybrid_eternet/u_choh_SSIM.py': '12354d6883b29d8cae8165031a8ba98cb5eda814',            # afa89a0 (05-22 작업): mask 인자 추가, mask=None = 원본 계산
    'models/hybrid_eternet/u_choh_model_ETER_ViT.py': 'e6391ef2c9f42e1e208cb759dde123900c2d2fc5',  # 0cdc914·e0dc4c9 (04-11·04-15, 규칙 이전)
    'configs/myConfig_choh_model3.py': '0f8748636dc90640b36671180cf455d9fa057858',                 # 0cdc914 (04-11, 규칙 이전)
    'models/vit_pytorch/vit-pytorch-main/tests/.DS_Store': 'D',                                    # 71478ad (09-03) macOS 잔재 삭제
}


def _git(*args):
    return subprocess.run(['git', '-C', _PROJECT_ROOT] + list(args), capture_output=True, text=True, check=True).stdout


def check_professor_originals():
    """교수님 원본이 7d4e4e0 과 같은지 (작업 트리 기준 — 커밋된 수정·커밋 안 된 수정 모두). 알려진 수정은 해시가 같을 때만 통과."""
    orig = [f for f in _git('ls-tree', '-r', '--name-only', ORIG_COMMIT).splitlines() if f.startswith(ORIG_PREFIXES)]
    changed = [ln.split('\t') for ln in _git('diff', '--name-status', ORIG_COMMIT, '--', *orig).splitlines() if ln.strip()]
    bad, known = [], []
    for st, path in ((c[0], c[-1]) for c in changed):
        want = KNOWN_MODIFIED.get(path)
        if want is None:
            bad.append(f'{st} {path}')
        elif want == 'D':
            (known if st == 'D' else bad).append(f'{st} {path}')
        else:
            ok = os.path.exists(os.path.join(_PROJECT_ROOT, path)) and _git('hash-object', path).strip() == want
            (known if ok else bad).append(f'{st} {path}' + ('' if ok else ' (알려진 수정과 내용이 다름)'))
    print(f"    원본 {len(orig)} 파일 vs {ORIG_COMMIT}: 알려진 수정 {len(known)} 건 (기록됨, 유지)")
    for k in known:
        print(f"      - {k}")
    if bad:
        print("    [WARN] 알려지지 않은 원본 변경:\n      " + "\n      ".join(bad))
        raise SystemExit("교수님 원본 파일이 수정됨 — 통제 비교 무효일 수 있음. 확인 요망.")
    return known
H = W = C.IMAGE_SIZE[0]


def build(seq, use_dc):
    if seq == 'gru':
        return PureETER_GRU(
            dim=H, n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    return PureETER_SS2D(
        n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
        unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
        ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
        use_dc=use_dc, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
    )


def unet_in_ch(model):
    # UNet_choh_skip.down_path[0] = UNetConvBlock; .block[0] = first Conv2d
    return model.unet.down_path[0].block[0].in_channels


def forward_check(model, use_dc):
    model.eval().to(DEV)
    x_img = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    x_ksp = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    mask  = torch.zeros(1, 1, H, W, device=DEV); mask[..., ::4] = 1.0
    sens  = torch.randn(1, C.INPUT_CHANNELS, H, W, device=DEV)
    with torch.no_grad():
        out = model(x_img, x_ksp, mask, sens)
    return tuple(out.shape)


def main():
    print(f"device={DEV}  H=W={H}  H2={C.N_HIDDEN_LRNN_2}  unet_depth={C.UNET_DEPTH}")
    results = {}
    for use_dc in (False, True):
        for seq in ('gru', 'ss2d'):
            key = f"{seq}_{'DC' if use_dc else 'noDC'}"
            m = build(seq, use_dc)
            in_ch = unet_in_ch(m)
            n_params = sum(p.numel() for p in m.parameters()) / 1e6
            shape = forward_check(m, use_dc)
            results[key] = (in_ch, shape, n_params)
            print(f"  [{key:12s}] unet_in_ch={in_ch}  out_shape={shape}  params={n_params:.1f}M")
            assert shape == (1, 1, H, W), f"{key} 출력 shape 오류: {shape}"
            del m
            if DEV.type == 'cuda':
                torch.cuda.empty_cache()

    # 2) 같은 DC 레벨에서 GRU/SS2D 의 U-Net in_channels 동일
    for tag in ('noDC', 'DC'):
        a = results[f'gru_{tag}'][0]
        b = results[f'ss2d_{tag}'][0]
        assert a == b, f"{tag}: GRU unet_in_ch({a}) != SS2D unet_in_ch({b}) — downstream 불일치!"
        print(f"  [contract:{tag}] GRU/SS2D unet_in_ch 동일 = {a}  OK")

    expected_in = 2 * C.N_HIDDEN_LRNN_2 + C.N_COIL * 2
    assert results['gru_noDC'][0] == expected_in, "in_channels 계약 위반"
    print(f"  [contract] in_channels == 2*H2+2*coil == {expected_in}  OK")

    # 4) 교수님/프로젝트 원본 파일 무수정 (git) — 교수님 원본은 7d4e4e0 대비, 프로젝트 공유 원본은 git status
    print("\n  [git] 교수님 원본 파일 무수정 확인:")
    check_professor_originals()
    targets = ['models/hybrid_eternet/', 'models/mamba_eternet/ss2d.py']
    dirty = subprocess.run(
        ['git', '-C', _PROJECT_ROOT, 'status', '--porcelain'] + targets,
        capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        print("    [WARN] 다음 원본 파일에 변경 감지:\n" + dirty)
        raise SystemExit("원본 파일이 수정됨 — 통제 비교 무효. 확인 요망.")
    print("    OK — 원본 무수정")

    print("\n✅ SANITY PASS — only-seq-model(±DC) differs, 차원 계약 충족, 원본 무수정.")


if __name__ == '__main__':
    main()
