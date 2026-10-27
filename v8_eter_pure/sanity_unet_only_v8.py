"""
v8 sanity: U-Net only 대조 모델(PureETER_UNET, 시퀀스 모듈 제거 = f_θ ≡ 0) 검증 (2026-10-01).

검사 (기본 = CPU 전용, 학습 중 GPU0 를 건드리지 않음):
  1. PureETER_UNET / PureETER_SS2D / PureETER_TRANSFORMER / PureETER_PIXELGRU
     (+ 메모리 허용 시 PureETER_GRU H=10, 668M) 의
     U-Net 서브모듈 state_dict key·tensor shape 가 완전히 동일 (U-Net in_channels=52, n_hidden=26 계약).
  2. 파라미터 수 (전체 / U-Net / 시퀀스 모듈).
  3. forward (B=1, 384²) → (1, 1, 384, 384).
  4. 1회 backward 후 0 채널(시퀀스 모듈 자리 20ch)을 읽는 첫 conv·마지막 1×1 conv 가중치 slice 의
     grad 가 정확히 0, 나머지(aliased image 32ch / decoder feature) slice 는 0 이 아님.
  5. 출력이 x_ksp 와 무관하고, 0 채널 가중치 slice 를 교란해도 출력이 비트 단위로 동일.

--gpu (선택, 학습 중에는 실행 금지): cuda:0 에서 trainer(main_train_pure_v8.py) 와 같은 학습 step
  (AMP forward → masked L1 + λ(1-SSIM) [u_choh_SSIM.SSIM] → GradScaler backward → unscale → clip 1.0
  → Adam step) 을 --bs 로 ~--iters 회 반복해 peak VRAM 과 s/iter 를 출력한다.
  합성 입력이므로 s/iter 는 데이터 로딩을 제외한 연산 시간. smoke_bs.txt·runs/ 에는 아무것도 쓰지 않는다.

실행 (저장소 루트):
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 nice -n 19 python v8_eter_pure/sanity_unet_only_v8.py
  CUDA_VISIBLE_DEVICES=0 python v8_eter_pure/sanity_unet_only_v8.py --gpu --bs 8      # GPU0 비어 있을 때만
trainer 모듈은 import 하지 않는다 (import 시 런 폴더가 생성됨) — build 인자는 trainer 의 build_model 과 동일하게 맞춘다.
"""

import os
import sys
import time
import argparse
import subprocess

import torch

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
sys.path.append(os.path.join(_HERE, 'configs'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'pure_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'hybrid_eternet'))   # trainer 와 동일 (u_choh_SSIM)
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'mamba_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'rnn_eternet'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'models', 'attn_eternet'))

import myConfig_pure_eter_v8 as C
from u_pure_eternet_unet import PureETER_UNET

H = W = C.IMAGE_SIZE[0]
ZERO_CH = 2 * C.N_HIDDEN_LRNN_2                     # 시퀀스 모듈 자리 (=20) — U-Net only 에서는 0
IMG_CH  = C.N_COIL * 2                              # aliased image (=32)
GRU_MIN_FREE_GB = 16.0                              # GRU(H=10, 668M) CPU 생성 허용 최소 가용 메모리


# ───────────────────────── build (trainer build_model 과 동일 인자) ─────────────────────────
def build(seq):
    if seq == 'unet':
        return PureETER_UNET(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    if seq == 'ss2d':
        from u_pure_eternet_ss2d import PureETER_SS2D
        return PureETER_SS2D(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            ss2d_d_inner=C.SS2D_D_INNER, ss2d_d_state=C.SS2D_D_STATE,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    if seq == 'transformer':
        from u_pure_eternet_transformer import PureETER_TRANSFORMER
        return PureETER_TRANSFORMER(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            axial_d_model=C.TRANSFORMER_D_MODEL, axial_n_pairs=C.TRANSFORMER_N_PAIRS,
            axial_n_heads=C.TRANSFORMER_N_HEADS,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    if seq == 'pixelgru':
        from u_pure_eternet_pixelgru import PureETER_PIXELGRU
        return PureETER_PIXELGRU(
            n_coil=C.N_COIL, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            pixelgru_hidden=C.PIXELGRU_HIDDEN,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    if seq == 'gru':
        from u_pure_eternet_gru import PureETER_GRU
        return PureETER_GRU(
            dim=H, n_coil=C.N_COIL,
            n_hidden_1=C.N_HIDDEN_LRNN_1, n_hidden_2=C.N_HIDDEN_LRNN_2,
            unet_depth=C.UNET_DEPTH, unet_wf=C.UNET_WF,
            use_dc=False, dc_k_scale_ratio=C.DC_K_SCALE_RATIO, dc_init_alpha=C.DC_INIT_ALPHA,
        )
    raise ValueError(seq)


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def unet_signature(model):
    return {k: tuple(v.shape) for k, v in model.unet.state_dict().items()}


def mem_available_gb():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    return 0.0


def first_conv(model):
    return model.unet.down_path[0].block[0]          # 3×3, in=52 (원입력 [0ch..20ch | aliased 32ch])


def last_conv(model):
    return model.unet.last                           # 1×1, in = cat([원입력 52ch, decoder 64ch]) = 116


def synthetic_batch(bs, device, seed=0):
    g = torch.Generator(device='cpu').manual_seed(seed)
    x_img = torch.randn(bs, IMG_CH, H, W, generator=g).to(device)
    x_ksp = torch.randn(bs, IMG_CH, H, W, generator=g).to(device)
    ref   = torch.randn(bs, 1, H, W, generator=g).abs().to(device)
    mask  = torch.zeros(bs, 1, H, W, device=device); mask[..., ::4] = 1.0
    sens  = torch.randn(bs, IMG_CH, H, W, generator=g).to(device)
    bm    = torch.zeros(bs, 1, H, W, device=device); bm[..., H // 5:4 * H // 5, W // 5:4 * W // 5] = 1.0
    return x_img, x_ksp, ref, mask, sens, bm


# ───────────────────────── CPU 검사 ─────────────────────────
def check_unet_identity(skip_gru):
    print('\n[1·2] U-Net 서브모듈 state_dict 동일성 + 파라미터 수')
    seqs = ['unet', 'ss2d', 'transformer', 'pixelgru']
    avail = mem_available_gb()
    if skip_gru:
        print('  GRU: --skip-gru 로 생략')
    elif avail >= GRU_MIN_FREE_GB:
        seqs.append('gru')
        print(f'  GRU(H=10) 생성: MemAvailable={avail:.0f} GB ≥ {GRU_MIN_FREE_GB:.0f} GB')
    else:
        print(f'  GRU 생략: MemAvailable={avail:.0f} GB < {GRU_MIN_FREE_GB:.0f} GB → SS2D/Transformer/pixel-GRU 와만 비교')

    ref_sig, rows = None, []
    for seq in seqs:
        torch.manual_seed(0)
        m = build(seq)
        sig = unet_signature(m)
        tot, un = n_params(m), n_params(m.unet)
        rows.append((seq, tot, un, tot - un, len(sig), first_conv(m).in_channels, last_conv(m).in_channels))
        if ref_sig is None:
            ref_sig = sig
        else:
            assert sig.keys() == ref_sig.keys(), \
                f'{seq}: U-Net state_dict key 불일치 — {set(sig) ^ set(ref_sig)}'
            diff = [k for k in sig if sig[k] != ref_sig[k]]
            assert not diff, f'{seq}: U-Net tensor shape 불일치 — {[(k, sig[k], ref_sig[k]) for k in diff]}'
        del m

    print(f'  {"model":11s} {"total":>14s} {"U-Net":>14s} {"seq module":>14s} {"#keys":>6s} '
          f'{"conv1_in":>8s} {"last_in":>7s}')
    for seq, tot, un, sq, nk, c1, cl in rows:
        print(f'  {seq:11s} {tot:>14,d} {un:>14,d} {sq:>14,d} {nk:>6d} {c1:>8d} {cl:>7d}')
    unet_counts = {r[2] for r in rows}
    assert len(unet_counts) == 1, f'U-Net 파라미터 수 불일치: {unet_counts}'
    assert rows[0][3] == 0, 'U-Net only 모델에 U-Net 외 파라미터가 존재'
    assert rows[0][5] == ZERO_CH + IMG_CH, f'conv1 in_channels != {ZERO_CH + IMG_CH}'
    print(f'  OK — U-Net {len(ref_sig)} keys·shape 동일 ({", ".join(r[0] for r in rows)}), '
          f'U-Net params = {rows[0][2]:,d}')
    return {r[0]: r for r in rows}


def check_forward_and_grad():
    print('\n[3·4·5] forward / backward / 0 채널 무기여 (B=1, 384², CPU)')
    torch.manual_seed(0)
    model = build('unet').train()
    x_img, x_ksp, ref, mask, sens, bm = synthetic_batch(1, torch.device('cpu'))

    t0 = time.time()
    out = model(x_img, x_ksp, mask, sens)
    t_fwd = time.time() - t0
    assert tuple(out.shape) == (1, 1, H, W), f'출력 shape 오류: {tuple(out.shape)}'
    print(f'  [3] out_shape={tuple(out.shape)}  ({t_fwd:.1f}s)  OK')

    # 4) 한 번 backward — trainer 의 masked L1 항 (SSIM 클래스는 CUDA 전용이라 CPU 검사에서는 L1 만 사용)
    m_sum = bm.sum().clamp(min=1.0)
    loss  = ((out.float() - ref).abs() * bm).sum() / m_sum
    model.zero_grad(set_to_none=True)
    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    g1 = first_conv(model).weight.grad               # (64, 52, 3, 3)
    gl = last_conv(model).weight.grad                # (1, 116, 1, 1)
    z1, i1 = g1[:, :ZERO_CH], g1[:, ZERO_CH:]
    zl, il, dl = gl[:, :ZERO_CH], gl[:, ZERO_CH:ZERO_CH + IMG_CH], gl[:, ZERO_CH + IMG_CH:]

    def nz(t):
        return int((t != 0).sum()), t.numel()

    print(f'  [4] loss={loss.item():.4f}  backward {t_bwd:.1f}s')
    for name, t in (('conv1.w[:, 0:20]  (0 채널)', z1), ('conv1.w[:, 20:52] (aliased)', i1),
                    ('last.w[:, 0:20]   (0 채널)', zl), ('last.w[:, 20:52]  (aliased)', il),
                    ('last.w[:, 52:116] (decoder)', dl)):
        k, n = nz(t)
        print(f'      {name:30s} |grad|max={t.abs().max().item():.3e}  nonzero={k}/{n}')
    assert torch.count_nonzero(z1) == 0 and torch.count_nonzero(zl) == 0, '0 채널 가중치 grad 가 0 이 아님'
    assert i1.abs().sum() > 0 and il.abs().sum() > 0 and dl.abs().sum() > 0, '정상 채널 grad 가 0'
    assert first_conv(model).bias.grad.abs().sum() > 0 and last_conv(model).bias.grad.abs().sum() > 0
    print('      OK — 0 채널 slice grad 정확히 0, aliased/decoder slice·bias grad ≠ 0')

    # 5) 출력이 x_ksp 와 무관 + 0 채널 가중치 slice 교란 불변
    model.eval()
    with torch.no_grad():
        o_ref = model(x_img, x_ksp, mask, sens)
        o_ksp = model(x_img, torch.randn_like(x_ksp), mask, sens)
        first_conv(model).weight[:, :ZERO_CH].add_(torch.randn_like(first_conv(model).weight[:, :ZERO_CH]))
        last_conv(model).weight[:, :ZERO_CH].add_(torch.randn_like(last_conv(model).weight[:, :ZERO_CH]))
        o_pert = model(x_img, x_ksp, mask, sens)
    d_ksp  = (o_ref - o_ksp).abs().max().item()
    d_pert = (o_ref - o_pert).abs().max().item()
    print(f'  [5] max|Δout| x_ksp 교체={d_ksp:.3e} (equal={torch.equal(o_ref, o_ksp)})  '
          f'0채널 가중치 교란={d_pert:.3e} (equal={torch.equal(o_ref, o_pert)})')
    assert torch.equal(o_ref, o_ksp), 'no-DC 출력이 x_ksp 에 의존'
    assert d_pert <= 1e-6 * max(1.0, o_ref.abs().max().item()), '0 채널 가중치가 출력에 기여'
    print('      OK — f_θ ≡ 0: 0 채널 가중치는 출력에 무기여')
    return t_fwd, t_bwd


def check_originals_untouched():
    print('\n[git] 교수님 원본 무수정 확인 (읽기 전용)')
    targets = ['models/hybrid_eternet/myUNet_DF.py', 'models/hybrid_eternet/u_choh_SSIM.py']
    dirty = subprocess.run(['git', '-C', _PROJECT_ROOT, 'status', '--porcelain'] + targets,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit('원본 파일이 수정됨 — 통제 비교 무효:\n' + dirty)
    print('  OK — ' + ', '.join(targets))


# ───────────────────────── GPU 학습 step (선택) ─────────────────────────
def gpu_step_bench(bs, iters, warmup, force):
    from u_choh_SSIM import SSIM                     # trainer 와 동일 import (cudnn.deterministic=True 설정 포함)
    if not torch.cuda.is_available():
        raise SystemExit('--gpu: CUDA 를 사용할 수 없음 (CUDA_VISIBLE_DEVICES=0 확인)')
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    used_gb = (total - free) / 1024 ** 3
    print(f'\n[gpu] {torch.cuda.get_device_name(device)}  total={total/1024**3:.1f} GB  '
          f'사용 중={used_gb:.1f} GB')
    if used_gb > 2.0 and not force:
        raise SystemExit('[gpu] GPU0 에 다른 프로세스가 점유 중 — 학습 런 종료 후 실행 (무시하려면 --force)')

    torch.cuda.reset_peak_memory_stats(device)
    dt, last_loss = run_train_steps(device, SSIM, bs, iters, warmup,
                                    sync=lambda: torch.cuda.synchronize(device))
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    peak_resv  = torch.cuda.max_memory_reserved(device) / 1024 ** 3
    print(f'[gpu] BS={bs}  iters={iters} (warmup {warmup})  {dt:.3f} s/iter  '
          f'peak_alloc={peak_alloc:.2f} GB  peak_reserved={peak_resv:.2f} GB  last_loss={last_loss:.4f}')
    print('[gpu] (합성 입력 — 데이터 로딩 제외 연산 시간. smoke_bs.txt·runs/ 미기록)')


def run_train_steps(device, ssim_cls, bs, iters, warmup, sync):
    """trainer 학습 step 복제 (main_train_pure_v8.py: forward 만 autocast, loss 는 fp32 밖에서 계산,
    scaler.scale(loss/ACCUM) → unscale → clip 1.0 → step → update → zero_grad). 반환 (s/iter, 마지막 loss)."""
    torch.manual_seed(0)
    model = build('unet').to(device).train()
    criterion_ssim_loss = ssim_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.LEARNING_RATE_ADAM,
                                 weight_decay=C.LAMBDA_REGULAR_PER_PIXEL)
    scaler = torch.amp.GradScaler('cuda')
    accum = int(getattr(C, 'ACCUM_STEPS', 1))
    x_img, x_ksp, ref, mask, sens, bm = synthetic_batch(bs, device)

    def one_step():
        with torch.amp.autocast('cuda'):
            out = model(x_img, x_ksp, mask, sens)
        out_fp    = out.float()
        m_sum     = bm.sum().clamp(min=1.0)
        loss_l1   = ((out_fp - ref).abs() * bm).sum() / m_sum
        loss_ssim = 1 - criterion_ssim_loss(out_fp, ref, mask=bm)
        loss      = loss_l1 + C.LAMBDA_SSIM_PER_PIXEL * loss_ssim
        scaler.scale(loss / accum).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return loss.item()

    for _ in range(warmup):
        one_step()
    sync()
    t0 = time.time()
    last_loss = float('nan')
    for _ in range(iters):
        last_loss = one_step()
    sync()
    return (time.time() - t0) / max(1, iters), last_loss


def main():
    ap = argparse.ArgumentParser(description='v8 U-Net only 대조 모델 sanity')
    ap.add_argument('--gpu', action='store_true', help='cuda:0 학습 step 벤치 (학습 중 실행 금지)')
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--force', action='store_true', help='--gpu: GPU0 점유 검사 무시')
    ap.add_argument('--skip-gru', action='store_true', help='GRU(668M) 생성 생략')
    ap.add_argument('--threads', type=int, default=4, help='CPU torch 스레드 수')
    args = ap.parse_args()

    if args.gpu:
        gpu_step_bench(args.bs, args.iters, args.warmup, args.force)
        return

    torch.set_num_threads(args.threads)
    print(f'device=cpu  threads={args.threads}  H=W={H}  H2={C.N_HIDDEN_LRNN_2}  '
          f'zero_ch={ZERO_CH}  img_ch={IMG_CH}  unet_depth={C.UNET_DEPTH}  wf={C.UNET_WF}')
    check_unet_identity(args.skip_gru)
    check_forward_and_grad()
    check_originals_untouched()
    print('\nSANITY PASS — U-Net only: U-Net 은 다른 비교 모델과 동일, 0 채널(f_θ ≡ 0)은 출력·gradient 에 무기여.')


if __name__ == '__main__':
    main()
