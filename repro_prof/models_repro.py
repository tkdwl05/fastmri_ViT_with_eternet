"""재현용 모델: 교수님 ETER-net 클래스 사본(o-ETER·d-ETER), 정제망 선택, U-Net 단독 기준선.

ETERNetRepro = 교수님 `ETER_hybrid_GRU_DFU`(231030:87-157, d-ETER)와 `oETER_GRU_DFU`(231022:90-163, o-ETER)의 사본.
원본과 다른 점은 셋뿐이다.
  (1) 모듈 수준 config 상수(N_INPUT_VERTICAL 등)를 생성자 인자로 받는다.
  (2) h0 를 `.cuda()` 대신 입력과 같은 장치·dtype 의 0 으로 만든다(GPU 에서는 같은 값).
  (3) 정제망을 고를 수 있다. 기본 'dfu' = 원본과 같은 UNet_choh_skip(myUNet_DF.py 를 그대로 import).
층 생성 순서(gru_h → gru_v → unet), state_dict 키, forward 식은 원본과 같다. 원문 클래스와의 동등성(같은 초기값·같은 출력)은
smoke_gpu.py 가 원문 클래스를 AST 로 잘라 실행해 비교한다.

정제망(refine)
  'dfu'       교수님 UNet_choh_skip (DFU 잔차 + 마지막에 원입력을 이어 붙인 1×1 conv). 입력 채널 = 2·n_hidden 이어야 한다
              (myUNet_DF.py:166,193).
  'dfu_plain' 교수님 같은 파일의 UNet 클래스(myUNet_DF.py:10-76). DFU 잔차는 같고 마지막 원입력 연결이 없다(입력 채널 제약 없음).
  'unet2015'  원래 U-Net(Ronneberger 2015, unet2015.py).
  'fastmri'   fastmri 패키지 0.3.0 의 fastmri.models.Unet(InstanceNorm, LeakyReLU 0.2, avg pooling, transpose conv).
              chans 기본 = 2**wf, num_pool_layers 기본 = depth − 1(해상도 수준 수 = depth 가 되도록).
U-Net 단독 기준선(UNetOnlyRepro)은 교수님 코드에 없다(설계). 로더의 zero-filled 코일 영상(data_img)만 받는다.
"""
import torch
import torch.nn as nn

from repro_prof import prof_paths as P
from repro_prof.unet2015 import UNet2015

REFINERS = ('dfu', 'dfu_plain', 'unet2015', 'fastmri')


class NormWrap(nn.Module):
    """fastMRI 식 표본별 정규화 감싸개(선택): out = net((x − μ)/σ)·σ + μ, μ·σ 는 표본마다 입력 전체의 평균·표준편차.

    손실은 원래 배율(×1e5)에서 계산되도록 출력을 되돌린다. 교수님 레시피에는 없다(설계).
    """

    def __init__(self, net, eps=1e-11):
        super().__init__()
        self.net, self.eps = net, eps

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        sd = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        return self.net((x - mu) / sd) * sd + mu


def build_refiner(kind, in_ch, n_hidden=None, depth=3, wf=6, fastmri_chans=None, fastmri_pools=None,
                  fastmri_drop=0.0, fastmri_norm='none', u2015_dropout=0.0, u2015_init='pytorch_default',
                  u2015_padding=True):
    if kind == 'dfu':
        unet_mod = P.prof_unet_module()
        n_hidden = in_ch // 2 if n_hidden is None else n_hidden
        if in_ch != 2 * n_hidden:
            raise ValueError(f'UNet_choh_skip 은 입력 채널 = 2·n_hidden 이어야 한다(in {in_ch}, n_hidden {n_hidden}; '
                             'myUNet_DF.py:166,193). 1채널 입력은 dfu_plain·unet2015·fastmri 를 쓴다')
        # 231030:119 와 같은 인자
        return unet_mod.UNet_choh_skip(in_channels=in_ch, n_classes=1, depth=depth, wf=wf, batch_norm=False,
                                       up_mode='upconv', n_hidden=n_hidden)
    if kind == 'dfu_plain':
        unet_mod = P.prof_unet_module()
        return unet_mod.UNet(in_channels=in_ch, n_classes=1, depth=depth, wf=wf, padding=True, batch_norm=False,
                             up_mode='upconv')
    if kind == 'unet2015':
        return UNet2015(in_ch, 1, depth=depth, wf=wf, padding=u2015_padding, dropout=u2015_dropout, init=u2015_init)
    if kind == 'fastmri':
        from fastmri.models import Unet
        chans = 2 ** wf if fastmri_chans is None else int(fastmri_chans)
        pools = depth - 1 if fastmri_pools is None else int(fastmri_pools)
        net = Unet(in_chans=in_ch, out_chans=1, chans=chans, num_pool_layers=pools, drop_prob=fastmri_drop)
        if fastmri_norm == 'instance':
            return NormWrap(net)
        if fastmri_norm != 'none':
            raise ValueError(fastmri_norm)
        return net
    raise ValueError(f'refine={kind!r} (가능: {REFINERS})')


class ETERNetRepro(nn.Module):
    def __init__(self, variant='d', n_in_x=384, n_freq=384, n_coil=16, n_out_x=384, n_out_y=384, h1=12, h2=12,
                 refine='dfu', depth=3, wf=6, **refine_kw):
        super().__init__()
        if variant not in ('d', 'o'):
            raise ValueError(variant)
        num_in_x, num_in_y, num_out_x, num_out_y = n_in_x, n_freq, n_out_x, n_out_y
        input_size = num_in_y * n_coil * 2                       # 231030:95
        num_layers = 1                                           # 231030:96
        num_out1 = num_out_y * h1                                # :97
        num_in2 = num_in_x * h1                                  # :98
        num_out2 = num_out_x * h2                                # :99
        if variant == 'd':
            num_feat_ch = int(num_out2 * 2 / num_out_x) + n_coil * 2   # 231030:101
            n_hidden = h2 + n_coil                                     # 231030:102
        else:
            num_feat_ch = int(num_out2 * 2 / num_out_x)                # 231022:103
            n_hidden = h2                                              # 231022:106
        self.variant = variant
        self.num_in_x = num_in_x
        self.num_in_y = num_in_y
        self.num_layers = num_layers
        self.num_out1 = num_out1
        self.num_out2 = num_out2
        self.num_out_x = num_out_x
        self.num_out_y = num_out_y
        self.unet_in_ch, self.unet_n_hidden = num_feat_ch, n_hidden
        self.gru_h = nn.GRU(input_size, num_out1, num_layers, batch_first=True, bidirectional=True)   # :114
        self.gru_v = nn.GRU(num_in2 * 2, num_out2, num_layers, batch_first=True, bidirectional=True)  # :115
        self.unet = build_refiner(refine, num_feat_ch, n_hidden if refine == 'dfu' else None, depth, wf, **refine_kw)

    def forward(self, x, x_img=None):
        # 231030:122-157 / 231022:127-163 와 같은 식. h0 만 입력 장치에 만든다.
        h_h0 = torch.zeros(self.num_layers * 2, x.size(0), self.num_out1, device=x.device, dtype=x.dtype)
        h_v0 = torch.zeros(self.num_layers * 2, x.size(0), self.num_out2, device=x.device, dtype=x.dtype)
        in_h = x.reshape([x.size(0), self.num_in_x, -1])
        out_h, _ = self.gru_h(in_h, h_h0)
        out_h = out_h.reshape([x.size(0), self.num_in_x, self.num_out_y, -1])
        out_h = out_h.permute(0, 2, 1, 3)
        out_h = out_h.reshape([x.size(0), self.num_out_y, -1])
        out_v, _ = self.gru_v(out_h, h_v0)
        out_v = out_v.reshape([x.size(0), self.num_out_y, self.num_out_x, -1])
        out_v = out_v.permute(0, 3, 2, 1)
        if self.variant == 'd':
            in_cnn = torch.cat((out_v, x_img), dim=1)            # 231030:151
        else:
            in_cnn = out_v                                       # 231022:157
        return self.unet(in_cnn)


class UNetOnlyRepro(nn.Module):
    """U-Net 단독 기준선(설계, 교수님 코드에 없음). 입력은 로더의 data_img(zero-filled 코일 영상, ×1e5)뿐이다.

    input_form
      'coil32' : data_img 32채널(코일 c: 채널 2c 실수, 2c+1 허수) 그대로. d-ETER 의 U-Net 이 받는 영상과 같다.
      'rss1'   : 32채널 제곱합의 제곱근(코일 결합 크기 영상) 1채널. 정답(16코일 RSS ×1e5)과 같은 배율이다.
    k-space 입력(data)은 받지 않는다. 학습 루프에서 d-ETER 와 같은 호출(model(data, data_img))을 쓰도록 인자는 둘이다.
    """

    def __init__(self, input_form='coil32', refine='dfu', depth=3, wf=6, n_coil=16, **refine_kw):
        super().__init__()
        if input_form not in ('coil32', 'rss1'):
            raise ValueError(input_form)
        self.input_form = input_form
        in_ch = 2 * n_coil if input_form == 'coil32' else 1
        if refine == 'dfu' and in_ch % 2:
            raise ValueError("rss1(1채널)은 UNet_choh_skip 에 넣을 수 없다 → refine='dfu_plain'·'unet2015'·'fastmri'")
        self.unet_in_ch = in_ch
        self.unet = build_refiner(refine, in_ch, in_ch // 2 if refine == 'dfu' else None, depth, wf, **refine_kw)

    def make_input(self, x_img):
        if self.input_form == 'coil32':
            return x_img
        return torch.sqrt(torch.sum(x_img * x_img, dim=1, keepdim=True))

    def forward(self, x=None, x_img=None):
        return self.unet(self.make_input(x_img))


def build_model(cfg):
    """config(dict 형) → 모델. cfg 키는 config_repro.py 의 설정 이름."""
    rk = dict(fastmri_chans=cfg['FASTMRI_CHANS'], fastmri_pools=cfg['FASTMRI_POOLS'], fastmri_drop=cfg['FASTMRI_DROP'],
              fastmri_norm=cfg['FASTMRI_NORM'], u2015_dropout=cfg['U2015_DROPOUT'], u2015_init=cfg['U2015_INIT'],
              u2015_padding=cfg['U2015_PADDING'])
    if cfg['MODEL'] == 'eter':
        return ETERNetRepro(variant=cfg['ETER_VARIANT'], n_in_x=cfg['N_INPUT_VERTICAL'], n_freq=cfg['N_FREQ_ENCODING'],
                            n_coil=cfg['N_RFCOIL'], n_out_x=cfg['N_OUT_X'], n_out_y=cfg['N_OUT_Y'],
                            h1=cfg['N_HIDDEN_LRNN_1'], h2=cfg['N_HIDDEN_LRNN_2'], refine=cfg['REFINE'],
                            depth=cfg['N_UNET_DEPTH'], wf=cfg['UNET_WF'], **rk)
    if cfg['MODEL'] == 'unet_only':
        return UNetOnlyRepro(input_form=cfg['BASELINE_INPUT'], refine=cfg['REFINE'], depth=cfg['N_UNET_DEPTH'],
                             wf=cfg['UNET_WF'], n_coil=cfg['N_RFCOIL'], **rk)
    raise ValueError(cfg['MODEL'])


def param_breakdown(model):
    out = {'total': sum(p.numel() for p in model.parameters())}
    for name in ('gru_h', 'gru_v', 'unet'):
        m = getattr(model, name, None)
        if m is not None:
            out[name] = sum(p.numel() for p in m.parameters())
    return out


# 기대 파라미터 수 — 교수님 코드 식·논문 수치 (부록 A 산술, verify_design §3). (설명, 생성 인자, 기대값)
EXPECTED_PARAMS = [
    ('d-ETER H12 depth3 (교수님 231030 + 잔재 깊이 3)', dict(kind='eter', variant='d', h=12, depth=3), 851_350_585),
    ('d-ETER H12 depth5 (= 공개 hybrid model.py:12-15)', dict(kind='eter', variant='d', h=12, depth=5), 880_519_481),
    ('o-ETER H12 depth3 (231022 클래스)', dict(kind='eter', variant='o', h=12, depth=3), 851_332_121),
    ('o-ETER H12 depth5', dict(kind='eter', variant='o', h=12, depth=5), 880_501_017),
    ('U-Net 단독 DFU 입력32 depth3', dict(kind='unet_only', depth=3), 1_879_585),
    ('U-Net 단독 DFU 입력32 depth4', dict(kind='unet_only', depth=4), 7_714_081),
    ('U-Net 단독 DFU 입력32 depth5', dict(kind='unet_only', depth=5), 31_048_481),
    ('우리 v8 ETER-net H10 (d-ETER 식, depth5; log params=668.2M)', dict(kind='eter', variant='d', h=10, depth=5), 668_162_101),
    ('S22 Cartesian o-ETER (Ky 54·8코일·216², Nf 32, DFU depth4; Sensors 2022 p.7)',
     dict(kind='eter', variant='o', h=16, depth=4, n_in_x=216, n_freq=54, n_coil=8, n_out_x=216, n_out_y=216), 312_367_393),
    ('S22 방사형 o-ETER (100 views·4코일·256², Nf 32, DFU depth5; Sensors 2022 p.7)',
     dict(kind='eter', variant='o', h=16, depth=5, n_in_x=256, n_freq=100, n_coil=4, n_out_x=256, n_out_y=256), 453_460_769),
    ('2020 공개 eternet_fastmri o-ETER (131줄 입력, H15, depth3, 출력 384×396)',
     dict(kind='eter', variant='o', h=15, depth=3, n_in_x=384, n_freq=131, n_coil=16, n_out_x=384, n_out_y=396), 960_320_111),
]
EXPECTED_SUBMODULE = [
    ('JI26 BiRNN(hidden 384×10) = GRU H10 두 개 (J. Imaging 2026 §2)', dict(kind='eter', variant='d', h=10, depth=5),
     ('gru_h', 'gru_v'), 637_102_080),
    ('교수님 d-ETER H12 GRU 두 개', dict(kind='eter', variant='d', h=12, depth=3), ('gru_h', 'gru_v'), 849_457_152),
]


def make_for_count(spec, refine='dfu'):
    """파라미터 수 확인용 생성(호출자는 torch.device('meta') 문맥에서 부른다)."""
    spec = dict(spec)
    kind = spec.pop('kind')
    if kind == 'eter':
        h = spec.pop('h')
        return ETERNetRepro(variant=spec.pop('variant'), h1=h, h2=h, refine=refine, depth=spec.pop('depth'), **spec)
    return UNetOnlyRepro(input_form=spec.pop('input_form', 'coil32'), refine=refine, depth=spec.pop('depth'), **spec)
