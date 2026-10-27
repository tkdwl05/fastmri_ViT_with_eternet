"""재현 설정 — 모든 값과 그 출처를 한 파일에 둔다.

각 설정 = (이름, 값, 출처, 상태). 상태 표기:
  code     교수님 코드에서 바로 읽힘(확정)                       → 확정
  data     데이터 모양이 정함(.mat (16,384,384,16,2))            → 확정
  residue  configs/myConfig_temp.py:108-221 주석 잔재            → [추정]  (그 블록은 231030 config 자체가 아니다: verify_design VD-04)
  run_name 런 폴더 이름 태그(잔재 :157 런 #22 등)                → [추정]  (#22 이름은 #21 에서 복사됐을 수 있다)
  paper    교수님 논문(다른 실험)                                → [추정]
  shim     없는 인자 3개 myOptimizer 의 대체 가정                → [추정]
  design   교수님 코드에 없어 우리가 정함                        → [설계·추정]
  ours     교수님 레시피에 없는 우리 추가(note 에 계산 영향 표시) → [추가]
  override 명령줄 --set 으로 바꿈                                → [변경]
값 이름은 교수님 config 변수 이름을 따른다(있는 경우). 시작할 때 교수님 스크립트처럼(231030:49-54) 전부 출력한다.

약칭: 231030 = scripts_legacy/choh_train_ETER_hybrid_fastmri_brain_acs32R4_231030.py (d-ETER),
      231022 = scripts_legacy/choh_train_oETER_fastmri_brain_fixed_acs32R4_231022.py (o-ETER),
      230425 = dataloaders/myDataloader_fastmri_brain_230425.py, 잔재 = configs/myConfig_temp.py,
      model.py = models/hybrid_eternet/hybrid_eternet_fastmri-main/model.py (공개 hybrid d-ETER).
"""
import ast
import datetime
from collections import OrderedDict
from dataclasses import dataclass, field

from repro_prof import prof_paths as P

ASSUMED_STATUSES = ('residue', 'run_name', 'paper', 'shim', 'design', 'override')
STATUS_TAG = {'code': '확정', 'data': '확정', 'residue': '추정', 'run_name': '추정', 'paper': '추정', 'shim': '추정',
              'design': '설계·추정', 'ours': '추가', 'override': '변경'}


@dataclass
class Setting:
    name: str
    value: object
    src: str
    status: str
    note: str = ''

    @property
    def assumed(self):
        return self.status in ASSUMED_STATUSES


def _S(name, value, src, status, note=''):
    assert status in STATUS_TAG, status
    return Setting(name, value, src, status, note)


# ----------------------------------------------------------------------------------------------- 기본 = d-ETER 런 #22
BASE = [
    # --- 데이터
    _S('MAT_DIR', P.PROF_MAT_DIR, "230425:489 '/mnt/sdb/...mat_brain_multi_train_16ch_396ky_ksp/' 와 같은 폴더(접두사만 이 서버의 /mnt/sda)", 'code'),
    _S('LIST_FILE', str(P.PROF_LIST_FILE), '230425:496 list_brain_train_320.txt (379줄, 학습 0~299·검증 300~319·시험 320~378)', 'code'),
    _S('TRAIN_IDX_START', 0, '231030:208 idx_start = NUM_VOL_PER_SET·n_set (0부터)', 'code'),
    _S('NUM_TRAIN_SET', 60, '잔재 :173 `# # NUM_TRAIN_SET = 60` (이중 주석 = 그때도 꺼진 줄). 대안 3(ViT config myConfig_choh_ViT_ETER_R4regular.py:35)', 'residue',
       '묶음 수 × 묶음 크기 = 학습 볼륨 수(300: 런 이름 t300, 231030:202 폭 4,800 = 300×16)'),
    _S('NUM_VOL_PER_SET', 5, '잔재 :174 `# # NUM_VOL_PER_SET = 5` (이중 주석). 대안 100(ViT config :36)', 'residue',
       '묶음마다 .mat 를 다시 읽고 묶음 안에서만 섞는다(231030:205-216). 묶음 크기는 섞임 범위만 바꾼다'),
    _S('VAL_IDX', 300, '231030:179 `_val(num_total_set=1)` → 230425:207 data[300:301] (목록 300번 1볼륨)', 'code'),
    _S('TEST_IDX_RANGE', (320, 379), '230425:304 `_test59` data[320:379]. 231030 은 시험을 하지 않는다 → 집합은 추정(강함)', 'residue'),
    _S('MASK', 'regular ky 3:384:4 ∪ ACS 176:208 (120/384줄, R_eff 3.2), 모든 슬라이스·epoch 같음', '230425:525-526 (_test_v2), 검증 클래스 :228-229 도 같음', 'code'),
    _S('AMP_X_KSP', 1e3, '230425:478 (k-space ×1e3)', 'code'),
    _S('AMP_X_IMG', 1e5, '230425:477 (zero-filled 영상 ×1e5)', 'code'),
    _S('AMP_Y', 1e5, '230425:479 (정답 ×1e5)', 'code'),
    _S('FFT', 'numpy 기본(backward) 정규화 중심 ifft2, 축1 fftshift·축2 ifftshift', '230425:28-50 (ortho 줄은 주석 :41)', 'code'),
    _S('LABEL', '완전 샘플 .mat 의 16코일 RSS (384×384 전체)', '230425:516-522', 'code'),
    _S('AUGMENT', None, '231030·230425 에 증강 없음', 'code'),
    _S('SHUFFLE_WITHIN_SET', True, '231030:212 DataLoader(shuffle=True) — 묶음 안에서만', 'code'),
    _S('LOADER_WORKERS', 0, '231030:212 DataLoader 기본 num_workers=0', 'code'),
    _S('DATA_SOURCE', 'cache', "우리: 'cache' = data_cache.py (원 로더 출력과 비트 단위 같음, 10-23 볼륨 0·1·43·62·300 확인) / 'prof' = 교수님 클래스를 그대로 실행(볼륨당 약 10.3 s)", 'ours',
       '계산 결과 무관(입력 비트 동일)'),
    _S('CACHE_DIR', '/home/snorlax/shared/cache_repro_prof/prof379_f32_v1', '우리: data_cache.DEFAULT_CACHE_DIR (NVMe, 39.34 GB 계획)', 'ours'),
    _S('PREFETCH_THREADS', 6, '우리: 다음 묶음을 백그라운드 스레드로 미리 만든다(cache 일 때만, numpy FFT 는 GIL 해제)', 'ours', '계산 결과 무관'),
    # --- 모델
    _S('MODEL', 'eter', "'eter' | 'unet_only'", 'code'),
    _S('ETER_VARIANT', 'd', "231030:101·151 d-ETER (U-Net 입력 = cat(GRU 출력, zero-filled 코일 영상)). 'o' = 231022:103·157", 'code'),
    _S('N_RFCOIL', 16, '데이터 .mat 16코일; 잔재 :196, model.py:6', 'data'),
    _S('N_INPUT_VERTICAL', 384, '데이터 384 격자; 잔재 :197, model.py:7', 'data'),
    _S('N_FREQ_ENCODING', 384, '데이터 384(영 채움 전체 격자); 잔재 :199, model.py:8', 'data'),
    _S('N_OUT_X', 384, '잔재 :200, model.py:9', 'data'),
    _S('N_OUT_Y', 384, '잔재 :201, model.py:10', 'data'),
    _S('N_HIDDEN_LRNN_1', 12, '잔재 :205, 런 #22 이름 nh12(:157), model.py:12 — 세 근거가 서로 독립은 아니다', 'residue'),
    _S('N_HIDDEN_LRNN_2', 12, '잔재 :206, 런 이름 nh12, model.py:13', 'residue'),
    _S('GRU', '1층 양방향 nn.GRU 2개(gru_h, gru_v), h0 = 0, dropout 없음', '231030:96·114-115·123-124 (잔재 R_GRU_DROPOUT=0.5·이름 drop05 는 코드에 효과 없음)', 'code'),
    _S('REFINE', 'dfu', "231030:119 UNet_choh_skip. 선택: 'dfu_plain'(myUNet_DF.UNet) · 'unet2015'(원래 U-Net) · 'fastmri'(fastmri.models.Unet)", 'code'),
    _S('N_UNET_DEPTH', 3, '잔재 :193 `N_UNET_DEPTH = 3 #4 #5`, 런 #22 이름 _u3_(:157). 공개 model.py:15 는 5 → 모름', 'residue'),
    _S('UNET_WF', 6, '231030:119 wf=6 (첫 층 64채널)', 'code'),
    _S('UNET_BATCH_NORM', False, '231030:119 batch_norm=False', 'code'),
    _S('UNET_UP_MODE', 'upconv', "231030:119 up_mode='upconv' (padding=True 는 myUNet_DF.py:146 기본값)", 'code'),
    _S('BASELINE_INPUT', 'coil32', "U-Net 단독일 때만: 'coil32' = data_img 32채널(d-ETER 의 U-Net 이 받는 영상) / 'rss1' = 코일 결합 크기 1채널. 교수님 기준선 입력은 어느 자료에도 없다(S22 'image-domain to image-domain' 뿐)", 'design'),
    _S('FASTMRI_CHANS', None, "fastmri 정제망일 때: None → 2**UNET_WF(=64, JI26 'UNet depth 5, 첫 층 64채널'). fastMRI 예제 기본 32, 리더보드 256", 'design'),
    _S('FASTMRI_POOLS', None, 'fastmri 정제망일 때: None → N_UNET_DEPTH − 1 (해상도 수준 수 = depth). fastMRI 기본 4', 'design'),
    _S('FASTMRI_DROP', 0.0, 'fastMRI Unet 기본 drop_prob 0.0', 'design'),
    _S('FASTMRI_NORM', 'none', "fastmri 정제망일 때: 'none' / 'instance'(표본별 평균·표준편차로 정규화 후 되돌림, fastMRI 식). 손실은 원 배율에서", 'design'),
    _S('U2015_DROPOUT', 0.0, '원래 U-Net: 논문은 수축 경로 끝 drop-out(확률 미기재) → 기본 끔', 'design'),
    _S('U2015_INIT', 'pytorch_default', "원래 U-Net: 'pytorch_default'(교수님 DFU 와 같은 기본) / 'he_normal'(논문 sqrt(2/N))", 'design'),
    _S('U2015_PADDING', True, '원래 U-Net: 3×3 conv zero padding → 출력 384×384 (논문은 unpadded + crop)', 'design'),
    # --- 손실
    _S('CRITERION', 'l1', "잔재 :191 `# # CRITERION = 'l1'`(이중 주석), 런 이름 'L1'. myCriterion 기본값은 'mse'(myUtils.py:111)", 'residue'),
    _S('LAMBDA_SSIM_PER_PIXEL', 0.2, '잔재 :179 (켜진 줄), 런 이름 lssim02 — 두 근거는 독립이 아니다', 'residue'),
    _S('LOSS', 'criterion(out, ref) + λ·(1 − mySSIM(out, ref)), 384×384 전체', '231030:234-236', 'code'),
    _S('SSIM_MODULE', 'mySSIM.SSIM() — 11×11 가우시안 σ1.5, L 은 예측값에서 자동(최댓값 >128 → 255, 아니면 1)', '231030:13·193, models/hybrid_eternet/mySSIM.py:24-37·94-111 (import 시 cudnn.deterministic=True, :6)', 'code'),
    # --- 최적화
    _S('OPTIMIZER', 'else', "잔재 :186 `# # OPTIMIZER = 'else' #'adam'`(이중 주석). 저장소 myUtils 'else' = Adam lr 1e-4, WD 없음(:104-106)", 'residue'),
    _S('LEARNING_RATE_ADAM', 1e-4, '잔재 :177 (켜진 줄). S22·JI26 도 Adam 1e-4. lr031(≈1e-3, 추론) 런은 "divergence nan"(:140)', 'residue'),
    _S('LAMBDA_REGULAR_PER_PIXEL', 1e-7, "잔재 :178, 런 이름 reg071. d-ETER 에서 실제 적용 여부 모름 → MYOPT_MODE='coupled_wd' 일 때만 쓴다", 'residue'),
    _S('MYOPT_MODE', 'lr_passthrough', "myoptimizer_shim.py: 231030:191 의 인자 3개 myOptimizer 대체. 'lr_passthrough'(분기 그대로·lr 만 전달·WD 0) / 'legacy_2arg' / 'coupled_wd'(231030:192 주석 줄 형태)", 'shim'),
    _S('ADAM_FOREACH', None, '우리: None = torch 2.3.1 기본(CUDA 에서 foreach). 교수님 당시 torch 판은 모름. False 면 step 임시 메모리 약 3.4 GB 감소', 'ours', '같은 알고리즘, 마지막 자리 반올림만 다를 수 있음'),
    _S('LR_SCHEDULE', None, '231030 에 스케줄러 없음(고정 lr)', 'code'),
    _S('GRAD_CLIP', None, '231030:237-239 clip 없음', 'code'),
    _S('PRECISION', 'fp32', '231030:218-220 torch.cuda.FloatTensor, AMP 없음', 'code'),
    _S('TF32', False, '우리: TITAN RTX(Turing)에는 TF32 가 없다. Ampere(3090)에서는 matmul·cudnn TF32 를 꺼서 fp32 를 맞춘다(JI26: 교수님 TITAN RTX)', 'ours', 'Ampere 에서만 계산에 영향'),
    _S('BATCH_SIZE', 2, '잔재 :184 `BATCH_SIZE = 2 #1 #2 #8`. 대안 1(231030:202 폭 4,800 = BS 1 의 epoch 당 step 수)', 'residue'),
    _S('NUM_EPOCHS', 50, '잔재 :185, 런 이름 ep50 — 두 근거는 독립이 아니다', 'residue'),
    # --- 검증·저장·기록
    _S('VAL_RECORD', '검증 1볼륨을 no_grad 로 돌고 마지막 배치 criterion 만 출력·저장, 모델 선택 없음', '231030:268-277', 'code'),
    _S('SAVE_EVERY_EPOCH', True, '231030:281 매 epoch tensors_{epoch}.pt, 끝에 tensors_entire.pt(:286), training/validation_loss.npy(:289-293)', 'code'),
    _S('CKPT_FORMAT', 'state_dict', "우리: 'state_dict'(이식성) / 'full_model'(교수님처럼 torch.save(model), 231030:281)", 'ours', '저장 형식만 다름'),
    _S('SAVE_RESUME', True, '우리: epoch 끝마다 model+optimizer+RNG 를 resume.pt 하나에 덮어씀(교수님 코드에는 재개 없음)', 'ours', '계산 결과 무관'),
    _S('SEED', 1, '우리: 교수님 코드에는 시드가 없다(231030 전문). 고정해 기록하고 반복 학습에 쓴다', 'ours', '초기값·순서가 정해짐(분포는 같음)'),
    _S('DATA_ORDER', 'independent', "우리: 'independent' = 묶음·epoch 마다 SEED 로 만든 전용 생성기로 섞음(모델이 달라도 같은 순서) / 'global' = 교수님처럼 전역 RNG", 'ours',
       '섞는 방식은 같고, 순서가 모델 초기화와 독립이 됨'),
    _S('HEALTH_EVERY_STEPS', 500, '우리: v8fix 감시(main_train_pure_v8fix.py:34-36). 그룹별 |w|<1e-6 비율 기록', 'ours', '기록만'),
    _S('COLLAPSE_ACTION', 'log', "우리: 'log'(기본, 개입 없음) / 'stop'(seq·unet_deep ≥ 50% 이면 ALERT_COLLAPSE 쓰고 exit 3)", 'ours'),
    _S('EXTRA_VAL_METRICS', True, '우리: 검증 볼륨 전체 배치의 평균 criterion·mySSIM 을 함께 기록(선택에 쓰지 않음)', 'ours', '기록만'),
    _S('LOG_ROOT', str(P.REPO / 'logs' / 'repro_prof'), '우리: log.txt·loss 배열·요약(교수님 PATH_FOLDER 역할)', 'ours'),
    _S('CKPT_ROOT', '/mnt/sda/snorlax/repro_prof/ckpt', '우리: 체크포인트(HDD, 10-23 여유 6.3 TB). d-ETER H12 depth3 한 런 ≈ 3.41 GB × 51 + resume 10.2 GB ≈ 184 GB', 'ours'),
    _S('RUN_NAME', None, '우리: None 이면 자동 이름', 'ours'),
]

# ----------------------------------------------------------------------------------------------- 실험 조건 프리셋
PRESETS = {
    'dETER': [],   # 기본 = 런 #22 재구성
    'oETER': [
        _S('ETER_VARIANT', 'o', '231022:103·157 o-ETER (U-Net 입력 = GRU 출력만, 2020 논문형 k-space 단독 입력)', 'code'),
        _S('MYOPT_MODE', 'coupled_wd', '231022:198 torch.optim.Adam(lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL) (형태 확정, 값은 잔재)', 'code'),
        _S('MASK', 'regular ky 3:384:4 ∪ ACS 176:208 (d-ETER 와 같음)',
           '설계: 교수님 o-ETER 런(#18~#21)은 없는 fixed 로더(아마 R4_idx_part 96줄)나 random 96줄 로더를 썼다(verify_design VD-03). 여기서는 d-ETER 와 같은 230425 로더', 'design'),
        _S('VAL_IDX', 300, '설계: d-ETER 와 같은 검증 볼륨. 교수님 o-ETER fixed 는 310번(231022:177)', 'design'),
    ],
    'unet_only': [
        _S('MODEL', 'unet_only', '설계: 교수님 코드에 U-Net 단독 경로 없음(7d4e4e0 grep). zero-filled 영상만 받는 독립 U-Net', 'design'),
        _S('N_UNET_DEPTH', 3, '설계: ETER-net 의 U-Net 깊이와 같게(시퀀스 모듈만 다르게). 교수님 최근 기준선은 depth 5·64채널(JI26) → 5 도 후보', 'design'),
    ],
}


class Config:
    """설정 묶음. cfg['NAME'] 으로 값, cfg.settings 로 (출처·상태) 를 본다."""

    def __init__(self, preset='dETER', overrides=None):
        if preset not in PRESETS:
            raise KeyError(f'preset {preset!r} (가능: {list(PRESETS)})')
        self.preset = preset
        self.settings = OrderedDict((s.name, s) for s in BASE)
        for s in PRESETS[preset]:
            if s.name not in self.settings:
                raise KeyError(s.name)
            self.settings[s.name] = s
        for k, v in (overrides or {}).items():
            if k not in self.settings:
                raise KeyError(f'알 수 없는 설정 {k}')
            old = self.settings[k]
            self.settings[k] = Setting(k, v, f'명령줄 --set (원래 {old.value!r}: {old.src})', 'override', old.note)
        self._derive()

    def __getitem__(self, k):
        return self.settings[k].value

    def get(self, k, default=None):
        return self.settings[k].value if k in self.settings else default

    def _derive(self):
        c = self
        if c['MODEL'] == 'eter':
            desc = f"{c['ETER_VARIANT']}ETER_H{c['N_HIDDEN_LRNN_2']}_{c['REFINE']}_u{c['N_UNET_DEPTH']}"
        else:
            desc = f"UNETonly_{c['REFINE']}_{c['BASELINE_INPUT']}_u{c['N_UNET_DEPTH']}"
        auto = (f"{datetime.date.today():%y%m%d}_repro_{desc}_bs{c['BATCH_SIZE']}_lr{c['LEARNING_RATE_ADAM']:g}"
                f"_lssim{c['LAMBDA_SSIM_PER_PIXEL']:g}_{c['CRITERION']}_ep{c['NUM_EPOCHS']}_s{c['SEED']}")
        self.run_name = c['RUN_NAME'] or auto
        self.train_idx = list(range(c['TRAIN_IDX_START'], c['TRAIN_IDX_START'] + c['NUM_TRAIN_SET'] * c['NUM_VOL_PER_SET']))
        self.sets = [list(range(c['TRAIN_IDX_START'] + c['NUM_VOL_PER_SET'] * n,
                                c['TRAIN_IDX_START'] + c['NUM_VOL_PER_SET'] * n + c['NUM_VOL_PER_SET']))
                     for n in range(c['NUM_TRAIN_SET'])]

    def as_dict(self):
        return OrderedDict((k, s.value) for k, s in self.settings.items())

    def assumed(self):
        return [s for s in self.settings.values() if s.assumed]

    def print_all(self, write=print):
        """교수님 스크립트처럼(231030:49-54) 시작할 때 모든 설정을 출력한다. 출처·상태를 덧붙인다."""
        write('\nBATCH_SIZE : %d' % self['BATCH_SIZE'])
        write(f'preset {self.preset}  run {self.run_name}')
        for s in self.settings.values():
            flag = ' <ASSUMED>' if s.assumed else ''
            write(f'{s.name} \t {s.value!r} \t [{STATUS_TAG[s.status]}{flag}] {s.src}' + (f'  ({s.note})' if s.note else ''))
        write(f'train volumes {self.train_idx[0]}..{self.train_idx[-1]} ({len(self.train_idx)}), '
              f'{len(self.sets)} sets × {self["NUM_VOL_PER_SET"]}, val {self["VAL_IDX"]}')
        a = self.assumed()
        write(f'assumed/design/override settings ({len(a)}): ' + ', '.join(s.name for s in a))
        write('BATCH_SIZE : %d\n' % self['BATCH_SIZE'])


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        k, _, v = p.partition('=')
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out
