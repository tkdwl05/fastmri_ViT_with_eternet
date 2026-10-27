"""d-ETER 최적화 교란 스모크 (2026-10-14) — 실험 조건 하나 = python 프로세스 하나.

질문: ETER-net 시퀀스 모듈(k-space bi-GRU → 20ch, zero-filled 32ch 와 concat → U-Net DFU)이 U-Net 단독보다 학습 손실이 높은 것은
최적화 교란 때문인가. 근거: docs/v8fix_setup_audit_2026-10-14.md §1-1·§1-2·§6-4, 설계 실측: results/smoke_deter_opt_2026-10-14/design_pilot/.
모든 조건: SEED 1, BS 8, 같은 배치 흐름(16 워커), 같은 U-Net 초기값, 같은 val 슬라이스, 4000 step, λ_SSIM = pod2 recipe.json 값
(손실 요인 조건 refF … A3P·ref2F … ref2P 만 예외 — 설계 결정 16).
step 의미는 pod2 ETER-net 트레이너(main_train_pure_v8fix_ampguard.py)와 같다: AdamW 분리형 + no-decay 그룹, eps 1e-8, clip 1e3(설계 결정 17 조건만 20),
fp16 autocast(gru_h 는 fp32), GradScaler + AMP_GRAD_RETRY 1 + AMP 하한 1024, NaN-retry 1, cosine T_max 406650(pod 50 epoch 값).

실험 조건 (CONDS):
  ref   U-Net 단독 (PureETER_UNET.forward — 시퀀스 자리 0)
  ref2  ref 와 같고 U-Net 초기값만 seed_all(2) — 회차 간 잡음 기준(null)                  — 기본 큐 밖, CONDS 로 지정할 때만
  A     현행 레시피 그대로
  B     GRU 그룹 LR ×0.1
  C     GRU 그룹만 선형 warmup W=1000 step (U-Net 그룹 그대로)
  D     seq_in = so.float()·0.06, GRU 그룹 eps = 0.06·1e-8 (Adam 갱신 법칙 보존)
  E     seq_in = α ⊙ so, α ∈ R^20 zero-init (no-decay 그룹, LR = base)
  EC    E + C 의 warmup (α 는 E 와 같음, LR = base)                                     — 기본 큐 밖
  A2    A 와 같고 GRU 초기값만 seed_all(2) (U-Net 초기값·배치 흐름은 A 와 같음) — ETER-net 회차 간 변동 확인 (10-15 추가, 기본 큐 밖)
  A3    A2 와 같고 GRU 초기값 seed_all(3) — A2 가 회차 의존으로 판정돼 회차 수를 늘림 (10-15 손실 요인 1단계, 기본 큐 밖)
  손실 설정 요인 (10-15 추가, 기본 큐 밖 — 설계 결정 16; 기준 조건(ref·ref2·A·A2·A3)과 preset 의 lam_ssim·loss_region 두 키만 다름):
  refF·ref2F·AF·A2F·A3F   F = 전체 영상 L1 + 전체 영상 SSIM (mask=None), λ = LAM (pod 값)
  refL·ref2L·AL·A2L·A3L   L = 뇌 마스크 L1 + 뇌 마스크 SSIM (현행과 같은 영역), λ = 0.2
  refP·ref2P·AP·A2P·A3P   P = 전체 영상 L1 + 전체 영상 SSIM, λ = 0.2 (교수님 ViT 학습 손실 형태)
  (어느 조건을 돌릴지는 런처 CONDS 로 정함 — 권장 1회차는 GRU seed 2·3 중심: smoke_deter_opt_loss_compare.py 머리말.
   ref2X = U-Net 초기값 seed 2 의 같은 설정 — 그 설정 아래 U-Net 회차 간 잡음 기준, 선택)
  Aeq   A 에 lam_ssim = LAM·loss_region = 'mask' 를 명시 — 새 분기가 기존 손실과 비트 단위로 같은지 보는 사전 점검 전용
        (--preflight 에서 설계 실측 meas.log 와 비트 비교; 본 실행 거부)
판정 규칙(사전 고정)·표·그림은 smoke_deter_opt_report.py (손실 요인 조건은 smoke_deter_opt_loss_compare.py,
학습 후 두 SSIM 규약 평가는 smoke_deter_opt_dualeval.py). 런처 = v8_eter_pure/runs/run_smoke_deter_opt.sh.

설계 결정 (spec 과 두 비판이 갈린 곳 — 이 판에서 고른 것과 이유):
  1. 기본 큐 = spec §8 그대로 ref A B C E D (사용자 승인 감사 §6-4 표의 조건 전부, 사전 점검 실측 ≈10.4 h). ref2(U-Net 초기값
     seed 2, 잡음 기준)는 과학 비판이 요구했지만 승인 표에 없으므로 CONDS 에 적을 때만 실행(맨 뒤, +≈0.45 h). 판정(1차)은
     spec §10 이고 ref2 는 보고서의 2차(비구속) 판정·잡음 기준 보고에만 쓰인다.
  2. 공정성 비판의 비트 단위 검증 채택: 사전 점검(--preflight)은 설계 실측 로그(meas*.log)와 step 별 loss·grad norm·amp_scale 을
     비트 단위로 비교한다(--compare-to). 1% 이내 비교로는 probe·기록이 학습을 바꾸는지 잡을 수 없다.
  3. 조건 간 동일성 기록: step 1·2·3 과 L_LIGHT 마다 unet_ck·gru_ck(%.17g), 매 step lr_unet. report 가 (a)~(e) 를 확인한다.
  4. 무작위 부분 표본은 파라미터 이름별 전용 generator (SUB_SEED + crc32(name)) — ref 와 GRU 모델의 U-Net 표본 원소가 같다.
  5. clip 발생 수를 step·구간별로 센다(E3 flag). E 의 gn 은 gate 포함(spec), 설계 실측 비교용으로 gate 제외 gn_model 도 기록.
  6. stream_md5 = 처음 steps 개 배치 위치(bpos 0..steps−1)만 — NaN-skip 으로 더 읽은 배치 때문에 거짓 불일치가 나지 않게.
  7. S2 는 양쪽 이웃 평균(i−1, i+1; 과학 비판) — i+1 우선 값은 보조 열(adj). GRU 특징 변화량 feat_change·var_change 를 probe 마다.
  8. EC 는 spec 대로 E + C 의 warmup, α LR = base (m_gate 1). 과학 비판의 α LR ×10 안(Adam 은 |α| ≤ lr·k 로 묶여 포화하지 않은
     GRU 의 크기에 닿지 못한다)은 사용자 결정 전에는 쓰지 않는다 — EC 는 기본 큐 밖.
  9. 선택 제안 중 채택: data_ck 를 GPU 에서 계산(CPU 합은 step 당 ~0.07 s), λ 를 pod 값과 fp32 비교, D eps = c·1e-8·fp32 비교,
     fp16 기울기 하한 감시(so 의 텐서 hook), partner 목록을 invariants.json 에 저장·대조, normal 출력도 시험과 같은 '저장 경로'로
     계산하고 직접 forward 와 같은지 1회 기록(I10), 최종 평가 전에 학습 워커 종료, gru_v 게이트 미분 재계산은 .half().float(),
     환경(torch·cuDNN·GPU·alloc conf)·LR 그룹 표를 meta·config_hash 에, S1 에 배치 sat_v 100-step 이동평균 추가, step 3500 에
     E256 정상 지표 스냅숏(report 가 3500·4000 평균으로 S4·순위 계산), 샘플별 L1 분자·마스크 합 기록.
     채택하지 않음: pod wandb 기록으로 V3 보정(외부 접속), annealed tail·3000 step 대안(예산·사용자 결정 필요).
  10. 출력 배치는 spec §7 ({cond}/summary.json 등)을 따른다. skip-done 은 status completed + 같은 config_hash(SCRIPT_VERSION 포함).
  11. 런처가 {cond}/stdout.log 를 job 시작 전에 열어 두므로, 재실행 때는 디렉터리 이름을 바꾸지 않고 stdout.log 를 뺀 이전 파일만
      {cond}.prev_YYYYmmdd-HHMMSS/ 로 옮긴다(삭제 없음).
  12. 끝 검사: E3(spec: nan_skip ≤ 10, amp_floor ≤ 5)은 flag, clip 발생은 따로 E3_clip(flag 전용, V2 에 쓰지 않음).
      E1·E2·E4·E5·E6 위반은 invalid(exit 5). E5(ref) = 0 입력 slice 가 갱신되지 않았는지(기울기 0, 감쇠도 실제로 0 — 아래 14).
  15. (10-15, 8 조건 완료 뒤 추가) A2 = A 에서 GRU 초기값만 seed 2. 8 조건 결과에서 V3(A 의 손실 열세)가 재현되지 않았고 pod ETER-net 은
      같은 구간 +10 % 였으므로, ETER-net 쪽 회차 간 변동을 잰다(사용자 10-15 승인). 바뀐 것은 preset 의 gru_seed 키 하나와 그 키가 있을
      때만 타는 분기(build_model 의 GRU 생성 seed, I2 대신 gru_ck_<cond> 기록)뿐이라 기존 8 조건의 동작·config_hash 는 그대로다
      (SCRIPT_VERSION 유지). 8 조건을 실행한 판 = results/smoke_deter_opt_2026-10-14/harness_as_run_ref-EC_md5_47aa0081.py.
      U-Net 은 지금처럼 seed_all(SEED) 로 만든 U-Net 단독 사본을 덮어쓰므로 A 와 같다(I1). 배치 흐름은 전용 generator 라 seed 와 무관(E2).
      보고 스크립트(ORDER)는 A2 를 읽지 않는다 — A2 비교는 따로.
  14. 분리형 weight decay 3e-5 는 이 LR(≤2e-4)에서 수치상 작동하지 않는다: 한 step 의 상대 감쇠 lr·wd ≤ 6e-9 가 fp32 반 ulp
      (≥2^-24 ≈ 5.96e-8)보다 작아 w·(1−lr·wd) 가 w 로 반올림된다(torch AdamW 는 스칼라도 fp32 로 계산). pod 런도 같다.
      재현에는 영향 없음(트레이너와 같은 코드) — meta.json I5 에 decay_fp32_noop 로 기록, 방법 서술 확인 필요.
  13. 통계 코드 규칙: 파라미터·기울기·optimizer 상태에서 나온 텐서에 in-place 연산 금지. 무기울기 forward 는 각자 자기 autocast
      구역에서만(학습 forward 와 같은 구역 금지, forward 와 backward 사이 금지) — autocast 의 fp16 가중치 캐시가 grad mode 를 보지 않음.
  16. (10-15, A2 완료 뒤 추가) 손실 설정 요인 조건 (위 표의 refF … A3P·ref2F … ref2P·Aeq). 근거: 교수님 ViT 학습 로그
      (/mnt/sda/choh/shared/W_python/myViT/logs/251124_model3_R4random_enc_B_dec_ismrm_adam041_exp3_18/log.txt — model3, LAMBDA_SSIM 0.2,
      loss = pix + 0.2·(1−SSIM), 전체 영상·마스크 없음; 기록된 loss 가 이 식과 로그 자릿수(≈2e-7)까지 같음)에서 SSIM 항이 손실의 52~58 % 다
      (epoch 별 step 평균의 비, epoch 1~121: pix : 0.2·(1−SSIM) = 0.71~0.91 : 1 — epoch 1 = 0.84, 최저 epoch 8 = 0.71 (SSIM 58.4 %),
      epoch 40 이후 0.83~0.92; epoch 당 1600~2400 step. 10-15 재계산).
      v8fix 레시피에서는 ≈1.5 % 다(pod log.txt 의 l1_over_ssim_term ≈ 64~69). 우리 정규화 단위(영상 / s, s = zero-filled RSS 99 백분위)에서
      ep1 train_l1 ≈ 0.058, 1−SSIM ≈ 0.29 이므로 λ = 0.2 이면 L1 : SSIM 항 ≈ 1.0 : 1 로 교수님 균형에 가깝다(이 스모크 ref·A 의 배치
      3001~4000 마스크 값도 l1 ≈ 0.053, 1−SSIM ≈ 0.28 → 0.95 : 1, SSIM 몫 51 %).
      설계 = P0(마스크 + LAM = 기존 ref·A·A2·A3)에서 요인을 하나씩 바꾼 2×2: F(영역만 전체 영상), L(λ 만 0.2), P(둘 다 = 교수님 ViT 손실 형태).
      L 은 교수님 균형(SSIM 몫)을 맞춘다. P 는 교수님 식·상수 그대로지만 균형까지 같다는 보장은 없다 — 아래 프레임·배경 때문에 SSIM 몫이
      더 클 수 있다(실측 전; loss_compare 가 refP 의 배치 3001~4000 몫을 사전 고정 범위 45~65 % 와 대조해 표시만 한다, λ 재조정 없음).
      '전체 영상' L1 = 384×384 전 화소 평균(교수님 nn.L1Loss 와 같은 F.l1_loss mean), SSIM = u_choh_SSIM.SSIM 의 mask=None 경로 = 교수님 원래 식
      (Gaussian 11, valid 합성곱, L 은 예측에서 자동 추정, 배치 지도 평균). '마스크' 영역 = 지금 식 그대로.
      프레임 주의(10-15 CPU 점검, val 슬라이스 24 개 표본): 약 절반의 볼륨은 GT 가 384² 로 0 채움되어 정확히 0 인 테두리(최대 64 행·64 열
      = 영상의 30.6 %, 표본 평균 GT == 0 화소 20 %)가 있고, zero-filled 입력은 그 테두리에서도 0 이 아니다(train 은 같은 로더 — 미측정).
      교수님 영상에는 이런 테두리가 없다(그의 AXT2 train 볼륨 2 개 표본에서 최대의 1 % 미만 화소 0 %).
      따라서 F·P 에서는 그 볼륨에서만 테두리를 정확히 0 으로 내는 부수 과제가 생긴다 — '전체 영상' 정의는 사용자 지정(384² 전 화소)대로 두고,
      대신 새 조건의 aux 에 GT 지지 영역(label ≠ 0) 항을 같이 기록해 사후에 나눠 본다.
      구현: preset 에 lam_ssim·loss_region 두 키가 있을 때만 타는 분기(loss_terms). 기존 10 조건(ref … A3)은 두 키가 없어 지금 식을 그대로 실행하므로
      동작·config_hash 가 그대로다(SCRIPT_VERSION 유지). 새 조건은 기준 조건(ref·ref2·A·A2·A3) 사전을 복사해 두 키만 더해 만든다(LOSS_DERIVED)
      — 다른 요인이 섞이지 않는다. A2·A3 계열은 GRU 초기 checksum 을 invariants.json 의 gru_ck_A2·gru_ck_A3 와, ref2 계열은 U-Net 초기
      checksum 을 unet_ck_ref2 와 대조한다(I1 은 ref2 와 같은 '건너뜀 + 대조' — ref2 자신의 분기는 그대로).
      기록의 뜻: steps.jsonl 의 loss = 그 조건 자신의 목적함수(l1 + λ·ssim_loss), l1·ssim_loss = 그 조건 영역의 항, l1_num·m_sum = 계속 마스크 기준.
      새 조건에만 aux = {l1_mask, ssim_loss_mask, l1_full, ssim_loss_full, loss_p0 = l1_mask + LAM·ssim_loss_mask, l1_supp, ssim_loss_supp,
      supp_frac}(무기울기, optimizer step 뒤 학습 출력의 detach 사본으로만; supp = label ≠ 0 의 L1 평균과 u_choh SSIM mask=supp, supp_frac =
      배치 안 지지 영역 비율)를 기록한다 — 설정 사이를 같은 척도로 보는 것은 aux 로만. 실제 λ·영역은 시작 줄·meta.json·summary.json 의 'loss'·I5·
      invariants.json 의 loss_<cond> 에 남긴다. I5 는 기존 검사(LAM == pod λ 등)를 그대로 두고, 새 조건에만 λ·영역이 LOSS_SETTINGS 표와
      (fp32 로) 같은지 더 본다. E2(배치 흐름 = ref)·E5(0 입력 slice 무갱신)·E4·probe·평가(OURS)는 손실과 무관하므로 그대로 적용한다
      (probe 의 sat_v 는 손실과 무관, 마스크 치환 시험은 P0 와 비교 가능하게 그대로 둠).
      Aeq(= A + 명시 키 LAM·mask)의 사전 점검이 meas.log 와 비트 단위로 같아야 새 분기를 믿는다(aux 계산이 학습을 바꾸지 않는지도 함께 확인).
      판정(사전 고정) = smoke_deter_opt_loss_compare.py: GRU 회차(seed 1·2·3)마다 자기 P0 회차를 기준으로 한 같은 배치 손실 비의 변화 Δ_k
      (블록 짝 차 t9 CI), 회차 모드(good·intermediate·bad), 2 회차 이상이 같은 쪽으로 움직여야 'changed' — 단일 회차 결과는 기술만.
      L 의 기준은 P0 회차의 기록 항을 L 목적함수로 다시 잰 값(재학습 없음), F·P 는 같은 척도인 마스크 항 비(aux.l1_mask·ssim_loss_mask)가 공동 1차.
      두 SSIM 규약(OURS·PROF_TEST) 분리 평가 = smoke_deter_opt_dualeval.py.
      이 판 이전 실행본: A2 = results/smoke_deter_opt_2026-10-14/harness_as_run_A2_md5_7da42e3f.py (보존됨),
      A3 = md5 3f184ec0 (A3 종료 뒤 harness_as_run_A3_md5_3f184ec0.py 로 보존 — 이 판은 A3 실행 중에 사본으로 작성).
  17. 기울기 급등 차단 확인 S4 (10-16, 사용자 "그래 이걸로 진행해"): 손실 요인 1단계의 P 세 회차가 모두 815~885 step 에서
      fp16 overflow 뒤 유한하지만 매우 큰 기울기(gn 1.5e3~1.5e6)를 받아 GRU 가 포화된 뒤 열세로 돌아섰다 — 상한 MAX_NORM 1000 은
      이를 거르지 못한다. 바꾼 것은 preset 의 max_norm 키 하나(있으면 clip_grad_norm_ 의 max_norm 과 clipped 판정에 그 값, 없으면
      MAX_NORM — 기존 조건은 키가 없어 동작·config_hash 가 그대로, SCRIPT_VERSION 유지). 새 조건 = 기준 조건 사전 + max_norm=20
      (CLIP_DERIVED): APc·A2Pc·A3Pc(P, GRU seed 1·2·3)·A2c(P0, seed 2). torch 2.3.1 clip_grad_norm_ 은 기울기에 clamp(max_norm/
      (gn+1e-6), max=1) 을 늘 곱하므로 gn < 20 인 step 은 정확히 1.0 을 곱해 비트 그대로다 — ref·ref2·A·A3·refP·ref2P 는 4000 step
      내내 gn < 20(최대 15.8)이라 상한 20 에서도 같은 궤적이고 다시 돌리지 않는다(판정 스크립트가 기록으로 다시 확인). A2 는 2 step
      (722·1602)이 걸려 A2c 로 다시 돌린다. 새 조건의 I1·I2 는 기준 조건과 같고(GRU 초기 checksum 은 invariants.json 의
      gru_ck_A2·gru_ck_A3 와 대조), E3(nan_skip ≤ 10·amp_floor ≤ 5)도 그대로, E3_clip 은 새 상한 기준으로 센다(flag 전용).
      Ac(= A + max_norm 20)는 사전 점검 전용(meas.log 와 비트 비교). 판정(실행 전 고정) = smoke_deter_opt_clip_compare.py.
      실행 = A2Pc·A3Pc·APc·A2c. AP 는 첫 유한 gn > 20 이 step 1591(그 전 885~1522 의 고립 overflow 건너뜀 14 회는 해가 없었음 —
      포화는 1591·1652 의 유한 급등 뒤 시작)이라 APc 는 1591 까지 AP 와 비트 같고 S3 의 E3(amp_floor ≤ 5)는 그 전에 이미 넘는다 —
      그래서 판정 스크립트는 새 조건에 E3 대신 s* 뒤만 보는 E3'(건너뜀 비율 ≤ 1 %, 연속 ≤ 20)를 쓴다(검토 지적, 실행 전 고정).
      이 판 이전 실행본(손실 요인 1단계 refP~ref2P) = harness_as_run_P_md5_fc7e5950.py (보존됨).

pod 대응 런 (meta.json pod_counterpart): ref ↔ pod2 PureETER_UNET_noDC_R4_brain384_v8fix_s1_fullval = main_train_pure_v8fix.py
(AMP_GRAD_RETRY·AMP_MIN_SCALE 없음 — report 가 amp_floor=0 이고 grad_retry 가 모두 amp_skip step 에서 회복 0 일 때만 같은 step 의미로 본다),
A ↔ pod2 PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval = ampguard 트레이너. pod 와는 train 목록(65,060 vs 로컬 65,028)·순열이 달라
배치가 다르므로 통계적 비교만 가능하다.

종료 코드: 0 정상 / 2 거부(GPU0 사용 중·정책) / 3 job 실패(예외·OOM) / 4 CUDA 사용 불가 / 5 불변식 위반.
사용:
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python v8_eter_pure/smoke_deter_opt.py --cond A \
      --out-dir results/smoke_deter_opt_2026-10-14 --skip-done
  사전 점검 (scratch 출력 — 실제 출력 폴더는 거부): --preflight --cond E --steps 30 --out-dir <scratch>   (설계 실측 로그와 비트 비교)
  CPU 단위 시험: CUDA_VISIBLE_DEVICES="" python v8_eter_pure/smoke_deter_opt.py --selftest [--selftest-model]
  손실 요인 조건 (설계 결정 16): --cond refP 등 (런처 CONDS 로 지정). 새 분기 비트 점검: --preflight --cond Aeq --out-dir <scratch>
"""

import os
import sys
import csv
import json
import math
import time
import zlib
import random
import shutil
import hashlib
import argparse
import datetime
import warnings
import traceback
import subprocess
import importlib.util
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ('v8_eter_pure', 'v8_eter_pure/configs', 'dataloaders', 'models/pure_eternet', 'models/hybrid_eternet',
           'models/mamba_eternet'):
    sys.path.insert(0, os.path.join(REPO, _p))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import smoke_recipe_fix as R                                    # 레시피 단일 출처 (읽기 전용 import)
import eval_paired_v8_nodc as _EV
from eval_paired_v8_nodc import C, slice_metrics
import dataloader_h5_v5 as _DL5
from dataloader_h5_v5 import FastMRI_H5_Dataloader
import u_choh_SSIM as _SSIM_MOD
from u_choh_SSIM import SSIM
import myUNet_DF as _UNET_MOD
import u_pure_eternet_gru as _GRU_MOD
import u_pure_eternet_unet as _UNO_MOD

# pod 전용 파일 — 디렉터리를 sys.path 에 넣지 않는다(저장소 모듈 사본이 같이 있음)
_GP_PATH = os.path.join(REPO, 'results', 'setup_audit_2026-10-14', 'pod2_src', 'v8_eter_pure', 'gru_precision.py')
_gp_spec = importlib.util.spec_from_file_location('gru_precision_pod2', _GP_PATH)
gru_precision = importlib.util.module_from_spec(_gp_spec)
_dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True     # results/ 아래에 .pyc 를 쓰지 않음
_gp_spec.loader.exec_module(gru_precision)
sys.dont_write_bytecode = _dwb

SCRIPT_VERSION = 'deter-opt-2026-10-14a'
OUT_DEFAULT = os.path.join(REPO, 'results', 'smoke_deter_opt_2026-10-14')
PILOT_DIR = os.path.join(OUT_DEFAULT, 'design_pilot')
POD_RECIPES = [os.path.join(REPO, 'results', 'setup_audit_2026-10-14', 'eval_evidence', 'logs', d, 'recipe.json')
               for d in ('pod2_gru', 'pod2_unet')]

# ── 고정값 (spec §2) ──
SEED, BS, WORKERS, STEPS = 1, 8, 16, 4000
LAM = 0.0030745725                    # pod2 recipe.json λ_SSIM (s̄ 325.25) — fp32 로 pod 전체 값과 같음 (I5)
LAM_PROF = 0.2                        # 교수님 ViT 학습 LAMBDA_SSIM_PER_PIXEL (설계 결정 16)
LOSS_REGIONS = ('mask', 'full')
# 손실 설정 요인 (설계 결정 16): 이름 → (λ_SSIM, 영역). '0' = P0 = 기존 조건의 손실 (Aeq 사전 점검만 명시 키로 씀)
LOSS_SETTINGS = OrderedDict([('0', (LAM, 'mask')), ('F', (LAM, 'full')), ('L', (LAM_PROF, 'mask')), ('P', (LAM_PROF, 'full'))])
T_MAX = 406650                        # pod 50 epoch opt-step (8133 × 50)
LR0, ETA_MIN = 2e-4, 1e-6
WD = 3e-5
EPS = 1e-8
BETAS = (0.9, 0.999)
MAX_NORM = 1e3
AMP_MIN_SCALE = 1024.0
AMP_GRAD_RETRY = 1
TRAIN_NAN_RETRY = 1
D_C = 0.06
FP16_TINY = 6.103515625e-05           # fp16 최소 정규수 (이보다 작으면 subnormal)
OVERLAP = {'file_brain_AXT2_200_2000306.h5', 'file_brain_AXT2_200_2000308.h5',
           'file_brain_AXT2_200_2000310.h5', 'file_brain_AXT2_200_2000311.h5'}   # 로컬 train 에도 있는 val 4 볼륨
L_FULL = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
L_LIGHT = sorted(set(L_FULL) | {750, 1250, 1750, 2250, 2750, 3250, 3750})
SNAP_STEP = 3500
CK_STEPS = (1, 2, 3)
R_WINDOWS = [(1, 100), (101, 500), (501, 1000), (1001, 2000), (2001, 3000), (3001, 4000)]

EXPECT = dict(unet_ck='5.9212156172e+02', gru_ck='-2.7356114523e+03', label_ck='7.673825e+07', data_ck='6.488181e+05',
              n_train=65028, n_val=7334, n_loader=8129, gru_tensors=16, no_decay_gru_model=99009)
MD5 = OrderedDict([
    ('v8_eter_pure/smoke_recipe_fix.py', '15fc853f02ddc05ffa2f8ab05fcec464'),
    ('results/setup_audit_2026-10-14/pod2_src/v8_eter_pure/gru_precision.py', '79a6905ed33bab6c131f04f5a51ffa17'),
    ('dataloaders/dataloader_h5_v5.py', 'f6cd0dc1a0e41d696d384aebed675ede'),
    ('models/pure_eternet/u_pure_eternet_gru.py', '3c5c9cae5b37cd254ca0de0ccb540866'),
    ('models/pure_eternet/u_pure_eternet_unet.py', 'e274872fc2c82d4ab3627cfe291816b1'),
    ('models/hybrid_eternet/myUNet_DF.py', 'e77a57332d5818b0f1b16f4490e0c022'),
    ('models/hybrid_eternet/u_choh_SSIM.py', '0a996f24625546ee2fc51c1cf0544e15'),
    ('v8_eter_pure/configs/myConfig_pure_eter_v8.py', '4b7e103ca767b586baaa7cb01fed7fed'),
    ('v8_eter_pure/eval_paired_v8_nodc.py', '795ea515e784df15af63d5bf1d76e68f'),
])
# 실제로 import 된 모듈이 위 파일인지 (sys.path 순서로 다른 사본을 집지 않았는지)
_MOD_FILES = [(R, 'v8_eter_pure/smoke_recipe_fix.py'),
              (gru_precision, 'results/setup_audit_2026-10-14/pod2_src/v8_eter_pure/gru_precision.py'),
              (_DL5, 'dataloaders/dataloader_h5_v5.py'), (_GRU_MOD, 'models/pure_eternet/u_pure_eternet_gru.py'),
              (_UNO_MOD, 'models/pure_eternet/u_pure_eternet_unet.py'), (_UNET_MOD, 'models/hybrid_eternet/myUNet_DF.py'),
              (_SSIM_MOD, 'models/hybrid_eternet/u_choh_SSIM.py'), (C, 'v8_eter_pure/configs/myConfig_pure_eter_v8.py'),
              (_EV, 'v8_eter_pure/eval_paired_v8_nodc.py')]

CONDS = OrderedDict([
    ('ref',  dict(model='unet', init_seed=1, seq=None)),
    ('ref2', dict(model='unet', init_seed=2, seq=None)),
    ('A',  dict(model='gru', init_seed=1, seq='raw',   m_gru=1.0, W_gru=0,    eps_gru=EPS)),
    ('B',  dict(model='gru', init_seed=1, seq='raw',   m_gru=0.1, W_gru=0,    eps_gru=EPS)),
    ('C',  dict(model='gru', init_seed=1, seq='raw',   m_gru=1.0, W_gru=1000, eps_gru=EPS)),
    ('D',  dict(model='gru', init_seed=1, seq='scale', c=D_C, m_gru=1.0, W_gru=0, eps_gru=D_C * 1e-8)),
    ('E',  dict(model='gru', init_seed=1, seq='gate',  m_gru=1.0, W_gru=0,    eps_gru=EPS, m_gate=1.0)),
    ('EC', dict(model='gru', init_seed=1, seq='gate',  m_gru=1.0, W_gru=1000, eps_gru=EPS, m_gate=1.0)),
    ('A2', dict(model='gru', init_seed=1, gru_seed=2, seq='raw', m_gru=1.0, W_gru=0, eps_gru=EPS)),   # 설계 결정 15
    ('A3', dict(model='gru', init_seed=1, gru_seed=3, seq='raw', m_gru=1.0, W_gru=0, eps_gru=EPS)),   # 설계 결정 15 (seed 3)
])
# 설계 결정 16 — 손실 설정 요인 조건: 기준 조건 사전을 복사해 lam_ssim·loss_region 두 키만 더한다 (기존 조건에는 두 키가 없음)
LOSS_BASE = OrderedDict([('ref', 'ref'), ('ref2', 'ref2'), ('A', 'A'), ('A2', 'A2'), ('A3', 'A3')])   # 접두 → 기준 조건
LOSS_DERIVED = OrderedDict()                                                     # 새 조건 → (기준 조건, 설정)
for _x in ('F', 'L', 'P'):
    for _pre, _base in LOSS_BASE.items():
        CONDS[_pre + _x] = dict(CONDS[_base], lam_ssim=LOSS_SETTINGS[_x][0], loss_region=LOSS_SETTINGS[_x][1])
        LOSS_DERIVED[_pre + _x] = (_base, _x)
CONDS['Aeq'] = dict(CONDS['A'], lam_ssim=LOSS_SETTINGS['0'][0], loss_region=LOSS_SETTINGS['0'][1])   # 사전 점검 전용
LOSS_DERIVED['Aeq'] = ('A', '0')
# 설계 결정 17 — 기울기 급등 차단(S4): 기준 조건 사전 + max_norm 키 하나 (기존 조건에는 키가 없음 → MAX_NORM)
MAX_NORM_S4 = 20.0
CLIP_DERIVED = OrderedDict()                                                     # 새 조건 → 기준 조건
for _base in ('AP', 'A2P', 'A3P', 'A2', 'A'):
    CONDS[_base + 'c'] = dict(CONDS[_base], max_norm=MAX_NORM_S4)
    CLIP_DERIVED[_base + 'c'] = _base
    if _base in LOSS_DERIVED:
        LOSS_DERIVED[_base + 'c'] = LOSS_DERIVED[_base]                           # 손실 설정·초기값 대조는 기준 조건과 같음
del _x, _pre, _base
PREFLIGHT_ONLY = ('Aeq', 'Ac')
PILOT_LOGS = {'A': 'meas.log', 'B': 'meas_B.log', 'D': 'meas_D.log', 'E': 'meas_E.log', 'Aeq': 'meas.log', 'Ac': 'meas.log'}
POD_COUNTERPART = dict(
    ref=dict(run='pod2 PureETER_UNET_noDC_R4_brain384_v8fix_s1_fullval',
             trainer='v8_eter_pure/main_train_pure_v8fix.py via infra/runpod/pod_run_v8fix.sh (AMP_GRAD_RETRY·AMP_MIN_SCALE 없음; '
                     'SCRATCH START 줄에 trainer=ampguard 없음)',
             equivalence='amp_floor = 0 이고 모든 grad_retry 가 amp_skip step 에서 recovered = 0 일 때만 같은 step 의미 (report 확인)'),
    A=dict(run='pod2 PureETER_GRU_noDC_R4_brain384_v8fix_s1_fullval',
           trainer='main_train_pure_v8fix_ampguard.py (trainer=ampguard amp_grad_retry=1 amp_min_scale=1024 gru_fp32_modules=gru_h)'),
    note='pod 와는 train 목록(pod 65,060 / 로컬 65,028 슬라이스)·순열이 달라 배치가 다르다 — 통계적 비교만 가능')

EXIT_OK, EXIT_REFUSE, EXIT_FAIL, EXIT_NOCUDA, EXIT_INVALID = 0, 2, 3, 4, 5


class InvariantError(Exception):
    pass


# ------------------------------------------------------------------ 공통 도구
def md5_file(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def _jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def jdumps(obj, **kw):
    return json.dumps(obj, default=_jdefault, **kw)


def write_json_atomic(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(jdumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def now_iso():
    return datetime.datetime.now().isoformat(timespec='seconds')


def base_lr(k):
    """닫힌 형식 cosine (설계 실측 meas*.py 의 cos_lr 과 같은 식·같은 연산 순서 → 같은 비트)."""
    return ETA_MIN + (LR0 - ETA_MIN) * (1 + math.cos(math.pi * k / T_MAX)) / 2


def group_lr(k, m, W):
    w = min(1.0, (k + 1) / W) if W > 0 else 1.0
    return base_lr(k) * m * w


def lr_recursion_maxrel(n_steps):
    """I5: 닫힌 형식 == torch CosineAnnealingLR 점화식 (k < n_steps 에서 최대 상대차)."""
    p = torch.zeros(1, requires_grad=True)
    o = torch.optim.SGD([p], lr=LR0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T_MAX, eta_min=ETA_MIN)
    mx = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for k in range(n_steps):
            a, b = o.param_groups[0]['lr'], base_lr(k)
            mx = max(mx, abs(a - b) / b)
            o.step()
            sch.step()
    return mx


def decay_fp32_noop():
    """분리형 감쇠 w·(1−lr·wd) 가 fp32 가중치를 바꾸는지 — 스칼라 fp32 반올림과 실제 AdamW(기울기 0, CPU) 둘 다 확인."""
    lrs = [LR0, base_lr(3999), 0.1 * LR0, ETA_MIN]
    scal = {repr(lr): bool(np.float32(1 - lr * WD) == np.float32(1.0)) for lr in lrs}
    moved = {}
    for fe in (False, True):
        g = torch.Generator()
        g.manual_seed(0)
        p = nn.Parameter(torch.randn(10000, generator=g) * 0.05)
        w0 = p.detach().clone()
        o = torch.optim.AdamW([p], lr=LR0, betas=BETAS, eps=EPS, weight_decay=WD, foreach=fe)
        for _ in range(20):
            p.grad = torch.zeros_like(p)
            o.step()
        moved[f'foreach={fe}'] = float((p.detach() - w0).abs().max())
    return dict(noop=all(scal.values()) and all(v == 0.0 for v in moved.values()), fp32_scalar_is_one=scal,
                adamw_zero_grad_maxabs_change=moved, lr_wd_max=LR0 * WD)


def preset_groups(P, has_gate):
    """(tag, weight_decay, m, W, eps) — spec §2.4 순서."""
    g = [('unet_decay', WD, 1.0, 0, EPS)]
    if P['model'] == 'gru':
        g.append(('gru_decay', WD, P['m_gru'], P['W_gru'], P['eps_gru']))
    g.append(('unet_nodecay', 0.0, 1.0, 0, EPS))
    if P['model'] == 'gru':
        g.append(('gru_nodecay', 0.0, P['m_gru'], P['W_gru'], P['eps_gru']))
    if has_gate:
        g.append(('gate', 0.0, P['m_gate'], 0, EPS))
    return g


def lr_table(P, has_gate, ks=(0, 99, 999, 3999)):
    return [dict(tag=t, wd=wd, m=m, W=W, eps=e, lr={str(k): group_lr(k, m, W) for k in ks})
            for t, wd, m, W, e in preset_groups(P, has_gate)]


def sign_test_p(n_pos, n_neg):
    return R.sign_test_p(n_pos, n_neg)


def drop_stats(base, test):
    st = R.drop_stats(base, test)
    return None if st is None else dict(st)


def even_sel(n_total, n):
    return R.even_indices(n_total, n)


def chan_rms(t):
    return t.float().pow(2).mean(dim=(-2, -1)).sqrt()


def tfloat(t):
    return float(t.detach().float()) if isinstance(t, torch.Tensor) else float(t)


def _rms(t):
    return float(t.detach().float().pow(2).mean().sqrt())


def metrics(o, ref, bm):
    """eval_paired_v8_nodc.slice_metrics (원 단위) — composite 는 버린다."""
    m = slice_metrics(o, ref, bm)
    m.pop('composite', None)
    return m


def loss_terms(crit, out_fp, lab, bm, region):
    """설계 결정 16 의 손실 항 (L1, 1−SSIM). 'mask' = 기존 train_forward 와 같은 식·같은 연산 순서(뇌 마스크 L1, u_choh SSIM mask=bm),
    'full' = 384×384 전 화소 L1 평균(교수님 nn.L1Loss = F.l1_loss mean) + u_choh SSIM mask=None(교수님 원래 식: 배치 지도 평균)."""
    if region == 'mask':
        m_sum = bm.sum().clamp(min=1.0)
        l1 = ((out_fp - lab).abs() * bm).sum() / m_sum
        lssim = 1 - crit(out_fp, lab, mask=bm)
    elif region == 'full':
        l1 = F.l1_loss(out_fp, lab)
        lssim = 1 - crit(out_fp, lab)
    else:
        raise ValueError(f'loss_region {region!r} (허용 {LOSS_REGIONS})')
    return l1, lssim


def supp_terms(crit, out_fp, lab):
    """설계 결정 16 — 기록 전용(aux, 학습에 쓰지 않음): GT 지지 영역 supp = (label ≠ 0) 의 L1 평균·u_choh SSIM(mask=supp)·지지 비율.
    0 채움 테두리를 뺀 '전체 영상' — 교수님 영상(테두리 없음)의 전체 영상에 가장 가까운 영역."""
    supp = (lab != 0).float()
    s_sum = supp.sum().clamp(min=1.0)
    l1 = ((out_fp - lab).abs() * supp).sum() / s_sum
    lssim = 1 - crit(out_fp, lab, mask=supp)
    return l1, lssim, supp.mean()


def param_key(name):
    if name.startswith('gru_h.'):
        return 'gru_h'
    if name.startswith('gru_v.'):
        return 'gru_v'
    if name.startswith('unet.'):
        return 'unet'
    if name == 'seq_gate':
        return 'gate'
    return 'other'


@torch.no_grad()
def unet_checksum(net):
    """spec §2.1-5 / 트레이너 SCRATCH START 와 같은 식."""
    return float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(net.unet.parameters())))


@torch.no_grad()
def gru_checksum(base):
    """Σ (i+1)·Σp (GRU 파라미터, float64) — 큰 텐서는 32M 원소씩 나눠 합(전체 fp64 사본 1.1 GB 회피, %.10e 에서 같은 값)."""
    tot = 0.0
    params = [p for n, p in base.named_parameters() if n.startswith(('gru_h.', 'gru_v.'))]
    for i, p in enumerate(params):
        acc = None
        for ch in p.detach().reshape(-1).split(R.STAT_CHUNK):
            v = (ch.double() * (i + 1)).sum()
            acc = v if acc is None else acc + v
        tot = tot + acc
    return float(tot)


def seq_parts(base, x):
    """PureETER_GRU._seq 와 같은 연산 + 중간 출력 (out_h, gru_v 입력 oh, out_v, so)."""
    B = x.size(0)
    out_h, _ = base.gru_h(x.reshape([B, base.num_in_x, -1]))
    oh = out_h.reshape([B, base.num_in_x, base.num_out_y, -1]).permute(0, 2, 1, 3).reshape([B, base.num_out_y, -1])
    out_v, _ = base.gru_v(oh)
    so = out_v.reshape([B, base.num_out_y, base.num_out_x, -1]).permute(0, 3, 2, 1)
    return out_h, oh, out_v, so


@torch.no_grad()
def gru_gate_derivs(gru, X, out, H, mimic_half):
    """GRU 게이트 미분 재계산 (fp32, 출력 시퀀스에서): dz = mean z(1−z), dn = mean(1−n²), zsat = mean(z>0.99), nsat = mean(|n|>0.99),
    recon_err = max|h_t − ((1−z)n + z·h_prev)|. 정방향 h_prev = h_{t−1}(h_{−1}=0), 역방향 h_prev = h_{t+1}. 게이트 순서 r|z|n."""
    q = (lambda t: t.detach().half().float()) if mimic_half else (lambda t: t.detach().float())
    res = {}
    with torch.autocast('cuda', enabled=False):
        Xq = q(X)
        for d, suf, sl in (('f', '', slice(0, H)), ('b', '_reverse', slice(H, 2 * H))):
            Wih, Whh = q(getattr(gru, 'weight_ih_l0' + suf)), q(getattr(gru, 'weight_hh_l0' + suf))
            bih, bhh = q(getattr(gru, 'bias_ih_l0' + suf)), q(getattr(gru, 'bias_hh_l0' + suf))
            h = out[..., sl].detach().float()
            z0 = torch.zeros_like(h[:, :1])
            hprev = torch.cat((z0, h[:, :-1]), 1) if d == 'f' else torch.cat((h[:, 1:], z0), 1)
            GI = F.linear(Xq, Wih, bih)
            GH = F.linear(hprev, Whh, bhh)
            r = torch.sigmoid(GI[..., :H] + GH[..., :H])
            z = torch.sigmoid(GI[..., H:2 * H] + GH[..., H:2 * H])
            n = torch.tanh(GI[..., 2 * H:] + r * GH[..., 2 * H:])
            hrec = (1 - z) * n + z * hprev
            res[d] = dict(dz=float((z * (1 - z)).mean()), dn=float((1 - n * n).mean()),
                          zsat=float((z > 0.99).float().mean()), nsat=float((n.abs() > 0.99).float().mean()),
                          recon_err=float((h - hrec).abs().max()))
            del GI, GH, r, z, n, hrec, hprev
    return res


@torch.no_grad()
def seq_stats(S, f0=None, partners=None, keys=None):
    """S: (n,20,H,W) fp32 — 공통 패턴 비율·W 상수성·입력 의존도·인접 슬라이스 차 (audit 정의, 설계 실측 probe 와 같은 식)."""
    n = S.shape[0]
    Fm = S.mean(0, keepdim=True)
    D = S - Fm
    e2 = lambda t: float(t.double().pow(2).sum())              # noqa: E731
    es = e2(S)
    ed = e2(D)
    common = n * e2(Fm) / max(es, 1e-300)
    r = dict(seq_rms=float(S.pow(2).mean().sqrt()), common_frac=common,
             common_frac_unbiased=1 - (1 - common) * n / (n - 1) if n > 1 else None,
             D_rms=float(D.pow(2).mean().sqrt()),
             wconst_D=D.shape[-1] * e2(D.mean(-1, keepdim=True)) / max(ed, 1e-300))
    ch = chan_rms(S)
    r.update(seq_ch_rms_med=float(ch.median()), seq_ch_rms_min=float(ch.min()), seq_ch_rms_max=float(ch.max()))
    if f0 is not None:
        f0n = float(f0.norm())
        r['dep'] = float(np.mean([float((S[q] - f0).norm()) / max(f0n, 1e-30) for q in range(n)]))
    if partners is not None:
        vals = [float((S[q] - partners[q]).norm()) / max(float(D[q].norm()), 1e-30) for q in range(n) if partners[q] is not None]
        r['adj_rel'] = float(np.mean(vals)) if vals else None
    return r, Fm, D


class _IdxDS(FastMRI_H5_Dataloader):
    """저장소 로더와 같은 __getitem__ + 'idx' (하위 클래스 — R._worker_init_fn 이 이 객체에 .rng 를 건다)."""

    def __getitem__(self, i):
        d = super().__getitem__(i)
        d['idx'] = i
        return d


class SubTracker:
    """파라미터별 고정 부분 표본 (원소 수 > R.SUB_N 이면 R.SUB_N 개 복원 추출, generator = SUB_SEED + crc32(이름) — 조건 간 같은 원소).
    초기값·직전값은 CPU. rel_change_from_init / rel_change_interval 을 키(gru_h·gru_v·unet)별로."""

    def __init__(self, named):
        self.items = []
        for name, p in named:
            N = p.numel()
            if N > R.SUB_N:
                g = torch.Generator()
                g.manual_seed(R.SUB_SEED + zlib.crc32(name.encode()))
                idx, scale = torch.randint(0, N, (R.SUB_N,), generator=g), N / R.SUB_N
            else:
                idx, scale = None, 1.0
            w0 = self._take(p, idx)
            self.items.append(dict(name=name, key=param_key(name), p=p, idx=idx, scale=scale, w0=w0, prev=w0))

    @staticmethod
    def _take(p, idx):
        a = p.detach().reshape(-1)
        if idx is not None:
            a = a[idx.to(a.device)]
        return a.float().cpu().clone()

    def idx_of(self, name):
        for it in self.items:
            if it['name'] == name:
                return it['idx']
        raise KeyError(name)

    @torch.no_grad()
    def update(self):
        acc = {}
        for it in self.items:
            cur = self._take(it['p'], it['idx'])
            a = acc.setdefault(it['key'], [0.0, 0.0, 0.0, 0.0])
            a[0] += float((cur - it['w0']).pow(2).sum()) * it['scale']
            a[1] += float(it['w0'].pow(2).sum()) * it['scale']
            a[2] += float((cur - it['prev']).pow(2).sum()) * it['scale']
            a[3] += float(it['prev'].pow(2).sum()) * it['scale']
            it['prev'] = cur
        return {k: dict(rel_change_from_init=math.sqrt(a[0] / a[1]) if a[1] > 0 else None,
                        rel_change_interval=math.sqrt(a[2] / a[3]) if a[3] > 0 else None) for k, a in acc.items()}


@torch.no_grad()
def near_dead_fracs(net, groups):
    per = {}
    for n, p in net.named_parameters():
        cnt = 0
        for ch in p.detach().reshape(-1).split(R.STAT_CHUNK):
            cnt += int((ch.abs() < R.NEAR_THR).sum())
        per[n] = (cnt, p.numel())
    out = {}
    for g, names in groups.items():
        out[g] = sum(per[n][0] for n in names) / max(sum(per[n][1] for n in names), 1)
    return out


@torch.no_grad()
def sat_probe(base, x_in, so):
    """gf_train.sat_probe 와 같은 정의 — 현재 학습 배치의 gru_h·gru_v 입력항 사전활성 포화와 출력 포화 (gru_h fp32)."""
    B = x_in.size(0)
    out = {}
    with torch.autocast('cuda', enabled=False):
        in_h = x_in.reshape([B, base.num_in_x, -1]).float()
        H = base.num_out1
        for d, suf in (('f', ''), ('b', '_reverse')):
            pre = F.linear(in_h, getattr(base.gru_h, 'weight_ih_l0' + suf), getattr(base.gru_h, 'bias_ih_l0' + suf))
            out[f'h_pre_std_{d}'] = float(pre.std())
            out[f'h_pre_sat4_rz_{d}'] = float((pre[..., :2 * H].abs() > 4).float().mean())
            out[f'h_pre_sat2_n_{d}'] = float((pre[..., 2 * H:].abs() > 2).float().mean())
            del pre
        out_h, _ = base.gru_h(in_h)
        out['h_out_rms'] = float(out_h.pow(2).mean().sqrt())
        out['h_out_sat99'] = float((out_h.abs() > 0.99).float().mean())
        oh = out_h.reshape([B, base.num_in_x, base.num_out_y, -1]).permute(0, 2, 1, 3).reshape([B, base.num_out_y, -1])
        Hv = base.num_out2
        pre = F.linear(oh, base.gru_v.weight_ih_l0, base.gru_v.bias_ih_l0)
        out['v_pre_std_f'] = float(pre.std())
        out['v_pre_sat4_rz_f'] = float((pre[..., :2 * Hv].abs() > 4).float().mean())
        out['v_pre_sat2_n_f'] = float((pre[..., 2 * Hv:].abs() > 2).float().mean())
        del pre, out_h, oh
    out['v_out_sat99'] = float((so.detach().float().abs() > 0.99).float().mean())
    return out


def stream_md5(lines, steps):
    h = hashlib.md5()
    for ln in lines:
        if ln['bpos'] < steps:
            h.update(f"{ln['bpos']}|{','.join(str(i) for i in ln['idx'])}|{ln['data_ck']!r}\n".encode())
    return h.hexdigest()


def parse_pilot_log(path):
    """설계 실측 로그 'STEP k {json}' → {k: dict(loss, gn, amp_scale, seq_rms, gate_absmean)}."""
    out = {}
    with open(path) as f:
        for ln in f:
            if not ln.startswith('STEP '):
                continue
            _, k, js = ln.split(' ', 2)
            d = json.loads(js)
            g = d.get('gate')
            out[int(k)] = dict(loss=d['loss'], gn=d['gn'], amp_scale=d['scale'], seq_rms=d.get('seq_rms'),
                               gate_absmean=None if not g else g.get('mean_abs'))
    return out


def parse_steps_jsonl(path):
    out = {}
    with open(path) as f:
        for ln in f:
            d = json.loads(ln)
            if d.get('skipped'):
                continue
            out[int(d['step'])] = dict(loss=d['loss'], gn=d['gn'], gn_model=d.get('gn_model'), amp_scale=d['amp_scale'],
                                       seq_rms=d.get('seq_rms'), gate_absmean=d.get('gate_absmean'))
    return out


def compare_runs(mine_path, other_path, use_gn_model):
    """step 별 loss·grad norm·amp_scale (+ seq_rms·|α| 평균) 비트 비교. 반환 (all_equal, rows)."""
    mine = parse_steps_jsonl(mine_path)
    other = parse_pilot_log(other_path) if not other_path.endswith('.jsonl') else parse_steps_jsonl(other_path)
    same = lambda a, b: (a == b) or (a is not None and b is not None and isinstance(a, float) and isinstance(b, float)
                                     and math.isnan(a) and math.isnan(b))   # noqa: E731
    rows, ok = [], True
    for k in sorted(set(mine) & set(other)):
        a, b = mine[k], other[k]
        gn_a = a['gn_model'] if (use_gn_model and a.get('gn_model') is not None) else a['gn']
        gn_b = b['gn_model'] if (use_gn_model and b.get('gn_model') is not None) else b['gn']
        r = dict(step=k, loss=same(a['loss'], b['loss']), gn=same(gn_a, gn_b), amp_scale=same(a['amp_scale'], b['amp_scale']))
        for opt in ('seq_rms', 'gate_absmean'):
            if a.get(opt) is not None and b.get(opt) is not None:
                r[opt] = same(a[opt], b[opt])
        r['vals'] = dict(loss=(a['loss'], b['loss']), gn=(gn_a, gn_b), amp_scale=(a['amp_scale'], b['amp_scale']))
        ok = ok and r['loss'] and r['gn'] and r['amp_scale'] and r.get('seq_rms', True) and r.get('gate_absmean', True)
        rows.append(r)
    return ok and bool(rows), rows


# ------------------------------------------------------------------ 본 job
class Job:
    def __init__(self, args):
        self.args = args
        self.cond = args.cond
        self.P = CONDS[args.cond]
        self.has_seq = self.P['model'] == 'gru'
        self.has_gate = self.P.get('seq') == 'gate'
        # 설계 결정 16: 두 키가 있을 때만 새 손실 분기 (기존 조건은 loss_custom False → 기존 식 그대로)
        self.loss_custom = 'lam_ssim' in self.P or 'loss_region' in self.P
        if self.loss_custom:
            assert 'lam_ssim' in self.P and 'loss_region' in self.P, f'{self.cond}: lam_ssim·loss_region 은 둘 다 있어야 함'
            assert self.P['loss_region'] in LOSS_REGIONS, self.P['loss_region']
        self.lam = self.P['lam_ssim'] if self.loss_custom else LAM
        self.loss_region = self.P['loss_region'] if self.loss_custom else 'mask'
        # 설계 결정 17: max_norm 키가 있을 때만 다른 상한 (기존 조건은 MAX_NORM 그대로)
        self.max_norm = float(self.P['max_norm']) if 'max_norm' in self.P else MAX_NORM
        assert math.isfinite(self.max_norm) and self.max_norm > 0, self.max_norm
        self.debug = bool(args.preflight or args.steps is not None or args.eval_n is not None or args.probe_n is not None)
        self.steps = args.steps if args.steps is not None else (30 if args.preflight else STEPS)
        self.eval_n = args.eval_n if args.eval_n is not None else (16 if args.preflight else 256)
        self.probe_n = args.probe_n if args.probe_n is not None else (8 if args.preflight else 32)
        self.out_root = os.path.abspath(args.out_dir)
        self.cdir = os.path.join(self.out_root, self.cond)
        self.l_full = sorted({k for k in L_FULL if k <= self.steps} | {self.steps})
        self.l_light = sorted({k for k in L_LIGHT if k <= self.steps} | {self.steps})
        self.snap_step = SNAP_STEP if self.steps == STEPS else (self.steps * 7) // 8
        self.inv = OrderedDict()
        self.flags = []
        self.t_start = time.time()
        self.summary_written = False

    def loss_desc(self):
        """설계 결정 16 — 이 조건이 실제로 쓰는 손실 설정 (시작 줄·meta·summary 기록용)."""
        name = next((k for k, v in LOSS_SETTINGS.items() if v == (self.lam, self.loss_region)), None)
        terms = (dict(l1='sum(|out-lab|*bm)/sum(bm)', ssim='u_choh_SSIM.SSIM(out, lab, mask=bm)') if self.loss_region == 'mask'
                 else dict(l1='F.l1_loss(out, lab) = mean over B x 384 x 384', ssim='u_choh_SSIM.SSIM(out, lab) (mask=None, map mean)'))
        return OrderedDict(lam_ssim=self.lam, loss_region=self.loss_region, setting=name,
                           objective='loss = l1 + lam_ssim * (1 - ssim)', terms=terms,
                           step_record='steps.jsonl loss/l1/ssim_loss = this condition objective & region; l1_num/m_sum = masked; '
                                       'aux = both regions + loss_p0 + GT-support (label != 0) terms and supp_frac (no grad)')

    # ── 불변식 기록 ──
    def check(self, name, ok, **info):
        self.inv[name] = dict(ok=bool(ok), **info)
        tag = 'OK ' if ok else 'FAIL'
        print(f'[{self.cond}] {tag} {name} {jdumps(info)[:600]}', flush=True)
        if not ok:
            raise InvariantError(f'{name}: {info}')

    def shared_invariants(self, upd):
        """invariants.json — 처음 쓴 조건이 값을 정하고 이후 조건은 대조한다 (원자적 쓰기)."""
        path = os.path.join(self.out_root, 'invariants.json')
        cur = {}
        if os.path.exists(path):
            with open(path) as f:
                cur = json.load(f)
        bad = []
        new = dict(cur)
        for k, v in upd.items():
            vj = json.loads(jdumps(v))
            if k in cur:
                if cur[k] != vj:
                    bad.append(k)
            else:
                new[k] = vj
        if bad:
            raise InvariantError('invariants.json 불일치: ' + ', '.join(f'{k}: {str(cur[k])[:120]} != {str(upd[k])[:120]}' for k in bad))
        if new != cur:
            write_json_atomic(new, path)

    # ── 1~3. 환경·가드 ──
    def startup_guards(self):
        R.gpu0_precheck(force=False)                     # exit 2 (사용 중) / 4 (CUDA 불가)
        try:
            torch.zeros(1, device='cuda')
        except RuntimeError as e:
            print(f'[refuse] CUDA 초기화 실패 (NVML/EPERM 가능): {e}', flush=True)
            raise SystemExit(EXIT_NOCUDA)
        self.dev = torch.device('cuda')
        torch.backends.cuda.matmul.allow_tf32 = False     # 트레이너 :84-87 과 같음
        torch.backends.cudnn.allow_tf32 = False
        self.md5 = OrderedDict()
        bad = []
        for rel, want in MD5.items():
            got = md5_file(os.path.join(REPO, rel))
            self.md5[rel] = got
            if got != want:
                bad.append(f'{rel}: {got} != {want}')
        for mod, rel in _MOD_FILES:
            if os.path.realpath(mod.__file__) != os.path.realpath(os.path.join(REPO, rel)):
                bad.append(f'import 경로 {mod.__name__}: {mod.__file__} != {rel}')
        if bad:
            print('[invalid] md5/경로 가드 실패:\n  ' + '\n  '.join(bad), flush=True)
            raise SystemExit(EXIT_INVALID)
        self.env = OrderedDict(torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                               gpu=torch.cuda.get_device_name(0), alloc_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), python=sys.version.split()[0],
                               harness_md5=md5_file(os.path.abspath(__file__)), host=os.uname().nodename)

    # ── val 슬라이스 (모든 조건 같음) ──
    def build_val(self):
        self.val_ds = R.build_val_dataset()
        vs = self.val_ds.samples
        ok = [i for i, (fp, s, _) in enumerate(vs) if os.path.basename(fp) not in OVERLAP]
        self.E = [ok[k] for k in even_sel(len(ok), self.eval_n)]
        self.Pp = [self.E[k] for k in even_sel(len(self.E), self.probe_n)]
        same_vol = lambda a, b: 0 <= a < len(vs) and 0 <= b < len(vs) and vs[a][0] == vs[b][0]   # noqa: E731
        self.E_nxt = [R.adjacent_partner(self.val_ds, i) for i in self.E]
        self.E_prv = [(i - 1) if (j == i + 1 and same_vol(i - 1, i)) else None for i, j in zip(self.E, self.E_nxt)]
        n = len(self.E)
        self.E_cross = [self.E[(k + n // 2) % n] for k in range(n)]
        self.P_adj = [R.adjacent_partner(self.val_ds, i) for i in self.Pp]
        npb = len(self.Pp)
        self.P_cross = [self.Pp[(k + npb // 2) % npb] for k in range(npb)]

    def config_hash(self):
        obj = dict(script_version=SCRIPT_VERSION, cond=self.cond, preset=self.P, steps=self.steps, tmax=T_MAX, seed=SEED,
                   bs=BS, workers=WORKERS, lam=LAM, E256=self.E, P32=self.Pp, L_FULL=self.l_full, L_LIGHT=self.l_light,
                   snap_step=self.snap_step, md5=self.md5, lr_table=lr_table(self.P, self.has_gate),
                   env={k: self.env[k] for k in ('torch', 'cudnn', 'gpu', 'alloc_conf')})
        return hashlib.sha1(jdumps(obj, sort_keys=True).encode()).hexdigest()

    def prepare_dir(self):
        os.makedirs(self.cdir, exist_ok=True)
        old = [f for f in os.listdir(self.cdir) if f != 'stdout.log']
        if old:
            dst = os.path.join(self.out_root, f'{self.cond}.prev_{datetime.datetime.now():%Y%m%d-%H%M%S}')
            os.makedirs(dst, exist_ok=False)
            for f in old:
                shutil.move(os.path.join(self.cdir, f), os.path.join(dst, f))
            print(f'[{self.cond}] 이전 결과 {len(old)} 개 → {dst} (삭제 없음)', flush=True)
        print(f'==== [{self.cond}] start {now_iso()} script={SCRIPT_VERSION} steps={self.steps} hash={self.hash} '
              f'{"DEBUG/PREFLIGHT" if self.debug else ""}', flush=True)
        if self.loss_custom:
            print(f'==== [{self.cond}] loss (설계 결정 16): lam_ssim={self.lam!r} region={self.loss_region} '
                  f'setting={self.loss_desc()["setting"]} — loss = l1 + lam_ssim·(1−SSIM), 둘 다 {self.loss_region} 영역', flush=True)

    # ── 4~6. 모델·체크섬·optimizer ──
    def build_model(self):
        dev = self.dev
        self.ref_u = None
        if self.P['model'] == 'unet':
            R.seed_all(self.P['init_seed'])
            self.model = R.build_model('unet', dev)
            self.base = None
        else:
            R.seed_all(self.P.get('gru_seed', SEED))         # A2 만 gru_seed (설계 결정 15) — U-Net 은 아래에서 덮어씀
            self.base = R.build_model('gru', dev)
            np_st, py_st = np.random.get_state(), random.getstate()
            with torch.random.fork_rng(devices=[]):
                R.seed_all(SEED)
                ref_u = R.build_model('unet', torch.device('cpu'))
            np.random.set_state(np_st)
            random.setstate(py_st)
            sd = ref_u.unet.state_dict()
            assert list(sd.keys()) == list(self.base.unet.state_dict().keys()), 'U-Net 구조 불일치'
            self.base.unet.load_state_dict(sd)
            del sd
            self.ref_u = ref_u.to(dev) if self.has_gate else None     # I7 용 — 확인 후 삭제
            del ref_u
            gru_precision.apply(self.base, ('gru_h',))
            self.model = self.base
        self.net = self.model
        self.gate = nn.Parameter(torch.zeros(1, R.SEQ_CH, 1, 1, device=dev)) if self.has_gate else None   # RNG 소비 없음
        self.model.train()
        self.model_params = list(self.model.parameters())
        self.clip_params = self.model_params + ([self.gate] if self.gate is not None else [])
        self.named_all = list(self.model.named_parameters()) + ([('seq_gate', self.gate)] if self.gate is not None else [])
        self.gru_params = [p for n, p in self.named_all if n.startswith(('gru_h.', 'gru_v.'))]
        self.key_params = OrderedDict()
        for n, p in self.named_all:
            self.key_params.setdefault(param_key(n), []).append(p)

        # 체크섬 (I1·I2)
        self.unet_ck0 = unet_checksum(self.net)
        if self.cond == 'ref2':
            self.inv['I1'] = dict(ok=True, skipped='ref2 (seed 2 init)', unet_ck=repr(self.unet_ck0))
            self.shared_invariants({'unet_ck_ref2': repr(self.unet_ck0)})
        elif self.loss_custom and LOSS_DERIVED[self.cond][0] == 'ref2':
            # 설계 결정 16: ref2F·ref2L·ref2P — U-Net 초기값 seed 2 == ref2 의 값 (invariants.json unet_ck_ref2 대조; 없으면 기록)
            self.inv['I1'] = dict(ok=True, skipped=f'{self.cond} (seed 2 init, = ref2)', unet_ck=repr(self.unet_ck0))
            self.shared_invariants({'unet_ck_ref2': repr(self.unet_ck0)})
        else:
            self.check('I1', '%.10e' % self.unet_ck0 == EXPECT['unet_ck'], unet_ck='%.10e' % self.unet_ck0,
                       expected=EXPECT['unet_ck'])
            self.shared_invariants({'unet_ck': repr(self.unet_ck0)})
        self.gru_ck0 = None
        if self.has_seq:
            self.gru_ck0 = gru_checksum(self.base)
            if 'gru_seed' in self.P:
                self.inv['I2'] = dict(ok=True, skipped=f'{self.cond} (GRU seed {self.P["gru_seed"]} init)', gru_ck=repr(self.gru_ck0),
                                      differs_from_seed1='%.10e' % self.gru_ck0 != EXPECT['gru_ck'])
                print(f'[{self.cond}] I2 skip (GRU seed {self.P["gru_seed"]}) gru_ck {self.gru_ck0:.10e} (seed 1: {EXPECT["gru_ck"]})', flush=True)
                self.shared_invariants({f'gru_ck_{self.cond}': repr(self.gru_ck0)})
                if self.loss_custom:                    # 설계 결정 16: A2F·A3P 등의 GRU 초기값 == 기준 조건(A2·A3)의 값
                    self.shared_invariants({f'gru_ck_{LOSS_DERIVED[self.cond][0]}': repr(self.gru_ck0)})
                elif self.cond in CLIP_DERIVED:         # 설계 결정 17: A2c 의 GRU 초기값 == A2 의 값
                    self.shared_invariants({f'gru_ck_{CLIP_DERIVED[self.cond]}': repr(self.gru_ck0)})
            else:
                self.check('I2', '%.10e' % self.gru_ck0 == EXPECT['gru_ck'], gru_ck='%.10e' % self.gru_ck0,
                           expected=EXPECT['gru_ck'])
                self.shared_invariants({'gru_ck': repr(self.gru_ck0)})
        self.w0_seq_slice = self.net.unet.down_path[0].block[0].weight.detach()[:, :R.SEQ_CH].float().cpu().clone()

    def build_optimizer(self):
        decay, no_decay, _, names_nd = R.split_decay(self.net)
        gru_ids = {id(p) for p in self.gru_params}
        lists = OrderedDict()
        for plist, kind in ((decay, 'decay'), (no_decay, 'nodecay')):
            lists[f'unet_{kind}'] = [p for p in plist if id(p) not in gru_ids]
            lists[f'gru_{kind}'] = [p for p in plist if id(p) in gru_ids]
        groups = []
        for tag, wd, m, W, eps in preset_groups(self.P, self.has_gate):
            params = [self.gate] if tag == 'gate' else lists[tag]
            groups.append(dict(params=params, weight_decay=wd, eps=eps, tag=tag, m=m, W=W))
        self.opt = torch.optim.AdamW(groups, lr=LR0, betas=BETAS, eps=EPS)
        # I4 — 파라미터 수·그룹 덮개
        n_model = sum(p.numel() for p in self.model_params if p.requires_grad)
        n_all = n_model + (self.gate.numel() if self.gate is not None else 0)
        ids = [id(p) for g in self.opt.param_groups for p in g['params']]
        cover = len(ids) == len(set(ids)) and set(ids) == {id(p) for p in self.clip_params}
        nd_numel = sum(p.numel() for p in no_decay)
        info = dict(n_model=n_model, n_all=n_all, cover_once=cover, no_decay_numel=nd_numel,
                    groups=[dict(tag=g['tag'], tensors=len(g['params']), numel=sum(p.numel() for p in g['params'])) for g in self.opt.param_groups])
        ok = cover
        if self.has_seq:
            n_gru_t = len(self.gru_params)
            n_gru = sum(p.numel() for p in self.gru_params)
            info.update(gru_tensors=n_gru_t, gru_numel=n_gru)
            ok = ok and n_gru_t == EXPECT['gru_tensors'] and nd_numel == EXPECT['no_decay_gru_model']
            if self.has_gate:
                ok = ok and n_all == n_model + R.SEQ_CH
            self.shared_invariants({'n_params_gru_model': n_model})
        else:
            self.shared_invariants({'n_params_unet_model': n_model})
        self.n_params = n_all
        self.check('I4', ok, **info)
        # I5 — LR 식·λ·T_max
        mx = lr_recursion_maxrel(self.steps)
        pod = []
        for p in POD_RECIPES:
            with open(p) as f:
                pod.append(json.load(f)['lambda_ssim'])
        lam_ok = len(set(pod)) == 1 and np.float32(LAM) == np.float32(pod[0]) and LAM == 0.0030745725
        eps_ok = True
        if self.P.get('seq') == 'scale':
            eps_ok = all(np.float32(g['eps']) == np.float32(D_C * 1e-8) for g in self.opt.param_groups if g['tag'].startswith('gru'))
        table = []
        for g in self.opt.param_groups:
            table.append(dict(tag=g['tag'], numel=sum(p.numel() for p in g['params']), wd=g['weight_decay'], eps=g['eps'],
                              lr={str(k): group_lr(k, g['m'], g['W']) for k in (0, 99, 999, 3999)}))
        print(f'[{self.cond}] LR 그룹 표 (base(k)·m·w(k)):', flush=True)
        for t in table:
            print(f"    {t['tag']:<13} numel {t['numel']:>11,}  wd {t['wd']:<7g} eps {t['eps']:<8g} "
                  + ' '.join(f"lr@{k}={v:.6e}" for k, v in t['lr'].items()), flush=True)
        self.lr_table = table
        self.decay_note = decay_fp32_noop()           # 정보용 (불변식 아님): 분리형 감쇠가 수치상 0 인지
        loss_ok, loss_info = True, {}
        if self.loss_custom:
            # 설계 결정 16 — 위 검사(모듈 상수 LAM == pod λ 등)는 그대로 두고, 실제 λ·영역이 LOSS_SETTINGS 의 한 칸과 같은지 더 본다
            want = [v for v in LOSS_SETTINGS.values() if v[1] == self.loss_region and np.float32(v[0]) == np.float32(self.lam)]
            loss_ok = (bool(want) and isinstance(self.lam, float) and math.isfinite(self.lam) and self.lam > 0
                       and self.lam == want[0][0] and self.loss_region in LOSS_REGIONS)
            loss_info = dict(lam_eff=self.lam, loss_region=self.loss_region, loss_setting=self.loss_desc()['setting'],
                             loss_table_ok=loss_ok)
        self.check('I5', mx < 1e-9 and lam_ok and eps_ok and T_MAX == 406650 and loss_ok, lr_formula_vs_torch_maxrel=mx,
                   lam=LAM, pod_lambda=pod, tmax=T_MAX, d_eps_ok=eps_ok, decay_fp32_noop=self.decay_note['noop'], max_norm=self.max_norm,
                   **loss_info)
        if self.loss_custom:
            self.shared_invariants({f'loss_{self.cond}': dict(lam_ssim=self.lam, loss_region=self.loss_region)})
        self.crit = SSIM().to(self.dev)
        self.scaler = torch.amp.GradScaler('cuda')
        self.groups_stat = R.build_groups(self.net, 'gru' if self.has_seq else 'unet')
        self.tracker = SubTracker([(n, p) for n, p in self.model.named_parameters()])

    def set_lrs(self, k):
        out = {}
        for g in self.opt.param_groups:
            g['lr'] = group_lr(k, g['m'], g['W'])
            out[g['tag']] = g['lr']
        return out

    # ── 7. 데이터 ──
    def build_train(self):
        ds = _IdxDS(R.DATA_TRAIN, num_files=None, target_size=C.IMAGE_SIZE[0],
                    augment=C.TRAIN_AUGMENT, augment_flip_p=C.TRAIN_AUGMENT_FLIP_P)
        ds.rng = np.random.default_rng(SEED + 1)                 # 트레이너와 같음 (워커가 다시 시드)
        g = torch.Generator()
        g.manual_seed(SEED)
        self.loader = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=WORKERS, pin_memory=True,
                                 worker_init_fn=R._worker_init_fn, generator=g, persistent_workers=False,
                                 prefetch_factor=C.PREFETCH_FACTOR)
        g.manual_seed(SEED + 1000 * 0)                           # epoch 0 재시드
        self.train_ds = ds
        files = [os.path.basename(self.val_ds.samples[i][0]) for i in self.Pp]
        flags_ok = (torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
                    and torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False)
        sizes_ok = len(ds) == EXPECT['n_train'] and len(self.loader) == EXPECT['n_loader'] and len(self.val_ds) == EXPECT['n_val']
        ok = (flags_ok and sizes_ok and C.NUM_WORKERS_TRAIN == WORKERS and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
              and len(set(files)) == len(self.Pp) and len(self.E) == self.eval_n and len(self.Pp) == self.probe_n
              and len(self.loader) > self.steps)
        n_none = sum(p is None for p in self.E_nxt) + sum(p is None for p in self.P_adj)
        self.check('I9', ok, tf32=[torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
                   deterministic=torch.backends.cudnn.deterministic, benchmark=torch.backends.cudnn.benchmark,
                   n_train=len(ds), n_loader=len(self.loader), n_val=len(self.val_ds), workers=C.NUM_WORKERS_TRAIN,
                   probe_distinct_files=len(set(files)), n_probe=len(self.Pp), n_eval=len(self.E), adj_none=n_none)
        self.shared_invariants({'E256': self.E, 'P32': self.Pp, 'E256_adj_next': self.E_nxt, 'E256_adj_prev': self.E_prv,
                                'E256_cross': self.E_cross, 'P32_adj': self.P_adj, 'P32_cross': self.P_cross,
                                'lambda': LAM, 'tmax': T_MAX})

    def _load_slices(self, ids, keys, bs=1):
        g = torch.Generator()
        g.manual_seed(0)                                          # 전역 RNG 를 건드리지 않게 전용 generator
        return DataLoader(Subset(self.val_ds, list(ids)), batch_size=bs, shuffle=False, num_workers=8,
                          pin_memory=False, generator=g)

    # ── 8. probe 슬라이스 캐시 ──
    def cache_probe(self):
        t0 = time.time()
        keys = list(self.Pp) + (sorted({p for p in self.P_adj if p is not None} - set(self.Pp)) if self.has_seq else [])
        self.cache = {}
        pos = 0
        for b in self._load_slices(keys, None, bs=1):
            i = keys[pos]
            pos += 1
            self.cache[i] = {k: b[k].float().clone() for k in ('data', 'data_img', 'label', 'brain_mask')}
        self.probe_keys = keys
        print(f'[{self.cond}] probe cache {len(keys)} slices {time.time() - t0:.0f}s', flush=True)

    def seq_transform(self, so):
        kind = self.P.get('seq')
        if kind == 'raw':
            return so
        if kind == 'scale':
            return so.float() * D_C
        if kind == 'gate':
            return self.gate * so
        raise ValueError(kind)

    # ── 첫 배치 불변식 (I3·I6·I7) — 학습 forward 전, 각자 자기 autocast 구역 ──
    @torch.no_grad()
    def first_batch_checks(self, sample):
        dev = self.dev
        lab_ck = float(sample['label'].double().sum())
        x_raw = sample['data'].float().to(dev)
        xi_raw = sample['data_img'].float().to(dev)
        data_ck = float(x_raw.abs().sum(dtype=torch.float64))
        self.check('I3', '%.6e' % lab_ck == EXPECT['label_ck'] and '%.6e' % data_ck == EXPECT['data_ck'],
                   label_ck=repr(lab_ck), data_ck=repr(data_ck), expected=[EXPECT['label_ck'], EXPECT['data_ck']],
                   first_idx=sample['idx'].tolist())
        self.shared_invariants({'first_label_ck': repr(lab_ck), 'first_data_ck': repr(data_ck),
                                'first_idx': sample['idx'].tolist()})
        self.first_data_ck = data_ck
        if not self.has_seq:
            return
        s = R.per_sample_scale(xi_raw)
        xk, xi = R.normalize_inputs(x_raw, xi_raw, s)
        mask, sens = sample['mask'].float().to(dev), sample['sens'].float().to(dev)
        base = self.base
        with torch.autocast('cuda'):
            so = base._seq(xk)
        with torch.autocast('cuda'):
            out_dec = base.unet(torch.cat((so, xi), 1))
        with torch.autocast('cuda'):
            out_full = base(xi, xk, mask, sens)
        with torch.autocast('cuda'):
            oh, _, ov, so2 = seq_parts(base, xk)
        d_dec = float((out_dec.float() - out_full.float()).abs().max())
        d_parts = float((so2.float() - so.float()).abs().max())
        self.check('I6', d_dec == 0.0 and d_parts == 0.0 and so2.dtype == so.dtype and oh.dtype == torch.float32
                   and ov.dtype == torch.float16, decomp_maxabs=d_dec, seq_parts_maxabs=d_parts,
                   gru_h_dtype=str(oh.dtype), gru_v_dtype=str(ov.dtype))
        del out_dec, out_full, oh, ov, so2
        if self.has_gate:
            with torch.autocast('cuda'):
                out_e = base.unet(torch.cat((self.gate * so, xi), 1))
            with torch.autocast('cuda'):
                out_u = self.ref_u(xi, xk)
            d7 = float((out_e.float() - out_u.float()).abs().max())
            self.check('I7', d7 == 0.0, out_E_vs_unet_only_maxabs=d7)
            del out_e, out_u
            self.ref_u = None
            torch.cuda.empty_cache()

    # ── 학습 forward (트레이너 :635-655 와 같은 식; autocast 는 forward 만) ──
    def train_forward(self, sample, want_hook):
        dev = self.dev
        x_raw = sample['data'].float().to(dev)
        x_img_raw = sample['data_img'].float().to(dev)
        lab = sample['label'].float().to(dev)
        bm = sample['brain_mask'].float().to(dev)
        s = R.per_sample_scale(x_img_raw)
        x_ksp, x_img = R.normalize_inputs(x_raw, x_img_raw, s)
        lab = lab / s
        so = seq_in = None
        with torch.autocast('cuda'):
            if self.has_seq:
                so = self.base._seq(x_ksp)
                if want_hook:
                    so.register_hook(self._so_hook)
                seq_in = self.seq_transform(so)
                out = self.base.unet(torch.cat((seq_in, x_img), 1))
            else:
                out = self.model(x_img, x_ksp)
        out_fp = out.float()
        if self.loss_custom:
            # 설계 결정 16 — preset 에 lam_ssim·loss_region 이 있는 조건만 ('mask' + LAM 이면 아래 기존 식과 같은 연산)
            l1, lssim = loss_terms(self.crit, out_fp, lab, bm, self.loss_region)
            loss = l1 + self.lam * lssim
        else:
            m_sum = bm.sum().clamp(min=1.0)
            l1 = ((out_fp - lab).abs() * bm).sum() / m_sum
            lssim = 1 - self.crit(out_fp, lab, mask=bm)
            loss = l1 + LAM * lssim
        return dict(x_raw=x_raw, x_ksp=x_ksp, x_img=x_img, lab=lab, bm=bm, s=s, so=so, seq_in=seq_in, out_fp=out_fp,
                    l1=l1, lssim=lssim, loss=loss)

    def _so_hook(self, g):
        # fp16 기울기 하한 감시 (값을 바꾸지 않음 — None 반환)
        ga = g.detach().abs()
        self.hook_st = dict(so_grad_zero=(ga == 0).float().mean(), so_grad_subnormal=(ga < FP16_TINY).float().mean(),
                            so_grad_scaled_rms=ga.float().pow(2).mean().sqrt(), so_grad_dtype=str(g.dtype))
        return None

    def grads_finite(self):
        gs = [p.grad for p in self.clip_params if p.grad is not None]
        return (not gs) or bool(torch.isfinite(torch.stack(torch._foreach_norm(gs))).all())

    @torch.no_grad()
    def group_grad_stats(self):
        out = {}
        for key, ps in self.key_params.items():
            gs = [p.grad for p in ps if p.grad is not None]
            out[f'gn_{key}'] = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs)))) if gs else None
        W = self.net.unet.down_path[0].block[0].weight.grad
        L = self.net.unet.last.weight.grad
        if W is not None:
            out['first_conv_seq_grad_rms'] = _rms(W[:, :R.SEQ_CH])
            out['first_conv_zf_grad_rms'] = _rms(W[:, R.SEQ_CH:])
        if L is not None:
            out['last_seq_grad_rms'] = _rms(L[:, :R.SEQ_CH])
        if self.has_seq:
            for mod in ('gru_h', 'gru_v'):
                zs = []
                for suf in ('weight_ih_l0', 'weight_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse'):
                    name = f'{mod}.{suf}'
                    p = getattr(getattr(self.base, mod), suf)
                    if p.grad is None:
                        continue
                    idx = self.tracker.idx_of(name)
                    a = p.grad.reshape(-1)
                    if idx is not None:
                        a = a[idx.to(a.device)]
                    zs.append(float((a == 0).float().mean()))
                out[f'{mod}_wgrad_zero_frac'] = float(np.mean(zs)) if zs else None
        return out

    @torch.no_grad()
    def adam_stats(self):
        hp = {id(p): g['eps'] for g in self.opt.param_groups for p in g['params']}
        acc = {}
        for n, p in self.named_all:
            r = R.adam_param_stats(self.opt.state.get(p), hp[id(p)], BETAS)
            if r is None:
                continue
            a = acc.setdefault(param_key(n), [0, 0.0, 0])
            a[0] += r[0]
            a[1] += r[1]
            a[2] += p.numel()
        return {k: dict(eps_dom_frac=a[0] / a[2], upd_ratio_mean=a[1] / a[2]) for k, a in acc.items()}

    @torch.no_grad()
    def gru_ck_now(self):
        return gru_checksum(self.base) if self.has_seq else None

    # ── 경량 기록 (L_LIGHT) ──
    def light_record(self, step, gstats, fw, win, step_line):
        t0 = time.time()
        rec = OrderedDict(step=step, time=now_iso(), elapsed=time.time() - self.t_train0 if self.t_train0 else 0.0)
        n = max(win['n'], 1)
        rec['window'] = dict(n=win['n'], loss=win['loss'] / n, l1=win['l1'] / n, ssim_loss=win['ssim'] / n)
        rec['counters'] = dict(self.cnt)
        rec['grad'] = gstats
        if self.hook_st and gstats is not None:
            rec['so_grad'] = {k: (v if isinstance(v, str) else tfloat(v)) for k, v in self.hook_st.items()}
        self.hook_st = None
        try:
            if self.has_seq and fw is not None:
                rec['sat'] = sat_probe(self.base, fw['x_ksp'], fw['so'])
            rec['adam'] = self.adam_stats()
            rec['rel_change'] = self.tracker.update()
            rec['io'] = R.io_slice_stats(self.net)
            if step % 500 == 0:
                rec['near_dead'] = near_dead_fracs(self.net, self.groups_stat)
            if self.P.get('seq') == 'scale' and step_line is not None and step_line.get('img_rms'):
                rec['d_scale'] = dict(c=D_C, seqin_over_img_rms=step_line['seqin_rms'] / step_line['img_rms'])
            if self.gate is not None:
                a = self.gate.detach().reshape(-1).float().cpu()
                rec['gate'] = dict(alpha=a.tolist(), mean_abs=float(a.abs().mean()), max_abs=float(a.abs().max()))
            rec['unet_ck'] = repr(unet_checksum(self.net))
            rec['gru_ck'] = repr(self.gru_ck_now()) if self.has_seq else None
        finally:
            self.net.train()
        rec['mem_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
        rec['sec'] = time.time() - t0
        self.f_log.write(jdumps(rec) + '\n')
        self.f_log.flush()
        return rec

    # ── 전체 probe (L_FULL) ──
    def full_probe(self, step):
        t0 = time.time()
        self.net.eval()
        try:
            rec = self._probe_body(step)
        finally:
            self.net.train()
        rec['sec'] = time.time() - t0
        self.f_probe.write(jdumps(rec) + '\n')
        self.f_probe.flush()
        t = rec.get('tests', {})
        msg = (f'[{self.cond}] PROBE step {step} ssim {rec["mid_eval"]["ssim"]:.4f} psnr {rec["mid_eval"]["psnr"]:.2f}')
        if self.has_seq:
            msg += (f' seq_rms {rec["seq"]["seq_rms"]:.4f} sat_v {rec["sat_v"]:.3f} common {rec["seq"]["common_frac"]:.4f}'
                    f' dep {rec["seq"].get("dep", float("nan")):.3f} var_change {rec.get("var_change", float("nan")):.3f}'
                    f' | zero {t["zero"]["mean"]:+.4f} adj {t["adj"]["mean"]:+.5f} ({t["adj"]["n_pos"]}/{t["adj"]["n"]})'
                    f' cross {t["cross"]["mean"]:+.5f} chanmean {t["chanmean"]["mean"]:+.4f}')
        print(msg + f' ({rec["sec"]:.1f}s)', flush=True)
        return rec

    @torch.no_grad()
    def _probe_body(self, step):
        dev, base, net = self.dev, self.base, self.net
        rec = OrderedDict(step=step)
        Pp = self.Pp
        n = len(Pp)
        SP = {}
        first = None
        sat_h, sat_v = {}, {}
        if self.has_seq:
            keys = self.probe_keys
            for j in range(0, len(keys), 8):
                ks = keys[j:j + 8]
                xk = torch.cat([self.cache[i]['data'] for i in ks]).to(dev)
                xi = torch.cat([self.cache[i]['data_img'] for i in ks]).to(dev)
                s = R.per_sample_scale(xi)
                xk_n, _ = R.normalize_inputs(xk, xi, s)
                with torch.autocast('cuda'):
                    out_h, oh, out_v, so = seq_parts(base, xk_n)
                sh = (out_h.abs() > 0.99).float().mean(dim=(1, 2)).tolist()
                sv = (out_v.abs() > 0.99).float().mean(dim=(1, 2)).tolist()
                for b, i in enumerate(ks):
                    SP[i] = so[b].contiguous()
                    sat_h[i], sat_v[i] = sh[b], sv[b]
                if j == 0:
                    first = dict(xk=xk_n, out_h=out_h, oh=oh, out_v=out_v)
                else:
                    del out_h, oh, out_v
            with torch.autocast('cuda'):
                f0 = seq_parts(base, torch.zeros((1,) + tuple(self.cache[Pp[0]]['data'].shape[1:]), device=dev))[3][0].float()
            S = torch.stack([SP[i].float() for i in Pp])
            partners = [SP[p].float() if p is not None else None for p in self.P_adj]
            st, Fm, D = seq_stats(S, f0=f0, partners=partners)
            seqin = self.seq_transform(S)
            st['seqin_rms'] = _rms(seqin)
            with torch.autocast('cuda', enabled=False):
                Wl = net.unet.last.weight.float()[:, :R.SEQ_CH]
                st['last_seq_contrib_rms'] = _rms(F.conv2d(seqin.float(), Wl))
            rec['seq'] = st
            rec['sat_v'] = float(np.mean([sat_v[i] for i in Pp]))
            rec['sat_h'] = float(np.mean([sat_h[i] for i in Pp]))
            # 초기 대비 특징 변화 (step 0 에 S_0 저장)
            if step == 0 or self.S0 is None:
                self.S0 = S.half().cpu()
            S0 = self.S0.to(dev).float()
            D0 = S0 - S0.mean(0, keepdim=True)
            rec['feat_change'] = float((S - S0).norm() / S0.norm().clamp(min=1e-30))
            rec['var_change'] = float((D - D0).norm() / D0.norm().clamp(min=1e-30))
            del S0, D0
            # 게이트 미분 (첫 8 슬라이스) — gru_h 는 fp32 계산, gru_v 는 fp16 입력·가중치 모사
            B = first['xk'].size(0)
            Xh = first['xk'].reshape([B, base.num_in_x, -1]).float()
            rec['gate_h'] = gru_gate_derivs(base.gru_h, Xh, first['out_h'], base.num_out1, mimic_half=False)
            rec['gate_v'] = gru_gate_derivs(base.gru_v, first['oh'], first['out_v'], base.num_out2, mimic_half=True)
            del first, Xh, S, Fm, D, seqin, partners
        # 기능 시험 + mid-eval (32 기준 슬라이스) — normal 도 저장 경로(cat(seq_in, x_img_n fp16))로 계산
        tests = ['normal'] + (['zero', 'adj', 'cross', 'chanmean'] if self.has_seq else [])
        rows = {t: {} for t in tests}
        mets = []
        outs_rms = []
        i10 = None
        for j in range(0, n, 8):
            ks = Pp[j:j + 8]
            xi_raw = torch.cat([self.cache[i]['data_img'] for i in ks]).to(dev)
            s = R.per_sample_scale(xi_raw)
            xi16 = (xi_raw / s).half()
            ref = torch.cat([self.cache[i]['label'] for i in ks]).to(dev)
            bm = torch.cat([self.cache[i]['brain_mask'] for i in ks]).to(dev)
            B = len(ks)
            zeros = torch.zeros((B, R.SEQ_CH) + tuple(xi_raw.shape[2:]), device=dev)
            variants = OrderedDict()
            if self.has_seq:
                so = torch.stack([SP[i] for i in ks])
                variants['normal'] = (self.seq_transform(so), list(range(B)))
                variants['zero'] = (zeros, list(range(B)))
                adj_rows = [b for b, i in enumerate(ks) if self.P_adj[Pp.index(i)] is not None]
                so_adj = torch.stack([SP[self.P_adj[Pp.index(i)]] if self.P_adj[Pp.index(i)] is not None else SP[i] for i in ks])
                variants['adj'] = (self.seq_transform(so_adj), adj_rows)
                so_cross = torch.stack([SP[self.P_cross[Pp.index(i)]] for i in ks])
                variants['cross'] = (self.seq_transform(so_cross), list(range(B)))
                so_cm = so.float().mean(dim=(-2, -1), keepdim=True).expand_as(so)
                variants['chanmean'] = (self.seq_transform(so_cm), list(range(B)))
            else:
                variants['normal'] = (zeros, list(range(B)))
            for t, (sq, use) in variants.items():
                with torch.autocast('cuda'):
                    o = net.unet(torch.cat((sq, xi16), 1))
                if t == 'normal':
                    outs_rms.append(_rms(o))
                    if step == 0 and j == 0:
                        # I10: 저장 경로 == 직접 forward (학습과 같은 fp32 x_img_n)
                        xk = torch.cat([self.cache[i]['data'] for i in ks]).to(dev)
                        xk_n, xi_n = R.normalize_inputs(xk, xi_raw, s)
                        with torch.autocast('cuda'):
                            if self.has_seq:
                                o_d = net.unet(torch.cat((self.seq_transform(base._seq(xk_n)), xi_n), 1))
                            else:
                                o_d = self.model(xi_n, xk_n)
                        i10 = float((o_d.float() - o.float()).abs().max())
                        del xk, xk_n, xi_n, o_d
                o = o.float() * s
                for b in use:
                    m = metrics(o[b:b + 1], ref[b:b + 1], bm[b:b + 1])
                    rows[t][ks[b]] = m['ssim']
                    if t == 'normal':
                        mets.append(m)
        if i10 is not None:
            self.inv['I10'] = dict(ok=i10 == 0.0, flag_only=True, stored_vs_direct_maxabs=i10)
            if i10 != 0.0:
                self.flags.append(f'I10 stored-path != direct forward (max|Δ|={i10:.3g}) — normal·시험은 같은 저장 경로라 비교는 일관')
            print(f'[{self.cond}] I10 stored-path vs direct max|Δ| = {i10}', flush=True)
        rec['mid_eval'] = {k: float(np.mean([m[k] for m in mets])) for k in ('ssim', 'psnr', 'nmse', 'l1')}
        rec['out_rms'] = float(np.mean(outs_rms))
        base_ssim = [rows['normal'][i] for i in Pp]
        rec['per_slice_ssim'] = base_ssim
        if self.has_seq:
            rec['tests'] = OrderedDict()
            rec['per_slice_drop'] = OrderedDict()
            for t in ('zero', 'adj', 'cross', 'chanmean'):
                tv = [rows[t].get(i) for i in Pp]
                rec['tests'][t] = drop_stats(base_ssim, tv)
                rec['per_slice_drop'][t] = [None if v is None else b - v for b, v in zip(base_ssim, tv)]
        else:
            rec['seq'] = dict(last_seq_contrib_rms=0.0)
        rec['io'] = R.io_slice_stats(net)
        if self.gate is not None:
            a = self.gate.detach().reshape(-1).float().cpu()
            rec['gate'] = dict(mean_abs=float(a.abs().mean()), max_abs=float(a.abs().max()))
        return rec

    # ── E256 평가 (최종: 시험 포함 / 스냅숏: 정상 지표만) ──
    def eval_set(self, full, tag):
        t0 = time.time()
        self.net.eval()
        try:
            res = self._eval_body(full, tag)
        finally:
            self.net.train()
        res['summary']['sec'] = time.time() - t0
        print(f'[{self.cond}] EVAL {tag} n={res["summary"]["n"]} ssim {res["summary"]["ssim"]:.4f} '
              f'psnr {res["summary"]["psnr"]:.2f} nmse {res["summary"]["nmse"]:.4f} ({res["summary"]["sec"]:.0f}s)', flush=True)
        return res

    @torch.no_grad()
    def _eval_body(self, full, tag):
        dev, net = self.dev, self.net
        ids = self.E
        idset = set(ids)
        pos_of = {i: k for k, i in enumerate(ids)}
        tests_on = full and self.has_seq
        extra = []
        if tests_on:
            extra = sorted({j for j in self.E_nxt + self.E_prv if j is not None} - idset)
        load = ids + extra
        seq_store, base_info, rows = {}, {}, {}
        it = iter(self._load_slices(load, None, bs=8))
        pos = 0
        for b in it:
            ks = load[pos:pos + b['label'].shape[0]]
            pos += len(ks)
            x_raw = b['data'].float().to(dev)
            xi_raw = b['data_img'].float().to(dev)
            lab = b['label'].float().to(dev)
            bm = b['brain_mask'].float().to(dev)
            s = R.per_sample_scale(xi_raw)
            xk_n, xi_n = R.normalize_inputs(x_raw, xi_raw, s)
            xi16 = xi_n.half()
            so = None
            if self.has_seq:
                with torch.autocast('cuda'):
                    so = self.base._seq(xk_n)
                for bi, i in enumerate(ks):
                    seq_store[i] = so[bi].contiguous().cpu()
            sel = [bi for bi, i in enumerate(ks) if i in idset]
            if not sel:
                continue
            zeros = torch.zeros((len(ks), R.SEQ_CH) + tuple(xi_raw.shape[2:]), device=dev)
            variants = OrderedDict(normal=self.seq_transform(so) if self.has_seq else zeros)
            if tests_on:
                variants['zero'] = zeros
                variants['chanmean'] = self.seq_transform(so.float().mean(dim=(-2, -1), keepdim=True).expand_as(so))
            outs = {}
            for t, sq in variants.items():
                with torch.autocast('cuda'):
                    outs[t] = net.unet(torch.cat((sq, xi16), 1)).float() * s
            zf = torch.sqrt((xi_raw ** 2).sum(1, keepdim=True))
            for bi in sel:
                i = ks[bi]
                fp, sl, _ = self.val_ds.samples[i]
                row = OrderedDict(idx=i, file=os.path.basename(fp), slice=sl)
                k = pos_of[i]
                row.update(adj_idx=self.E_nxt[k], adj2_idx=self.E_prv[k], cross_idx=self.E_cross[k])
                mz = metrics(zf[bi:bi + 1], lab[bi:bi + 1], bm[bi:bi + 1])
                row.update(zf_ssim=mz['ssim'], zf_psnr=mz['psnr'], zf_nmse=mz['nmse'])
                m = metrics(outs['normal'][bi:bi + 1], lab[bi:bi + 1], bm[bi:bi + 1])
                row.update(ssim=m['ssim'], psnr=m['psnr'], nmse=m['nmse'], l1=m['l1'])
                for t in ('zero', 'chanmean'):
                    if t in outs:
                        mt = metrics(outs[t][bi:bi + 1], lab[bi:bi + 1], bm[bi:bi + 1])
                        row.update({f'ssim_{t}': mt['ssim'], f'psnr_{t}': mt['psnr'], f'nmse_{t}': mt['nmse']})
                rows[i] = row
                if tests_on:
                    base_info[i] = dict(xi16=xi16[bi].cpu(), s=s[bi].cpu(), lab=lab[bi].cpu(), bm=bm[bi].cpu())
            del outs, zeros, so, x_raw, xi_raw, xk_n, xi_n, xi16
        del it
        summ = OrderedDict(n=len(rows), tag=tag)
        for k in ('ssim', 'psnr', 'nmse', 'l1', 'zf_ssim', 'zf_psnr', 'zf_nmse'):
            summ[k] = float(np.mean([rows[i][k] for i in ids]))
        if tests_on:
            # 256 슬라이스 시퀀스 통계 (스트리밍, fp64 누적)
            acc = torch.zeros(seq_store[ids[0]].shape, dtype=torch.float64, device=dev)
            es = 0.0
            for i in ids:
                Si = seq_store[i].to(dev).double()
                acc += Si
                es += float(Si.pow(2).sum())
            n = len(ids)
            Fm = (acc / n).float()
            common = n * float(Fm.double().pow(2).sum()) / max(es, 1e-300)
            with torch.autocast('cuda'):
                f0 = self.base._seq(torch.zeros((1, 2 * C.N_COIL) + tuple(Fm.shape[1:]), device=dev))[0].float()
            ed = ew = 0.0
            deps = []
            for i in ids:
                Si = seq_store[i].to(dev).float()
                Di = Si - Fm
                ed += float(Di.double().pow(2).sum())
                ew += Di.shape[-1] * float(Di.mean(-1, keepdim=True).double().pow(2).sum())
                deps.append(float((Si - f0).norm()) / max(float(f0.norm()), 1e-30))
                r = rows[i]
                r['seq_rms'] = _rms(Si)
                j = self.E_nxt[pos_of[i]]
                r['seq_adj_rel'] = (float((Si - seq_store[j].to(dev).float()).norm()) / max(float(Di.norm()), 1e-30)
                                    if j is not None else None)
            summ['seq256'] = dict(common_frac=common, common_frac_unbiased=1 - (1 - common) * n / (n - 1),
                                  wconst_D=ew / max(ed, 1e-300), dep=float(np.mean(deps)), D_rms=math.sqrt(ed / (n * Fm.numel())),
                                  seq_rms=math.sqrt(es / (n * Fm.numel())))
            # pass 2: 이웃(다음·이전)·다른 볼륨 seq 치환
            for c0 in range(0, len(ids), 8):
                cs = ids[c0:c0 + 8]
                xi16 = torch.stack([base_info[i]['xi16'] for i in cs]).to(dev)
                s = torch.stack([base_info[i]['s'] for i in cs]).to(dev)
                lab = torch.stack([base_info[i]['lab'] for i in cs]).to(dev)
                bm = torch.stack([base_info[i]['bm'] for i in cs]).to(dev)
                for t, plist in (('adj', [self.E_nxt[pos_of[i]] for i in cs]), ('adjprev', [self.E_prv[pos_of[i]] for i in cs]),
                                 ('cross', [self.E_cross[pos_of[i]] for i in cs])):
                    use = [b for b, j in enumerate(plist) if j is not None]
                    if not use:
                        continue
                    sq = torch.stack([seq_store[plist[b]] for b in use]).to(dev)
                    with torch.autocast('cuda'):
                        o = net.unet(torch.cat((self.seq_transform(sq), xi16[use]), 1)).float() * s[use]
                    for q, b in enumerate(use):
                        mt = metrics(o[q:q + 1], lab[b:b + 1], bm[b:b + 1])
                        rows[cs[b]].update({f'ssim_{t}': mt['ssim'], f'psnr_{t}': mt['psnr'], f'nmse_{t}': mt['nmse']})
            base_ssim = [rows[i]['ssim'] for i in ids]
            tests = OrderedDict()
            for t in ('zero', 'adj', 'adjprev', 'cross', 'chanmean'):
                tests[t] = drop_stats(base_ssim, [rows[i].get(f'ssim_{t}') for i in ids])
            sym = []
            for i in ids:
                r = rows[i]
                ds = [r['ssim'] - r[f'ssim_{t}'] for t in ('adj', 'adjprev') if r.get(f'ssim_{t}') is not None]
                r['d_adj_sym'] = float(np.mean(ds)) if ds else None
                sym.append(r['d_adj_sym'])
            # 대칭 이웃 하락 = base − (base − d) 꼴로 drop_stats 재사용
            tests['adj_sym'] = drop_stats([0.0] * len(sym), [None if d is None else -d for d in sym])
            summ['tests'] = tests
        return dict(summary=summ, rows=[rows[i] for i in ids])

    def write_eval_csv(self, res, name):
        cols = ['idx', 'file', 'slice', 'adj_idx', 'adj2_idx', 'cross_idx', 'zf_ssim', 'zf_psnr', 'zf_nmse',
                'ssim', 'psnr', 'nmse', 'l1']
        for t in ('zero', 'adj', 'adjprev', 'cross', 'chanmean'):
            cols += [f'ssim_{t}', f'psnr_{t}', f'nmse_{t}']
        cols += ['d_adj_sym', 'seq_rms', 'seq_adj_rel']
        path = os.path.join(self.cdir, name)
        with open(path + '.tmp', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in res['rows']:
                w.writerow(['' if r.get(c) is None else (repr(r[c]) if isinstance(r[c], float) else r[c]) for c in cols])
        os.replace(path + '.tmp', path)

    # ── 요약 ──
    def write_summary(self, status, extra=None):
        summ = OrderedDict(status=status, cond=self.cond, script_version=SCRIPT_VERSION, config_hash=getattr(self, 'hash', None),
                           debug=self.debug, steps_target=self.steps, time=now_iso(), preset=self.P,
                           flags=self.flags, invariants=self.inv, pod_counterpart=POD_COUNTERPART.get(self.cond),
                           total_sec=time.time() - self.t_start)
        if self.loss_custom:
            summ['loss'] = self.loss_desc()                     # 설계 결정 16
        if extra:
            summ.update(extra)
        write_json_atomic(summ, os.path.join(self.cdir, 'summary.json'))
        self.summary_written = True

    # ── 실행 ──
    def run(self):
        a = self.args
        self.startup_guards()
        self.build_val()
        self.hash = self.config_hash()
        if a.skip_done:
            sp = os.path.join(self.cdir, 'summary.json')
            if os.path.exists(sp):
                with open(sp) as f:
                    prev = json.load(f)
                if prev.get('status') == 'completed' and prev.get('config_hash') == self.hash:
                    print(f'[{self.cond}] skip-done: completed, 같은 config_hash {self.hash}', flush=True)
                    return EXIT_OK
        self.prepare_dir()
        meta = OrderedDict(cond=self.cond, preset=self.P, script_version=SCRIPT_VERSION, config_hash=self.hash, debug=self.debug,
                           args=vars(a), steps=self.steps, tmax=T_MAX, seed=SEED, bs=BS, workers=WORKERS, lam=LAM,
                           L_FULL=self.l_full, L_LIGHT=self.l_light, snap_step=self.snap_step, md5=self.md5, env=self.env,
                           pod_counterpart=POD_COUNTERPART, start=now_iso())
        if self.loss_custom:
            meta['loss'] = self.loss_desc()                     # 설계 결정 16 (meta 의 lam 은 모듈 상수 LAM 그대로)
        write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))
        try:
            rc = self._run_inner(meta)
        except InvariantError as e:
            print(f'[{self.cond}] INVARIANT VIOLATION: {e}', flush=True)
            meta['invariants'] = self.inv
            write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))
            self.write_summary('invalid', dict(error=str(e)))
            return EXIT_INVALID
        except SystemExit:
            raise
        except BaseException as e:                                          # noqa: BLE001
            tb = traceback.format_exc()
            print(tb, flush=True)
            meta['invariants'] = self.inv
            write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))
            self.write_summary('failed', dict(error=f'{type(e).__name__}: {e}', traceback=tb))
            return EXIT_FAIL
        return rc

    def _run_inner(self, meta):
        a = self.args
        self.build_model()
        self.build_optimizer()
        meta['lr_table'] = self.lr_table
        meta['n_params'] = self.n_params
        self.build_train()
        self.cache_probe()
        self.S0 = None
        self.hook_st = None
        self.t_train0 = None
        self.cnt = OrderedDict(amp_skip=0, grad_retry=0, grad_recovered=0, amp_floor=0, nan_retry=0, nan_recovered=0,
                               nan_skip=0, clipped=0)
        self.f_steps = open(os.path.join(self.cdir, 'steps.jsonl'), 'w')
        self.f_log = open(os.path.join(self.cdir, 'log.jsonl'), 'w')
        self.f_probe = open(os.path.join(self.cdir, 'probe.jsonl'), 'w')
        self.aux_bad = 0                    # 설계 결정 16 — aux 항 ≠ 학습 항인 step 수 (손실 요인 조건만 셈, E7)

        # step 0: 기록 + probe (학습 전)
        win = dict(n=0, loss=0.0, l1=0.0, ssim=0.0)
        t_probe = time.time()
        self.light_record(0, None, None, win, None)
        self.full_probe(0)
        probe_sec = time.time() - t_probe
        meta['invariants'] = self.inv
        write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))

        it = iter(self.loader)
        self.t_train0 = time.time()
        global_step, bpos, consec = 0, -1, 0
        steps = self.steps
        lines_meta = []                     # (bpos, idx, data_ck) — stream_md5·batch_idx.npy
        step_secs = []
        self.gn_hist = []                   # (bpos, gn, clipped) — 구간별 clip 수·최대 gn
        i8_done = not self.has_gate
        snap = None
        n_lines = 0
        self.opt.zero_grad(set_to_none=True)
        while global_step < steps:
            t_it = time.time()
            sample = next(it)
            bpos += 1
            lrs = self.set_lrs(global_step)
            idx = sample['idx'].tolist()
            if bpos == 0:
                self.first_batch_checks(sample)
                meta['invariants'] = self.inv
                write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))
            want_light = (global_step + 1) in self.l_light
            fw = self.train_forward(sample, want_light)
            with torch.no_grad():
                data_ck = float(fw['x_raw'].abs().sum(dtype=torch.float64))
            nan_retry_here = False
            for _ in range(TRAIN_NAN_RETRY):
                if torch.isfinite(fw['loss']):
                    break
                first = fw['loss'].item()
                del fw
                fw = self.train_forward(sample, want_light)
                ok = bool(torch.isfinite(fw['loss']))
                self.cnt['nan_retry'] += 1
                self.cnt['nan_recovered'] += int(ok)
                nan_retry_here = True
                print(f'[{self.cond}] NaN-retry bpos {bpos} first={first} {"recovered" if ok else "still non-finite"}', flush=True)
            if not torch.isfinite(fw['loss']):
                lval = fw['loss'].item()
                del fw
                self.opt.zero_grad(set_to_none=True)
                consec += 1
                self.cnt['nan_skip'] += 1
                line = dict(step=global_step, bpos=bpos, idx=idx, skipped=True, loss=lval, data_ck=data_ck,
                            lr_unet=lrs['unet_decay'], nan_retry=nan_retry_here, sec=time.time() - t_it)
                self.f_steps.write(jdumps(line) + '\n')
                lines_meta.append(dict(bpos=bpos, idx=idx, data_ck=data_ck))
                print(f'[{self.cond}] NaN-skip bpos {bpos} consec {consec} total {self.cnt["nan_skip"]}', flush=True)
                if consec >= C.MAX_CONSEC_SKIP:
                    raise RuntimeError(f'{consec} consecutive non-finite losses')
                continue
            consec = 0
            self.scaler.scale(fw['loss']).backward()
            grad_retry_here = grad_rec_here = False
            fin = None
            for _ in range(AMP_GRAD_RETRY):
                fin = self.grads_finite()
                if fin:
                    break
                del fw
                self.opt.zero_grad(set_to_none=True)
                fw = self.train_forward(sample, want_light)
                self.scaler.scale(fw['loss']).backward()
                fin = self.grads_finite()
                self.cnt['grad_retry'] += 1
                self.cnt['grad_recovered'] += int(fin)
                grad_retry_here, grad_rec_here = True, fin
            if fin is None:
                fin = self.grads_finite()
            self.scaler.unscale_(self.opt)
            if not i8_done and fin and bool((self.gate.detach() == 0).all()):
                gru_zero = all(p.grad is not None and bool((p.grad == 0).all()) for p in self.gru_params)
                gg = self.gate.grad
                ok8 = gru_zero and gg is not None and bool(torch.isfinite(gg).all()) and bool((gg != 0).any())
                self.check('I8', ok8, bpos=bpos, gru_grads_all_zero=gru_zero,
                           gate_grad_absmax=None if gg is None else float(gg.abs().max()))
                i8_done = True
            gstats = self.group_grad_stats() if want_light else None
            gn_model = None
            if self.gate is not None:
                with torch.no_grad():
                    gs = [p.grad for p in self.model_params if p.grad is not None]
                    gn_model = float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gs, 2.0)), 2.0))
            gn = float(torch.nn.utils.clip_grad_norm_(self.clip_params, max_norm=self.max_norm))   # 설계 결정 17
            self.scaler.step(self.opt)
            self.scaler.update()
            amp_skip = not math.isfinite(gn)
            clipped = math.isfinite(gn) and gn > self.max_norm
            self.cnt['amp_skip'] += int(amp_skip)
            self.cnt['clipped'] += int(clipped)
            self.gn_hist.append((bpos, gn, clipped))
            if self.scaler.get_scale() < AMP_MIN_SCALE:
                self.scaler.update(AMP_MIN_SCALE)
                self.cnt['amp_floor'] += 1
                print(f'[{self.cond}] AMP-floor bpos {bpos} total {self.cnt["amp_floor"]}', flush=True)
            self.opt.zero_grad(set_to_none=True)
            global_step += 1

            # step 줄 (통계는 새 텐서로만 — in-place 없음)
            with torch.no_grad():
                vals = [fw['loss'], fw['l1'], fw['lssim'], fw['s'].mean(), fw['x_img'].pow(2).mean().sqrt()]
                if self.has_seq:
                    sod = fw['so'].detach()
                    vals += [sod.float().pow(2).mean().sqrt(), fw['seq_in'].detach().float().pow(2).mean().sqrt(),
                             (sod.abs() > 0.99).float().mean()]
                st = torch.stack([v.detach().double().reshape(()) for v in vals]).tolist()
                l1_num = ((fw['out_fp'].detach() - fw['lab']).abs() * fw['bm']).sum(dim=(1, 2, 3)).tolist()
                m_sum = fw['bm'].sum(dim=(1, 2, 3)).tolist()
                aux = None
                if self.loss_custom:
                    # 설계 결정 16 — 두 영역의 항 + P0 목적함수 (optimizer step 뒤, 학습 forward 의 출력 사본으로만 — 기울기 없음)
                    o_d = fw['out_fp'].detach()
                    l1m, lsm = loss_terms(self.crit, o_d, fw['lab'], fw['bm'], 'mask')
                    l1f, lsf = loss_terms(self.crit, o_d, fw['lab'], fw['bm'], 'full')
                    l1s, lss, sfr = supp_terms(self.crit, o_d, fw['lab'])          # GT 지지 영역 (label ≠ 0) — 기록 전용
                    av = torch.stack([v.detach().double().reshape(()) for v in (l1m, lsm, l1f, lsf, l1m + LAM * lsm,
                                                                                 l1s, lss, sfr)]).tolist()
                    aux = OrderedDict(zip(('l1_mask', 'ssim_loss_mask', 'l1_full', 'ssim_loss_full', 'loss_p0',
                                           'l1_supp', 'ssim_loss_supp', 'supp_frac'), av))
                    del o_d, l1m, lsm, l1f, lsf, l1s, lss, sfr
                    # 같은 영역의 aux 항 == 학습 항 (같은 텐서·같은 식 → 비트 단위); P0 설정(Aeq)이면 loss_p0 == loss 까지
                    sfx = 'mask' if self.loss_region == 'mask' else 'full'
                    same = aux[f'l1_{sfx}'] == st[1] and aux[f'ssim_loss_{sfx}'] == st[2]
                    if (self.lam, self.loss_region) == LOSS_SETTINGS['0']:
                        same = same and aux['loss_p0'] == st[0]
                    self.aux_bad += int(not same)
            line = OrderedDict(step=global_step, bpos=bpos, idx=idx, skipped=False, loss=st[0], l1=st[1], ssim_loss=st[2],
                               gn=gn, gn_model=gn_model, clipped=clipped, amp_skip=amp_skip, amp_scale=self.scaler.get_scale(),
                               grad_retry=grad_retry_here, grad_recovered=grad_rec_here, nan_retry=nan_retry_here,
                               lr_unet=lrs['unet_decay'], lr_gru=lrs.get('gru_decay'), lr_gate=lrs.get('gate'),
                               s_mean=st[3], data_ck=data_ck, img_rms=st[4])
            if self.has_seq:
                line.update(seq_rms=st[5], seqin_rms=st[6], sat_v_batch=st[7])
            if self.gate is not None:
                ga = self.gate.detach().abs()
                line.update(gate_absmean=float(ga.mean()), gate_absmax=float(ga.max()))
            line['l1_num'] = l1_num
            line['m_sum'] = m_sum
            if aux is not None:
                line['aux'] = aux
            if global_step in CK_STEPS:
                line['unet_ck'] = repr(unet_checksum(self.net))
                line['gru_ck'] = repr(self.gru_ck_now()) if self.has_seq else None
            line['sec'] = time.time() - t_it
            step_secs.append(line['sec'])
            self.f_steps.write(jdumps(line) + '\n')
            n_lines += 1
            if n_lines % 50 == 0:
                self.f_steps.flush()
            lines_meta.append(dict(bpos=bpos, idx=idx, data_ck=data_ck))
            win['n'] += 1
            win['loss'] += st[0]
            win['l1'] += st[1]
            win['ssim'] += st[2]
            if global_step % 50 == 0 or global_step <= 10:
                print(f'[{self.cond}] step {global_step}/{steps} loss {st[0]:.5f} gn {gn:.4g} scale {line["amp_scale"]:g} '
                      f'lr_u {lrs["unet_decay"]:.4e}'
                      + (f' lr_g {lrs["gru_decay"]:.4e} seq_rms {st[5]:.4f} sat_v {st[7]:.3f}' if self.has_seq else '')
                      + (f' |α| {line["gate_absmean"]:.5f}' if self.gate is not None else '')
                      + f' {line["sec"]:.2f}s mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G', flush=True)
            t_p = time.time()
            if global_step in self.l_light:
                self.f_steps.flush()
                self.light_record(global_step, gstats, fw, win, line)
                win = dict(n=0, loss=0.0, l1=0.0, ssim=0.0)
            if global_step in self.l_full:
                try:
                    self.full_probe(global_step)
                except Exception as e:                                      # noqa: BLE001
                    raise RuntimeError(f'probe step {global_step} 실패 (S1·V4 에 모든 probe 필요): {e}') from e
            if global_step == self.snap_step and global_step < steps:
                snap = self.eval_set(full=False, tag=f'step{global_step}')
                self.write_eval_csv(snap, f'eval{global_step}_per_slice.csv')
            probe_sec += time.time() - t_p
            del fw
        train_sec = time.time() - self.t_train0
        self.f_steps.flush()
        del it
        self.loader = None
        import gc
        gc.collect()

        # 최종 평가
        t_e = time.time()
        fin_eval = self.eval_set(full=True, tag=f'step{global_step}')
        self.write_eval_csv(fin_eval, 'eval_per_slice.csv')
        write_json_atomic(fin_eval['summary'], os.path.join(self.cdir, 'eval_summary.json'))
        eval_sec = time.time() - t_e

        # 끝 검사
        md5s = stream_md5(lines_meta, steps)
        end = OrderedDict()
        end['E1'] = dict(ok=global_step == steps, global_step=global_step)
        stream_check = 'unchecked'
        refp = os.path.join(self.out_root, 'ref', 'summary.json')
        if self.cond != 'ref' and os.path.exists(refp):
            with open(refp) as f:
                rs = json.load(f)
            if rs.get('status') == 'completed' and rs.get('steps_target') == steps and rs.get('stream_md5'):
                stream_check = 'match' if rs['stream_md5'] == md5s else 'mismatch'
        end['E2'] = dict(ok=stream_check != 'mismatch', stream_md5=md5s, vs_ref=stream_check)
        # E3 = spec 정의 (V2 가 읽음). clip 은 따로 flag 전용 E3_clip (V2 에 쓰지 않음)
        e3 = self.cnt['nan_skip'] <= 10 and self.cnt['amp_floor'] <= 5
        end['E3'] = dict(ok=e3, flag_only=True, nan_skip=self.cnt['nan_skip'], amp_floor=self.cnt['amp_floor'])
        if not e3:
            self.flags.append(f"E3 nan_skip={self.cnt['nan_skip']} amp_floor={self.cnt['amp_floor']}")
        end['E3_clip'] = dict(ok=self.cnt['clipped'] == 0, flag_only=True, clipped=self.cnt['clipped'], max_norm=self.max_norm)
        if self.cnt['clipped']:
            self.flags.append(f"E3_clip clipped={self.cnt['clipped']} (gn > {self.max_norm:g}; 판정에는 쓰지 않음)")
        nd = near_dead_fracs(self.net, self.groups_stat)
        end['E4'] = dict(ok=all(nd.get(k, 0.0) < 0.5 for k in ('seq', 'unet_deep')), near_dead=nd)
        if not self.has_seq:
            # 0 입력 slice 는 기울기가 정확히 0 이고 분리형 감쇠도 fp32 에서 0 이므로 갱신이 없어야 한다 (값은 0.0 이 정상)
            w = self.net.unet.down_path[0].block[0].weight.detach()[:, :R.SEQ_CH].float().cpu()
            rel = float((w - self.w0_seq_slice).norm() / self.w0_seq_slice.norm())
            end['E5'] = dict(ok=rel <= 1e-4, seq_slice_rel_change=rel, exact_zero=rel == 0.0,
                             meaning='zero-input slice received no update (decoupled decay is a numerical no-op in fp32 here)')
        if self.P.get('seq') == 'scale':
            eps_now = [g['eps'] for g in self.opt.param_groups if g['tag'].startswith('gru')]
            end['E6'] = dict(ok=all(np.float32(e) == np.float32(D_C * 1e-8) for e in eps_now), gru_eps=eps_now)
        if self.gate is not None:
            end['E6'] = dict(ok=bool(torch.isfinite(self.gate).all()), alpha=self.gate.detach().reshape(-1).tolist())
        if self.loss_custom:
            # 설계 결정 16 — 기록(aux)의 같은 영역 항이 학습 항과 비트 단위로 같았는지 (flag 전용: 기록 일관성, 학습 타당성과 무관)
            end['E7'] = dict(ok=self.aux_bad == 0, flag_only=True, aux_mismatch_steps=self.aux_bad, lam_ssim=self.lam,
                             loss_region=self.loss_region)
            if self.aux_bad:
                self.flags.append(f'E7 aux 항 ≠ 학습 항 {self.aux_bad} step')
        for k, v in end.items():
            print(f'[{self.cond}] {"OK " if v["ok"] else ("FLAG" if v.get("flag_only") else "FAIL")} {k} {jdumps(v)[:300]}', flush=True)

        # 저장
        np.save(os.path.join(self.cdir, 'batch_idx.npy'), np.asarray([m['idx'] for m in lines_meta], dtype=np.int32))
        if not a.no_save_end and not self.debug:
            sd = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            if self.gate is not None:
                sd['seq_gate'] = self.gate.detach().cpu()
            tmp = os.path.join(self.cdir, 'end_state.pt.tmp')
            torch.save(sd, tmp)
            os.replace(tmp, os.path.join(self.cdir, 'end_state.pt'))
            del sd
        clip_w = OrderedDict()
        for lo, hi in R_WINDOWS:
            sel = [(g, c) for bp, g, c in self.gn_hist if lo <= bp + 1 <= hi]
            fin_g = [g for g, _ in sel if math.isfinite(g)]
            clip_w[f'{lo}-{hi}'] = dict(n=len(sel), clipped=int(sum(c for _, c in sel)), gn_max=max(fin_g) if fin_g else None)
        bad_end = [k for k, v in end.items() if not v['ok'] and not v.get('flag_only')]
        status = 'completed'
        if bad_end:
            status = 'invalid_stream' if bad_end == ['E2'] else 'invalid'
        extra = OrderedDict(
            steps_done=global_step, batches_consumed=bpos + 1, stream_md5=md5s, stream_check=stream_check, end_checks=end,
            counters=self.cnt, clip_windows=clip_w, n_params=self.n_params, unet_ck_init=repr(self.unet_ck0),
            gru_ck_init=repr(self.gru_ck0) if self.gru_ck0 is not None else None, unet_ck_end=repr(unet_checksum(self.net)),
            eval=fin_eval['summary'], eval_snap=None if snap is None else snap['summary'],
            timing=dict(train_sec=train_sec, probe_and_record_sec=probe_sec, eval_sec=eval_sec,
                        s_per_step_median=float(np.median(step_secs[10:])) if len(step_secs) > 10 else None),
            peak_mem_gb=torch.cuda.max_memory_allocated() / 2 ** 30, env=self.env, decay_note=self.decay_note)
        meta['invariants'] = self.inv
        meta['end_checks'] = end
        meta['end'] = now_iso()
        write_json_atomic(meta, os.path.join(self.cdir, 'meta.json'))
        rc = EXIT_OK if status == 'completed' else EXIT_INVALID
        # 비트 비교 (사전 점검 — 설계 실측 로그 또는 다른 런의 steps.jsonl)
        cmp_path = a.compare_to
        if cmp_path is None and a.preflight and self.cond in PILOT_LOGS:
            cmp_path = os.path.join(PILOT_DIR, PILOT_LOGS[self.cond])
        if cmp_path:
            self.f_steps.flush()
            same, rows = compare_runs(os.path.join(self.cdir, 'steps.jsonl'), cmp_path, use_gn_model=self.gate is not None)
            extra['compare'] = dict(path=cmp_path, all_equal=same, n_steps=len(rows),
                                    first_mismatch=next((r for r in rows if not (r['loss'] and r['gn'] and r['amp_scale']
                                                                                 and r.get('seq_rms', True)
                                                                                 and r.get('gate_absmean', True))), None))
            write_json_atomic(dict(path=cmp_path, all_equal=same, rows=rows), os.path.join(self.cdir, 'compare.json'))
            print(f'[{self.cond}] BIT-COMPARE vs {cmp_path}: {"ALL EQUAL" if same else "MISMATCH"} over {len(rows)} steps', flush=True)
            for r in rows:
                if not (r['loss'] and r['gn'] and r['amp_scale']):
                    print(f'    step {r["step"]} {r["vals"]}', flush=True)
            if not same:
                rc = EXIT_INVALID
        for f in (self.f_steps, self.f_log, self.f_probe):
            f.close()
        self.write_summary(status if not (a.preflight and status == 'completed') else 'preflight_completed', extra)
        print(f'[{self.cond}] DONE status={status} rc={rc} train {train_sec / 60:.1f} min, probes {probe_sec / 60:.1f} min, '
              f'eval {eval_sec / 60:.1f} min, total {(time.time() - self.t_start) / 60:.1f} min', flush=True)
        return rc


# ------------------------------------------------------------------ CPU 단위 시험
OLD_CONDS = ('ref', 'ref2', 'A', 'B', 'C', 'D', 'E', 'EC', 'A2', 'A3')  # 설계 결정 16 이전 조건 (손실 키 없음)


def _ssim_hand(o, t, m=None):
    """u_choh_SSIM 과 독립인 손계산 (numpy float64, scipy correlate 'valid'): Gaussian 11·σ 1.5, L 은 예측 o 에서 자동 추정,
    m 이 있으면 m 을 같은 창으로 평활화해 > 0.5 인 위치만 평균, 없으면 배치 전체 지도 평균."""
    from scipy.signal import correlate2d
    x = np.arange(11) - 5
    g1 = np.exp(-x ** 2 / (2 * 1.5 ** 2))
    g1 = g1 / g1.sum()
    w = np.outer(g1, g1)
    L = (255 if o.max() > 128 else 1) - (-1 if o.min() < -0.5 else 0)
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    f = lambda a: correlate2d(a, w, mode='valid')      # noqa: E731
    maps, sel = [], []
    for b in range(o.shape[0]):
        a, c = o[b, 0], t[b, 0]
        mu1, mu2 = f(a), f(c)
        s1, s2, s12 = f(a * a) - mu1 ** 2, f(c * c) - mu2 ** 2, f(a * c) - mu1 * mu2
        maps.append(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)))
        if m is not None:
            sel.append(f(m[b, 0]) > 0.5)
    S = np.stack(maps)
    return float(S.mean()) if m is None else float(S[np.stack(sel)].mean())


def selftest_loss_factor():
    """설계 결정 16 (CPU): (a) 기존 10 조건(OLD_CONDS)에는 손실 키가 없고 새 조건은 기준 조건 + 두 키뿐, (b) Job 의 loss 속성,
    (c) loss_terms 'mask' 가 기존 train_forward 식과 비트 단위로 같고 'full'·'mask'·supp_terms(GT 지지 영역) 모두 손계산과 같음 (u_choh 의 .cuda() 는
    이 시험 안에서만 R._cuda_noop 으로 우회 — R.install_cpu_shims 와 같은 방식), (d) 결과 폴더가 있으면 meta.json 이 있는 기존
    조건(실행 중 포함)의 config_hash 를 이 판의 CONDS·config_hash 로 재현 (summary.json 이 있으면 그 값과도 대조)."""
    import io
    import contextlib
    from types import SimpleNamespace
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())      # noqa: E731  (u_choh create_window 의 print 숨김)
    res = OrderedDict()
    # (a) 조건 표
    for c in OLD_CONDS:
        assert 'lam_ssim' not in CONDS[c] and 'loss_region' not in CONDS[c], c
    new = OrderedDict()
    for x in ('F', 'L', 'P'):
        for pre, b in LOSS_BASE.items():
            c = pre + x
            P = CONDS[c]
            assert {k: v for k, v in P.items() if k in CONDS[b]} == CONDS[b], c
            assert {k: v for k, v in P.items() if k not in CONDS[b]} == dict(lam_ssim=LOSS_SETTINGS[x][0],
                                                                               loss_region=LOSS_SETTINGS[x][1]), c
            g = P.get('seq') == 'gate'
            assert lr_table(P, g) == lr_table(CONDS[b], g) and preset_groups(P, g) == preset_groups(CONDS[b], g), c
            new[c] = (P['lam_ssim'], P['loss_region'])
            assert LOSS_DERIVED[c] == (b, x), c
    assert CONDS['Aeq'] == dict(CONDS['A'], lam_ssim=LAM, loss_region='mask') and 'Aeq' in PREFLIGHT_ONLY
    assert all(CONDS[f'ref2{x}']['init_seed'] == 2 and CONDS[f'ref2{x}']['model'] == 'unet' for x in 'FLP')
    assert all(LOSS_DERIVED[f'ref2{x}'][0] == 'ref2' for x in 'FLP')    # → build_model 의 I1 = ref2 와 같은 건너뜀 + unet_ck_ref2 대조
    assert set(LOSS_DERIVED) == {c for c in CONDS if 'lam_ssim' in CONDS[c]} and not set(OLD_CONDS) & set(LOSS_DERIVED)
    assert LOSS_SETTINGS['F'] == (LAM, 'full') and LOSS_SETTINGS['L'] == (0.2, 'mask') and LOSS_SETTINGS['P'] == (0.2, 'full')
    res['loss_conds'] = new
    # (b) Job 속성 (I/O 없음 — __init__ 만)
    ns = lambda c: SimpleNamespace(cond=c, preflight=False, steps=None, eval_n=None, probe_n=None, out_dir=OUT_DEFAULT)  # noqa: E731
    for c in OLD_CONDS:
        j = Job(ns(c))
        assert j.loss_custom is False and j.lam == LAM and j.loss_region == 'mask', c
    for c in list(new) + ['Aeq']:
        j = Job(ns(c))
        assert j.loss_custom is True and (j.lam, j.loss_region) == (CONDS[c]['lam_ssim'], CONDS[c]['loss_region']), c
        assert j.loss_desc()['setting'] == ('0' if c == 'Aeq' else c[-1]), (c, j.loss_desc())
    # (b2) 설계 결정 17 — 상한 조건: 기준 조건 + max_norm 키 하나, 기존·손실 요인 조건에는 키 없음, Job.max_norm
    assert set(CLIP_DERIVED) == {'APc', 'A2Pc', 'A3Pc', 'A2c', 'Ac'} == {c for c in CONDS if 'max_norm' in CONDS[c]}, CLIP_DERIVED
    assert 'Ac' in PREFLIGHT_ONLY and PILOT_LOGS['Ac'] == PILOT_LOGS['A']
    for c, b in CLIP_DERIVED.items():
        P = CONDS[c]
        assert {k: v for k, v in P.items() if k != 'max_norm'} == CONDS[b] and P['max_norm'] == MAX_NORM_S4 == 20.0, c
        g = P.get('seq') == 'gate'
        assert lr_table(P, g) == lr_table(CONDS[b], g) and preset_groups(P, g) == preset_groups(CONDS[b], g), c
        assert (c in LOSS_DERIVED) == (b in LOSS_DERIVED) and LOSS_DERIVED.get(c) == LOSS_DERIVED.get(b), c
        jc, jb = Job(ns(c)), Job(ns(b))
        assert jc.max_norm == 20.0 and jb.max_norm == MAX_NORM == 1e3, c
        assert (jc.loss_custom, jc.lam, jc.loss_region) == (jb.loss_custom, jb.lam, jb.loss_region), c
    for c in CONDS:
        if c not in CLIP_DERIVED:
            assert 'max_norm' not in CONDS[c] and Job(ns(c)).max_norm == MAX_NORM, c
    # clip_grad_norm_ 이 gn < max_norm 에서 기울기를 비트 그대로 두는지 (CPU, 이 torch 판) — A·A3·ref·refP 재사용의 근거
    gen = torch.Generator().manual_seed(0)
    ps = [torch.nn.Parameter(torch.zeros(n)) for n in (7, 1000, 33)]
    for target in (0.03, 4.5, 15.8, 19.99):                       # 재사용 조건의 gn 범위(최대 15.8)와 상한 바로 밑
        gs = [torch.randn(p_.shape, generator=gen) for p_ in ps]
        cur = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gs])))
        for p_, g in zip(ps, gs):
            p_.grad = g * (target / cur)
        before = [p_.grad.clone() for p_ in ps]
        tot = float(torch.nn.utils.clip_grad_norm_(ps, max_norm=MAX_NORM_S4))
        assert tot < MAX_NORM_S4, (target, tot)
        assert all(torch.equal(x, p_.grad) for x, p_ in zip(before, ps)), target
    for p_ in ps:                                                 # 상한을 넘으면 실제로 줄어듦
        p_.grad = torch.randn(p_.shape, generator=gen) * 10.0
    tot = float(torch.nn.utils.clip_grad_norm_(ps, max_norm=MAX_NORM_S4))
    new = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(p_.grad) for p_ in ps])))
    assert tot > MAX_NORM_S4 and abs(new - MAX_NORM_S4) < 1e-3, (tot, new)
    res['clip_conds'] = OrderedDict((c, dict(base=b, max_norm=CONDS[c]['max_norm'])) for c, b in CLIP_DERIVED.items())
    # (c) 손실 항
    _cuda = torch.Tensor.cuda
    try:
        torch.Tensor.cuda = R._cuda_noop
        with quiet():
            crit = SSIM()
        g = torch.Generator()
        g.manual_seed(16)
        B, H, W = 2, 48, 44
        lab = torch.rand(B, 1, H, W, generator=g) * 1.5
        bm = torch.zeros(B, 1, H, W)
        bm[0, :, 8:40, 6:36] = 1.0
        bm[1, :, 14:44, 10:30] = 1.0
        out = lab + 0.05 * torch.randn(B, 1, H, W, generator=g) + 0.3 * (1 - bm) * torch.rand(B, 1, H, W, generator=g)
        # ↑ 뇌 밖 오차가 더 큼 → 'full' ≠ 'mask'
        with quiet():
            l1m, lsm = loss_terms(crit, out, lab, bm, 'mask')
            l1f, lsf = loss_terms(crit, out, lab, bm, 'full')
            m_sum = bm.sum().clamp(min=1.0)                                     # 기존 train_forward 식 그대로
            l1_old = ((out - lab).abs() * bm).sum() / m_sum
            ls_old = 1 - crit(out, lab, mask=bm)
        assert torch.equal(l1m, l1_old) and torch.equal(lsm, ls_old) and torch.equal(l1m + LAM * lsm, l1_old + LAM * ls_old)
        o, t, m = out.double().numpy(), lab.double().numpy(), bm.double().numpy()
        hand = dict(l1_mask=float((np.abs(o - t) * m).sum() / m.sum()), l1_full=float(np.abs(o - t).mean()),
                    ssim_loss_mask=1 - _ssim_hand(o, t, m), ssim_loss_full=1 - _ssim_hand(o, t))
        got = dict(l1_mask=float(l1m), l1_full=float(l1f), ssim_loss_mask=float(lsm), ssim_loss_full=float(lsf))
        err = {k: abs(got[k] - hand[k]) for k in hand}                       # fp32 합성곱 대 float64 손계산
        res['loss_terms'] = dict(got=got, hand=hand, abs_err=err, loss_P=float(l1f + LAM_PROF * lsf),
                                 loss_P_hand=hand['l1_full'] + LAM_PROF * hand['ssim_loss_full'])
        assert all(v < 1e-6 for v in err.values()), err
        assert abs(got['l1_full'] - got['l1_mask']) > 1e-3 and abs(got['ssim_loss_full'] - got['ssim_loss_mask']) > 1e-3, got
        # GT 지지 영역 항 (aux 전용): 표본 0 에 정확히 0 인 테두리(위 6 행·오른쪽 5 열)를 둔 GT — supp ≠ full
        lab_z = lab.clone()
        lab_z[0, :, :6, :] = 0.0
        lab_z[0, :, :, -5:] = 0.0
        with quiet():
            l1s, lss, sfr = supp_terms(crit, out, lab_z)
            l1fz, lsfz = loss_terms(crit, out, lab_z, bm, 'full')
        tz, sz = lab_z.double().numpy(), (lab_z != 0).double().numpy()
        hand_s = dict(l1_supp=float((np.abs(o - tz) * sz).sum() / sz.sum()), ssim_loss_supp=1 - _ssim_hand(o, tz, sz),
                      supp_frac=float(sz.mean()))
        got_s = dict(l1_supp=float(l1s), ssim_loss_supp=float(lss), supp_frac=float(sfr))
        err_s = {k: abs(got_s[k] - hand_s[k]) for k in hand_s}
        res['supp_terms'] = dict(got=got_s, hand=hand_s, abs_err=err_s)
        assert all(v < 1e-6 for v in err_s.values()), err_s
        assert abs(got_s['l1_supp'] - float(l1fz)) > 1e-4 and abs(got_s['ssim_loss_supp'] - float(lsfz)) > 1e-4, (got_s, l1fz, lsfz)
        with quiet():                                                           # 0 이 없으면 supp = 전체 (L1 은 같은 평균)
            l1s1, _, sfr1 = supp_terms(crit, out, lab)
        assert float(sfr1) == 1.0 and abs(float(l1s1) - float(l1f)) < 1e-6
        try:
            loss_terms(crit, out, lab, bm, 'brain')
            raise AssertionError('알 수 없는 영역을 받아들임')
        except ValueError:
            pass
    finally:
        torch.Tensor.cuda = _cuda
    # (d) 기존 완료 조건의 config_hash 재현
    inv_p = os.path.join(OUT_DEFAULT, 'invariants.json')
    if os.path.exists(inv_p):
        with open(inv_p) as f:
            inv = json.load(f)
        rh = OrderedDict()
        for c in [c for c in CONDS if c not in CLIP_DERIVED and c not in PREFLIGHT_ONLY]:   # 설계 결정 17: 손실 요인 조건도 재현
            sp, mp = os.path.join(OUT_DEFAULT, c, 'summary.json'), os.path.join(OUT_DEFAULT, c, 'meta.json')
            if not os.path.exists(mp):
                continue
            with open(mp) as f:
                m = json.load(f)                     # 시작 때 쓰임 (실행 중인 조건도 config_hash 가 있음)
            if m.get('debug') or not m.get('config_hash'):
                continue
            s = None
            if os.path.exists(sp):
                with open(sp) as f:
                    s = json.load(f)
            j = Job(ns(c))
            j.E, j.Pp, j.md5, j.env = inv['E256'], inv['P32'], m['md5'], m['env']
            h = j.config_hash()
            rh[c] = h == m['config_hash'] and (s is None or s.get('config_hash') in (None, h))
        res['config_hash_reproduced_old'] = rh
        assert rh and all(rh.values()), rh
    return res


def selftest(with_model, with_forward):
    """GPU 를 쓰지 않는 검사: LR 식·torch 점화식, 그룹 LR(B·C·EC), D 의 eps 보정(Adam 갱신 불변), gate 0 의 출력·기울기,
    부분 표본 결정성, seq_stats 식, 게이트 미분 재구성(작은 GRU), 부호 검정, 비트 비교 파서, 손실 요인 조건(설계 결정 16 —
    selftest_loss_factor: 조건 표·loss_terms 손계산·기존 조건 config_hash 재현), (--selftest-model) 668M 모델의
    체크섬·파라미터 수·no-decay 원소 수, (--selftest-forward) _seq == seq_parts 와 실제 GRU 게이트 재구성(BS 1, fp32)."""
    assert os.environ.get('CUDA_VISIBLE_DEVICES', None) == '', 'selftest 는 CUDA_VISIBLE_DEVICES="" 로'
    res = OrderedDict()
    # 1) LR
    res['lr_base'] = {k: base_lr(k) for k in (0, 999, 1999, 3999)}
    assert abs(base_lr(0) - 2e-4) < 1e-18 and abs(base_lr(3999) / 1.99953e-4 - 1) < 1e-5, res['lr_base']
    mx = lr_recursion_maxrel(4000)
    res['lr_formula_vs_torch_maxrel'] = mx
    assert mx < 1e-9
    PB, PC, PE, PEC = CONDS['B'], CONDS['C'], CONDS['E'], CONDS['EC']
    assert group_lr(0, PB['m_gru'], PB['W_gru']) == base_lr(0) * 0.1
    assert group_lr(5, 1.0, 0) == base_lr(5)                       # m=1, w=1 → base 와 같은 비트
    c = {k: group_lr(k, PC['m_gru'], PC['W_gru']) for k in (0, 99, 499, 998, 999, 3999)}
    res['lr_C'] = c
    assert abs(c[0] / (base_lr(0) / 1000) - 1) < 1e-12 and abs(c[99] / (base_lr(99) * 0.1) - 1) < 1e-12
    assert abs(c[499] / (base_lr(499) * 0.5) - 1) < 1e-12 and c[999] == base_lr(999) and c[3999] == base_lr(3999)
    assert group_lr(10, PEC['m_gate'], 0) == base_lr(10) and group_lr(10, PE['m_gate'], 0) == base_lr(10)
    # 분리형 감쇠가 fp32 가중치를 바꾸지 못함 (finding: wd 3e-5 는 수치상 비활성)
    res['decay_fp32_noop'] = decay_fp32_noop()
    assert res['decay_fp32_noop']['noop'], res['decay_fp32_noop']
    res['lr_table_D'] = lr_table(CONDS['D'], False)
    assert CONDS['D']['eps_gru'] == 0.06 * 1e-8 and np.float32(CONDS['D']['eps_gru']) == np.float32(6e-10)
    # 2) D 의 eps 보정: 기울기 c·g 와 eps c·ε 의 AdamW 갱신 == 기울기 g 와 eps ε (fp64 에서 상대 1e-12)
    torch.manual_seed(0)
    g_seq = [torch.randn(1000, dtype=torch.float64) * 10 ** torch.empty(1000, dtype=torch.float64).uniform_(-11, -5)
             for _ in range(6)]
    upd = []
    for scale, eps in ((1.0, 1e-8), (D_C, D_C * 1e-8), (D_C, 1e-8)):
        p = torch.zeros(1000, dtype=torch.float64, requires_grad=True)
        o = torch.optim.AdamW([p], lr=2e-4, eps=eps, weight_decay=3e-5, foreach=False)
        for g in g_seq:
            p.grad = g * scale
            o.step()
        upd.append(p.detach().clone())
    rel_comp = float((upd[1] - upd[0]).abs().max() / upd[0].abs().max())
    rel_nocomp = float((upd[2] - upd[0]).abs().max() / upd[0].abs().max())
    res['D_eps_comp_rel'] = rel_comp
    res['D_eps_without_comp_rel'] = rel_nocomp
    assert rel_comp < 1e-6 < rel_nocomp, (rel_comp, rel_nocomp)
    # 3) gate 0: 출력은 0 채널과 같고 so 의 기울기는 정확히 0, gate 기울기는 0 이 아님
    gate = nn.Parameter(torch.zeros(1, 20, 1, 1))
    so = torch.randn(2, 20, 8, 8, requires_grad=True)
    xi = torch.randn(2, 32, 8, 8)
    conv = nn.Conv2d(52, 4, 3, padding=1)
    y1 = conv(torch.cat((gate * so, xi), 1))
    y0 = conv(torch.cat((torch.zeros(2, 20, 8, 8), xi), 1))
    res['gate0_out_maxabs'] = float((y1 - y0).abs().max())
    y1.square().sum().backward()
    res['gate0_so_grad_allzero'] = bool((so.grad == 0).all())
    res['gate0_gate_grad_nonzero'] = bool((gate.grad != 0).any())
    assert res['gate0_out_maxabs'] == 0.0 and res['gate0_so_grad_allzero'] and res['gate0_gate_grad_nonzero']
    # 4) 부분 표본: 같은 이름 → 같은 원소 (모델·순서와 무관), 다른 이름 → 다른 원소
    pa = nn.Parameter(torch.randn(1_500_000))
    pb = nn.Parameter(torch.randn(1_500_000))
    t1 = SubTracker([('unet.x', pa), ('unet.y', pb)])
    t2 = SubTracker([('gru_h.z', nn.Parameter(torch.randn(2_000_000))), ('unet.x', pa)])
    assert torch.equal(t1.idx_of('unet.x'), t2.idx_of('unet.x')) and not torch.equal(t1.idx_of('unet.x'), t1.idx_of('unet.y'))
    with torch.no_grad():
        pa.mul_(1.01)
    rc = t1.update()['unet']
    res['subtracker_rel_change'] = rc
    assert 0.003 < rc['rel_change_from_init'] < 0.008
    # 5) seq_stats: 모든 슬라이스가 같은 패턴이면 common 1, 독립 잡음이면 ≈ 1/n
    base_pat = torch.randn(1, 20, 6, 7)
    s_same = base_pat.expand(8, -1, -1, -1).clone()
    st_same, _, _ = seq_stats(s_same, f0=base_pat[0], partners=[None] * 8)
    s_ind = torch.randn(64, 20, 6, 7)
    st_ind, _, _ = seq_stats(s_ind)
    s_wc = torch.randn(16, 20, 6, 1).expand(-1, -1, -1, 7).clone()
    st_wc, _, _ = seq_stats(s_wc)
    res['seq_stats'] = dict(same_common=st_same['common_frac'], same_dep=st_same['dep'], indep_common=st_ind['common_frac'],
                            indep_common_unbiased=st_ind['common_frac_unbiased'], wconst_of_W_constant=st_wc['wconst_D'])
    assert abs(st_same['common_frac'] - 1) < 1e-6 and st_same['dep'] < 1e-6      # fp32 평균 반올림 ~1e-8
    assert abs(st_ind['common_frac'] - 1 / 64) < 0.01 and abs(st_ind['common_frac_unbiased']) < 0.02
    assert abs(st_wc['wconst_D'] - 1) < 1e-6
    # 6) 게이트 미분 재구성 (작은 양방향 GRU, fp32 CPU) — recon_err ≈ fp32 반올림
    gru = nn.GRU(12, 5, 1, batch_first=True, bidirectional=True)
    X = torch.randn(3, 9, 12)
    with torch.no_grad():
        out, _ = gru(X)
    gd = gru_gate_derivs(gru, X, out, 5, mimic_half=False)
    res['gate_derivs_small_gru'] = gd
    assert gd['f']['recon_err'] < 1e-5 and gd['b']['recon_err'] < 1e-5
    assert 0 < gd['f']['dz'] <= 0.25 and 0 < gd['f']['dn'] <= 1
    # 7) 부호 검정·drop_stats
    assert abs(sign_test_p(16, 16) - 1.0) < 1e-12 and sign_test_p(20, 0) < 2e-6
    ds = drop_stats([1.0, 1.0, 1.0, 1.0], [0.9, None, 0.8, 1.1])
    assert ds['n'] == 3 and ds['n_pos'] == 2 and ds['n_neg'] == 1
    # 8) 비트 비교 파서 (설계 실측 로그 vs 자기 자신)
    pl = os.path.join(PILOT_DIR, 'meas_B.log')
    if os.path.exists(pl):
        d = parse_pilot_log(pl)
        res['pilot_B_steps'] = len(d)
        assert d[1]['gn'] == float('inf') and d[1]['loss'] == 0.5491266250610352 and d[2]['amp_scale'] == 32768.0
    # 9) stream_md5: NaN-skip 으로 더 읽은 배치는 제외
    lm = [dict(bpos=b, idx=[b, b + 1], data_ck=float(b)) for b in range(5)]
    assert stream_md5(lm, 4) == stream_md5(lm[:4], 4) != stream_md5(lm, 5)
    # 10) md5 가드·import 경로
    res['md5'] = {rel: md5_file(os.path.join(REPO, rel)) == want for rel, want in MD5.items()}
    res['import_paths'] = {rel: os.path.realpath(m.__file__) == os.path.realpath(os.path.join(REPO, rel)) for m, rel in _MOD_FILES}
    assert all(res['md5'].values()) and all(res['import_paths'].values()), (res['md5'], res['import_paths'])
    pod = [json.load(open(p))['lambda_ssim'] for p in POD_RECIPES]
    res['lambda_fp32_equal_pod'] = bool(np.float32(LAM) == np.float32(pod[0])) and pod[0] == pod[1]
    assert res['lambda_fp32_equal_pod']
    res.update(selftest_loss_factor())
    if with_model:
        t0 = time.time()
        R.seed_all(1)
        base = R.build_model('gru', torch.device('cpu'))
        np_st, py_st = np.random.get_state(), random.getstate()
        with torch.random.fork_rng(devices=[]):
            R.seed_all(1)
            ref_u = R.build_model('unet', torch.device('cpu'))
        np.random.set_state(np_st)
        random.setstate(py_st)
        base.unet.load_state_dict(ref_u.unet.state_dict())
        gru_precision.apply(base, ('gru_h',))
        uck = unet_checksum(base)
        gck = gru_checksum(base)
        params = [p for n, p in base.named_parameters() if n.startswith(('gru_h.', 'gru_v.'))]
        gck_plain = float(sum((p.double() * (i + 1)).sum() for i, p in enumerate(params)))
        uck_ref = unet_checksum(ref_u)
        decay, no_decay, _, _ = R.split_decay(base)
        R.seed_all(2)
        ref2 = R.build_model('unet', torch.device('cpu'))
        res['model'] = dict(unet_ck='%.10e' % uck, unet_ck_unet_model='%.10e' % uck_ref, gru_ck='%.10e' % gck,
                            gru_ck_unchunked='%.10e' % gck_plain, gru_ck_absdiff=abs(gck - gck_plain),
                            unet_ck_ref2='%.10e' % unet_checksum(ref2),
                            n_gru_model=sum(p.numel() for p in base.parameters()),
                            n_unet_model=sum(p.numel() for p in ref_u.parameters()),
                            gru_tensors=len(params), gru_numel=sum(p.numel() for p in params),
                            no_decay_numel=sum(p.numel() for p in no_decay), build_sec=time.time() - t0)
        print('selftest model', jdumps(res['model']), flush=True)
        assert res['model']['unet_ck'] == EXPECT['unet_ck'] == res['model']['unet_ck_unet_model']
        assert res['model']['gru_ck'] == EXPECT['gru_ck'] == res['model']['gru_ck_unchunked']
        assert res['model']['gru_tensors'] == EXPECT['gru_tensors']
        assert res['model']['no_decay_numel'] == EXPECT['no_decay_gru_model']
        # 그룹 구성 (실제 optimizer 와 같은 코드 경로의 축소판)
        gru_ids = {id(p) for p in params}
        tags = []
        for tag, wd, m, W, eps in preset_groups(CONDS['E'], True):
            tags.append(tag)
        assert tags == ['unet_decay', 'gru_decay', 'unet_nodecay', 'gru_nodecay', 'gate']
        n_cov = sum(p.numel() for p in decay + no_decay)
        assert n_cov == res['model']['n_gru_model'] and len({id(p) for p in decay + no_decay}) == len(decay) + len(no_decay)
        assert sum(1 for p in decay + no_decay if id(p) in gru_ids) == 16
        if with_forward:
            t0 = time.time()
            torch.set_grad_enabled(False)
            x = torch.randn(1, 32, 384, 384) * 0.05
            so_a = base._seq(x)
            out_h, oh, out_v, so_b = seq_parts(base, x)
            res['forward'] = dict(seq_parts_eq=bool(torch.equal(so_a, so_b)), gru_h_dtype=str(out_h.dtype),
                                  gate_h=gru_gate_derivs(base.gru_h, x.reshape([1, 384, -1]), out_h, base.num_out1, False),
                                  gate_v=gru_gate_derivs(base.gru_v, oh, out_v, base.num_out2, False),
                                  sec=time.time() - t0)
            torch.set_grad_enabled(True)
            print('selftest forward', jdumps(res['forward']), flush=True)
            assert res['forward']['seq_parts_eq']
            assert res['forward']['gate_h']['f']['recon_err'] < 1e-4 and res['forward']['gate_v']['b']['recon_err'] < 1e-4
    print('SELFTEST OK', jdumps(res, indent=1), flush=True)
    return 0


def parse_args():
    ap = argparse.ArgumentParser(description='d-ETER 최적화 교란 스모크 (조건 하나 = 프로세스 하나)')
    ap.add_argument('--cond', choices=list(CONDS))
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--skip-done', action='store_true')
    ap.add_argument('--no-save-end', action='store_true')
    ap.add_argument('--steps', type=int, default=None, help='디버그 전용 (config_hash 가 바뀜, scratch 출력만)')
    ap.add_argument('--eval-n', type=int, default=None, help='디버그 전용')
    ap.add_argument('--probe-n', type=int, default=None, help='디버그 전용')
    ap.add_argument('--preflight', action='store_true', help='짧은 사전 점검 (기본 30 step, eval 16, probe 8; 설계 실측 로그와 비트 비교)')
    ap.add_argument('--compare-to', default=None, help='설계 실측 로그(meas*.log) 또는 다른 런의 steps.jsonl — step 별 비트 비교')
    ap.add_argument('--selftest', action='store_true', help='CPU 단위 시험 (CUDA_VISIBLE_DEVICES="")')
    ap.add_argument('--selftest-model', action='store_true', help='+ 668M 모델 CPU 생성 (체크섬·파라미터 수)')
    ap.add_argument('--selftest-forward', action='store_true', help='+ CPU forward 1 회 (BS 1, 느림)')
    return ap.parse_args()


GPU_LOCK = '/tmp/smoke_deter_opt.gpu.lock'      # 이 하네스의 GPU job 은 동시에 하나 (런처 pgrep 가드의 보강 — python -u 등도 잡음)


def take_gpu_lock():
    import fcntl
    f = open(GPU_LOCK, 'a')
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.seek(0)
    f.truncate()
    f.write(f'{os.getpid()} {now_iso()} {" ".join(sys.argv[1:])}\n')
    f.flush()
    return f                                     # 프로세스가 끝날 때까지 열어 둔다 (종료 시 커널이 해제)


def main():
    args = parse_args()
    if args.selftest:
        return selftest(args.selftest_model or args.selftest_forward, args.selftest_forward)
    if args.cond is None:
        print('[refuse] --cond 필요', flush=True)
        return EXIT_REFUSE
    if args.cond in PREFLIGHT_ONLY and not args.preflight:
        print(f'[refuse] {args.cond} 는 사전 점검 전용 (--preflight, scratch 출력 — 설계 결정 16·17)', flush=True)
        return EXIT_REFUSE
    lock = take_gpu_lock()
    if lock is None:
        try:
            holder = open(GPU_LOCK).read().strip()
        except OSError:
            holder = '?'
        print(f'[refuse] 다른 smoke_deter_opt GPU job 이 실행 중 ({GPU_LOCK}: {holder})', flush=True)
        return EXIT_REFUSE
    job = Job(args)
    if job.debug:
        real = os.path.realpath(OUT_DEFAULT)
        out = os.path.realpath(job.out_root)
        if out == real or out.startswith(real + os.sep):
            print(f'[refuse] 디버그·사전 점검 출력은 scratch 로만 (받은 값 {job.out_root})', flush=True)
            return EXIT_REFUSE
    elif args.compare_to:
        print('[refuse] --compare-to 는 사전 점검(--preflight)에서만', flush=True)
        return EXIT_REFUSE
    return job.run()


if __name__ == '__main__':
    sys.exit(main())
