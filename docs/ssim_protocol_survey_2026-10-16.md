# fastMRI 공식 SSIM 과 문헌별 SSIM 계산 방식 조사 (2026-10-16)

요청(10-16 사용자): "fastMRI 공식 GitHub 의 SSIM 평가 방식을 참고하고, 다른 논문들은 어떻게 평가했는지 확인". 조사 = 워크플로 3 갈래(공식 코드·논문 / 문헌 / 로컬 코드) + 갈래마다 독립 검증 + 종합. 검증에서 확인(89 건) 또는 정정(10 건)된 항목만 썼고, 정정 항목은 고친 내용으로 적었다. 열람하지 못한 것은 (d) 에 따로 모았다. 앞선 비교 문서 = `docs/ssim_impl_compare_2026-10-15.md`.

이 문서에서 쓰는 약어는 다음과 같다.
- **skimage 기본**: 균일 7×7 창, K1 0.01·K2 0.03, 표본 공분산 NP/(NP−1), 평균 낼 때 가장자리 3 px 제외.
- **A**: 아래 표의 fastMRI 공식 `evaluate.py` 행과 같다는 뜻.
- **B**: 아래 표의 fastMRI Lightning 검증 행과 같다는 뜻.

## (a) SSIM 계산 방식 비교표

### fastMRI 공식 코드 · 데이터셋 논문 · 챌린지

| 계산 방식 | data_range | 마스크/전경 | crop/크기 | 창·공분산 | 집계(슬라이스/볼륨) | 구현 | 출처 |
|---|---|---|---|---|---|---|---|
| **A. fastMRI 공식 `evaluate.py`** (GitHub main HEAD 91f2df4711 = 설치본 0.3.0, diff 동일) | crop 뒤 정답 볼륨 전체의 최댓값 `gt.max()`. min 은 빼지 않음 | 없음. 배경을 포함한 전체 영상 | 정답과 재구성을 모두 (W, W) 로 가운데 자름(W = 정답 폭). multicoil 대상은 `reconstruction_rss` | skimage 기본 | 슬라이스 SSIM → 볼륨 평균 → 볼륨 간 평균(runstats). 출력 형식 mean ± 2·SD | skimage `structural_similarity`. 테스트는 0.19.3 고정, 설치본 0.25.2 와 기본값 동일 | `fastmri/evaluate.py:40-57, 91-98, 114-122, 165-167`; skimage 0.25.2 `_structural_similarity.py:161-183, 272-275`; 0.19.3 `:143-165, 205, 233-236`; `setup.cfg:51, 85` |
| **B. fastMRI Lightning 검증** (`MriModule`, 학습 중 검증) | h5 `attrs['max']`. 로컬 val 462/464 에서 RSS 볼륨 최댓값과 같음 | 없음 | `center_crop_to_smallest` (정답 크기) | skimage 기본 | 슬라이스 → 볼륨 평균 → 볼륨 평균 | `evaluate.ssim` 재사용 | `pl_modules/mri_module.py:128, 138-140, 205-207`; `varnet_module.py:114`; `data/transforms.py:336, 462` |
| (참고) fastMRI 학습 손실 `SSIMLoss` | 샘플별 `attrs['max']` | 없음 | — | 균일 7×7, k 0.01/0.03, 표본 공분산. valid conv 라 테두리 자르기 없음. 1 − S.mean() | — | torch | `losses.py:18-62`; `varnet_module.py:93, 102-104` |
| **Zbontar 2018** (fastMRI 데이터셋 논문, arXiv 1811.08839v2) | 타깃 볼륨 최댓값 L = max(v) [Q1] | §5.3 에 언급 없음 | §4.7 은 중앙 320×320 [Q2]. 이 문장은 knee 전용 v1 판에도 그대로 있음. App. A.1 은 320 crop 을 knee 에 한정 [Q3] | 7×7, k1 0.01·k2 0.03. 가우시안 언급 없음 | §5.3 에 슬라이스 평균 방식 없음. NMSE·PSNR 은 볼륨 단위 [Q4] | 명시 없음 | §5.3 식 (9) p.14; §4.7 p.9; App. A.1 p.33; §5.1–5.2 p.14 |
| **2019 knee 챌린지** (Knoll, MRM 2020, arXiv 2001.02518v1) | 본문에 없음 | 언급 없음. 배경 잡음도 논의하지 않음 | 중앙 320×320 [Q5] | 본문에 없음 | 리더보드 순위는 R=8 SSIM [Q6]. test 전체와 대조도별로 계산 | 본문에 없음 | §2.1 p.3-4; §2.3 p.5 |
| **2020 brain 챌린지** (Muckley, TMI 2021, arXiv 2012.06318v3) | 논문에 명시 없음. "저장소 평가 스크립트 + skimage 기본값" [Q7] 이라고만 함. A 와 같다고 보는 근거는 이 문장과 Discussion 293 [Q8] | 공식 순위는 마스크 없이 매김. 사후 분석에서만 제출 영상과 정답을 함께 마스킹 [Q9]. 마스크 알고리즘은 공개하지 않음 | 브레인은 재구성 행렬 크기가 다양함 [Q10] | skimage 기본 | Transfer 트랙 AVG 는 전 볼륨 단순 평균 [Q11]. 표 III 은 평균과 2.5·97.5% 분위 95% CI | fastMRI 저장소 스크립트 | §II-B·§II-C p.3; §III-B p.6; Fig. 4 p.7; §IV-B p.9; Supp. §2·Fig. S1 p.13-15 |

### fastMRI 를 쓴 재구성 논문

| 논문 | data_range | 마스크/전경 | crop/크기 | 창·공분산 | 집계 | 구현 | 출처 |
|---|---|---|---|---|---|---|---|
| **E2E-VarNet** (Sriram 2020, arXiv 2004.06688) | 본문에 없음. 코드는 B | B | B | B | test 표 3 은 리더보드 값 [Q12]. 브레인 4× SSIM 0.959 | fastMRI `VarNetModule`(MriModule 상속). 학습 손실 −SSIM | §3; 표 3 p.8. "리더보드 = `evaluate.py`" 는 Discussion 293 에 근거한 추론 |
| **HUMUS-Net** (NeurIPS 2022, arXiv 2203.08213). knee multi-coil 8× 만 실험 | `attrs['max']` | 없음 | 입력을 384×384 로 crop/pad [Q13]. 평가는 `center_crop_to_smallest` | B | B | `HUMUSNetModule(MriModule)` | §5.1 p.7; repo `pl_modules/humus_module.py:7, 10, 80, 90`; `data/data_transforms.py:125` |
| **PromptMR** (arXiv 2309.13839). knee val 199 개를 99/100 으로 나눔 [Q14] | A (코드 복사본) | 없음 | (W, W) | A | A | `promptmr_examples/fastmri/evaluate.py`. 원본과 다른 점은 가속 배율 필터(`--test-path`)뿐 | App. B·표 B.1; repo `evaluate.py:40-57, 113-121` |
| **PromptMR+** (ECCV 2024) 보충자료 + 코드 `external/PromptMR-plus@934eeda` | 코드는 `attrs['max']` | fastMRI 에서는 없음. CMRxRecon 에서는 공식 코드에 배경 crop 을 추가했고 이유는 적지 않음 [Q16] | fastMRI knee 는 공식 코드로 320×320 crop [Q15]. 코드상 검증은 RSS 크기로 `center_crop_to_smallest`. 입력은 384 reflect pad | A | 슬라이스 → 볼륨 → 볼륨 평균 | fastMRI 코드 복사(`mri_utils/utils.py:35-52`) | 보충 Sec. C p.5-6; `pl_modules/mri_module.py:166, 177, 231-268`; `data/transforms.py:446, 483-485`; README 브레인 공식 test 4× 0.9615 [Q22] |
| **ReconFormer** (arXiv 2201.09376). knee single-coil PD, 320×320 | 정답 볼륨 최댓값 | 없음 | (W, W) | skimage 기본 + `multichannel=True`. 이 인수는 0.25.2 에서 조용히 무시되어 결과가 3D 호출과 같음 | 볼륨 → `np.mean` | 자체 `utils/evaluate.py` | repo `utils/evaluate.py:34-38, 97-98`; `models/evaluation.py:56-62` |
| **MambaRecon** (arXiv 2409.12401). fastMRI 브레인 부분집합 100/10/40 명, 5 코일 압축. 표 1 만 fastMRI | 1.0. 영상마다 자기 최댓값으로 정규화한 뒤 계산 | 없음 | 미확인 | Gaussian σ 1.5, 모분산 [Q17] | 슬라이스 단위, SSIM(%) | 자체 코드 | repo `code/utils/psnr.py:18-27`; `code/val_2D.py:14-25`; §5.1 |
| **LMO** (CVPR 2025). fastMRI 는 knee single-channel. 코드는 비공식 저장소 `weili419/LMO` (논문 링크 `ZhengJianwei2/LMO` 에는 README 뿐) | 검증 단계는 정답 슬라이스 최댓값. 테스트 단계는 지정하지 않음. 구판 skimage 는 이때 float 기본값 L=2 [Q23], 0.25.2 는 오류 | 없음 | 본문에 없음 | skimage 기본 | 슬라이스 | 자체 코드 | `mc_rec_train_and_test_ac3m.py:144-146, 251-252`; 논문 §4 |
| **MambaMIR** (arXiv 2402.18451). knee PD, 사례당 중앙 20 슬라이스 | 본문에 없음 | 본문에 없음 | 320×320 중앙 crop [Q18] | 본문에 없음 | 본문에 없음 | 저장소 404 | §3.1 p.6 |

### 다른 데이터셋의 공식 평가 · ETER-net 계열 · 마스크 적용 문헌

| 계산 방식 | data_range | 마스크/전경 | crop/크기 | 창·공분산 | 집계 | 구현 | 출처 |
|---|---|---|---|---|---|---|---|
| Calgary-Campinas 공식 | 슬라이스마다 max(ref, rec) − min(ref, rec) | 없음 | — | skimage 기본 | 슬라이스별 배열 반환 | `mrrec/metrics.py` | `metrics.py:22-24` |
| CMRxRecon 공식 | 1. 정답과 예측을 각자 자기 최댓값으로 나눈 뒤 계산 | 없음 | — | skimage 기본 | 2D 영상 단위 | `Evaluation/Evaluation.py` | `:35-41, 54, 64` |
| ETER-net Sensors 2022 (자체 3T 데이터, fastMRI 아님) | 언급 없음. 라벨은 [0,1] 로 정규화 | 언급 없음 | — | 식 (10) 은 전역 평균·분산 형태이고 창 언급 없음 [Q19] | 표 1 은 테스트 48 영상 | 언급 없음 | §3.1 p.5; §3.3 식 (10) pp.7-8; 표 1 p.12 |
| 교수님 테스트 스크립트 (해당 블록은 파일에서 비활성: `:394 flag_evalu=False`) | max(정답 최댓값, 예측 최댓값) | 없음 | 영상 전체 | skimage 기본 | 슬라이스. 집계 코드 없음 | skimage | `scripts_legacy/myTEST_ViT_ETER_regular_251008.py:394, 403-406` |
| Jalal 2021 (NeurIPS, arXiv 2108.01368). 정답은 MVUE | 본문에서 확인 못 함 | 마스크 SSIM 을 따로 보고. 완전 샘플 MVUE 최댓값의 0.05 배 미만 좌표를 0 으로 바꿈 [Q20]. 두 영상 모두에 적용하는지는 명시 없음 | — | — | — | — | App. A, Fig. 6-7 |
| P-VDAMP (TMI 2022, arXiv 2203.04180) | 확인 못 함 | 기준 영상 최댓값의 5% 미만 화소를 마스크 [Q21] | — | — | — | — | §III-D |
| Shimron 2024 (arXiv 2411.06704) | 확인 못 함 | 기준 영상으로 만든 이진 마스크로 배경 제외 [Q24] | — | — | 슬라이스별 SSIM 의 평균 | — | §2.3.4 p.7 |

### 우리 계산 방식

| 계산 방식 | data_range | 마스크/전경 | crop/크기 | 창·공분산 | 집계 | 구현 | 출처 |
|---|---|---|---|---|---|---|---|
| **우리 (v8·v8fix 표)** | 슬라이스마다 뇌 마스크 안 정답의 max − min | 뇌 마스크(Otsu×0.4 + 최대 연결 성분) 안에서 SSIM 맵 평균. `full=True` 맵을 쓰므로 3 px 테두리를 빼지 않음 | 384×384 로 crop_or_pad. 320² 볼륨은 0 테두리가 생김 | skimage 기본 | 슬라이스 → 볼륨 평균 → mean ± SD (ddof=1) | skimage | `v8_eter_pure/eval_paired_v8_nodc.py:50-55`; `dataloaders/dataloader_h5_v5.py:44-72, 196-198, 243-263`; `paper/make_tables.py:289-299` |
| 우리 공개 모델 기준선 행 (`baselines_384_full`) | 위와 같음 | 위와 같음 | 384² 재-FFT, 16 코일 절단, 같은 R4 마스크 | 같음 | 같음. 여기에 슬라이스별 최소제곱 강도 배율 보정 추가 | 같음 | `eval_baselines_full.py:4-7, 78-79, 188-194, 394`; `visualize_multimodel_compare.py:81-110` |

저장소 안에 `fastmri.evaluate` 를 쓰거나 공식 SSIM 을 계산하는 스크립트는 하나도 없다(grep 결과 0 건). `eval_paired_baselines.py` 의 "native" 행도 입력과 crop 만 공식에 맞췄고, 지표는 여전히 마스크 지표다(299 슬라이스 표본).

### 권고만 담은 문헌 (계산 방식이 아니라 권고)
- **Gourdeau 2022** (Med Phys, Crossref 초록만 읽음): 조직이 있는 부피에서 평균하고 동적 범위를 적절히 정하라고 권한다 [Q25].
- **Breger 2024** (arXiv 2405.19097): PSNR/SSIM/LPIPS 가 더 낫다고 판정한 fastMRI 브레인 FLAIR 영상이 실제로는 흐림과 링잉이 더 심했다(정정: LPIPS 는 낮을수록 좋은 지표다). 넓은 검은 배경이 결과를 좌우한다고 경고한다 [Q26].
- **Dohmen 2024** (arXiv 2405.08431): L 이 크면 SSIM 이 1 에 가까워진다고 지적하고, 범위는 적어도 두 영상, 가능하면 데이터셋 전체에서 정하라고 권한다 [Q27]. 이웃 화소가 필요한 SSIM 에 마스크를 적용하는 방법은 향후 과제로 남겼다.

### 인용문 (원문, 25 단어 이내)
- [Q1] Zbontar §5.3: "define L as the maximum value of the target volume, L = max(v)."
- [Q2] Zbontar §4.7: "All ground truth images are cropped to the central 320 × 320 pixel region"
- [Q3] Zbontar App. A.1: "For knee images, height and width have been cropped to 320 x 320."
- [Q4] Zbontar §5.1: "we report NMSE values computed and normalized over full image volumes rather than individual slices"
- [Q5] Knoll §2.1: "cropping the reconstructed images to the central 320 × 320 pixel region."
- [Q6] Knoll §2.3: "Submissions were ranked by SSIM of R=8 undersampling."
- [Q7] Muckley §II-C: "so as a result we used the default parameters in scikit-image"
- [Q8] fastMRI Discussion #293: "The website essentially runs the same code as in evaluate.py"
- [Q9] Muckley §III-B: "we masked both the submission and the reference RSS ground truth before calculating SSIM" / "Our custom mask would not have changed finalist rankings otherwise."
- [Q10] Muckley §II-B: "Unlike the knee challenge, this data set exhibits a wide variety of reconstruction matrix sizes."
- [Q11] Muckley Fig. 4 캡션: "a simple average across all volumes in the Transfer track."
- [Q12] E2E-VarNet 표 3: "Results on the test data compared with the best models on the fastMRI leaderboard"
- [Q13] HUMUS-Net §5.1: "For HUMUS-Net, we center crop and pad inputs to 384 × 384."
- [Q14] PromptMR App. B: "we divided the original 199 validation cases into 99 for validation and 100 for testing."
- [Q15] PromptMR+ 보충 Sec. C: "using the code in the official fastMRI repository, with images center-cropped to 320 × 320."
- [Q16] PromptMR+ 보충 Sec. C: "with the modification that we cropped the background of the image."
- [Q17] MambaRecon `psnr.py`: "sigma=1.5, gaussian_weights=True, use_sample_covariance=False, data_range=1.0"
- [Q18] MambaMIR §3.1: "all slices were centre-cropped to 320 × 320"
- [Q19] ETER-net Sensors §3.3: "µx and µy are the means of x and y ... small constants are included such as c1 and c2."
- [Q20] Jalal App. A: "zeros out all coordinates whose absolute value is smaller than 0.05 times the maximum absolute value in the fully-sampled MVUE."
- [Q21] P-VDAMP §III-D: "all pixels with absolute value less than 5% of the maximum of the reference were masked."
- [Q22] PromptMR+ README: "test result is on the official test set. (Not reported in the paper)"
- [Q23] skimage 0.25.2 docstring: "this estimate yields a result double the value of the desired range"
- [Q24] Shimron §2.3.4: "excluded from metric calculations by calculating a binary mask on the fully-sampled reference image."
- [Q25] Gourdeau 초록: "reporting the average over the volume of the image containing tissue and appropriately defining the dynamic range."
- [Q26] Breger 권고 1: "being aware that huge black background areas influence the outcome."
- [Q27] Dohmen: "the data range should be derived at least from both images if not from the whole dataset."
- Muckley §IV-B: "The influence of background noise on SSIM scores is substantial."
- Muckley §III-B: "set the background air to nearly 0s, which led to a clinically irrelevant SSIM loss of approximately 0.3"

## (b) 공식 방식과 다른 점, 그리고 공개 수치와 비교할 때의 의미

### 지표 정의에서 다른 점

| 항목 | fastMRI 공식 (`evaluate.py`) | 우리 | 영향 방향과 근거 |
|---|---|---|---|
| data_range | crop 뒤 정답 볼륨 최댓값. 한 볼륨의 모든 슬라이스가 같은 값을 씀 | 슬라이스마다 마스크 안 정답의 max − min | 마스크 안 최댓값은 볼륨 최댓값보다 클 수 없고 여기서 min 을 다시 빼므로, 우리 L 은 항상 공식 L 이하다. 그만큼 C1=(K1·L)²·C2=(K2·L)² 가 작아져, 같은 오차라도 SSIM 이 낮게 나오는 방향이다(Dohmen §2.3.1: L 이 크면 SSIM 이 1 에 가까워짐). 로컬 파일럿에서는 범위만 바꿔도 0.9216 → 0.9557 (+0.034) |
| 평균 영역 | 전체 영상(배경 공기와 RSS 잡음 포함). 테두리 3 px 는 뺌 | 뇌 마스크 안. 테두리 3 px 를 빼지 않음 | Muckley §IV-B 는 배경 잡음의 영향을 "substantial" 이라고 썼고, 배경을 0 으로 낸 팀은 약 0.3 을 잃었다. 로컬 파일럿에서는 마스크만 없애면 −0.268. 테두리 차이는 측정하지 않음 |
| 영상 크기 | 원래 RSS 크기에서 W×W | 384×384 (모자라면 0 으로 채움) | 로컬 val 464 개는 320² 285, 384² 157, 320×270 11, 320×260 7, 276² 2, 213² 1, 384×312 1 이다. 비정사각 19 개는 모두 H>W 라 위아래 행만 잘린다. 마스크 지표에서는 0 테두리가 마스크 밖이라 영향이 거의 없다고 본다(측정하지 않음). 마스크 없는 지표를 384 틀에서 그대로 계산하면 공식값과 다르다 |
| 집계 | 슬라이스 → 볼륨 → 볼륨 평균. 출력은 mean ± 2·SD | 같은 구조. 표는 mean ± SD | 평균의 정의는 같다. 같은 것은 이 항목뿐이다. ± 의 뜻은 따로 적어야 한다 |
| PSNR·NMSE | 볼륨 단위, 범위는 볼륨 최댓값, 마스크 없음 | 슬라이스 단위, 마스크 안 | 정의 자체가 다르다 |

파일럿 수치의 출처는 `docs/ssim_impl_compare_2026-10-15.md:57-85` 다(48 슬라이스, U-Net 단독 v8fix ep48). 이 문서는 `:59` 에서 "경향 확인용이다. 인용하지 않는다" 고 밝혀 두었다.

### 지표 밖에서 다른 점
- **입력**: 앞 16 코일만 쓰고, 검증 마스크는 고정 equispaced R4 다(offset = R − 1). `dataloader_h5_v5.py:79-93, 228-232`
- **데이터**: 로컬 val 464 파일은 공식 val 1,378 의 일부다. pod 에는 val 0~2 1,378 볼륨과 test_full 558 볼륨이 있다. 공개 수치는 대부분 리더보드 test 값이다.
- **학습 데이터**: 리더보드 U-Net·VarNet 은 train+val 로 학습했다(`fastmri_examples/varnet/README.md:92-93`). E2E-VarNet 본문에도 "we trained on both the training and validation sets" 라고 적혀 있다.

### 의미
1. 지금 표의 뇌 마스크 SSIM 은 공개 fastMRI 수치와 같은 줄에 놓을 수 없다. 공개 수치는 Zbontar 표 12 의 브레인 U-Net 4× test 0.9446, E2E-VarNet 표 3 의 브레인 4× 0.959, PromptMR+ README 의 공식 test 4× 0.9615 다. 지표 정의, 데이터 분할, 입력 조건이 모두 다르다.
2. 공식 정의로 다시 계산해도 맞춰지는 것은 지표 정의뿐이다. 분할(val 과 test), 입력(16 코일, 고정 마스크), 학습 데이터(train+val) 차이는 남는다. 그래서 순위 비교가 아니라 참고 수치로만 쓸 수 있고, 이는 CLAUDE.md 에 이미 정해 둔 방침과 같다.
3. 우리 공개 모델 행(U-Net† 0.8971 / E2E-VarNet† 0.9181 / PromptMR+ 0.9417, CLAUDE.md 기재)과 각 논문 수치의 차이를 지표 차이 하나로 설명할 수 없다. 이 행에는 최소제곱 강도 배율 보정과 384 재-FFT·16 코일 입력이 함께 들어 있다.
4. 연구 안의 비교(각 비교 모델 vs 기준 모델 ETER-net)는 모든 모델이 같은 정의를 쓰므로 정의 면에서는 공정하다. 그러나 정의를 바꿔도 순위가 유지된다는 근거는 아직 없다.
   - 로컬 파일럿(인용 금지)에서 마스크 없는 공식 정의를 쓰면 U-Net 단독 0.9072, ETER-net(ep36 중간 ckpt) 0.8631 이었다.
   - 같은 문서는 이 차이의 약 84% 를 뇌 밖 FOV 배경 탓으로 보았다(`:85, :99`).
   - Muckley 의 챌린지에서는 마스킹해도 결선 순위가 한 곳(Transfer 트랙의 ATB)만 바뀌었다. 하지만 이 결과를 우리 모델에 그대로 옮길 근거는 없다.
5. 정의 차이가 만드는 변화가 모델 간 차이보다 훨씬 크다. 2020 챌린지 4× 트랙 상위 4팀은 0.955~0.965 에 몰려 있었다(Muckley §III-B). 반면 파일럿에서는 범위 하나로 +0.034, 마스크 하나로 −0.268 이 움직였다. 정의를 밝히지 않은 SSIM 숫자는 해석할 수 없다.
6. 방법 절에 브레인 공식 평가를 "320×320 crop" 으로 적으면 틀린다. 320 crop 은 knee 에 대한 서술이고(Zbontar App. A.1), 공식 코드는 W×W 로 자르며, 브레인 RSS 크기는 볼륨마다 다르다.

## (c) 권고

### 결론: 뇌 마스크 SSIM 을 대체하지 말고 공식 SSIM 을 함께 보고
- **주 지표**: 지금의 뇌 마스크 SSIM 을 유지하고, 방법 절에 data_range·마스크·평균 영역의 정의를 적는다. 근거로 들 수 있는 문헌은 Muckley 의 사후 마스킹, Shimron·P-VDAMP·Jalal 의 마스크 평가, Gourdeau 초록과 Breger 의 권고다.
- **보조 지표**: fastMRI 공식 `evaluate.py` 정의의 SSIM 을, 같은 정의의 볼륨 단위 PSNR·NMSE 와 함께 한 열 또는 부록 표로 낸다. fastMRI 코드, 두 챌린지, E2E-VarNet·HUMUS-Net·PromptMR·PromptMR+ 코드가 모두 이 정의를 쓴다. 그래서 독자가 우리 수치의 위치를 가늠할 수 있고, 두 정의에서 순위가 같은지도 함께 보여 줄 수 있다.
- **같이 정리할 점**: 우리 data_range(슬라이스 단위, 정답만, 마스크 안 max − min)는 Dohmen 의 권고 [Q27] 와 다르다. 방법 절에 이렇게 고른 이유를 적거나, 볼륨 단위 범위를 민감도 점검으로 추가하는 방안이 있다.
- **대체하지 않는 이유**: 원고·표·주장 범위가 이미 마스크 지표를 기준으로 짜여 있다. 게다가 공식 정의로 바꿔도 공개 수치와 직접 비교할 수 없다((b) 2).

### 384 출력에서 공식 정의로 계산하는 절차
1. 모델 출력을 그 파일 `reconstruction_rss` 의 (H, W) 로 가운데 자른다. 출력은 `Normalized` 래퍼가 ×s 를 되돌린 라벨 단위다.
   - 패딩 오프셋 `(384−H)//2` 는 `fastmri` `center_crop` 의 오프셋과 같다.
   - 로컬 val 의 7 가지 크기마다 한 파일씩 확인했다. `center_crop(label/1e6, rss.shape)` 가 `reconstruction_rss` 를 절대오차 ≤5.8e-11(상대 ≤8.9e-8)로 재현했다.
2. 파일 단위로 슬라이스를 쌓는다. `ds.samples` 는 파일별로 연속되어 있다(`dataloader_h5_v5.py:152-157`).
3. 정답은 h5 의 `reconstruction_rss` 를 그대로 쓴다.
   - 정답과 예측의 단위만 맞으면 배율(/1e6)은 결과에 영향이 없다.
   - 최소제곱 강도 배율 보정은 하지 않는다.
4. 정답과 예측을 (W, W) 로 가운데 자른다(W = 정답 폭). 로컬 val 에서 실제로 잘리는 것은 비정사각 19 개 볼륨의 위아래 행뿐이다.
5. data_range 는 잘라낸 정답 볼륨의 최댓값으로 한다(`evaluate.py` 정의).
   - Lightning 검증과 PromptMR+ 가 쓰는 `attrs['max']` 와는 로컬 val 6/464 볼륨에서 다르다.
   - 그중 FLAIR_203 276² 2 개는 attrs 가 20.2%·2.0% 낮다.
   - 나머지 4 개(AXT1_202 3 개, AXT2_207 1 개)는 crop 으로 최댓값이 바뀌며, 최대 7.9% 낮다.
   - 어느 정의를 썼는지 표 주석에 적는다.
6. 슬라이스마다 skimage `structural_similarity(gt, pred, data_range=볼륨 최댓값)` 를 기본 인수로 계산해 볼륨 평균을 낸다.
   - PSNR 은 볼륨 전체에 `peak_signal_noise_ratio(..., data_range=같은 최댓값)` 를 적용한다.
   - NMSE 는 볼륨 단위 ‖gt − pred‖² / ‖gt‖² 다.
7. 볼륨들을 평균하고, ± 가 SD 인지 2·SD 인지(`evaluate.py` 는 2·SD) 적는다.

### 구현할 때 주의할 점
- **새 파일로 만든다.**
  - pod3 의 ETER-net 런처는 학습이 끝나면 `eval_v8fix_grufp32.py` 를 거쳐 `eval_v8fix.py`·`eval_unet_only_v8.py` 를 부른다(10-14 스냅숏 `pod_run_arm_ampguard.sh:57-63`). 그래서 이 두 파일은 고치지 않는다.
  - `eval_unet_only_v8.run_inference` 의 루프(`:114-139`)와 `eval_v8fix.Normalized`(`:50-60`)를 가져다 쓴다.
  - 같은 패스에서 마스크 지표와 공식 지표를 함께 계산하면 추론을 한 번만 돌리면 된다.
- **runstats 가 없다.** 이 환경에서 `from fastmri.evaluate import ...` 는 runstats 미설치로 실패한다(`evaluate.py:15`). fastmri 패키지 메타데이터는 `runstats>=1.8.0` 을 의존성으로 적어 두었다. `ssim`·`psnr`·`nmse` 세 함수는 skimage·numpy 만 쓰므로 그대로 옮기면 환경을 바꿀 필요가 없다.
- **CPU 로 돌릴 때**: `run_inference` 의 autocast 가 `'cuda'` 로 고정되어 있다(`:129-130`).
- **선택 사항**: 예측을 fastMRI 형식 h5(`reconstruction` 키)로 저장하면 공식 CLI 로 교차 확인할 수 있다. 필요한 디스크는 val 0~2 약 9–13 GB, test_full 약 3.6–5.2 GB 다(float32 산술 추정, 측정하지 않음).
- **pod 데이터 확인**: pod 의 val 0~2·test_full 헤더를 읽어 384 를 넘는 RSS 축이 있는지 먼저 본다. 있으면 dataloader 가 정답을 잘라 버리므로 원래 크기로 되돌릴 수 없다. 로컬 464 개에는 없다.
- **공개 모델 행**: 이 행에도 공식 정의를 적용한다면 최소제곱 배율 보정을 뺄지 정해야 한다. 공식 정의에는 이 보정이 없다. 입력은 그대로 384 재-FFT·16 코일이다.

### 비용

| 항목 | 값 | 근거 |
|---|---|---|
| 공식 SSIM 계산 자체 | CPU 1 스레드로 4.9 ms/슬라이스(320²), 7.7 ms(384²). val 0~2 21,842 슬라이스 전체가 약 1.8–2.8 분 | 검증 단계 재측정(합성 볼륨) |
| 추론 재실행 | 필요하다. 예측은 저장되어 있지 않고, per-slice CSV 에는 마스크 지표만 있다(`idx,file,slice_idx,ssim,psnr,nmse,l1`) | `logs/runpod_mirror*/results/v8fix_s1_fullval_*/per_slice_*.csv` |
| GPU (pod3 RTX 3090, SS2D) | test_full 18.5 분, val 0~2 42.8 분 (batch 1, AMP, workers 8) | `eval_ss2d_s1_fullval.log:15, 77` |
| GPU (U-Net 단독, ETER-net) | 측정 없음 | — |
| CPU 로컬 (8 스레드, v8 ckpt 12 슬라이스) | SS2D 26.4 s/슬라이스, GRU 32.6 s/슬라이스. SS2D / GRU 기준으로 로컬 val 7,334 슬라이스 약 54 / 66 h, test_full 약 65 / 80 h, val 0~2 약 160 / 198 h | `results/vis/multimodel_compare/metrics_summary.txt:22-29` |
| ckpt 상태 | U-Net 단독 best = ep48. SS2D best = ep50 (best.pt 와 epoch_50.pt 의 텐서가 같음. `summary_ss2d.md` 의 "ep 46" 은 소수 4 자리 동률을 첫 epoch 로 고른 결과). ETER-net 은 아직 학습 중으로, 10-16 09:18 UTC 미러 기준 ep44 val 0.9212 이고 50 epoch ckpt 와 DONE 이 없다 | DONE 파일, CPU `torch.equal`, `log.txt` |

판단은 다음과 같다.
- 비용은 거의 전부 재추론에서 나온다. 공식 지표 계산 자체는 무시할 만하다.
- CPU 경로는 모델 하나에 약 2~8 일이 걸려 현실적이지 않다.
- 실측 근거가 있는 경로는 pod3 GPU 이고, SS2D 기준으로 모델당 약 1 시간(test_full + val)이다.
- ETER-net 은 학습이 끝나야 평가할 수 있다(메모리 기재 종료 예정 약 10-16 23 UTC).
- IEIE 학술대회판 마감(10-19) 전에 넣을지 MDPI 판으로 미룰지는 사용자가 정할 일이다.
- pod3 에 지금 빈 GPU 가 있는지는 확인하지 않았다.

## (d) 확인하지 못한 항목

- **리더보드 서버가 실제로 돌린 코드**: 근거는 Muckley §II-C 와 Discussion 293 의 "essentially runs the same code" 두 문장뿐이다. 원래 사이트의 Submission Guidelines 페이지(JS 앱)는 Wayback 보관본에서 본문을 얻지 못했다. 2019 Official Rules PDF 에는 SSIM 을 쓴다는 말만 있고 세부 설정은 없다.
- **E2E-VarNet·HUMUS-Net 리더보드 수치**: 어떤 평가 코드 판으로 계산했는지 서버 쪽에서 확인할 수 없다. "E2E-VarNet = `evaluate.py`" 는 추론이다.
- **Muckley 의 사후 마스크 알고리즘**: 종류와 임계값을 본문·보충자료 어디에도 적지 않았다.
- **Zbontar·Knoll 의 '마스크 없음'·'슬라이스 평균 방식 부재' 판단**: 추출 텍스트 grep 에 근거했다. 그림·표 캡션을 빠짐없이 읽은 것은 아니다. Knoll 의 텍스트가 깨졌다는 문제는 PyMuPDF 로 다시 추출해 해소했다. "브레인 크기는 고정이 아니다" 는 Zbontar 에 적힌 문장이 아니라, knee 한정 서술·Muckley §II-B·로컬 실측을 종합한 판단이다.
- **브레인 320×320 crop 을 공식으로 명시한 문헌**: 찾지 못했다.
- **크기·분할 관련 측정 범위**: 로컬 val 464 는 공식 val 1,378 의 일부다. pod 의 val 0~2·test_full 은 RSS 크기 분포를 측정하지 않았다. 384 를 넘는 축이 있는지도 확인하지 않았다.
- **ETER-net 계열 원문**:
  - MLMIR 2018: 본문을 열람하지 못했다.
  - Medical Physics 2021: 초록 수치만 확인했다(fastMRI R4 SSIM 0.931, R8 0.884). 방법 절을 읽지 못해 계산 방식을 모르므로 우리 수치와도, 공개 수치와도 비교할 수 없다.
- **MambaMIR**: 저장소가 404 라 SSIM 구현을 확인하지 못했다.
- **LMO**: 보충자료를 읽지 않았다. `weili419/LMO` 가 1저자 저장소인지, 이 코드가 논문 수치를 만들었는지, 어느 skimage 판을 썼는지 확인하지 못했다.
- **Gourdeau 2022**: 초록만 읽었다.
- **PromptMR 표 B.1**: ± 가 SD 인지 2·SD 인지 확인하지 못했다.
- **PromptMR+**:
  - README 의 브레인 test 0.9615 를 어떤 평가 스크립트로 냈는지 README 에 없다.
  - CMRxRecon 배경 crop 의 이유도 적혀 있지 않다. 처음에 쓴 "배경 비중 축소" 목적은 추론이라 정정했다.
  - CC brain 의 끝 50 슬라이스 제외 이유는 "little meaningful anatomy" 다.
- **MambaRecon**: 전처리된 fastMRI 브레인 데이터의 출처와 세부를 확인하지 못했다.
- **ReconFormer**: skimage 가 `multichannel` 인수를 어느 판부터 무시하는지 확인하지 않았다.
- **Jalal**: 정답과 재구성을 둘 다 0 으로 바꾸는지 본문에 명시가 없다.
- **우리 평가의 3 px 테두리 차이**: 영향을 측정하지 않았다. "마스크 지표에서는 0 테두리 영향이 거의 없다" 도 측정 없이 내린 판단이다.
- **평가 시간 미측정**: U-Net 단독·ETER-net 의 GPU 평가 시간, U-Net 단독의 CPU 추론 시간. 메모리에 있는 ETER-net 약 34 분/val 은 학습 중 검증(batch 4)에서 잰 값이다.
- **pod3 의 runstats**: 없다는 판단은 wandb `requirements.txt` 에서 추론했고, pod 에서 import 해 보지 않았다.
- **pod3 의 빈 GPU 여부**: 확인하지 않았다.
- **예측 h5 저장 디스크 크기**: 산술 추정이다.
- **부수 발견 (영향 미확인)**: `file_brain_AXT2_200_2000306/308/310/311` 4 개 파일이 로컬 `multicoil_train` 과 `multicoil_val` 양쪽에 있다. PromptMR+ 의 `fm-brain.json` 은 이 파일들을 train 으로 분류한다. 공식 tar 목록과 대조하지 않았고, 과거 로컬 런(v8)에 영향이 있었는지도 확인하지 않았다. **(주: 이미 알려진 사항이다. 10-12 Pod 확인 결과 이 4 볼륨은 공식 train 볼륨이 로컬 val 0번에 섞인 것이고, Pod 의 val 0~2·test_full 에는 없다. 로컬 평가와 스모크 하네스는 이 4 볼륨을 뺀다 — `docs/professor_settings_compare_2026-10-12.md` §6-1, `smoke_deter_opt.py` 의 OVERLAP.)**

## (e) 다음 단계 제안 (사용자 결정 필요)

1. **IEIE(10-19)**: 방법 절에 SSIM 정의(뇌 마스크 안 평균, data_range = 마스크 안 GT max−min, 슬라이스 → 볼륨 평균)를 정확히 적고, 공개 fastMRI 수치와 같은 줄에 놓지 않는다. 브레인 공식 평가를 "320×320 crop" 으로 쓰지 않는다.
2. **공식 정의 SSIM 을 보조 지표로 함께 내기 (권고)**: 새 평가 스크립트(기존 `eval_v8fix.py`·`eval_unet_only_v8.py` 는 pod3 런처가 쓰므로 고치지 않음)로 한 번의 추론에서 마스크 지표와 공식 지표(볼륨 최댓값 data_range, 원래 크기로 되자른 전체 영상, 볼륨 PSNR·NMSE)를 같이 계산한다.
   - 비용은 거의 재추론이다. pod3 GPU 기준 SS2D 가 test_full 18.5 분 + val 0~2 42.8 분이다.
   - 지금 pod3 의 GPU1 은 비어 있다(SS2D 학습 종료). 하지만 GPU0 의 ETER-net 학습과 CPU·메모리를 나눠 써서 학습이 느려질 수 있다(10-12 호스트 경합 사례). 그래서 지금 돌릴지, ETER-net 학습이 끝난 뒤(≈10-17 00:45 UTC 이후) 세 모델을 한꺼번에 돌릴지 정해야 한다.
   - pod 의 val 0~2·test_full 에 384 를 넘는 RSS 축이 있는지 먼저 확인한다(있으면 384 틀에서 원래 크기로 되돌릴 수 없음).
3. **data_range 민감도 점검(선택)**: 볼륨 단위 범위로 바꾼 마스크 SSIM 을 같은 패스에서 함께 계산해, 범위 정의를 바꿔도 모델 간 순위가 유지되는지 확인한다(Dohmen 권고 대응).


## (f) 실측 결과 — 세 모델 공식 정의 평가 (10-16~17, 사용자 "직접 확인을 진행해보자")

(e) 2·3 을 실행했다. 스크립트 = `v8_eter_pure/eval_v8fix_official.py`(한 번의 추론에서 마스크 지표·공식 지표·볼륨 단위 data_range 마스크 SSIM·공식 SSIM map 의 뇌 안/밖 분해를 함께 계산, `--selftest` 로 fastmri.evaluate 함수와 수치 일치 확인) + 짝 비교 `v8_eter_pure/eval_v8fix_official_compare.py`. 원자료 `results/eval/v8fix_official_test_full/`(표 `three_model_table.md`, 짝 비교 `compare_*.md`). 평가 집합 test_full 558 볼륨 / 8,852 슬라이스, pod3 GPU1. 모든 RSS 축이 384 이하·H ≥ W 라 384 틀에서 원래 크기로 정확히 되자를 수 있다(사전 확인).

- **점검**: 공식 SSIM = 분해 합(최대 오차 0), 라벨 vs h5 RSS 상대 오차 ≤ 1.2e-7. 마스크 SSIM 은 런처 평가 CSV 와 U-Net 단독·SS2D 에서 비트 일치(최대 차 0). ETER-net 은 TF32 를 끄고(학습과 같음, `--tf32 off`) 돌려 런처 평가(cuDNN TF32 켜짐)와의 차이로 TF32 영향을 쟀다 → 슬라이스 SSIM 최대 4.8e-5·평균 −4e-8, 볼륨 평균 동일(0.91907). **세팅 점검 §1-6(평가 TF32 미해제)은 영향 없음으로 종결.**
- 세 모델 모두 seed 1 단일 학습, best = val 0~2 기준(U-Net 단독 ep48, ETER-net ep46, SS2D ep50).

| 지표 (볼륨 평균±SD) | U-Net only | ETER-net (bi-GRU) | SS2D |
|---|---:|---:|---:|
| 마스크 SSIM (현행 주 지표) | 0.9213±0.0466 | 0.9191±0.0474 | 0.9199±0.0469 |
| 마스크 SSIM, data_range 볼륨 단위 | 0.9509±0.0295 | 0.9493±0.0300 | 0.9498±0.0299 |
| 공식 SSIM (evaluate.py 정의) | 0.8965±0.0400 | 0.8652±0.0399 | 0.8389±0.0575 |
| 공식 SSIM map — 뇌 안 | 0.9532 | 0.9517 | 0.9522 |
| 공식 SSIM map — 뇌 밖 | 0.8592 | 0.8123 | 0.7725 |
| 마스크 PSNR (dB) | 34.91 | 34.69 | 34.78 |
| 공식 PSNR (dB) | 34.81 | 32.88 | 32.04 |
| 공식 NMSE (%) | 1.50 | 2.30 | 2.82 |

짝 비교 (Δ 평균, 나은 볼륨 수 / 558, Wilcoxon p):

| 비교 | 마스크 SSIM | 공식 SSIM | 공식 SSIM map 뇌 안 | 공식 SSIM map 뇌 밖 |
|---|---|---|---|---|
| ETER-net − U-Net only | −0.0022 (2, 7e-92) | −0.0313 (1, 4e-93) | −0.0016 (4) | −0.0469 (1) |
| SS2D − U-Net only | −0.0014 (12) | −0.0575 (0, 4e-93) | −0.0011 | −0.0867 |
| SS2D − ETER-net | +0.0008 (453, 6e-58) | −0.0262 (80, 5e-79) | +0.0005 (456) | −0.0398 (73) |

해석:
1. **뇌 안에서는 정의를 바꿔도 순위가 같다** — 마스크 SSIM·볼륨 범위 마스크 SSIM·공식 SSIM map 뇌 안 세 정의 모두 U-Net only > SS2D > ETER-net. 시퀀스 모듈을 넣은 두 모델이 U-Net 단독보다 낮은 것은 평가 정의 탓이 아니다(Dohmen 권고형 민감도 점검 통과).
2. **공식 정의(전체 영상)의 차이는 대부분 뇌 밖 배경에서 나온다** — 공식 SSIM Δ 근사 분해: ETER-net − U-Net 안 −0.0006 / 밖 −0.0296, SS2D − U-Net 안 −0.0004 / 밖 −0.0547, SS2D − ETER-net 안 +0.0002 / 밖 −0.0251. 학습 손실도 마스크 안에서만 계산하므로 배경은 학습되지 않은 영역이고, 시퀀스 모듈 출력 채널이 그 영역에 구조를 더해 공식 정의에서만 크게 벌어진다(배경 SSIM 0.859 → 0.812 → 0.773). **공식 정의에서는 SS2D 와 ETER-net 의 순서가 뒤집힌다**(뇌 안 +0.0005 → 전체 −0.0262) — 이 순서는 재구성 품질이 아니라 배경 처리 차이다.
3. **한계**: 각 모델 1 회 학습이라 짝 비교 p 값은 볼륨 간 변동만 반영하고 학습 회차 간 변동은 반영하지 못한다(d-ETER 스모크 ref2−ref 차 +0.0081 @4000 step, v8 1b 재현 실패 사례). 뇌 안 차이(0.0005~0.0022)는 회차 간 변동과 구분된다고 말할 수 없다.
4. **원고 권고**: 주 지표는 지금처럼 뇌 마스크 SSIM, 방법 절에 정의를 정확히 적고, 한계 문장으로 "fastMRI 공식 정의(마스크 없음·볼륨 최댓값)로는 학습하지 않은 배경이 포함되어 수치가 낮아지고 모델 간 차이도 배경에서 커진다"를 쓴다. 공식 정의 수치를 공개 리더보드 수치와 같은 줄에 놓지 않는다(학습 데이터·크기 처리가 달라 여전히 비교 불가).
