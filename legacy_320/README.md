# legacy_320/ — 루트 320×320 트랙 스크립트 (v6_1 ~ v6_3 · eval · visualize)

2026-09-03 저장소 루트에서 이동. **옛 8GB 머신(RTX 5060Ti, `mri_env`) 전용**이며 이 저장소에는 해당
ckpt/`runs/` 가 없어 실행·재현 불가 — 역사 기록으로 보존한다(`docs/script_version_history.md`,
`docs/presentation_overview.md`, `docs/logs_archive.md`). 320 결과는 384 서버 트랙과 직접 비교하지 않는다.

- `main_train_{ss2d,eter}_v6_{1,2,3}.py` — gradient-loss fine-tune(v6_1/v6_2, 폐기) · sharp ablation(v6_3, 채택후보). config 는 `configs/myConfig_choh_*_v6_x.py`(루트 유지).
- `eval_full_compare.py` / `eval_tta_ensemble.py` — 풀평가·TTA/앙상블(Tier 1, negative).
- `visualize_compare.py` / `visualize_compare_versions.py` / `visualize_diagnostic_v6.py` — 4-way 비교·버전 cross·raw↔masked SSIM 진단.

각 스크립트는 `current_dir` 를 상위 폴더(저장소 루트)로 잡도록 한 줄만 수정했으므로 저장소 루트에서
`python legacy_320/<script>.py` 로 호출하면 import 경로는 그대로 성립한다(ckpt 부재로 실행은 실패).
현행 4-way 시각화 스크립트(`visualize_v7_titan_compare.py`·`visualize_v8_pure_compare.py`·`visualize_v9_compare.py`)는 루트에 남아 있다.
