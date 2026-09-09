# 논문 표 양식 관례 조사 + IEIE 초안 표 갈아엎기 (2026-09-03)

사용자 요청(09-03): "최근 MRI reconstruction 논문이 표를 어떻게 쓰고 지표를 어떻게 비교하는지" 확인하고, 우리 표가
"하고 싶은 대로 만든" 형태에서 벗어나는 지점을 **전부** 관례에 맞게 고친다. 이 문서는 (1) 확인한 양식 근거, (2) 어긋난
지점과 조치, (3) 남은 [TBD] 를 기록한다. 표 자체는 `paper/make_tables.py` 가 생성(`paper/tables/tableC*.md|tex`,
`ieie_table1_block.md`), IEIE 초안은 `paper/ieie/draft_ieie_v2.src.md`(09-09 통합 단일 소스; v1 두 소스는 `paper/ieie/archive/`) → 빌더 2종.

## 1. 확인한 양식 근거

### IEIE(대한전자공학회) 투고규정·템플릿
- 투고규정(2022-05-13 개정, `paper_submission_guideline.pdf`): 정규논문 12매, 국문 작성, 제목·요약·그림/표 설명은 **영문 병기**,
  **표 설명은 표 위**, "표와 그림: 영문" → 표 안 내용(머리글·행 이름)은 영문. 문장식 표제(sentence case).
- 논문지 템플릿 예시 표: **전체 괘선(hairline, sz=3) 격자**, 8 pt, 가운데 정렬, 머리글 굵게, 표제 위(국문+영문).
  → `paper/ieie/build_ieie_docx.py::table()` 이 이 틀을 그대로 재현(틀 자체는 통과). 학술대회 2쪽 양식은 국문 표제만.

### 최근 MRI 재구성 논문(원문 확인)
| 논문 | 표 형식 | 산포 | 강조 | 기준선 행 | 비고 |
|---|---|---|---|---|---|
| MMR-Mamba (MedIA 2025, PMC12842503) | booktabs + 그룹 세로선 | mean±std | 최고 굵게+음영 | Zero-filling 행, Param (M) | ↑↓ 머리글, ✓/✗ ablation |
| DM-Mamba (arXiv 2501.08163) | booktabs, 데이터셋→AF→NMSE/SSIM/PSNR 다층 머리글 | 없음 | 최고 굵게, 자기 행 회색 | — | AF 별 열 |
| HiFi-Mamba (arXiv 2508.09179) | 위와 동일 | 없음 | 굵게 | — | 효율 표 별도, 표제 아래 |
| MambaRecon (arXiv 2409.12401) | 전체 괘선 | 없음 | 굵게 | — | params 표 별도 |
| SO-Mamba (arXiv 2605.22031) | booktabs | 작은 글씨 std | 굵게 | — | |
| Oh 2025 (교수님, MDPI) | 가로 괘선만, 지표=행/모델=열 | "1.92±0.230" | 없음 | — | nMSE (%)·SSIM·VIF |
| fastMRI 공식 `evaluate.py` | — | 볼륨 단위 | — | — | 320 center-crop, 마스크 없음, NMSE/PSNR/SSIM |

공통 관례: **SSIM·PSNR·NMSE(대개 %)** 세 지표, **Params (M)** 열, **최고 굵게(차선 밑줄)**, mean±std, 머리글 ↑↓, 표 내용 영문,
zero-filled/공개 모델 기준선 행, 가속률(R) 축 표, 효율(시간·메모리) 표 분리. L1 열은 어느 논문에도 없음.

## 2. 어긋난 지점 7 + 조치(09-03 적용)
| # | 어긋남(기존 IEIE 초안) | 조치 |
|---|---|---|
| 1 | 표 내용 한국어 | 머리글·행 이름 영문(bi-GRU (original) / SS2D (controlled) / Enhanced SS2D / Zero-filled) |
| 2 | L1 열 | 결과 표에서 제거(손실 항으로만 언급). 본문 "4지표" → "세 지표(SSIM·PSNR·nMSE)" 일괄 수정 — per-contrast 범위·우위 비율 수치는 L1 제외해도 불변(min/max 가 SSIM·PSNR/NMSE 에서 나옴) |
| 3 | NMSE 0.00438 (소수) | nMSE (%) 0.438 |
| 4 | 평균만 | mean±SD(슬라이스 단위, SD ddof=1). 볼륨 단위 변형(0.9127/0.9141/0.9146 ± 0.037) 은 `ieie_table1_block.md` 두 번째 블록 |
| 5 | 강조 없음 | 최고 `**굵게**`·차선 `__밑줄__`(빌더 2종이 마크업 해석; 저자 `***` 자리표시자는 매치 안 되도록 정규식 제한) |
| 6 | 기준선 행 없음 | Zero-filled 행 추가(`v8_eter_pure/eval_zero_filled_v8.py`, CPU, 우리 좌표계·지표식 그대로; raw 변형 사용, ls 변형은 md 에만). 공개 U-Net/E2E-VarNet 은 train+val 학습(누수)이라 표에서 제외하고 각주 |
| 7 | Params "668M" 문자열, 프로토콜 미기재 | Params (M) 숫자 열; 표제에 데이터·볼륨/슬라이스 수·R·brain-masked·단위 명시; 효율 표(표 5) 분리 |

paired 통계(우위 비율·Wilcoxon)는 관례상 주 표가 아니므로 **보조 표 하나**로 압축(표 3: 두 비교 × 3지표), 강화판 전용 paired 표는 삭제.
그림 4(4패널, L1 포함)는 유지하고 표제에 "L1 은 손실 항 참고용" 명시.

## 3. 남은 [TBD] (GPU0 큐 확보 후 — `docs/v8_fairness_followup_plan.md` 큐에 편입 예정)
- ~~Zero-filled 행~~ **완료(09-03 16:32)**: `results/eval/zero_filled/`(CPU, 7,334 슬라이스) → `make_tables.py` 조인 → 두 초안 표에 반영. raw 변형 슬라이스 단위 SSIM 0.7521±0.0767 / PSNR 24.76±2.96 dB / nMSE 3.938±3.806 % (볼륨 단위 0.7523±0.0410 / 24.76±2.11 / 3.935±2.166). ls(강도 정합) 변형은 md 요약에만(0.7620 / 25.36 / 3.10 %).
- 가속률 일반화 표(R∈{2,4,6,8}): 현재 `results/eval/v8_r_sweep/` 는 stride-4 서브샘플(n=1,834, 상향 편향)이라 논문 표 불가 → 전체 val 재실행 필요.
- 효율 표의 추론 ms/slice·peak VRAM, fastMRI 표준 프로토콜(320 crop·무마스크·볼륨 단위) 수치.
- ~~공개 모델 전체 평가(n=299 → 7,334)~~ **우리 프로토콜 행 09-06 CPU 런 → 09-08 논문 반영**: `make_tables.py` Table C4(`paper/tables/tableC4_reference.{md,tex}`·IEIE 블록 `ieie_table_ref_block.md`, 순위 표시 없음·마지막 열 = SS2D 대비 우위 슬라이스 % SSIM/PSNR) → 학술지판 **표 4**(신설 Ⅳ장 6절 "공개 모델 참고선"; contrast 표 4→5, 효율 표 5→6 재번호)·학술대회판 한 문장. U-Net† 0.8971±0.0366 / 30.95±2.29 / 0.973±0.796 %, E2E-VarNet† 0.9181±0.0386 / 32.78±3.21 / 1.133±1.291 %(볼륨 단위, LS 정합 후) 확정, **PromptMR+ 행은 `[TBD]`**(추론 진행 중 — 완주 후 `make_tables.py` 재실행 → 초안 `[TBD]` 교체). 공개 가중치의 원 프로토콜 행·ms/VRAM 은 GPU 큐 7단계.
- ~~대표 수치를 볼륨 단위로 바꿀지~~ **결정·적용(09-04): 볼륨 단위**. 두 초안의 결과 표(학술지 표 2·표 4, 학술대회 표 1)·초록·본문 대표 수치를 볼륨 단위(n=464, ddof=1)로 교체 — SSIM bi-GRU 0.9127±0.0366 / SS2D 0.9141±0.0365 / Enhanced 0.9146±0.0361, PSNR 33.78 / 33.91 / 33.92 dB, nMSE 0.448 / 0.438 / 0.439 %, Zero-filled 0.7523 / 24.76 / 3.935. paired 분석(표 3·그림 4)은 슬라이스 단위 Δ + 볼륨 단위 검정 병행으로 유지, 학습 로그 수치(matched-epoch 0.9130 vs 0.9140)는 "(학습 로그 기준)" 명시. 순위·유의성은 두 단위에서 동일. 블록 출처 `paper/tables/ieie_table1_block.md`(첫 블록)·`ieie_table4_block.md`.

## 4. 렌더 확인 — 구조 점검기 + Word/한글 체크리스트 (09-04)
컨테이너에 docx 렌더러(Word/LibreOffice/pandoc/poppler)가 없어 **구조 점검만 자동화**했다: `python paper/ieie/check_docx_structure.py`(표준 라이브러리 + PIL). 검사 항목 — zip/XML 무결성, 관계 id·media 해결, style id, 섹션(pgSz/pgMar/cols) = 양식 일치(문단 수준 sectPr 포함), 표 gridCol/tcW/셀 수·허용 폭(부동 wrapper 는 본문 폭, 단 내 표는 단 폭), 그림 폭·높이·유효 dpi(≥300), 수식 문단 내 배치·우측 번호, 표 캡션 위/그림 캡션 아래 + 영문 병기, 표·그림·식 참조 번호 ≤ 개수, 잔존 마크다운·지시어·`***`/`[TBD]`, run 공백 보존, 연속 빈 문단, 글꼴 = 양식 범위, 쪽수 추정(휴리스틱; 2쪽 예시 문서를 1.4쪽으로 추정할 만큼 낙관적이므로 참고만). 09-04 결과: 학술지판 FAIL 0 / WARN 1([TBD] 7건 = GPU 큐 대기 수치), 학술대회판 FAIL 0 / WARN 1(`***` 저자 자리표시자 7건). 학술지판 추정 ≈15쪽(양식 본문 10pt·줄간격 160%) — 실제 쪽수는 Word 확인 필요.

구조로는 판정 불가 → **Word/한글에서 직접 볼 것**:
1. 부동 배치: 학술지판의 페이지 폭 표·그림(표 2~5, 그림 1~4)은 `tblpPr`(margin/top, overlap never) 부동 wrapper 표 — 같은 쪽에 두 개 이상 앉을 때 겹침·다음 쪽 밀림·본문 공백이 생기는지. 학술대회판 그림 1은 단 내 inline.
2. 표 안 줄바꿈: 8pt 고정 폭(학술지 page 표 9,100~9,500 twips, 단 표 4,563/4,535)에서 `0.9141±0.0365` 같은 셀이 두 줄로 꺾이지 않는지, 헤더 `PSNR (dB) ↑`·`SS2D vs. bi-GRU (%)` 가 한 줄인지.
3. 수식(학술지판 6개 oMath): 기호(ỹ, ⊙, ‖·‖, ∘)·첨자 렌더와 우측 정렬 번호 (1)~(6)의 탭 위치; 한글(HWP)에서는 OMML 이 수식 개체로 변환되는지.
4. 글꼴 치환: 바탕·HY신명조가 없는 PC 에서 대체 글꼴로 바뀌면 분량이 변한다 — 제출 PC 에 글꼴 설치 확인, 한글 변환본은 별도 검토.
5. 쪽수: 학술대회판은 09-08 그림 1 을 세부 그림 3장(파이프라인·두 팔 내부·강화 SS2D)으로 나눠 빌더 추정 2.88쪽/점검기 2.6쪽 — 2026 추계 규정은 "Double Column 1페이지 이상"(하계는 "1페이지 이상 5페이지 이내")이라 2쪽은 작성예시의 분량이지 상한이 아님(웹 확인 09-08). 2쪽에 맞춰야 하면 그림 배율 `| col | 0.85` + 방법 절 축약으로 조정. 학술지판 총 쪽수·게재료 기준 확인.
6. 캡션 규칙: 표 위 캡션(국문+영문), 그림 아래 캡션(국문+영문), 표·그림 본문 영문 — 캡션이 표/그림과 다른 쪽으로 분리되지 않는지.
   - 그림 문법(09-08, 사용자 "어색하다" 지적으로 학술대회판 그림 1~3 재작성): 어색함의 원인은 슬라이드 문법 — 상자 안 서술문·굵은 제목/색 띠·크기 제각각 상자·실데이터 없음·위아래로 꺾이는 흐름·RNN/SS2D/Mamba 를 관례와 다르게 그림. 교정 규칙 = ① 블록은 한 줄 라벨(+작은 부라벨)로 균일, 크기·채널은 화살표 위 작은 회색 글씨 ② 실데이터 썸네일(정본 슬라이스 k-space·마스크·zero-filled·복원·GT) ③ 좌→우 한 줄 흐름, 손실은 끝에 점선 괄호 ④ bi-RNN 은 펼친 체인(순/역방향 두 행, 아래 x_t·위 y_t) ⑤ SS2D 는 cross-scan 격자 4개 → S6 ×4 → Ⓒ(concat)→LN·Linear(코드 `ss2d.py` 확인: 덧셈 아님) ⑥ Mamba 블록은 분기(SSM 가지/SiLU 게이트)→⊗→Linear→⊕ 잔차 ⑦ 본문 글꼴 Liberation Sans(Arial 메트릭, OFL, `paper/fonts/`), 6.5 pt 라벨/5 pt 주석. 학술지판 그림 1 도 09-09 같은 규칙으로 재작성(`paper/make_fig1_architecture.py` — page 폭 6.69 in 3-패널 (a)파이프라인/(b)두 팔/(c)강화 SS2D, 도우미는 `make_figs_conf_arch.py` import; 캡션 (a)(b)(c) 갱신, 본문 '그림 1(b)'→'그림 1(c)' 정정).
   - 그림 인쇄 크기(09-09, 삽입 그림 점검에서 발견): 빌더가 그림을 page 폭 6.69 in 으로 삽입하므로 figsize 가 그보다 넓으면 글자가 비례 축소된다 — 옛 그림 2(10.5 in 조판 → ×0.64, 범례 4.8 pt)·그림 4(12.5 in → ×0.54, 라벨 ~4 pt)·그림 3(7.5 in → ×0.89, 패널 수치 4.8 pt). 교정 = 세 스크립트 모두 `PAGE_W = 6.69` 로 직접 조판(최소 글자 6.5 pt — 그림 3 패널 안 수치만 5.5 pt, Liberation Sans, 그림 2·4 600 dpi, 그림 3 은 384 px 슬라이스가 패널에 1:1 로 들어가는 506 dpi; 그림 2 범례는 패널 밖 하단 공통 한 줄, 그림 4 는 2×4 패널 높이 3.7 in + 우상단 2행 범례). `check_docx_structure.py` 는 유효 dpi 만 검사하고 글자 크기는 검사하지 않으므로 재생성한 PNG 를 실제 폭(6.69 in)으로 눈으로 확인한다(그림 수치·슬라이스는 조판 전과 동일).
7. 참고문헌 51건(학술지)·8건(학술대회) 번호 순서와 본문 [n] 상첨자 표기, 저자 `***` 자리표시자 교체.
선택지: 진짜 렌더가 필요하면 scratchpad 에 LibreOffice AppImage 를 내려받아(환경 무변경) PDF 로 변환해 볼 수 있다 — 요청 시 진행.


Sources: https://www.theieie.org/download/paper_submission_guideline.pdf · https://www.theieie.org/pages_journal/journal_info.vm ·
https://pmc.ncbi.nlm.nih.gov/articles/PMC12842503/ · https://arxiv.org/abs/2406.18950 · https://arxiv.org/abs/2501.08163 ·
https://arxiv.org/abs/2508.09179 · https://arxiv.org/abs/2409.12401 · https://arxiv.org/abs/2605.22031 · https://github.com/facebookresearch/fastMRI (evaluate.py)

## 5. 09-09 통합판 v2 (교수님 검토 반영)

- 교수님 지시: 학술지 양식 = **서면 심사용(4쪽, blind)**, 학술대회 양식 = **프로시딩 게재용(1~5쪽, 저자 포함)** — 두 파일은 **내용 동일**, 저자·소속만 차이.
  → 단일 소스 `paper/ieie/draft_ieie_v2.src.md` 를 두 빌더로 빌드(`--src/--out` 지원). 학술지 빌더는 `check_blind()` 로 본문 저자·소속 단서를 막는다(논문지 양식 자체에 저자란 없음).
- **캡션 규칙 변경**: 국문만, 한 줄로 간결하게(세부는 본문). 논문지 양식 예시에는 "그림 1." + "Fig. 1." 두 줄이 있으나 교수님이 영문 불필요라 하심 → `@cap_en` 은 선택(있을 때만 병기). `check_docx_structure.py` 는 영문 캡션 부재를 더 이상 경고하지 않고 개수만 알린다.
- **표 1 하나로 통합**: 본 연구 4행(Zero-filled / bi-GRU / SS2D / Enhanced SS2D, 최고 굵게·차선 밑줄) + 공개 모델 참고선 3행(U-Net† / E2E-VarNet† / PromptMR+, 순위 표시 없음) — 수치는 `ieie_table1_block.md`·`ieie_table_ref_block.md` 그대로, `@note` 에 † 와 PromptMR+ 계열 차이 명시.
- 초록 c44/c71 코멘트 반영: "모든 구성요소(데이터·마스크·U-Net 구조·손실·최적화 설정)가 동일(가중치 공유 없이 각각 독립 학습)", 우위 비율은 지표별로 슬라이스 78.2/73.8/73.8 %·볼륨 94.8/89.9/90.1 %.
- 쪽수(휴리스틱 ±15 %): 학술지판 ≈3.7쪽(참고문헌 12편 ≈0.65쪽 포함), 학술대회판 ≈3.3쪽. Word 실측 후 4쪽 초과 시 1순위 절감 = 참고문헌·Ⅱ.2 문단.

