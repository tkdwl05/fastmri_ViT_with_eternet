# 발표 덱(이미지 중심 v6 비교) — 세션 핸드오프

작성: 2026-06-01. 다음 세션이 **그대로 이어받아 진행**할 수 있도록 정리한 문서.

> ✅ **완료 (2026-06-01)**: 세션 재시작 후 Figma 인증이 `tkdwl05@g.hongik.ac.kr · Full seat`로 갱신되어 §3 신뢰 레시피로 빌드 완료. 덱 = [ViT MRI Recon — v6 비교](https://www.figma.com/slides/W7zo3SteyNjTJMm4eFDaG3) (fileKey `W7zo3SteyNjTJMm4eFDaG3`, 9장). 초기 빌드는 v6_3가 이미지에 없어 Option A(v6 주인공 + 표 정량 참고)로 처리했으나, **이후 사용자가 슬라이드를 직접 수정** — 비교 이미지를 **6컬럼(v4·v6·v6_1·v6_2·v6_3 포함)** 으로 교체, 슬라이스 세트를 0000·0660·1321·1982·3304로, 내러티브를 "평균은 U-Net 추월하나 정성/개별은 U-Net 우세 → 과적합 추정"으로 신중화함. 상세 발표 대본 = [presentation_script_v6_deck.md](presentation_script_v6_deck.md) (실제 수정본 기준 **v2**). 아래 §1~§7은 최초 빌드 당시 참고 기록으로 보존.

## 0. 목표 / 결정 사항

- **목표**: `results/vis/root_track/compare_versions_aligned/`의 정렬 비교 이미지를 넣은 **이미지 중심 발표 덱**.
- **구성 범위 (사용자 확정)**: **이미지 중심 · v6 비교**. v7_titan 구간은 vis 이미지가 없어 제외(텍스트 전용이라 이미지 덱 범위 밖).
- **형식 (사용자 확정)**: **Figma Slides가 1순위**. 폴백으로 `.pptx`(아래 §6)는 **이미 완성**되어 있음.

## 1. ⚠️ 첫 단계 — Figma 계정/시트 확인

**왜 막혔나**: Figma MCP(`claude.ai Figma`)는 커넥터 승인 시 발급된 자체 OAuth 토큰으로 동작. 브라우저 figma.com 재로그인·데스크톱 플러그인 재연결로는 **MCP 계정이 안 바뀜**. 이번 세션 내내 `whoami`가 `tkdwl05@gmail.com · Starter · View`로 고정됨. View 시트는 **월 6회** MCP 호출만 허용 → 제대로 된 덱을 만들기엔 빠듯.

**갱신 방법(사용자가 추정)**: MCP의 OAuth 신원은 **클라이언트(세션) 재시작 후** 새 인증으로 갱신되는 듯. 다른(Full/Dev) 계정으로 재인증하려면: 브라우저에서 그 계정으로 먼저 로그인 → 커넥터(`/mcp` 또는 커넥터 설정) 재인증.

**다음 세션 FIRST STEP**:
```
mcp__claude_ai_Figma__whoami()
```
- `seat`/`seat_type`이 **Full 또는 Dev** → §3(신뢰 레시피)로 진행. `plans[].key`를 `planKey`로 사용.
- 아직 **View** → §4(쿼터 절약 레시피)로 최소 호출만, 또는 사용자에게 폴백 pptx(§6) 안내.

## 2. 자산 인벤토리 (이미 준비됨)

| 경로 | 내용 |
|---|---|
| `results/vis/root_track/compare_versions_aligned/compare_*.png` | 원본 12장 (3065×3068, 5~6MB). 4×4 그리드. |
| `results/vis/_slides_assets_v6/v6cmp_{0660,1982,3304,5286,6608}.png` | **업로드용 최적화본 5장** (2597×2600, ~4MB, <10MB 한도). 선정 슬라이스. |
| `results/vis/_slides_assets_v6/metrics_bar.png` | SSIM/PSNR 막대 차트 (matplotlib). |
| `발표 자료/ViT_v6_비교_이미지중심.pptx` | **완성된 폴백 덱** (9슬라이드, 21MB). |
| `tools/build_v6_deck.py` | 차트+pptx 재생성 스크립트 (conda `python`로 실행). |

**그리드 읽는 법**: 1행 재구성 `GT │ SS2D v4 │ SS2D v6 │ SS2D v6_1`, 2행 SS2D 오차맵, 3행 재구성 `U-Net │ ETER v4 │ ETER v6 │ ETER v6_1`, 4행 ETER 오차맵. 패널 상단 = 해당 슬라이스 PSNR/SSIM(raw).

## 3. Figma 빌드 — 신뢰 레시피 (Full/Dev 시트, 200회/일)

매핑이 확실한 방식. 슬라이스↔이미지 대응이 깨지지 않음.

1. **`create_new_file`** `editorType="slides"`, `fileName="ViT MRI Recon — v6 비교"`, `planKey=<whoami의 key>` → `fileKey` 확보.
2. **`use_figma`**(1회): 슬라이드 골격을 한 번에 생성 — 타이틀/범례/5개 이미지 슬라이드(빈 이미지용 사각형 + 캡션 텍스트)/요약/결론. 각 이미지용 사각형의 `name`을 `img_0660` 등으로 두고 **node id를 description 응답으로 반환**시켜 다음 단계 nodeId로 사용. 골격 스크립트 스켈레톤은 §5.
3. **`upload_assets`** `count=1`, `nodeId=<해당 사각형 id>`, `scaleMode="FIT"` 를 **5장 각각** 호출 → 각 사각형에 이미지 fill. (5회)
4. **`upload_assets`** `count=1`, `nodeId=<요약 슬라이드 차트 사각형>` 로 `metrics_bar.png` 배치. (1회)
5. (선택) **`get_screenshot`** 으로 슬라이드 검수 → 필요 시 `use_figma`로 미세 조정.

대략 8~10회 호출 (Full/Dev는 200/일이라 여유).

## 4. Figma 빌드 — 쿼터 절약 레시피 (아직 View, 6회/월)

1. `create_new_file` slides. (1)
2. `upload_assets` `count=5`, `batchCommit=true` (nodeId 없이) → 5장이 현재 페이지에 image-fill 프레임으로 생성. 반환 URL들에 PNG 바이트 POST 후 commitUrl 1회. (1)
3. `use_figma`(1): 페이지에서 image-fill 프레임 5개를 찾아 각각 슬라이드로 만들고 크기 맞춤 + 타이틀/캡션/범례/요약/결론 텍스트 일괄 생성. (1~2)

총 3~4회. 단, 이미지↔슬라이스 매핑이 업로드 순서에 의존하므로 캡션을 일반화하거나 스크린샷으로 확인.

## 5. `use_figma` 슬라이드 골격 스켈레톤 (§3-2용 참고)

> Figma Slides Plugin API. 슬라이드 크기는 보통 1920×1080. **한글 폰트 주의**: `Inter`는 한글 미지원 → `loadFontAsync({family:"Noto Sans KR", style:"Regular"})` 시도, 실패 시 사용 가능한 한글 폰트로 대체(첫 호출 전 `figma.listAvailableFontsAsync()`로 확인 가능). 반환값에 생성한 사각형들의 `{name,id}`를 JSON으로 담아 다음 nodeId로 사용.

```js
// pseudo-skeleton — 실제 API 시그니처는 첫 use_figma read로 확인 후 확정
const W = 1920, H = 1080;
const KFONT = {family: "Noto Sans KR", style: "Regular"};
const KBOLD = {family: "Noto Sans KR", style: "Bold"};
try { await figma.loadFontAsync(KFONT); await figma.loadFontAsync(KBOLD); }
catch(e) { /* fallback: pick a Korean-capable font from listAvailableFontsAsync */ }

const slices = [
  {key:"0660", title:"슬라이스 #0660 — 중간 뇌 (뇌실 레벨)"},
  {key:"1982", title:"슬라이스 #1982 — 중간 뇌"},
  {key:"3304", title:"슬라이스 #3304 — 두정부 (sulci 미세구조)"},
  {key:"5286", title:"슬라이스 #5286 — 두정부 고주파 (어려운 케이스)"},
  {key:"6608", title:"슬라이스 #6608 — 두정부 고detail"},
];
const out = [];
function txt(slide, s, size, x, y, w, bold){
  const t = figma.createText();
  t.fontName = bold ? KBOLD : KFONT;
  t.fontSize = size; t.characters = s;
  t.x = x; t.y = y; t.resize(w, t.height);
  slide.appendChild(t); return t;
}
for (const sl of slices){
  const slide = figma.createSlide();          // 시그니처 확인 필요
  txt(slide, sl.title, 48, 80, 60, 1760, true);
  const r = figma.createRectangle();
  r.resize(980, 880); r.x = 60; r.y = 160; r.name = "img_"+sl.key;
  slide.appendChild(r);
  out.push({name:r.name, id:r.id});           // ← upload_assets nodeId
}
// 캡션 텍스트는 발표 자료/ViT_v6_비교_이미지중심.pptx 의 슬라이드별 bullet 사용(§아래)
figma.notify("skeleton done");
return JSON.stringify(out);
```

**슬라이드별 캡션(복붙용)** — 폴백 pptx와 동일:
- 0660: v4→v6에서 뇌실·실질 경계 또렷 / SS2D 오차맵이 ETER보다 옅음 → SS2D 우위 / v6_1은 미세 과샤프닝.
- 1982: 구조 복원 일관성 / SS2D v6 오차맵이 ETER보다 균일하게 옅음 / U-Net 대비 동등~우위.
- 3304: SS2D v6 SSIM 0.9367 > ETER v6 0.9242 (U-Net 0.9447) / v6_1 PSNR 33.71→32.48dB 하락=over-sharpening / 고주파 sulci에서 차이 뚜렷.
- 5286: fine detail에 오차 집중 / mean-prediction blurring 가시화(발견②) / 정량↑ 시각↓ 괴리 근거.
- 6608: 여러 해부 레벨에서 SS2D>ETER 일관 / 4-방향 SSM의 장거리 의존성이 k-space aliasing에 효과적.

## 6. 폴백 .pptx (이미 완성·검증됨)

- 파일: `발표 자료/ViT_v6_비교_이미지중심.pptx` (9슬라이드, 16:9, 5비교이미지+차트 임베드).
- 재생성: `python tools/build_v6_deck.py` (conda `python` = mri_env. **주의: `python3`에는 PIL 없음**, 반드시 `python`).
- Keynote/PowerPoint에서 바로 열림. Figma로의 직접 import는 깔끔하지 않으므로, Figma는 §3/§4로 네이티브 생성 권장.

## 7. 핵심 수치 (presentation_overview.md §6.1, 표준 SSIM 7,270 슬라이스 풀평가)

| 모델 | PSNR(dB) | SSIM | NMSE |
|---|---|---|---|
| U-Net (pretrained, 496M) | 34.66 | 0.8858 | 0.00737 |
| SS2D-ViT v6 ▲U-Net 추월 | 35.96 | 0.8913 | 0.00810 |
| ETER-ViT v6 | 34.63 | 0.8862 | 0.01080 |
| SS2D-ViT v6_3 ★채택후보 | 36.05 | 0.8924 | 0.00800 |

핵심 메시지: ① 표준 metric에서 v6가 U-Net 추월, ② SS2D(Mamba) > ETER(GRU) 일관, ③ raw SSIM 배경 부풀림 + mean-prediction blurring(발견②), ④ v6_3 채택 후보.
