# 발표 대본 — ViT 기반 fastMRI Brain 재구성 (ETER vs SS2D)

총 21슬라이드  ·  시간 무관 완전판 (자연스러운 발표용 풀어쓴 대본)

---

## 슬라이드 1 — 표지

안녕하세요, 발표를 맡은 정동욱입니다.

오늘 발표 주제는 ViT 인코더 위에 시퀀스 디코더를 붙여서 fastMRI Brain 영상을 재구성하는 연구입니다. 시퀀스 디코더로 두 가지 변형을 비교했어요. 하나는 양방향 GRU 기반의 ETER, 다른 하나는 Mamba를 영상에 맞게 확장한 SS2D 입니다.

핵심 질문은 단순합니다. R=4, 즉 k-space의 25%만 사용하는 환경에서 — ViT 인코더라는 같은 출발점을 두고 — 어떤 시퀀스 모델링 전략이 더 좋은 복원을 만들어내는가. 이게 오늘 답하고 싶은 질문입니다.

발표는 크게 다섯 부분으로 흘러갑니다. 문제와 데이터, 모델 구조, 관련 논문 리뷰, 왜 SS2D를 선택했는지의 이론적 근거, 그리고 마지막으로 실험 결과와 향후 계획입니다.

---

## 슬라이드 2 — 문제 정의

먼저 문제부터 짚고 넘어가겠습니다.

MRI는 우리가 알고 있듯이 굉장히 유용한 영상 진단 도구지만, 동시에 본질적으로 느린 imaging 기법입니다. 환자가 MRI 장비 안에 누워있는 동안 k-space라는 주파수 도메인 데이터를 라인 단위로 하나씩 채워야 하는데, 이 시간이 길면 환자도 힘들고 장비 효율도 떨어집니다.

그래서 임상 현장에서는 k-space의 일부만 샘플링하는 방식으로 시간을 단축합니다. 이걸 undersampling이라고 부르고, 본 연구에서 다루는 R=4는 4배 가속 — 즉 원래 데이터의 25%만 사용한다는 뜻입니다. 그런데 이렇게 일부만 샘플링하면 영상 도메인에서 aliasing artifact라는 겹침 현상이 발생합니다. 그냥 iFFT로 변환만 하면 영상 위에 겹친 그림자처럼 나타나는 거죠.

본 연구의 과제는 이 R=4 조건에서 ViT 인코더와 시퀀스 디코더 구조를 사용해서 aliased 입력을 GT, 즉 ground truth 영상에 가깝게 복원하는 것입니다. 디코더 변형 두 가지를 비교한다고 말씀드렸는데, 하나는 Bi-GRU 기반 ETER, 다른 하나는 SS2D 기반 Mamba 입니다.

오른쪽 다이어그램을 보시면 전체 파이프라인이 정리되어 있어요. 위에서부터 — Full k-space에서 R=4 equispaced mask를 적용해 undersample하고, iFFT로 영상 도메인으로 가져와서 320×320으로 center crop을 하고, 다시 fft2c로 k-space를 재구성합니다. 그리고 그 결과로 만들어진 aliased image가 모델의 입력이 되고요. 모델을 통과하면 reconstruction된 영상이 나오는데, 이걸 GT와 비교해서 L1 + 0.2 × (1 − SSIM) 손실로 학습합니다.

---

## 슬라이드 3 — 데이터

학습과 검증에 사용한 데이터입니다.

fastMRI는 NYU와 Facebook AI Research가 공개한 대규모 의료영상 벤치마크인데, 그 중에서도 Brain Multicoil AXFLAIR 시퀀스를 사용했습니다. AXFLAIR는 axial 평면에서 fluid-attenuated inversion recovery 시퀀스로 찍은 영상이에요.

입력은 두 종류입니다. 첫째 `data`는 32 채널의 real/imag interleave된 k-space인데, 원래 16개 코일에서 받은 복소수 데이터를 real과 imaginary 부분을 따로 채널로 풀어서 32 채널로 만든 겁니다. 둘째 `data_img`는 그 k-space를 iFFT로 변환한 aliased image입니다. label은 fastMRI 공식에서 제공하는 reconstruction_rss라는 root-sum-of-squares 매그니튜드 영상이고요.

마스크는 phase encoding 축에서 R=4 equispaced 패턴, center fraction 0.08, offset은 3에서 시작해 3, 7, 11 식으로 4 간격으로 샘플링합니다. 스케일은 — k-space 입력은 1e4, 영상 입력은 1e6, GT는 1e6으로 정규화했어요.

오른쪽 stat card에 데이터셋 규모가 정리돼 있습니다. fastMRI brain은 파일마다 RSS shape이 320×320, 320×264, 768×396 식으로 굉장히 이기종으로 섞여 있어서 — 학습 일관성을 위해 320×320 정합 파일만 필터링했습니다. 그 결과 train은 541 파일에 8,548 슬라이스, val은 285 파일에 4,492 슬라이스를 확보했어요. 이 필터링 때문에 약 40% 데이터를 못 쓴 셈인데, 나중에 v5에서 이 부분을 완화하는 작업을 합니다 — 잠시 후에 다시 나옵니다.

---

## 슬라이드 4 — 공통 아키텍처

이제 본격적으로 모델 구조를 보겠습니다. ETER와 SS2D 두 변형이 공유하는 공통 아키텍처입니다.

핵심은 — 두 변형이 시퀀스 모듈만 다르고 나머지는 다 동일하다는 점입니다. 이게 공정한 비교를 가능하게 하는 핵심 설계예요. ViT 인코더, ViT 디코더, 그리고 마지막 합성 헤드는 두 모델 모두 똑같습니다.

다이어그램을 따라가면 — 입력이 두 개로 들어옵니다. 위쪽은 aliased image (B, 32, 320, 320), 아래쪽은 masked k-space (B, 32, 320, 320). aliased image는 ViT Encoder를 거쳐 ViT Decoder로 전달돼서 영상 도메인의 글로벌 컨텍스트를 학습합니다. ViT Encoder는 patch 32×32, 6 layer, hidden dim 384, head 6개. ViT Decoder는 6 layer에 hidden 512, 그리고 PixelShuffle을 두 번 적용한 up_tail로 해상도를 복원합니다.

그동안 k-space 쪽은 별도의 sequence module로 들어갑니다. 이게 ETER 일 때는 양방향 GRU 두 개 (수평·수직), SS2D 일 때는 4-way Mamba scan 입니다.

두 경로의 출력을 channel concat 한 뒤 — 보시면 가운데 큰 네이비 박스 — Concat + Refinement 블록에서 합쳐서 최종 1채널 영상으로 만듭니다. 즉 입력 영상, ViT가 본 영상, 그리고 시퀀스 모듈이 본 k-space 정보가 한 자리에서 만나는 거예요.

마지막에 Conv2d 3×3을 통과해서 reconstructed image (B, 1, 320, 320)이 나옵니다. 손실은 앞서 말씀드린 L1 + 0.2 × (1−SSIM)이고, mixed precision 학습이라 forward는 fp16, loss 계산은 fp32로 갈라놨습니다 — 이건 학습 설정 슬라이드에서 다시 다뤄요.

---

## 슬라이드 5 — ETER 디코더

먼저 ETER 디코더부터 보겠습니다. 이름 ETER는 End-To-End Recurrent에서 따왔습니다.

핵심 아이디어는 — k-space를 채널 시퀀스로 보고, 두 축에 대해 양방향 GRU로 순회한다는 거예요. 좌측 본문에 정리한 것처럼, 행 방향으로 forward + backward GRU 한 번 (gru_h), 그리고 열 방향으로 forward + backward GRU 한 번 (gru_v). 각각 hidden은 2 입니다. 두 방향의 출력 채널을 합치면 out_v = 2 × hidden = 4 채널이 되고, 이걸 ViT 디코더 출력과 in_imgs에 concat한 다음 Conv2d 3×3 한 번으로 1채널 복원 영상을 만듭니다.

그런데 여기서 중요한 제약이 하나 있어요. 좌측 하단 노란 박스에 강조해뒀습니다. 원본 ETER-Net 논문에서는 384×384 입력에 hidden=10이었어요. 이걸 본 연구의 8 GB GPU 환경에 맞추기 위해 320×320으로 줄이고, hidden을 10에서 2로 다섯 배 줄였습니다. 거기에 원본의 마지막 합성 단계가 U-Net이었던 걸 Conv2d 한 개로 단순화했고요.

이 축소가 실제 성능 한계의 한 축이 됐습니다. v3까지 ETER가 SS2D 대비 일관되게 낮은 성능을 보인 이유 중 하나가 바로 이 capacity 손실이에요. 자세한 건 실험 변천 슬라이드에서 다시 다룹니다.

오른쪽 다이어그램은 ETER 디코더의 데이터 흐름인데 — 좌측이 ViT 경로 (in_imgs → patch_embedding → encoder.transformer → enc_to_dec linear → decoder.transformer → final_linear + up_tail), 우측이 ETER 경로 (in_ksp → gru_h → gru_v → out_v 4 channels). 마지막에 두 경로가 만나서 concat + Conv2d 3×3으로 최종 출력이 만들어집니다.

---

## 슬라이드 6 — ETER-Net 논문 리뷰

여기서부터 4장 슬라이드는 본 연구가 직접 참고한 논문 4편의 리뷰입니다. 본 연구가 어떤 prior art 위에 서 있는지 명확히 하기 위해 정리했어요.

첫 번째는 방금 본 ETER 디코더의 출처 논문입니다. Oh, Chung, Han 세 분이 2022년 9월에 Sensors 저널에 발표한 "An End-to-End Recurrent Neural Network for Radial MR Image Reconstruction" 입니다. 한국 가천대학교 연구진의 작업이에요.

이 논문이 등장한 배경 맥락이 흥미로운데요. 그 이전에 AUTOMAP이라는 모델이 있었어요. 2018년 Nature에 발표된 모델인데, fully connected layer로 k-space를 영상으로 바로 변환하는 방식이었습니다. 아이디어는 강력했지만 메모리 요구량이 너무 커서 실제 해상도의 영상에 적용하기 어려웠어요. ETER-Net은 이 메모리 문제를 RNN으로 해결한 겁니다.

핵심 기여를 네 가지로 정리했습니다.

첫째, Domain transform RNN. 양방향 RNN을 써서 k-space 데이터를 수평으로 한 번, 수직으로 한 번 sweep해서 잠재 이미지로 변환합니다. AUTOMAP의 거대한 FC 레이어 대신 RNN을 쓰니까 메모리가 훨씬 효율적이죠.

둘째, Refinement network. 이미지 도메인에서 multi-channel feature를 단일 채널 magnitude image로 정제합니다.

셋째, 이 논문의 강점 중 하나가 Cartesian과 non-Cartesian — 즉 radial trajectory — 모두 지원한다는 점이에요. 임상에서 motion robustness 때문에 radial을 쓰는 경우가 많은데, 그것까지 한 모델로 처리할 수 있게 만든 거죠.

넷째, multi-channel coil 정보를 시퀀스 차원에 자연스럽게 흡수해서 별도의 coil combination 단계가 필요 없습니다.

오른쪽 Figure 1이 전체 구조입니다. 빨간색 화살표가 forward RNN, 파란색이 reverse RNN 방향이에요. 위쪽 큰 회색 박스가 Domain transform network (Horizontal RNN + Vertical RNN), 아래가 Refinement network입니다.

본 연구는 이 논문의 양방향 GRU 모델링 부분을 직접 차용했고, 다만 8 GB GPU 환경에 맞추기 위해 hidden 차원을 10에서 2로, 그리고 Refinement를 U-Net에서 Conv2d 1개로 축소해서 Cartesian R=4 brain 데이터로 재현 실험을 진행했습니다.

---

## 슬라이드 7 — Mamba 논문 리뷰

두 번째 논문은 SS2D의 base가 되는 Mamba 입니다. Albert Gu — 카네기멜런 — 와 Tri Dao — 프린스턴, FlashAttention으로 유명한 분 — 이 함께 2023년 12월에 arXiv에 올렸고, ICML 2024에서 발표됐습니다.

이 논문의 등장 배경을 짧게 짚으면, 그 이전부터 SSM, Structured State Space Model 계열의 연구가 진행 중이었어요. S4, S5, H3 같은 모델들이 있었는데, 이들은 long-range arena 같은 벤치마크에서는 강했지만 언어 모델링 같은 dense modality에서는 Transformer를 못 따라잡았습니다. 저자들은 그 이유를 "input-content-based reasoning이 안 된다"고 진단했고, 이 한계를 풀기 위한 핵심 기여를 세 축으로 제시합니다.

첫째, Selection Mechanism. SSM의 핵심 파라미터인 (Δ, B, C)를 입력 의존 함수로 만들었습니다. 즉 같은 시퀀스 안에서도 어떤 토큰에서는 정보를 강하게 받아들이고, 어떤 토큰에서는 무시할 수 있게 된 거예요. 이게 "selective"의 의미입니다.

둘째, Hardware-aware Parallel Algorithm. 기존 SSM이 효율적이었던 이유는 파라미터가 시간 불변이라 convolution으로 표현될 수 있었기 때문인데, Mamba는 입력 의존이라 그 효율을 못 쓰게 됩니다. 저자들은 이를 해결하려고 GPU SRAM을 활용하는 fused parallel scan 알고리즘을 직접 작성했고, A100 기준으로 기존 SSM보다 3배 빠르게 만들었습니다.

셋째, 단순화된 아키텍처. Attention이나 MLP 블록 없이 SSM 블록만으로 모델 전체를 구성합니다. 굉장히 깔끔한 디자인이죠.

이 결과 — Transformer 대비 5배 inference throughput, 시퀀스 길이에 대해 linear 복잡도, 그리고 1M 토큰까지 quality가 향상되는 것까지 보였습니다. 언어, 오디오, 게놈 같은 다양한 modality에서 SOTA를 찍었어요.

오른쪽 Figure 1을 보시면 — 입력 x_t가 들어와서 가운데 Project 박스를 거쳐 (Δ, B, C)를 만듭니다. 이게 selection mechanism이고 파란색 화살표예요. 그 위로 A는 정적 파라미터로 유지되고, Discretize 단계에서 (Ā, B̄)로 바뀌어서 SSM 연산을 거칩니다. 우하단 작은 삼각형 그림이 GPU SRAM과 HBM의 메모리 계층인데, 모든 expanded state를 SRAM에 머무르게 해서 IO를 최소화하는 게 hardware-aware의 핵심입니다.

---

## 슬라이드 8 — VMamba / SS2D 논문 리뷰

세 번째는 본 연구가 직접 차용한 SS2D 모듈의 출처 논문입니다. Yue Liu 외 여러 분이 중국 UCAS, Huawei, Pengcheng Lab에서 함께 작성한 VMamba 논문이에요. 2024년 1월에 arXiv에 올라왔고, NeurIPS 2024에서 발표됐습니다.

문제 의식은 명확해요. Mamba는 본래 1D 시퀀스 모델이라서 영상 같은 2D 데이터에 그대로 적용하면 한계가 있습니다. 영상은 본질적으로 ordering이 없는 — non-sequential한 — 구조잖아요. 한 방향으로만 스캔하면 반대 방향의 의존성을 놓치게 되고요.

이 논문의 핵심 contribution이 SS2D, 2D Selective Scan 입니다. 같은 feature map을 4가지 방향 — 좌→우, 우→좌, 위→아래, 아래→위 — 로 스캔한 다음 결과를 합치는 방식이에요. 이렇게 하면 양방향 공간 의존성을 모두 잡을 수 있습니다.

오른쪽 Figure 1(b)가 이 아이디어의 시각적 설명입니다. 자세히 보시면 — 가장 왼쪽 그림에서 빨간 박스로 표시된 동일 토큰이 시작점인데, 청록색과 노란색 화살표가 좌우 양방향으로 뻗어나가요. 가운데 그림은 위아래 양방향이고요. 그리고 마지막 그림에서 4 방향이 다 합쳐진 결과로 — 빨간 박스 토큰이 주변 모든 컨텍스트와 연결되는 — compressed contextual knowledge가 완성됩니다.

이 방식의 장점은 self-attention 없이도 글로벌 receptive field를 얻을 수 있다는 점이에요. 그리고 비용은 quadratic이 아니라 linear입니다. 그 결과 ImageNet에서 VMamba-Base가 top-1 accuracy 83.9%로 Swin Transformer를 0.4% 앞섰고, 처리량은 Swin 대비 40% 이상 빨랐어요. COCO object detection에서는 mAP 47-49%, ADE20K semantic segmentation에서는 mIoU 47-51%로 Swin과 ConvNeXt를 모두 능가했습니다.

본 연구는 이 SS2D 모듈을 ViT 디코더의 보조 경로로 직접 차용했어요. k-space 입력 (B, 32, 320, 320)에 SS2D를 적용하고, 그 출력을 ViT 디코더 출력과 합치는 구조입니다. 다만 원본 VSS block 전체를 가져오기보다는 single-shot SS2D + Conv 합성으로 단순화해서 사용했습니다.

---

## 슬라이드 9 — MambaRecon 논문 리뷰

마지막 네 번째 논문은 MRI 도메인의 동시기 prior art입니다. Yilmaz Korkmaz와 Vishal M. Patel — 둘 다 Johns Hopkins University 소속 — 이 WACV 2025에서 발표한 MambaRecon 입니다.

이 논문이 의미 있는 이유는 — Mamba 기반 MRI 재구성 모델 중 최초의 physics-guided 구조이기 때문입니다. 즉 Data Consistency block을 모델 안에 통합한 unrolled 구조라는 거죠. 기존 CNN 기반 MRI recon 연구에서는 VarNet, E2E-VarNet 같은 unrolled 모델이 주류였는데, 이 흐름을 Mamba 기반으로 가져온 첫 번째 연구입니다.

오른쪽 Figure 1을 보시면 architecture가 잘 정리되어 있어요. 위쪽 큰 흐름 — zero-filled 입력 X_us가 왼쪽에서 들어와서 Patchify를 거쳐 VSSM Block으로 들어갑니다. VSSM Block은 LayerNorm + VSSM 두 번 반복 (skip connection 포함)이고요. 그 다음 DC Block — Unpatchify → Data Consistency → SiLU → Patchify 순서입니다. 이 VSSM + DC 쌍이 6번 반복돼서 마지막에 reconstruction X_r이 나오고, 이걸 fully sampled GT X_fs와 L1 손실로 비교합니다.

아래쪽에는 두 핵심 블록의 상세도가 있어요. 좌하단 VSSM은 Linear → DWConv → SS2D → LayerNorm 순서, 우하단 SS2D는 입력 토큰들을 4 방향으로 unfold한 뒤 (B, C, Δ)를 입력 의존 Linear로 만들고 SSM에 넣어 출력 y를 합니다 (Σ summation 기호).

본 연구와의 핵심 차별점은 두 가지입니다. 첫째, MambaRecon은 multi-iter unrolled 구조 — 즉 6번 반복 — 인데, 본 연구는 ViT encoder + SS2D decoder의 단일 forward 구조입니다. 둘째, 본 연구는 v4에서 1-iter soft DC block을 시도했고 (잠시 후 deep dive 슬라이드에서 다시 다룹니다), 다음 마일스톤은 이 DC를 multi-iter unrolled로 확장해서 MambaRecon식 구조를 본 모델에 도입하는 것입니다. 그런 의미에서 이 논문은 향후 발전 방향의 청사진이라고 할 수 있어요.

---

## 슬라이드 10 — 왜 SS2D? (1) Transformer Self-Attention의 한계

지금까지 각 논문이 무엇이었는지 봤다면, 이제부터 세 슬라이드는 "왜 우리가 SS2D를 선택했는가" 라는 질문에 차례로 답합니다.

첫 번째 이유는 가장 널리 알려진 한계 — Transformer self-attention의 O(N²) 비용입니다.

좌측 수식을 보시면 — Attention(Q, K, V) = softmax(Q Kᵀ / √d) · V. 여기서 Q와 Kᵀ의 내적이 N×N 크기의 attention 행렬을 만들어내고, 이게 시퀀스 길이 N에 대해 quadratic 메모리와 연산을 요구합니다.

이게 본 연구 환경에서 왜 문제가 되는지 구체적으로 짚어볼게요. 320×320 영상을 32×32 patch로 토큰화하면 토큰이 100개입니다. 여기에 32 채널 multi-coil 정보까지 시퀀스로 결합한다고 하면 실효 시퀀스 길이는 더 커지죠. 8 GB GPU에서 N²로 폭증하는 메모리는 학습 자체를 어렵게 만듭니다. 빨간색으로 강조한 마지막 줄처럼 "메모리·학습시간이 모두 N²로 폭증" 이라는 게 정확한 표현이에요.

오른쪽 차트는 cost vs sequence length를 정성적으로 비교한 그림입니다. 빨간 곡선이 O(N²) Transformer attention, 청록 직선이 O(N) Mamba SS2D예요. 시퀀스가 짧을 때는 차이가 거의 없지만 길어질수록 빨간 곡선은 위로 휘어 올라가고 청록 직선은 그대로 선형 — 이 격차가 vision 도메인처럼 토큰이 많은 환경에서 결정적입니다.

물론 Vaswani 외 2017년 Attention is All You Need 논문 이후로 이 N² 문제는 잘 알려져 있었고, 이를 우회하기 위한 시도들 — patch size 키우기, local window attention, linear attention 등 — 이 많이 나왔습니다. 본 연구는 그중에서도 SSM 기반 방식을 선택했고요.

---

## 슬라이드 11 — 왜 SS2D? (2) Mamba — Selective State Space Model

두 번째 이유는 Mamba SSM이 가진 이론적 매력입니다.

좌상단을 먼저 보시면 — SSM은 원래 제어이론에서 출발한 개념입니다. 입력 시퀀스 x(t)를 잠재 상태 h(t)로 누적해서 출력 y(t)를 만드는 구조예요. 연속 시간에서는 h'(t) = A h(t) + B x(t), y(t) = C h(t) — 이게 표준 형태죠. 이걸 학습 가능한 신경망에 쓰려면 이산화해야 하는데, step size Δ로 이산화하면 h_t = Ā h_{t−1} + B̄ x_t, y_t = C h_t로 RNN과 비슷한 점화식이 됩니다.

여기까지가 기존 SSM의 형태이고, Mamba가 추가한 게 selection mechanism입니다. 좌측 하단 작은 글씨로 적어둔 것처럼, Mamba는 (Δ, B, C)를 입력 x_t에 의존하는 함수로 만들었어요.

이게 직관적으로 무슨 의미냐 하면 — 기존 SSM은 모든 토큰에 같은 동역학을 적용했어요. 마치 RNN이 시간에 따라 같은 weight를 반복 적용하는 것처럼요. 그런데 Mamba는 토큰마다 "이번에는 정보를 얼마나 받아들일지(B), 얼마나 출력에 반영할지(C), 이산화 step을 얼마나 크게 할지(Δ)"를 그때그때 다르게 정합니다. 그래서 시퀀스의 어느 토큰을 강하게 기억하고 어떤 토큰을 무시할지 동적으로 선택할 수 있게 되는 거예요. 이게 "selective"의 정확한 의미입니다.

이 추상적 설명을 우상단 Mamba 논문 Algorithm 2에서 구체적으로 확인할 수 있어요. 보시면 1번 줄 A는 그냥 Parameter — 입력과 무관한 상수입니다. 그런데 2번, 3번 줄을 보면 B와 C가 입력 x에 대한 함수 s_B(x), s_C(x) 로 정의돼 있어요. 빨간색으로 강조된 부분입니다. 4번 줄도 마찬가지로 Δ가 입력 의존이고요. 6번 줄에서 SSM 연산이 수행되는데, 이렇게 시간에 따라 변하는 파라미터 때문에 — 빨간색 마지막 줄 — recurrence/scan으로만 처리 가능합니다.

좌측 하단에는 이 모든 특성을 다섯 가지로 요약했어요. Selective — 입력 의존 동역학, Linear O(N) — 시퀀스 길이에 선형, Parallel scan — RNN처럼 순차적이지 않고 GPU에서 병렬화 가능, Hardware-aware — SRAM 활용 fused kernel, Long-range — HiPPO 행렬 기반의 강한 long-context 누적. 이 다섯 가지가 본 연구가 SS2D를 선택한 이론적 근거입니다.

---

## 슬라이드 12 — 왜 SS2D? (3) 4-way 2D Scan과 MRI 적합성

세 번째 이유는 — 위 두 가지 이론적 근거가 본 연구의 도메인인 MRI 재구성과 잘 맞는다는 점입니다.

좌측을 보시면 — Mamba는 본래 1D 시퀀스 모델이라고 말씀드렸어요. 영상에 그대로 적용하면 한 방향 스캔만으로는 양방향 공간 의존성을 잡기 어렵습니다. SS2D가 이 문제를 풀어요. 같은 feature map을 4 방향 — 좌→우, 우→좌, 위→아래, 아래→위 — 로 스캔하고 결과를 합쳐서 2D 글로벌 컨텍스트를 만듭니다. 좌측 하단 4-way Selective Scan 미니 다이어그램이 그 4 방향을 각각 시각화한 거예요.

우측이 본 발표의 핵심 메시지입니다. MRI 재구성 도메인에 SS2D가 잘 맞는 이유 네 가지를 정리했어요.

첫째, R=4 undersampling에서 발생하는 aliasing artifact는 영상 전반에 퍼지는 long-range 패턴입니다. 한 픽셀의 artifact를 풀려면 멀리 떨어진 픽셀의 정보가 필요해요. 즉 전역 컨텍스트가 필수입니다.

둘째, 32 채널 multi-coil 정보를 어떻게 통합하느냐가 fastMRI 같은 multi-coil 데이터의 핵심인데, 채널 간 long-range dependency를 잡는 것이 결정적입니다. SSM의 long-range 누적 특성이 여기에 잘 들어맞아요.

셋째, 8 GB GPU 제약 환경에서 quadratic Transformer는 patch가 작아질수록 비용이 폭증합니다. 더 세밀한 patch를 쓰면 더 좋은 디테일을 얻을 수 있는데 비용이 따라오지 못해요. SS2D는 선형이니까 더 큰 표현력을 가능하게 합니다.

넷째, ViT 디코더와 상보적이라는 점이에요. ViT는 patch 단위 — 즉 32×32 블록 단위 — 의 글로벌 attention을 담당하고, SS2D는 픽셀 단위의 글로벌 dependency를 담당합니다. 이 두 스케일이 합쳐져서 멀티스케일 통합이 자연스럽게 일어나요.

이 네 가지가 본 연구가 SS2D를 ETER의 비교군으로 — 단지 비교하는 게 아니라 적극적으로 더 나은 대안으로 — 선택한 이유입니다.

---

## 슬라이드 13 — SS2D 디코더 구현

이제 실제로 어떻게 구현했는지 보겠습니다.

좌측 본문에 핵심 흐름이 정리돼 있어요. k-space 입력 (B, 32, 320, 320) — 32 채널이 real/imag interleave된 값들 — 에 대해 SS2D 모듈을 적용합니다. 단계는 norm → 1×1 projection → depthwise conv → 4-way SelectiveScan1D → merge 입니다. 1×1 projection은 채널 차원을 d_inner로 맞추는 역할이고, depthwise conv는 지역 문맥을 추가해주는 보조 단계예요. 그 다음 핵심인 4-way SelectiveScan1D를 거치고, 마지막에 합칩니다.

좌측 하단에 하이퍼파라미터를 정리했어요. 내부 projection 차원 d_inner는 32, 잠재 상태 차원 d_state는 8 (HiPPO 기반), 최종 출력 채널은 20. depthwise conv는 3×3 커널, scan direction은 4 방향 양방향 결합이고요. 이 d_inner=32와 d_state=8이 사실 v1에서는 너무 작은 값이었어요 — 잠시 후 실험 변천 슬라이드에서 v4에서 64와 16으로 늘리는 부분을 다시 다룹니다.

우측은 SS2D.forward(in_ksp)의 실제 코드 흐름입니다. 위에서 아래로 — LayerNorm → Linear (d_inner로) → Depthwise Conv 3×3 → 4-way SelectiveScan1D (mamba_ssm CUDA, 보라색 강조) → concat (네 방향 연결) → LayerNorm + Linear merge → out_proj (20 channels) — 이 결과가 마지막에 ViT up_tail, in_imgs와 함께 concat되어 Conv 3×3 합성으로 최종 출력이 됩니다.

좌측 하단 노란 글씨 하나 더 — 의존성 부분이 중요해요. mamba_ssm 패키지의 CUDA selective_scan kernel이 반드시 있어야 합니다. 이건 환경 점검 단계에서 강제로 검증해서, 없으면 학습이 시작도 안 되도록 해뒀어요.

---

## 슬라이드 14 — 학습 설정

학습 설정입니다. 4분면으로 정리했어요.

좌상단 — 옵티마이저와 스케줄러. Adam 옵티마이저, learning rate 2e-4, weight decay는 1e-7로 시작했다가 v5부터 3e-5로 올렸어요. 스케줄러는 v3 이후 CosineAnnealingLR — 단일 부드러운 cosine decay — 를 씁니다. v1과 v2에서는 CosineAnnealingWarmRestarts라는 톱니 LR을 썼는데, 이게 픽셀 단위 정밀 수렴에 해롭다는 가설로 v3에서 바꿨어요. 그런데 이게 정확히 어떻게 작용했는지는 결과가 좀 미묘합니다 — deep dive 슬라이드에서 다시 다뤄요.

우상단 — Mixed Precision 학습. forward는 autocast('cuda')로 fp16에서 돌리고, loss 계산은 autocast 밖으로 빼서 fp32로 합니다. 이렇게 분리한 사유가 — SSIM 함수 내부에 제곱 연산이 있는데, label 값이 약 959 정도면 959²이 약 92만이 되거든요. fp16의 max가 65504니까 overflow가 일어납니다. 그래서 출력 텐서를 .float()로 fp32로 변환한 뒤에 SSIM을 계산하는 방식이에요. GradScaler로 mixed precision backward를 처리하고요.

좌하단 — 손실과 데이터로더. 손실은 L1 + 0.2 × (1 − SSIM). Train loader는 batch size 8, workers 4, prefetch factor 2. Val loader는 batch 4, workers 2로 줄였습니다. 왜 이렇게 줄였냐면 — 시스템 RAM 사용량이 25.4 GB까지 치솟는 걸 관측했어요. 그러면 systemd-oomd가 프로세스를 죽여버리는 일이 생겨서, workers를 제한하지 않으면 학습이 끊깁니다. 그리고 GUI 터미널에서 실행하면 OOM이 나면 터미널까지 같이 사라지는 일이 있어서, tmux로 격리해서 실행하는 걸 권장하고 있어요.

우하단 — 검증과 체크포인트. validation은 train SSIM이 새로운 best를 갱신할 때마다 트리거합니다. 매 epoch마다 하면 시간이 너무 걸려서요. val에서는 PSNR, NMSE, SSIM, L1 네 가지 지표를 계산하고, composite score — best 대비 비율 기반의 종합 점수 — 를 만들어서 best ckpt를 결정합니다. 5 epoch마다 ckpt도 따로 저장하고요. 모든 metric은 wandb에 매 batch 단위로 로깅돼서 ViT-MRI-Recon 프로젝트에서 실시간으로 모니터링이 가능합니다.

---

## 슬라이드 15 — 실험 변천 timeline

이제 실험 변천을 봅니다. 본 연구는 6번의 버전을 거쳤고, 각 버전에서 무엇을 바꿨는지 한 줄씩 정리한 timeline입니다. 여기서는 큰 흐름만 빠르게 잡고, 다음 슬라이드에서 각 버전의 구체적 문제를 깊이 들여다보겠습니다.

가운데 가로축이 시간 흐름이에요. 색깔이 있는 점이 각 버전입니다.

v1 baseline — 50 epoch까지 학습한 결과 val SSIM이 0.5846. 수렴이 안 끝났고, 시각화 결과도 blurry해서 7가지 동시 문제로 진단됐어요.

v2 문제 진단 — v1의 7가지 문제를 체크리스트로 정리하고 일부를 해결한 버전입니다.

v3 스케줄러 변경 — CosineAnnealingWarmRestarts라는 톱니 LR에서 CosineAnnealingLR이라는 단일 부드러운 decay로 교체했습니다.

v4 capacity ↑ + DC block — capacity 증설, regularization, Data Consistency 세 축을 동시에 적용했어요. 본 연구의 메인 모델입니다.

v5 regularization + EarlyStop — dropout과 weight_decay를 강화하고 EarlyStopping을 도입했고, 데이터 분포도 넓혔습니다.

v6 resume + skimage SSIM — 운영 안정화 단계로, 체크포인트 이어 학습 기능과 표준화된 SSIM validation을 추가했어요.

이렇게 여섯 개 버전을 거치면서 점진적으로 발전한 흐름이고, 각 버전 안에서 정확히 어떤 문제를 어떻게 진단하고 처방했는지 — 다음 슬라이드에서 본격적으로 들어갑니다.

---

## 슬라이드 16 — 버전별 문제 진단 deep dive

이 슬라이드가 본 발표에서 가장 중요한 슬라이드입니다. 6개 카드로 v1부터 v6까지의 진단과 처방을 정리했어요.

**v1 baseline (빨간색) — "blurry recon, 7가지 동시 문제"**

50 epoch까지 학습했지만 val SSIM이 0.5846으로 수렴이 안 됐고, 시각화를 보니 복원 영상이 입력보다 오히려 더 흐릿해진 — blurry해진 — 결과가 나왔어요. 신기하게도 PSNR은 35.68 dB로 나쁘지 않았는데, 이건 평균적으로 GT와 밝기가 가깝기만 해도 PSNR이 올라가는 metric의 특성 때문이에요. 실제 지각적 품질은 SSIM이 더 정확히 반영하고요.

원인 분석을 통해 7가지 문제를 진단했고, 그중 4가지를 카드에 정리했어요.

첫째, patch가 32×32라서 320×320 영상을 토큰 100개로 표현하는데, 각 patch 내부 32×32 × 32 채널 = 32,768 값이 384차원 토큰으로 압축됩니다. 압축비 85대 1이에요. 의료영상에서 SSIM과 시각 품질을 결정하는 edge, texture, 조직 경계 같은 fine detail이 인코딩 단계에서 이미 비가역적으로 손실되는 거죠.

둘째, 최종 합성이 Conv2d 한 개입니다. 308 채널에서 1 채널로 가는 conv 한 개, 파라미터 2,773개에 불과해요. ViT 브랜치, in_imgs, SS2D 브랜치의 풍부한 정보를 비선형적으로 결합할 능력이 없습니다. 사실상 가중 평균이라서 결과가 평균값 — 즉 blurry — 에 수렴해요.

셋째, SSIM weight가 0.2인데, 실제로 손실 전체에서 SSIM이 차지하는 비중이 0.7%에 불과했어요. L1이 99.3%를 차지하니까, 모델은 사실상 L1만 최소화하고 있었고 — L1 최소화는 모든 가능한 출력의 픽셀별 평균이 되니까 blurry output을 선호합니다.

넷째, SS2D의 capacity가 d_inner=32, d_state=8로 매우 작았어요. 참고로 VMamba는 분류 작업에서도 d_inner를 192 이상으로 씁니다. 320×320 MRI의 복잡한 k-space 상관관계를 잡기에는 턱없이 부족했죠.

거기에 더해 train SSIM은 계속 올라가지만 val SSIM은 epoch 100 이후 정체되는 — gap 0.066의 — 과적합도 보였습니다.

**v2 구조 보강 (앰버) — "7개 중 4개 해결"**

v1의 7가지 문제 중 4가지를 한꺼번에 해결한 버전입니다.

patch size를 32×32에서 16×16으로 줄여서 패치 수를 100에서 400으로 늘렸어요. 압축비도 85대 1에서 21대 1로 떨어져서 공간 정보가 4배 더 보존됩니다.

최종 합성을 Conv2d 1개에서 RefinementBlock — 3개의 ResBlock으로 구성된 — 으로 바꿨어요. 파라미터가 2,773개에서 약 12만 개로 늘었고, 비선형 특징 결합과 점진적 정제가 가능해졌습니다.

SSIM weight를 0.2에서 1.0으로 올려서 손실 기여도를 0.7%에서 약 3.5%로 끌어올렸어요. 구조 보존 학습이 실질적으로 작동하게 된 거죠.

ETER 쪽에서는 GRU hidden을 2에서 4로 늘려서 표현력을 두 배로 만들었습니다.

다만 잔여 문제는 두 가지였어요. SS2D capacity 천장이 여전했고, Data Consistency가 없었습니다.

**v3 스케줄러 교체 (보라) — "WarmRestarts → CosineAnnealingLR"**

이 버전에서는 모델은 그대로 두고 스케줄러만 바꿨습니다. 동기는 — 기존 CosineAnnealingWarmRestarts가 epoch 1, 3, 7, 15, 31, 63, 127에서 LR을 최댓값으로 되돌리는 톱니 패턴이었는데, 이게 픽셀 단위 정밀 수렴이 중요한 MRI 재구성에서는 득보다 실이 클 수 있다는 가설이었어요.

그래서 CosineAnnealingLR — 200 epoch 동안 단일 cosine decay, 즉 2e-4에서 1e-6으로 한 번만 부드럽게 감소 — 로 교체했습니다.

그런데 흥미로운 발견이 있었어요. SS2D는 안정화됐는데, ETER에서는 오히려 성능이 회귀했습니다. v3 best 0.7475가 v4 best 0.7320으로 떨어진 거예요. 사후 분석으로는, WarmRestarts의 LR 재가열이 saddle escape 역할을 하고 있었던 것 같습니다. 단일 cosine decay는 평탄한 minimum에서 모델이 못 빠져나오는 효과가 있을 수 있어요. 이건 향후 ETER 실험에서 WarmRestarts와 EarlyStop 조합으로 다시 시도해볼 가치가 있는 부분입니다.

**v4 capacity ↑ + reg + DC (틸) — "A·B·C 세 축 동시 적용"**

본 연구의 메인 모델입니다. v1 분석 체크리스트에서 v3까지 미해결이던 항목을 동시에 공격했어요.

A. Capacity. SS2D의 d_inner를 32에서 64로, d_state를 8에서 16으로 두 배씩 늘렸습니다.

B. Regularization. weight_decay를 1e-7에서 1e-5로 올렸고, Transformer decoder에 dropout 0.1을 추가했습니다.

C. Data Consistency block. 1-iteration soft DC를 추가했어요. 흐름은 — 모델이 추정한 영상을 sens map과 곱해서 multicoil 이미지로 만든 뒤, FFT를 통해 k-space로 변환하고, 측정된 k-space와 비교해서 mask 영역에 학습 가능한 α 가중치로 보정하고, 다시 iFFT하고 sens conjugate로 coil-combine하는 — physics-aware한 단계입니다. sens map은 ACS 영역의 저주파만 사용해서 추정했어요.

부작용도 있었습니다. d_inner=64로 capacity를 늘리니까 SS2D의 merge 단계 LayerNorm에서 약 800 MiB 임시 버퍼가 필요해져서 첫 batch에서 OOM이 났어요. 해결책으로 SS2D forward에 gradient checkpointing을 넣었고, 그래도 backward에서 다시 OOM이 나서 batch size를 8에서 4로 절반 줄였습니다. 그 과정 자체가 8 GB GPU 한계가 어디까지인지 보여준 셈이고요.

결과는 — val SSIM 0.7340. fastMRI Pretrained U-Net의 0.8865 대비 약 15% 갭이 남아있는 상태예요. 이 갭이 다음 두 버전의 동기가 됩니다.

**v5 데이터 분포 + 일반화 (네이비)**

v4 결과 분석에서 두 가지 가설이 나왔어요. 첫째, 학습 분포가 좁다 — 우리가 320×320 정합 파일만 필터링했더니 실제 학습 슬라이스가 fastMRI brain 코퍼스의 약 60%밖에 안 됐거든요. 둘째, 일반화가 부족하다 — train-val gap이 0.055로 capacity ceiling에 도달한 상태에서 val이 정체됐습니다.

처방은 네 가지입니다.

데이터 — 사이즈 필터를 완화하고 image-domain center-crop이나 zero-pad로 다양한 사이즈 파일을 모두 흡수했어요. 그 결과 train이 +67% (8,548에서 14,262 슬라이스), val이 +62% (4,492에서 7,270 슬라이스) 회복됐습니다. 이 val 7,270이라는 숫자가 중요한데, fastMRI Pretrained U-Net 평가셋과 정확히 같은 슬라이스 모집단이에요. 그래서 직접적인 leaderboard 비교가 가능해집니다.

Dropout을 0.1에서 0.2로, weight_decay를 1e-5에서 3e-5로 강화했고요.

H/V flip augmentation을 image domain에서 추가했습니다. mask는 W축 1D 패턴이라 image flip과 독립적이고, GT도 같이 flip해서 일관성을 유지했어요.

EarlyStopping을 도입했어요. patience 5로, 약 50 epoch 무개선 시 학습을 정지합니다. v4가 epoch 30~40에 피크였던 패턴 기준 충분한 마진이에요.

**v6 운영 안정화 (앰버)**

마지막 버전은 운영 측면 정비입니다.

체크포인트 이어 학습 — resume capability — 이 추가됐어요. 학습이 끊겨도 마지막 ckpt에서 이어갈 수 있게 됐습니다.

SSIM validation을 skimage 라이브러리 기반으로 통일했어요. 기존엔 자체 구현한 SSIM이었는데, 표준 라이브러리로 바꿔서 다른 연구와 직접 비교할 수 있게 한 거죠.

EarlyStopping을 SSIM 중심으로 통일했고, ETER v6도 동일한 레시피를 공유하도록 만들었습니다.

이 정도가 v1부터 v6까지의 흐름이고, 다음 슬라이드부터 실제 결과를 보겠습니다.

---

## 슬라이드 17 — 결과 (1) 정량 평가

이제 결과입니다. vis_compare_v5의 12개 샘플 평균이고, R=4 brain320 val set 기준입니다.

표를 위에서부터 보시면 —

U-Net Pretrained baseline. MICCAI fastMRI 공식 weights를 사용한 모델이고, PSNR 34.92 dB ± 2.00, SSIM 0.8964 ± 0.0891. 이게 본 연구의 reference point예요.

SS2D-ViT v4. 본 연구의 메인 모델인 Mamba decoder입니다. PSNR 33.35 dB ± 1.70, SSIM 0.8765 ± 0.0620. U-Net 대비 −1.57 dB, SSIM 차이 −0.020.

ETER-ViT v4. 비교군인 Bi-GRU decoder입니다. PSNR 30.75 dB ± 1.80, SSIM 0.8612 ± 0.0379. U-Net 대비 −4.17 dB, SSIM 차이 −0.035.

표 아래 INSIGHT 박스가 핵심 메시지예요. 두 가지를 짚고 있습니다.

첫째 — SS2D-ViT v4가 ETER-ViT v4 대비 PSNR을 +2.60 dB, SSIM을 +0.015 개선했습니다. 이게 왜 의미 있냐면 — 두 모델이 ViT 인코더, ViT 디코더, RefinementBlock 등 거의 모든 구성요소를 공유하고 있고, 차이는 시퀀스 모듈 한 곳뿐이에요. 즉 이 +2.60 dB은 순수하게 시퀀스 모듈이 GRU냐 SS2D냐의 차이에서 오는 것입니다. Selective SSM의 long-range 모델링 효과가 양방향 GRU 대비 분명한 우수성을 보였다는 결과예요.

둘째 — 다만 fastMRI Pretrained U-Net 대비로는 여전히 −1.57 dB의 격차가 남아있습니다. 이건 모델 구조의 한계라기보다는 — ViT를 random initialization에서 from-scratch로 학습한 것의 한계예요. 의료영상은 데이터가 상대적으로 적어서 ImageNet에서 사전학습된 표현이 없으면 underfit이 일어나기 쉽습니다. 이 갭은 한계 슬라이드에서 다시 다루고, 향후 MAE pretrain 같은 방법으로 좁힐 계획이에요.

---

## 슬라이드 18 — 결과 (2) 정성 비교 첫 번째

정량 결과 옆에 정성 결과도 봐야 진짜 차이가 보입니다.

비교 모델은 세 가지 — SS2D-ViT v4, ETER-ViT v4, U-Net Pretrained. 위에 굵은 글씨로 정리해뒀습니다.

각 row 형식을 잠깐 설명드리면 — 위쪽 4개 이미지는 GT, SS2D 복원, ETER 복원, U-Net 복원 순서이고, 아래쪽 4개는 각각의 error map (GT − recon)입니다. error map은 빨간색이 진할수록 에러가 큰 영역이에요.

좌측 샘플 #0과 우측 샘플 #3304, 두 슬라이스를 보여드리고 있어요.

복원 영상부터 보시면 — 사실 셋 다 육안으로는 굉장히 비슷해 보입니다. 정량 차이가 크지만 정성으로는 미묘한 수준이에요. 그런데 아래 error map을 보면 차이가 명확합니다. ETER의 error map이 가장 빨갛게 진하고, 특히 뇌 구조 경계부 — 즉 회백질과 백질 경계, 측뇌실 주변 — 에 강한 에러 패턴이 집중돼 있어요. SS2D는 ETER보다는 에러가 분산되어 있고 강도도 약합니다. U-Net은 가장 깨끗하고요.

이 패턴이 의미하는 건 — ETER의 hidden=2 capacity가 정밀한 경계 복원에 부족하다는 거예요. SS2D는 같은 8 GB 환경에서도 더 큰 capacity를 활용해 경계를 더 잘 보존했고, U-Net은 pretrained 가중치 덕분에 가장 좋은 결과를 보여준 거죠.

---

## 슬라이드 19 — 결과 (3) 정성 비교 두 번째

다른 샘플에서도 같은 경향이 나타나는지 확인해야 결과가 안정적이라고 할 수 있죠. 그래서 두 번째 정성 비교 슬라이드를 넣었습니다.

같은 형식이고, 샘플은 #4625와 #6608 두 개입니다.

여기서도 동일한 경향을 확인할 수 있어요. 디코더 변형 — SS2D vs ETER — 의 차이가 다양한 샘플에서 일관되게 나타나고, U-Net과의 격차도 일관됩니다. 즉 정량 결과의 +2.60 dB가 우연히 운 좋은 샘플에서 나온 게 아니라, 모델 구조에서 비롯된 안정적인 차이라는 뜻입니다.

이 두 슬라이드에서 보여드린 4개 샘플은 vis_compare_v5의 12개 샘플 중 4개를 추렸고, 나머지 8개도 — 4625와 6608을 포함해서 — 같은 경향이 일관되게 나타납니다.

---

## 슬라이드 20 — 한계와 향후 과제

결과를 정직하게 보고 나면 한계가 명확해집니다. 두 컬럼으로 정리했어요.

좌측, 현재 한계 네 가지입니다.

첫째 — ViT random-init from scratch의 한계입니다. Pretrained U-Net 대비 −1.57 dB의 갭은 결국 사전학습 가중치 유무에서 옵니다. 의료영상은 데이터가 상대적으로 적기 때문에, ImageNet이나 fastMRI 자체에서 사전학습한 표현이 없이는 ViT가 충분히 학습되기 어려워요.

둘째 — patch 32×32의 정보 손실. 320을 32로 나누면 10×10 patch grid인데, patch 내부 디테일이 토큰화 과정에서 손실됩니다. 이게 blurry recon 경향의 한 축이고, v2에서 16×16으로 줄여봤지만 8 GB GPU에서 수렴까지 끌고 가기 어려워서 다시 32×32로 돌아갔어요.

셋째 — 8 GB GPU capacity 천장. hidden=2, d_inner=32 등 축소된 상태로만 학습이 가능했고, 본래 ETER-Net의 hidden=10 대비 표현력이 부족합니다. 이 한계가 ETER 쪽에서 특히 두드러져요.

넷째 — composite score의 한계. best 대비 비율 기반이라 초반에 큰 점프가 한 번 일어나면 그 이후 갱신이 잘 안 됩니다. 그래서 *_best.pt가 epoch 10 시점의 ckpt일 수 있어요. 마지막 epoch ckpt가 실제로는 더 나은 경우가 많아서, 평가 시 이 부분도 고려해야 합니다.

우측, 향후 과제 네 가지입니다.

첫째 — MAE pretrain 적용. ViT 인코더를 fastMRI 전체 코퍼스로 self-supervised pretrain한 뒤 fine-tune하면 from-scratch 격차를 회복할 수 있을 거예요. 이게 가장 큰 효과를 기대하는 방향입니다.

둘째 — patch overlap 또는 더 작은 patch. 16×16 + overlap 같은 조합으로 디테일을 보존하는 거죠. SS2D가 선형 비용이니까 patch를 작게 만드는 시도가 가능합니다.

셋째 — 강한 Data Consistency. v4에서 시도한 1-iter soft DC를 multi-iter unrolled DC로 확장하는 겁니다. VarNet, E2E-VarNet, MambaRecon 식 구조죠. 앞에서 본 MambaRecon이 좋은 청사진입니다.

넷째 — R=8, R=16 일반화. R=4보다 어려운 가속비에서 SS2D의 우수성이 더 두드러질 가능성이 있어요. ETER 같은 단순 시퀀스 모델은 더 많은 정보 손실을 다루기 어려운데, SS2D의 long-range 모델링이 그 환경에서 더 빛날 수 있습니다.

---

## 슬라이드 21 — 결론과 다음 단계

마지막 슬라이드입니다. 결론은 단순합니다 — 환경 업그레이드가 필요합니다.

상단 빨간색 박스를 먼저 보시면 — 현재 환경의 한계를 정리했어요. 개인 데스크톱 RTX 5060 Ti 8 GB 환경에서는 hidden=2, d_inner=32 등 축소된 capacity로만 학습이 가능했고, 원본 ETER-Net의 hidden=10이나 더 큰 SS2D 설정의 완전한 성능을 재현하지 못했습니다. 본 연구를 한 단계 발전시키려면 더 큰 GPU 메모리와 연산 자원이 가능한 새로운 환경이 필수예요.

그 아래 두 가지 옵션을 검토하고 있습니다.

**옵션 1 — 학교의 남는 데스크톱 활용**

장점은 명확합니다. 추가 비용 없이 즉시 가용하고, 장기간 점유 가능해서 긴 학습 일정을 운용하기 좋아요. 그리고 현재 8 GB보다 큰 GPU 메모리가 확보되면 hidden과 d_inner를 원래 ETER-Net 수준 — 또는 그 이상 — 으로 원복해서 capacity 문제를 정면으로 풀 수 있습니다.

주의할 점은 — 단기에 가용한 데스크톱의 GPU 사양 확인이 우선입니다. 8 GB보다 작거나 비슷하면 의미가 없으니까요.

**옵션 2 — 학습용 서버 임대**

장점은 즉각성과 고사양이에요. 필요한 시점에 A100, H100 급 GPU를 즉시 사용할 수 있고, v6의 chain 자동화 기능을 활용하면 병렬 실험을 돌려서 일정을 단축할 수 있습니다. vast.ai, Lambda, runpod 등 단기 임대 옵션이 다양해요.

주의할 점은 비용입니다. 지속 사용하면 비용이 누적되니까 실험 일정을 압축하고 한 번에 끝내는 전략이 필요해요.

이 두 옵션 중에서 — 사실 어느 쪽이 더 좋다고 단정하기 어렵습니다. 단기 실험이 많고 빠른 iteration이 중요하면 옵션 2가 좋고, 장기적으로 같은 환경에서 여러 실험을 돌릴 거면 옵션 1이 좋고요. 두 옵션 다 시도해보면서 가장 효율적인 환경을 찾아갈 계획입니다.

이상으로 ViT 기반 fastMRI Brain 재구성 발표를 마치겠습니다. 질문 있으시면 답변드리겠습니다. 감사합니다.

---

## 발표 팁 (시간 무관)

- **슬라이드 6-9 (논문 리뷰)** 는 청중이 prior art에 익숙하지 않을 수 있으니 — "이 논문이 나왔을 때 어떤 문제를 풀려고 했는지" 한 줄씩 짚으면서 진행하면 따라오기 편함.
- **슬라이드 11 (Selective SSM 수식)** 에서 "Δ가 입력 의존이 된다" 부분은 화이트보드에 손짓으로 "토큰마다 다른 값" 이라고 한 번 더 강조하면 청중이 이해하기 쉬움.
- **슬라이드 16 (deep dive)** 6개 카드를 빠르게 넘기지 말고, 색깔 컬러 strip이 v1=빨강(문제 제기), v2-v6=점차 진단·처방으로 넘어가는 흐름이라는 걸 한 번 짚어주면 구조가 잘 보임.
- **결과 슬라이드 (17, 18, 19)** 에서 숫자만 읽지 말고 — 17번에서는 "디코더만 바꿔서 +2.60 dB" 가 핵심 메시지, 18-19에서는 "error map 색깔이 ETER가 가장 진하고 SS2D가 그 다음" 을 손가락으로 가리키면서 설명.
- **결론 (21)** 에서 두 옵션의 트레이드오프를 정직하게 말하고, "여러분이라면 어떻게 하시겠습니까?" 같은 질문으로 마무리해도 자연스러움.
