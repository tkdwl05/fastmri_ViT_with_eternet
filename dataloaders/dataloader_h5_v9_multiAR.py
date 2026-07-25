"""
Multi-AR dataloader (v9 radapt) — 학습 시 배치별(샘플별) 가속률 R 랜덤 노출.

단일-R 학습이 그 R 의 aliasing 구조를 외워 OOD(다른 R)에서 파국적으로 무너지는 문제(§4.2-⑤)를
완화하기 위해, `__getitem__` 마다 R∈AR_CHOICES 를 랜덤 선택한다. R-대응(radapt) 학습의 핵심.

**원본 `dataloader_h5_v5.FastMRI_H5_Dataloader` 무수정** — 얇은 서브클래스로 `self.acceleration` 만
매 __getitem__ 직전 랜덤 세팅 후 부모 로직 호출. 부모의 mask 생성(`build_r4_mask`)·masked k-space·
반환 'mask'(샘플링 패턴) 가 모두 그 R 을 반영한다. ACS(center_fraction 기반)는 R-무관.

주의: DataLoader worker 는 별도 프로세스(각자 dataset 복사본) + __getitem__ 순차 호출이라
self.acceleration 의 per-call 뮤테이션은 안전(경합 없음).
"""

import numpy as np

from dataloader_h5_v5 import FastMRI_H5_Dataloader


class FastMRI_H5_MultiAR(FastMRI_H5_Dataloader):
    def __init__(self, *args, ar_choices=(2, 3, 4, 5, 6, 8), **kwargs):
        # random_mask=True 권장(offset 도 랜덤). 부모가 rng 를 만들지 않는 조합 대비 아래서 보강.
        super().__init__(*args, **kwargs)
        self.ar_choices = tuple(int(r) for r in ar_choices)
        if self.rng is None:
            self.rng = np.random.default_rng()
        print(f"    [MultiAR] per-sample R ∈ {self.ar_choices} (랜덤 노출)")

    def __getitem__(self, idx):
        # 이 호출에 쓸 R 을 랜덤 선택 → 부모 __getitem__ 이 build_r4_mask/mask_out 에 반영
        self.acceleration = int(self.rng.choice(self.ar_choices))
        return super().__getitem__(idx)
