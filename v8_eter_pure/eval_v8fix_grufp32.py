"""eval_v8fix.py 를 ETER-net(GRU) 의 k-space 쪽 GRU(gru_h)를 fp32 로 계산해 실행한다 (2026-10-10).

학습(main_train_pure_v8fix_ampguard.py, GRU_FP32_MODULES 기본 gru_h)과 같은 정밀도로 평가하기 위한 얇은 래퍼 — 인자는
eval_v8fix.py 와 같다. 이유는 gru_precision.py 주석. 환경 변수 GRU_FP32_MODULES(기본 gru_h, 쉼표 구분).
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_HERE)
sys.path.append(os.path.join(os.path.dirname(_HERE), 'models', 'pure_eternet'))
import gru_precision  # noqa: E402

_names = tuple(n for n in os.environ.get('GRU_FP32_MODULES', 'gru_h').split(',') if n)
gru_precision.install_class_patch(_names)
print(f'[eval_v8fix_grufp32] PureETER_GRU {_names} → fp32')
sys.argv[0] = os.path.join(_HERE, 'eval_v8fix.py')
runpy.run_path(sys.argv[0], run_name='__main__')
