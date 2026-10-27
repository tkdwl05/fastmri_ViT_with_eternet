"""없는 인자 3개짜리 myOptimizer 의 대체 (새 파일; 교수님 myUtils.py 무수정).

사실
- d-ETER 학습 스크립트는 `optimizer = myOptimizer(eternet, OPTIMIZER, LEARNING_RATE_ADAM)` 로 인자 3개를 넘긴다(231030:191).
  바로 아래 주석 줄은 `torch.optim.Adam(eternet.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)` 이다(231030:192).
- 저장소의 myOptimizer 는 인자 2개(model, flag_optim)이고 분기마다 lr 을 박아 둔다(scripts_legacy/myUtils.py:21).
  'adam' 분기 = Adam lr 1e-7(:84-88), 'else' 분기 = Adam lr 1e-4(:104-106). 어느 분기에도 weight decay 가 없다.
  저장소 판에 인자 3개를 넘기면 TypeError 가 난다. git 전 이력과 보이는 디스크에 인자 3개 판은 없다.
- o-ETER 스크립트는 myOptimizer 를 쓰지 않고 `torch.optim.Adam(..., lr=LEARNING_RATE_ADAM, weight_decay=LAMBDA_REGULAR_PER_PIXEL)`
  (결합형 L2)을 쓴다(231022:198, random 231027:187).
- 잔재 config 의 OPTIMIZER 는 'else'(이중 주석, configs/myConfig_temp.py:186)다.

가정(확인 불가 — config 에 '추정'으로 표시)
- mode='lr_passthrough'(기본): 저장소 판과 같은 분기·같은 나머지 인자를 쓰고, lr 만 넘겨받은 값으로 바꾼다.
  weight decay 는 저장소 판처럼 없다. OPTIMIZER='else', LR=1e-4 이면 저장소 판과 결과가 같다(Adam lr 1e-4, WD 0).
- mode='legacy_2arg': 넘겨받은 lr 을 버리고 저장소 판(인자 2개)을 그대로 부른다.
- mode='coupled_wd': torch.optim.Adam(params, lr=lr, weight_decay=weight_decay) — 231030:192 주석 줄과 o-ETER 스크립트 형태.
- foreach: None 이면 torch 기본(이 서버 torch 2.3.1 은 CUDA 텐서에 foreach 구현). 교수님 당시 torch 판은 모른다.
  foreach 와 for-loop 구현은 같은 알고리즘이고 마지막 자리 반올림만 다를 수 있다. foreach=False 는 step 의 임시 메모리를 줄인다.
"""
import torch

from repro_prof import prof_paths as P

MODES = ('lr_passthrough', 'legacy_2arg', 'coupled_wd')

# 저장소 판 분기의 lr 이외 인자(scripts_legacy/myUtils.py:23-106 그대로)
_BRANCHES = {
    'sgd': lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.0, dampening=0, weight_decay=0.0, nesterov=False),
    'nesterov': lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9, dampening=0, weight_decay=0.0, nesterov=True),
    'asgd': lambda p, lr: torch.optim.ASGD(p, lr=lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.0),
    'rmsprop': lambda p, lr: torch.optim.RMSprop(p, lr=lr, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0, centered=False),
    'rprop': lambda p, lr: torch.optim.Rprop(p, lr=lr, etas=(0.5, 1.2), step_sizes=(1e-06, 50)),
    'adagrad': lambda p, lr: torch.optim.Adagrad(p, lr=lr, lr_decay=0.1, weight_decay=0, initial_accumulator_value=0),
    'adadelta': lambda p, lr: torch.optim.Adadelta(p, lr=lr, rho=0.9, eps=1e-06, weight_decay=0),
    'adamax': lambda p, lr: torch.optim.Adamax(p, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
}


def myOptimizer(model, flag_optim, lr=None, *, mode='lr_passthrough', weight_decay=0.0, foreach=None):
    """교수님 호출 `myOptimizer(eternet, OPTIMIZER, LEARNING_RATE_ADAM)` 의 대체."""
    if mode not in MODES:
        raise ValueError(mode)
    print('\t OPTIMIZER : %s  (repro shim mode=%s)' % (flag_optim, mode))
    if mode == 'legacy_2arg':
        opt = P.prof_utils_module().myOptimizer(model, flag_optim)   # 인자 2개 원본 그대로(lr 무시)
        if foreach is not None:
            for g in opt.param_groups:
                g['foreach'] = foreach
        return opt
    if mode == 'coupled_wd':
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, foreach=foreach)
        print('\t adam (coupled L2), lr {}, weight_decay {}'.format(lr, weight_decay))
        return opt
    # lr_passthrough
    params = model.parameters()
    if flag_optim == 'sparseadam':
        return torch.optim.SparseAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08)
    if flag_optim in _BRANCHES:
        opt = _BRANCHES[flag_optim](params, lr)
        print('\t {}, lr {}'.format(flag_optim, lr))
        return opt
    # 'adam' 분기(:84-88)와 그 밖의 값 = 'else' 분기(:104-106): 둘 다 Adam, 기본 betas·eps, WD 없음
    opt = torch.optim.Adam(params, lr=lr, foreach=foreach)
    print('\t {}(adam), lr {}'.format('adam' if flag_optim == 'adam' else 'else', lr))
    return opt
