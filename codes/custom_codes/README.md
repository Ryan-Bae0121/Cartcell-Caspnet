# custom_codes — cartcell rebuild (2차: Phase 3 Multi-branch CapsNet 까지)

`cartcell_rebuild_briefing.md` 스펙을 근거로 CapsNet-MHC 베이스 위에 재구현한 MHC-II
peptide binding 예측 코드.

- **1차 완료** = 데이터 파이프라인 + Phase 2 모델(CNN + Core Register) + 5-fold 학습/평가.
- **2차 완료** = Phase 3 모델(Multi-branch CapsNet + Dynamic Routing), Phase2 체크포인트
  fold별 warm-start. **전체 학습(5-fold × 40ep, CPU)은 미실행** — 스모크 테스트만 통과.
- 3차(다음 단계) = Ensemble(α 스윕, OOF) → per-allele(≥30) → LOMO(≥50, 60 allele) → figure 재생성.

## 구성

| 파일 | 역할 |
|---|---|
| `seq_encoding.py` | BLOSUM62 23차원 (peptide, max_len 25) / 축소 10차원 (allele pseudo-seq, 34) |
| `mhc_pseudo.py` | allele 이름 정규화 + pseudo-sequence 조회. **실서열 미보유 → placeholder** (briefing 1.4 미해결) |
| `data_pipeline.py` | raw CSV → 부등호 변환 → affinity 정규화 → label → 길이필터 → dedup → stratified split |
| `data_provider.py` | TSV 로드, 인코딩 캐시, 배치 iterator (CPU 텐서) |
| `models/common.py` | `CoreRegister`(9-mer soft binding core), `PseudoSeqEncoder`, `DynamicRouting`(squash + routing-by-agreement, Phase 3용) |
| `models/phase2.py` | `Phase2Model` — CNN + Core Register → (cls_logit, reg_yhat, pos_prob) |
| `models/phase3.py` | `Phase3Model` — 3-branch(56 primary capsule) → Dynamic Routing ×3 → (cls_logit, reg_yhat, pos_prob, coupling_cls, coupling_reg). `warm_start_from_phase2()` 로 Phase2 체크포인트 이식 |
| `losses.py` | `Phase2Loss`: `L = L_cls(BCE, pos_weight) + λ·L_reg(MSE) + β·entropy(pos_prob)`. `Phase3Loss` = 동일 목적함수, 기본 λ=0.5 |
| `train_phase2.py` | 5-fold CV, OOF 수집, 체크포인트, summary JSON, test 예측 (λ=0.1, 30 epoch) |
| `train_phase3.py` | 동일 구조 + fold별 Phase2 checkpoint warm-start (λ=0.5, 40 epoch) |
| `evaluate.py` | AUROC / AUPRC / Pearson r / MSE, per-allele (≥30 샘플) |
| `config/data.json`, `config/phase2.json`, `config/phase3.json` | 설정 |

## 실행

```bash
cd codes/custom_codes

# 1) 데이터셋 생성  -> custom_dataset/Anthem_dataset/{train_data.txt,test_data.txt,mhc_ii_pseudo.csv}
bash run_pipeline.sh

# 2) 스모크 테스트
python models/phase2.py --selftest
python models/phase3.py --selftest
python data_provider.py
EPOCHS=2 N_FOLDS=2 bash run_train_phase2.sh
EPOCHS=2 N_FOLDS=2 bash run_train_phase3.sh   # Phase2 ckpt 없으면 fold별로 scratch로 자동 폴백

# 3) 전체 학습 (CPU)
bash run_train_phase2.sh                      # 5 folds x 30 epochs, 먼저 실행해야 warm-start 가능
bash run_train_phase3.sh                      # 5 folds x 40 epochs, phase2_fold{k}.pt 로 자동 warm-start
```

Phase3 는 fold `k`의 train/val split이 Phase2와 같은 `KFold(seed=42, n_folds=5)`라서
`reports_phase2_affinity_from_scratch/checkpoints/phase2_fold{k}.pt` 가 존재하면
그 fold를 그대로 warm-start 한다. 없으면 (아직 Phase2 전체 학습 전이면) 경고를 찍고
scratch로 학습 — 실행이 막히지는 않지만, briefing §2 근거(회귀 안정성: full warm-start
std 0.022 vs core-only 0.067)를 얻으려면 Phase2 전체 학습을 먼저 끝내야 한다.

```bash
WARM_START=0 python train_phase3.py                 # warm-start 끄고 scratch로
WARM_START_CORE_ONLY=1 python train_phase3.py        # 구 버그 재현용(core만, MHC 스킵)
PHASE2_CKPT_DIR=/other/path python train_phase3.py   # 다른 Phase2 run 사용
```

산출물:
- `reports_phase2_affinity_from_scratch/` — `phase2_affinity_from_scratch_summary.json`,
  `oof_predictions.tsv`, `test_predictions.tsv`, `test_metrics.json`, `checkpoints/phase2_fold{k}.pt`
- `reports_phase3_capsnet/` — `phase3_capsnet_summary.json`, `oof_predictions.tsv`,
  `test_predictions.tsv`, `test_metrics.json`, `checkpoints/phase3_fold{k}.pt`

## Phase 3 아키텍처 메모

- 56 primary capsule = full-peptide branch(pep_stem → GAP → Linear proj, **32개**) +
  core branch(pep_stem → `CoreRegister` → Linear proj, **16개**) +
  MHC branch(`pseudo_encoder` → Linear proj, **8개**). capsule 차원(`cap_dim`)은
  briefing에 명시 안 돼 있어 표준 CapsNet 관례를 따라 **primary 8차원 / 출력 16차원**으로
  결정 (`config/phase3.json`에서 조정 가능).
- `DynamicRouting`(`models/common.py`) 이 56개 primary capsule → 3개 출력 capsule
  (음성/양성 class capsule 2개 + regression capsule 1개)로 routing-by-agreement 3회 반복.
  `cls_logit = ||v_pos|| - ||v_neg||`, `reg_yhat = Linear(v_reg)`.
- `pep_stem` / `core` / `pseudo_encoder` 모듈명이 Phase2Model과 완전히 동일 — briefing이
  지적한 `mhc_encoder` vs `pseudo_encoder` 이름 불일치 버그를 처음부터 없앤 설계.
  `warm_start_from_phase2()` 가 이름+shape 매칭되는 텐서만 골라 복사(strict 아님, 안전).

## 알려진 한계

- **MHC-II pseudo-sequence 가 placeholder** (allele 해시 기반 결정적 더미). 실제 IMGT/HLA
  유래 서열을 `custom_dataset/Anthem_dataset/mhc_ii_pseudo.csv` 의 해당 행에 넣고 `source`
  컬럼을 `real` 로 바꾼 뒤 재학습해야 의미 있는 성능이 나옴. 그 전까지 수치는 배관 검증용.
- `data.json` 의 `collapse_dra: true` → `HLA-DRA*01:01/DRB1*xx` 와 `HLA-DRB1*xx` 를 같은
  allele 로 병합 (DRA 단형). 이 때문에 allele 수가 브리핑의 101 → 85 로 줄어듦.
- 서버는 고정 `cv_fold` 컬럼을 썼지만 로컬은 `KFold(shuffle=True, seed=42)` 랜덤 분할 →
  fold 단위 수치 1:1 재현은 불가, mean 수준 비교만 유효.
- Phase2 `lambda_reg` 값이 브리핑(§3, 0.1)과 서버 state 노트(warmfix_full_rerun, 0.5) 간
  불일치 — 로컬 `config/phase2.json` 은 현재 브리핑값 0.1 유지, 확인 필요(재구성 우선순위 4).
- CPU 전용. GPU 필요 시 `train_phase2.py`/`train_phase3.py` 의 `DEVICE` 만 교체.
- Phase 3 **전체 5-fold × 40epoch 학습 미실행** — 스모크(2fold×2epoch, scratch)만 확인.
