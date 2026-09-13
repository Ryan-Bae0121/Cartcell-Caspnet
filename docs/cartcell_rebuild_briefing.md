# cartcell 프로젝트 재구성 브리핑

> 이 문서는 `eantec-server`가 막혀서 서버에 있던 custom code에 접근할 수 없는 상황에서,
> 과거 Cursor 대화 기록과 Claude 대화 기록에서 확인된 스펙을 바탕으로
> 로컬에서 프로젝트를 재구성하기 위해 정리한 문서입니다.
> **주의: 아래는 대화 중 언급/확인된 스펙이며, 실제 코드 원본이 아닙니다.**
> 서버 복구 시 반드시 실제 코드와 대조해서 검증할 것.

---

## 0. 베이스 & 디렉토리 구조

- 베이스 프레임워크: GitHub의 **CapsNet-MHC** 저장소를 클론해서 사용 (34MB, DNS 이슈로 curl 대안 사용해 다운로드했었음)
- 커스텀 코드 위치: `CapsNet-MHC-main/codes/custom_codes/`
- 설정 파일: `custom_train/config.json`, `custom_test/config.json`
- 실행 스크립트: `run_train.sh`, `run_test.sh`
- 데이터 위치: `custom_dataset/Anthem_dataset/`
  - `train_data.txt` — 52,303 샘플
  - `test_data.txt` — 13,076 샘플
- 원본 데이터: `mhc_ligand_table_DR.xlsx` (65,379 샘플, IEDB) → Excel → CapsNet 입력 형식(Anthem & IEDB 포맷)으로 변환 완료했던 이력 있음

**할 일**: GitHub에서 CapsNet-MHC 원본 저장소를 다시 클론하는 것부터 시작 (이 부분은 서버 없이 100% 복구 가능).

---

## 1. 데이터 파이프라인

### 1.1 데이터셋 규모
- 총 34,408 peptide–allele 쌍 (전처리 후 최종본, 원본은 65,379)
- 101개 MHC-II allele (HLA-DR, -DP, -DQ)
- Peptide 길이: 13–25 aa (범위 밖 약 3.7% 제외)
- Strong binder 비율: 약 5.13% (class imbalance 근거)

### 1.2 Inequality IC50 처리 (Aggressive Transformation)
```
IC50_numeric = x/2   if IC50 < x   (truncated less-than)
IC50_numeric = 2x    if IC50 > x   (truncated greater-than)
IC50_numeric = x     if IC50 = x   (exact value)
```
목적: truncate된 값도 버리지 않고 순서+강도 정보를 부여. 이 변환 적용 전후로 회귀 성능이 크게 개선됨 (변환 전 Pearson r ≈ 0.099 → 후 0.495, 이건 초기 버전 수치이고 최종 5-fold는 0.70대까지 감).

### 1.3 Affinity 정규화 & 라벨
```
affinity = 1 - log(IC50_numeric) / log(50000)
```
- 범위 [0, 1], 높을수록 strong binder
- Binary label: `affinity >= 0.8` → strong binder(1), 그 외 weak(0)

### 1.4 인코딩
- Peptide: BLOSUM62, 23차원, 최대 길이 25로 zero-padding
- Allele pseudo-sequence: 34 residue, 10차원 단순화 BLOSUM62 인코딩 사용 (23차원 아님 — 주의)
- **알려진 이슈**: pseudo-sequence가 현재 placeholder(예: "VVVV" 반복 문자)로 채워져 있고, 실제 IMGT/HLA 유래 서열로 교체 + 재학습이 필요한 상태였음 (미해결 항목)

---

## 2. 모델 아키텍처 (코드 레벨로 확인된 설명)

### Phase 1 (초기 버전, 참고용)
- 슬라이딩 윈도우로 9-mer 조각 생성 → 각각 MLP → 최고 점수를 펩타이드 예측값으로 사용
- allele은 원핫 인코딩(서열 정보 없음)
- binding core 위치에 대한 명시적 표현 없음 → Phase2/3의 출발점 역할만

### Phase 2 — CNN + Core Register Module
- **펩타이드 브랜치**: 전체 서열 BLOSUM 인코딩 → Conv 스택 → kernel=9 valid conv로 T=L-8개 "9-mer 후보 위치" 생성 → softmax로 위치 확률분포(`pos_prob`) 산출 → 가중합으로 `core_vec` 추출
- **Allele 브랜치**: JSON에서 allele별 pseudo-sequence 조회 → BLOSUM(10차원) → Conv → pooling → `mhc_vec` 추출
- **결합**: `core_vec` + `mhc_vec` concat → MLP → 분류 logit (+ 회귀 yhat)
- 학습 시 `pos_prob` 엔트로피 패널티 추가 (core 위치가 너무 분산되지 않도록)

### Phase 3 — Multi-branch CapsNet + Dynamic Routing
3개 브랜치에서 총 56개 primary capsule 생성:
- Full peptide branch: 전체 서열 Conv → Global Avg Pool → **32개** primary capsule
- Core branch: Phase 2와 동일한 CoreRegister 구조 → **16개** primary capsule
- MHC branch: 동일 PseudoSeqEncoder → **8개** primary capsule

→ 56개 primary capsule concat → **Dynamic Routing 3회 반복** → class capsule 2개 + regression capsule 1개

**출력**:
- 분류: 두 class capsule의 norm 차이 → logit
- 회귀: regression capsule → Linear → yhat
- 해석용: `pos_prob`(core 위치 분포), `coupling_cls`/`coupling_reg`(각 primary capsule의 기여도)

**수식**:
```
Squash function:
v_j = (||s_j||² / (1 + ||s_j||²)) · (s_j / ||s_j||)

Dynamic routing update:
b_ij ← b_ij + û_(j|i) · v_j
c_ij = exp(b_ij) / Σ_k exp(b_ik)
```

**알려진 버그(수정됨)**: Phase 2 가중치로 warm-start 할 때 모듈 이름이 `mhc_encoder` vs `pseudo_encoder`로 불일치해서 MHC 브랜치 가중치가 실제로는 로드되지 않고 core만 이식되던 문제가 있었음. 이후 정정 후 전체 재학습해서 fold 간 Pearson r 안정성이 개선됨. **재구현 시 이 이름 불일치를 처음부터 통일할 것.**

### Ensemble
- 새 아키텍처 아님. Phase2, Phase3를 각각 독립적으로 forward 후 출력만 선형 혼합 (파라미터 공유/추가 학습 없음)
```
분류: α·σ(logit_phase2) + (1-α)·σ(logit_phase3), 기본 α=0.5
회귀: α·yhat_phase2 + (1-α)·yhat_phase3
```
- Simple averaging을 택한 이유: 샘플 수(~34,400)가 제한적이라 meta-learner는 overfitting 위험 → averaging이 더 안전

---

## 3. 학습 설정

- Multi-task loss: `L = L_cls + λ·L_reg`
  - λ_reg = 0.1 (Phase2, 5-fold) / λ_reg = 0.5 (Phase3 5-fold, 그리고 LOMO에서는 두 모델 다 0.5)
- Classification: BCEWithLogitsLoss, `pos_weight ≈ 18.5` (strong binder 비율 5%에서 유도된 값)
- Regression: MSE
- Optimizer: AdamW
- Epoch: Phase2 = 30, Phase3 = 40

## 4. 평가 프로토콜 3종

1. **5-fold Cross-Validation**: 전체 데이터 분할, out-of-fold(OOF) 예측 concat 후 전체 성능 계산
2. **Per-allele analysis**: OOF 예측을 allele별로 grouping, 최소 30 샘플 이상 allele만 분석
3. **LOMO (Leave-One-Allele-Out)**: 특정 allele 완전히 제외하고 처음부터(from scratch) 재학습, 최소 50 test 샘플 allele 대상. 총 60개 allele × 2모델 반복, 약 6시간 소요.
   - **알려진 데이터 누수 이슈**: LOMO early stopping 시 `val_df = test_df`로 설정되어 있던 리크 존재. 재실험하지 않고 Limitation에 투명하게 명시하기로 결정한 상태.

## 5. 참고 성능 수치 (5-fold CV 기준, 최종본)

| Model    | AUROC | AUPRC | Pearson r | MSE   |
|----------|-------|-------|-----------|-------|
| Phase2   | ~0.88 | ~0.38 | 0.49      | 0.063 |
| Phase3   | ~0.86 | ~0.30 | **0.70**  | **0.043** |
| Ensemble | ~0.88 | ~0.41 | 0.68      | 0.046 |

---

## 6. 새 Cursor 세션에서 쓰는 법

새 로컬 워크스페이스를 열고 첫 프롬프트에 이 문서 전체를 붙여넣은 뒤 이렇게 시작하면 돼:

> "서버 접속이 막혀서 기존 custom_codes를 잃어버렸어. 위 스펙을 기준으로 CapsNet-MHC 베이스 저장소 위에 Phase2, Phase3, Ensemble 모델을 codes/custom_codes/ 아래에 재구현해줘. 먼저 디렉토리 구조랑 Phase2부터 짜자."

한 번에 다 시키지 말고, Phase2 → Phase3 → 평가 스크립트 순서로 단계별로 진행하는 걸 추천.
