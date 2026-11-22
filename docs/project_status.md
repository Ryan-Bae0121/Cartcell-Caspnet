# cartcell MHC-II predictor — 전체 히스토리와 현재 상황

최종 갱신: **2026-11-19**

> 이 문서는 프로젝트의 **단일 기준 문서**입니다. 서버 유실 이후의 로컬 재구성,
> 세운 가설, 그 가설을 검증한 실험, 그리고 **가설이 기각된 경위**까지를
> 숫자와 근거 파일 경로로 남깁니다.
>
> 같은 폴더의 다른 두 문서는 성격이 다릅니다.
>
> - `cartcell_rebuild_briefing.md` — 서버 유실 직후, 대화 기록에서 복원한 **스펙 추정치**. 검증 전 자료.
> - `cartcell_server_state.md` — 유실된 서버 환경의 상태 기록.
>
> 아래 숫자는 모두 이 저장소의 `reports_*` 산출물에서 직접 읽은 값입니다.
> **틀렸다고 밝혀진 서술은 지우지 않고 정정 표시와 함께 남깁니다** — 어느 숫자가
> 어떤 이유로 바뀌었는지가 기록의 핵심입니다.

---

## 0. 한 줄 요약

**원래 가설은 기각됐습니다.** 그러나 그 자리에 더 구체적이고 검증 가능한 결론이 섰습니다:
CapsNet은 binding core를 식별해서 잘하는 게 아니고, 회귀에서는 이기지만
**그 이득은 dynamic routing에서 오지 않습니다**(+0.002, p=0.373). 그리고 왜 오지 않는지 메커니즘까지 측정했습니다.

> **[정정] 2026-11-11 — 회귀 격차의 *크기*는 현재 신뢰할 수 없습니다.**
> 수렴 프로브에서 50 epoch 예산이 **Phase2를 훨씬 더 심하게 굶긴 것**이 드러났습니다.
> 양쪽 150 epoch에서는 격차가 **+0.128 → +0.037로 줄고**(§5.6), **세 모델 모두 400 epoch**을
> 받으면 **부호가 뒤집힙니다**(−0.032, §5.8 — 2026-11-19 확정). Phase2의 공정 예산은
> 50이 아니라 **300 이상** — 구 예산은 6배 부족했습니다.
> **§5의 표와 §7의 서술은 5-fold 재학습 뒤에 다시 써야 합니다.**
> 기각된 가설(core 식별)과 routing null 결과(+0.002, p=0.373)는 **영향 없습니다** —
> 둘 다 예산과 무관한 비교이고, routing null은 `|W|`가 6–10배 커진 뒤에도 유지됩니다.

---

## 1. 배경

### 1.1 무엇을 예측하는가
MHC-II 분자와 peptide의 결합 친화도. 13–25 aa peptide가 주어지면 그 안의
**9-mer binding core**가 MHC-II groove에 들어가고, 양 옆은 flank로 남습니다.
어느 9-mer가 core인지는 **라벨이 없습니다** — 이게 이 문제의 본질적 어려움입니다.

라벨 변환:
```
affinity = 1 - log(IC50) / log(50000)
```

### 1.2 데이터
- `custom_dataset/Anthem_dataset/` — train 25,628 / test 6,407 쌍
- 68 MHC-II allele (register-shift 실험 기준), HLA-DR/DP/DQ
- strong binder(≤500 nM) 비율 약 5% → `pos_weight ≈ 18.5`로 weighted BCE
- 500 nM 임계값은 affinity 척도로 **0.4256**, 50 nM은 **0.6384**

### 1.3 모델 두 개
| | 구조 |
|---|---|
| **Phase2 (CNN)** | CNN + CoreRegister. T=17개 후보 위치에 softmax(`pos_prob`) → 가중합 `core_vec`. `core_conditioning`으로 allele 정보를 core 선택에 주입 |
| **Phase3 (CapsNet)** | multi-branch primary capsule 56개(full 32 + core 16 + MHC 8) + dynamic routing → 3 output capsule. Phase2에서 core/MHC 가중치 35개 텐서를 warm-start |

**pseudo-sequence는 실서열입니다.** `custom_dataset/Anthem_dataset/mhc_ii_pseudo.csv`
68행 전부 `source=real` (34-mer). 과거의 placeholder 더미 단계는 지나갔습니다.

> `register_shift_summary.json`의 `pseudo_sequence_warning`이 한동안
> `"0/68 alleles use PLACEHOLDER ... carries no real signal"`로 출력되어
> **placeholder로 돌린 run처럼 읽혔습니다.** 경고 문장이 카운트와 무관하게
> 무조건 붙던 버그이고, 지금은 카운트가 0이면 문장도 달라지게 고쳤습니다.
> **숫자 0/68이 맞는 값이고, 이 run들은 실서열 기반입니다.**

---

## 2. 검증하려던 가설

> **CapsNet + dynamic routing은 binding core(register)를 일관되게 식별하며,
> 이는 CNN보다 정확하고, 그 능력은 dynamic routing에서 나온다.**

지도교수 피드백으로 이 문장이 **세 개의 독립적 주장**으로 쪼개진다는 걸 확인했고,
세 개를 각각 측정했습니다.

1. core를 일관되게 식별한다
2. CNN보다 정확하게 추적한다
3. (숨은 전제) 그게 dynamic routing 덕분이다

---

## 3. 실험 설계 — register-shift

핵심 함정: **불변성만 재면 아무 의미가 없습니다.** 입력을 무시하는 상수 모델이
불변성 100%를 받습니다. 그래서 **민감도 대조군을 함께** 넣었습니다.

`codes/custom_codes/register_shift.py`, 조건당 300 케이스:

| 조건 | 무엇을 바꾸나 | 기대 |
|---|---|---|
| `shift_flank` | core 위치만 옮김 (flank 길이 재분배) | 예측이 **안 흔들려야** |
| `shift_pad` | padding으로 위치만 옮김 | 예측이 **안 흔들려야** |
| `flank_resample` | 위치 고정, flank 서열만 교체 | shift와 비교할 기준선 |
| `core_mutate` | core 내부 잔기 변이 | 예측이 **흔들려야** |
| `scramble_all` | 전체 서열 셔플 (조성 보존) | 예측이 **흔들려야** |

누출 감사도 함께 돌렸습니다: `exact_pair_in_train = 0` / 16,468 variant.
`core9_in_train_same_allele`이 1,783로 높지만 이건 **설계상 의도된 것**입니다 —
anchor가 실제 peptide를 재사용하므로. 이 실험은 **예측 안정성**을 재는 것이고
미지 core에 대한 일반화를 재는 게 아닙니다.

---

## 4. 결과 (1) — 가설 세 조항 전부 기각

### 4.1 core를 일관되게 식별하는가 → 아니다

`shift_flank`에서 **같은 9-mer를 집는 비율(`same_core_rate`)**:

| 모델 | same_core_rate | fixed_index_rate |
|---|---|---|
| Phase2 (CNN) | **0.412** | 0.050 |
| Phase3 (routing 3) | **0.403** | 0.053 |
| Phase3 no-routing | **0.420** | 0.045 |

> **정정 (2026-11-21) — 0.416은 null이 아닙니다.** 이전 판은 "랜덤 9-mer 일치
> 기대값(null) 0.416"이라고 적었습니다. **틀린 라벨입니다.** 0.416의 출처는
> `reports_register_shift_final/register_shift_summary.json` →
> `condition_summary[1]` (`condition = "flank_resample"`) 의
> `mean_same_core_rate = 0.41556`, 즉 **Phase2의 `flank_resample` 대조군 값**입니다.
> `flank_resample`은 [`register_shift.py`](../codes/custom_codes/register_shift.py)
> 주석대로 *"flanks resampled R times at a FIXED j"* — **위치를 고정하고 flank만
> 교체**하는 조건입니다. 무작위 기대값이 아니라 **측정값**이고, `shift_flank`와
> 같은 성질의 양입니다.
>
> 실제 chance level은 [`plot_core_calibration.py`](../codes/custom_codes/plot_core_calibration.py)
> 의 `baselines()`가 열거법으로 계산합니다 — 기준 A(모델 출력공간 T=17 슬롯 균등),
> 기준 B(anchor 자신의 9-mer window 균등). 이 값이 **PNG에만 그려지고 JSON에
> 저장되지 않았던 것**이 두 달간 라벨 오류가 남은 직접적 원인입니다. 이제
> `core_calibration_baselines.json`으로 저장합니다.

**chance 대비 배수 (2026-11-22, §5.9 GPU 5-fold 체크포인트 기준).**
출처: `reports_register_shift_{phase2,phase3,norouting}_gpu600/core_calibration_baselines.json`
→ `conditions.shift_flank`. 2,084 variant 행 / 300 case.

| 모델 | 측정 `shift_flank` | 기준 A (출력공간 17슬롯) | A 대비 | 기준 B (위치 기지) | B 대비 |
|---|---|---|---|---|---|
| Phase2 | 0.4175 | 0.0549 | **7.60×** | 0.1332 | **3.13×** |
| Phase3 | 0.4040 | 0.0518 | **7.81×** | 0.1251 | **3.23×** |
| no-routing | 0.4064 | 0.0533 | **7.62×** | 0.1290 | **3.15×** |

**"무작위와 구분되지 않는다"는 완전히 틀린 진술이었습니다.** core 선택은 가장
보수적인 기준(B: peptide 위치를 이미 안다고 가정)으로도 chance의 **3.1–3.2배**,
모델이 실제로 내놓는 출력공간 기준(A)으로는 **7.6–7.8배**입니다. 세 모델이 거의
동일한 배수를 보입니다.

따라서 §4.1의 주장은 "무작위"가 아니라 **"대조군 대비 차이가 없다"**로
바뀝니다:

| 모델 | `flank_resample` (위치 고정) | `shift_flank` (위치 이동) | 차이 |
|---|---|---|---|
| Phase2 | 0.4150 | 0.4198 | +0.005 |
| Phase3 | 0.4250 | 0.4069 | −0.018 |
| no-routing | 0.4167 | 0.4095 | −0.007 |

**위치를 옮기는 것이 flank만 바꾸는 것과 다를 바 없습니다.** 즉 core 선택의
불안정성은 **위치 이동과 무관**합니다 — 문맥이 조금이라도 바뀌면 다른 9-mer를
집고, 그 교체율은 peptide가 실제로 움직였는지 여부에 거의 반응하지 않습니다.

> **차이의 부호는 재현되지 않습니다.** CPU 런에서는 Phase3 +0.005 /
> no-routing +0.010 (즉 `shift_flank`가 더 높음)이었는데, GPU 런에서는 각각
> −0.018 / −0.007로 **부호가 뒤집혔습니다**. Phase2도 −0.004 → +0.005로
> 뒤집혔습니다. 세 모델 전부 부호가 반대로 나왔으므로 **±0.02 수준의 이
> 차이는 noise이고, 방향성을 해석해서는 안 됩니다.** 이전 판에 적었던
> "Phase3·no-routing은 shift_flank가 오히려 높다 — 위치 이동에 페널티가
> 없다는 뜻"이라는 문장은 **철회합니다.** 남는 주장은 부호 없는 쪽,
> 즉 "두 조건의 차이가 0에 가깝다"입니다.

`fixed_index_rate`가 0.05로 낮으니 "그냥 고정 위치를 집는다"는 경쟁 설명도 아닙니다 —
**위치를 옮기면 다른 9-mer를 집습니다.**

`shift_pad`에서는 0.51–0.54로 조금 높지만(drift 1.80–2.06), 이건 padding이
문맥을 덜 바꾸기 때문이고 여전히 "일관되게 식별한다"고 부를 수준이 아닙니다.

> **중요:** core 추적이 불안정한 것과 **예측 자체가 안정한 것**은 별개입니다.
> 예측은 안정합니다(`shift_flank` / `flank_resample` 변동 비 0.88–0.95).
> 모델은 register를 명시적으로 식별하지 않고도 안정적으로 예측합니다.

### 4.2 CNN보다 정확한가 → 아니다

0.412(Phase2) vs 0.403(Phase3). 차이 없습니다.

> **기록해 둘 사실:** 이 비교표가 한때 **뒤집혀 있었습니다.** 0.412가 Phase3 숫자라고
> 적힌 적이 있는데, `register_shift_summary.json`의 `args.ckpt_dir`을 확인한 결과
> `reports_phase2_final/checkpoints` — **0.412는 Phase2**가 맞습니다.
> 또 "이 숫자들은 `core_conditioning` 적용 전"이라는 지적이 있었으나,
> 체크포인트의 `cond_gate` 값(0.351 / 0.251 / 0.228 / 0.352 / −0.216)이
> 0이 아니므로 **적용 후**의 숫자입니다.

### 4.3 dynamic routing 덕분인가 → 아니다

`config/phase3_norouting.json` — `routing_iters: 1`만 다르고 나머지는 전부 동일(diff로 확인).
`n_iters=1`이면 `b`가 0에 머물러 `c = softmax(0) = 1/3` **정확히 균일**,
즉 `u_hat`의 단순 평균이 됩니다. **W와 squash와 capsule 구조는 그대로 학습되므로,
이건 "routing 반복"만 제거하는 올바른 ablation**입니다.

→ 모든 지표에서 full routing과 구분되지 않음. 자세한 수치는 §5.

### 4.4 추가로 드러난 것 — Phase3는 민감도 대조군을 통과하지 못한다

`scramble_all / shift_flank` 변동 비 (**1보다 커야** 서열에 민감한 것):

| 모델 | 비 | 해석 |
|---|---|---|
| Phase2 (CNN) | **1.48** | 통과 (p = 5e-31) |
| Phase3 (routing 3) | **0.81** | **실패**, 유의하지 않음 |
| Phase3 no-routing | **0.82** | **실패** |

`scramble_all`은 **아미노산 조성을 보존**합니다. Phase2는 셔플하면 흔들리는데
Phase3는 오히려 덜 흔들립니다 → **Phase3는 조성/길이 같은 더 거친 특징에 의존**한다는 증거.
Phase3의 기준선 wobble도 Phase2의 2배입니다.

`core_mutate / shift_flank` 비는 Phase2 2.32, Phase3 1.67, no-routing 1.60 —
core 변이에는 세 모델 다 반응하지만 Phase2가 가장 강합니다.

---

## 5. 결과 (2) — 공정 비교 (round 2, 최종)

### 5.1 왜 다시 돌렸나: 두 개의 교란 요인

**(a) 체크포인트 선택 기준이 불안정했습니다.**
weighted BCE val_loss는 ranking 지표가 좋아지는 동안에도 올라갑니다
(이미 알려진 `phase2-early-stopping-bug`: val_loss로 고르면 epoch 0을 고름).
그래서 AUPRC로 바꿨었는데, AUPRC도 흔들렸습니다. AUROC 단독은 **fold 2, 3에서 epoch 0**을 골라
더 나빴습니다. capsule 모델에서는 AUROC가 epoch 0–1에 피크를 찍고 내려가는 반면
Pearson r은 epoch 38에도 여전히 오르고 있어서, **어느 한 지표로도 안 됩니다.**

→ `train_phase2.py`에 공유 헬퍼 `selection_score()`를 넣고
**`mean(AUROC, Pearson r)` 복합 기준**을 도입. `train_phase3.py`와 `lomo.py`가 같이 씁니다.

**(b) λ_reg가 모델 간에 달랐습니다** — 회귀 손실 가중치가 다른데 회귀 성능을 비교하고 있었습니다.

→ 셋 다 **λ_reg = 0.5, epochs = 50, 복합 기준**으로 통일해 재학습.

```
Phase2 (CNN)        lam=0.5 ep=50 sel=['auroc','pearson_r'] routing=-
Phase3 (routing 3)  lam=0.5 ep=50 sel=['auroc','pearson_r'] routing=3
no-routing          lam=0.5 ep=50 sel=['auroc','pearson_r'] routing=1
```

### 5.2 성능

| 모델 | split | AUROC | AUPRC | Pearson r | MSE |
|---|---|---|---|---|---|
| Phase2 (CNN) | oof | 0.861 | 0.379 | 0.510 | 0.0531 |
| Phase2 (CNN) | **test** | **0.895** | **0.426** | 0.552 | 0.0506 |
| Phase3 (routing 3) | oof | 0.824 | 0.282 | 0.634 | 0.0429 |
| Phase3 (routing 3) | **test** | 0.882 | 0.373 | **0.680** | **0.0382** |
| no-routing (1) | oof | 0.826 | 0.300 | 0.638 | 0.0428 |
| no-routing (1) | **test** | 0.883 | 0.347 | **0.682** | **0.0380** |

### 5.3 부트스트랩 검정 (2,000 resample, 동일 test 행)

행 정렬은 `affinity` 컬럼 `np.allclose`로 검증했습니다.

| 비교 | Pearson r 차이 | AUPRC 차이 |
|---|---|---|
| **Phase3 − Phase2** | **+0.1284** [+0.1165, +0.1400] p<0.001 | **−0.0516** [−0.0820, −0.0219] p=0.001 |
| **no-routing − Phase3** | +0.0021 [−0.0024, +0.0069] **p=0.373** | −0.0247 [−0.0512, +0.0008] p=0.057 |

절대값:

| | Pearson r | 95% CI | AUPRC | 95% CI |
|---|---|---|---|---|
| Phase2 | 0.552 | [0.533, 0.569] | 0.427 | [0.369, 0.481] |
| Phase3 | 0.680 | [0.665, 0.694] | 0.375 | [0.320, 0.430] |
| no-routing | 0.682 | [0.667, 0.697] | 0.351 | [0.301, 0.404] |

**읽는 법:**
- **회귀는 CapsNet, 분류는 CNN.** 둘 다 유의하고, 트레이드오프가 실재합니다.
- **routing 기여는 0.** r 차이의 CI가 0을 포함하고 p=0.373. AUPRC만 −0.025로 경계선(p=0.057)인데
  이 지표는 노이즈가 크고 방향도 routing에 유리해서 주장 근거로 쓰기 어렵습니다.
  **독립적인 두 번의 확인**입니다 — round 1(λ_reg 0.5 / epoch 40 / AUPRC 선택)과 round 2(현 조건).

### 5.4 기록해 둘 틀린 예측

재학습 전에 이렇게 예측했습니다:
> "Phase3의 회귀 우위가 상당 부분 λ_reg였을 가능성이 크고, 크게 축소될 것이다."

**틀렸습니다.**

| Pearson r | λ_reg 불일치 (0.1 / 0.5) | λ_reg 통일 (0.5) |
|---|---|---|
| Phase2 | 0.483 | 0.552 (+0.069) |
| Phase3 | 0.608 | 0.680 (+0.072) |
| **격차** | **0.125** | **0.128** |

λ_reg를 올리자 **두 모델이 거의 같은 폭으로 같이 올라갔습니다.** Phase2만 오를 거라고
가정한 게 잘못이었습니다. 결과적으로 **Phase3의 회귀 우위는 아키텍처 효과로 확정**됐고,
이제는 통제된 조건에서 얻은, CI가 0을 한참 벗어나는 숫자입니다.

### 5.5 남은 흠 — epoch 예산

`best_epoch`이 여전히 50 예산에 붙어 있습니다:

| 모델 | best_epoch / fold |
|---|---|
| Phase2 | 42, 37, 47, 35, 44 |
| Phase3 | 48, 41, 38, 34, 47 |
| no-routing | 49, 49, 39, 34, 47 |

50 epoch도 부족합니다. 처음에는 **"세 모델이 똑같이 부족하므로 비교의 공정성은 유지된다"**고
적었는데, **2026-11-11 수렴 프로브로 반증됐습니다 → §5.6.**

### 5.6 수렴 프로브 (2026-11-11) — 예산이 §5.2의 격차를 만들었습니다

fold 0 하나를 세 모델 모두 150 epoch까지 돌리며 매 epoch을 기록했습니다
(`run_convergence_probe.sh`, 트레이스는 `reports_probe_*/trace/`).

**세 모델은 똑같이 부족하지 않았습니다. Phase2가 압도적으로 더 굶었습니다.**

| fold 0 val | r @49 (구 예산) | r @best | Δ | best_epoch | 평평해지는 epoch |
|---|---|---|---|---|---|
| Phase2 | 0.523 | **0.658** | **+0.135** | 146 | **아직 상승 중** |
| Phase3 | 0.655 | 0.694 | +0.039 | 125 | 125 |
| no-routing | 0.654 | 0.683 | +0.029 | 124 | 113 |

**그래서 Phase3 − Phase2 회귀 격차가 72% 줄어듭니다:**

| | Phase3 − Phase2 (Pearson r) |
|---|---|
| @49 — 구 예산 | **+0.132** |
| @best — 공정 예산 | **+0.037** |

@49의 +0.132가 §5.3이 test에서 보고한 **+0.128과 거의 일치**합니다. fold 0 val 궤적이
보고된 비교를 제대로 대리한다는 뜻이고, 그래서 +0.037로의 축소를 신뢰할 수 있습니다.

**방향이 바뀐 것은 없지만 크기는 바뀝니다.** 분류 격차는 오히려 **벌어집니다**
(AUROC @best — Phase2 0.860 vs Phase3 0.809). 복합 지표로는 **Phase2가 역전합니다**
(0.759 vs 0.751). routing 결론은 **그대로**입니다 — no-routing − Phase3 = −0.011.

### 5.7 Phase2 400 epoch (2026-11-11) — 격차의 **부호가 뒤집힙니다**

Phase2가 150에서도 상승 중이었으므로 400 epoch까지 연장했습니다. **epoch ~301에서
평평해집니다**(복합 지표 블록 최대: … 0.7841 → 0.7889 → 0.7908 → 0.7893 → 0.7921 → 0.7920).
즉 Phase2의 공정 예산은 **50이 아니라 300 이상**입니다 — 구 예산은 **6배 부족**했습니다.

| fold 0 val | epoch | 복합 | Pearson r | AUROC |
|---|---|---|---|---|
| **Phase2 (400 예산)** | 370 | **0.7921** | **0.7289** | **0.8553** |
| Phase3 (150 예산) | 125 | 0.7514 | 0.6942 | 0.8087 |
| no-routing (150 예산) | 124 | 0.7428 | 0.6834 | 0.8022 |

Phase2의 r 궤적: @49 0.523 → @149 0.657 → @249 0.705 → @**370 0.729**.

| Phase3 − Phase2 (r) | |
|---|---|
| @49 (구 예산 50) | **+0.132** |
| 양쪽 150 예산 | +0.037 |
| Phase2 300+ vs Phase3 150 | **−0.035 (부호 역전)** |

> **이 −0.035는 Phase3·no-routing이 150 epoch만 받은 상태의 비교였습니다** — 그 자체가
> §5.6에서 진단한 오류입니다. 두 capsule 모델도 같은 400 예산으로 돌렸고,
> 결과는 **§5.8**입니다. 부호는 유지됩니다(−0.032).

> **지금 상태로 확정된 것:** §5.2/§5.3의 숫자는 **논문에 쓸 수 없습니다.**
> 구 예산 50은 Phase2에 6배 부족했고, 그 결과 회귀 격차가 최소 2.6배 과대평가됐으며
> 부호마저 뒤집힐 수 있습니다. 재학습 전까지 §5·§7의 모든 성능 주장은 **보류**입니다.

### 5.8 세 모델 동일 400 예산 (2026-11-19) — 회귀 우위가 **사라집니다**

§5.7의 −0.035는 Phase2만 400을 받은 비교였습니다. 두 capsule 모델도 같은 예산으로
돌려 **같은 조건에서 비교**했습니다(fold 0, 복합 기준, `reports_probe_*_long/trace/`).

| fold 0 val | epoch 수 | 복합 최대 | 그 때 r | 그 때 AUROC | r 최대 |
|---|---|---|---|---|---|
| **Phase2** | 400 | **0.7921** @370 | **0.7289** | **0.8553** | 0.7308 @372 |
| Phase3 | 374 | 0.7522 @160 | 0.6972 | 0.8072 | 0.7243 @299 |
| no-routing | 400 | 0.7590 @392 | 0.7047 | 0.8132 | 0.7105 @376 |

| Phase3 − Phase2 | 값 |
|---|---|
| @49 (구 예산 50) | **+0.132** |
| 양쪽 150 예산 | +0.037 |
| **양쪽 400 예산, 복합 기준 epoch의 r** | **−0.032** |
| 양쪽 400 예산, 각자의 r 최대끼리 | **−0.007** |
| 양쪽 400 예산, 복합 지표 | **−0.040** |
| 양쪽 400 예산, AUROC | **−0.048** |

**읽는 법:** 예산을 맞추면 **회귀 우위가 남지 않습니다.** 복합 기준으로 고른 체크포인트끼리는
Phase2가 −0.032만큼 앞서고, 각 모델의 r 최대끼리 비교하면 −0.007로 **사실상 동률**입니다.
어느 쪽이든 §5.3이 보고한 **+0.128은 재현되지 않습니다.** 분류 격차는 §5.6에서 본 대로
**더 벌어집니다**(AUROC −0.048).

**routing 결론은 그대로입니다.** no-routing − Phase3 = **+0.0075**(r), +0.0068(복합) —
400 epoch에서도 routing 쪽이 유리하지 않습니다. 그리고 `|W|`는 예산만으로
**Phase3 8.7×, no-routing 15.2×**(init 0.00798 대비)까지 자랐는데도 기여는 0입니다.
§6.2의 "6–10배" 서술은 이제 **9–15배**로 읽어야 하고, 결론은 더 강해집니다.

> **주의 — 아직 1 fold입니다.** 부호는 세 번의 서로 다른 예산에서 일관되게 움직였지만
> (+0.132 → +0.037 → −0.032), 이건 fold 0 validation 궤적 하나입니다.
> **5-fold 재학습과 test-set 부트스트랩 CI 없이는 논문에 못 씁니다.**
> Phase3는 374/400에서 죽었으나 복합 최대가 160, r 최대가 299라 예산 부족은 아닙니다.

---

### 5.9 사전 등록 프로토콜 — GPU 5-fold 재학습 (2026-11-21)

§5.8이 1 fold val 궤적이라 결론을 쓸 수 없었습니다. 이 절은 **실행 전에** 확정한
규칙입니다. 결과를 보고 규칙을 고치지 않기 위해 미리 적어 둡니다.

#### 환경 이전 — CPU → RTX 4090

| 모델 | CPU s/epoch | GPU s/epoch | 배속 |
|---|---|---|---|
| Phase2 | ~49 | **4.0** | 12.3× |
| Phase3 | ~64 | **6.9** | 9.3× |
| no-routing | ~50 | **5.75** | 8.7× |

CPU 기준값은 `reports_probe_*_long/trace/*.jsonl`의 `secs` 필드 중 오염되지 않은
구간(ep 0–199)에서 가져왔습니다. 15 run × 600 epoch = **약 13.9시간**
(CPU로는 136시간).

**GPU 이전에서 고친 버그 3개** — 전부 CPU 전용 경로에서는 드러나지 않던 것입니다:

| 파일 | 문제 | 수정 |
|---|---|---|
| `train_phase2.py:140` | `vt = torch.tensor(0.0)`이 CPU에 생성되어 cuda loss 누적 시 `RuntimeError` | `torch.zeros((), device=DEVICE)` |
| `train_phase3.py:160` | 동일 | 동일 |
| `diag_routing_iters.py:72` | routing `b`가 CPU, `c.numpy()`가 cuda 텐서에서 실패 | `device=u_hat.device`, `.detach().cpu().numpy()` |

**GPU utilization은 median 34%입니다** (30회 샘플, min 30 / max 44). 병목은 GPU가
아니라 `batch_size = 32`입니다. 벤치마크: 데이터셋 전체가 89.6 MB로 GPU에 1.3초면
올라가고, GPU 상주 인덱싱 + bs=256이면 train epoch이 9.2배, bs=1024면 37배
빨라집니다. **그래도 bs=32를 유지합니다** — §5.6–5.8의 모든 프로브가 bs=32였고,
배치/lr을 바꾸면 "예산이 만든 착시"라는 현재 결론에 **두 번째 변인**이 들어갑니다.
13.9시간은 하룻밤에 들어가므로 재현성을 속도와 바꿀 이유가 없습니다.

#### 확정된 설정

| 항목 | 값 | 근거 |
|---|---|---|
| **epochs** | **600** | 400에서 Phase2 복합 peak @370, no-routing @392 — **셋 중 둘이 천장에 붙어** 예산 반론을 닫지 못합니다 |
| **미수렴 판정** | **선택 epoch ≥ 540이면 그 모델은 미수렴으로 보고 예산을 늘린다** | 사전 규칙. 사후에 "충분했다"고 주장하지 않기 위함 |
| **선택 기준 (주)** | **AUPRC** | 5% 양성 클래스의 올바른 요약이고, **회귀 지표와 독립**이라 Pearson r 비교가 순환이 아닙니다 |
| 선택 기준 (민감도) | 복합 `(auroc+pearson_r)/2` | §5.8이 쓴 규칙 |
| 선택 기준 (참조) | 각 모델의 `pearson_r` 최대 | 선택 규칙과 무관한 상한 |
| batch_size | 32 | 위 참조 |
| lr / weight_decay / grad_clip | 2e-4 / 1e-6 / 1.0 | 변경 없음 |
| λ_reg | **0.5 (세 모델 동일)** | 커밋 `8526eee`에서 이미 통일됨 (§5.1b) |
| seed / n_folds / split | 42 / 5 / `KFold(shuffle=True, random_state=42)` | 세 모델 전수 확인 동일 |
| pseudo-sequence | `mhc_ii_pseudo.csv` (실서열) | 세 모델 동일 |
| `core_conditioning` | true | 세 모델 동일 |
| `entropy_beta` | 0.01 | 세 모델 동일 |
| `auto_pos_weight` | true (18.71) | 세 모델 동일 |
| `routing_iters` | 3 / 1 | **의도된 ablation** |
| **`dropout`** | **0.5 / 0.3 / 0.3 — 통일하지 않음** | 아래 한계 참조 |

설정 파일: `config/phase2_gpu600.json`, `config/phase3_gpu600.json`,
`config/norouting_gpu600.json`. 출력: `reports_*_gpu600/`.

> **한계 — `dropout`을 통일하지 않은 이유.** 값만 보면 불일치(0.5 vs 0.3)지만
> **적용 위치가 다릅니다**: Phase2는 MLP head에 `nn.Dropout` 2개
> (`models/phase2.py:48,51`), Phase3는 capsule projection 입력에 1개
> (`models/phase3.py:88`). 같은 이름의 다른 하이퍼파라미터이므로 숫자를 맞추는
> 것은 "공정하게 만드는" 조작이 아닙니다. 아키텍처별 재튜닝이 원칙적으로 맞지만
> 현재 질문(routing이 기여하는가)에 필요하지 않아 **현재값을 유지하고 한계로
> 명시합니다.**

#### 선택 규칙이 결론의 부호를 바꿉니다 — 이것 자체가 결과입니다

400-epoch 프로브(fold 0 val)에서 **Phase3 − Phase2 (Pearson r)**:

| 선택 규칙 | 고른 epoch (P2/P3) | Phase3 − Phase2 | 부호 |
|---|---|---|---|
| AUPRC (주) | 156 / 227 | **+0.037** | 회귀 우위 **있음** |
| 복합 (민감도) | 370 / 160 | **−0.032** | 회귀 우위 **없음** |
| 각자 r 최대 (참조) | 372 / 299 | **−0.007** | 사실상 동률 |

**AUPRC 선택이 Phase3에 유리한 이유는 routing이 아닙니다.** Phase2는 AUPRC가
@156에서 최대인데 그 지점의 r이 0.6703이고, 자기 r 최대는 @372의 0.7308입니다 —
**0.06 떨어진 지점에서 평가받습니다.** Phase3는 AUPRC @227에서 r=0.7068,
r 최대 @299에서 0.7243으로 간격이 0.018뿐입니다. 즉 **Phase2의 두 헤드가 서로
멀리서 peak**하고, AUPRC 기준은 그 간격에 벌점을 줍니다. capsule 구조의 이득이
아니라 **두 헤드의 수렴 시점 불일치**가 만드는 차이입니다.

> **결론 금지 조항.** 위 세 숫자는 **fold 0 단일 측정**입니다. ±0.03은 5-fold
> 부트스트랩 CI 안에 들어갈 가능성이 높습니다. **5-fold + CI가 나오기 전까지
> 어느 부호도 결론으로 쓰지 않습니다.** 보고는 세 규칙 전부를, 부호가 규칙에
> 의존한다는 사실과 함께 적습니다.

#### 측정 항목과 집계

세 규칙 각각의 체크포인트를 fold별로 저장합니다
(`checkpoints/{prefix}_fold{k}.pt`, `.pt.composite`, `.pt.rmax`;
`fold_report_{k}.json`의 `alt_selection`에 규칙별 선택 epoch 기록).
core tracking과 성능은 **동일 체크포인트**에서 뽑습니다.

- **성능:** AUROC, AUPRC, Pearson r, MSE — **test 기준** (val peak 아님)
- **core tracking:** `same_core_rate`, 평균 밀림(`mean_core_drift`)
- **부트스트랩 CI 1000회**, fold별 값도 따로 보고

> **CPU 결과와 섞지 않습니다.** §5.2/§5.3/§5.8은 CPU 런이고 예산·선택 기준이
> 다릅니다. §5.9의 표에는 **이번 GPU 런에서 새로 나온 세 모델만** 들어갑니다.

---

## 6. 결과 (3) — routing은 왜 기여가 없는가 (메커니즘)

"routing 제거해도 같다"는 결과에는 전혀 다른 두 설명이 있습니다:

- **(a)** routing은 작동하지만 agreement 신호가 이 과제에 쓸모없다
- **(b)** routing이 `n_iters=1`이 주는 균일 coupling으로 붕괴해서, **애초에 제거할 게 없었다**

학습된 가중치만으로 구분할 수 있습니다. 진단 스크립트 두 개를 새로 썼습니다.

### 6.1 `diag_coupling.py` — coupling 텐서를 직접 꺼낸다

`model.routing`에 forward hook을 걸어 **(B, 56, 3) 전체 coupling**을 수집하고,
분산을 **입력 의존(within-cell)** vs **고정 패턴(between-cell)** 으로 분해합니다.
forward() 출력만으로는 절대 보이지 않는 양입니다.

| fold | \|c − 1/3\| | 입력 간 std | cell-mean std | entropy (균일 = 1.0986) |
|---|---|---|---|---|
| 0 | 0.126 | 0.056 | 0.150 | 0.982 |
| 1 | 0.063 | 0.027 | 0.072 | 1.072 |
| 2 | 0.112 | 0.043 | 0.139 | 1.005 |
| 3 | 0.134 | 0.049 | 0.159 | 0.973 |
| 4 | 0.102 | 0.039 | 0.122 | 1.023 |

no-routing은 5 fold 전부 **정확히 0.0000 / entropy 1.0986** — 설계대로 균일입니다(검증 완료).

**결론:** coupling은 균일에서 **벗어납니다**(설명 b는 기각). 그런데
**변동의 87.8–91.4%가 고정된 per-cell 패턴**이고 **입력 의존 성분은 8.6–12.2%뿐**입니다.
고정 재가중은 뒤의 `reg_out`과 norm 기반 분류가 흡수할 수 있습니다 →
**그래서 제거해도 비용이 0입니다.**

### 6.2 `diag_routing_iters.py` — 반복을 늘리면 되는가 → 아니다

학습된 `W`로 routing loop를 손으로 10회까지 재생하며, 매 단계
"c의 변동 중 peptide에 따라 달라지는 비율"을 기록했습니다.
올라가면 예산이 부족했던 것이고, 평평하면 agreement 신호 자체에 정보가 없는 것입니다.

fold 0:
```
 iter  |c-1/3|   input-dep   mean|b|   mean|Δb|
    1  0.00000      (n/a)     0.0000   0.17875
    2  0.06317      19.1%     0.1788   0.18185
    3  0.12550      19.1%     0.3606   0.18329
    5  0.21489      20.1%     0.7271   0.18321
   10  0.30393      21.5%     1.6498   0.19004
```
(iter 1은 b=0이라 분산이 0/0 — 의미 없는 값입니다)

5 fold 전부 iter 3 → iter 10에서 **평평**: 19.1→21.5 / 17.6→17.9 / 12.5→13.7 / 12.1→12.5 / 12.9→13.8 %.

가중치 스케일:

| fold | \|W\| | ‖u‖ | ‖u_hat‖ |
|---|---|---|---|
| 0 | 0.0362 | 0.882 | 0.349 |
| 1 | 0.0195 | 0.880 | 0.171 |
| 2 | 0.0322 | 0.870 | 0.296 |
| 3 | 0.0374 | 0.892 | 0.373 |
| 4 | 0.0336 | 0.870 | 0.319 |

`W`는 **작지만 갇혀 있지는 않습니다.** 초기화는 `0.01·randn`이고 그 평균 절대값은
`0.01·√(2/π) = 0.0080`이므로, 학습 후 0.0195–0.0374는 fold별로 **2.4–4.7배 성장**한
값입니다. 즉 "초기화에서 움직이지 못했다"가 아니라 **"느리게 자라던 중"**입니다.
**2026-11-11 프로브로 확정: `W`는 50 epoch에서 자라는 중에 끊긴 것이 맞습니다.**
150 epoch까지 돌리면 `|W|`가 계속 자랍니다 — Phase3 4.5× → **6.1×**,
no-routing 6.8× → **9.8×**(성장은 느려짐, Q4/Q1 비 0.17 / 0.23).
**400 epoch에서는 Phase3 8.7×, no-routing 15.2×**까지 갑니다(Q4/Q1 0.29 / 0.28, §5.8).

**그리고 이게 `W` 초기화 스케일 실험을 사실상 불필요하게 만듭니다.**
스케일을 키워서 얻고 싶었던 게 "더 큰 `W`에서는 routing이 기여하는가"인데,
예산만 늘려도 `W`가 **init의 9–15배**까지 저절로 커지고 — 그 상태에서도
routing 기여는 여전히 0입니다(400 예산 no-routing − Phase3 = **+0.0075**, §5.8).
즉 **routing의 null 결과는 `W`가 9–15배 커진 뒤에도 살아남습니다.**
이건 방어해야 할 약점이 아니라 **주장을 강화하는 측정**입니다.

측정된 결과만 보면: `‖u_hat‖`(0.17–0.37)이 `‖u‖`(0.87)보다 훨씬 작고,
`Δb`가 반복마다 **거의 같은 값**(0.1788 → 0.1819 → 0.1833 → …)을 더하기만 합니다.

> **핵심 진단: 이건 갱신이 아니라 증폭입니다.**
> `b`가 커지면서 `|c−1/3|`은 늘어나지만, 늘어나는 방향이 매 반복 거의 같아서
> coupling은 입력과 무관한 고정 패턴을 더 선명하게 만들 뿐입니다.
> **반복을 늘리는 sweep은 불필요합니다 — 측정으로 반증됐습니다.**

---

## 7. 논문에 쓸 서술

가설 세 조항은 그대로 기각입니다. 대신 훨씬 구체적인 대체 서술이 섰습니다.
**회귀 숫자는 §5.9 GPU 5-fold로 채웠습니다** (test, paired bootstrap 1000회;
출처 `reports_gpu600_aggregate.json`):

> CapsNet은 register를 식별해서 잘하는 게 아니다. core 추적은 CNN과 같고
> (`shift_flank` 0.407 vs 0.420 — 둘 다 위치를 이미 안다고 가정한 chance의
> 3.2배 / 3.1배로, **"무작위 null 0.416"은 철회한다**, §4.1 정정 참조)
> 서열 민감도는 오히려 낮다 (scramble 비 0.81× vs 1.48×).
> **회귀 우위의 부호는 체크포인트 선택 규칙에 의존한다**: 사전 등록한 AUPRC
> 기준에서 Pearson r은 **+0.038** [+0.031, +0.045] p=0.001로 CapsNet이
> 유리하지만, 복합 기준에서는 **−0.014** [−0.020, −0.008] p=0.001로
> 부호가 뒤집히고 **두 방향 모두 CI가 0을 배제한다**. 어느 쪽도 단독으로
> 인용할 수 없다. 반면 분류 성능은 세 규칙 전부에서 CapsNet이 낮다
> (AUROC −0.036 [−0.052, −0.019], AUPRC −0.043 [−0.075, −0.007]).
> 그리고 그 어떤 이득도 dynamic routing에서 오지 않는다 — routing을 끈
> 모델과의 차이는 Pearson r **−0.001** [−0.006, +0.003] p=0.542,
> AUPRC **+0.007** [−0.023, +0.038] p=0.672로 **CI가 0을 포함한다**.
> routing의 coupling은 입력 의존 성분이 9–12%에 불과해 고정 재가중과
> 구분되지 않으며, 반복을 10회로 늘려도 그 비율이 평평하다. 예산을 600
> epoch으로 늘려 `W`가 init의 **9–15배**까지 자란 뒤에도 기여는 여전히 0이다.
> **남는 것은 capsule 구조의 이득이 아니라, 예산이 만든 착시였다.**

**취소선 그을 문장:** 이전 판은 여기에 *"회귀는 확실히 낫다 (+0.128,
95% CI [+0.117, +0.140], p<0.001; MSE 25% 감소)"*와
*"routing을 작동하게 만들면? `W` 초기화 스케일을 키우는 실험이 답이고,
성공하면 negative result가 contribution이 된다"*를 적었습니다. 둘 다 철회합니다 —
전자는 예산 artifact가 섞인 크기이고(§5.6), 후자는 예산만 늘려도 `W`가 6–10배
커지는데 routing 기여가 그대로 0이라 **실험할 이유가 없어졌습니다**(§6.2).

**지금 더 정직한 후속 질문:** routing이 아니라 **capsule 구조의 무엇이** 회귀를 돕는가.
그리고 공정 예산에서 그 이득(+0.037)이 추가 연산 비용(epoch당 70초 vs 31초, **2.2배**)을
정당화하는가 — 프로브 기준으로는 **그렇지 않습니다.**

### 정직하게 적어야 할 한계
1. 이 실험들은 **peptide branch 진단**입니다. pseudo-sequence는 실서열이지만,
   register-shift는 allele을 고정하고 peptide만 교란하므로 **cross-allele 주장의 근거는 아닙니다.**
2. **epoch 예산이 부족합니다** (§5.5). 세 모델 동일하게 부족하므로 비교는 유효.
3. `scramble_all`은 조성을 보존하지만 **길이/위치 분포까지 완전히 통제하지는 않습니다.**
   "거친 특징 의존" 가설을 확정하려면 조성 보존 대조군을 하나 더 설계해야 합니다.
4. core 추적 실험은 **예측 안정성**을 재며, 미지 core에 대한 **일반화를 재지 않습니다**
   (`core9_in_train_same_allele = 1783`, 설계상 의도).

---

## 8. 재현 방법

```bash
cd codes/custom_codes

# round 2 공정 비교 (순차, Phase3는 Phase2 체크포인트로 warm-start하므로 순서 중요)
python train_phase2.py                                          # -> reports_phase2_m2/
python train_phase3.py                                          # -> reports_phase3_m2/
PHASE3_CONFIG=config/phase3_norouting.json python train_phase3.py  # -> reports_phase3_norouting_m2/

# register-shift (모델별)
python register_shift.py --model phase2 --ckpt-dir reports_phase2_m2/checkpoints
python register_shift.py --model phase3 --ckpt-dir reports_phase3_m2/checkpoints
python register_shift.py --model phase3 --config config/phase3_norouting.json \
                         --ckpt-dir reports_phase3_norouting_m2/checkpoints

# routing 진단
python diag_coupling.py      --ckpt-dir reports_phase3_m2/checkpoints
python diag_routing_iters.py --ckpt-dir reports_phase3_m2/checkpoints
```

**경고:** Phase2가 실패하면 Phase3를 돌리지 마세요 — 이전 round의 낡은 가중치로
warm-start해서 조용히 오염됩니다. `run_m2.sh`는 `set -e`로 이걸 막습니다.

### 주요 파일

| 파일 | 역할 |
|---|---|
| `train_phase2.py` | Phase2 학습 + 공유 `selection_score()` 헬퍼 |
| `train_phase3.py` | Phase3 학습, `PHASE3_CONFIG` 환경변수로 config 교체 |
| `lomo.py` | leave-one-molecule-out, 같은 선택 기준 사용 |
| `register_shift.py` | register-shift 실험 (`--model phase2|phase3`) |
| `diag_coupling.py` | coupling 텐서 분산 분해 |
| `diag_routing_iters.py` | 학습된 W로 routing loop 재생 |
| `config/phase3_norouting.json` | `routing_iters: 1` ablation |

### 환경
CPU 전용 PyTorch 2.7.1+cpu, 22 logical core (Intel Core Ultra 7 155H).
epoch당 26–120s(경합에 따라 변동), 3개 모델 × 5 fold × 50 epoch = 약 **61시간**.

> **학습을 밤새 돌릴 때 반드시 확인:** Windows **Modern Standby(S0)** 가 학습을 멈춥니다.
> 증상이 교묘합니다 — 프로세스가 죽지 않고 그냥 느려집니다
> (11시간 20분에 2 epoch → 176일 완주 예상). Kernel-Power event 506/507로 확인 가능.
>
> ```
> powercfg /change standby-timeout-ac 0
> powercfg /change hibernate-timeout-ac 0
> powercfg /change monitor-timeout-ac 0
> ```
>
> Win+L 잠금과 화면 끄기는 **안전**합니다. 위 설정은 작업이 끝나면 되돌리세요.

---

## 9. 다음에 할 일

**끝난 것 — 수렴 프로브 (2026-11-11, §5.6)**
- [x] 1 fold × 150 epoch × 3모델. 결론: 50 epoch 예산이 **Phase2를 훨씬 더 심하게**
      굶겼고, 그 때문에 §5.2/§5.3의 회귀 격차가 **+0.132 → +0.037로 72% 과대평가**됐습니다.
- [x] `|W|` 질문도 같이 닫혔습니다 — 예산만 늘려도 init의 6–10배까지 자라고,
      그 상태에서도 routing 기여는 0. **init 스케일 실험은 불필요해졌습니다**(§6.2).

**1순위 — 공정 예산 찾고 전부 재학습 (이게 지금 유일하게 중요한 일)**

§5.2/§5.3 숫자는 현재 **논문에 쓸 수 없습니다.** 방향은 살아 있지만 크기가 틀렸습니다.

- [x] **Phase2 평평 지점 확인** — 1 fold × 400 epoch. **epoch ~301에서 평평**(§5.7).
      공정 예산은 50이 아니라 **300 이상**이고, r이 0.523 → 0.729까지 올라
      격차의 **부호가 뒤집힙니다**(−0.035).
- [x] **Phase3 / no-routing도 동일한 400 예산으로** 1 fold 프로브 (2026-11-19 완료, §5.8).
      **회귀 우위가 사라집니다** — Phase3 − Phase2 = **−0.032**(복합 기준 체크포인트),
      r 최대끼리는 −0.007로 동률. routing 기여는 400에서도 0(**+0.0075**).
- [ ] **세 모델 5-fold 전부 재학습**, 400 epoch 기준 합계 ≈ **86 h**
      (Phase2 17 h + Phase3 39 h + no-routing 31 h). 예산이 6배로 커져서
      비용도 50 epoch 시절의 8배입니다. 이게 끝나야 §5의 모든 표와
      부트스트랩 CI, §7의 서술을 다시 씁니다.
- [ ] 재학습 후 **분류 트레이드오프 재확인.** 프로브에서는 AUROC 격차가 오히려
      **벌어졌고**(0.860 vs 0.809) 복합 지표로는 Phase2가 역전했습니다. 사실이면
      "회귀는 CapsNet, 분류는 CNN"이 "CapsNet의 이득은 회귀에 한정되고 그마저 작다"로
      약해집니다 — 서술 전체가 달라집니다.

**3순위 — 재학습 이후로 미룸** (지금 측정해도 예산 바뀌면 다시 해야 합니다)
- [ ] **국소 셔플 대조군.** `scramble_all`이 이미 조성을 보존하므로 "조성 보존 대조군"은
      새로 얻는 게 없습니다. 필요한 건 3-mer/4-mer 블록 단위 셔플 — 국소 순서는 보존하고
      전역 배치만 깨서 Phase3가 쓰는 특징의 **입자 크기**를 재는 설계입니다.
      전역에는 둔감한데 국소에 민감하면 "짧은 motif", 둘 다 둔감하면 "정말 조성만".
- [ ] per-allele register 측정 — pseudo-sequence가 실서열이므로 이제 가능
- [ ] register-shift 그림에 core-tracking 패널 추가

**사람이 해야 하는 일**
- [ ] NetMHCIIpan-4.3 라이선스 — DTU 서버가 이 환경에서 접근 불가(HTTP 000)
- [ ] 선행 연구가 core 식별을 어떻게 평가하는지 문헌 조사
