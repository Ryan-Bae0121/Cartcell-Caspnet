# cartcell 서버 상태 복원 노트 (채팅 기록 기반)

> `cartcell_rebuild_briefing.md` 를 보완하는 문서. 유실된 `eantec-server`
> (`/home/eantec/etc/cartcell/`) 에서 실제로 무엇을 돌렸는지, 과거 Cursor/Claude
> 대화 복붙에서 확인된 내용만 정리. **코드 원본이 아니라 대화에서 언급된 스펙.**
> 서버 복구 시 실제 파일과 대조 필수.

---

## 1. 실제로 쓴 데이터 파일

### (A) 메인 — `ic50_processed_with_affinity_aggressive.csv`  ★
- 위치: 프로젝트 루트. 약 **34,408 샘플**.
- 컬럼: `species, mhc, peptide_length, sequence, inequality, meas,
  IC50_numeric_aggressive, affinity_aggressive, cv_fold`
- 모델 입력 매핑:
  - peptide = `sequence`,  allele = `mhc`
  - 회귀 타깃 / 임계값 = `affinity_aggressive` (있으면 이 컬럼으로 affinity 덮어씀)
  - **5-fold = `cv_fold ∈ {1..5}` 고정 컬럼** (랜덤 KFold 아님).
    train = `cv_fold != k`, val = `cv_fold == k`
  - 이진 라벨(strong binder) = `affinity >= 0.8`
- Phase2 scratch / Phase3 aggressive / ensemble / LOMO / interpretability 전부 이 파일 사용.
- `phase3_capsnet_aggressive_summary.json` 의 `data_path` 도 이 파일.

### (B) non-aggressive 변형 — `ic50_processed_with_affinity.csv`
- 컬럼: `IC50_clean, affinity, cv_fold` (aggressive 컬럼 없음).
- 일부 baseline / README (`README_PHASE2_AFFINITY_FROM_SCRATCH.md`,
  `run_baseline_on_ic50_affinity.py`) 가 기본 경로로 사용. 최신 실험은 (A).

### (C) 구 Phase2 표준 스크립트용 — `capsnet_mhc2_phase2/data/merged_with_split_phase1.csv`
- 컬럼: `peptide, allele, label, split(train/val), measure` — IEDB/merged Phase1 형식.
- `capsnet_mhc2_phase2/scripts/train_phase2.py` 가 사용.

### (D) Phase 1 (슬라이딩 9-mer) — `train_phase1.py --csv <파일>`
- 스키마: `peptide, allele, label, split`. 고정 파일 없음.

### (E) allele 보조 데이터
- **`capsnet_mhc2_phase2/resources/mhc_pseudoseq.json`** — allele 문자열 → pseudo-sequence
  매핑 파일. 대화에서 "allele → pseudo-sequence" 라고만 언급됨. **브리핑 §1.4 는 서버에서도
  pseudo-seq 가 placeholder("VVVV" 반복류) 상태였고 실서열 교체가 미해결이라고 명시** →
  이 JSON 이 실서열이었는지 placeholder 였는지 불확실. (2026-09-10: 사용자 "못 구할 것 같다"
  → 복구 포기. 서버가 AUROC 0.86 을 placeholder pseudo-seq 로 낸 것으로 보이므로 로컬도
  placeholder 로 진행해도 성능 재현에 큰 지장 없을 것으로 판단.)

> ⚠️ `ic50_processed*.csv` 를 만든 가공 스크립트/노트북은 이 워크스페이스 밖에 있을 수 있음.
> 원본 IEDB 필터 조건은 CSV 안의 `species, meas, inequality` 로만 추적 가능.

### 로컬 재구성과의 차이
| 항목 | 서버 | 현재 로컬 |
|---|---|---|
| 데이터 | `ic50_processed_with_affinity_aggressive.csv` (fold 컬럼 포함) | `merged_data_without.csv` → `data_pipeline.py` 로 재생성 |
| fold | `cv_fold` 고정 컬럼 | `KFold(shuffle, seed=42)` 랜덤 |
| pseudo-seq | `mhc_pseudoseq.json` (실서열) | 해시 기반 placeholder |
| aggressive 변환 | `IC50_numeric_aggressive` 컬럼 | `data_pipeline.py` 가 동일 로직 재현 |

---

## 2. 서버 디렉토리 구조 (복원)

```
/home/eantec/etc/cartcell/
├── ic50_processed_with_affinity_aggressive.csv        # 메인 데이터
├── ic50_processed_with_affinity.csv
├── reports_phase2_affinity_from_scratch/checkpoints/  # Phase2 ckpt (fold_k/best.ckpt | phase2/best.ckpt)
├── phase3_capsnet_aggressive/checkpoints/             # Phase3 ckpt
├── phase3_capsnet_aggressive_summary.json
├── capsnet_mhc2_phase2/
│   ├── data/merged_with_split_phase1.csv
│   ├── resources/mhc_pseudoseq.json                   # 실 pseudo-seq
│   ├── scripts/train_phase2.py
│   └── checkpoints/phase2/best.ckpt
├── capsnet_mhc2_phase3/
│   ├── phase3_discriminative_lr/     Val AUROC 0.7839  ✅
│   ├── phase3_cap_dim16/             Val AUROC 0.7812  ✅
│   ├── phase3_core_topk/             config만          ⏳
│   ├── phase3_augmentation/          config만          ⏳
│   └── phase3_multiseed/             README+config      ⏳
├── warmfix_full_rerun/                                # 최종 전체 재학습 (~2.6h GPU)
│   ├── phase2_cv/phase2_affinity_from_scratch_summary.json
│   ├── phase3_full_warmstart/phase3_capsnet_aggressive_summary.json
│   └── phase3_core_only_warmstart/phase3_capsnet_aggressive_summary.json
├── ensemble/
│   ├── oof_predictions/oof_predictions_{phase2,phase3,ensemble}.csv   # 각 34,408행
│   └── ensemble_phase2_phase3_ic50_affinity_summary__default.json
├── train_phase3_ic50_affinity_aggressive.py           # --phase2_ckpt_dir, --warm_start_core_only
├── train_phase1.py
├── create_performance_comparison_figure.py            # create_grouped_comparison_figure()
├── create_lomo_figure.py
└── interpretability/
    ├── phase3_core_position_analysis.py               # 4-category core position
    ├── core_hitrate_evaluation.py                     # NetMHCIIpan hit-rate, --synthetic_reference
    ├── per_allele_evaluation*.py
    └── (figure PNG / tex 다수 — 4절)
```

---

## 3. 최종 성능 (warmfix_full_rerun, 5-fold CV — 권위 있는 수치)

| 모델 | AUROC | AUPRC | Pearson r | MSE |
|---|---|---|---|---|
| Phase3 **full warm-start** (수정 로더, core+MHC) | 0.8601 ± 0.0116 | 0.3073 ± 0.0315 | 0.6934 ± 0.0223 | ~0.0442 |
| Phase3 core-only warm-start (구 방식) | 0.8623 ± 0.0080 | 0.2968 ± 0.0312 | 0.6643 ± 0.0666 | ~0.0467 |
| 저장돼 있던 옛 Phase3 | 0.8588 ± 0.0090 | 0.3027 ± 0.0130 | 0.6986 ± 0.0038 | — |

**OOF (ensemble/ 체크포인트 기준):** Phase2 r = 0.4892 · Phase3 r = 0.6937 · Ensemble r = 0.6771

- → 브리핑 §5 표(Phase2 r≈0.49 / Phase3 r≈0.70 / Ensemble r≈0.68)와 일치. 브리핑 §5 = 이 러닝.
- 앞서 언급됐던 "Val AUROC 0.80" 계열(discriminative_lr 0.7839 등)은 **더 이른 HP 튜닝 단계**의
  단일 split 값. 최종 5-fold OOF 와 다른 지표.

### full warm-start vs core-only 결론
- AUROC: core-only 가 평균 아주 조금 위지만 **노이즈 수준**.
- Pearson r: **full warm 이 평균 높고 fold 간 std 훨씬 작음 (0.022 vs 0.067).**
  fold 3 에서 core-only 는 r 0.536 까지 무너지는데 full warm 은 0.652 유지.
- MSE 도 full warm 이 약간 유리.
- **→ 배포/재구현 기준 = full warm-start (core + MHC).** (로컬 재구성은 `pseudo_encoder`
  이름을 이미 통일해 놔서 이 방식이 바로 가능.)

---

## 4. LOMO 평가 (60개 unseen allele, 각 ≥ 50 test 샘플)

| Metric | Phase2 (CNN) | Phase3 (CapsNet) | Ensemble |
|---|---|---|---|
| MSE | 0.074 ± 0.031 | 0.070 ± 0.029 | **0.069 ± 0.030** |
| MAE | 0.223 ± 0.052 | **0.215 ± 0.050** | 0.215 ± 0.051 |
| Pearson r | 0.409 ± 0.180 | 0.421 ± 0.160 | **0.444 ± 0.162** |
| AUROC | 0.719 ± 0.199 | 0.706 ± 0.184 | **0.726 ± 0.199** |
| AUPRC | 0.179 ± 0.189 | 0.158 ± 0.181 | **0.190 ± 0.193** |
| F1 | 0.147 ± 0.157 | **0.173 ± 0.170** | 0.164 ± 0.170 |
| Best on (r / AUROC) | 16 / 32 | 15 / 12 | 29 / 16 |

- Ensemble = 평균 AUROC·Pearson r 최고. Phase3 = variance 최저 (robust generalization).
- ⚠️ 브리핑 §4.3 데이터 누수(`val_df = test_df`) 는 재현하지 말고 Limitation 에 명시.

---

## 5. 논문 Figure 인벤토리

| Figure | 파일 | 내용 |
|---|---|---|
| **Fig 1** — 5-fold CV 성능 비교 | `interpretability/performance_comparison_grouped.png` | grouped bar, Phase2/3/Ensemble, mean±std. 상단 Regression(MSE·MAE·Pearson r·Spearman ρ), 하단 Classification(AUROC·AUPRC·Accuracy). 생성: `create_performance_comparison_figure.py :: create_grouped_comparison_figure()` |
| (Fig 1 보조 표) | `performance_table_phase2_phase3_ensemble.png`, `performance_table_simple.png` | 테이블 이미지 |
| **Fig 3** — Per-allele 평가 | `interpretability/per_allele_evaluation_comprehensive.png` | 2패널: A) Pearson r box plot, B) AUROC box plot. allele별 분포, ≥30 샘플. |
| Fig 3 보조 | `per_allele_boxplot_all_metrics.png`, `per_allele_heatmap_{pearson,auroc}.png` (Top 30), `per_allele_scatter_phase2_vs_phase3.png`, `per_allele_top_alleles_barplot.png` (Top 10) | |
| **Fig 4a** — binding core position heatmap | `interpretability/core_position_global_heatmap.png` | x=core 시작 위치, y=카테고리, 색=평균 `pos_prob`. DRB1*07:01 strong 집중 vs weak 분산 패턴은 `core_position_by_allele_HLA-DRB1_07_01.png` 에서 확인. |
| **Fig 4b** — ROC/PR + scatter 합본 | `interpretability/figure4_roc_pr_scatter.png` | A) `performance_comparison_phase2_phase3_ensemble.png` + B) `per_allele_scatter_phase2_vs_phase3.png` |
| (Fig 4b A 최종본) | `performance_comparison_phase2_phase3_ensemble.png` | Regression: MSE·MAE·Pearson r / Classification: AUROC·AUPRC. 타이틀 "(A) Regression Performance (Phase3 leads)", "(B) Classification Performance (Ensemble strongest)" |
| **LOMO figure** | `interpretability/lomo_evaluation_comprehensive.png` | 2패널 box plot (A Pearson r, B AUROC), ★ best, Phase3 low-variance 강조 |
| LOMO 보조 | `lomo_variance_comparison.png`, `lomo_scatter_comparison.png` | |
| LOMO scatter 대안 (`create_lomo_figure.py`) | `lomo_performance_comparison.png` (ECDF + 승패 매트릭스), `lomo_performance_raincloud.png` (violin+box+점), `lomo_performance_delta_bars.png` (정렬된 Δ 막대) | |
| **LOMO LaTeX 표** | `interpretability/lomo_evaluation_summary_table.tex` | `\caption{...N=60...}`, best 값 `\mathbf`. `booktabs` 필요. |

---

## 6. Interpretability 스크립트 상세

### `phase3_core_position_analysis.py` — 4 카테고리 core position
| 카테고리 | 조건 | 색 |
|---|---|---|
| Non-binders | `affinity < nonbinder_threshold` (기본 0.4) | 파랑 |
| Weak binders | `nonbinder_threshold ≤ affinity < affinity_threshold` | 주황 |
| Strong binders | `affinity ≥ affinity_threshold` (기본 0.8) | 빨강 |
| All binders | `affinity ≥ nonbinder_threshold` (Weak ∪ Strong) | 회색 점선 |

- CLI: `--nonbinder_threshold 0.4  --affinity_threshold 0.8  --min_category_samples 5
  --top_n_alleles 8  --device {cpu,cuda}`
- `nonbinder_threshold >= affinity_threshold` 이면 에러.
- 출력: `core_position_global_heatmap.png`, `core_position_global_strong_vs_weak.png`(내용은 4카테고리),
  `core_position_global_{nonbinder,weak,strong,all_binders}.npy`,
  `core_position_by_allele_*.png` (상위 allele), `core_position_data_summary.csv` (34,408행).
- 건드리지 않은 부분: `collect_core_positions`, fold 루프, ckpt/데이터 로딩.
- 실행 예 (~18s CPU):
  ```
  python3 interpretability/phase3_core_position_analysis.py \
    --nonbinder_threshold 0.4 --affinity_threshold 0.8 --min_category_samples 5
  ```

### `core_hitrate_evaluation.py` — NetMHCIIpan core hit-rate
- NetMHCIIpan **4.1 설치 필요.** 없으면 `--synthetic_reference` (랜덤 reference 로 파이프라인 테스트).
- `scipy.stats.mcnemar` 없음 → chi2 로 직접 구현.
- synthetic 실행 결과: McNemar p ≈ 0.36 (Phase2 vs Phase3 exact hit 차이 무의미).
- 출력: `interpretability/core_hitrate/{core_hitrate_summary.csv, allele_core_hitrate.csv,
  core_hitrate_overall_comparison.png, core_hitrate_allele_boxplot.png,
  phase2_with_reference.csv, phase3_with_reference.csv}`
- 실제 실행:
  ```
  python interpretability/core_hitrate_evaluation.py \
    --data_path ic50_processed_with_affinity_aggressive.csv \
    --netmhciipan_path /path/to/netMHCIIpan \
    --output_dir interpretability/core_hitrate
  ```

---

## 7. 복구된 학습 설정 & 버그 수정

### 설정
- **Phase2** (warmfix rerun): 5-fold CV, 30 epoch, batch 32, `affinity_aggressive`, **regression λ = 0.5**.
  ⚠️ 브리핑 §3 은 Phase2 5-fold λ_reg = 0.1 이라고 함 → **불일치**. 최종 rerun 은 0.5 사용.
  (로컬 `config/phase2.json` 은 현재 0.1 — 재확인 필요.)
- **Phase3**: warm-start from Phase2 **fold별** 체크포인트 (`fold_k/best.ckpt`), core + MHC.
  `--warm_start_core_only` 로 MHC 스킵(구 방식 재현). `train_phase3_ic50_affinity_aggressive.py --phase2_ckpt_dir`.
  λ_reg = 0.5, 40 epoch. (브리핑 §2 아키텍처: 3-branch 56 primary capsule, dynamic routing ×3.)
- 전체 재학습 3단계(Phase2 → Phase3 full warm → Phase3 core-only) ~2.6h GPU, `DONE_ALL_THREE`.

### 버그 수정 (재구현 시 처음부터 반영)
- `scipy.stats.mcnemar` 미존재 → McNemar chi2 직접 구현.
- Phase2 클래스명 `MHC2Phase2Model` 로 통일. path 순서에서 phase2 가 phase3 보다 먼저 로드.
- Phase2 ckpt fallback: `fold_k` 없으면 `phase2/best.ckpt`.
- matplotlib `labels=` → `tick_labels=` (deprecation).
- 브리핑 §2 Phase3 버그: warm-start 시 `mhc_encoder` vs `pseudo_encoder` 이름 불일치로 MHC
  브랜치가 로드 안 되던 문제 → 이름 통일로 해결 (로컬 재구성은 `pseudo_encoder` 로 이미 통일).

---

## 8. 재구성 우선순위 (이 노트 반영)

> 2026-09-10: `mhc_pseudoseq.json` / `ic50_processed_with_affinity_aggressive.csv` 원본 복구 불가 확정.
> → placeholder pseudo-seq + `merged_data_without.csv` 로컬 파이프라인으로 진행.
> 잃는 것: 서버의 고정 `cv_fold` (→ per-fold 수치 1:1 재현 불가, mean 수준 비교는 유효),
> 실서열 pseudo-seq (→ 서버도 placeholder였을 가능성 높아 영향 제한적).

1. 2차 Phase3: full warm-start (core + MHC) 기준으로 구현. λ_reg = 0.5, 40 epoch.
2. 3차: ensemble(α 스윕, OOF) → per-allele(≥30) → LOMO(≥50, 60 allele, 누수 재현 X).
3. Figure 재생성: Fig1 grouped bar / Fig3 per-allele box / Fig4 core heatmap + ROC·PR·scatter 합본 / LOMO box + LaTeX 표.
4. Phase2 λ_reg 0.1 vs 0.5 불일치 확인 후 config 확정.
5. (선택, 나중) 실 MHC-II pseudo-seq 를 공개 소스에서 재구축 — NetMHCIIpan 34-position 정의 +
   IMGT/HLA DRB/DQA·DQB/DPA·DPB 정렬. 블로커 아님, 성능 개선용.
