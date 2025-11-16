Ooredoo TAC Quantum Use-Case — Dataset Bundle
---------------------------------------------
Files:
  - cells.csv
  - neighbors.csv
  - paging_affinity.csv
  - handovers.csv
  - constraints.json
  - baseline_assignment.csv
  - example_submission.csv
  - validator.py

Quick start:
  python validator.py --data_dir . --assignment example_submission.csv

Objective (Ooredoo):
  C = λ_p*H_paging + λ_m*H_mobility + λ_g*H_geo + λ_b*H_balance
  with H_onehot enforced as a hard constraint by the validator.
