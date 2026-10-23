## Results (105 trials)

| config | task | pass | mean score | submit | tool calls | repeats | tokens (k) | agent s |
|---|---|---|---|---|---|---|---|---|
| A_raw | seq-align-affine | 0/5 | 0.12 | 4/5 | 13.0 | 6.6 | 101 | 123 |
| A_raw | reaction-network-stiff | 0/5 | 0.23 | 5/5 | 9.6 | 3.2 | 55 | 36 |
| A_raw | seir-changepoint-inference | 0/5 | 0.09 | 2/5 | 27.8 | 18.8 | 151 | 115 |
| B_guard | seq-align-affine | 0/5 | 0.15 | 5/5 | 12.8 | 6.2 | 48 | 75 |
| B_guard | reaction-network-stiff | 0/5 | 0.00 | 3/5 | 22.6 | 17.4 | 103 | 67 |
| B_guard | seir-changepoint-inference | 0/5 | 0.00 | 5/5 | 16.2 | 4.6 | 115 | 146 |
| C_method | seq-align-affine | 0/5 | 0.03 | 5/5 | 9.0 | 5.0 | 27 | 32 |
| C_method | reaction-network-stiff | 0/5 | 0.00 | 5/5 | 9.8 | 6.8 | 26 | 35 |
| C_method | seir-changepoint-inference | 0/5 | 0.00 | 5/5 | 16.0 | 8.0 | 87 | 68 |
| D_think | seq-align-affine | 0/5 | 0.11 | 5/5 | 2.4 | 0.2 | 38 | 171 |
| D_think | reaction-network-stiff | 0/5 | 0.25 | 5/5 | 2.6 | 0.0 | 13 | 172 |
| D_think | seir-changepoint-inference | 0/5 | 0.14 | 5/5 | 2.8 | 0.0 | 43 | 190 |
| E_think_method | seq-align-affine | 0/5 | 0.11 | 5/5 | 3.6 | 0.4 | 63 | 218 |
| E_think_method | reaction-network-stiff | 0/5 | 0.05 | 5/5 | 3.0 | 0.4 | 24 | 158 |
| E_think_method | seir-changepoint-inference | 0/5 | 0.08 | 5/5 | 2.4 | 0.0 | 38 | 164 |
| F_check | seq-align-affine | 0/5 | 0.12 | 3/5 | 24.0 | 17.6 | 192 | 430 |
| F_check | reaction-network-stiff | 0/5 | 0.85 | 5/5 | 11.8 | 1.8 | 91 | 75 |
| F_check | seir-changepoint-inference | 0/5 | 0.03 | 5/5 | 15.0 | 6.2 | 116 | 162 |
| G_think_check | seq-align-affine | 0/5 | 0.09 | 5/5 | 2.2 | 0.0 | 49 | 353 |
| G_think_check | reaction-network-stiff | 1/5 | 0.80 | 3/5 | 14.6 | 3.4 | 124 | 1293 |
| G_think_check | seir-changepoint-inference | 0/5 | 0.28 | 5/5 | 4.2 | 0.6 | 54 | 609 |

## Failure modes

| failure mode | seq-align-affine | reaction-network-stiff | seir-changepoint-inference | total |
|---|---|---|---|---|
| submitted wrong/partial solution | 25 | 26 | 32 | 83 |
| too slow (grader timeout) | 7 | 4 | 0 | 11 |
| stuck in repetition loop | 3 | 2 | 3 | 8 |
| ran out of turns/time | 0 | 2 | 0 | 2 |
| solved | 0 | 1 | 0 | 1 |

## Numerical approach in final code

| config | task | approaches in final code (count) |
|---|---|---|
| A_raw | reaction-network-stiff | implicit: 5, tight-rtol: 2 |
| A_raw | seir-changepoint-inference | scans-change-day: 4 |
| B_guard | reaction-network-stiff | implicit: 5, tight-rtol: 1 |
| B_guard | seir-changepoint-inference | scans-change-day: 5 |
| C_method | reaction-network-stiff | implicit: 5, tight-rtol: 5 |
| C_method | seir-changepoint-inference | model-selection: 5, scans-change-day: 3 |
| D_think | reaction-network-stiff | implicit: 5, tight-rtol: 3 |
| D_think | seir-changepoint-inference | scans-change-day: 5 |
| E_think_method | reaction-network-stiff | implicit: 5, tight-rtol: 5 |
| E_think_method | seir-changepoint-inference | scans-change-day: 5, model-selection: 5 |
| F_check | reaction-network-stiff | implicit: 5, tight-rtol: 5 |
| F_check | seir-changepoint-inference | scans-change-day: 3 |
| G_think_check | reaction-network-stiff | implicit: 5, tight-rtol: 5 |
| G_think_check | seir-changepoint-inference | scans-change-day: 5 |

## Verification behaviour

| config | trials | mean score | silent-success submit | empty runs / trial | truncated replies | used check.py |
|---|---|---|---|---|---|---|
| A_raw | 15 | 0.144 | 10/15 | 2.6 | 3 | 0/15 |
| B_guard | 15 | 0.048 | 9/15 | 8.3 | 5 | 0/15 |
| C_method | 15 | 0.010 | 15/15 | 7.9 | 0 | 0/15 |
| D_think | 15 | 0.165 | 6/15 | 0.5 | 11 | 0/15 |
| E_think_method | 15 | 0.079 | 5/15 | 0.5 | 12 | 0/15 |
| F_check | 15 | 0.335 | 1/15 | 0.4 | 0 | 15/15 |
| G_think_check | 15 | 0.390 | 0/15 | 0.0 | 9 | 8/15 |
