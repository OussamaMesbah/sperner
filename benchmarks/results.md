Rent 3000, 200 simulated flats per row.

### Questions

| Flatmates | Precision | Questions per person, mean | 90th percentile | Answers only | Largest envy / precision |
|---|---|---|---|---|---|
| 2 | 30 | 3.3 | 4.0 | 100% | 0.98 |
| 2 | 10 | 3.9 | 5.0 | 100% | 0.95 |
| 2 | 3 | 4.4 | 5.5 | 100% | 0.87 |
| 3 | 30 | 10.1 | 14.0 | 100% | 0.99 |
| 3 | 10 | 12.3 | 17.0 | 100% | 0.98 |
| 3 | 3 | 14.4 | 20.7 | 100% | 0.98 |
| 4 | 30 | 14.3 | 21.0 | 100% | 1.24 |
| 4 | 10 | 19.2 | 28.8 | 100% | 1.24 |
| 4 | 3 | 38.5 | 49.8 | 100% | 1.23 |
| 5 | 30 | 27.5 | 40.6 | 100% | 1.36 |
| 5 | 10 | 38.7 | 62.8 | 100% | 1.40 |
| 5 | 3 | 52.8 | 87.0 | 100% | 1.37 |

### Methods compared (precision 10)

Inputs are questions answered, or values reported for Spliddit. The guarantee bounds envy for quasi-linear flatmates: twice the precision reached.

| Method | People | Tolerance | Flats | Inputs per person, mean | 90th percentile | Answers only | Envy, mean | Envy, max | Largest envy / guarantee |
|---|---|---|---|---|---|---|---|---|---|
| sperner | 2 | 10 | 200 | 3.9 | 5.0 | 100% | 0.21 | 5.84 | 0.47 |
| single walk | 2 | 10 | 200 | 69.7 | 88.5 | 100% | 1.10 | 9.92 | 0.50 |
| divide and choose | 2 | 10 | 200 | 5.0 | 5.0 | 100% | 1.64 | 5.79 | 0.58 |
| spliddit | 2 | 10 | 200 | 2.0 | 2.0 | 100% | 0.00 | 0.00 | exact |
| sperner | 3 | 10 | 200 | 12.3 | 17.0 | 100% | 0.57 | 4.05 | 0.49 |
| single walk | 3 | 10 | 200 | 149.2 | 191.7 | 100% | 2.60 | 9.96 | 0.50 |
| spliddit | 3 | 10 | 200 | 3.0 | 3.0 | 100% | 0.00 | 0.00 | exact |

### Three rooms, two flatmates, one newcomer

| Precision | Questions per person, mean | 90th percentile |
|---|---|---|
| 30 | 26.6 | 35.0 |
| 10 | 33.1 | 42.0 |
| 3 | 40.0 | 52.0 |

### Budgets (three flatmates, precision 10)

Each flatmate has a budget between 34% and 44% of the rent, and rent above it hurts four times as much. Spliddit gets the values without the budgets, because its model cannot express them.

| Method | Flats where someone pays over budget | Amount over budget per person, mean | Envy under true preferences, mean | Largest envy |
|---|---|---|---|---|
| Spliddit (values only) | 66% | 27.0 | 179.9 | 1259.9 |
| sperner (answers) | 50% | 7.9 | 1.3 | 13.2 |
