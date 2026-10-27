# ML-Driven Intermediate Code Optimization
Predicts the best sequence of compiler optimization passes for a three-address-code (TAC) program.

## Run
    pip install -r requirements.txt
    python -m mltac.run_experiments   # builds dataset, trains models, writes results/ (≈1 min)
    python -m mltac.demo              # source -> TAC -> ML-chosen passes -> optimized TAC

## Files
- mltac/tac.py              IR, interpreter, cost model, 5 passes (cf, alg, cse, cp, dce), 10 pass sequences
- mltac/data.py             synthetic program generator, 22 static features, mini front end (source -> TAC)
- mltac/run_experiments.py  labelling, models, CV, threshold extension, robustness test, figures
- results/                  results.json + figures used in the paper

## Method in one paragraph
For each program, all 10 candidate pass sequences are applied; the label is the sequence with the lowest
cost (ties -> fewest passes). Every optimized program is checked for semantic equivalence by random testing.
Models predict the label from static features of the unoptimized TAC.
