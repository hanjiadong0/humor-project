TRAINING DATA EXPORTS

1) train_detector.csv
   - Use for humor detection (binary): predict is_humor from text_norm.
   - Rows: 8000

2) train_scorer.csv
   - Use for humor scoring on humor subset only (is_humor should be 1 here).
   - Predict humor_rating (regression) or humor_bucket3 (classification).
   - Rows: 4932

3) train_multitask.csv
   - Use for multi-head training.
   - Always train is_humor and offense_rating heads.
   - Train humor_rating / humor_controversy heads ONLY when has_humor_score==True (mask loss).
   - Rows: 8000
