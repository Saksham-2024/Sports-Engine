Flow of data
1. Detect_KeyFrames.py 
  -> Extract timestamps from json files and save the frames in Keyframes dir
  -> Save frames in outputs folder as keyframes_metadata.csv
2. player_positions.py 
  -> Detect players in the frames and save the annotated frames in Skeleton_marked_frames dir
  -> Save the player positions in outputs folder as player_positions.csv
3. feature_extraction.py 
  -> Extract features from the player positions and save the features in outputs folder as features.csv
4. reduce_classes.py
    -> reduce the number of classes in the features.csv
    -> save the processed features in outputs folder as features1.csv
5. preprocessor.py 
  -> Preprocess the features and save the processed features in outputs folder as processed_data/sequences.pkl, processed_data/categorical_encoders.pkl, processed_data/shot_label_encoder.pkl, processed_data/feature_list.json
6. decision_tree.py
    -> defines 6 tree models (3 without context and 3 with context)
    -> saves the models in models/naive_approach_models/ml_models
7. inference_classical_ml.py
    -> inference on the classical ml models
8. lstm.py
    -> defines lstm architecture 
    -> saves the models in ../models/naive_approach_models/lstm
9. train_lstm.py
    -> train the lstm models 
    -> saves the models in ../models/naive_approach_models/
10. inference_lstm.py
    -> inference on the lstm models
11. train_YOLO_on_shuttlecock.py
    -> fine tunes the yolov8 model on the shuttlecock images
    -> saves the models in ../runs


FILES MAY NEED TO BE RERUN AND CHECKED FOR CORRECTNESS OF PATHS