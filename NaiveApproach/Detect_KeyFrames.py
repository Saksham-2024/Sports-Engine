import cv2
import os
import pandas as pd
import json
# import get_data from NaiveBayes_Predictor
# Function to get FPS of a video. called for each video in video_path while processing
output_dir = "KeyFrames"
video_dir = "../dataset/videos/"
json_dir = "../dataset/json/"
os.makedirs(output_dir, exist_ok=True)
records = []

def get_fps(video_path):
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def get_data(match):
    with open(match, 'r') as f:
        Data = json.load(f)
    return Data

def extract_timestamps(Data):
    data = {}
    for point, val in Data.items():
        data[point] = []
        rally = val["PointInfo"]["Rally"]
        for stroke in rally:
            strokeBegin = stroke.get("StrokeBegin")
            strokeEnd = stroke.get("StrokeEnd")
            strokeType = stroke.get("StrokeType")
            playingSide = stroke.get("Player")
            if playingSide == "T1P1":
                playingSide = 1
            else:
                playingSide = 2
                
            if strokeBegin is None or strokeEnd is None or strokeType is None:
                continue
            timestamp = (strokeBegin + strokeEnd) * 3 / 4
            data[point].append([stroke["StrokeNum"], strokeType, playingSide, timestamp])
    return data

def extract_frame(fps, timestamp):
    frame_no = int(fps * timestamp)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def save_frame(match, frame, point, stroke_num, stroke_type, playingSide):
    filename = f"match_{match}_point_{point}_stroke_{stroke_num}.jpg"
    image_path = os.path.join(output_dir, filename)
    cv2.imwrite(image_path, frame)
    records.append({
        "match_no": match,
        "rally_id": point,
        "stroke_num": stroke_num,
        "stroke_type": stroke_type,
        "hitting_player": playingSide,
        "hit_time": timestamp,
        "image_path": image_path
    })

video_paths = [video_dir + "match1.mp4", video_dir + "match2.mp4", video_dir + "match3.mp4", video_dir + "match4.mp4", video_dir + "match5.mp4", video_dir + "match6.mp4", video_dir + "match7.mp4", video_dir + "match8.mp4", video_dir + "match9.mp4"]
dataset = [json_dir + "match1.json", json_dir + "match2.json", json_dir + "match3.json", json_dir + "match4.json", json_dir + "match5.json", json_dir + "match6.json", json_dir + "match7.json", json_dir + "match8.json", json_dir + "match9.json"]
for i, match in enumerate(dataset):
    Data = get_data(match)
    data = extract_timestamps(Data)
    video_path = video_paths[i] # Corresponding video path ## both dataset and video_paths should be in same order and have same size
    cap = cv2.VideoCapture(video_path)
    fps = get_fps(video_path)
    for point, val in data.items():
        for stroke in val:
            stroke_num, stroke_type, playingSide, timestamp = stroke
            frame = extract_frame(fps, timestamp)
            if frame is not None:
                save_frame(i+1, frame, point, stroke_num, stroke_type, playingSide)

df = pd.DataFrame(records)
df.to_csv("keyframes_metadata.csv", index=False)
cap.release()
    


