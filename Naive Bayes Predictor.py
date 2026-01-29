import math
import json

# Function to load data from a JSON file
def get_data(match):
    with open(match, 'r') as f:
        Data = json.load(f)
    return Data

# Function to extract usable data from the loaded JSON structure (For the whole match)
def extract_usable_data(Data):
    data = {}
    for point, val in Data.items():
        data[point] = []
        for i in range(len(val["PointInfo"]["Rally"])):
            rally = val["PointInfo"]["Rally"]
            player = rally[i]["Player"]
            stroke = rally[i].get("StrokeType", None)
            data[point].append([player, stroke])
    return data

# calculate prob of winning a point given the strokes played in the rally
def prob(point: str):
    if point not in data:
        print(f"{point} not found in data.")
        return
    rally = data[point]
    prob = 0
    cnt_T1 = {}
    cnt_T2 = {}
    Nw, Nw_ = 0, 0
    rally_won_T1P1 = Data[point]["PointInfo"]["T1P1"]["Point"]
    rally_won_T2P1 = Data[point]["PointInfo"]["T2P1"]["Point"]
    if rally_won_T2P1 == 0:
        rally_won_T2P1 = 1
    ratio = (rally_won_T1P1) / (rally_won_T2P1)
    if ratio > 0:
        prob += math.log(ratio)
    for i in range(len(rally)):
        [player, stroke] = rally[i]
        if player == "T1P1":
            cnt_T1[stroke] = cnt_T1.get(stroke, 0) + 1
            Nw += 1
        else:
            cnt_T2[stroke] = cnt_T2.get(stroke, 0) + 1
            Nw_ += 1
        cnt1, cnt2 = cnt_T1.get(stroke, 0), cnt_T2.get(stroke, 0)
        if Nw != 0 and cnt2 != 0:
            val = (cnt1 * Nw_) / (Nw * cnt2)
            if val > 0:
                prob += math.log(val)
    
    if prob > 0:
        print(f"T1P1 is more likely to win the point {point}")
    else:
        print(f"T2P1 is more likely to win the point {point}")
    print(prob)
        
dataset = ["match1.json"] # Placeholder for dataset paths
for match in dataset:
    Data = get_data(match)
    data = extract_usable_data(Data)
    for point in data.keys():
        prob(point)


