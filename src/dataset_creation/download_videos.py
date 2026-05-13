import yt_dlp #type: ignore
import os
import yaml

with open('../../configs/configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)
    
video_links = [
    "https://www.youtube.com/watch?v=TXT-qlniM90",
    'https://www.youtube.com/watch?v=gloiZ_gTJaE',
    'https://www.youtube.com/watch?v=5W6txLGZ1Rs&t=8s',
    'https://www.youtube.com/watch?v=YP8YlZkrQq8',
    'https://www.youtube.com/watch?v=j7_cjmJDYNU',
    'https://www.youtube.com/watch?v=yu9oyMXRGHY',
    'https://www.youtube.com/watch?v=5kS_a7vS5xI',
    'https://www.youtube.com/watch?v=uXs4fI3CeHE',
    'https://www.youtube.com/watch?v=NlDrJyQUTSI',
    'https://www.youtube.com/watch?v=y6QbtrTV-K0',
    'https://www.youtube.com/watch?v=-aOI9_JxoWc',
    'https://www.youtube.com/watch?v=32j2Tg64Zbg',
    'https://www.youtube.com/watch?v=yD6WKVqsAKc',
    'https://www.youtube.com/watch?v=8E98Gpk-fOM',
    'https://www.youtube.com/watch?v=li1sbr6S34g',
    'https://www.youtube.com/watch?v=ou5geWUE4Bw',
    'https://www.youtube.com/watch?v=XmJ-OdVFQtk',
    'https://www.youtube.com/watch?v=gJ_KHu0EC6I',
    'https://www.youtube.com/watch?v=vfzkc3lFTdM&list=PLA7ZcagI0frA0VuKY2ryd7C6OQUE5EAct&index=20',
    'https://www.youtube.com/watch?v=FZPrpoGdyHI',
    'https://www.youtube.com/watch?v=IDSr0z5f52k',
    'https://www.youtube.com/watch?v=D27aAZvuRTw',
    'https://www.youtube.com/watch?v=ROAnTfC_8zA',
    'https://www.youtube.com/watch?v=SYyvDrUgClc',
    'https://www.youtube.com/watch?v=xhUi2KpmVkI',
    'https://www.youtube.com/watch?v=O669aZhH0LI',
    'https://www.youtube.com/watch?v=eugfCRwSBJo',
    'https://www.youtube.com/watch?v=Y0tCJ6DWXKM',
    'https://www.youtube.com/watch?v=boQC4J4E1ZQ',
    'https://www.youtube.com/watch?v=8u_UHCnYSkk',
    'https://www.youtube.com/watch?v=G9r400zkkz8',
    'https://www.youtube.com/watch?v=maLGQ7fjCt4',
    'https://www.youtube.com/watch?v=yr2JQTdzNjY',
    'https://www.youtube.com/watch?v=rSK9Qx8LapE',
    'https://www.youtube.com/watch?v=Mawo3l3Hb9E',
    'https://www.youtube.com/watch?v=o51ingUOU20',
    'https://www.youtube.com/watch?v=8lHAsyRhYYQ',
    'https://www.youtube.com/watch?v=YmW83aQFADg',
    'https://www.youtube.com/watch?v=4e3JJ4rvT3Q',
    'https://www.youtube.com/watch?v=stGBwLEQB0Y', # not the mentioned match
    'https://www.youtube.com/watch?v=4rQUHv9oGpI',
    'https://www.youtube.com/watch?v=RRI_k2KZgOM',
    'https://www.youtube.com/watch?v=OzRtd3D0hEo',
    'https://www.youtube.com/watch?v=IuXmsimDOW8'
]

output_dir = os.path.join(configs['global']['project_root'], configs['global']['video_dir'])
os.makedirs(output_dir, exist_ok=True)

# Using a stable index (1 to N) prevents numbering drift when re-running the script
for i, link in enumerate(video_links, start=10):
    output_path = f'{output_dir}/match{i}.mp4'
    
    if os.path.exists(output_path):
        print(f"Skipping video {i} - already exists at {output_path}")
        continue

    ydl_opts = {
        'format': 'bestvideo[height<=720][ext=mp4][vcodec^=avc1]',
        'outtmpl': output_path,
        'noplaylist': True,
        'ignoreerrors': True,
        'socket_timeout': 30,
        'extractor_args': {
            'youtube': {
                'player_client': ['web'],  # Try web client first
                'player_skip': ['configs'],  # Skip some challenges
            }
        },
        'postprocessors': [{
            'key': 'FFmpegVideoRemuxer',
            'preferedformat': 'mp4',
        }],
    }
    
    print(f"\n--- Downloading video {i}: {link} ---")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([link])
    except Exception as e:
        print(f"Error downloading video {i}: {e}")

