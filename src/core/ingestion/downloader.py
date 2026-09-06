import os
import yt_dlp
from pathlib import Path
from src.core.utils.configs import PathResolver

class Downloader:
    def __init__(self, urls):
        self.resolver = PathResolver()
        self.video_dir = self.resolver.get_path("global.video_dir")
        if not os.path.exists(self.video_dir):
            os.makedirs(self.video_dir, exist_ok=True)
        self.urls = urls
    
    def download(self):
        for i, link in enumerate(self.urls, start=1):
            match_name = f'match{i}.mp4'
            match_path = Path(self.video_dir) / match_name
            if match_path.exists():
                print(f"Skipping video {i} - already exists at {match_path}")
                continue

            ydl_opts = {
                'format': 'bestvideo[height<=720][ext=mp4][vcodec^=avc1]',
                'outtmpl': match_path,
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
                    ydl.download([str(link)])
            except Exception as e:
                print(f"Error downloading video {i}: {e}")




            

        
