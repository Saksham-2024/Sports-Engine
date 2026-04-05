import yt_dlp #type: ignore
import os

video_links = [
    "https://www.youtube.com/watch?v=5W6txLGZ1Rs&t=8s",
    'https://www.youtube.com/watch?v=YP8YlZkrQq8',
    'https://www.youtube.com/watch?v=y6QbtrTV-K0',
    'https://www.youtube.com/watch?v=-aOI9_JxoWc',
    'https://www.youtube.com/watch?v=32j2Tg64Zbg'
]
count = len(os.listdir('./unlabeled_videos'))

for i, link in enumerate(video_links, start=count+1):
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]',
        'outtmpl': f'unlabeled_videos/{i}.mp4',
        'noplaylist': True,
        'merge_output_format': 'mp4'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

