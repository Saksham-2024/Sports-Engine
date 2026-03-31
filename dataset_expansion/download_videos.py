import yt_dlp #type: ignore
import os

video_links = [
    "https://www.youtube.com/watch?v=qvBL6GVZR4c&list=PLA7ZcagI0frCRsdX5bIGfKU2c1xldBImV&index=3",
    "https://www.youtube.com/watch?v=3BAg0Za8yCE&list=PLA7ZcagI0frCRsdX5bIGfKU2c1xldBImV&index=9",
    "https://www.youtube.com/watch?v=tgHjFhBHWfE&list=PLA7ZcagI0frCRsdX5bIGfKU2c1xldBImV&index=15",
    "https://www.youtube.com/watch?v=L-J5Dz6TmIE&list=PLA7ZcagI0frCRsdX5bIGfKU2c1xldBImV&index=17",
    "https://www.youtube.com/watch?v=byTOesv1980&list=PLA7ZcagI0frCRsdX5bIGfKU2c1xldBImV&index=24",
    "https://www.youtube.com/watch?v=8PxSHZoOOSI&list=PLA7ZcagI0frDqjYVgX120NXz0Vq94Ga3Q&index=14",
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

