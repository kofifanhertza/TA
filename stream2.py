import os
def get_youtube_stream_url(youtube_url):
    # Fetch the best quality stream URL
    stream_url = os.popen(f"youtube-dl -g {youtube_url}").read().strip()
    return stream_url


print(get_youtube_stream_url("https://www.youtube.com/live/g1tb--rpfVE?si=f_qSuE-C54H7Xxbz"))
