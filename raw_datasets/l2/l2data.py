import os
import pandas as pd
from pytube import YouTube


def download_audio(video_url, output_path):
    try:
        yt = YouTube(video_url)
        yt_stream = yt.streams.filter(only_audio=True).first()
        audio_path = yt_stream.download(output_path=output_path, filename="temp_audio")
        return audio_path
    except Exception as e:
        print(f"Error downloading audio from {video_url}: {e}")
        return None


if __name__ == "__main__":
    df = pd.read_csv("L2_hindi_speech.csv")
    ctr = 0
    for index, row in df.iterrows():
        video_url = row["url"]
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])

        audio_path = download_audio(video_url, "temp")
        if audio_path:
            try:
                ctr += 1
                file = f"L2_data/{ctr}.wav"
                os.system(
                    f"ffmpeg -i {audio_path} -ss {start_time} -to {end_time} -c:a pcm_s16le -ar 44100 {file}"
                )
                print(f"Audio conversion completed successfully for {file}")
            except Exception as e:
                print(f"Error converting audio: {e}")
            finally:
                os.remove(audio_path)
