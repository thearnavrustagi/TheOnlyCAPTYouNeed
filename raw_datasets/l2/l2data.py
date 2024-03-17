import os
import pandas as pd
from pytube import YouTube
import moviepy.editor as mp


def convert_to_audio(video_url, start_time, end_time, output_file):
    try:
        yt = YouTube(video_url)
        yt_stream = yt.streams.filter(only_audio=True).first()
        audio_path = yt_stream.download(output_path="temp", filename="temp_audio")

        temp_file = audio_path.replace(".webm", ".mp3")
        output_file = output_file.replace(".mp3", ".wav")
        os.rename(temp_file, output_file)

        audio = mp.AudioFileClip(output_file)
        trimmed_audio = audio.subclip(start_time, end_time)
        trimmed_audio.write_audiofile(output_file.replace(".mp3", ".wav"), fps=44100)
        print(f"Audio conversion completed successfully for {output_file}")

    except Exception as e:
        print(f"An error occurred for {output_file}: {e}")


if __name__ == "__main__":
    df = pd.read_csv("L2 hindi speech.csv")
    for index, row in df.iterrows():
        video_url = row["url"]
        start_time = float(row["start_time"])
        end_time = float(row["end_time"])
        output_file = f"L2_data/{index}.wav"
        convert_to_audio(video_url, start_time, end_time, output_file)
