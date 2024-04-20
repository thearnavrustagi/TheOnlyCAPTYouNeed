# Open the file in read mode
with open(
    "../raw_datasets/mucs/train/transcription.txt", "r", encoding="utf-8"
) as file:
    length = []
    for line_num, line in enumerate(file, 1):
        num_chars = len(line.strip())
        length.append(num_chars)
    print(max(length))
