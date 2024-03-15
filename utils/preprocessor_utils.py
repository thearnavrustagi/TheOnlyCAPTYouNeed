def load_mucs_transcription(fpath, *, lines=None):
    with open(fpath) as file:
        if not lines:
            lines = file.read().splitlines()
        return dict(tuple(line.strip().split(maxsplit=1)) for line in lines)


if __name__ == "__main__":
    print(
        load_mucs_transcription(
            "__init__.py",
            lines=["a key and value pair", "b of very cool keys", "c i want"],
        )
    )
