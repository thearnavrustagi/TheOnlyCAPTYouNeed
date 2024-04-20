from feedback.hindi_sounds_feedback import sound_info


def pronounce_hindi_sound(sound):
    if sound in sound_info:
        return sound_info[sound]
    else:
        return f"No record for '{sound}'."


if __name__ == "__main__":
    sound = "ज"
    print(pronounce_hindi_sound(sound))
