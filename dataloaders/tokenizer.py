def tokenize(arg):
    arg = arg.replace(" ", "")
    tokenized_line = []

    for char in arg:
        if 0x0900 <= ord(char) and ord(char) <= 0x0D7F:
            tokenized_line.append(ord(char) - 0x0900)

    return tokenized_line
