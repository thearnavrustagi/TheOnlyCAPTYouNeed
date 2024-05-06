import json

if __name__ == "__main__":
    getlines = lambda x: open(x).read().splitlines()
    script = getlines("./script.txt")
    misp_script = getlines("./misp_script.txt")

    e_errs = []

    for correct, incorrect in zip(script, misp_script):
        line_errs = []
        for corr_word, incorrect_word in zip(correct.split(), incorrect.split()):
            line_errs.append(int(corr_word != incorrect_word))
        e_errs.append(line_errs)
    json.dump({"e_err": e_errs}, open("./e_err.json", "w"))
