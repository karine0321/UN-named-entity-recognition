import json

with open("outfile-pretty.json", "r") as f:
    contents = json.load(f)

data = []

for t in contents:
    word_dict = t[0]
    predictions = {
        "me_pred": t[1]["me_pred"][1],
        "dt_pred": t[1]["dt_pred"][1],
        "nb_pred": t[1]["nb_pred"][1],
    }

    data.append([word_dict, predictions])

with open('outfile2.json', 'w') as f:
    json.dump(data, f, indent = 2)