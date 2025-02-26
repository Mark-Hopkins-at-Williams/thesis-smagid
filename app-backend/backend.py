from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import faiss

Z_SIZE = 6
DATA_FILE = 'merged.csv'
TSNE_FILE = '/mnt/storage/smagid/thesis-smagid/tsne/data/tnse.csv'

app = Flask(__name__)
CORS(app)

# set up faiss
data = pd.read_csv(DATA_FILE)
fontNames = data["font"].values
# remove attribute data
labels = ["font", "z0","z1","z2","z3","z4","z5"]
data = data[labels]
zLabels = ["z0","z1","z2","z3","z4","z5"]
vectors = data[zLabels].values.astype("float32")
faissSearch = faiss.IndexFlatL2(Z_SIZE)
faissSearch.add(vectors)

@app.route('/getfont', methods=['GET'])
def get_font():
    centerFont = request.args.get('center')
    magnitude = float(request.args.get('mag'))
    if centerFont is not None:
        if centerFont in data['font'].values:
            datumRow = data[data['font'] == centerFont].iloc[:, 1:]
            datum = torch.tensor(datumRow.values)
        else:
            return
    else:
        # select random center and magnitude 1
        datumRow = data.sample()
        centerFont = datumRow['font'].values[0]
        print(centerFont)
        datum = torch.tensor(datumRow.iloc[:, 1:].values)
    response = dict()
    response["centerFont"] = centerFont
    selectedFonts = []
    for i in range(datum.shape[1]):
        modTensor = torch.zeros_like(datum)
        modTensor[0, i] = 1 * magnitude
        newTensor = datum + modTensor
        queryVector = newTensor.numpy().astype("float32")
        print(queryVector)
        _, indices = faissSearch.search(queryVector, 10)
        selectedFont = None
        i = 0
        # try to avoid duplicates and same fonts
        # print(indices)
        while not selectedFont:
            index = indices[0][i]
            font = fontNames[index]
            # print(font)
            if font != centerFont and font not in selectedFonts:
                selectedFont = font
            if i == 9 and selectedFont is None: # last resort just choose the first one
                print('resorting :(')
                selectedFont = fontNames[indices[0][0]]
            i += 1
        selectedFonts.append(selectedFont)
    response["selectedFonts"] = selectedFonts
    return jsonify(response)

@app.route('/gettsne', methods=['GET'])
def get_tsne():
    df = pd.read_csv(TSNE_FILE)
    data = df.to_dict(orient="records")
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=18812)