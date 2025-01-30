from flask import Flask, request
import pandas as pd
import torch

app = Flask(__name__)

data = pd.read_csv('srivatsan-6.csv')

@app.route('/getfont', methods=['GET'])
def get_data():
    centerFont = request.args.get('center')  # Retrieve 'param1' from the URL
    magnitude = int(request.args.get('mag'))  # Retrieve 'param2' from the URL

    if centerFont in data['font'].values:
        datumRow = data[data['font'] == centerFont].iloc[:, 1:]
        datum = torch.tensor(datumRow.values)
        selectedFonts = []
        for i in range(datum.shape[1]):
            modTensor = torch.zeros_like(datum)
            print(modTensor.shape)
            modTensor[0, i] = 1 * magnitude
            newTensor = datum + modTensor
            minDistance = 9999
            minFont = None
            for otherFont in data['font'].values:
                otherDatum = torch.tensor(data[data['font'] == otherFont].iloc[:,1:].values)
                distance = torch.sqrt(torch.sum((newTensor - otherDatum) ** 2))
                if distance < minDistance and otherFont != centerFont and otherFont not in selectedFonts and str(otherFont) != 'nan':
                    print('help!')
                    minDistance = distance
                    minFont = otherFont
                    print(minFont)
            selectedFonts.append(str(minFont))
            print(selectedFonts)
        return f'Found these fonts: [{', '.join(selectedFonts)}]'

    else:
        return 'Font not found.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=18812)