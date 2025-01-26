from flask import Flask, request
import torch
import enum

app = Flask(__name__)

data = torch.load('srivatsan-6.pt')

@app.route('/getfont', methods=['GET'])
def get_data():
    centerFont = request.args.get('center')  # Retrieve 'param1' from the URL
    magnitude = int(request.args.get('mag'))  # Retrieve 'param2' from the URL

    if centerFont in data.keys():
        datum = data[centerFont]
        selectedFonts = []
        for i in range(len(datum)):
            modTensor = torch.zeros_like(datum)
            modTensor[i] = 1 * magnitude
            newTensor = datum + modTensor
            minDistance = 9999
            minFont = None
            for otherFont in data.keys():
                otherDatum = data[otherFont]
                distance = torch.sqrt(torch.sum((newTensor - otherDatum) ** 2))
                if distance < minDistance and otherFont != centerFont and otherFont not in selectedFonts:
                    minDistance = distance
                    minFont = otherFont
            selectedFonts.append(minFont)
        return f'Found these fonts: {selectedFonts}'

    else:
        return 'Font not found.'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=18812)