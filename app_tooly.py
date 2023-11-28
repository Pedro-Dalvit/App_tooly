import torch
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Cargar modelo y tokenizer
def load_model_and_tokenizer(model_filename='modelo_entrenado.pth', tokenizer_filename='tokenizer.pth'):
    directorio_guardado = os.path.dirname(os.path.realpath(__file__))  # Obtener el directorio del script actual
    ruta_modelo = os.path.join(directorio_guardado, model_filename)
    ruta_tokenizer = os.path.join(directorio_guardado, tokenizer_filename)

    # Cargar el modelo y el tokenizer con torch.load
    model = torch.load(ruta_modelo, map_location=torch.device('cpu'))
    tokenizer = torch.load(ruta_tokenizer, map_location=torch.device('cpu'))
    return model,tokenizer

# Definir la función de predicción
def get_prediction(text, model, tokenizer):
    id2str = {0: 'Amoladora', 1: 'Escalera', 2: 'Hidrolavadora', 3: 'Martillo', 4: 'Rodillo', 5: 'Serrucho',
              6: 'Soldadora', 7: 'Taladro', 8: 'Destornillador', 9: 'Lijadora', 10: 'Bordeadora', 11: 'Sierra',
              12: 'Minitorno'}

    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt').to("cpu")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    probs = probs[0].tolist()

    # Obtener los tres mayores resultados
    top_3_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]

    # Obtener nombres de herramientas para los tres mayores resultados
    top_3_herramientas = [id2str[index] for index in top_3_indices]

    return top_3_herramientas

    #return [{'prob': prob * 100, 'herramienta': id2str[probs.index(prob)]} for prob in probs]

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Hacer predicción utilizando el modelo y tokenizer cargados
    predictions = get_prediction(text, loaded_model, loaded_tokenizer)

    return jsonify(predictions)

if __name__ == '__main__':
    # Cargar modelo y tokenizer al iniciar la aplicación
    loaded_model, loaded_tokenizer = load_model_and_tokenizer()

    # Iniciar la aplicación Flask
    app.run(port=int(os.environ.get('PORT', 5000)), debug=False)