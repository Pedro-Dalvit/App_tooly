import torch
from flask import Flask, request, jsonify
import os
import io
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)

# Ruta al archivo JSON de las credenciales de la cuenta de servicio
SERVICE_ACCOUNT_FILE = 'handy-ia-b007002fae0e.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def create_drive_service():
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('drive', 'v3', credentials=credentials)

def download_file(file_id, destination):
    drive_service = create_drive_service()
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")
    with open(destination, 'wb') as f:
        f.write(fh.getvalue())

def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        print(content)


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


modelo_entrenado_file_id = '1dfmrhgnOUPTlJucbuEMuPSTxbJBuFhUQ'
tokenizer_file_id = '1cQ2lfSfQEOReUJRVzbFVjfV9y0MYkXBB'
modelo_entrenado_destination_path = 'modelo_entrenado.pth'
tokenizer_destination_path = 'tokenizer.pth'

download_file(modelo_entrenado_file_id, modelo_entrenado_destination_path)
download_file(tokenizer_file_id, tokenizer_destination_path)
loaded_model, loaded_tokenizer = load_model_and_tokenizer()

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Hacer predicción utilizando el modelo y tokenizer cargados
    predictions = get_prediction(text, loaded_model, loaded_tokenizer)

    return jsonify(predictions)

@app.route('/')
def hello():
    return "Hello world!"

# Punto de entrada para la aplicación
if __name__ == '__main__':
    # Configurar para que la aplicación se ejecute en el puerto proporcionado por Heroku o en el puerto 5000 de manera local
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)