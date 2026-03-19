import os
import tempfile
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Import your unmodified inference functions
from whisper_model import transcribe_whisper
from hubert_model import transcribe_hubert

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # 1. Validate the file upload
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided in the request'}), 400
        
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
        
    # 2. Extract and validate the model choice (defaults to whisper)
    model_choice = request.form.get('model', 'whisper').lower()
    if model_choice not in ['whisper', 'hubert']:
        return jsonify({'error': 'Invalid model choice. Must be "whisper" or "hubert"'}), 400

    # 3. Save the audio file temporarily
    filename = secure_filename(audio_file.filename)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"upload_{filename}")
    
    try:
        audio_file.save(temp_path)
        
        # 4. Route to the correct unmodified inference function
        if model_choice == 'whisper':
            text = transcribe_whisper(temp_path)
        else:
            text = transcribe_hubert(temp_path)
            
        # 5. Return the transcription in JSON
        return jsonify({
            'status': 'success',
            'model': model_choice,
            'transcription': text
        }), 200
        
    except Exception as e:
        # Catch any errors gracefully
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
        
    finally:
        # 6. Clean up the temporary audio file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Use dynamic port for deployment (defaults to 5000)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
