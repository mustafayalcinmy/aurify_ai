import pika
import json
import uuid
import os
from main import generate_music, sequence_to_midi, enhance_midi_quality, midi_to_wav, convert_to_mp3

class MusicGenerationWorker:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost'))
        self.channel = self.connection.channel()
        
        self.channel.queue_declare(queue='music_requests')
        self.channel.queue_declare(queue='music_responses')
    
    def process_request(self, ch, method, properties, body):
        task_id = None
        try:
            request = json.loads(body)
            task_id = request['task_id']
            params = request['params']
            
            # Müzik oluşturma işlemi
            generated_seq = generate_music(
                model=params.get('model', None),  # Model referansını uygun şekilde alın
                model_type=params.get('model_type', 'transformer'),
                start_sequence=params['start_sequence'],
                length=params.get('length', 100),
                temperature=params.get('temperature', 1.0)
            )
            
            # Dosya oluşturma
            output_dir = f"generated/{task_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            midi_path = os.path.join(output_dir, "output.mid")
            sequence_to_midi(generated_seq, midi_path)
            
            # Diğer format dönüşümleri
            enhanced_path = os.path.join(output_dir, "enhanced.mid")
            wav_path = os.path.join(output_dir, "output.wav")
            mp3_path = os.path.join(output_dir, "output.mp3")
            
            enhance_midi_quality(midi_path, enhanced_path)
            midi_to_wav(enhanced_path, wav_path, params['soundfont_path'])
            convert_to_mp3(wav_path, mp3_path)
            
            # Response hazırla
            response = {
                'task_id': task_id,
                'status': 'completed',
                'download_url': f"/generated/{task_id}/output.mp3",
                'midi_url': f"/generated/{task_id}/enhanced.mid"
            }
            
            # Sonucu gönder
            self.channel.basic_publish(
                exchange='',
                routing_key='music_responses',
                body=json.dumps(response)
            )
            
        except Exception as e:
            error_response = {
                'task_id': task_id if task_id else 'unknown',
                'status': 'error',
                'message': str(e)
            }
            self.channel.basic_publish(
                exchange='',
                routing_key='music_responses',
                body=json.dumps(error_response)
            )

    def start_consuming(self):
        self.channel.basic_consume(
            queue='music_requests',
            on_message_callback=self.process_request,
            auto_ack=True)
        self.channel.start_consuming()


if __name__ == "__main__":
    worker = MusicGenerationWorker()
    worker.start_consuming()