# File: OrpheusLabs/orpheuslabs_ai/worker.py
# Rewritten worker supporting both Sequence and GAN models based on project structure

import pika
import json
import uuid
import os
import logging
import configparser
import torch
import glob
import time
from datetime import datetime

# Logging configuration (consistent with main.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("MusicGenerationWorker")

# Import necessary functions and constants from the project
# Ensure these paths are correct relative to where the worker is run
try:
    from app.utils.config_handler import load_config, create_default_config
    from app.utils.model import (
        load_sequence_model, create_sequence_model,
        load_gan_models, create_gan_models
    )
    from app.utils.audio_converter import (
        sequence_to_midi, midi_to_wav, convert_to_mp3, enhance_midi_quality,
        pianoroll_tensor_to_midi
    )
    from app.generation.generator import generate_music, generate_music_gan
    # Constants for model parameters (assuming these are correctly located)
    from app.models.constants_sequence import SEQUENCE_MODEL_PARAMS
    from app.models.constants_gan import (
        GAN_TRAINING_PARAMS, PIANOROLL_PARAMS, GENERATOR_PARAMS,
        DISCRIMINATOR_PARAMS, GAN_GENERATION_DEFAULTS
    )
except ImportError as e:
    logger.error(f"Error importing necessary modules: {e}", exc_info=True)
    # Worker cannot run without these modules
    raise ImportError(f"Could not import necessary modules: {e}")


class MusicGenerationWorker:
    def __init__(self, config_path='app/config.ini'): # config.ini path relative to worker execution dir
        logger.info(f"Loading configuration: {config_path}")
        # Load config or create default if it doesn't exist
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Creating default.")
            try:
                # Ensure the 'app' directory exists if creating default there
                os.makedirs(os.path.dirname(config_path), exist_ok=True)
                create_default_config(config_path) # create_default_config handles writing
                self.config = load_config(config_path) # Load the newly created config
                logger.info(f"Default config created: {config_path}. Please review [PATHS], especially 'soundfont_path'.")
                if not self.config: # Check if loading after creation failed
                     raise ValueError(f"Failed to load the newly created configuration from {config_path}")

            except Exception as ce:
                logger.error(f"Failed to create default config: {ce}", exc_info=True)
                raise ValueError(f"Failed to create or load configuration from {config_path}") from ce
        else:
            self.config = load_config(config_path)
            if not self.config:
                raise ValueError(f"Failed to load configuration from {config_path}")

        # Read essential PATHS and Settings (with error handling)
        try:
            # --- Paths ---
            # Use 'models_save_dir' as the primary base for models unless specific paths are given
            self.base_models_dir = self.config.get('PATHS', 'models_save_dir', fallback='models')
            self.generation_output_dir = self.config.get('PATHS', 'generation_output_dir', fallback='generated')
            self.soundfont_path = self.config.get('PATHS', 'soundfont_path', fallback=None)
            # Specific load paths from config (optional, overrides finding latest)
            self.sequence_model_load_path_cfg = self.config.get('PATHS', 'model_load_path', fallback=None) # Reuse 'model_load_path'
            self.gan_model_load_path_cfg = self.config.get('PATHS', 'model_load_path', fallback=None) # Reuse 'model_load_path'

            # Create necessary directories
            os.makedirs(self.generation_output_dir, exist_ok=True)
            os.makedirs(self.base_models_dir, exist_ok=True) # Ensure base model dir exists

            # Log SoundFont status
            if not self.soundfont_path:
                logger.warning("Config '[PATHS] -> soundfont_path' not found. WAV/MP3 conversion will be skipped.")
            elif not os.path.exists(self.soundfont_path):
                logger.warning(f"SoundFont path '{self.soundfont_path}' in config not found. WAV/MP3 conversion might fail.")
            else:
                logger.info(f"Using SoundFont: {self.soundfont_path}")

        except (configparser.NoSectionError, configparser.NoOptionError) as e:
            logger.error(f"Error reading PATHS from config: {e}")
            raise ValueError(f"Error reading essential paths from config: {e}") from e

        self.loaded_models = {} # Cache loaded models (model_key -> model)
        self.determine_device() # Determine device ('mps', 'cuda', 'cpu')

        # RabbitMQ Connection
        # Prioritize environment variable, then config, then default
        rabbitmq_url = os.environ.get('RABBITMQ_URL')
        if not rabbitmq_url:
             try:
                 rabbitmq_url = self.config.get('RABBITMQ', 'url', fallback='amqp://morgar:password@localhost:5672/')
                 logger.info("Using RabbitMQ URL from config file.")
             except (configparser.NoSectionError, configparser.NoOptionError):
                 rabbitmq_url = 'amqp://morgar:password@localhost:5672/'
                 logger.warning("RabbitMQ URL not found in environment or config. Using default: amqp://morgar:password@localhost:5672/")
        else:
            logger.info("Using RabbitMQ URL from environment variable.")


        logger.info(f"Connecting to RabbitMQ: {rabbitmq_url}")
        try:
            # Increased connection timeout and heartbeat
            parameters = pika.URLParameters(rabbitmq_url)
            # parameters.heartbeat = 600 # Example: 10 minutes heartbeat
            # parameters.blocked_connection_timeout = 300 # Example: 5 minutes timeout

            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare the durable queue the worker consumes from
            self.channel.queue_declare(queue='music_requests', durable=True)
            # The reply queue is declared by the client (Go service)

            # Process one message at a time (Quality of Service)
            self.channel.basic_qos(prefetch_count=1)
            logger.info("RabbitMQ connection successful. 'music_requests' queue ready.")
        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}", exc_info=True)
            raise # Worker cannot start without connection

    def determine_device(self):
        """Determines and sets the PyTorch device (mps, cuda, cpu)."""
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            logger.info("Using Apple Silicon (MPS) device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA device.")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU device.")

    def find_latest_model(self, model_pattern):
        """Finds the most recently modified model file matching the pattern."""
        list_of_files = glob.glob(model_pattern)
        if not list_of_files:
            return None
        try:
            return max(list_of_files, key=os.path.getctime)
        except Exception as e:
            logger.error(f"Error finding latest model for pattern '{model_pattern}': {e}")
            return None

    def get_sequence_model(self, model_type):
        """Loads or retrieves the specified sequence model from cache."""
        cache_key = f"sequence_{model_type}"
        if cache_key in self.loaded_models:
            logger.debug(f"Returning cached sequence model: {model_type}")
            return self.loaded_models[cache_key]

        # Determine load path: Config > Latest
        load_path = self.sequence_model_load_path_cfg # Check specific path first
        model_file_pattern = os.path.join(self.base_models_dir, f'{model_type}_*.pt') # Pattern to find latest

        if load_path and os.path.exists(load_path):
             logger.info(f"Using sequence model path from config: {load_path}")
        else:
            if load_path: # Config path was specified but not found
                logger.warning(f"Sequence model path from config ('{load_path}') not found.")
            logger.info(f"Searching for latest sequence model in '{self.base_models_dir}' matching '{model_type}_*.pt'...")
            load_path = self.find_latest_model(model_file_pattern)
            if not load_path:
                 raise FileNotFoundError(f"No suitable model file found in '{self.base_models_dir}' for type '{model_type}'. Searched pattern: '{model_file_pattern}'")
            logger.info(f"Using latest modified sequence model: {load_path}")


        logger.info(f"Loading sequence model: {load_path} -> Device: {self.device}")
        try:
            # load_sequence_model returns model and the checkpoint dict
            model, checkpoint = load_sequence_model(load_path, device=self.device)
            # Verify model type from checkpoint if possible
            loaded_type = checkpoint.get('model_params', {}).get('model_type', model_type) # Assumes 'model_type' is saved in params
            if loaded_type != model_type:
                logger.warning(f"Requested model type '{model_type}' differs from loaded model type '{loaded_type}' from checkpoint.")

            self.loaded_models[cache_key] = model # Cache the model
            logger.info(f"Sequence model '{loaded_type}' loaded successfully.")
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found during load attempt: {load_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading sequence model ({load_path}): {e}", exc_info=True)
            raise

    def get_gan_generator(self):
        """Loads or retrieves the GAN Generator model from cache."""
        cache_key = "gan_generator"
        if cache_key in self.loaded_models:
            logger.debug("Returning cached GAN Generator.")
            return self.loaded_models[cache_key]

        # Determine load path: Config > Latest Checkpoint > Latest Final Generator
        load_path = self.gan_model_load_path_cfg # Check specific path first
        checkpoint_pattern = os.path.join(self.base_models_dir, 'gan_checkpoint_epoch_*.pt')
        final_g_pattern = os.path.join(self.base_models_dir, 'generator_final.pt')

        if load_path and os.path.exists(load_path):
             logger.info(f"Using GAN model path from config: {load_path}")
        else:
            if load_path: # Config path was specified but not found
                logger.warning(f"GAN model path from config ('{load_path}') not found.")

            logger.info(f"Searching for latest GAN checkpoint in '{self.base_models_dir}'...")
            load_path = self.find_latest_model(checkpoint_pattern)

            if not load_path:
                logger.warning(f"No GAN checkpoint found. Searching for final generator...")
                load_path = self.find_latest_model(final_g_pattern)
                if not load_path:
                    raise FileNotFoundError(f"No GAN checkpoint or final generator found in '{self.base_models_dir}'. Searched patterns: '{checkpoint_pattern}', '{final_g_pattern}'")
                logger.info(f"Using final GAN generator: {load_path}")
            else:
                 logger.info(f"Using latest modified GAN checkpoint: {load_path}")


        logger.info(f"Loading GAN Generator: {load_path} -> Device: {self.device}")
        try:
            # load_gan_models handles checkpoints, needs constants
            # If loading final .pt, it might just contain state_dict, handle this
            if 'final' in os.path.basename(load_path):
                 # Create a new model instance and load state dict
                 netG, _ = create_gan_models(GENERATOR_PARAMS, DISCRIMINATOR_PARAMS, PIANOROLL_PARAMS, GAN_TRAINING_PARAMS)
                 try:
                    state_dict = torch.load(load_path, map_location=self.device)
                    netG.load_state_dict(state_dict)
                    logger.info(f"Loaded final GAN Generator state dict successfully.")
                 except Exception as sd_err:
                     logger.error(f"Error loading state dict from final generator file {load_path}: {sd_err}", exc_info=True)
                     raise
                 netG.to(self.device)
                 checkpoint_data = {'epoch': 'final'} # Indicate it's the final model
            else:
                 # Load from checkpoint using the utility function
                 netG, _, checkpoint_data = load_gan_models(
                     load_path, GENERATOR_PARAMS, DISCRIMINATOR_PARAMS,
                     PIANOROLL_PARAMS, GAN_TRAINING_PARAMS, device=self.device
                 )
                 logger.info(f"GAN Generator loaded successfully (Epoch {checkpoint_data.get('epoch', 'Unknown')}).")

            self.loaded_models[cache_key] = netG # Cache the model
            return netG
        except FileNotFoundError:
            logger.error(f"GAN model/checkpoint file not found during load attempt: {load_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading GAN model ({load_path}): {e}", exc_info=True)
            raise


    def process_request(self, ch, method, properties, body):
        """Processes a single music generation request."""
        task_id = properties.correlation_id
        reply_to = properties.reply_to
        logger.info(f"Received request | TaskID: {task_id} | ReplyTo: {reply_to}")


        # Initialize response payload matching Go struct
        response_payload = {
            'task_id': task_id,
            'status': 'processing', # Start with processing status
            'message': 'Request received, starting processing.',
            'mp3_url': '',
            'midi_url': '', # URL for the enhanced/final MIDI
            'raw_midi_url': '', # URL for the initial raw MIDI output
            'wav_url': '',
            'model_used': '',
            'length_generated': 0,
        }

        try:
            start_time = time.time()
            request_data = json.loads(body)
            logger.debug(f"[TaskID: {task_id}] Request Body: {request_data}")

            # --- Extract Parameters ---
            params = request_data.get('params', {})

            # Determine Run Mode (Sequence or GAN)
            # Prefer 'model_type' from params, fallback to experiment mode from config
            requested_model_type = params.get('model_type', '').lower() # e.g., 'lstm', 'transformer', 'gan'
            config_mode = self.config.get('EXPERIMENT', 'mode', fallback='sequence').lower()
            config_seq_type = self.config.get('EXPERIMENT', 'sequence_model_type', fallback='transformer').lower()

            run_mode = 'gan' if requested_model_type == 'gan' else 'sequence'
            if run_mode == 'sequence' and not requested_model_type:
                # If no specific sequence type requested, use the default from config
                requested_model_type = config_seq_type
            elif run_mode == 'sequence' and requested_model_type not in SEQUENCE_MODEL_PARAMS:
                 logger.warning(f"[TaskID: {task_id}] Invalid sequence 'model_type' requested: '{requested_model_type}'. Falling back to config default '{config_seq_type}'.")
                 requested_model_type = config_seq_type

            response_payload['model_used'] = requested_model_type if run_mode == 'sequence' else 'gan'
            logger.info(f"[TaskID: {task_id}] Determined Run Mode: {run_mode.upper()} | Model Type: {response_payload['model_used'].upper()}")


            # --- Setup Output ---
            # Create a unique directory for this task's generated files
            task_output_dir = os.path.join(self.generation_output_dir, task_id)
            os.makedirs(task_output_dir, exist_ok=True)
            # Base URL path for response (relative to the static file server root)
            relative_output_dir_url = f"{os.path.basename(self.generation_output_dir)}/{task_id}" # e.g., "generated/task_id"
            # Base filename for generated files
            ts_now = datetime.now().strftime("%H%M%S_%d%m%Y")
            base_filename = f"{response_payload['model_used']}_{uuid.uuid4().hex[:6]}_{ts_now}"


            all_generated_midi_paths = [] # Keep track of generated MIDI files
            raw_midi_path = None

            # --- Music Generation ---
            if run_mode == 'sequence':
                # Get parameters for sequence generation
                model_type = requested_model_type
                start_sequence = params.get('start_sequence', [60, 64, 67]) # Default start
                length = params.get('length', self.config.getint('GENERATION', 'sequence_generation_length', fallback=250))
                temperature = params.get('temperature', self.config.getfloat('GENERATION', 'sequence_temperature', fallback=0.85))
                bpm = params.get('bpm', self.config.getint('MUSIC_FORMAT', 'bpm', fallback=120))
                note_duration = params.get('note_duration', self.config.getfloat('MUSIC_FORMAT', 'sequence_note_duration', fallback=0.4))
                instrument_program = params.get('instrument_program', self.config.getint('MUSIC_FORMAT', 'instrument_program', fallback=0))

                # Load the model
                model = self.get_sequence_model(model_type)

                # Generate the note sequence
                logger.info(f"[TaskID: {task_id}] Generating sequence (Model: {model_type}, Length: {length}, Temp: {temperature})...")
                gen_start_time = time.time()
                generated_seq = generate_music(
                    model=model,
                    model_type=model_type, # Pass type for potential internal logic
                    start_sequence=start_sequence,
                    length=length,
                    temperature=temperature,
                    device=self.device
                )
                gen_duration = time.time() - gen_start_time
                logger.info(f"[TaskID: {task_id}] Sequence generation complete ({gen_duration:.2f}s). Length: {len(generated_seq)}")
                response_payload['length_generated'] = len(generated_seq)

                # Save the raw MIDI output
                raw_midi_filename = f"{base_filename}_raw.mid"
                raw_midi_path = os.path.join(task_output_dir, raw_midi_filename)
                sequence_to_midi(generated_seq, raw_midi_path, bpm, note_duration, instrument_program)
                logger.info(f"[TaskID: {task_id}] Raw MIDI saved: {raw_midi_path}")
                response_payload['raw_midi_url'] = f"/{relative_output_dir_url}/{raw_midi_filename}"
                all_generated_midi_paths.append(raw_midi_path)

            elif run_mode == 'gan':
                # Get parameters for GAN generation
                num_samples = 1 # Worker currently generates one sample per request
                bpm = params.get('bpm', self.config.getint('MUSIC_FORMAT', 'bpm', fallback=120))
                instrument_program = params.get('instrument_program', self.config.getint('MUSIC_FORMAT', 'instrument_program', fallback=0))
                vel_threshold = params.get('gan_velocity_threshold', self.config.getfloat('MUSIC_FORMAT', 'gan_velocity_threshold', fallback=GAN_GENERATION_DEFAULTS['velocity_threshold']))

                # Load the GAN generator model
                generator_model = self.get_gan_generator()

                # Generate using GAN (returns list of MIDI paths)
                logger.info(f"[TaskID: {task_id}] Generating with GAN...")
                gen_start_time = time.time()
                gan_filename_pattern = f"{base_filename}"+"_{index}.mid" # Use base_filename
                gan_midi_paths = generate_music_gan(
                    generator=generator_model,
                    latent_dim=GAN_TRAINING_PARAMS['latent_dim'],
                    num_samples=num_samples,
                    device=self.device,
                    output_dir=task_output_dir, # Save to task-specific dir
                    filename_pattern=gan_filename_pattern,
                    # Parameters for pianoroll_tensor_to_midi
                    fs=PIANOROLL_PARAMS['fs'],
                    pitch_range=(PIANOROLL_PARAMS['pitch_min'], PIANOROLL_PARAMS['pitch_max']),
                    velocity_threshold=vel_threshold,
                    instrument_program=instrument_program,
                    bpm=bpm
                )
                gen_duration = time.time() - gen_start_time
                logger.info(f"[TaskID: {task_id}] GAN generation complete ({gen_duration:.2f}s). Files generated: {len(gan_midi_paths)}")

                if gan_midi_paths:
                    # Use the first generated MIDI as the raw output
                    raw_midi_path = gan_midi_paths[0]
                    raw_midi_filename = os.path.basename(raw_midi_path)
                    response_payload['raw_midi_url'] = f"/{relative_output_dir_url}/{raw_midi_filename}"
                    all_generated_midi_paths.append(raw_midi_path)
                    # Estimate length based on pianoroll constants
                    response_payload['length_generated'] = PIANOROLL_PARAMS.get('seq_length', 0)
                else:
                    raise ValueError("GAN model failed to generate any MIDI files.")
            else:
                 raise ValueError(f"Invalid run mode determined: {run_mode}")

            # --- Post-processing: Enhance and Convert ---
            if not raw_midi_path or not os.path.exists(raw_midi_path):
                 raise ValueError("No raw MIDI file was generated or found for post-processing.")

            # 1. Enhance MIDI Quality (Optional but recommended)
            raw_midi_filename = os.path.basename(raw_midi_path)
            # Create a distinct name for the enhanced file
            enhanced_midi_filename = raw_midi_filename.replace("_raw.mid", ".mid").replace(".mid", "_enhanced.mid")
            enhanced_midi_path = os.path.join(task_output_dir, enhanced_midi_filename)
            enhanced_midi_url = response_payload['raw_midi_url'] # Default to raw URL if enhancement fails

            logger.info(f"[TaskID: {task_id}] Enhancing MIDI quality: {raw_midi_path} -> {enhanced_midi_path}")
            if enhance_midi_quality(raw_midi_path, enhanced_midi_path, bpm, instrument_program):
                 logger.info(f"[TaskID: {task_id}] Enhanced MIDI saved: {enhanced_midi_path}")
                 enhanced_midi_url = f"/{relative_output_dir_url}/{enhanced_midi_filename}"
            else:
                 logger.warning(f"[TaskID: {task_id}] MIDI enhancement failed. Using raw MIDI URL for 'midi_url'.")
                 enhanced_midi_path = raw_midi_path # Use raw path for conversion if enhancement failed

            response_payload['midi_url'] = enhanced_midi_url # Set the final MIDI URL

            # 2. Convert to WAV (if SoundFont is available)
            wav_path = None
            wav_filename = os.path.splitext(os.path.basename(enhanced_midi_path))[0] + ".wav"
            wav_full_path = os.path.join(task_output_dir, wav_filename)
            if self.soundfont_path and os.path.exists(self.soundfont_path):
                logger.info(f"[TaskID: {task_id}] Converting to WAV: {enhanced_midi_path} -> {wav_full_path}")
                if midi_to_wav(enhanced_midi_path, wav_full_path, self.soundfont_path):
                    logger.info(f"[TaskID: {task_id}] WAV file saved: {wav_full_path}")
                    response_payload['wav_url'] = f"/{relative_output_dir_url}/{wav_filename}"
                    wav_path = wav_full_path # Set path for MP3 conversion
                else:
                    logger.error(f"[TaskID: {task_id}] MIDI to WAV conversion failed.")
            else:
                logger.warning(f"[TaskID: {task_id}] SoundFont not configured or found. Skipping WAV conversion.")

            # 3. Convert to MP3 (if WAV was created)
            mp3_path = None
            if wav_path and os.path.exists(wav_path):
                 mp3_filename = os.path.splitext(wav_filename)[0] + ".mp3"
                 mp3_full_path = os.path.join(task_output_dir, mp3_filename)
                 logger.info(f"[TaskID: {task_id}] Converting to MP3: {wav_path} -> {mp3_full_path}")
                 if convert_to_mp3(wav_path, mp3_full_path):
                     logger.info(f"[TaskID: {task_id}] MP3 file saved: {mp3_full_path}")
                     response_payload['mp3_url'] = f"/{relative_output_dir_url}/{mp3_filename}"
                     mp3_path = mp3_full_path
                 else:
                     logger.error(f"[TaskID: {task_id}] WAV to MP3 conversion failed.")
                 # Optionally remove the intermediate WAV file
                 try:
                     os.remove(wav_path)
                     logger.debug(f"[TaskID: {task_id}] Intermediate WAV file removed: {wav_path}")
                 except OSError as rm_err:
                     logger.warning(f"[TaskID: {task_id}] Could not remove intermediate WAV file {wav_path}: {rm_err}")
            elif self.soundfont_path: # SoundFont was configured, but WAV failed
                 logger.warning(f"[TaskID: {task_id}] Skipping MP3 conversion because WAV file was not created.")
            else: # SoundFont wasn't configured
                 logger.info(f"[TaskID: {task_id}] Skipping MP3 conversion as WAV was not generated.")

            # --- Finalize Response ---
            if response_payload['mp3_url'] or response_payload['midi_url']: # Success if at least MP3 or enhanced MIDI exists
                 response_payload['status'] = 'completed'
                 response_payload['message'] = 'Music generation and conversion successful.'
                 total_duration = time.time() - start_time
                 logger.info(f"[TaskID: {task_id}] Processing successful ({total_duration:.2f}s).")
            else:
                 # If we reached here but have no output URLs, something went wrong during conversion
                 raise RuntimeError("Generation complete, but failed to produce MP3 or Enhanced MIDI output.")


        except FileNotFoundError as e:
            logger.error(f"[TaskID: {task_id}] Processing Error - File Not Found: {e}", exc_info=True)
            response_payload['status'] = 'error'
            response_payload['message'] = f"A required file was not found: {e}"
        except ValueError as e: # Config, mode selection, or parameter errors
            logger.error(f"[TaskID: {task_id}] Processing Error - Value Error: {e}", exc_info=False)
            response_payload['status'] = 'error'
            response_payload['message'] = f"Error during processing: {e}"
        except RuntimeError as e: # Catch specific runtime errors like failed conversions
             logger.error(f"[TaskID: {task_id}] Processing Error - Runtime Error: {e}", exc_info=True)
             response_payload['status'] = 'error'
             response_payload['message'] = f"Runtime error during processing: {e}"
        except Exception as e: # Catch-all for unexpected errors
            logger.error(f"[TaskID: {task_id}] Unexpected Processing Error: {e}", exc_info=True)
            response_payload['status'] = 'error'
            response_payload['message'] = f"An unexpected server error occurred: {e}"

        # --- Send Response ---
        response_body_str = json.dumps(response_payload)
        if reply_to:
            try:
                ch.basic_publish(
                    exchange='',
                    routing_key=reply_to,
                    properties=pika.BasicProperties(
                        correlation_id=task_id,
                        content_type='application/json' # Set content type
                        # delivery_mode=2, # Make message persistent if reply queue is durable
                    ),
                    body=response_body_str
                )
                logger.info(f"[TaskID: {task_id}] Response sent successfully -> {reply_to}")
                logger.debug(f"[TaskID: {task_id}] Response Payload: {response_body_str}")
            except Exception as pub_err:
                logger.error(f"[TaskID: {task_id}] Failed to send response to {reply_to}: {pub_err}", exc_info=True)
        else:
            logger.warning(f"[TaskID: {task_id}] No 'reply_to' queue specified in request. Cannot send response.")

        # --- Acknowledge Message ---
        # Acknowledge the message regardless of success or failure,
        # as we've processed it (or attempted to).
        try:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.debug(f"[TaskID: {task_id}] Message acknowledged (delivery_tag: {method.delivery_tag}).")
        except Exception as ack_err:
            logger.error(f"[TaskID: {task_id}] Failed to acknowledge message: {ack_err}", exc_info=True)
            # This might indicate a channel/connection issue.


    def start_consuming(self):
        """Starts consuming messages from the 'music_requests' queue."""
        logger.info(' [*] Waiting for music generation requests. To exit press CTRL+C')
        # Register the consumer callback
        self.channel.basic_consume(
            queue='music_requests',
            on_message_callback=self.process_request
            # auto_ack is False by default when using basic_consume without it
        )
        try:
            # Start the blocking consumer loop
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down worker...")
        except pika.exceptions.ConnectionClosedByBroker:
            logger.error("Connection closed by broker. Check RabbitMQ server and credentials.")
        except pika.exceptions.AMQPChannelError as err:
            logger.error(f"Caught a channel error: {err}, stopping consumer.")
            self.channel.stop_consuming()
        except pika.exceptions.AMQPConnectionError:
            logger.error("Connection was closed, stopping consumer.")
        except Exception as e:
            logger.error(f"Unexpected error during consumption: {e}", exc_info=True)
        finally:
            # Ensure connection is closed cleanly
            self.stop_consuming()

    def stop_consuming(self):
        """Closes the RabbitMQ connection."""
        if self.connection and self.connection.is_open:
            logger.info("Closing RabbitMQ connection...")
            try:
                self.connection.close()
                logger.info("RabbitMQ connection closed.")
            except Exception as e:
                logger.error(f"Error closing RabbitMQ connection: {e}", exc_info=True)
        else:
            logger.info("RabbitMQ connection already closed or not established.")


if __name__ == "__main__":
    # Ensure the config file path is correct relative to the execution directory
    # Often, workers are run from the project root, so 'app/config.ini' might be correct
    config_file_path = 'app/config.ini'

    try:
        worker = MusicGenerationWorker(config_path=config_file_path)
        worker.start_consuming()
    except (ValueError, ImportError, configparser.Error) as init_err: # Initialization errors
        logger.critical(f"Worker failed to initialize (config/import error): {init_err}", exc_info=True)
    except pika.exceptions.AMQPConnectionError as conn_err: # RabbitMQ connection errors
        logger.critical(f"Worker failed to initialize (RabbitMQ connection error): {conn_err}", exc_info=False)
    except Exception as main_err: # Other unexpected errors during startup or runtime
        logger.critical(f"Worker encountered an unexpected critical error: {main_err}", exc_info=True)