import os
import hashlib
import numpy as np
from PIL import Image
import logging
import time


def _generate_hash(filename, logger):
    """Verilen dosyanın SHA256 hash'ini oluşturur."""
    hash_func = hashlib.sha256()
    try:
        with open(filename, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.digest()
    except FileNotFoundError:
        logger.error(f"Hash oluşturulacak dosya bulunamadı: {filename}")
        return None
    except Exception as e:
        logger.error(f"Hash oluşturulurken hata: {filename}, Hata: {e}", exc_info=True)
        return None

def _create_image_from_hash(file_hash, logger, final_size=(512, 512), upscale_factor=4):
    """Verilen hash değerinden deterministik bir görsel oluşturur."""
    try:
        seed = int.from_bytes(file_hash, "big") % (2**32)
        rng = np.random.default_rng(seed)

        high_size = (final_size[0] * upscale_factor, final_size[1] * upscale_factor)
        x = np.linspace(0, 2 * np.pi, high_size[0])
        y = np.linspace(0, 2 * np.pi, high_size[1])
        X, Y = np.meshgrid(x, y)

        pattern = (
            np.sin(X * rng.uniform(0.5, 2.0) + rng.uniform(0, 2*np.pi)) +
            np.sin(Y * rng.uniform(0.5, 2.0) + rng.uniform(0, 2*np.pi)) +
            np.sin((X + Y) * rng.uniform(0.5, 2.0) + rng.uniform(0, 2*np.pi)) +
            np.sin((X - Y) * rng.uniform(0.5, 2.0) + rng.uniform(0, 2*np.pi))
        )

        pattern_min = pattern.min()
        pattern_max = pattern.max()
        if pattern_max == pattern_min:
            pattern = np.zeros_like(pattern)
        else:
            pattern = (pattern - pattern_min) / (pattern_max - pattern_min)

        mask = pattern > 0.5
        color1 = np.array([0xF5, 0x5B, 0x47], dtype=np.uint8)  # #F55B47
        color2 = np.array([0xCC, 0xF2, 0x4E], dtype=np.uint8)  # #CCF24E
        img_array = np.where(mask[..., None], color1, color2)

        img = Image.fromarray(img_array)
        img = img.resize(final_size, resample=Image.LANCZOS)
        return img

    except Exception as e:
        logger.error(f"Hash'ten görsel oluşturulurken hata: {e}", exc_info=True)
        return None

def generate_and_save_image_from_file(
    input_filepath,
    output_dir,
    base_filename, # Uzantısız dosya adı (örn: gan_5_191236_27_04_2025)
    relative_url_base, # URL için temel yol (örn: generated/task_id)
    task_id, # Loglama için
    logger # Worker'dan gelen logger objesi
    ):
    """
    Verilen dosyadan (MP3) hash oluşturur, bu hash'ten bir görsel yaratır,
    PNG olarak kaydeder ve göreceli URL'sini döndürür.

    Args:
        input_filepath (str): Görselin oluşturulacağı kaynak dosyanın (örn: MP3) yolu.
        output_dir (str): Görselin kaydedileceği dizin.
        base_filename (str): Kaydedilecek görsel için uzantısız temel dosya adı.
        relative_url_base (str): Sunucudaki statik dosya yoluna göreli temel URL.
        task_id (str): Log mesajlarında kullanılacak görev ID'si.
        logger (logging.Logger): Loglama için kullanılacak logger nesnesi.

    Returns:
        str or None: Başarılı olursa görselin göreceli URL'si (örn: /generated/task_id/dosya.png),
                     başarısız olursa None.
    """
    logger.info(f"[TaskID: {task_id}] Generating image from file: {input_filepath}")
    start_time = time.time()

    file_hash = _generate_hash(input_filepath, logger)
    if file_hash is None:
        logger.error(f"[TaskID: {task_id}] Could not generate hash for {input_filepath}. Skipping image generation.")
        return None

    generated_image = _create_image_from_hash(file_hash, logger)
    if generated_image is None:
        logger.error(f"[TaskID: {task_id}] Image generation function returned None for file: {input_filepath}")
        return None

    try:
        image_filename = f"{base_filename}.png" # Uzantıyı ekle
        image_full_path = os.path.join(output_dir, image_filename)
        generated_image.save(image_full_path, "PNG")
        duration = time.time() - start_time
        logger.info(f"[TaskID: {task_id}] Image saved: {image_full_path} ({duration:.2f}s)")
        image_url = f"/{relative_url_base}/{image_filename}" # Başına / ekleyerek göreceli URL oluştur
        return image_url

    except Exception as e:
        logger.error(f"[TaskID: {task_id}] Failed to save generated image {image_filename}: {e}", exc_info=True)
        return None