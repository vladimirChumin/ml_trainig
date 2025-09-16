import csv

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import logging
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_name):
    """Загрузка модели и токенизатора с обработкой ошибок"""
    try:
        logger.info(f"Загрузка модели {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используется устройство: {device}")
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        return tokenizer, model, device
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        raise

def zero_shot_classify(title, description, site_profile, tokenizer, model, device):
    try:
        prompt = (
            f"Определи, является ли сайт компанией, которая занимается {site_profile}.\n"
            f"Заголовок: {title}\n"
            f"Описание: {description}\n"
            "Ответь только 'да' или 'нет':"
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():  # Отключаем вычисление градиентов для экономии памяти
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Получаем ответ после prompt и анализируем его более тщательно
        result = answer[len(prompt):].strip().lower()

        # Более надежная проверка ответа
        if "да" in result and "нет" not in result:
            return "да"
        elif "нет" in result:
            return "нет"
        else:
            logger.warning(f"Неоднозначный ответ: '{result}'. Возвращаем 'нет'")
            return "нет"
    except Exception as e:
        logger.error(f"Ошибка при классификации: {e}")
        return "ошибка"

def batch_process(df, tokenizer, model, device, site_profile, batch_size=100):
    """Обработка данных батчами для лучшей производительности"""
    results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Обработка батчей"):
        batch = df.iloc[i:i+batch_size]
        batch_results = []

        for _, row in batch.iterrows():
            result = zero_shot_classify(
                row['title'],
                row['description'],
                site_profile,
                tokenizer,
                model,
                device
            )
            batch_results.append(result)

        results.extend(batch_results)

    return results

def main():
    try:
        # Загрузка данных
        logger.info("Загрузка данных...")
        df = pd.read_csv("filtered_output.csv", sep="|")
        logger.info(f"Загружено {df.shape[0]} записей")

        # Параметры классификации
        site_profile = "Оптовый продавец или дистрибьютор обуви"
        model_name = "sberbank-ai/rugpt3small_based_on_gpt2"

        # Загрузка модели
        tokenizer, model, device = load_model(model_name)

        # Классификация
        logger.info("Начало классификации...")
        df['classification'] = batch_process(df, tokenizer, model, device, site_profile)

        # Сохранение результатов
        df.to_csv("classified_output.csv", sep="|", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

        # Статистика результатов
        yes_count = (df['classification'] == 'да').sum()
        no_count = (df['classification'] == 'нет').sum()
        error_count = (df['classification'] == 'ошибка').sum()

        logger.info(f"Классификация завершена: {yes_count} - да, {no_count} - нет, {error_count} - ошибки")
        logger.info(f"Результаты сохранены в 'classified_output.csv'")

    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")

if __name__ == "__main__":
    main()