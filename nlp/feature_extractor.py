"""
Feature Extraction Pipeline:
1. Запрос критериев для протоколов
2. NLP/LLM для извлечения признаков из текста
3. Извлечение сырых данных (таблицы, анализы)
4. Построение недостающих признаков ML-моделями
"""

class FeatureExtractor:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base

    def extract_features(self, patient_raw, protocol_ids):
        # TODO: запрос требований к признакам
        # TODO: NLP + LLM
        # TODO: Structured features
        # TODO: ML/derived features
        return {}
