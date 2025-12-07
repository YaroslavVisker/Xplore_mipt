"""
Оценка соответствия пациента критериям протокола:
 - выполнение DSL правил
 - агрегирование
 - финальный verdict: eligible / not eligible
"""

class ScoringEngine:
    def evaluate(self, features, protocol):
        # TODO: DSL evaluator
        # TODO: агрегирование
        return {
            "status": "eligible",
            "score": 95,
            "report": {}
        }
