import json
import numpy as np


def handle_results(results, file_path="metrics.json"):
    """Загружает вопросы из JSON-файла."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            metrics = json.load(file)
            answer = ''
            for result in results:
                current_metric = metrics.get(str(result[0]))
                answer += current_metric + f' Вероятность: {result[1]:.3f}' + '\n'
            return answer
    except FileNotFoundError:
        print("Файл не найден!")
        return []
    except json.JSONDecodeError:
        print("Ошибка декодирования JSON!")
        return []


if __name__ == "__main__":
    print(handle_results([[0, 10], [2, 9]]))