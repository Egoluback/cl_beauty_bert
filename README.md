# Curriculum Learning based on Loss Differences for text classification using ModernBERT
Эксперименты с Curriculum Learning в рамках классификации текстов для задачи классификации текстов.


**Датасеты:**
1. https://huggingface.co/datasets/stanfordnlp/imdb - Sentiment analysis dataset с бинарными метками
2. https://huggingface.co/datasets/wics/strategy-qa - Q/A dataset с бинарными ответами (true/false)


**Модель:** [ModernBERT-Base](https://huggingface.co/answerdotai/ModernBERT-base) (149m параметров)

**Как воспроизвести?**
1. Склонировать репозиторий
2. Установить среду 
<code>conda create --n your_env</code>
3. Установить необходимые библиотеки <code>pip3 install -r requirements</code>
4. Запустить интересующий эксперимент: все параметры меняются глобальными переменными в начале файла <code>cl_beauty_qa.py</code> или <code>cl_beauty_imdb.py</code>
