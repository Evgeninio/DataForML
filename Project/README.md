
# SMS Spam — Dagster Pipeline

Учебный проект, демонстрирующий полный цикл работы с данными с помощью Dagster.

## 🔧 Что делает проект
1. **Сбор данных**: скачивание датасета UCI (SMS Spam Collection) 
2. **Разметка/препроцессинг**: подготовка признаков TF‑IDF, расчёт метрик неопределённости/энтропии для базовой модели.
3. **Очистка**: удаление дубликатов, простая фильтрация «аномалий» по длине.
4. **Активное обучение**: стратегия *uncertainty sampling* (modAL) — выбор наиболее неопределённых примеров для разметки и дообучения.

## 🚀 Запуск локально
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

dagster dev 

```

## 📂 Структура
```
sms_spam_dagster/
├─ README.md
├─ repository.py
├─ requirements.txt
├─ workspace.yaml
├─ assets/
│  ├─ __init__.py
│  ├─ collect_data.py
│  ├─ preprocess_and_entropy.py
│  ├─ clean_data.py
│  ├─ active_learning.py
│  └─ train_model.py
├─ jobs/
│  ├─ __init__.py
│  ├─ data_collection_job.py
│  └─ active_learning_job.py
├─ data/ (создаётся во время выполнения)
│  ├─ raw/
│  └─ processed/
└─ artifacts/ (создаётся во время выполнения)
```
