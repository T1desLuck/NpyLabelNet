# NpyLabelNet: Классификация изображений для создания меток `.npy`

## Описание проекта

`NpyLabelNet` — это легковесная сверточная нейронная сеть (CNN), которая классифицирует RGB-изображения размером 512x512 пикселей на 1000 классов и сохраняет предсказанные идентификаторы классов в файлы `.npy` (тип `np.int64`). Проект создан для генерации меток для нейронной сети [TintoraAI](https://github.com/T1desLuck/TintoraAi), предназначенной для раскрашивания черно-белых или выцветших фотографий, но может использоваться и для других задач классификации изображений.

Репозиторий включает скрипты для обучения модели (`train.py`), инференса с созданием `.npy` меток (`auto_label.py`) и набор тестов (`tests/`). Автоматизация проверки кода реализована через GitHub Actions. Подробное руководство по подготовке датасета находится в [DATASET_GUIDE.md](DATASET_GUIDE.md).

### Основные возможности
- **Модель**: Простая CNN (`MiniCNN`) с тремя сверточными слоями, пулингом и полносвязными слоями, реализованная на PyTorch.
- **Входные данные**: RGB-изображения 512x512 пикселей в формате `.jpg` или `.png`.
- **Выходные данные**: Файлы `.npy`, содержащие один целочисленный идентификатор класса (от 0 до 999).
- **Классы**: Описаны в `classes.json` (1000 категорий, например, "трава_лужайка", "человек_лицо", "неопределённый_объект").
- **Интеграция с TintoraAI**: Генерирует `.npy` метки для обучения TintoraAI, требующей синхронизированных папок `bw/`, `color/` и меток.
- **CI/CD**: Автоматическое тестирование и линтинг с помощью GitHub Actions.

## Требования

Для работы с проектом нужны:
- Python 3.9 или выше.
- Зависимости из `requirements.txt`:
  - `torch>=1.13.0`
  - `torchvision>=0.14.0`
  - `numpy>=1.24.0`
  - `pillow>=9.0.0`
- Набор данных с изображениями 512x512 (подробности в [DATASET_GUIDE.md](DATASET_GUIDE.md)).
- Файл `classes.json` с описанием 1000 классов.
- Обученные веса модели (по умолчанию: `models/tiny_cnn.pth`).
- Опционально: GPU с поддержкой CUDA для ускорения обучения и инференса (автоматически доступно в Google Colab с GPU).

## Установка

### Локальная установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/T1desLuck/NpyLabelNet.git
   cd NpyLabelNet
   ```

2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. (Опционально) Установите инструменты для тестирования:
   ```bash
   pip install pytest flake8
   ```

### Установка в Google Colab
1. Откройте Google Colab и создайте новый ноутбук.
2. Клонируйте репозиторий:
   ```python
   !git clone https://github.com/T1desLuck/NpyLabelNet.git
   %cd NpyLabelNet
   ```
3. Установите зависимости:
   ```python
   !pip install -r requirements.txt
   !pip install pytest flake8
   ```
4. (Опционально) Включите GPU: в меню Colab выберите `Runtime > Change runtime type > GPU`.
5. Загрузите датасет и `classes.json` в Colab (например, через `Files > Upload` или Google Drive).

## Использование

Для подготовки датасета см. [DATASET_GUIDE.md](DATASET_GUIDE.md). Ниже описаны основные шаги работы с моделью.

### 1. Обучение модели
#### Локально
Для обучения модели `MiniCNN` выполните:
```bash
python train.py --data_path /путь/к/данным --save_path models/tiny_cnn.pth --epochs 10 --batch_size 32
```
- `--data_path`: Папка с подпапками, названными `0`, `1`, ..., `999`, содержащими изображения соответствующих классов.
- `--save_path`: Путь для сохранения весов модели (по умолчанию: `models/tiny_cnn.pth`).
- `--epochs`: Количество эпох обучения (рекомендуется: 10 и выше).
- `--batch_size`: Размер батча (рекомендуется: 32).

#### В Google Colab
1. Загрузите датасет в Colab (например, через Google Drive):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Запустите обучение:
   ```python
   !python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth --epochs 10 --batch_size 32
   ```
- Убедитесь, что датасет находится в `/content/drive/MyDrive/train_data`.
- Для хорошей точности используйте 5,000–10,000 изображений (5–10 на класс).

Скрипт применяет аугментации (случайное вращение, обрезка, изменение яркости и контраста) и сохраняет веса модели.

### 2. Генерация меток
#### Локально
Для создания `.npy` меток выполните:
```bash
python auto_label.py --input /путь/к/изображениям --output /путь/к/меткам --model_path models/tiny_cnn.pth --classes_path classes.json
```
- `--input`: Папка с изображениями 512x512 (`.jpg` или `.png`, например, `dataset/color`).
- `--output`: Папка для сохранения `.npy` файлов (например, `dataset/label`).
- `--model_path`: Путь к весам модели.
- `--classes_path`: Путь к файлу `classes.json`.

#### В Google Colab
1. Загрузите изображения и `classes.json` в Colab (например, через Google Drive).
2. Выполните инференс:
   ```python
   !python auto_label.py --input /content/drive/MyDrive/dataset/color --output /content/drive/MyDrive/dataset/label --model_path /content/drive/MyDrive/models/tiny_cnn.pth --classes_path /content/drive/MyDrive/classes.json
   ```

Для каждого изображения:
- Предсказывается класс с наибольшей вероятностью.
- Если уверенность ниже 0.5, присваивается класс 999 ("неопределённый_объект").
- Сохраняется `.npy` файл с идентификатором класса (например, `img_0001.npy` содержит `[150]` для "человек_лицо").
- Логируются результаты, например: `Processed img_0001.jpg: class 150 (человек_лицо), confidence 0.85`.
- Скрипт пропускает изображения, если `.npy` файл уже существует.

### 3. Запуск тестов
#### Локально
Для проверки проекта:
```bash
pytest tests/ -s
```

#### В Google Colab
```python
!pytest tests/ -s
```

Тесты проверяют:
- Инициализацию модели.
- Загрузку классов из `classes.json`.
- Предобработку изображений.
- Корректность сохранения `.npy` файлов.

### 4. Проверка кода
#### Локально
Для проверки стиля кода:
```bash
flake8 *.py
```

#### В Google Colab
```python
!flake8 *.py
```

Конфигурация `.flake8` задает максимальную длину строки 120 символов.

## Интеграция с TintoraAI

`NpyLabelNet` разработан для поддержки [TintoraAI](https://github.com/T1desLuck/TintoraAi), которая раскрашивает черно-белые или выцветшие фотографии. **NpyLabelNet** создает `.npy` метки, необходимые для обучения **TintoraAI**. Подробности подготовки датасета, включая структуру папок `bw/`, `color/` и `labels/`, описаны в [DATASET_GUIDE.md](DATASET_GUIDE.md).

## Структура проекта

```
NpyLabelNet/
├── .github/workflows/ci.yml    # Настройка CI/CD
├── tests/                      # Тесты
│   ├── __init__.py
│   └── test_model.py
├── .flake8                    # Конфигурация Flake8
├── __init__.py
├── auto_label.py              # Скрипт для инференса
├── classes.json               # Описание 1000 классов
├── requirements.txt           # Зависимости
├── train.py                   # Скрипт для обучения
├── README.md                  # Этот файл
└── DATASET_GUIDE.md           # Руководство по созданию датасета
```

## Вклад в проект

1. Сделайте форк репозитория.
2. Создайте ветку для изменений: `git checkout -b имя-ветки`.
3. Зафиксируйте изменения: `git commit -m "Добавлена фича"`.
4. Отправьте ветку: `git push origin имя-ветки`.
5. Откройте пулл-реквест на [GitHub](https://github.com/T1desLuck/NpyLabelNet).

Убедитесь, что код проходит тесты (`pytest tests/`) и проверку `flake8` перед отправкой.

## Лицензия

Проект распространяется под лицензией MIT.