# NpyLabelNet: Классификация изображений для создания меток `.npy`

## Описание проекта

`NpyLabelNet` — это легковесная сверточная нейронная сеть (CNN), которая классифицирует RGB-изображения размером 512x512 пикселей на 100 классов и сохраняет предсказанные идентификаторы классов в файлы `.npy` (тип `np.int64`). Проект разработан для генерации меток для нейронной сети [TintoraAI](https://github.com/T1desLuck/TintoraAi), предназначенной для раскрашивания черно-белых или выцветших фотографий, но может использоваться и для других задач классификации изображений.

Репозиторий включает скрипты для обучения модели (`train.py`), инференса с созданием `.npy` меток (`auto_label.py`) и набор тестов (`tests/`). Проверка кода автоматизирована через GitHub Actions. Подробное руководство по подготовке датасета находится в [DATASET_GUIDE.md](DATASET_GUIDE.md).

### Основные возможности
- **Модель**: Простая CNN (`MiniCNN`) с тремя сверточными слоями, пулингом и полносвязными слоями, реализованная на PyTorch.
- **Входные данные**: RGB-изображения 512x512 пикселей в формате `.jpg` или `.png`.
- **Выходные данные**: Файлы `.npy`, содержащие один целочисленный идентификатор класса (0–99).
- **Классы**: Описаны в `classes.json` (100 категорий, например, "глаза", "дерево", "неопределённый_объект").
- **Интеграция с TintoraAI**: Генерирует `.npy` метки для обучения `TintoraAI`, требующей синхронизированных папок `bw/`, `color/` и `labels/`.
- **Прогресс**: Использует `tqdm` для отображения прогресс-баров при обучении и инференсе.
- **Чекпоинты**: Сохраняет веса модели каждые 5 эпох.
- **CI/CD**: Автоматическое тестирование (`pytest`) и проверка стиля кода (`flake8`) через GitHub Actions.

## Требования

- Python 3.9 или выше
- Зависимости (`requirements.txt`):
  - `torch>=1.13.0`
  - `torchvision>=0.14.0`
  - `numpy>=1.24.0`
  - `pillow>=9.0.0`
  - `tqdm>=4.65.0`
- Набор данных с изображениями 512x512 (см. [DATASET_GUIDE.md](DATASET_GUIDE.md))
- Файл `classes.json` с описанием 100 классов
- Обученные веса модели (по умолчанию: `models/tiny_cnn.pth`)
- Опционально: GPU с поддержкой CUDA для ускорения обучения и инференса

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
1. Создайте новый ноутбук в Google Colab.
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
4. (Опционально) Включите GPU: `Runtime > Change runtime type > GPU`.
5. Загрузите датасет и `classes.json` (например, через Google Drive).

## Использование

См. [DATASET_GUIDE.md](DATASET_GUIDE.md) для подготовки датасета.

### 1. Обучение модели
Запустите `train.py` для обучения `MiniCNN`:
```bash
python train.py --data_path /путь/к/данным --save_path models/tiny_cnn.pth --epochs 20 --batch_size 32
```
- `--data_path`: Папка с подпапками `глаза`, `рот`, ..., `неопределённый_объект`, содержащими изображения классов.
- `--save_path`: Путь для сохранения весов модели (по умолчанию: `models/tiny_cnn.pth`).
- `--epochs`: Количество эпох обучения (рекомендуется: 20+).
- `--batch_size`: Размер батча (рекомендуется: 32).

**В Colab**:
```python
from google.colab import drive
drive.mount('/content/drive')
!python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth --epochs 20 --batch_size 32
```

Обучение включает:
- Аугментации данных (случайное вращение, обрезка, изменение яркости/контраста).
- Прогресс-бары через `tqdm`.
- Сохранение чекпоинтов каждые 5 эпох (например, `models/checkpoint_epoch_5.pth`).

### 2. Генерация меток
Запустите `auto_label.py` для создания `.npy` меток:
```bash
python auto_label.py --input /путь/к/изображениям --output /путь/к/меткам --model_path models/tiny_cnn.pth --classes_path classes.json
```
- `--input`: Папка с изображениями 512x512 (например, `dataset/color/`).
- `--output`: Папка для `.npy` файлов (например, `dataset/labels/`).
- `--model_path`: Путь к весам модели.
- `--classes_path`: Путь к `classes.json`.

**В Colab**:
```python
!python auto_label.py --input /content/drive/MyDrive/dataset/color --output /content/drive/MyDrive/dataset/labels --model_path /content/drive/MyDrive/models/tiny_cnn.pth --classes_path /content/drive/MyDrive/classes.json
```

Поведение:
- Предсказывает класс с наибольшей вероятностью.
- Присваивает класс 99 ("неопределённый_объект") при уверенности ниже 0.5.
- Сохраняет `.npy` файлы с одним ID класса (например, `img_0001.npy` содержит `[0]` для "глаза").
- Пропускает изображения, если `.npy` файл уже существует.
- Логирует прогресс с `tqdm` и уверенностью предсказания.

### 3. Запуск тестов
**Локально**:
```bash
pytest tests/ -s
```

**В Colab**:
```python
!pytest tests/ -s
```

Тесты проверяют:
- Инициализацию модели (`MiniCNN`).
- Загрузку `classes.json`.
- Предобработку изображений.
- Корректность создания `.npy` файлов.

### 4. Проверка стиля кода
**Локально**:
```bash
flake8 *.py
```

**В Colab**:
```python
!flake8 *.py
```

Конфигурация `.flake8` задает максимальную длину строки 120 символов.

## Интеграция с TintoraAI

`NpyLabelNet` создает `.npy` метки для [TintoraAI](https://github.com/T1desLuck/TintoraAi), которая раскрашивает черно-белые или выцветшие фотографии. Подробности структуры датасета (`bw/`, `color/`, `labels/`) описаны в [DATASET_GUIDE.md](DATASET_GUIDE.md).

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
├── classes.json               # Описание 100 классов
├── requirements.txt           # Зависимости
├── train.py                   # Скрипт для обучения
├── README.md                  # Этот файл
└── DATASET_GUIDE.md           # Руководство по датасету
```

## Вклад в проект

1. Сделайте форк репозитория.
2. Создайте ветку: `git checkout -b имя-ветки`.
3. Зафиксируйте изменения: `git commit -m "Добавлена фича"`.
4. Отправьте ветку: `git push origin имя-ветки`.
5. Откройте пулл-реквест на [GitHub](https://github.com/T1desLuck/NpyLabelNet).

Убедитесь, что код проходит тесты (`pytest tests/`) и проверку `flake8` перед отправкой.

## Лицензия

Проект распространяется под лицензией MIT.
