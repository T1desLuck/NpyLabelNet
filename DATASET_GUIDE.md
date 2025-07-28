# Руководство по созданию датасета для NpyLabelNet и TintoraAI

Это руководство описывает, как подготовить датасет для обучения и инференса `NpyLabelNet`, а также для интеграции с [TintoraAI](https://github.com/T1desLuck/TintoraAi), нейронной сетью для раскрашивания черно-белых или выцветших фотографий. Проект использует только собственные изображения и модели, без сторонних данных или весов.

## Назначение NpyLabelNet

`NpyLabelNet` — это сверточная нейронная сеть (CNN), которая классифицирует RGB-изображения размером 512x512 на 100 классов, описанных в `classes.json` (например, "глаза", "дерево", "неопределённый_объект"). Она создает `.npy` файлы с идентификаторами классов (тип `np.int64`) для обучения `TintoraAI` или других задач классификации.

### Что делает NpyLabelNet
- **Обучение**: Учится классифицировать изображения на основе датасета с метками.
- **Инференс**: Предсказывает классы для новых изображений и сохраняет ID в `.npy` файлы (например, `[0]` для "глаза").
- **Выходные данные**: Каждый `.npy` файл содержит одно целое число (ID класса, 0–99).

### Как это работает
- Модель `MiniCNN` обрабатывает изображения через три сверточных слоя, пулинг и полносвязные слои.
- При инференсе выбирается класс с наибольшей вероятностью. Если уверенность ниже 0.5, присваивается класс 99 ("неопределённый_объект").
- Классы сопоставляются с именами в `classes.json`.

## Требования к датасету

### Для обучения NpyLabelNet
- **Структура папок**:
  ```
  train_data/
  ├── глаза/          # Класс 0
  │   ├── img_0001.jpg
  │   ├── img_0002.png
  │   └── ...
  ├── рот/            # Класс 1
  │   ├── img_0003.jpg
  │   └── ...
  ├── ...
  └── неопределённый_объект/  # Класс 99
      ├── img_9999.jpg
      └── ...
  ```
- **Требования**:
  - Подпапки названы по ключам из `classes.json` (например, "глаза", "рот", ..., "неопределённый_объект").
  - Изображения: RGB, 512x512, `.jpg` или `.png`.
  - Рекомендуется: 5,000–10,000 изображений (50–100 на класс для 100 классов).
  - Используйте только собственные изображения (например, личные фото или созданные вами), исключая поврежденные файлы.

- **Советы**:
  - Используйте разнообразные и качественные изображения для повышения точности.
  - Убедитесь, что имена папок совпадают с ключами в `classes.json`.

### Для инференса NpyLabelNet
- **Структура папок**:
  ```
  dataset/color/
  ├── img_0001.jpg
  ├── img_0002.png
  └── ...
  ```
- **Требования**:
  - Изображения: RGB, 512x512, `.jpg` или `.png`.
  - Все изображения в одной папке (без подпапок).
  - В выходной папке (например, `dataset/labels/`) создаются `.npy` файлы с такими же именами (например, `img_0001.npy`).

### Для интеграции с TintoraAI
`TintoraAI` требует три синхронизированных компонента:
- **Черно-белые изображения** (`bw/`): Градации серого.
- **Цветные изображения** (`color/`): Оригинальные цветные версии.
- **Метки** (`labels/`): `.npy` файлы с ID классов от `NpyLabelNet`.

- **Структура папок**:
  ```
  tintora_dataset/
  ├── bw/
  │   ├── img_0001.png
  │   ├── img_0002.png
  │   └── ...
  ├── color/
  │   ├── img_0001.png
  │   ├── img_0002.png
  │   └── ...
  └── labels/
      ├── img_0001.npy
      ├── img_0002.npy
      └── ...
  ```
- **Требования**:
  - Имена файлов в `bw/`, `color/` и `labels/` должны совпадать (например, `img_0001.png` и `img_0001.npy`).
  - Черно-белые изображения — градации серого цветных изображений.
  - `.npy` файлы содержат одно целое число (ID класса).
  - Изображения: 512x512, `.png` или `.jpg`.

## Как создать датасет

### Шаг 1: Сбор изображений
- Соберите или создайте RGB-изображения 512x512 для каждого класса из `classes.json`. Используйте только собственные изображения.
- Для `TintoraAI`:
  - Подготовьте цветные изображения для `color/`.
  - Сконвертируйте их в черно-белые для `bw/` (например, с помощью ImageMagick или Pillow).

**Локально**:
```bash
convert color/img_0001.jpg -colorspace Gray bw/img_0001.png
```

**В Colab**:
```python
from PIL import Image
import os
os.makedirs('tintora_dataset/bw', exist_ok=True)
img = Image.open('tintora_dataset/color/img_0001.jpg').convert('L')
img.save('tintora_dataset/bw/img_0001.png')
```

### Шаг 2: Организация датасета для обучения
- Создайте папку `train_data/` с подпапками, названными по классам (`глаза`, `рот`, ..., `неопределённый_объект`).
- Разместите изображения в соответствующие папки.

**Локально**:
```bash
mkdir -p train_data/глаза train_data/рот train_data/неопределённый_объект
mv eyes.jpg train_data/глаза/
mv tree.jpg train_data/дерево/
```

**В Colab**:
```python
import os
for cls in ['глаза', 'рот', 'дерево', 'неопределённый_объект']:
    os.makedirs(f'train_data/{cls}', exist_ok=True)
!mv eyes.jpg train_data/глаза/
!mv tree.jpg train_data/дерево/
```

### Шаг 3: Генерация `.npy` меток
- Обучите `NpyLabelNet`:
  **Локально**:
  ```bash
  python train.py --data_path train_data --save_path models/tiny_cnn.pth --epochs 20
  ```
  **В Colab**:
  ```python
  !python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth --epochs 20
  ```

- Выполните инференс:
  **Локально**:
  ```bash
  python auto_label.py --input tintora_dataset/color --output tintora_dataset/labels --model_path models/tiny_cnn.pth --classes_path classes.json
  ```
  **В Colab**:
  ```python
  !python auto_label.py --input /content/drive/MyDrive/tintora_dataset/color --output /content/drive/MyDrive/tintora_dataset/labels --model_path /content/drive/MyDrive/models/tiny_cnn.pth --classes_path /content/drive/MyDrive/classes.json
  ```

Это создаст `.npy` файлы в `tintora_dataset/labels/`.

### Шаг 4: Проверка датасета
- Убедитесь, что все изображения 512x512 и не повреждены.
- Проверьте, что имена файлов в `bw/`, `color/` и `labels/` совпадают.
- Убедитесь, что `classes.json` соответствует ID классов.

**В Colab**:
```python
from PIL import Image
img = Image.open('/content/drive/MyDrive/tintora_dataset/color/img_0001.jpg')
print(img.size)  # Должно быть (512, 512)
```

## Обучение NpyLabelNet

1. Подготовьте датасет, как описано выше.
2. Запустите обучение:
   **Локально**:
   ```bash
   python train.py --data_path train_data --save_path models/tiny_cnn.pth --epochs 20 --batch_size 32
   ```
   **В Colab**:
   ```python
   !python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth --epochs 20 --batch_size 32
   ```
3. Следите за потерями в выводе, чтобы убедиться в прогрессе обучения.
4. Сохраняйте веса для инференса (чекпоинты сохраняются каждые 5 эпох).

## Использование результатов с TintoraAI

- `.npy` файлы от `NpyLabelNet` содержат ID классов для `TintoraAI`.
- Убедитесь, что папка `labels/` содержит `.npy` файлы с именами, соответствующими `bw/` и `color/`.
- Следуйте документации `TintoraAI` для обучения.

## Решение проблем

- **Изображения не 512x512**:
  **Локально**:
  ```bash
  convert image.jpg -resize 512x512! resized_image.jpg
  ```
  **В Colab**:
  ```python
  from PIL import Image
  img = Image.open('image.jpg').resize((512, 512))
  img.save('resized_image.jpg')
  ```

- **Несоответствие классов**: Проверьте, что имена папок совпадают с ключами в `classes.json`.
- **Низкая точность**: Увеличьте количество эпох, добавьте больше изображений (50–100 на класс) или настройте `MiniCNN` (например, добавьте сверточные слои).
- **Ошибки CI/CD**: Проверьте ошибки `flake8`:
  **Локально**:
  ```bash
  flake8 *.py
  ```
  **В Colab**:
  ```python
  !flake8 *.py
  ```

## Дополнительные замечания
- **Производительность**: `MiniCNN` — простая модель, но для сложных данных может потребовать углубления архитектуры.
- **Размер датасета**: Больший датасет (5,000–10,000 изображений) улучшает обобщение.
- **TintoraAI**: Убедитесь, что черно-белые изображения в `bw/` соответствуют цветным в `color/`.
- **Google Colab**: Используйте Google Drive для хранения датасета и весов. Включите GPU для ускорения.

Для помощи создайте issue в [репозитории NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet).