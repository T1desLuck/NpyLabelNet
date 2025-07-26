# Руководство по созданию датасета для NpyLabelNet и TintoraAI

Это руководство описывает, как подготовить датасет для обучения и инференса `NpyLabelNet`, а также как организовать данные для совместимости с [TintoraAI](https://github.com/T1desLuck/TintoraAi), нейронной сетью для раскрашивания черно-белых или выцветших фотографий.

## Назначение NpyLabelNet

`NpyLabelNet` — это сверточная нейронная сеть (CNN), которая классифицирует RGB-изображения размером 512x512 на 1000 классов, описанных в `classes.json` (например, "трава_лужайка", "человек_лицо", "неопределённый_объект"). Она создает `.npy` файлы с идентификаторами классов (тип `np.int64`), которые используются для обучения `TintoraAI`. Проект также подходит для других задач классификации изображений.

### Что делает NpyLabelNet
- **Обучение**: Учится классифицировать изображения на основе набора данных с метками.
- **Инференс**: Предсказывает классы для новых изображений и сохраняет их ID в `.npy` файлы (например, `[150]` для "человек_лицо").
- **Выходные данные**: Каждый `.npy` файл содержит одно целое число (ID класса от 0 до 999).

### Как это работает
- Модель `MiniCNN` обрабатывает изображения через три сверточных слоя, пулинг и полносвязные слои.
- При инференсе выбирается класс с наибольшей вероятностью. Если уверенность ниже 0.5, присваивается класс 999 ("неопределённый_объект").
- Классы сопоставляются с именами в `classes.json`.

## Требования к датасету

### Для обучения NpyLabelNet
Датасет должен быть организован следующим образом:
- **Структура папок**:
  ```
  train_data/
  ├── 0/              # Класс 0 (например, "трава_лужайка")
  │   ├── img_0001.jpg
  │   ├── img_0002.png
  │   └── ...
  ├── 1/              # Класс 1 (например, "трава_сухая")
  │   ├── img_0003.jpg
  │   └── ...
  ├── ...
  └── 999/            # Класс 999 (например, "неопределённый_объект")
      ├── img_9999.jpg
      └── ...
  ```
- **Требования**:
  - Подпапки названы по ID классов (`0`, `1`, ..., `999`), соответствующим ключам в `classes.json`.
  - Изображения — RGB, 512x512 пикселей, в формате `.jpg` или `.png`.
  - Рекомендуется 5,000–10,000 изображений (5–10 на класс для 1000 классов).
  - Исключите поврежденные файлы или файлы не в формате изображений.

- **Советы**:
  - Используйте качественные и разнообразные изображения для повышения точности.
  - Убедитесь, что имена классов в `classes.json` (например, `"0": {"name": "трава_лужайка", "id": 0}`) соответствуют папкам.

### Для инференса NpyLabelNet
- **Структура папок**:
  ```
  dataset/color/
  ├── img_0001.jpg
  ├── img_0002.png
  └── ...
  ```
- **Требования**:
  - Изображения — RGB, 512x512, в формате `.jpg` или `.png`.
  - Все изображения в одной папке (без подпапок).
  - В выходной папке (например, `dataset/label/`) создаются `.npy` файлы с такими же базовыми именами (например, `img_0001.npy`).

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
  - Черно-белые изображения — это градации серого цветных изображений.
  - `.npy` файлы содержат одно целое число (ID класса).
  - Изображения — 512x512, в формате `.png` или `.jpg`.

## Как создать датасет

### Шаг 1: Сбор изображений
- Соберите или создайте RGB-изображения 512x512 для каждого класса из `classes.json`.
- Для `TintoraAI`:
  - Создайте цветные изображения для папки `color/`.
  - Сконвертируйте их в черно-белые для папки `bw/` (например, с помощью ImageMagick или Python с Pillow).

#### Локально
```bash
convert color/img_0001.jpg -colorspace Gray bw/img_0001.png
```

#### В Google Colab
```python
from PIL import Image
import os
os.makedirs('tintora_dataset/bw', exist_ok=True)
img = Image.open('tintora_dataset/color/img_0001.jpg').convert('L')
img.save('tintora_dataset/bw/img_0001.png')
```

### Шаг 2: Организация датасета для обучения
- Создайте папку (например, `train_data/`) с подпапками `0` до `999`.
- Разместите изображения в соответствующие папки классов.

#### Локально
```bash
mkdir -p train_data/0 train_data/1 ... train_data/999
mv grass_lawn.jpg train_data/0/
mv dry_grass.jpg train_data/1/
```

#### В Google Colab
```python
import os
for i in range(1000):
    os.makedirs(f'train_data/{i}', exist_ok=True)
!mv grass_lawn.jpg train_data/0/
!mv dry_grass.jpg train_data/1/
```

### Шаг 3: Генерация `.npy` меток
- Обучите `NpyLabelNet`:
  #### Локально
  ```bash
  python train.py --data_path train_data --save_path models/tiny_cnn.pth
  ```
  #### В Google Colab
  ```python
  !python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth
  ```

- Выполните инференс:
  #### Локально
  ```bash
  python auto_label.py --input tintora_dataset/color --output tintora_dataset/labels --model_path models/tiny_cnn.pth --classes_path classes.json
  ```
  #### В Google Colab
  ```python
  !python auto_label.py --input /content/drive/MyDrive/tintora_dataset/color --output /content/drive/MyDrive/tintora_dataset/labels --model_path /content/drive/MyDrive/models/tiny_cnn.pth --classes_path /content/drive/MyDrive/classes.json
  ```

- Это создаст `.npy` файлы в папке `tintora_dataset/labels/`.

### Шаг 4: Проверка датасета
- Убедитесь, что все изображения 512x512 и не повреждены.
- Проверьте, что имена файлов в `bw/`, `color/` и `labels/` совпадают.
- Убедитесь, что `classes.json` соответствует ID классов.

#### В Google Colab
- Для проверки размеров изображений:
  ```python
  from PIL import Image
  img = Image.open('/content/drive/MyDrive/tintora_dataset/color/img_0001.jpg')
  print(img.size)  # Должно быть (512, 512)
  ```

## Обучение NpyLabelNet

1. Подготовьте датасет, как описано выше.
2. Запустите скрипт обучения:
   #### Локально
   ```bash
   python train.py --data_path train_data --save_path models/tiny_cnn.pth --epochs 10 --batch_size 32
   ```
   #### В Google Colab
   ```python
   !python train.py --data_path /content/drive/MyDrive/train_data --save_path /content/drive/MyDrive/models/tiny_cnn.pth --epochs 10 --batch_size 32
   ```
3. Следите за выводом потерь, чтобы убедиться, что модель обучается.
4. Сохраните веса для инференса.

## Использование результатов с TintoraAI

- `.npy` файлы от `NpyLabelNet` содержат ID классов для `TintoraAI`.
- Убедитесь, что папка `labels/` содержит `.npy` файлы с именами, соответствующими `bw/` и `color/`.
- Следуйте документации `TintoraAI` для обучения с этими файлами.

## Решение проблем

- **Изображения не 512x512**:
  #### Локально
  ```bash
  convert image.jpg -resize 512x512! resized_image.jpg
  ```
  #### В Google Colab
  ```python
  from PIL import Image
  img = Image.open('image.jpg').resize((512, 512))
  img.save('resized_image.jpg')
  ```

- **Несоответствие классов**: Проверьте, что имена подпапок совпадают с ключами в `classes.json`.
- **Низкая точность**: Увеличьте количество эпох, добавьте больше изображений (5–10 на класс) или используйте более сложную модель (например, ResNet).
- **Ошибки CI/CD**: Проверьте ошибки `flake8`:
  #### Локально
  ```bash
  flake8 *.py
  ```
  #### В Google Colab
  ```python
  !flake8 *.py
  ```

## Дополнительные замечания
- **Производительность**: `MiniCNN` — простая модель, но для сложных данных может давать низкую точность. Попробуйте более глубокие архитектуры.
- **Размер датасета**: Больший датасет (5,000–10,000 изображений) улучшает обобщение.
- **TintoraAI**: Убедитесь, что черно-белые изображения в `bw/` точно соответствуют цветным для корректного обучения раскрашиванию.
- **Google Colab**: Используйте Google Drive для хранения датасета и весов, чтобы избежать повторной загрузки. Убедитесь, что GPU включен для ускорения.

Для помощи создайте issue в [репозитории NpyLabelNet](https://github.com/T1desLuck/NpyLabelNet).
