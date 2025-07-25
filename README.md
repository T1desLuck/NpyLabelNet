# NpyLabelNet
Мини-нейронка для автоматической разметки изображений.

## Описание
`NpyLabelNet` классифицирует изображения (512x512, `.jpg` или `.png`) и создаёт `.npy` файлы с метками `[class_id]` (0–999, np.int64). Совместима с проектами, ожидающими такой формат (например, `tintora_ai`).

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/your_npy_labelnet
   ```
2. Установите Python 3.9+.
3. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Использование
1. Подготовьте изображения (512x512) в папке (например, `dataset/color`).
2. Запустите классификацию:
   ```bash
   python auto_label.py --input dataset/color --output dataset/label
   ```
   - Пропускает изображения, если `.npy` уже существует.
   - Логирует: "Processed img_0001.jpg: class 150 (человек_лицо), confidence 0.85".

## Обучение
1. Подготовьте данные: 5,000–10,000 изображений (5–10 на класс × 1000 классов) в `train_data/класс/`.
2. Запустите обучение:
   ```bash
   python train.py --data_path train_data --save_path models/tiny_cnn.pth --epochs 10 --batch_size 32
   ```

## Требования
- Изображения: 512x512, `.jpg` или `.png`.
- Зависимости: См. `requirements.txt`.
- Модель: `models/tiny_cnn.pth` (обучите через `train.py`).
- Классы: `classes.json` (1000 классов).

## Лицензия
MIT