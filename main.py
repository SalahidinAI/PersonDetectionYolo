import cv2
from ultralytics import YOLO
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('test.mp4')

if not cap.isOpened():
    print('Camera not found')
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

# Переменные для рисования прямоугольника
drawing = False
start_point = None
end_point = None
rect_defined = False


def put_russian_text(img, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """Функция для отображения русского текста"""
    try:
        # Пытаемся загрузить системный шрифт
        font = ImageFont.truetype("arial.ttf", int(30 * font_scale))
    except:
        # Если не удалось, используем шрифт по умолчанию
        font = ImageFont.load_default()

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color[::-1])  # BGR -> RGB
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_result


def mouse_callback(event, x, y, flags, param):
    """Callback для рисования прямоугольника мышью"""
    global drawing, start_point, end_point, rect_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)
        rect_defined = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        rect_defined = True
        print(f'Rectangle drawn: from {start_point} to {end_point}')


def is_point_in_rect(point, rect_start, rect_end):
    """Проверка, находится ли точка внутри прямоугольника"""
    x, y = point
    x1, y1 = rect_start
    x2, y2 = rect_end

    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    return x_min <= x <= x_max and y_min <= y <= y_max


def get_person_bottom_center(box):
    """Получить центральную точку внизу bounding box (ноги человека)"""
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    bottom_y = y2
    return (center_x, bottom_y)


# Получаем первый кадр для настройки зоны
ret, first_frame = cap.read()
if not ret:
    print('Failed to read video')
    exit()

setup_frame = first_frame.copy()
cv2.namedWindow('Setup: Draw restricted zone')
cv2.setMouseCallback('Setup: Draw restricted zone', mouse_callback)

print('=== INSTRUCTIONS ===')
print('1. Hold left mouse button and draw rectangle')
print('2. Release button when finished')
print('3. Press ENTER to start detection')
print('4. Press R to redraw')
print('5. Press Q to exit')
print('==================')

# Режим настройки зоны
setup_mode = True
while setup_mode:
    display_frame = setup_frame.copy()

    # Рисуем прямоугольник
    if start_point and end_point:
        overlay = display_frame.copy()
        cv2.rectangle(overlay, start_point, end_point, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
        cv2.rectangle(display_frame, start_point, end_point, (0, 0, 255), 3)

        x1, y1 = start_point
        x2, y2 = end_point
        cv2.putText(display_frame, f'({x1},{y1})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display_frame, f'({x2},{y2})', (x2, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Инструкции на английском
    cv2.rectangle(display_frame, (5, 5), (450, 100), (0, 0, 0), -1)
    cv2.putText(display_frame, 'Draw rectangle with mouse',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if rect_defined:
        cv2.putText(display_frame, 'Press ENTER to start',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, 'Drawing...',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(display_frame, 'R - reset | Q - exit',
                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Setup: Draw restricted zone', display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()
    elif key == ord('r'):
        start_point = None
        end_point = None
        rect_defined = False
        print('Rectangle reset')
    elif key == 13 and rect_defined:
        setup_mode = False
        cv2.destroyWindow('Setup: Draw restricted zone')
        print(f'Restricted zone set!')
        print(f'Coordinates: {start_point} -> {end_point}')

# Возвращаем видео в начало
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Основной цикл детекции
print('Starting detection...')
while True:
    fps_start = time.time()
    ret, frame = cap.read()

    if not ret:
        print('Video ended, restarting...')
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Детекция объектов
    result = model(frame, conf=0.3)
    boxes = result[0].boxes

    person_count = 0
    persons_in_zone = 0

    # Рисуем красную запретную зону
    overlay = frame.copy()
    cv2.rectangle(overlay, start_point, end_point, (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.rectangle(frame, start_point, end_point, (0, 0, 255), 3)

    # Текст на зоне (на английском)
    center_x = (start_point[0] + end_point[0]) // 2
    center_y = (start_point[1] + end_point[1]) // 2
    cv2.putText(frame, ' ' , (center_x - 100, center_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Обработка людей - показываем ТОЛЬКО тех, кто в красной зоне
    for n in boxes:
        cls = int(n.cls[0])
        label = model.names[cls]
        conf = round(float(n.conf[0]), 2)

        if label == 'person':
            x1, y1, x2, y2 = map(int, n.xyxy[0])

            person_point = get_person_bottom_center(n.xyxy[0])
            in_zone = is_point_in_rect(person_point, start_point, end_point)

            # Показываем ТОЛЬКО людей в красной зоне
            if in_zone:
                person_count += 1
                persons_in_zone += 1

                # Красный прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(frame, f'ALERT! {conf * 100:.0f}%',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)

                # Мигающий восклицательный знак
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(frame, '!!!', (x2 + 5, y1 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # Точка отслеживания
                cv2.circle(frame, person_point, 6, (0, 0, 255), -1)

    fps_end = time.time()
    fps = 1 / (fps_end - fps_start)

    # Панель информации (на английском)
    panel_height = 120
    cv2.rectangle(frame, (5, 5), (320, panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (5, 5), (320, panel_height), (255, 255, 255), 2)

    cv2.putText(frame, f'FPS: {round(fps, 1)}', (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f'People in zone: {person_count}', (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if person_count > 0:
        cv2.putText(frame, 'VIOLATION!', (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, 'All clear', (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Person Detection - Restricted Zone', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Program finished')
