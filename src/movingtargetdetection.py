import cv2
import numpy as np

frame_count = 0
DETECT_INTERVAL = 5  # her 15 karede bir yeniden detect & re‐init

cap = cv2.VideoCapture("videos/MTD5.mp4")   # video file will be here
if not cap.isOpened():
    print("Video is not opened.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("outputs/mtd_output.avi", fourcc, fps, (width, height))


fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)

ret, prev_frame = cap.read()
if not ret:
    print("Video açılamadı.")
    exit()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Döngüden önce, cap açıldıktan sonra:
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
motionHistory = np.zeros((h, w), dtype=np.float32)
mh_duration = 1.0  # Geçmişi 1 saniye tut


def merge_nearby_boxes(boxes, distance_threshold=100):
    """Yakın bounding boxları birleştir - gelişmiş versiyon"""
    if len(boxes) == 0:
        return []

    # Recursive merging için
    def merge_two_boxes(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        min_x = min(x1, x2)
        min_y = min(y1, y2)
        max_x = max(x1 + w1, x2 + w2)
        max_y = max(y1 + h1, y2 + h2)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    # Tüm boxları birbirleriyle kontrol et
    merged = list(boxes)
    changed = True

    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            current_box = merged[i]
            x1, y1, w1, h1 = current_box
            used[i] = True

            # Bu box ile birleştirilecek diğer boxları bul
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue

                x2, y2, w2, h2 = merged[j]

                # Mesafe kontrolü
                cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
                cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
                distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                # Veya overlap kontrolü
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y

                if distance < distance_threshold or overlap_area > 0:
                    current_box = merge_two_boxes(current_box, merged[j])
                    x1, y1, w1, h1 = current_box  # Güncelle
                    used[j] = True
                    changed = True

            new_merged.append(current_box)

        merged = new_merged

    return merged


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)

    # ─── Optical Flow Filtre ───
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_mask = (mag > 1.5).astype('uint8') * 255  # Threshold'u artır
    prev_gray = gray.copy()
    fgmask = cv2.bitwise_and(fgmask, flow_mask)

    # Orijinal morfolojik işlemler + ek bağlantı
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    # Daha büyük kernel ile bağlantı kur
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_large)

    fgmask = cv2.medianBlur(fgmask, 5)

    h, w = fgmask.shape
    m = 30
    fgmask[:m, :] = 0
    fgmask[-m:, :] = 0
    fgmask[:, :m] = 0
    fgmask[:, -m:] = 0

    # ───────── Motion History Image ile birleştirme ─────────
    # 1) Zaman damgası
    timestamp = cv2.getTickCount() / cv2.getTickFrequency()

    # 2) Motion History güncellemesi
    motionHistory[fgmask > 0] = timestamp

    # 3) Geçerli silüet: son mh_duration saniyede hareket eden pikseller
    silh = ((timestamp - motionHistory) <= mh_duration).astype('uint8') * 255

    # 4) Morfolojik temizleme - daha agresif bağlantı
    k_mhi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    silh = cv2.morphologyEx(silh, cv2.MORPH_CLOSE, k_mhi)
    silh = cv2.morphologyEx(silh, cv2.MORPH_OPEN, k_mhi)

    # Ek dilation ile parçaları birleştir
    k_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    silh = cv2.morphologyEx(silh, cv2.MORPH_CLOSE, k_connect)

    # 5) Kontur çıkar ve tekil kutuları çiz
    cnts, _ = cv2.findContours(silh,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 800:  # Threshold'u biraz artır
            continue

        x, y, w_box, h_box = cv2.boundingRect(cnt)

        # Aspect ratio filtresi - daha sıkı
        aspect_ratio = w_box / float(h_box)
        if aspect_ratio > 6 or aspect_ratio < 0.15:
            continue

        detections.append((x, y, w_box, h_box))

    # ─── Sadece yakın boxları birleştir ───
    detections = merge_nearby_boxes(detections, distance_threshold=150)

    # ─── Sonuçları çiz ───
    for (x, y, w_box, h_box) in detections:
        cv2.rectangle(frame,
                      (x, y),
                      (x + w_box, y + h_box),
                      (0, 255, 0), 2)

        # Area bilgisi
        area = w_box * h_box
        cv2.putText(frame, f"Area: {area}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Debug görüntüleri (isteğe bağlı)
    # cv2.imshow("Silhouette", silh)
    # cv2.imshow("FG Mask", fgmask)

    # Son: göster ve kaydet
    out.write(frame)
    cv2.imshow("MTD Combined", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()