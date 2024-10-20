import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import smtplib
from email.message import EmailMessage
import os

def send_email_with_attachment(subject, body, to, attachment_path):
    # SMTP server details
    smtp_server = 'smtp.office365.com'
    smtp_port = 587
    sender_email = 'mail'
    sender_password = 'sifre'

    
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to
    msg.set_content(body)

    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        file_name = os.path.basename(attachment_path)
    msg.add_attachment(file_data, maintype='image', subtype='png', filename=file_name)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


subject = "Yetkisiz Giriş Tespiti"
body = "İlk yetkisiz giriş ekran görüntüsü ekte bulunmaktadır."
to = "mail"
attachment_path = "first_unauthorized_entry.png"



class_names = {
    0: 'human', 1: 'blue', 2: 'yellow',
}

model = YOLO("son.pt")


cap = cv2.VideoCapture('son.mp4')

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_file = "output_video.mp4"
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

def calculate_intersection_over_union(points, box2):
    x1, y1, x2, y2 = min(points, key=lambda x: x[0])[0], min(points, key=lambda x: x[1])[1], max(points, key=lambda x: x[0])[0], max(points, key=lambda x: x[1])[1]
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    intersection_x1 = max(x1, box2[0])
    intersection_y1 = max(y1, box2[1])
    intersection_x2 = min(x2, box2[2])
    intersection_y2 = min(y2, box2[3])
    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    union_area = area_box1 + area_box2 - intersection_area
    iou = intersection_area / union_area
    return iou

# Belirtilen beş nokta
points = [(137, 109), (137, 475), (542, 477), (537, 116)]

start_time = time.time()
frame_count = 0
entry_times = []
current_state = 0
first_unauthorized_screenshot_taken = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = frame_count / fps  
    cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    detections = model(frame)[0]

    detected_state = 0  
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, ID = detection
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        if score >= 0.4:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            box2 = [x1, y1, x2, y2]

            if int(ID) in class_names:
                class_name = class_names[int(ID)]
            else:
                class_name = 'Unknown'

            cv2.putText(frame, class_name, (x1 + 50, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            intersection_over_union = calculate_intersection_over_union(points, box2)
            if intersection_over_union >= 0.1:
                print("Belirlenen alanda insan var")
                roi = frame[y1:y2, x1:x2]
                baret_detections = model(roi)

                for baret_detection in baret_detections[0].boxes.data.tolist():
                    bx1, by1, bx2, by2, baret_score, baret_ID = baret_detection
                    bx1, by1, bx2, by2 = map(int, [bx1, by1, bx2, by2])

                    if int(baret_ID) in [1, 2]:
                        baret_color = class_names[int(baret_ID)]
                        cv2.putText(frame, f"Baret: {baret_color}, {baret_score:.2f}", (x1 + bx1 + 10, y1 + by1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

                        if baret_color == 'yellow':
                            print("Belirtilen alanda insan var ve sarı baretle çalışıyor.")
                            cv2.putText(frame, "Yetkisiz Giris!", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                            detected_state = 1  # Sarı baret veya insan tespiti
                            if not first_unauthorized_screenshot_taken:
                                cv2.imwrite("first_unauthorized_entry.png", frame)
                                first_unauthorized_screenshot_taken = True
                                #Oto Mail
                                # send_email_with_attachment(subject, body, to, attachment_path)
                        elif baret_color == 'blue':
                            print("Belirtilen alanda insan var ve mavi baretle çalışıyor.")
                            cv2.putText(frame, "Yetkili Giris!", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                            detected_state = 2  # Mavi baret tespiti

    if detected_state != current_state:
        if current_state != 0:
            entry_times.append((entry_start_time, current_time, current_state))
        entry_start_time = current_time
        current_state = detected_state

    elapsed_time = time.time() - start_time
    real_fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {real_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

    cv2.imshow("Video", frame)
    out.write(frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

if current_state != 0:
    entry_times.append((entry_start_time, current_time, current_state))

cap.release()
out.release()
cv2.destroyAllWindows()

# Grafiği oluştur
total_duration = int(frame_count / fps)
time_series = [0] * (total_duration + 1)

for entry_start, entry_end, state in entry_times:
    start_sec = int(entry_start)
    end_sec = int(entry_end)
    for sec in range(start_sec, end_sec + 1):
        time_series[sec] = state

plt.plot(time_series)
plt.xlabel("Saniye")
plt.ylabel("Giriş Durumu")
plt.title("Yetkili ve Yetkisiz Giriş Zamanları")
plt.show()
