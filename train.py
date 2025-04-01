from ultralytics import YOLO

if __name__ == '__main__':
    # โหลดโมเดล YOLO11m
    model = YOLO("runs/detect/train/weights/best.pt")  # เปลี่ยน trainX เป็นโฟลเดอร์ที่เทรนล่าสุด


    # ตั้งค่าการฝึกโมเดล
    model.train(
    data="D:/flameai/dataset/data.yaml",  # ไฟล์ YAML ที่กำหนด path ของ dataset
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="cuda"  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # ประเมินผลโมเดลที่เทรนเสร็จแล้ว
    results = model.val()
    









