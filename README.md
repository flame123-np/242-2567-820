# 🧈 Butter Classification — Object Detection (YOLO)

โปรเจกต์ Object Detection สำหรับตรวจจับและจำแนกประเภทผลิตภัณฑ์ที่คล้ายเนย 4 ชนิด จากภาพถ่าย โดยใช้โมเดล **YOLO (Ultralytics)** ฝึกจากชุดข้อมูลที่รวบรวมและติดป้ายกำกับเองผ่าน Roboflow

---

## 🎯 เป้าหมายของระบบ

ตรวจจับตำแหน่ง (bounding box) และจำแนกประเภทสิ่งที่อยู่ในภาพ ออกเป็น 4 คลาส:

* Butter (เนย)
* Margarine (มาการีน)
* Peanut Butter (เนยถั่ว)
* Shortening (เนยขาว)

---

## 🛠️ เทคโนโลยีที่เลือกใช้ (Tech Stack)

* **[Ultralytics YOLO](https://docs.ultralytics.com/):** เฟรมเวิร์กสำหรับเทรนและรัน object detection model
* **Python:** ภาษาหลักที่ใช้เขียนสคริปต์ฝึกโมเดลและทดสอบ (`train.py`, `detect.py`)
* **[Roboflow](https://roboflow.com/):** ใช้จัดการ/เตรียมชุดข้อมูล (dataset: `butter-yoycg`, version 1, license CC BY 4.0)
* **CUDA:** ฝึกโมเดลบน GPU (`device="cuda"`)

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
242-2567-820/
├── data.yaml           # กำหนด path ของ train/val/test set และชื่อคลาส
├── train.py            # สคริปต์สำหรับเทรนโมเดล (100–200 epochs, imgsz 640)
├── detect.py            # สคริปต์สำหรับรันตรวจจับภาพด้วยโมเดลที่เทรนแล้ว
├── testbutter.jpg       # ภาพตัวอย่างสำหรับทดสอบโมเดล
└── train/               # ผลลัพธ์จากการเทรน (กราฟ, confusion matrix, ตัวอย่างภาพ)
    ├── results.png / results.csv
    ├── confusion_matrix(_normalized).png
    ├── P_curve.png, R_curve.png, PR_curve.png, F1_curve.png
    └── train_batch*.jpg, val_batch*_labels/pred.jpg
```

> หมายเหตุ: `data.yaml` อ้าง path แบบ local (`D:/flameai/dataset/...`) ซึ่งผูกกับเครื่องที่ใช้เทรนตอนแรก หากจะรันต่อบนเครื่องอื่นต้องแก้ path ให้ตรงกับตำแหน่งไฟล์จริง และไฟล์ weight ที่ผ่านการเทรน (`best.pt`) ไม่ได้แนบมาในรีโปนี้ ต้องเทรนเองหรือหามาใส่เพิ่มก่อนจึงจะรัน `detect.py` ได้

---

## 📊 ผลการเทรนล่าสุด (Training Result Snapshot)

จากการเทรนครบ 200 epochs (`imgsz=640`, `batch=16`):

| Metric | ค่า |
|---|---|
| Precision | ~0.55 |
| Recall | ~0.39 |
| mAP@50 | ~0.41 |
| mAP@50-95 | ~0.17 |

(ดูรายละเอียดเต็มได้ที่ `train/results.csv` และกราฟใน `train/results.png`)

---

## 🚀 วิธีใช้งาน (Getting Started)

```bash
# ติดตั้งไลบรารีหลัก
pip install ultralytics

# เทรนโมเดล (แก้ path ใน data.yaml ให้ตรงกับเครื่องตัวเองก่อน)
python train.py

# ทดสอบตรวจจับภาพด้วยโมเดลที่เทรนแล้ว
python detect.py
```

---

## 👥 ผู้พัฒนา (Developer)

* **Nuttaphat** ([@flame123-np](https://github.com/flame123-np))

*(หมายเหตุ: โปรเจกต์นี้จัดทำในภาคการศึกษา 2567 กลุ่มเรียน 820 — ตามชื่อรีโป `242-2567-820`)*
