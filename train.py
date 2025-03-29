from ultralytics import YOLO

# training yolo11m model with rotifera dataset with 50, 100 and 200 epochs
model = YOLO("yolo11m.pt")
results = model.train(data="data.yaml", epochs=50, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/50epoch/rotifera", name="exp1")

model = YOLO("yolo11m.pt")
results = model.train(data="data.yaml", epochs=100, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/100epoch/rotifera", name="exp1")

model = YOLO("yolo11m.pt")
results = model.train(data="data.yaml", epochs=200, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/200epoch/rotifera", name="exp1")

# training yolo11m model with no rotifera dataset with 50, 100 and 200 epochs
model = YOLO("yolo11m.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=50, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/50epoch/no_rotifera", name="exp1")

model = YOLO("yolo11m.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=100, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/100epoch/no_rotifera", name="exp1")

model = YOLO("yolo11m.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=200, imgsz=640, 
                      device=0, batch=10, workers=12, amp=True, 
                      pretrained=False, project="yolo11m/200epoch/no_rotifera", name="exp1")

# training yolo11n model with rotifera dataset with 50, 100 and 200 epochs
model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=50, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/50epoch/rotifera", name="exp1")

model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=100, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/100epoch/rotifera", name="exp1")

model = YOLO("yolo11n.pt")
results = model.train(data="data.yaml", epochs=200, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/200epoch/rotifera", name="exp1")

# training yolo11n model with no rotifera dataset with 50, 100 and 200 epochs
model = YOLO("yolo11n.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=50, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/50epoch/no_rotifera", name="exp1")

model = YOLO("yolo11n.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=100, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/100epoch/no_rotifera", name="exp1")

model = YOLO("yolo11n.pt")
results = model.train(data="data_no_rotifera.yaml", epochs=200, imgsz=640, 
                      device=0, batch=40, workers=9, amp=True, 
                      pretrained=False, project="yolo11n/200epoch/no_rotifera", name="exp1")