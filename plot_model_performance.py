from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

YOLOM = "yolo11m"
YOLON = "yolo11n"

EPOCH50 = "50epoch"
EPOCH100 = "100epoch"
EPOCH200 = "200epoch"

ROTIFERA = "rotifera"
NO_ROTIFERA = "no_rotifera"

EXPERIMET = "exp1"

TOBEST = "weights/best.pt"

OUTPUT_DIR = "compare_results"

# all models to be compared
models = [
    f"{YOLOM}/{EPOCH50}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLOM}/{EPOCH100}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLOM}/{EPOCH200}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH50}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH100}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH200}/{ROTIFERA}/{EXPERIMET}",
    f"{YOLOM}/{EPOCH50}/{NO_ROTIFERA}/{EXPERIMET}",
    f"{YOLOM}/{EPOCH100}/{NO_ROTIFERA}/{EXPERIMET}",
    f"{YOLOM}/{EPOCH200}/{NO_ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH50}/{NO_ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH100}/{NO_ROTIFERA}/{EXPERIMET}",
    f"{YOLON}/{EPOCH200}/{NO_ROTIFERA}/{EXPERIMET}",
]

model_recalls = []
model_map_scores = []
model_labels = []

os.makedirs(OUTPUT_DIR, exist_ok=True)

# evaluate all models and save results into vaiables
with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as file:
    for model_path in models:
        model = YOLO(f"{model_path}/{TOBEST}")
        metrics = model.val()
        
        model_name = model_path.split('/')[0]
        epoch = model_path.split('/')[1]
        rotifera_status = model_path.split('/')[2]
        
        recall = metrics.box.mr
        map50_95 = metrics.box.map
        
        model_map_scores.append(map50_95)
        model_recalls.append(recall)
        model_labels.append(f"{model_name}\n{epoch}\n{'Dataset 1' if rotifera_status == 'rotifera' else 'Dataset 2'}")

        # write results to results file
        file.write(f"Model: {model_name}, Epoch: {epoch}, Rotifera: {rotifera_status} - ")
        file.write(f"Recall: {recall}, mAP0.5-0.95: {map50_95}\n")


# plot the results
fig, ax1 = plt.subplots(figsize=(10, 5))

# double chart
ax2 = ax1.twinx()

width = 0.4

x = range(len(models))

ax1.bar(x, [recall for recall in model_recalls], width, label='Recall', color="g", align='center')
ax2.bar([p + width for p in x], model_map_scores, width, label='mAP0.5-0.95', color="b", align='center')

ax1.set_xlabel('Model')
ax1.set_ylabel('Recall' )
ax2.set_ylabel('mAP0.5-0.95')

# set litims for y axes to show differenceis clearly
ax1.set_ylim(0.7, 0.9)
ax2.set_ylim(0.6, 0.9)

ax1.set_xticks([p + width/2 for p in x])
ax1.set_xticklabels(model_labels, rotation=90, ha='center') 

fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(top=0.9)

fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.title('Recall and mAP0.5-0.95 for Each Model')
# save the plot
plt.savefig(os.path.join(OUTPUT_DIR, "recall_map_combined.png"))
