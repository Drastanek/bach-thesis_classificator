Toto prostředí slouží k práci s modelem umělé inteligence na YOLO frameworku.

Spusťte příkazový řádek v této složce.

V případě použití linux conda prostředí, spusťte:

    conda env create -f environment_linux.yml

Případně (Windows/nefungující conda) nainstalujte závislosti a knihovny pomocí

    pip install -r requirements.txt


Scripty se spouští v příkazové řádce pomocí

    python train.py
    python predict.py
    python plot_model_performance.py

V yolo11n a yolo11m složkách jsou natrénované modely z experimentů. Složky predict_results a
data_to_classify slouží ke scriptu predict.py, kam se ukládají výsledy a odkud se berou
obrázky ke klasifikaci. Compare_results složka obsahuje výstup z plot_model_performance.py,
tento script na kažném modelu provede .val() funkci a výsledky porovná, uloží do results.txt
hodnoty Recall a mAP0.5-0.95 a z těchto hodnot udělá graf.
Script train.py využívá yolo11n.pt a yolo11m.pt PyTorch struktury a data.yaml a data_no_rotifera.yaml
s cestami k datasetům a seznam tříd.
Pro vyzkoušení je potřeba cesty k datasetům vyplnit v textovém editoru.
