# AIOT Lab

# see more I made

https://drive.google.com/drive/folders/1hj1ySTmT1uvt-aySBAlxRnxvX-v0IKXv?usp=share_link

# referrence
https://github.com/google-coral


程式有個資不給看🤣

這個腳本從預設的網路攝影機擷取影像，應用預先訓練好的 TensorFlow Lite 模型對影像流中的物體進行分類，並在偵測到特定物體時發送通知到 IFTTT。具體而言，如果腳本在影像中檢測到未戴口罩的人，它會發送一個 LINE 和一個 Google Sheets 通知。

腳本需要兩個參數：--model 和 --video。--model 指定 TensorFlow Lite 模型的目錄路徑，而 --video 則指定網路攝影機的視訊號碼。如果未指定 --video，則使用默認值 0。

腳本從指定的標籤文件加載標籤，使用指定的模型文件初始化 TensorFlow Lite 解譯器，設置輸入和輸出索引。然後，它從攝像頭擷取幀，將幀裁剪成正方形，並將它們調整為 224x224。接著，它應用 TensorFlow Lite 模型對幀中的物體進行分類，並在幀上顯示結果。

如果偵測到的物體是一個未戴口罩的人，其置信分數大於等於 0.8，且自上一次發送通知以來的延遲時間至少為 2 秒，則腳本會發送 LINE 和 Google Sheets 通知，指示在影片中偵測到了一個未戴口罩的人，以及執行腳本的計算機的 IP 地址。通知使用兩個獨立的線程發送，以提高性能。

總的來說，該腳本旨在從預設的網路攝影機中檢測和通知使用者任何未戴口罩的人在影像流中。
