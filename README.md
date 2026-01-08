# Nvidia-cuPHY-OSC-note
此檔案專門記錄我閱讀完以下程式碼所得出的程式邏輯

且專注於cuPHY本身，在另外一個資料夾cuPHY-CP(控制層)將會額外再開一個頁面來記錄筆記

資料來源為https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main
###  Core cuPHY library implementation(核心程式碼)[src/]
src為核心程式碼，所有功能的 root裡面再細分為以下三個類別

- **src/cuphy/**

  此為最重要的資料夾內含LDPC編解碼器，Modulation mapping(調變)，PDSCH，PDCCH，PUSCH，PUCCH，等重要部分，是「核心 L1 演算法 + 控制」，為最需要研究的部分!!!
- **src/cuphy_channels/**

  這是 cuPHY-CP（L1 controller）真正會調用的 API ， 在上面cuphy/資料夾為元件，此處為cuphy將上述部件組合起來的地方
- **src/cuphy_hdf5/**

  用來儲存測試向量、配置參數、非即時 log
  用途：

  -- 儲存 Golden reference（讓你比較 GPU output 正不正確）

  -- 可讀取 .h5 格式 test vectors

  -- 訓練/驗證模型或 pipeline

  -- 用在 examples / unit tests

  特別是 OTA 前 offline 模擬階段會很常用
###  Reference implementations and sample applications that demonstrate how to build complete(完整 PHY Pipeline 範例)[examples/]

  裡面包含：

  -- pdsch_tx（完整 PDSCH 下行傳送管線）
  
  -- pusch_rx_multi_pipe（多cell uplink 接收）
  
  -- srs_rx_pipeline（Sounding Reference Signal，用來做頻道估測）
  
  -- prach_receiver_multi_cell（隨機接入）
###  Unit and component tests for cuPHY kernels and higher-level primitives (e.g., rate matching, modulation/demodulation, error correction, HDF5 IO, etc.). This directory also includes helper scripts such as cuphy_unit_test.sh to orchestrate running collections of tests.(測試 LDPC、調變、FFT、Rate matching、HDF5 IO)[test/]

測試 LDPC、調變、FFT、Rate matching、HDF5 IO ，用來測試各部件的功能

###  Structured logging utility used throughout cuPHY (and other parts of the SDK when integrated). It is tailored to the performance requirements of the library (low overhead, high throughput logging).[nvlog/]

是用來支援 L1 運行時期的結構化、低 latency 的執行 trace 與除錯工具，配合 FAPI trace，檢查 L1↔L2互動是否正常

###  Doxygen configuration (docs/Doxyfile.in) and related documentation assets.([docs/]

- 用 Doxygen 自動產生程式碼 API 說明

- 把 .hpp/.cpp/.cu 裡的註解抓出來產生 HTML/PDF 

###  CMake utilities and toolchain files used when building cuPHY as a standalone component.[cmake/]

提供 cmake scripts，支援不同 GPU 架構編譯，是讓 cuPHY 支援各類平台的關鍵

###  Supporting utilities (e.g., MATLAB integration, performance collection scripts, LDPC helper data).(雜項)[util/]

MATLAB / 效能測試工具 / LDPC helper

###  額外名詞解釋位置

此處將放置一些我不太了解的名詞經過我了解後的地方

| 資料夾	| 功能	| 用途時間| 	針對誰| 
| ------------- | ------------- |------------- |------------- |
| nvlog/| 	執行時期 logging 系統	| 程式在跑時	| 工程師分析 L1 真實行為| 
| docs/| 	自動產生 API 文檔	| 程式在編譯前| 	工程師閱讀 source code|

此表格是nvlog/和docs/的差異

| 通道|	方向	|類型	|承載內容|
| ------------- | ------------- |------------- |------------- |
| PDCCH	|gNB → UE|	控制	|DCI（分配、排程資訊）|
| PDSCH|	gNB → UE|	資料	|下行 user data / paging / RRC|
| PUCCH	|UE → gNB|	控制|	ACK/NACK, SR, CSI|
| PUSCH|	UE → gNB|	資料|	上行 user data / MAC 控制|


其中gNB 用 PDCCH 下命令、用 PDSCH 傳資料；UE 用 PUCCH 回報、用 PUSCH傳資料
