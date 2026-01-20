# NVIDIA cuPHY PRACH Receiver 完整文檔

## 1. 概述與應用場景

### 1.1 基本定義
PRACH（Physical Random Access Channel）接收器是 5G NR 上行鏈路**隨機接入**流程中的關鍵組件，負責檢測使用者終端（UE）發送的**前導序列（preambles）**，並估計其**時延、功率**和**干擾**等參數。

### 1.2 信號處理流程
```
接收信號（O-RAN）
    ↓
[PRACH Receiver] ← 本組件
    ↓
+─────────────────────────────────────+
│ FFT 變換（時域 → 頻域）              │
│ Zadoff-Chu (ZC) 序列相關             │
│ 功率延遲輪廓 (PDP) 計算              │
│ 峰值檢測與閾值比較                  │
│ 時延估計（IFFT）                     │
│ 功率和 RSSI 估計                     │
└─────────────────────────────────────┘
    ↓
檢測結果：
- 檢測到的前導序列個數
- 前導序列索引 (0-63)
- 時延估計（秒）
- 功率估計（dBm）
- 天線 RSSI / 總體 RSSI
- 干擾估計
    ↓
隨機接入程序
```

### 1.3 核心功能

1. **ZC 序列生成**：根據根序列索引、迴圈移位區域、前導索引生成 64 個 Zadoff-Chu 參考序列
2. **相關計算**：對每個接收天線進行頻域相關（接收信號 × 共軛 ZC 參考序列）
3. **功率延遲輪廓計算**：IFFT 後計算平方功率，得到 PDP
4. **峰值檢測**：在 PDP 中搜索超過閾值的峰值，檢測前導序列
5. **時延估計**：根據 PDP 峰值位置、PDP 寬度估計到達時間（ToA）
6. **功率估計**：計算檢測到的每個前導序列的功率和信噪比

### 1.4 應用場景

- **5G NR 隨機接入**：UE 初始連接、握手程序
- **多小區聚合**：同步處理多個小區的多個 PRACH 時機
- **多天線接收**：支持 MIMO 接收、波束成形
- **幀結構支援**：支持 FDD、TDD、兼容 3GPP TS 38.211

---

## 2. 理論基礎

### 2.1 Zadoff-Chu 序列

#### 2.1.1 ZC 序列定義
Zadoff-Chu 序列是一種恆功率、低互相關特性的序列，定義為：

$$x_u(i) = \exp\left(-j\pi u \frac{i(i+1)}{L_{RA}}\right), \quad i = 0, 1, \ldots, L_{RA}-1$$

其中：
- $u$：根索引（根序列編號）
- $L_{RA}$：序列長度（839 或 139）
- $i$：樣本索引

#### 2.1.2 迴圈移位序列
應用迴圈移位 $C_v$ 後的序列：

$$x_{u,v}(n) = x_u((n + C_v) \bmod L_{RA})$$

其中 $C_v = v \cdot N_{CS}$，$v$ 是迴圈移位索引，$N_{CS}$ 是零相關區配置。

#### 2.1.3 頻域 ZC 參考序列
通過 DFT 得到頻域序列：

$$y_{u,v}(k) = \text{DFT}\{x_{u,v}(n)\} = \sum_{n=0}^{L_{RA}-1} x_{u,v}(n) e^{-j2\pi kn/L_{RA}}, \quad k = 0, 1, \ldots, L_{RA}-1$$

### 2.2 前導序列檢測算法

#### 2.2.1 相關計算
對於接收信號 $r(n)$ 和參考序列 $y_u(k)$，頻域相關為：

$$z_u(n) = \text{FFT}\{r_{\text{segment}}(t)\} \cdot \overline{y_u(k)}$$

其中 $\overline{(\cdot)}$ 表示複共軛。

#### 2.2.2 功率延遲輪廓 (PDP)
將相關結果轉換回時域並計算功率：

$$\text{PDP}_u(n) = \left|\text{IFFT}\{z_u(k)\} \cdot \frac{N_{\text{FFT}}}{L_{RA}}\right|^2, \quad n = 0, 1, \ldots, N_{\text{FFT}}-1$$

#### 2.2.3 峰值檢測
在 PDP 中搜索超過動態閾值的峰值：

$$\text{Peak} = \max_{n} \text{PDP}_u(n)$$

**兩級閾值比較**：
- **Threshold 0 ($\text{thr0}$)**：SNR 閾值，過濾噪聲峰值
- **Threshold 1 ($\text{thr1}$)**：相對功率閾值，選擇主要檢測

#### 2.2.4 時延估計
根據峰值位置估計 ToA：

$$\text{Delay} = \frac{\text{PDP}_{\text{peak\_idx}}}{N_{\text{FFT}} \cdot \Delta f_{RA}}$$

其中 $\Delta f_{RA}$ 是 PRACH 子載波間距（1.25 kHz、5 kHz 或 15 kHz）。

### 2.3 3GPP 標準參考

**相關標準**：
- **TS 38.211 第 6.3.3 節**：PRACH 序列生成和映射
- **TS 38.211 表 6.3.3.1-3 和 6.3.3.1-4**：ZC 序列根索引映射表
- **TS 38.211 表 6.3.3.1-5 至 6.3.3.1-8**：零相關區配置
- **TS 38.212 第 6.2.3 節**：前導序列檢測指示

### 2.4 前導格式與參數

| 格式 | $\Delta f_{RA}$ | $L_{RA}$ | $N_{\text{FFT}}$ | $N_{\text{rep}}$ | $N_{CS}$ 範圍 | 應用 |
|-----|----------------|---------|-----------------|-----------------|-------------|------|
| 0, 1, 2 | 1.25 kHz | 839 | 1024 | 1-2 | 13-77 | FR1（主要） |
| 3 | 5 kHz | 839 | 1024 | 1-2 | 13-77 | FR1 高 SCS |
| 4 | 15 kHz | 139 | 256 | 1-2 | 13-77 | FR1 mu≥2 |
| 5 | 30 kHz | 139 | 256 | 1-4 | 13-77 | FR2（毫米波） |
| 6 | 60 kHz | 139 | 256 | 4-8 | 13-77 | FR2 高 SCS |

---

## 3. 數據結構

### 3.1 PRACH 參數結構

#### 3.1.1 PrachParams（導出的參數）
```cpp
struct PrachParams
{
    // 基本參數
    uint32_t L_RA;          // ZC 序列長度（839 或 139）
    uint32_t N_CS;          // 零相關區步長（配置編號）
    uint32_t delta_f_RA;    // PRACH 子載波間距（Hz）：1250, 5000, 15000, 60000
    uint32_t Nfft;          // FFT 大小（1024 或 256）
    uint32_t N_ant;         // 接收天線數
    uint32_t mu;            // 數值學（0-4）
    
    // 導出參數
    uint32_t uCount;        // 唯一根序列個數（基於 N_CS）
    uint32_t N_rep;         // 前導序列重複次數（1-8）
    uint32_t N_nc;          // 非相干合並次數
    uint32_t kBar;          // 保護子載波數
};
```

#### 3.1.2 內部動態參數（GPU）
```cpp
struct PrachInternalDynParamPerOcca
{
    __half2* dataRx;                // 接收信號指針（頻域，複數）
    uint16_t occaPrmStatIdx;        // 指向靜態參數的索引
    uint16_t occaPrmDynIdx;         // 指向動態參數的索引
    float thr0;                     // SNR 閾值
    uint16_t nUplinkStreams;        // 上行流數量
};
```

#### 3.1.3 內部靜態參數（設備）
```cpp
struct PrachDeviceInternalStaticParamPerOcca
{
    PrachParams prach_params;               // PRACH 參數
    float* prach_workspace_buffer;          // 工作區（FFT、PDP、檢測結果）
    __half2* d_y_u_ref;                     // ZC 參考序列（GPU）
    uint8_t enableUlRxBf;                   // 上行波束成形標誌
};
```

### 3.2 檢測輸出結構

#### 3.2.1 檢測結果
```cpp
struct prach_det_t<Tscalar>
{
    Tscalar power_db;           // 檢測功率（dB）
    int32_t delay_samp;         // 延遲（樣本）
    uint8_t prmbIdx;            // 前導索引
};
```

#### 3.2.2 功率延遲輪廓
```cpp
struct prach_pdp_t<Tscalar>
{
    Tscalar power;              // 功率
    int32_t delay_samp;         // 延遲（樣本）
};
```

### 3.3 輸出緩衝區

| 名稱 | 數據類型 | 大小 | 說明 |
|-----|---------|------|------|
| num_detectedPrmb | uint32_t | 1 × nOccasions | 每個時機檢測到的前導序列數 |
| prmbIndex_estimates | uint32_t | 64 × nOccasions | 檢測到的前導索引 |
| prmbDelay_estimates | float | 64 × nOccasions | 時延估計（秒） |
| prmbPower_estimates | float | 64 × nOccasions | 功率估計（dBm） |
| ant_rssi | float | nAntennas × nOccasions | 每天線 RSSI |
| rssi | float | nOccasions | 總體 RSSI（dBm） |
| interference | float | nOccasions | 干擾功率估計 |

---

## 4. CUDA 核函數詳解

### 4.1 核函數流水線

PRACH 接收器包含 7 個主要 CUDA 核函數階段（按執行順序）：

```
[1] FFT Kernel
    ├─ 輸入：時域接收信號（O-RAN 格式）
    ├─ 操作：cufftDx 或 cuFFT 快速傅立葉變換
    └─ 輸出：頻域信號
        ↓
[2] Compute Correlation Kernel
    ├─ 輸入：頻域信號 + ZC 參考序列
    ├─ 操作：元素級相乘（接收信號 × 共軛參考序列）
    └─ 輸出：相關信號
        ↓
[3] Compute PDP Kernel
    ├─ 輸入：相關信號
    ├─ 操作：平方功率計算 |correlation|²
    └─ 輸出：功率延遲輪廓
        ↓
[4] Search PDP Kernel
    ├─ 輸入：PDP + 閾值參數
    ├─ 操作：搜索峰值、時延估計
    └─ 輸出：檢測結果（前導索引、時延、功率）
        ↓
[5] Compute RSSI Kernel
    ├─ 輸入：接收功率
    ├─ 操作：聚合計算 RSSI、干擾
    └─ 輸出：RSSI（dB）
        ↓
[6] Memcpy RSSI Kernel
    └─ 設備到主機複製（可選）
        ↓
[7] CPU 後處理
    └─ 格式化輸出
```

### 4.2 相關計算核函數：compute_corr

#### 4.2.1 核函數簽名
```cuda
template<typename Tcomplex, typename Tscalar>
__global__ void compute_corr(const PrachInternalDynParamPerOcca* d_dynParam,
                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam)
```

#### 4.2.2 主要邏輯
```cuda
__global__ void compute_corr(...)
{
    // 1. 獲取時機索引和參數
    uint32_t batchIndex = blockIdx.x;
    uint16_t occaPrmStatIdx = d_dynParam[batchIndex].occaPrmStatIdx;
    const PrachParams* prach_params = &(d_staticParam[occaPrmStatIdx].prach_params);
    
    // 2. 提取參數
    int N_ant = prach_params->N_ant;
    int uCount = prach_params->uCount;
    int L_RA = prach_params->L_RA;
    int Nfft = prach_params->Nfft;
    
    // 3. 計算工作區指針
    Tcomplex* d_fft = (Tcomplex*)(d_staticParam[occaPrmStatIdx].prach_workspace_buffer);
    __half2* d_y_u_ref = d_staticParam[occaPrmStatIdx].d_y_u_ref;
    
    // 4. 並行計算：每個線程計算一個（天線、U 序列）對
    int antIdx = blockIdx.y;              // 天線索引
    int uIdx = blockIdx.z;                // U 序列索引
    
    for (int fftIdx = threadIdx.x; fftIdx < L_RA; fftIdx += blockDim.x) {
        // 相關計算：r[fftIdx] = fft[antIdx][fftIdx] * conj(y_ref[uIdx][fftIdx])
        Tcomplex rfft_val = d_fft[...];
        Tcomplex yref_conj = conj(d_y_u_ref[...]);
        Tcomplex corr = rfft_val * yref_conj;
        d_corr[...] = corr;
    }
}
```

### 4.3 功率延遲輪廓計算

#### 4.3.1 PDP 計算核函數：compute_pdp
```cuda
__global__ void compute_pdp(
    const PrachInternalDynParamPerOcca* d_dynParam,
    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
    ...)
{
    // 並行 IFFT + 平方功率計算
    int uIdx = blockIdx.x;
    int antIdx = blockIdx.y;
    
    // 每個線程塊處理一個（天線, U）對的 PDP
    __shared__ Tscalar local_power[NUM_THREAD];
    __shared__ Tscalar local_max[NUM_THREAD];
    __shared__ int local_loc[NUM_THREAD];
    
    // 執行 IFFT（每個線程塊）
    prach_ifft<Tcomplex>(d_corr, d_pdp, L_RA, Nfft);
    
    // 計算功率：power[n] = |ifft_result[n]|² * (Nfft/L_RA)²
    for (int idx = threadIdx.x; idx < Nfft; idx += blockDim.x) {
        Tcomplex val = d_pdp[...];
        Tscalar power = (val.x * val.x + val.y * val.y) * 
                       (Nfft / L_RA) * (Nfft / L_RA);
        d_pdp_power[idx] = power;
    }
}
```

### 4.4 峰值搜索核函數：search_pdp

#### 4.4.1 算法流程
```cuda
__global__ void search_pdp(
    const prach_pdp_t<Tscalar>* d_pdp,
    uint32_t* num_detectedPrmb_addr,
    uint32_t* prmbIndex_estimates_addr,
    float* prmbDelay_estimates_addr,
    ...)
{
    // 1. 初始化每個前導索引的檢測結果
    __shared__ Tscalar local_max[NUM_PREAMBLE];
    __shared__ int local_loc[NUM_PREAMBLE];
    __shared__ Tscalar local_snr[NUM_PREAMBLE];
    
    // 2. 迴圈遍歷 PRACH 區域
    for (int prmbCount = threadIdx.x; prmbCount < 64; prmbCount += blockDim.x) {
        // 計算區域參數
        int NzonePerU = L_RA / N_CS;
        int uIdx = prmbCount / NzonePerU;
        int zone_idx = prmbCount % NzonePerU;
        
        // 計算區域邊界
        int zone_start = (Nfft - C_v[prmbCount] * Nfft / L_RA) % Nfft;
        int zoneSize = (N_CS * Nfft + L_RA - 1) / L_RA;
        
        // 搜索區域內的最大值
        Tscalar max_power = 0;
        int max_loc = 0;
        
        for (int idx = 0; idx < zoneSize; idx++) {
            Tscalar power = d_pdp[...].power;
            if (power > max_power) {
                max_power = power;
                max_loc = idx;
            }
        }
        
        // 3. 計算背景噪聲和 SNR
        Tscalar bg_power = compute_background_power(d_pdp, zone_start, zoneSize);
        Tscalar snr_db = 10 * log10(max_power / bg_power) - 10 * log10(L_RA);
        
        // 4. 階段 1：SNR 閾值比較
        if (snr_db > thr0) {
            local_max[prmbCount] = max_power;
            local_loc[prmbCount] = max_loc;
            local_snr[prmbCount] = snr_db;
        }
    }
    
    __syncthreads();
    
    // 5. 階段 2：相對功率排序
    sort_and_detect(local_max, local_loc, local_snr, thr1);
    
    // 6. 輸出檢測結果
    uint32_t detIdx = count_detections();
    if (threadIdx.x == 0) {
        num_detectedPrmb_addr[occaPrmDynIdx] = detIdx;
    }
    
    for (int i = threadIdx.x; i < detIdx; i += blockDim.x) {
        prmbIndex_estimates_addr[...] = prmb_indices[i];
        prmbDelay_estimates_addr[...] = delays[i];
        prmbPower_estimates_addr[...] = powers[i];
    }
}
```

### 4.5 RSSI 計算核函數

```cuda
__global__ void compute_rssi(
    const __half2* d_prach_rx,
    float* d_ant_rssi,
    float* d_rssi,
    float* d_interference,
    ...)
{
    // 計算每天線功率
    float ant_power = 0.0f;
    for (int idx = threadIdx.x; idx < nSamples; idx += blockDim.x) {
        __half2 sample = d_prach_rx[idx];
        float real = __half2float(sample.x);
        float imag = __half2float(sample.y);
        ant_power += real * real + imag * imag;
    }
    
    // 歸一化和聚合
    ant_power /= nSamples;
    float ant_rssi_db = 10.0f * log10(ant_power + 1e-9f);
    
    // 儲存結果
    d_ant_rssi[threadIdx.z] = ant_rssi_db;
}
```

---

## 5. 最佳實踐與優化

### 5.1 計算複雜度

| 階段 | 操作 | 複雜度 | 說明 |
|-----|-----|-------|------|
| **FFT** | 一維快速傅立葉變換 | $O(N \log N)$ | N = Nfft = 1024 或 256 |
| **相關** | 逐元素乘法 | $O(N \cdot uCount)$ | uCount = 1-64 |
| **IFFT** | 逆 FFT | $O(N \log N)$ | 恢復時域 |
| **PDP** | 平方功率 + 搜索 | $O(N)$ | 線性掃描 |
| **檢測** | 排序 + 閾值比較 | $O(64 \log 64)$ | 最多 64 前導 |

**總體複雜度**：$O(N_{\text{ant}} \cdot uCount \cdot N \log N)$

### 5.2 內存使用優化

#### 5.2.1 工作區分配
```
prach_workspace_buffer:
├─ FFT 輸出：N_ant × uCount × Nfft × sizeof(complex)
├─ PDP：N_ant × uCount × Nfft × sizeof(float)
├─ 檢測結果：64 × sizeof(detection_struct)
├─ RSSI 聚合：N_ant × sizeof(float)
└─ 臨時緩衝區

總大小 ≈ 10-50 MB（取決於配置）
```

#### 5.2.2 共享內存優化
- 使用共享內存存儲局部 PDP（減少全局內存訪問）
- 使用 warp-level shuffle 進行歸約操作
- 避免銀行衝突（偏移佈局）

### 5.3 CUDA 設備架構適配

```cpp
// 根據計算能力選擇內核
if (computeCapability >= 80) {  // Ampere
    // 使用張量核心、更高的佔有率
    use_cufftdx_kernels();
} else if (computeCapability >= 70) {  // Volta
    // 使用標準 cuFFT
    use_cufft_kernels();
}
```

---

## 6. 主機端 API

### 6.1 創建函數

```cpp
cuphyStatus_t cuphyCreatePrachRx(
    cuphyPrachRxHndl_t* pPrachRxHndl,
    cuphyPrachStatPrms_t const* pStatPrms)
{
    // 驗證靜態參數
    cuphyStatus_t status = validateStaticParams(pStatPrms);
    if (status != CUPHY_STATUS_SUCCESS) return status;
    
    // 為每個時機創建 PrachRx 對象
    PrachRx* pObj = new PrachRx(pStatPrms, &status);
    *pPrachRxHndl = static_cast<cuphyPrachRxHndl_t>(pObj);
    
    return status;
}
```

### 6.2 設置函數

```cpp
cuphyStatus_t cuphySetupPrachRx(
    cuphyPrachRxHndl_t prachRxHndl,
    cuphyPrachDynPrms_t* pDynPrms)
{
    if (!prachRxHndl || !pDynPrms) 
        return CUPHY_STATUS_INVALID_ARGUMENT;
    
    PrachRx* pipeline_ptr = static_cast<PrachRx*>(prachRxHndl);
    
    // 擴展動態參數、分配臨時緩衝區
    return pipeline_ptr->expandParameters(pDynPrms);
}
```

### 6.3 執行函數

```cpp
cuphyStatus_t cuphyRunPrachRx(cuphyPrachRxHndl_t prachRxHndl)
{
    if (!prachRxHndl) 
        return CUPHY_STATUS_INVALID_ARGUMENT;
    
    PrachRx* pipeline_ptr = static_cast<PrachRx*>(prachRxHndl);
    
    // 執行核函數流水線
    return pipeline_ptr->Run();
}
```

### 6.4 銷毀函數

```cpp
cuphyStatus_t cuphyDestroyPrachRx(cuphyPrachRxHndl_t prachRxHndl)
{
    if (!prachRxHndl) 
        return CUPHY_STATUS_INVALID_ARGUMENT;
    
    PrachRx* pObj = static_cast<PrachRx*>(prachRxHndl);
    delete pObj;
    
    return CUPHY_STATUS_SUCCESS;
}
```

---

## 7. 使用示例

### 7.1 完整使用流程

#### 7.1.1 初始化與配置
```cpp
#include "cuphy.h"
#include <cuda_runtime.h>

// 創建 PRACH 接收器
cuphyPrachRxHndl_t prach_handle;
cuphyPrachStatPrms_t stat_prms = {
    .nMaxCells = 2,           // 最大小區數
    .nMaxOccaProc = 4,        // 最大時機數
    // ... 其他靜態參數
};

cuphyStatus_t status = cuphyCreatePrachRx(&prach_handle, &stat_prms);
if (status != CUPHY_STATUS_SUCCESS) {
    printf("Failed to create PRACH RX\n");
    return -1;
}
```

#### 7.1.2 設置參數
```cpp
// 準備動態參數
cuphyPrachDynPrms_t dyn_prms;
dyn_prms.nPolUciSegs = 1;
// ... 設置 UCI 分段參數

// 複製接收信號到 GPU
cudaMemcpyAsync(dyn_prms.pTDataRx[0].pAddr, 
                host_rx_signal, 
                rx_signal_size,
                cudaMemcpyHostToDevice, 
                stream);

// 設置 PRACH 接收器
status = cuphySetupPrachRx(prach_handle, &dyn_prms);
```

#### 7.1.3 執行檢測
```cpp
// 運行 PRACH 接收器
status = cuphyRunPrachRx(prach_handle);
if (status != CUPHY_STATUS_SUCCESS) {
    printf("PRACH detection failed\n");
    return -1;
}

cudaStreamSynchronize(stream);
```

#### 7.1.4 讀取結果
```cpp
// 從 GPU 複製結果到主機
uint32_t num_detected_prmb;
uint32_t prmbIndex[64];
float prmbDelay[64];
float prmbPower[64];

cudaMemcpy(&num_detected_prmb, 
           dyn_prms.pNumDetectedPrmb,
           sizeof(uint32_t),
           cudaMemcpyDeviceToHost);

cudaMemcpy(prmbIndex, 
           dyn_prms.pPrmbIndexEstimates,
           64 * sizeof(uint32_t),
           cudaMemcpyDeviceToHost);

// 處理檢測結果
printf("Detected preambles: %u\n", num_detected_prmb);
for (uint32_t i = 0; i < num_detected_prmb; ++i) {
    printf("  Preamble %u: index=%u, delay=%.3e s, power=%.1f dBm\n",
           i, prmbIndex[i], prmbDelay[i], prmbPower[i]);
}
```

#### 7.1.5 清理
```cpp
// 銷毀 PRACH 接收器
status = cuphyDestroyPrachRx(prach_handle);
```

### 7.2 多時機聚合示例

```cpp
// 同時處理多個小區和時機
for (uint32_t cell = 0; cell < num_cells; ++cell) {
    for (uint32_t occ = 0; occ < num_occasions; ++occ) {
        uint32_t idx = cell * num_occasions + occ;
        
        // 設置每個時機的接收信號
        dyn_prms.pTDataRx[idx].pAddr = rx_buffers[idx];
        
        // 設置閾值參數
        dyn_prms.pDynPrmsPerOcca[idx].thr0 = 7.5f;  // SNR 閾值
    }
}

// 一次性執行所有時機
status = cuphySetupPrachRx(prach_handle, &dyn_prms);
status = cuphyRunPrachRx(prach_handle);

// 檢索所有結果
for (uint32_t idx = 0; idx < num_cells * num_occasions; ++idx) {
    uint32_t num_det = num_detected_prmb[idx];
    printf("Cell %u, Occasion %u: %u preambles detected\n",
           idx / num_occasions, idx % num_occasions, num_det);
}
```

---

## 8. 性能分析

### 8.1 吞吐量估計

| 配置 | 天線數 | U 個數 | 時機數 | 執行時間 | 吞吐量 |
|------|-------|-------|-------|---------|-------|
| 小 | 1 | 1 | 1 | ~10 ms | ~100 檢測/s |
| 中 | 4 | 16 | 4 | ~50 ms | ~320 檢測/s |
| 大 | 8 | 64 | 8 | ~200 ms | ~1280 檢測/s |

### 8.2 帶寬分析

**輸入帶寬**：
$$BW_{\text{in}} = \frac{N_{\text{ant}} \times L_{RA} \times \text{sizeof}(\text{complex})}{T_{\text{exec}}}$$

典型值：50-200 GB/s（4-8 天線）

**輸出帶寬**：
$$BW_{\text{out}} = \frac{64 \times \text{num\_results} \times \text{sizeof}(\text{float})}{T_{\text{exec}}}$$

典型值：1-10 GB/s（非常小，計算密集）

### 8.3 功耗估計

- **GPU 功耗**：50-150 W（取決於配置和使用率）
- **能效**：2-5 檢測/瓦（取決於檢測複雜度）

---

## 9. 3GPP 標準映射

### 9.1 前導格式映射表

| TS 38.211 表 | 前導格式 | $\Delta f_{RA}$ | $L_{RA}$ | 實現 |
|-------------|---------|----------------|---------|------|
| 6.3.3.1-1 | 0, 1, 2, 3 | 1.25, 5 kHz | 839 | ✓ |
| 6.3.3.1-1 | 4, 5, 6 | 15, 30, 60 kHz | 139 | ✓ |

### 9.2 ZC 根序列映射

- **TS 38.211 表 6.3.3.1-3**：L_RA = 839 的邏輯索引到 u 的映射
- **TS 38.211 表 6.3.3.1-4**：L_RA = 139 的邏輯索引到 u 的映射
- **實現**：lookup tables `table_logIdx2u_839[]`、`table_logIdx2u_139[]`

### 9.3 零相關區配置

- **TS 38.211 表 6.3.3.1-5 至 6.3.3.1-8**：N_CS 配置編號 [0-419]
- **實現**：三維查找表基於 delta_f_RA、restrictedSet、zeroCorrelationZone

---

## 10. 調試與故障排除

### 10.1 常見問題

#### 10.1.1 未檢測到前導序列
**症狀**：num_detected_prmb = 0  
**原因**：
- SNR 閾值過高（thr0）
- 接收信號損壞或不存在
- 參數配置錯誤（前導格式、根序列）

**解決方案**：
```cpp
// 降低 SNR 閾值進行測試
dyn_prms.thr0 = 3.0f;  // 從 7.5 降至 3.0

// 驗證接收信號
float max_power = 0;
for (int i = 0; i < signal_len; i++) {
    max_power = max(max_power, abs(rx_signal[i]));
}
printf("Max signal level: %f\n", max_power);
```

#### 10.1.2 誤檢（假正例）
**症狀**：檢測到不應存在的前導序列  
**原因**：
- SNR 閾值過低（thr0）
- 相對功率閾值不適當（thr1）
- 噪聲干擾太強

**解決方案**：
```cpp
// 提高閾值
dyn_prms.thr0 = 10.0f;   // 增大 SNR 閾值
thr1 = 3.0f;              // 提高相對功率比
```

#### 10.1.3 時延估計錯誤
**症狀**：delay_estimates 不準確  
**原因**：
- FFT 大小不匹配
- 接收信號時域失真
- 時鐘偏差

**驗證**：
```cpp
// 檢查前導序列的 PDP 寬度
float expected_width = 1.0f / (L_RA * delta_f_RA);  // 秒
printf("Expected PDP width: %.6e s\n", expected_width);
```

### 10.2 性能分析

#### 10.2.1 使用 NVIDIA Profiler
```bash
# 測量核函數執行時間
nsys profile --output=prach_profile_%h_%t ./app

# 詳細分析
ncu --set full --o prach.ncu-rep ./app
```

#### 10.2.2 性能計數器
```cpp
// 在 PRACH 執行前後記錄事件
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
cuphyRunPrachRx(prach_handle);
cudaEventRecord(stop, stream);

float elapsed_ms = 0;
cudaEventElapsedTime(&elapsed_ms, start, stop);
printf("PRACH execution time: %.3f ms\n", elapsed_ms);
```

---

## 11. 最佳實踐

### 11.1 內存管理

1. **使用 Pinned Memory**：主機端動態參數應使用 pinned memory 加速複製
   ```cpp
   cuphyPrachDynPrms_t* pDynPrmsCpu;
   cudaMallocHost(&pDynPrmsCpu, sizeof(cuphyPrachDynPrms_t));
   ```

2. **流管理**：為避免阻塞，使用獨立的 CUDA 流
   ```cpp
   cudaStream_t prach_stream;
   cudaStreamCreate(&prach_stream);
   // 所有 PRACH 操作在該流上執行
   ```

3. **預分配**：在初始化時預分配所有工作區緩衝區，避免動態分配開銷

### 11.2 算法正確性驗證

1. **ZC 序列驗證**：與 MATLAB 參考實現比對
   ```matlab
   % MATLAB 參考
   [u, C_v] = findZcPar(prmbIdx, rootSeq, L_RA, N_CS);
   y_ref = genZcPreamble(L_RA, C_v, u);
   ```

2. **PDP 驗證**：確認 PDP 峰值與已知前導序列位置匹配

3. **時延估計驗證**：與通道延遲知識值比對

### 11.3 性能優化

1. **批量處理**：同時處理多個小區/時機以提高 GPU 利用率
2. **非同步複製**：重疊前導序列複製和計算
3. **CUDA 圖**：使用 CUDA 圖減少核函數啟動開銷

---

## 12. 總結與關鍵要點

### 12.1 核心功能回顧

PRACH 接收器實現了 5G NR 隨機接入的**前導序列檢測和定時**：

1. **ZC 序列生成**：根據標準參數動態生成 64 個參考序列
2. **頻域相關**：計算接收信號與每個 ZC 序列的複相關
3. **時域 PDP**：通過 IFFT 恢復時域功率延遲輪廓
4. **峰值檢測**：在多個 PDP 區域搜索超過閾值的檢測
5. **估計輸出**：前導索引、時延、功率、RSSI、干擾

### 12.2 關鍵實現細節

| 方面 | 實現 |
|-----|------|
| **並行策略** | 天線級並行 + U 序列級並行 + 數據級並行 |
| **FFT 加速** | cufftDx（編譯時）或 cuFFT（運行時） |
| **內存優化** | 工作區預分配、共享內存 PDP 緩存 |
| **數值精度** | float32（功率、RSSI）、float16（LLR） |
| **多時機** | 支持聚合多小區、多時機同步處理 |

### 12.3 典型性能指標

- **延遲**：10-50 ms（單時機），50-200 ms（多時機聚合）
- **吞吐量**：100-1280 檢測/秒（配置相關）
- **準確度**：>99%（SNR > 10 dB 時）
- **功耗**：50-150 W（GPU）

---

## 附錄 A：MATLAB 參考實現

### A.1 前導序列檢測函數

```matlab
function prach = detectPreamble(prach, carrier, SimCtrl)

% 提取參數
y_uv_rx = prach.y_uv_rx;
N_CS = prach.N_CS;
L_RA = prach.L_RA;
Nfft = prach.Nfft;
u_ref = prach.u_ref;
C_v_ref = prach.C_v_ref;

% 計算 PDP
for prmbIdx = 0:63
    % 相關計算
    z_u = fft(y_uv_rx, Nfft) .* conj(fft(y_ref, Nfft));
    
    % IFFT 和 PDP 計算
    pdp = abs(ifft(z_u, Nfft)).^2;
    
    % 峰值搜索
    [peak_power, peak_idx] = max(pdp);
    
    % 時延估計
    delay_samp = peak_idx;
    delay_time = delay_samp / (Nfft * delta_f_RA);
    
    % 儲存結果
    prach.prmbIdx_det(prmbIdx+1) = prmbIdx;
    prach.delay_time_det(prmbIdx+1) = delay_time;
end

end
```

### A.2 ZC 序列生成

```matlab
function y_uv = genZcPreamble(L_RA, C_v, u)

% ZC 序列生成
i = 0:L_RA-1;
x_u = exp(-1j * pi * u * i .* (i+1) / L_RA);

% 迴圈移位
n = 0:L_RA-1;
x_uv = x_u(mod(n + C_v, L_RA) + 1);

% 頻域 DFT
y_uv = fft(x_uv) / sqrt(L_RA);

end
```

---

文檔編寫時間：2026 年 1 月 20 日  
基於 cuPHY PRACH 模組版本：NVIDIA Aerial CUDA-Accelerated RAN  
標準參考：3GPP TS 38.211 v17.0.0、3GPP TS 38.212 v17.0.0
