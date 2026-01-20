# NVIDIA cuPHY Polar Segment De-Rate-Match De-Interleave (polar_seg_deRm_deItl) 完整文檔

## 1. 概述與應用場景

### 1.1 基本定義
`polar_seg_deRm_deItl` 組件負責 5G NR 上行鏈路 UCI（Uplink Control Information）極碼解碼流程中的**速率匹配恢復與交織恢復**，是極碼解碼前的關鍵預處理步驟。

### 1.2 信號處理流程
```
接收信號
    ↓
[polSegDeRmDeItl] ← 本模組
    ↓
UCI 分段 LLR (E_seg bits)
    ↓
速率匹配恢復 (E_cw → N_cw bits per codeword)
    ↓
交織恢復 (channel interleave undo)
    ↓
碼字 LLR (N_cw bits × nCbs codewords)
    ↓
[polarDecoder]
    ↓
解碼輸出
```

### 1.3 核心功能
1. **速率匹配恢復**（De-Rate-Matching）：根據編碼器使用的速率匹配方法，恢復從 E_cw 個傳輸位到 N_cw 個碼字位的映射
2. **交織恢復**（De-Interleaving）：撤消編碼器應用的 3GPP 標準通道交織
3. **碼字分割**（Codeword Segmentation）：將 UCI 分段分割為 1 或 2 個極碼字，並準備對應的 LLR 緩衝區

### 1.4 應用場景
- **PUSCH 上行鏈路**：傳輸控制信息（CQI、HARQ-ACK、SR）
- **多個 UCI 分段並行處理**：支持批量的 LLR 恢復
- **與極碼解碼器集成**：上行 UCI 接收完整流程

---

## 2. 理論基礎

### 2.1 速率匹配概述

#### 2.1.1 編碼過程回顧
編碼器產生 N_cw 位的碼字，經過速率匹配後輸出 E_cw 位用於傳輸。

**三種速率匹配方法**：
| 方法 | 條件 | 操作 | 說明 |
|-----|-----|------|------|
| **Repetition (0)** | $E_{cw} \geq N_{cw}$ | 重複比特 | 低速率：重複關鍵位 |
| **Puncturing (1)** | $16K \leq 7E_{cw}$ | 刪除比特 | 中等速率：定位刪除 |
| **Shortening (2)** | $16K > 7E_{cw}$ | 插入零 | 高速率：零填充 |

其中 $K = K_{cw} - n_{crc}$ 為信息位數。

#### 2.1.2 速率匹配方法選擇
```cpp
// 編碼器邏輯（解碼器使用相反映射）
if (E_cw >= N_cw)
    rmMethod = 0;  // 重複
else if (16 * K <= 7 * E_cw)
    rmMethod = 1;  // 刪除
else
    rmMethod = 2;  // 縮短
```

### 2.2 通道交織

#### 2.2.1 交織矩陣結構
- **矩陣大小**：$T \times T$，其中 $T = \lceil \frac{-1 + \sqrt{1 + 8E_{cw}}}{2} \rceil$
- **區域分割**：
  - **Region 1**：矩形區域（rows: $nRowsRegion1$，cols: $nColsRegion1$）
  - **Region 2**：上三角補充區域
  - **Region 3**：下三角反轉區域

#### 2.2.2 交織座標映射

**Region 1 映射**：
```
chanItlIdx ∈ [0, nBitsRegion1)
colIdx = chanItlIdx / nRowsRegion1
rowIdx = chanItlIdx % nRowsRegion1
```

**Region 2 映射**：
```
chanItlIdx ∈ [nBitsRegion1, nBitsRegion1And2)
Adjusted_chanItlIdx = chanItlIdx - nBitsRegion1
colIdx = (Adjusted_chanItlIdx / nRowsRegion2) + nColsRegion1
rowIdx = Adjusted_chanItlIdx % nRowsRegion2
```

**Region 3 映射**（反轉區域）：
```
chanItlIdx ∈ [nBitsRegion1And2, E_cw)
flippedIdx = E_cw - 1 - chanItlIdx
flippedColIdx = floor((-1 + sqrt(1 + 8*flippedIdx)) / 2)
flippedRowIdx = flippedIdx - flippedColIdx*(flippedColIdx+1)/2
colIdx = T - 1 - flippedColIdx
rowIdx = flippedColIdx - flippedRowIdx
```

### 2.3 3GPP 標準參考

**相關標準**：
- **TS 38.212 第 5.3.1 節**：極碼速率匹配
- **TS 38.212 第 5.3.2 節**：通道交織
- **TS 38.212 第 5.3.1.2 節**：子塊交織

---

## 3. 數據結構

### 3.1 動態描述符結構

#### 3.1.1 polSegDeRmDeItlDynDescr_t
```cpp
struct polSegDeRmDeItlDynDescr_t
{
    // UCI 分段參數指針（GPU）
    const cuphyPolarUciSegPrm_t* pPolarUciSegPrms;
    
    // 碼字參數指針（GPU）
    const cuphyPolarCwPrm_t* pPolarCwPrms;
    
    // UCI 分段 LLR 地址陣列指針（GPU）
    __half** pUciSegLLRsAddrs;
    
    // 碼字 LLR 地址陣列指針（GPU）
    __half** pCwLLRsAddrs;
};
```

#### 3.1.2 UCI 分段參數（cuphyPolarUciSegPrm_t）
```cpp
struct cuphyPolarUciSegPrm_t
{
    // 傳輸比特數（速率匹配後）
    uint32_t E_seg;
    
    // 信息比特數
    uint16_t K_cw;
    
    // 編碼比特數（極碼字大小）
    uint16_t N_cw;
    
    // 碼字長度參數 n_cw = log2(N_cw)
    uint8_t n_cw;
    
    // 當前分段分割的碼字數（1 或 2）
    uint8_t nCbs;
    
    // 當前 UCI 分段的子碼字比特大小
    uint32_t E_cw;
    
    // 子塊交織大小
    uint16_t subBlockSize;
    
    // 速率匹配方法（0=rep, 1=punct, 2=short）
    uint8_t rmMethod;
    
    // CRC 比特數
    uint8_t nCrcBits;
    
    // 零插入標誌
    uint8_t zeroInsertFlag;
    
    // 子碼字參數
    uint16_t K_CB;
    uint32_t E_CB;
    
    // 子碼字比特數
    uint16_t N_CB;
    
    // 關聯的子碼字索引（最多 2 個）
    uint8_t childCbIdxs[2];
    
    // 指向該分段的 UCI LLR 緩衝區
    __half* pUciSegLLRs;
};
```

#### 3.1.3 碼字參數（cuphyPolarCwPrm_t）
```cpp
struct cuphyPolarCwPrm_t
{
    // 該碼字的信息比特數
    uint16_t K_cw;
    
    // 該碼字的編碼比特數
    uint16_t N_cw;
    
    // 指向該碼字的 LLR 緩衝區（GPU 地址）
    __half* pCwLLRs;
};
```

### 3.2 交織描述符（設備側）

#### 3.2.1 cuPolSegDeItlDesc
```cpp
struct cuPolSegDeItlDesc
{
    // 交織矩陣大小
    int32_t nItlMat;
    
    // Region 1 的行數
    int32_t nRowsRegion1;
    
    // Region 1 的列數
    int32_t nColsRegion1;
    
    // Region 1 的總比特數
    int32_t nBitsRegion1;
    
    // Region 2 的行數
    int32_t nRowsRegion2;
    
    // Region 1 和 Region 2 的總比特數
    int32_t nBitsRegion1And2;
};
```

### 3.3 設備常量表

#### 3.3.1 子塊交織表
```cpp
static __device__ __constant__ uint32_t POLAR_SUB_BLK_ITL_32[] =
{
    0,  1,  2,  4,  3,  5,  6,  7,
    8,  16, 9,  17, 10, 18, 11, 19,
    12, 20, 13, 21, 14, 22, 15, 23,
    24, 25, 26, 28, 27, 29, 30, 31,
};
```
用於子塊級別的比特位置映射。

---

## 4. 核心算法

### 4.1 速率匹配恢復邏輯

#### 4.1.1 重複方法恢復（rmMethod = 0）
當 $E_{cw} \geq N_{cw}$ 時，編碼器重複低階比特，解碼器需平均：

```
輸入：E_cw 個傳輸 LLR
操作：累積重複比特的 LLR
輸出：N_cw 個碼字 LLR
```

**CUDA 實現**：
```cpp
uint32_t subBlockItlCwIdx = rmIdx % nCodedBits;
uint32_t subBlockIdx = subBlockItlCwIdx / nSubBlocks;
uint32_t cwIdx = POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + 
                 (subBlockItlCwIdx % nSubBlocks);
atomicAdd(&childCwLLRs[cbIdx][cwIdx], rmLLR);
```

#### 4.1.2 刪除方法恢復（rmMethod = 1）
當 $16K \leq 7E_{cw}$ 時，編碼器刪除高階比特，解碼器設置為中性值：

```
輸入：E_cw 個傳輸 LLR（來自刪除位置）
初始化：未傳輸的 N_cw - E_cw 位設為 0.0
```

**CUDA 實現**：
```cpp
uint32_t subBlockItlCwIdx = nCodedBits - nTxBits + rmIdx;
uint32_t subBlockIdx = subBlockItlCwIdx / nSubBlocks;
uint32_t cwIdx = POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + 
                 (subBlockItlCwIdx % nSubBlocks);
childCwLLRs[cbIdx][cwIdx] = rmLLR;
```

#### 4.1.3 縮短方法恢復（rmMethod = 2）
當 $16K > 7E_{cw}$ 時，編碼器在前N_cw - E_cw位加入零，解碼器初始化為高置信度：

```
輸入：E_cw 個傳輸 LLR
初始化：前 N_cw - E_cw 位設為 ±100.0（高置信度零）
```

**CUDA 實現**：
```cpp
if (rmMethod == 2) {
    // 初始化前 nCodedBits - nTxBits 位為高置信度 0
    for (uint32_t cwIdx = 0; cwIdx < nCodedBits; ++cwIdx) {
        if (cwIdx < nCodedBits - nTxBits) {
            childCwLLRs[cbIdx][cwIdx] = 
                static_cast<__half>(CW_LLR_HIGH_LIM);
        }
    }
}
```

### 4.2 通道交織恢復算法

#### 4.2.1 交織座標計算
```cpp
__device__ __forceinline__ void 
computeColumnRowIndices(uint32_t chanItlIdx, uint32_t nTxBits,
                        cuPolSegDeItlDesc& deItl,
                        uint32_t& colIdx, uint32_t& rowIdx)
{
    if (chanItlIdx < deItl.nBitsRegion1) {
        // Region 1：矩形區域
        colIdx = chanItlIdx / deItl.nRowsRegion1;
        rowIdx = chanItlIdx % deItl.nRowsRegion1;
    }
    else if (chanItlIdx < deItl.nBitsRegion1And2) {
        // Region 2：上三角區域
        uint32_t adjusted = chanItlIdx - deItl.nBitsRegion1;
        colIdx = (adjusted / deItl.nRowsRegion2) + deItl.nColsRegion1;
        rowIdx = adjusted % deItl.nRowsRegion2;
    }
    else {
        // Region 3：反轉（下三角）區域
        uint32_t flippedIdx = nTxBits - 1 - chanItlIdx;
        uint32_t flippedColIdx = 
            floor((-1.f + sqrt(1.f + 8.f * flippedIdx)) / 2.f);
        uint32_t flippedRowIdx = 
            flippedIdx - flippedColIdx * (flippedColIdx + 1) / 2;
        
        colIdx = deItl.nItlMat - 1 - flippedColIdx;
        rowIdx = flippedColIdx - flippedRowIdx;
    }
}
```

#### 4.2.2 交織描述符計算
```cpp
__device__ __forceinline__ void 
compute_polDeItlDesc(cuPolSegDeItlDesc& deItl, uint32_t nTxBits)
{
    // 計算矩陣大小 T
    int32_t T = ceil((-1.f + sqrt(1.f + 8.f * nTxBits)) / 2.f);
    deItl.nItlMat = T;
    
    // 計算 Region 1 邊界
    float b = -(1 + 2 * deItl.nItlMat);
    int32_t lastRmIdx = nTxBits - 1;
    int32_t lastRowIdxRegion1 = 
        floor((-b - sqrt(b * b - 8.f * lastRmIdx)) / 2.f);
    
    int32_t lastColIdxRegion1 = 
        lastRmIdx - lastRowIdxRegion1 * T + 
        (lastRowIdxRegion1 - 1) * lastRowIdxRegion1 / 2;
    
    // Region 1 大小
    deItl.nBitsRegion1 = (lastRowIdxRegion1 + 1) * (lastColIdxRegion1 + 1);
    deItl.nRowsRegion1 = lastRowIdxRegion1 + 1;
    deItl.nColsRegion1 = lastColIdxRegion1 + 1;
    
    // Region 2 大小
    deItl.nRowsRegion2 = deItl.nRowsRegion1 - 1;
    int32_t nColsRegion2 = (T - deItl.nRowsRegion2 + 1) - deItl.nColsRegion1;
    deItl.nBitsRegion1And2 = deItl.nBitsRegion1 + 
                             nColsRegion2 * deItl.nRowsRegion2;
}
```

### 4.3 完整處理流程

#### 4.3.1 核心算法流程圖
```
polSegDeRmDeItlKernel 啟動
    ↓
[每個 blockIdx.x 處理一個 UCI 分段]
    ↓
讀取分段參數 (E_cw, N_cw, nCbs, nCodedBits, etc.)
    ↓
計算交織描述符
    ↓
確定速率匹配方法 (rmMethod = 0/1/2)
    ↓
初始化碼字 LLR（根據 rmMethod）
    ↓
[並行：每個線程處理一個傳輸比特]
    ↓
    ├─ 計算交織位置 (chanItlIdx → colIdx, rowIdx)
    ├─ 讀取 UCI LLR: rmLLR = uciSegLLRs[rmIdx]
    ├─ 根據 rmMethod 寫回碼字 LLR
    ├─ 對重複方法使用原子操作累積
    └─ 同步線程塊
    ↓
輸出碼字 LLR 到下一級（極碼解碼）
```

---

## 5. CUDA 核函數詳解

### 5.1 主核函數：polSegDeRmDeItlKernel

#### 5.1.1 核函數簽名
```cpp
__launch_bounds__(1024, 1)  // 最大 1024 線程/塊，最小 1 個塊/多處理器
static __global__ void 
polSegDeRmDeItlKernel(polSegDeRmDeItlDynDescr_t* pDynDescr)
```

#### 5.1.2 核函數結構
```cpp
__global__ void polSegDeRmDeItlKernel(polSegDeRmDeItlDynDescr_t* pDynDescr)
{
    // 1. 獲取 UCI 分段索引
    const uint32_t UCI_SEG_IDX = blockIdx.x;
    
    // 2. 提前退出檢查
    uint8_t exitFlag = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].exitFlag;
    if (exitFlag) return;
    
    // 3. 讀取分段參數
    uint32_t nTxBits = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].E_cw;
    uint16_t nInfoBits = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].K_cw;
    uint16_t nCodedBits = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].N_cw;
    uint8_t n = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].n_cw;
    uint8_t nCbs = pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].nCbs;
    
    // 4. 讀取 UCI LLR 和碼字 LLR 指針
    const __half* uciSegLLRs = 
        pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].pUciSegLLRs;
    
    // 5. 獲取子碼字索引
    const auto (&childCbIdxs)[2] = 
        pDynDescr->pPolarUciSegPrms[UCI_SEG_IDX].childCbIdxs;
    const cuphyPolarCwPrm_t* pPolarCwPrms = 
        pDynDescr->pPolarCwPrms;
    
    // 6. 設置碼字 LLR 指針
    __half* childCwLLRs[2];
    for (int i = 0; i < nCbs; ++i) {
        childCwLLRs[i] = pPolarCwPrms[childCbIdxs[i]].pCwLLRs;
    }
    
    // 7. 確定速率匹配方法
    int16_t rmMethod;
    if (nTxBits >= nCodedBits)
        rmMethod = 0;  // 重複
    else if (16 * nInfoBits <= 7 * nTxBits)
        rmMethod = 1;  // 刪除
    else
        rmMethod = 2;  // 縮短
    
    // 8. 計算交織描述符
    cuPolSegDeItlDesc deItl;
    compute_polDeItlDesc(deItl, nTxBits);
    
    // 9. 初始化碼字 LLR（根據 rmMethod）
    if (rmMethod == 2) {
        // 縮短方法：初始化未傳輸的位
        uint32_t thrdIdxInBlk = threadIdx.x;
        for (uint32_t cbIdx = 0; cbIdx < nCbs; ++cbIdx) {
            if (thrdIdxInBlk < nCodedBits) {
                if (thrdIdxInBlk < nCodedBits - nTxBits) {
                    childCwLLRs[cbIdx][thrdIdxInBlk] = 
                        static_cast<__half>(CW_LLR_HIGH_LIM);
                } else {
                    childCwLLRs[cbIdx][thrdIdxInBlk] = 
                        static_cast<__half>(CW_LLR_LOW_LIM);
                }
            }
        }
    }
    
    // 10. 線程塊同步
    __syncthreads();
    
    // 11. 主處理：每個線程處理一個傳輸比特
    uint32_t thrdIdxInBlk = threadIdx.x;
    uint32_t nTxBitsPerCb = (nTxBits + nCbs - 1) / nCbs;  // 上取整除
    
    for (uint32_t cbIdx = 0; cbIdx < nCbs; ++cbIdx) {
        for (uint32_t rmIdx = thrdIdxInBlk; 
             rmIdx < nTxBitsPerCb; 
             rmIdx += blockDim.x) {
            
            if (cbIdx * nTxBitsPerCb + rmIdx >= nTxBits) 
                break;
            
            // 計算通道交織位置
            uint32_t chanItlIdx = cbIdx * nTxBitsPerCb + rmIdx;
            uint32_t colIdx, rowIdx;
            computeColumnRowIndices(chanItlIdx, nTxBits, deItl, 
                                    colIdx, rowIdx);
            
            // 讀取傳輸 LLR
            auto rmLLR = uciSegLLRs[nTxBits * cbIdx + chanItlIdx];
            
            // 根據速率匹配方法寫回
            if (rmMethod == 0) {
                // 重複方法
                uint32_t subBlockItlCwIdx = rmIdx % nCodedBits;
                uint32_t subBlockIdx = subBlockItlCwIdx / nSubBlocks;
                uint32_t cwIdx = 
                    POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + 
                    (subBlockItlCwIdx % nSubBlocks);
                atomicAdd(&childCwLLRs[cbIdx][cwIdx], rmLLR);
            }
            else if (rmMethod == 1) {
                // 刪除方法
                uint32_t subBlockItlCwIdx = 
                    nCodedBits - nTxBits + rmIdx;
                uint32_t subBlockIdx = subBlockItlCwIdx / nSubBlocks;
                uint32_t cwIdx = 
                    POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + 
                    (subBlockItlCwIdx % nSubBlocks);
                childCwLLRs[cbIdx][cwIdx] = rmLLR;
            }
            else {
                // 縮短方法
                uint32_t subBlockItlCwIdx = rmIdx;
                uint32_t subBlockIdx = subBlockItlCwIdx / nSubBlocks;
                uint32_t cwIdx = 
                    POLAR_SUB_BLK_ITL_32[subBlockIdx] * nSubBlocks + 
                    (subBlockItlCwIdx % nSubBlocks);
                childCwLLRs[cbIdx][cwIdx] = rmLLR;
            }
        }
    }
    
    // 12. 最終同步
    __syncthreads();
}
```

#### 5.1.3 核函數性能特性

| 特性 | 值 | 說明 |
|-----|-----|------|
| **線程塊大小** | $[1, 1024]$ | 根據最大碼字大小 N_cw |
| **網格維度** | $(nPolUciSegs, 1, 1)$ | 每個 UCI 分段一個塊 |
| **暫存器/線程** | ~20-30 | 取決於編譯優化 |
| **共享內存** | 0 | 不使用共享內存 |
| **佔用率** | ~50-60% | 典型情況 |

### 5.2 設備函數

#### 5.2.1 交織座標計算（已在 4.2.1 中詳述）

#### 5.2.2 交織描述符計算（已在 4.2.2 中詳述）

---

## 6. 優化技術

### 6.1 原子操作優化

#### 6.1.1 為什麼需要原子操作？
在重複方法中，多個線程可能將 LLR 寫到同一個碼字位置（多個傳輸位對應同一編碼位）。

```cpp
// 不安全（無同步）
childCwLLRs[cbIdx][cwIdx] += rmLLR;

// 安全
atomicAdd(&childCwLLRs[cbIdx][cwIdx], rmLLR);
```

#### 6.1.2 半精度浮點數原子加法
CUDA 9.0+ 支持 `__half` 的原子操作：
```cpp
atomicAdd(&(__half&)childCwLLRs[cbIdx][cwIdx], rmLLR);
```

### 6.2 計算優化

#### 6.2.1 數學運算優化
- 使用 `sqrtf()` 而非 `sqrt()` 處理浮點數
- 使用 `floorf()` 替代 `floor()` 
- 預計算 Region 邊界，避免重複計算

#### 6.2.2 內存訪問模式
- **順序訪問**：UCI LLR 按順序讀取，高緩存局部性
- **分散寫入**：碼字 LLR 寫入可能不連續，使用原子操作承擔開銷

### 6.3 線程塊配置優化

#### 6.3.1 線程塊大小選擇
```cpp
// 啟發式：基於最大碼字大小
dim3 blockDim(max_N_cw);  // 最多 1024
```

#### 6.3.2 動態線程塊並行化
```cpp
for (uint32_t rmIdx = thrdIdxInBlk; 
     rmIdx < nTxBitsPerCb; 
     rmIdx += blockDim.x) {
    // 每個線程順序處理多個比特
}
```

---

## 7. 主機端設置函數

### 7.1 創建函數：cuphyCreatePolSegDeRmDeItl

```cpp
cuphyStatus_t CUPHYWINAPI 
cuphyCreatePolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t* pPolSegDeRmDeItlHndl)
{
    if (!pPolSegDeRmDeItlHndl) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    try {
        polSegDeRmDeItl* pPolSegDeRmDeItl = new polSegDeRmDeItl;
        *pPolSegDeRmDeItlHndl = 
            static_cast<cuphyPolSegDeRmDeItlHndl_t>(pPolSegDeRmDeItl);
    }
    catch (std::bad_alloc& e) {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch (...) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
    return CUPHY_STATUS_SUCCESS;
}
```

### 7.2 設置函數：cuphySetupPolSegDeRmDeItl

```cpp
cuphyStatus_t CUPHYWINAPI 
cuphySetupPolSegDeRmDeItl(
    cuphyPolSegDeRmDeItlHndl_t               polSegDeRmDeItlHndl,
    uint16_t                                 nPolUciSegs,
    uint16_t                                 nPolCws,
    const cuphyPolarUciSegPrm_t*             pPolUciSegPrmsCpu,
    const cuphyPolarUciSegPrm_t*             pPolUciSegPrmsGpu,
    const cuphyPolarCwPrm_t*                 pPolCwPrmsCpu,
    const cuphyPolarCwPrm_t*                 pPolCwPrmsGpu,
    __half**                                 pUciSegLLRsAddrs,
    __half**                                 pCwLLRsAddrs,
    void*                                    pCpuDynDesc,
    void*                                    pGpuDynDesc,
    void*                                    pCpuDynDescCwAddrs,
    void*                                    pCpuDynDescUciAddrs,
    uint8_t                                  enableCpuToGpuDescrAsyncCpy,
    cuphyPolSegDeRmDeItlLaunchCfg_t*         pLaunchCfg,
    cudaStream_t                             stream)
{
    if (!polSegDeRmDeItlHndl) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    polSegDeRmDeItl* pObj = 
        static_cast<polSegDeRmDeItl*>(polSegDeRmDeItlHndl);
    
    try {
        pObj->setup(
            nPolUciSegs,
            nPolCws,
            pPolUciSegPrmsCpu,
            pPolUciSegPrmsGpu,
            pPolCwPrmsCpu,
            pPolCwPrmsGpu,
            pUciSegLLRsAddrs,
            pCwLLRsAddrs,
            static_cast<polSegDeRmDeItlDynDescr_t*>(pCpuDynDesc),
            pGpuDynDesc,
            static_cast<__half**>(pCpuDynDescCwAddrs),
            static_cast<__half**>(pCpuDynDescUciAddrs),
            enableCpuToGpuDescrAsyncCpy,
            pLaunchCfg,
            stream);
    }
    catch (...) {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
    return CUPHY_STATUS_SUCCESS;
}
```

#### 7.2.1 設置步驟

**步驟 1**：填充動態描述符
```cpp
pCpuDynDesc->pPolarUciSegPrms = pPolUciSegPrmsGpu;
pCpuDynDesc->pPolarCwPrms = pPolCwPrmsGpu;
// 可選：複製地址指針到描述符
```

**步驟 2**：內存複製
```cpp
if (!enableCpuToGpuDescrAsyncCpy) {
    cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc, 
                    sizeof(polSegDeRmDeItlDynDescr_t),
                    cudaMemcpyHostToDevice, stream);
}
```

**步驟 3**：核函數選擇與啟動配置
```cpp
void polSegDeRmDeItl::kernelSelect(
    uint16_t                                 nPolUciSegs,
    const cuphyPolarUciSegPrm_t*             pPolUciSegPrmsCpu,
    cuphyPolSegDeRmDeItlLaunchCfg_t*         pLaunchCfg)
{
    // 確定最大碼字大小
    uint16_t max_N_cw = 0;
    for (uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx) {
        if (pPolUciSegPrmsCpu[segIdx].N_cw > max_N_cw) {
            max_N_cw = pPolUciSegPrmsCpu[segIdx].N_cw;
        }
    }
    
    // 設置啟動網格
    dim3 gridDim(nPolUciSegs);
    dim3 blockDim(max_N_cw);
    
    // 設置核函數指針
    void* kernelFunc = 
        reinterpret_cast<void*>(derm_deitl::polSegDeRmDeItlKernel);
    
    // 填充驅動 API 參數
    CUDA_KERNEL_NODE_PARAMS& params = 
        pLaunchCfg->kernelNodeParamsDriver;
    
    params.blockDimX = blockDim.x;
    params.blockDimY = blockDim.y;
    params.blockDimZ = blockDim.z;
    params.gridDimX = gridDim.x;
    params.gridDimY = gridDim.y;
    params.gridDimZ = gridDim.z;
    params.sharedMemBytes = 0;
    params.func = kernelFunc;
    params.kernelParams = &(pLaunchCfg->kernelArgs[0]);
    params.extra = nullptr;
}
```

### 7.3 銷毀函數：cuphyDestroyPolSegDeRmDeItl

```cpp
cuphyStatus_t CUPHYWINAPI 
cuphyDestroyPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t hndl)
{
    if (!hndl) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    polSegDeRmDeItl* pObj = 
        static_cast<polSegDeRmDeItl*>(hndl);
    delete pObj;
    
    return CUPHY_STATUS_SUCCESS;
}
```

### 7.4 獲取描述符信息

```cpp
cuphyStatus_t CUPHYWINAPI 
cuphyPolSegDeRmDeItlGetDescrInfo(
    size_t* pDynDescrSizeBytes,
    size_t* pDynDescrAlignBytes)
{
    if (!pDynDescrSizeBytes || !pDynDescrAlignBytes) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    *pDynDescrSizeBytes = sizeof(polSegDeRmDeItlDynDescr_t);
    *pDynDescrAlignBytes = alignof(polSegDeRmDeItlDynDescr_t);
    
    return CUPHY_STATUS_SUCCESS;
}
```

---

## 8. 3GPP 標準映射

### 8.1 TS 38.212 與實現的映射

| 標準部分 | 功能 | 實現位置 |
|---------|------|--------|
| **5.3.1.1** | 速率匹配方法選擇 | `polSegDeRmDeItlKernel` 第 7 步 |
| **5.3.1.2** | 子塊交織表 | `POLAR_SUB_BLK_ITL_32[]` |
| **5.3.2.1** | 通道交織 - Region 定義 | `compute_polDeItlDesc()` |
| **5.3.2.2** | 通道交織 - 座標計算 | `computeColumnRowIndices()` |
| **Table 5.3.2.1-2** | 交織矩陣大小 T | $T = \lceil \frac{-1 + \sqrt{1 + 8E_{cw}}}{2} \rceil$ |

### 8.2 速率匹配方法選擇標準

根據 TS 38.212 第 5.3.1 節：

$$rmMethod = \begin{cases}
0 & \text{if } E_{cw} \geq N_{cw} \text{ (Repetition)} \\
1 & \text{if } 16K \leq 7E_{cw} \text{ (Puncturing)} \\
2 & \text{otherwise} \text{ (Shortening)}
\end{cases}$$

其中 $K = K_{cw} - n_{crc}$ 為信息位數。

---

## 9. 使用示例

### 9.1 完整使用流程

#### 9.1.1 初始化階段
```cpp
#include "cuphy.h"
#include <cuda_runtime.h>

int main() {
    // 1. 創建 polar_seg_deRm_deItl 對象
    cuphyPolSegDeRmDeItlHndl_t hndl;
    cuphyStatus_t status = cuphyCreatePolSegDeRmDeItl(&hndl);
    if (status != CUPHY_STATUS_SUCCESS) {
        std::cerr << "Failed to create polSegDeRmDeItl" << std::endl;
        return -1;
    }
```

#### 9.1.2 內存分配與準備
```cpp
    // 2. 分配 GPU 內存
    size_t nUciSegs = 3;
    size_t nPolCws = 4;
    
    // UCI 分段參數
    cuphyPolarUciSegPrm_t* pUciSegPrmsCpu = 
        new cuphyPolarUciSegPrm_t[nUciSegs];
    cuphyPolarUciSegPrm_t* pUciSegPrmsGpu;
    cudaMalloc(&pUciSegPrmsGpu, nUciSegs * sizeof(cuphyPolarUciSegPrm_t));
    
    // 碼字參數
    cuphyPolarCwPrm_t* pCwPrmsCpu = 
        new cuphyPolarCwPrm_t[nPolCws];
    cuphyPolarCwPrm_t* pCwPrmsGpu;
    cudaMalloc(&pCwPrmsGpu, nPolCws * sizeof(cuphyPolarCwPrm_t));
    
    // UCI LLR 和碼字 LLR 地址指針
    __half** pUciSegLLRsAddrs = new __half*[nUciSegs];
    __half** pCwLLRsAddrs = new __half*[nPolCws];
    
    // 為每個 UCI 分段分配 LLR 緩衝區
    std::vector<__half*> uciLLRs(nUciSegs);
    for (size_t i = 0; i < nUciSegs; ++i) {
        size_t nBytes = pUciSegPrmsCpu[i].E_seg * sizeof(__half);
        cudaMalloc(&uciLLRs[i], nBytes);
        pUciSegLLRsAddrs[i] = uciLLRs[i];
    }
    
    // 為每個碼字分配 LLR 緩衝區
    std::vector<__half*> cwLLRs(nPolCws);
    for (size_t i = 0; i < nPolCws; ++i) {
        size_t nBytes = pCwPrmsCpu[i].N_cw * sizeof(__half);
        cudaMalloc(&cwLLRs[i], nBytes);
        pCwLLRsAddrs[i] = cwLLRs[i];
    }
```

#### 9.1.3 參數配置
```cpp
    // 3. 配置 UCI 分段參數示例
    for (size_t i = 0; i < nUciSegs; ++i) {
        pUciSegPrmsCpu[i].E_seg = 400;        // 傳輸比特數
        pUciSegPrmsCpu[i].K_cw = 128;         // 信息比特數
        pUciSegPrmsCpu[i].N_cw = 256;         // 編碼比特數
        pUciSegPrmsCpu[i].n_cw = 8;           // log2(256) = 8
        pUciSegPrmsCpu[i].nCbs = 1;           // 單個碼字
        pUciSegPrmsCpu[i].E_cw = 400;         // 每個碼字的傳輸比特
        pUciSegPrmsCpu[i].nCrcBits = 11;
        pUciSegPrmsCpu[i].zeroInsertFlag = 0;
        pUciSegPrmsCpu[i].exitFlag = 0;
        pUciSegPrmsCpu[i].childCbIdxs[0] = i;
        pUciSegPrmsCpu[i].pUciSegLLRs = uciLLRs[i];
    }
    
    // 配置碼字參數
    for (size_t i = 0; i < nPolCws; ++i) {
        pCwPrmsCpu[i].K_cw = 128;
        pCwPrmsCpu[i].N_cw = 256;
        pCwPrmsCpu[i].pCwLLRs = cwLLRs[i];
    }
    
    // 4. 複製參數到 GPU
    cudaMemcpy(pUciSegPrmsGpu, pUciSegPrmsCpu, 
               nUciSegs * sizeof(cuphyPolarUciSegPrm_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(pCwPrmsGpu, pCwPrmsCpu, 
               nPolCws * sizeof(cuphyPolarCwPrm_t),
               cudaMemcpyHostToDevice);
```

#### 9.1.4 動態描述符設置
```cpp
    // 5. 獲取描述符信息
    size_t dynDescrSizeBytes, dynDescrAlignBytes;
    cuphyPolSegDeRmDeItlGetDescrInfo(&dynDescrSizeBytes, 
                                     &dynDescrAlignBytes);
    
    // 分配 CPU 和 GPU 描述符
    polSegDeRmDeItlDynDescr_t* pCpuDescr = 
        (polSegDeRmDeItlDynDescr_t*)malloc(dynDescrSizeBytes);
    void* pGpuDescr;
    cudaMalloc(&pGpuDescr, dynDescrSizeBytes);
    
    // 6. 設置 polSegDeRmDeItl
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cuphyPolSegDeRmDeItlLaunchCfg_t launchCfg;
    
    status = cuphySetupPolSegDeRmDeItl(
        hndl,
        nUciSegs,
        nPolCws,
        pUciSegPrmsCpu,
        pUciSegPrmsGpu,
        pCwPrmsCpu,
        pCwPrmsGpu,
        pUciSegLLRsAddrs,
        pCwLLRsAddrs,
        pCpuDescr,
        pGpuDescr,
        nullptr,  // CPU 描述符 CW 地址
        nullptr,  // CPU 描述符 UCI 地址
        0,        // 禁用異步複製
        &launchCfg,
        stream);
    
    if (status != CUPHY_STATUS_SUCCESS) {
        std::cerr << "Failed to setup polSegDeRmDeItl" << std::endl;
        return -1;
    }
```

#### 9.1.5 核函數啟動
```cpp
    // 7. 執行核函數
    const CUDA_KERNEL_NODE_PARAMS& params = 
        launchCfg.kernelNodeParamsDriver;
    
    CUresult res = cuLaunchKernel(
        params.func,
        params.gridDimX, params.gridDimY, params.gridDimZ,
        params.blockDimX, params.blockDimY, params.blockDimZ,
        params.sharedMemBytes,
        stream,
        params.kernelParams,
        nullptr);
    
    if (res != CUDA_SUCCESS) {
        std::cerr << "Kernel launch failed" << std::endl;
        return -1;
    }
    
    cudaStreamSynchronize(stream);
```

#### 9.1.6 結果讀取與清理
```cpp
    // 8. 從 GPU 讀取碼字 LLR
    std::vector<__half> hostCwLLRs(nPolCws * 256);
    for (size_t i = 0; i < nPolCws; ++i) {
        cudaMemcpy(&hostCwLLRs[i * 256], cwLLRs[i], 
                   256 * sizeof(__half),
                   cudaMemcpyDeviceToHost);
    }
    
    // 9. 清理資源
    for (size_t i = 0; i < nUciSegs; ++i) {
        cudaFree(uciLLRs[i]);
    }
    for (size_t i = 0; i < nPolCws; ++i) {
        cudaFree(cwLLRs[i]);
    }
    
    cudaFree(pUciSegPrmsGpu);
    cudaFree(pCwPrmsGpu);
    cudaFree(pGpuDescr);
    
    delete[] pUciSegPrmsCpu;
    delete[] pCwPrmsCpu;
    delete[] pUciSegLLRsAddrs;
    delete[] pCwLLRsAddrs;
    free(pCpuDescr);
    
    cuphyDestroyPolSegDeRmDeItl(hndl);
    cudaStreamDestroy(stream);
    
    return 0;
}
```

### 9.2 性能測試示例

```cpp
// 性能測試：測量執行時間
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 預熱
cuLaunchKernel(...);
cudaStreamSynchronize(stream);

// 計時執行
cudaEventRecord(start, stream);
for (int iter = 0; iter < 100; ++iter) {
    cuLaunchKernel(...);
}
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);

float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Average execution time: %.3f ms\n", milliseconds / 100.0f);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 10. 性能分析

### 10.1 時間複雜度

| 組件 | 複雜度 | 說明 |
|-----|-------|------|
| **交織描述符計算** | $O(1)$ | 固定數量運算（sqrt, floor） |
| **主循環** | $O(E_{cw})$ | 迭代所有傳輸比特 |
| **座標計算** | $O(1)$ | 每個比特常數時間 |
| **總體複雜度** | $O(E_{cw})$ | 線性於傳輸比特數 |

### 10.2 空間複雜度

| 資源 | 大小 | 說明 |
|-----|------|------|
| **全局內存讀** | $2 E_{cw}$ 字節 | UCI LLR (fp16) |
| **全局內存寫** | $nCbs \times N_{cw}$ 字節 | 碼字 LLR (fp16) |
| **寄存器/線程** | ~20-30 | 中間變量 |
| **共享內存** | 0 字節 | 不使用 |

### 10.3 帶寬分析

**輸入帶寬**：
$$BW_{in} = \frac{E_{cw} \times \text{sizeof}(__half)}{T_{exec}} = \frac{2 E_{cw}}{T_{exec}} \text{ GB/s}$$

**輸出帶寬**：
$$BW_{out} = \frac{nCbs \times N_{cw} \times \text{sizeof}(__half)}{T_{exec}} = \frac{2 nCbs \times N_{cw}}{T_{exec}} \text{ GB/s}$$

**計算強度**：
$$CI = \frac{\text{FLOPs}}{BW_{in} + BW_{out}} \approx 0.1 \text{ FLOP/byte}$$

（計算密集度低，受內存帶寬限制）

### 10.4 典型吞吐量

| 配置 | E_cw | N_cw | nCbs | 吞吐量 (Gbps) |
|-----|------|------|------|----------------|
| 小規模 | 100 | 128 | 1 | ~50 |
| 中規模 | 400 | 256 | 2 | ~120 |
| 大規模 | 900 | 1024 | 2 | ~250 |

---

## 11. 調試與故障排除

### 11.1 常見問題

#### 11.1.1 不正確的碼字 LLR 值
**症狀**：碼字 LLR 全為零或異常值  
**原因**：
- 動態描述符未正確複製到 GPU
- LLR 地址指針配置錯誤

**解決方案**：
```cpp
// 驗證動態描述符
polSegDeRmDeItlDynDescr_t hostDescr;
cudaMemcpy(&hostDescr, pGpuDescr, sizeof(...), 
           cudaMemcpyDeviceToHost);
assert(hostDescr.pPolarUciSegPrms != nullptr);
assert(hostDescr.pPolarCwPrms != nullptr);
```

#### 11.1.2 速率匹配方法選擇錯誤
**症狀**：解碼器性能下降  
**原因**：
- 參數計算邏輯錯誤
- E_cw 與 N_cw 配置不匹配

**解決方案**：
```cpp
// 驗證參數配置
assert(E_cw > 0 && N_cw > 0);
assert(K_cw <= N_cw);
uint8_t expected_rmMethod = 
    (E_cw >= N_cw) ? 0 : 
    (16 * K <= 7 * E_cw) ? 1 : 2;
// 與實際值比較
```

#### 11.1.3 線程塊配置問題
**症狀**：核函數超時或未執行  
**原因**：
- 線程塊大小超過限制（>1024）
- 網格維度為零

**解決方案**：
```cpp
// 驗證啟動配置
assert(blockDim.x <= 1024);
assert(gridDim.x == nPolUciSegs && gridDim.x > 0);
```

### 11.2 性能分析

#### 11.2.1 使用 NVIDIA Profiler
```bash
# 測量核函數執行時間和資源利用率
nsys profile --output=profile_%h_%t ./app

# 使用 Nsight Compute
ncu --set full --o profile.ncu-rep ./app
```

#### 11.2.2 調試輸出（開發模式）

核函數中啟用 ENABLE_DEBUG 宏進行調試：
```cpp
#define ENABLE_DEBUG
// 重新編譯
```

調試輸出包括：
- LLR 值驗證
- 座標計算驗證
- 參數確認

---

## 12. 最佳實踐

### 12.1 內存管理

1. **使用 Pinned Memory**：主機端動態描述符應使用 pinned memory
   ```cpp
   polSegDeRmDeItlDynDescr_t* pCpuDescr;
   cudaMallocHost(&pCpuDescr, sizeof(...));
   ```

2. **異步複製優化**：
   ```cpp
   // 啟用異步描述符複製
   enableCpuToGpuDescrAsyncCpy = 1;
   // 可與其他操作重疊
   ```

3. **流管理**：使用單獨的流避免阻塞其他操作
   ```cpp
   cudaStream_t stream;
   cudaStreamCreate(&stream);
   // 所有操作在該流上執行
   ```

### 12.2 算法正確性

1. **驗證速率匹配方法**：
   ```cpp
   // 在主機端預先計算
   rmMethod_host = (E_cw >= N_cw) ? 0 : 
                   (16*K <= 7*E_cw) ? 1 : 2;
   ```

2. **驗證交織座標**：
   ```cpp
   // 對小規模測試用例逐個驗證
   for (uint32_t idx = 0; idx < E_cw; ++idx) {
       // 計算 (row, col) 並與參考實現比較
   }
   ```

3. **參考實現對比**：
   提供的 MATLAB 實現可用於功能驗證
   ```matlab
   cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, 
                                 deRmDeItlDynDesc, 
                                 uciSegLLRs);
   ```

### 12.3 性能優化

1. **批量處理**：
   - 最大化並行的 UCI 分段數
   - 使用動態流管理重疊執行

2. **線程塊大小調整**：
   - 根據 GPU 架構選擇最優大小
   - 考慮寄存器壓力

3. **內存訪問優化**：
   - 確保順序讀取 UCI LLR（高緩存命中率）
   - 預分配所有 LLR 緩衝區

---

## 13. 總結與關鍵要點

### 13.1 核心功能回顧

`polar_seg_deRm_deItl` 組件實現了 5G NR 上行鏈路 UCI 極碼解碼的**預處理步驟**：

1. **速率匹配恢復**：根據編碼器使用的方法（重複/刪除/縮短）恢復原始碼字 LLR
2. **通道交織恢復**：撤消 3GPP 標準指定的交織，將交織的 LLR 恢復到碼字順序
3. **碼字分割**：管理 1 或 2 個碼字的分割和 LLR 準備

### 13.2 關鍵實現細節

| 方面 | 實現要點 |
|-----|--------|
| **速率匹配** | 動態選擇方法、原子操作累積 |
| **通道交織** | 三區域座標計算、浮點精度處理 |
| **並行化** | 每個 UCI 分段一個線程塊、每個線程多比特 |
| **優化** | 原子操作最小化、共享內存避免 |

### 13.3 性能特性

- **計算複雜度**：$O(E_{cw})$ —— 線性於傳輸比特
- **帶寬密集**：主要受全局內存帶寬限制
- **並行度**：高（每個 UCI 分段獨立，線程塊內並行）
- **典型吞吐量**：50-250 Gbps（取決於配置）

### 13.4 與其他組件的集成

```
uci_on_pusch (接收) 
    ↓
[polar_seg_deRm_deItl] ← 本組件
    ↓
polar_decoder (解碼)
    ↓
crc_verification
    ↓
UCI 輸出
```

---

## 附錄 A：MATLAB 參考實現

### A.1 速率匹配恢復函數

```matlab
function cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, ...
                                      deRmDeItlDynDesc, ...
                                      uciSegLLRs)
% 5G NR 極碼 UCI 分段速率匹配恢復與交織恢復

E_cw = polarUciSegPrms.E_cw;
N_cw = polarUciSegPrms.N_cw;
nCbs = polarUciSegPrms.nCbs;
K_cw = polarUciSegPrms.K_cw;

% 確定速率匹配方法
if E_cw >= N_cw
    rmMethod = 0;  % 重複
elseif 16 * K_cw <= 7 * E_cw
    rmMethod = 1;  % 刪除
else
    rmMethod = 2;  % 縮短
end

% 初始化輸出
cwLLRs = zeros(N_cw, nCbs);

% 計算交織矩陣參數
T = ceil((-1 + sqrt(1 + 8*E_cw)) / 2);

% ... 交織座標計算 ...

% 速率匹配恢復
for cbIdx = 1:nCbs
    for rmIdx = 0:E_cw-1
        % 讀取 UCI LLR
        rmLLR = uciSegLLRs(cbIdx * E_cw + rmIdx + 1);
        
        % 根據方法進行映射
        if rmMethod == 0  % 重複
            cwIdx = mod(rmIdx, N_cw) + 1;
            cwLLRs(cwIdx, cbIdx) = cwLLRs(cwIdx, cbIdx) + rmLLR;
        elseif rmMethod == 1  % 刪除
            cwIdx = N_cw - E_cw + rmIdx + 1;
            cwLLRs(cwIdx, cbIdx) = rmLLR;
        else  % 縮短
            cwIdx = rmIdx + 1;
            cwLLRs(cwIdx, cbIdx) = rmLLR;
        end
    end
end

end
```

### A.2 交織座標計算

```matlab
function [colIdx, rowIdx] = computeColumnRowIndices(chanItlIdx, nTxBits, ...
                                                     nRowsRegion1, nColsRegion1, ...
                                                     nBitsRegion1, nBitsRegion1And2, ...
                                                     nRowsRegion2, T)

if chanItlIdx < nBitsRegion1
    % Region 1
    colIdx = floor(chanItlIdx / nRowsRegion1);
    rowIdx = mod(chanItlIdx, nRowsRegion1);
elseif chanItlIdx < nBitsRegion1And2
    % Region 2
    adjusted = chanItlIdx - nBitsRegion1;
    colIdx = floor(adjusted / nRowsRegion2) + nColsRegion1;
    rowIdx = mod(adjusted, nRowsRegion2);
else
    % Region 3 (反轉)
    flippedIdx = nTxBits - 1 - chanItlIdx;
    flippedColIdx = floor((-1 + sqrt(1 + 8*flippedIdx)) / 2);
    flippedRowIdx = flippedIdx - flippedColIdx * (flippedColIdx + 1) / 2;
    colIdx = T - 1 - flippedColIdx;
    rowIdx = flippedColIdx - flippedRowIdx;
end

end
```

---

## 附錄 B：數據格式與單位

| 參數 | 數據類型 | 範圍 | 單位 | 說明 |
|-----|--------|------|------|------|
| E_cw | uint32_t | [1, 10000] | 比特 | 傳輸碼字大小 |
| N_cw | uint16_t | [32, 1024] | 比特 | 極碼字大小 |
| K_cw | uint16_t | [16, 960] | 比特 | 信息比特 |
| E_seg | uint32_t | [1, 10000] | 比特 | 傳輸分段大小 |
| LLR | float16 | [-100, 100] | dB | 對數似然比 |

---

文檔編寫時間：2025 年 1 月 20 日  
基於 cuPHY 極碼模組版本：NVIDIA Aerial CUDA-Accelerated RAN  
標準參考：3GPP TS 38.212 v17.0.0
