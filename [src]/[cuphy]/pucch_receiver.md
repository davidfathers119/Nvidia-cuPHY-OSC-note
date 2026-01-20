# PUCCH Receiver (PucchRx)

## 1. 概述 (Overview)

**PUCCH Receiver (PucchRx)** 是NVIDIA cuPHY中用於物理上行控制信號 (Physical Uplink Control Channel) 接收的主要管道 (pipeline)。

### 主要功能：
- **多格式支持**: PUCCH Format 0, 1, 2, 3 (F0, F1, F2, F3)
- **UCI解碼**: 上行控制信息 (Uplink Control Information) - HARQ-ACK, SR (調度請求), CSI (通道狀態信息)
- **多小區處理**: 支持單個時隙中多個小區的聚合處理
- **管道架構**: F0/F1/F2/F3 并行接收 → 極化碼率匹配 → 極化解碼 → UCI輸出
- **性能指標**: SINR, RSSI, RSRP, 干擾功率, 噪聲方差, 定時超前 (TA)

### 關鍵特性：
- **並行處理**: 最多18個PUCCH UCI可同時處理
- **CUDA圖形支持**: 支持CUDA Graph模式 (圖形執行) 和流模式 (Stream模式)
- **動態配置**: 每個時隙可配置不同的UCI參數
- **內存優化**: 預分配内存，避免碎片化

---

## 2. 架構與組件 (Architecture & Components)

### 2.1 管道組件結構

```
PUCCH RX Pipeline Components:
├─ PUCCH_CELL_INFO (0)          → 小區信息緩衝
├─ PUCCH_F0_Rx (1)               → PUCCH Format 0 接收器
├─ PUCCH_F1_Rx (2)               → PUCCH Format 1 接收器
├─ PUCCH_F2_RX (3)               → PUCCH Format 2 前端
├─ PUCCH_F3_RX (4)               → PUCCH Format 3 前端
├─ PUCCH_F234_UCI_SEG (5)        → PUCCH F234 UCI分段器
├─ POL_COMP_CW_TREE (6)          → 極化CW樹類型
├─ POL_COMP_CW_TREE_ADDRS (7)    → 樹地址
├─ POL_SEG_DERM_DEITL (8)        → 極化分段De-Rm De-Itl
├─ POL_SEG_DERM_DEITL_CW_ADDRS (9)
├─ POL_SEG_DERM_DEITL_UCI_ADDRS (10)
├─ POL_DECODE (11)               → 極化解碼器
├─ POL_DECODE_LLR_ADDRS (12)
├─ POL_DECODE_CB_ADDRS (13)
├─ LIST_POL_DECODE_SCRATCH_ADDRS (14)
├─ RM_DECODE (15)                → 速率匹配解碼器
└─ [16]                          → 總組件數 = N_PUCCH_COMPONENTS
```

### 2.2 核心數據結構

#### PucchRx 類
- **角色**: PUCCH接收管道的主要類
- **繼承**: 從 `cuphyPucchRx` (抽象接口) 繼承
- **主要成員**:
  - 子組件句柄 (F0/F1/F2/F3 RX, 極化解碼器等)
  - 內存緩衝區 (LLR, 性能指標)
  - CUDA圖形執行對象
  - 內核啟動配置

#### 重要參數結構
- `cuphyPucchStatPrms_t`: 靜態參數 (小區配置, 天線數等)
- `cuphyPucchDynPrms_t`: 動態參數 (每時隙配置)
- `cuphyPucchCellGrpDynPrm_t`: 小區組動態參數

---

## 3. 主要Function Call 生命周期

### 3.1 創建 (Creation)

```cpp
// 函數簽名
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchRx(
    cuphyPucchRxHndl_t* pPucchRxHndl,        // [OUT] 返回的PUCCH RX句柄
    cuphyPucchStatPrms_t const* pStatPrms,   // [IN] 靜態參數
    cudaStream_t cuStream                    // [IN] CUDA流
);

// 用途
// 創建新的PucchRx管道對象
// 分配內部資源、初始化組件、配置內存
// 驗證參數有效性

// 返回碼
// CUPHY_STATUS_SUCCESS           - 成功
// CUPHY_STATUS_INVALID_ARGUMENT  - 無效輸入 (NULL句柄或參數)
// CUPHY_STATUS_ALLOC_FAILED      - 內存分配失敗
// CUPHY_STATUS_INTERNAL_ERROR    - 內部錯誤

// 示例
cuphyPucchRxHndl_t pucchRxHndl = nullptr;
cuphyStatus_t status = cuphyCreatePucchRx(
    &pucchRxHndl,
    &pucchStatPrms,
    cuStream
);
if (status != CUPHY_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create PUCCH RX");
}
```

**內部操作**:
1. 驗證輸入指針非NULL
2. 創建PucchRx類實例
3. 調用構造函數初始化:
   - 分配描述符緩衝區
   - 創建子組件 (F0, F1, F2, F3 RX)
   - 初始化內存分配器
   - 配置向量 (LLR緩衝區地址, 性能指標)
4. 返回句柄

---

### 3.2 設置 (Setup)

```cpp
// 函數簽名
cuphyStatus_t CUPHYWINAPI cuphySetupPucchRx(
    cuphyPucchRxHndl_t pucchRxHndl,              // [IN] PUCCH RX句柄
    cuphyPucchDynPrms_t* pDynPrms,               // [IN] 動態參數
    cuphyPucchBatchPrmHndl_t const batchPrmHndl // [IN] 批次參數句柄
);

// 用途
// 在每個時隙開始前設置管道
// 配置UCI參數、內存指針、內核啟動參數
// 構建CUDA圖或流模式執行配置

// 返回碼
// CUPHY_STATUS_SUCCESS           - 成功
// CUPHY_STATUS_INVALID_ARGUMENT  - 無效句柄或參數

// 示例
cuphyPucchDynPrms_t dynPrms;
dynPrms.cuStream = cuStream;
dynPrms.cpuCopyOn = 1;  // 立即複製輸出到CPU

cuphyStatus_t setupStatus = cuphySetupPucchRx(
    pucchRxHndl,
    &dynPrms,
    batchPrmHndl
);
```

**內部操作**:
1. 驗證輸入有效性
2. 調用 `PucchRx::setup()` 方法
3. 為各個組件調用 `setupComponents()`:
   - `cuphySetupPucchF0Rx()` - 如果有F0 UCI
   - `cuphySetupPucchF1Rx()` - 如果有F1 UCI
   - `cuphySetupPucchF2Rx()` - 如果有F2 UCI
   - `cuphySetupPucchF3Rx()` - 如果有F3 UCI
   - 極化編碼器/解碼器設置
4. 構建動態描述符
5. 填充內核啟動參數 (CUDA_KERNEL_NODE_PARAMS)
6. 創建或更新CUDA圖

---

### 3.3 運行 (Execution)

```cpp
// 函數簽名
cuphyStatus_t CUPHYWINAPI cuphyRunPucchRx(
    cuphyPucchRxHndl_t pucchRxHndl,  // [IN] PUCCH RX句柄
    uint64_t procModeBmsk             // [IN] 處理模式位掩碼 (目前未使用)
);

// 用途
// 執行PUCCH接收管道
// 啟動所有內核（CUDA圖或流模式）
// 處理所有UCI、解碼、性能指標計算

// 返回碼
// CUPHY_STATUS_SUCCESS           - 成功
// CUPHY_STATUS_INVALID_ARGUMENT  - 無效句柄
// CUPHY_STATUS_INTERNAL_ERROR    - 內核執行失敗

// 示例
uint64_t procMode = 0;  // 標準模式
cuphyStatus_t runStatus = cuphyRunPucchRx(pucchRxHndl, procMode);
if (runStatus != CUPHY_STATUS_SUCCESS) {
    throw std::runtime_error("PUCCH RX execution failed");
}
```

**內部流程** (CUDA圖模式):
```
cuGraphLaunch(m_graphExec, m_cuStream)
  ├─ F0 RX內核 (如果n_F0Ucis > 0)
  ├─ F1 RX內核 (如果n_F1Ucis > 0)
  ├─ F2 RX內核 (如果n_F2Ucis > 0)
  ├─ F3 RX內核 (如果n_F3Ucis > 0)
  ├─ 率匹配解碼器內核
  ├─ 計算CW樹類型內核
  ├─ 極化分段De-Rm De-Itl內核
  └─ 極化解碼器內核 (可能多個列表)
```

**內部流程** (流模式):
```
為每個組件:
  CUresult = cuLaunchKernel(
    func,
    gridDim.x, gridDim.y, gridDim.z,
    blockDim.x, blockDim.y, blockDim.z,
    sharedMem,
    cuStream,
    kernelParams,
    extra
  )
```

---

### 3.4 輸出複製 (Optional)

```cpp
// 函數簽名 (內部調用)
cuphyStatus_t PucchRx::copyOutputToCPU();

// 用途
// 將GPU端的UCI輸出複製到CPU端內存
// 性能指標複製 (SINR, RSSI等)
// CRC標誌複製

// 條件執行
if (pDynPrms->cpuCopyOn) {
    copyOutputToCPU();  // 異步或同步複製
}
```

---

### 3.5 銷毀 (Destruction)

```cpp
// 函數簽名
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchRx(
    cuphyPucchRxHndl_t pucchRxHndl  // [IN] PUCCH RX句柄
);

// 用途
// 釋放所有資源、銷毀組件、清理GPU內存
// 銷毀CUDA圖對象

// 返回碼
// CUPHY_STATUS_SUCCESS           - 成功
// CUPHY_STATUS_INVALID_ARGUMENT  - 無效句柄

// 示例
cuphyStatus_t destroyStatus = cuphyDestroyPucchRx(pucchRxHndl);
if (destroyStatus != CUPHY_STATUS_SUCCESS) {
    NVLOG_ERR("Failed to destroy PUCCH RX");
}
```

**內部操作**:
1. 調用 `destroyComponents()`:
   - `cuphyDestroyPucchF0Rx()`
   - `cuphyDestroyPucchF1Rx()`
   - `cuphyDestroyPucchF2Rx()`
   - `cuphyDestroyPucchF3Rx()`
   - `cuphyDestroyPucchF234UciSeg()`
   - `cuphyDestroyCompCwTreeTypes()`
   - `cuphyDestroyPolSegDeRmDeItl()`
   - `cuphyDestroyPolarDecoder()`
   - `cuphyDestroyRmDecoder()`
2. 銷毀CUDA圖: `cudaGraphDestroy()`, `cudaGraphExecDestroy()`
3. 釋放GPU/CPU內存
4. 刪除PucchRx類實例

---

## 4. 子組件 Function Calls

### 4.1 PUCCH F0 接收器

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF0Rx(
    cuphyPucchF0RxHndl_t* pPucchF0RxHndl,
    cudaStream_t strm
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF0Rx(
    cuphyPucchF0RxHndl_t pucchF0RxHndl,
    cuphyTensorPrm_t* pDataRx,
    cuphyPucchF0F1UciOut_t* pF0UcisOut,
    uint16_t nCells,
    uint16_t nF0Ucis,
    uint8_t enableUlRxBf,
    cuphyPucchUciPrm_t* pF0UciPrms,
    cuphyPucchCellPrm_t* pCmnCellPrms,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    void* pCpuDynDesc,
    void* pGpuDynDesc,
    cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF0Rx(
    cuphyPucchF0RxHndl_t pucchF0RxHndl
);
```

### 4.2 PUCCH F1 接收器

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF1Rx(
    cuphyPucchF1RxHndl_t* pPucchF1RxHndl,
    cudaStream_t strm
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF1Rx(
    cuphyPucchF1RxHndl_t pucchF1RxHndl,
    cuphyTensorPrm_t* pDataRx,
    cuphyPucchF0F1UciOut_t* pF1UcisOut,
    uint16_t nCells,
    uint16_t nF1Ucis,
    uint8_t enableUlRxBf,
    cuphyPucchUciPrm_t* pF1UciPrms,
    cuphyPucchCellPrm_t* pCmnCellPrms,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    void* pCpuDynDesc,
    void* pGpuDynDesc,
    cuphyPucchF1RxLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF1Rx(
    cuphyPucchF1RxHndl_t pucchF1RxHndl
);
```

### 4.3 PUCCH F2 前端

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF2Rx(
    cuphyPucchF2RxHndl_t* pPucchF2RxHndl,
    cudaStream_t strm
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF2Rx(
    cuphyPucchF2RxHndl_t pucchF2RxHndl,
    cuphyTensorPrm_t* pDataRx,
    __half** pDescramLLRaddrs,
    uint8_t* pDTXflags,
    float* pSinr,
    float* pRssi,
    float* pRsrp,
    float* pInterf,
    float* pNoiseVar,
    float* pTaEst,
    uint16_t nCells,
    uint16_t nF2Ucis,
    uint8_t enableUlRxBf,
    cuphyPucchUciPrm_t* pF2UciPrms,
    cuphyPucchCellPrm_t* pCmnCellPrms,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    void* pCpuDynDesc,
    void* pGpuDynDesc,
    cuphyPucchF2RxLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF2Rx(
    cuphyPucchF2RxHndl_t pucchF2RxHndl
);
```

### 4.4 PUCCH F3 前端

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Rx(
    cuphyPucchF3RxHndl_t* pPucchF3RxHndl,
    cudaStream_t strm
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Rx(
    cuphyPucchF3RxHndl_t pucchF3RxHndl,
    cuphyTensorPrm_t* pDataRx,
    __half** pDescramLLRaddrs,
    uint8_t* pDTXflags,
    float* pSinr,
    float* pRssi,
    float* pRsrp,
    float* pInterf,
    float* pNoiseVar,
    float* pTaEst,
    uint16_t nCells,
    uint16_t nF3Ucis,
    uint8_t enableUlRxBf,
    cuphyPucchUciPrm_t* pF3UciPrms,
    cuphyPucchCellPrm_t* pCmnCellPrms,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    void* pCpuDynDesc,
    void* pGpuDynDesc,
    cuphyPucchF3RxLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Rx(
    cuphyPucchF3RxHndl_t pucchF3RxHndl
);
```

### 4.5 PUCCH F234 UCI 分段器

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t* pPucchF234UciSegHndl
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t pucchF234UciSegHndl,
    // ... 各式各樣的參數
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t pucchF234UciSegHndl
);
```

### 4.6 極化解碼器相關

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreatePolarDecoder(
    cuphyPolarDecoderHndl_t* pPolarDecoderHndl
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupPolarDecoder(
    cuphyPolarDecoderHndl_t polarDecoderHndl,
    // ... 參數
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolarDecoder(
    cuphyPolarDecoderHndl_t polarDecoderHndl
);
```

### 4.7 率匹配解碼器

```cpp
// 創建
cuphyStatus_t CUPHYWINAPI cuphyCreateRmDecoder(
    CUcontext cuContext,
    cuphyRmDecoderHndl_t* pRmDecoderHndl,
    unsigned int rmFlags,
    size_t* pMemoryFootprint
);

// 設置
cuphyStatus_t CUPHYWINAPI cuphySetupRmDecoder(
    cuphyRmDecoderHndl_t rmDecoderHndl,
    // ... 參數
);

// 銷毀
cuphyStatus_t CUPHYWINAPI cuphyDestroyRmDecoder(
    cuphyRmDecoderHndl_t rmDecoderHndl
);
```

---

## 5. 關鍵內部方法

### 5.1 PucchRx 類內部方法

```cpp
// 構造函數
PucchRx(cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t strm);

// 析構函數
~PucchRx();

// 設置 (核心方法)
[[nodiscard]] cuphyStatus_t setup(cuphyPucchDynPrms_t *pDynPrm);

// 運行 (執行管道)
[[nodiscard]] cuphyStatus_t run();

// 輸出複製
cuphyStatus_t copyOutputToCPU();

// 調試
void writeDbgBufSynch(cudaStream_t cuStream);

// 內部組件管理
void createComponents();
void setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPucchDynPrms_t *pDynPrm);
void destroyComponents();

// CUDA圖管理
void createGraph();
void updateGraph();

// 內存相關
cuphyStatus_t allocateDescr();
cuphyStatus_t allocateDeviceMemory();

// 速率匹配相關
F234RmSizes_t compRateMatchSizesF2(cuphyPucchUciPrm_t& F2uciPrms);
F234RmSizes_t compRateMatchSizesF3(cuphyPucchUciPrm_t& F3uciPrms);
```

### 5.2 圖構建方法

```cpp
// 創建完整的CUDA圖
void PucchRx::createGraph();
// - 創建root節點
// - 添加所有組件的內核節點
// - 設置依賴關係
// - 編譯圖為可執行形式

// 更新CUDA圖 (性能優化)
void PucchRx::updateGraph();
// - 根據當前UCI數量啟用/禁用節點
// - 無需重新創建圖，只更新節點狀態
```

---

## 6. 完整使用流程示例

```cpp
#include <cuphy.h>
#include <cuda_runtime.h>

int main() {
    // 初始化CUDA
    cudaStream_t cuStream;
    cudaStreamCreate(&cuStream);
    
    // 第1步: 準備靜態參數
    cuphyPucchStatPrms_t statPrms = {...};  // 小區配置、天線數等
    
    // 第2步: 創建PUCCH RX
    cuphyPucchRxHndl_t pucchRxHndl = nullptr;
    cuphyStatus_t status = cuphyCreatePucchRx(
        &pucchRxHndl,
        &statPrms,
        cuStream
    );
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Create failed");
    }
    
    // 第3步: 為每個時隙循環
    for (int slot = 0; slot < numSlots; ++slot) {
        // 準備動態參數
        cuphyPucchDynPrms_t dynPrms = {...};  // UCI參數、內存指針等
        dynPrms.cuStream = cuStream;
        dynPrms.cpuCopyOn = 1;
        
        // 第4步: 設置管道
        cuphyStatus_t setupStatus = cuphySetupPucchRx(
            pucchRxHndl,
            &dynPrms,
            batchPrmHndl
        );
        if (setupStatus != CUPHY_STATUS_SUCCESS) {
            throw std::runtime_error("Setup failed");
        }
        
        // 第5步: 執行管道
        cuphyStatus_t runStatus = cuphyRunPucchRx(pucchRxHndl, 0);
        if (runStatus != CUPHY_STATUS_SUCCESS) {
            throw std::runtime_error("Run failed");
        }
        
        // 第6步: 同步流並處理輸出
        cudaStreamSynchronize(cuStream);
        // 輸出已複製到CPU (如果 cpuCopyOn=1)
    }
    
    // 第7步: 銷毀
    cuphyStatus_t destroyStatus = cuphyDestroyPucchRx(pucchRxHndl);
    if (destroyStatus != CUPHY_STATUS_SUCCESS) {
        NVLOG_ERR("Destroy failed");
    }
    
    // 清理CUDA
    cudaStreamDestroy(cuStream);
    return 0;
}
```

---

## 7. 性能特性

| 指標 | 值 |
|------|-----|
| **最大並行UCI** | 18個/時隙 |
| **支持的格式** | F0 (1符號), F1 (2-7符號), F2 (1-2符號), F3 (4-14符號) |
| **內核數量** | 16+ (依配置而定) |
| **CUDA圖優化** | 支持 (顯著降低主機開銷) |
| **流並發** | 支持多流執行 |
| **記憶體預分配** | 最大化性能，避免動態分配 |

---

## 8. 錯誤處理

常見錯誤碼及處理：

| 錯誤碼 | 含義 | 解決方法 |
|-------|------|---------|
| `CUPHY_STATUS_INVALID_ARGUMENT` | 無效參數 (NULL指針等) | 檢查輸入參數有效性 |
| `CUPHY_STATUS_ALLOC_FAILED` | GPU內存分配失敗 | 檢查可用GPU內存 |
| `CUPHY_STATUS_INTERNAL_ERROR` | 內核執行失敗 | 檢查CUDA錯誤、設備狀態 |

```cpp
// 推薦的錯誤處理模式
if (status != CUPHY_STATUS_SUCCESS) {
    const char* errName = cuphyGetErrorName(status);
    const char* errMsg = cuphyGetErrorString(status);
    NVLOG_ERR("Error: {} - {}", errName, errMsg);
    // 清理資源
    cuphyDestroyPucchRx(pucchRxHndl);
    return 1;
}
```

---

## 9. 相關文件位置

- **主文件**: `cuPHY/src/cuphy_channels/pucch_rx.hpp`, `pucch_rx.cpp`
- **API定義**: `cuPHY/src/cuphy/cuphy_api.h`
- **子組件**: `cuPHY/src/cuphy/pucch_F0_receiver/`, `pucch_F1_receiver/`, `pucch_F2_front_end/`, `pucch_F3_front_end/`
- **示例**: `cuPHY/examples/pucch_rx_pipeline/`

---

## 10. 總結

PUCCH Receiver是一個複雜的多組件管道，通過精心設計的Function Call序列實現高效的上行控制信號接收：

1. **創建** → 初始化資源
2. **設置** → 配置參數和內存
3. **運行** → 執行所有內核
4. **輸出複製** (可選) → 將結果轉移到CPU
5. **銷毀** → 釋放資源

通過支持CUDA圖、並行UCI處理和流並發，PucchRx實現了低延遲、高吞吐量的5G NR上行控制信息解碼。
