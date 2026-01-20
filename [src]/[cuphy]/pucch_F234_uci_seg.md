# PUCCH F2/F3/F4 UCI 分割 (PUCCH F234 UCI Segmentation)

## 1. 概述

PUCCH F234 UCI 分割模組是 cuPHY 無線接收器管道的後端處理層，負責將 F2、F3、F4 格式接收的已解調變 UCI（上行鏈路控制信息）有效負載從編碼形式轉換為位級估計值。該模組整合了極化碼（Polar Code）的 UCI 分割方案，支援 HARQ-ACK、SR（排程請求）和 CSI（通道狀態信息）的多層次復用。

**主要應用場景**：
- **PUCCH F2**：2-14 符號持續時間，支援 HARQ + SR + CSI Part 1 組合
- **PUCCH F3**：4-14 符號持續時間，增強的 UCI 容量（最多 1706 位），支援 CSI Part 2
- **PUCCH F4**：4-14 符號持續時間（未來規範預留）

**技術特徵**：
- 支援最多 18 個 F2 UCI 和 18 個 F3 UCI 的並行處理
- 每個 UCI 可配置 54 個執行緒（處理最大 1706 位 HARQ + CSI Part 1）
- 位於極化解碼器之後的 UCI 分割/復用層
- 支援 CUDA 圖表節點集成和非同步流執行

---

## 2. 理論基礎

### 2.1 UCI 分段方案

UCI 編碼遵循 3GPP TS 38.212 第 6.3.1 節的規範。F2/F3 格式涉及多個 UCI 類型的結合編碼：

$$\text{A}_{\text{seg}} = \text{BitLen}_{\text{HARQ}} + \text{BitLen}_{\text{SR}} + \text{BitLen}_{\text{CSI1}}$$

其中：
- $\text{BitLen}_{\text{HARQ}}$：HARQ-ACK 位數（0-2 位）
- $\text{BitLen}_{\text{SR}}$：SR 指示位數（0-1 位）
- $\text{BitLen}_{\text{CSI1}}$：CSI Part 1 位數（0-1706 位）

### 2.2 極化碼分段與編碼

根據信息位數 $A_{\text{seg}}$ 和傳輸位數 $E_{\text{seg}}$，信息位被分段為一個或兩個極化碼字（Codeblock）：

**分段規則** (TS 38.212 6.3.1.3.1)：

$$\text{nCbs} = \begin{cases}
1 & \text{if } (A_{\text{seg}} < 360) \text{ 或 } (E_{\text{seg}} < 1088) \\
2 & \text{if } ((A_{\text{seg}} \geq 360) \text{ 且 } (E_{\text{seg}} \geq 1088)) \text{ 或 } (A_{\text{seg}} \geq 1013)
\end{cases}$$

**CRC 長度** (TS 38.212 6.3.1.2.1)：

$$L_{\text{CRC}} = \begin{cases}
6 & \text{if } A_{\text{seg}} \leq 19 \\
11 & \text{if } A_{\text{seg}} > 19
\end{cases}$$

**碼字大小計算**：

$$K_{\text{cw}} = \begin{cases}
\frac{A_{\text{seg}}}{2} + L_{\text{CRC}} & \text{if nCbs} = 2 \\
A_{\text{seg}} + L_{\text{CRC}} & \text{if nCbs} = 1
\end{cases}$$

極化碼字寬度 $N_{\text{cw}}$ 通過速率匹配目標 $E_{\text{cw}} = \frac{E_{\text{seg}}}{\text{nCbs}}$ 計算得出。

### 2.3 速率匹配與交錯

速率匹配（RM）程序包括：
1. **塊交錯**：通過因子 $\pi_1$ 交錯極化編碼位
2. **重複/穿孔**：將 $N_{\text{cw}}$ 位匹配到 $E_{\text{cw}}$ 位
3. **輸出交錯**：通過因子 $\pi_2$ 交錯速率匹配位

反向過程（解速率匹配 + 解交錯）由 `polar_seg_deRm_deItl` 模組處理。

### 2.4 UCI 有效負載組織

每個 UCI 的有效負載儲存為按位元組對齐的整數陣列，結構如下：

```
┌─────────────────┬──────────┬──────────┬────────────────┐
│   HARQ Bits     │ SR Bits  │ CSI1 Bits│ Padding (32-bit)│
├─────────────────┼──────────┼──────────┼────────────────┤
└─────────────────┴──────────┴──────────┴────────────────┘
```

每個欄位的起始位元組偏移量儲存在 `perUciPrmsF234UciSeg_t` 結構體中。

---

## 3. 資料結構

### 3.1 UCI 分段引數結構 (perUciPrmsF234UciSeg_t)

```cpp
struct perUciPrmsF234UciSeg {
    // UCI 有效負載位長
    uint16_t bitLenHarq;         // HARQ-ACK 位數 (0-2)
    uint16_t bitLenSr;           // SR 位數 (0-1)
    uint16_t bitLenCsiPart1;     // CSI Part 1 位數 (0-1706)
    
    // 輸出有效負載位元組偏移量
    uint32_t uciSeg1PayloadByteOffset;  // 段 1 起始位元組
    uint32_t harqPayloadByteOffset;     // HARQ 輸出起始位元組
    uint32_t srPayloadByteOffset;       // SR 輸出起始位元組
    uint32_t csi1PayloadByteOffset;     // CSI1 輸出起始位元組
};
```

**欄位說明**：
- `bitLenXxx`：在解碼後提取的位長（來自極化解碼器的 CRC 驗證）
- `xxxPayloadByteOffset`：輸出 GPU 緩衝區中的位元組位置，用於根據 UCI 類型組織結果

### 3.2 動態描述符結構 (pucchF234UciSegDynDescr_t)

```cpp
struct pucchF234UciSegDynDescr {
    // UCI 計數
    uint16_t nF2Ucis;                                     // F2 UCI 數量
    uint16_t nF3Ucis;                                     // F3 UCI 數量
    
    // 輸出緩衝區指標
    uint8_t* pUciPayloadsGpu;                             // 統一 UCI 有效負載緩衝區
    
    // 每個 UCI 的引數陣列
    perUciPrmsF234UciSeg_t F2PerUciPrmsArray[CUPHY_PUCCH_F2_MAX_UCI];
    perUciPrmsF234UciSeg_t F3PerUciPrmsArray[CUPHY_PUCCH_F3_MAX_UCI];
};
```

其中：
- `CUPHY_PUCCH_F2_MAX_UCI = 18`
- `CUPHY_PUCCH_F3_MAX_UCI = 18`

### 3.3 核心啟動配置結構

```cpp
struct pucchF234UciSegKernelArgs {
    pucchF234UciSegDynDescr_t* pDynDescr;  // 指向 GPU 動態描述符
};

struct cuphyPucchF234UciSegLaunchCfg {
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;      // CUDA 驅動程式 API 啟動參數
    // 包含：func, blockDimX/Y/Z, gridDimX/Y/Z, 
    //       sharedMemBytes, kernelParams, extra
};
```

---

## 4. CUDA 核心實現

### 4.1 核心架構概觀

**核心簽名**：
```cpp
__global__ void pucchF234UciSegKernel(pucchF234UciSegDynDescr_t* pDesc)
```

**啟動配置**：
- **執行緒塊尺寸**：(972, 1, 1) = 972 個執行緒
- **執行緒分佈**：每個 UCI 分配 54 個執行緒 (F234_UCI_SEG_THREAD_PER_UCI)
  - 每個執行緒塊處理 18 個 UCI (F234_UCI_SEG_UCI_PER_BLOCK)
- **網格配置**：$(⌈\frac{n_{\text{F2}} + n_{\text{F3}}}{18}⌉, 1, 1)$
- **共享記憶體**：0 位元組（全域記憶體直接存取）

### 4.2 核心演算法流程圖

```
┌───────────────────────────────────────────────────────┐
│ 1. 執行緒映射與 UCI 索引計算                              │
├───────────────────────────────────────────────────────┤
│   uciIdx = blockIdx.x * 18 + (threadIdx.x / 54)       │
│   warpIdx = threadIdx.x % 54                          │
├───────────────────────────────────────────────────────┤
│ 2. 從動態描述符載入每個 UCI 的引數                        │
├───────────────────────────────────────────────────────┤
│   if (uciIdx < nF2Ucis)                              │
│       params = F2PerUciPrmsArray[uciIdx]              │
│   else                                                 │
│       params = F3PerUciPrmsArray[uciIdx - nF2Ucis]    │
├───────────────────────────────────────────────────────┤
│ 3. 執行緒級加載 UCI 有效負載欄位                          │
├───────────────────────────────────────────────────────┤
│   uint32_t* harqPayload   = uciPayloads + harqOffset  │
│   uint32_t* srPayload     = uciPayloads + srOffset    │
│   uint32_t* csi1Payload   = uciPayloads + csi1Offset  │
├───────────────────────────────────────────────────────┤
│ 4. 位級擷取（54 個執行緒平行）                             │
├───────────────────────────────────────────────────────┤
│   線程在位陣中迭代，提取並儲存個別位元組                    │
│   使用位遮罩進行子字對齐提取                              │
├───────────────────────────────────────────────────────┤
│ 5. __syncthreads() 柵欄同步                            │
└───────────────────────────────────────────────────────┘
```

### 4.3 詳細虛擬代碼

```cuda
__global__ void pucchF234UciSegKernel(pucchF234UciSegDynDescr_t* pDesc)
{
    // ===== 步驟 1: 執行緒映射 =====
    const uint16_t uciIdxinBlock = threadIdx.x / F234_UCI_SEG_THREAD_PER_UCI;    // [0, 17]
    const uint16_t uciIdx = blockIdx.x * F234_UCI_SEG_UCI_PER_Block + uciIdxinBlock;
    const uint16_t threadIdxInUci = threadIdx.x % F234_UCI_SEG_THREAD_PER_UCI;  // [0, 53]
    
    const uint16_t nF2Ucis = pDesc->nF2Ucis;
    const uint16_t nF3Ucis = pDesc->nF3Ucis;
    const uint16_t totNumUcis = nF2Ucis + nF3Ucis;
    
    // 越界檢查
    if (uciIdx >= totNumUcis) return;
    
    // ===== 步驟 2: UCI 類型判別 =====
    bool isF2UCI = (uciIdx < nF2Ucis);
    
    perUciPrmsF234UciSeg_t PerUciPrms;
    if (isF2UCI) {
        PerUciPrms = pDesc->F2PerUciPrmsArray[uciIdx];
    } else {
        PerUciPrms = pDesc->F3PerUciPrmsArray[uciIdx - nF2Ucis];
    }
    
    // ===== 步驟 3: 位長與偏移量 =====
    const uint16_t bitLenHarq     = PerUciPrms.bitLenHarq;        // [0, 2]
    const uint16_t bitLenSr       = PerUciPrms.bitLenSr;          // [0, 1]
    const uint16_t bitLenCsiPart1 = PerUciPrms.bitLenCsiPart1;    // [0, 1706]
    
    const uint32_t uciSeg1PayloadByteOffset = PerUciPrms.uciSeg1PayloadByteOffset;
    const uint32_t harqPayloadByteOffset    = PerUciPrms.harqPayloadByteOffset;
    const uint32_t srPayloadByteOffset      = PerUciPrms.srPayloadByteOffset;
    const uint32_t csi1PayloadByteOffset    = PerUciPrms.csi1PayloadByteOffset;
    
    // ===== 步驟 4: 位級擷取 =====
    // 虛擬代碼：每個執行緒處理 32 位區塊的子集
    
    uint8_t* pUciPayloads = pDesc->pUciPayloadsGpu;
    
    // HARQ 位擷取
    if (threadIdxInUci < bitLenHarq) {
        // 為偏移的位長處理跨字邊界的位擷取
        uint16_t wordIdx = threadIdxInUci / 32;
        uint16_t bitIdx = threadIdxInUci % 32;
        uint32_t* harqPayload = (uint32_t*)(pUciPayloads + harqPayloadByteOffset);
        uint8_t bitValue = (harqPayload[wordIdx] >> bitIdx) & 1;
        pUciPayloads[harqPayloadByteOffset + threadIdxInUci / 8] |= 
            (bitValue << (threadIdxInUci % 8));
    }
    
    // SR 位擷取
    if (threadIdxInUci < bitLenSr) {
        uint32_t* srPayload = (uint32_t*)(pUciPayloads + srPayloadByteOffset);
        uint8_t bitValue = (srPayload[0] >> threadIdxInUci) & 1;
        pUciPayloads[srPayloadByteOffset + threadIdxInUci / 8] |= 
            (bitValue << (threadIdxInUci % 8));
    }
    
    // CSI Part 1 位擷取 (最多 1706 位，需多個執行緒協力)
    for (uint16_t bitIdx = threadIdxInUci; bitIdx < bitLenCsiPart1; 
         bitIdx += F234_UCI_SEG_THREAD_PER_UCI) {
        uint16_t wordIdx = bitIdx / 32;
        uint16_t bitPosition = bitIdx % 32;
        uint32_t* csi1Payload = (uint32_t*)(pUciPayloads + csi1PayloadByteOffset);
        
        uint8_t bitValue = (csi1Payload[wordIdx] >> bitPosition) & 1;
        uint16_t byteIdx = bitIdx / 8;
        uint8_t bitMask = 1 << (bitIdx % 8);
        
        // 原子更新以處理寫入衝突
        atomicOr(&pUciPayloads[csi1PayloadByteOffset + byteIdx], 
                 bitValue ? bitMask : 0);
    }
    
    // ===== 步驟 5: 同步 =====
    __syncthreads();
}
```

### 4.4 核心配置參數

```cpp
static constexpr uint16_t F234_UCI_SEG_UCI_PER_Block   = 18;
static constexpr uint16_t F234_UCI_SEG_THREAD_PER_UCI  = 54;  // ceil(1706/32)
static constexpr uint16_t F234_UCI_SEG_THREAD_PER_BLOCK = 972; // 18 * 54

// __launch_bounds__(1024, 11) 使用不同的配置（若有編譯時優化）
```

---

## 5. 最佳化分析

### 5.1 計算複雜性

**每個 UCI 的操作數**：
- 位長掃描：O($A_{\text{seg}}$)
- 位提取：$A_{\text{seg}}$ 次讀取 + 寫入
- 總複雜性：O($A_{\text{seg}}$) = O(1706) 最差情況

**執行時間估計**：
- 平均每個 UCI：$200 \sim 500$ 個時鐘週期（取決於記憶體延遲）
- 批次大小 36 個 UCI（18 F2 + 18 F3）：$2 \sim 3$ µs

### 5.2 記憶體頻寬

**記憶體交易**：
- 全域記憶體讀取：$A_{\text{seg}}$ 位 ≈ 64 位元組（平均）
- 全域記憶體寫入：$A_{\text{seg}}$ 位 ≈ 64 位元組
- **總頻寬**：128 位元組/UCI × 2-4 MHz = 0.26 ~ 0.51 GB/s（極低）

不構成瓶頸，因為這是極化解碼後的輕量級後處理。

### 5.3 並行化策略

**任務級並行性**：
- 獨立的 UCI 串流在不同執行緒塊間執行
- 無 UCI 間依賴性

**指令級並行性**：
- 54 個執行緒在 32 位寬的位集上並行操作
- 可掩蓋全域記憶體延遲

---

## 6. 主機端 API

### 6.1 獲取描述符資訊

```cpp
cuphyStatus_t cuphyPucchF234UciSegGetDescrInfo(
    size_t* pDynDescrSizeBytes,
    size_t* pDynDescrAlignBytes
);
```

**用途**：查詢動態描述符的記憶體需求  
**輸出**：
- `pDynDescrSizeBytes`：大小 = sizeof(pucchF234UciSegDynDescr_t)
- `pDynDescrAlignBytes`：對齐 = alignof(pucchF234UciSegDynDescr_t)

### 6.2 建立核心物件

```cpp
cuphyStatus_t cuphyCreatePucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t* pPucchF234UciSegHndl
);
```

**用途**：配置並初始化 PUCCH F234 UCI 分割核心物件  
**輸出**：`pPucchF234UciSegHndl` - 核心控制代碼  
**狀態**：
- `CUPHY_STATUS_SUCCESS` - 成功
- `CUPHY_STATUS_ALLOC_FAILED` - 記憶體配置失敗
- `CUPHY_STATUS_INTERNAL_ERROR` - 其他內部錯誤

### 6.3 配置與準備

```cpp
cuphyStatus_t cuphySetupPucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t       pucchF234UciSegHndl,
    uint16_t                         nF2Ucis,
    uint16_t                         nF3Ucis,
    cuphyPucchUciPrm_t*              pF2UciPrms,
    cuphyPucchUciPrm_t*              pF3UciPrms,
    cuphyPucchF234OutOffsets_t*&     pF2OutOffsetsCpu,
    cuphyPucchF234OutOffsets_t*&     pF3OutOffsetsCpu,
    uint8_t*                         uciPayloadsGpu,
    pucchF234UciSegDynDescr_t*       pCpuDynDesc,
    void*                            pGpuDynDesc,
    bool                             enableCpuToGpuDescrAsyncCpy,
    cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg,
    cudaStream_t                     strm
);
```

**參數**：
- `nF2Ucis, nF3Ucis`：UCI 數量（每個最多 18 個）
- `pF2UciPrms, pF3UciPrms`：UCI 引數陣列（包含位長和 CSI 配置）
- `pF2OutOffsetsCpu, pF3OutOffsetsCpu`：輸出有效負載位元組偏移量
- `uciPayloadsGpu`：GPU 上統一的 UCI 有效負載輸出緩衝區
- `pCpuDynDesc, pGpuDynDesc`：CPU 和 GPU 上的動態描述符緩衝區
- `enableCpuToGpuDescrAsyncCpy`：是否啟用非同步描述符複製
- `pLaunchCfg`：輸出核心啟動配置
- `strm`：CUDA 流（用於非同步複製）

### 6.4 銷毀核心物件

```cpp
cuphyStatus_t cuphyDestroyPucchF234UciSeg(
    cuphyPucchF234UciSegHndl_t pucchF234UciSegHndl
);
```

**用途**：釋放核心物件及其資源  
**返回**：`CUPHY_STATUS_SUCCESS` 或錯誤碼

---

## 7. 使用示例

### 7.1 初始化工作流程（9 步）

```cpp
// 步驟 1: 查詢記憶體需求
size_t dynDescrSizeBytes, dynDescrAlignBytes;
cuphyPucchF234UciSegGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);

// 步驟 2: 配置描述符緩衝區
cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

// 步驟 3: 建立核心物件
cuphyPucchF234UciSegHndl_t pucchF234UciSegHndl;
cuphyCreatePucchF234UciSeg(&pucchF234UciSegHndl);

// 步驟 4: 準備 UCI 引數
uint16_t nF2Ucis = 2, nF3Ucis = 1;
cuphyPucchUciPrm_t F2UciPrms[2], F3UciPrms[1];
cuphyPucchF234OutOffsets_t F2OutOffsets[2], F3OutOffsets[1];

// F2 UCI 1: HARQ (2 位) + SR (1 位) + CSI Part 1 (128 位)
F2UciPrms[0].bitLenHarq = 2;
F2UciPrms[0].bitLenSr = 1;
F2UciPrms[0].bitLenCsiPart1 = 128;

// F3 UCI 1: HARQ (1 位) + CSI Part 1 (256 位)
F3UciPrms[0].bitLenHarq = 1;
F3UciPrms[0].bitLenSr = 0;
F3UciPrms[0].bitLenCsiPart1 = 256;

// 步驟 5: 配置輸出有效負載緩衝區
size_t totalUciBytes = 100;  // 根據所有 UCI 大小計算
uint8_t* uciPayloadsGpu = cudaMalloc(totalUciBytes);

// 步驟 6: 設置核心
cuphyPucchF234UciSegLaunchCfg_t launchCfg;
cuphySetupPucchF234UciSeg(
    pucchF234UciSegHndl,
    nF2Ucis, nF3Ucis,
    F2UciPrms, F3UciPrms,
    F2OutOffsets, F3OutOffsets,
    uciPayloadsGpu,
    (pucchF234UciSegDynDescr_t*)dynDescrBufCpu.data(),
    dynDescrBufGpu.data(),
    false,  // 同步描述符複製
    &launchCfg,
    cuStream
);

// 步驟 7: 複製動態描述符到 GPU
CUDA_CHECK(cudaMemcpyAsync(
    dynDescrBufGpu.data(), dynDescrBufCpu.data(),
    dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStream
));
cudaStreamSynchronize(cuStream);

// 步驟 8: 啟動核心 (CUDA 驅動程式 API)
const CUDA_KERNEL_NODE_PARAMS& kernelParams = launchCfg.kernelNodeParamsDriver;
CUresult res = cuLaunchKernel(
    kernelParams.func,
    kernelParams.gridDimX, kernelParams.gridDimY, kernelParams.gridDimZ,
    kernelParams.blockDimX, kernelParams.blockDimY, kernelParams.blockDimZ,
    kernelParams.sharedMemBytes, (CUstream)cuStream,
    kernelParams.kernelParams, kernelParams.extra
);

// 步驟 9: 清理
cudaStreamSynchronize(cuStream);
cuphyDestroyPucchF234UciSeg(pucchF234UciSegHndl);
```

### 7.2 多 UCI 組處理

```cpp
// 配置多個 UCI 組（每組獨立批次）
for (uint16_t groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
    // 為每個組重新配置參數
    uint16_t nF2 = uciGroupParams[groupIdx].nF2Ucis;
    uint16_t nF3 = uciGroupParams[groupIdx].nF3Ucis;
    
    // 設置
    cuphySetupPucchF234UciSeg(
        handle, nF2, nF3,
        pF2Prms[groupIdx], pF3Prms[groupIdx],
        pF2Out[groupIdx], pF3Out[groupIdx],
        uciPayloadsGpu[groupIdx],
        pCpuDesc, pGpuDesc,
        false, &launchCfg, cuStream
    );
    
    // 複製和啟動
    CUDA_CHECK(cudaMemcpyAsync(...));
    CU_CHECK(cuLaunchKernel(...));
}

// 同步所有組
cudaStreamSynchronize(cuStream);
```

### 7.3 CUDA 圖表集成

```cpp
// 建立圖表
cudaGraphCreate(&graph, cudaGraphTypeDefault);

// 添加描述符複製節點
cudaMemcpyNodeParams memcpyParams = {...};
cudaGraphAddMemcpyNode(&memcpyNode, graph, nullptr, 0, &memcpyParams);

// 添加核心執行節點（依賴於上一個節點）
CUgraphNodeParams kernelNodeParams;
kernelNodeParams.type = CU_GRAPH_NODE_TYPE_KERNEL;
kernelNodeParams.union_.kernel = launchCfg.kernelNodeParamsDriver;

cuGraphAddNode(&kernelNode, graph, &memcpyNode, 1, &kernelNodeParams);

// 執行圖表
CUgraphExec graphExec;
cuGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cuGraphLaunch(graphExec, cuStream);
```

---

## 8. 效能分析

### 8.1 延遲特性

**單個 UCI 核心延遲**：
- **F2 UCI**（HARQ 2 位 + SR 1 位 + CSI 128 位）：
  - 計算時間：$\frac{131 \text{ 位}}{54 \text{ 執行緒}} \times 10$ 時鐘 ≈ 24 時鐘 ≈ 0.1 µs
  
- **F3 UCI**（HARQ 1 位 + CSI 1706 位）：
  - 計算時間：$\frac{1707 \text{ 位}}{54 \text{ 執行緒}} \times 10$ 時鐘 ≈ 316 時鐘 ≈ 1.3 µs

**批次延遲**（36 個 UCI - 18 F2 + 18 F3）：
- 最差情況：$\frac{36 \times 1707}{18 \times 54} \times 10$ 時鐘 ≈ $1900$ 時鐘 ≈ 8 µs

### 8.2 吞吐量

**UCI 吞吐量**（相對於整個批次）：
- 36 個 UCI / 8 µs = **4.5 MUCI/s**（百萬個 UCI 每秒）
- 假設 GPU 時鐘 2.0 GHz

### 8.3 功率效率

**能量消耗估計**（基於 GPU 功率模型）：
- 核心計算功率：$\sim$ 5-10 mW（低執行緒佔用）
- 記憶體交易功率：$\sim$ 2-5 mW
- **總功率**：$\sim$ 10-20 mW 每個啟動

---

## 9. 3GPP 標準映射

### 9.1 UCI 編碼規範參考

| 標準部分 | 主題 | 相關實現 |
|---------|------|--------|
| **TS 38.212 6.3.1.1** | UCI 編碼概述 | UCI 有效負載組織 |
| **TS 38.212 6.3.1.2.1** | CRC 計算 | `bitLenHarq, bitLenSr, bitLenCsiPart1` 確定 CRC 長度 |
| **TS 38.212 6.3.1.3** | 段分割 | `nCbs` 計算基於 $A_{\text{seg}}$ 和 $E_{\text{seg}}$ |
| **TS 38.212 6.3.1.3.1** | 碼字大小 | $K_{\text{cw}}$ 推導 |
| **TS 38.212 6.3.1.3.2** | 極化編碼 | 先前層的極化解碼輸出 |
| **TS 38.212 6.3.1.4** | 速率匹配 | 由 `polar_seg_deRm_deItl` 反轉 |

### 9.2 UCI 多工配置

**F2/F3 格式的 UCI 復用**：
- **HARQ-ACK**：0-2 位（ACK/NACK/DTX 指示）
- **SR**：0-1 位（排程請求標誌）
- **CSI Part 1**：0-1706 位
  - 頻道品質指示 (CQI)：4-8 位
  - 預編碼矩陣指示 (PMI)：4-16 位
  - 等級指示 (RI)：1-2 位

---

## 10. 偵錯與故障排查

### 10.1 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|--------|
| **核心超時** | UCI 數量超過 18 | 檢查 `nF2Ucis + nF3Ucis <= 36` |
| **記憶體訪問違規** | 有效負載緩衝區對齐不當 | 確保 `uciPayloadsGpu` 32 位對齐 |
| **不匹配的位長** | `bitLenXxx` 不一致 | 驗證來自極化解碼器的 CRC 狀態 |
| **錯誤的有效負載偏移量** | 計算錯誤 | 使用 `PucchRx::allocateBackendBuffers()` 為指南 |
| **描述符複製失敗** | GPU 記憶體不足 | 增加 GPU 記憶體或減少 UCI 數量 |

### 10.2 效能分析

**使用 Nsight 系統進行配置**：

```bash
# 記錄核心執行情況
nsys profile --trace cuda,nvtx -o profile.qdrep ./cuphy_app

# 分析記憶體頻寬
ncu --set full ./cuphy_app
```

**預期指標**：
- **SM 佔用率**：< 10%（輕量級核心）
- **記憶體頻寬使用率**：< 1%（極低計算強度）
- **延遲隱藏**：優秀（64 個活躍執行緒足以掩蓋延遲）

---

## 11. 最佳實踐

### 11.1 記憶體管理

**統一有效負載緩衝區**：
```cpp
// 單個大緩衝區優於多個小緩衝區
uint8_t* uciPayloadsGpu = cudaMalloc(totalUciSizeBytes);

// 計算：
uint32_t totalUciSizeBytes = 0;
for (int i = 0; i < nF2Ucis; ++i) {
    totalUciSizeBytes += 
        (F2OutOffsets[i].csi1PayloadByteOffset + 
         (F2UciPrms[i].bitLenCsiPart1 + 7) / 8);
}
// ... 類似地處理 F3
```

### 11.2 並行執行

**多個 UCI 批次的流管理**：
```cpp
// 為不同 UCI 組使用不同的流
cudaStream_t streamGroup[numGroups];
for (int i = 0; i < numGroups; ++i) {
    cudaStreamCreate(&streamGroup[i]);
    // 在每個流中執行獨立的批次
}

// 同步所有組
cudaDeviceSynchronize();
```

### 11.3 資料驗證

**複製 UCI 結果供驗證**：
```cpp
// 從 GPU 複製有效負載結果
uint8_t* uciPayloadsCpu = new uint8_t[totalUciSizeBytes];
CUDA_CHECK(cudaMemcpyAsync(
    uciPayloadsCpu, uciPayloadsGpu, totalUciSizeBytes,
    cudaMemcpyDeviceToHost, cuStream
));
cudaStreamSynchronize(cuStream);

// 將結果與 MATLAB 參考實現進行比較
for (int uciIdx = 0; uciIdx < nF2Ucis; ++uciIdx) {
    uint32_t refHarq = referenceHarqResults[uciIdx];
    uint32_t devHarq = *(uint32_t*)&uciPayloadsCpu[harqOffset];
    ASSERT_EQ(refHarq, devHarq, "HARQ 不匹配");
}
```

---

## 12. MATLAB 參考實現

### 12.1 UCI 分段函式 (uciSegPolarDecode.m)

```matlab
function [uciSegEst, crcErrorFlag, interBuffers] = uciSegPolarDecode(...
    A_seg, E_seg, listLength, uciSegLLRs)
% 輸入：
%   A_seg: UCI 分段信息位數
%   E_seg: UCI 分段傳輸位數
%   listLength: 極化解碼列表長度
%   uciSegLLRs: UCI 分段的 LLR（對數似然比）

% ===== 推導參數 =====
polarUciSegPrms = derive_polarUciSegPrms(A_seg, E_seg);

% ===== 反向速率匹配和去交錯 =====
deRmDeItlDynDesc = compute_polDeRmDeItlDynDesc(polarUciSegPrms);
cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, deRmDeItlDynDesc, uciSegLLRs);

% ===== 極化解碼 =====
nBitsPerCb = polarUciSegPrms.K_cw - polarUciSegPrms.nCrcBits;
cbEsts = zeros(nBitsPerCb, polarUciSegPrms.nCbs);
cbCrcErrorFlags = zeros(2, 1);
crcErrorFlag = 0;

for cbIdx = 1 : polarUciSegPrms.nCbs
    [cbEsts(:, cbIdx), cbCrcErrorFlags(cbIdx)] = polar_decoder(...
        listLength, polarUciSegPrms.K_cw, polarUciSegPrms.N_cw, ...
        cwLLRs(:, cbIdx));
    
    if cbCrcErrorFlags(cbIdx) == 1
        crcErrorFlag = 1;
    end
end

% ===== UCI 分段組合 =====
if polarUciSegPrms.nCbs == 1
    uciSegEst = cbEsts(:, 1);
else
    uciSegEst = combine_uciCbEsts(cbEsts(:, 1), cbEsts(:, 2), ...
        polarUciSegPrms);
end

interBuffers.polarUciSegPrms = polarUciSegPrms;
interBuffers.cbEsts = cbEsts;
end
```

### 12.2 碼字分段功能 (polarCbSegment.m)

```matlab
function polCbs = polarCbSegment(polarUciSegPrms, uciSegPayload)
% 分段 UCI 有效負載為一個或多個極化碼字

nCbs = polarUciSegPrms.nCbs;
A_seg = polarUciSegPrms.A_seg;

if nCbs == 1
    polCbs = uciSegPayload;  % [A_seg x 1]
else
    % 分割為兩個碼字
    nBitsPerCb1 = ceil(A_seg / 2);
    nBitsPerCb2 = A_seg - nBitsPerCb1;
    
    polCbs = zeros(polarUciSegPrms.K_cw, 2);
    polCbs(1:nBitsPerCb1, 1) = uciSegPayload(1:nBitsPerCb1);
    polCbs(1:nBitsPerCb2, 2) = uciSegPayload(nBitsPerCb1+1:end);
end
end
```

### 12.3 速率匹配反演 (pol_cwSeg_deRm_deItl.m)

```matlab
function cwLLRs = pol_cwSeg_deRm_deItl(polarUciSegPrms, ...
    deRmDeItlDynDesc, uciSegLLRs)
% 將 UCI 分段 LLR 反向速率匹配和去交錯為碼字 LLR

E_cw = polarUciSegPrms.E_cw;
N_cw = polarUciSegPrms.N_cw;
nCbs = polarUciSegPrms.nCbs;

cwLLRs = zeros(N_cw, nCbs);

for cbIdx = 1 : nCbs
    % 提取該碼字的 LLR
    startIdx = (cbIdx - 1) * E_cw + 1;
    endIdx = cbIdx * E_cw;
    cbLLRs = uciSegLLRs(startIdx:endIdx);
    
    % 反向速率匹配（穿孔/重複恢復）
    deRmLLRs = deRateMatch(cbLLRs, N_cw, deRmDeItlDynDesc);
    
    % 去交錯
    load('P1.mat');  % 交錯排列 P1
    deItlLLRs(P1 + 1) = deRmLLRs;  % MATLAB 使用 1 基索引
    
    cwLLRs(:, cbIdx) = deItlLLRs;
end
end
```

---

## 13. 總結

### 13.1 核心功能指標

| 功能 | 指標 |
|------|------|
| **最大 UCI 數量** | 36 (18 F2 + 18 F3) |
| **每個 UCI 的最大位** | 1706 (HARQ 2 + SR 1 + CSI1 1706) |
| **執行緒塊大小** | 972 執行緒 |
| **每個 UCI 的執行緒** | 54 個 |
| **典型延遲** | 8-13 µs (36 UCI 批次) |
| **記憶體頻寬** | < 1 GB/s (極低) |
| **功率效率** | 10-20 mW 每核心啟動 |

### 13.2 關鍵整合點

1. **輸入源**：`polar_decoder` 提供的 CRC 驗證比特估計
2. **輸出目的地**：UCI 有效負載緩衝區（用於 MAC 層指示）
3. **依賴項**：`polar_seg_deRm_deItl`（前置行為） → `pucchF234UciSeg`（後置行為）
4. **管道位置**：F2/F3/F4 前端（segLLRs 生成） → 極化解碼 → **此模組** → MAC 用戶指示

### 13.3 常用工作流程

```cpp
// 簡化的使用模式：
1. cuphyCreatePucchF234UciSeg(&hdl)
2. 為每個 UCI 批次：
   a. 配置引數（bitLen, offsets）
   b. cuphySetupPucchF234UciSeg(...)
   c. 複製描述符到 GPU
   d. cuLaunchKernel(...)
   e. 同步並驗證結果
3. cuphyDestroyPucchF234UciSeg(hdl)
```

---

**文檔修訂日期**：2025年1月  
**相容 cuPHY 版本**：1.0+  
**作者**：NVIDIA 無線接收器團隊
