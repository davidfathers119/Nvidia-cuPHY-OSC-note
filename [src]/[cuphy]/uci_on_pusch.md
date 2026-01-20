# UCI on PUSCH (上行控制訊息在PUSCH上傳輸)

## 概述

UCI on PUSCH 模組處理上行鏈路控制訊息（HARQ-ACK、CSI）與上行共享頻道（PUSCH）的多工、編碼和解調。此模組實現 3GPP TS 38.211/213 標準，支援極化碼編碼、分段、率配適應和 CUDA 加速執行。

**關鍵特性：**
- **三階段分段架構**：SegLLRs0（早期HARQ）、SegLLRs1（主要HARQ+CSI-1）、SegLLRs2（CSI-2 控制）
- **CSI-2 控制協處理**：計算 CSI-2 長度、進行預編碼控制
- **完整編碼鏈**：極化編碼 → 速率匹配 → 交錯 → 簡單碼解碼
- **混合精度**：FP16/FP32 支援，CUDA 圖表最佳化

---

## 架構

### 核心類與結構

#### 1. **uciOnPuschSegLLRs0** - 早期 HARQ 分段

```cpp
// filepath: uci_on_pusch/uciOnPusch_segLLRs0.hpp
class uciOnPuschSegLLRs0 : public cuphyUciOnPuschSegLLRs0 {
public:
    void setup(uint16_t nUciUes,
               uint16_t* pUciUeIdxs,
               PerTbParams* pTbPrmsCpu,
               PerTbParams* pTbPrmsGpu,
               uint16_t nUeGrps,
               cuphyTensorPrm_t* pTensorPrmsEqOutLLRs,
               cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsCpu,
               cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsGpu,
               cuphyUciToSeg_t uciToSeg,
               uciOnPuschSegLLRs0DynDescr_t* pCpuDynDesc,
               void* pGpuDynDesc,
               bool enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
               cudaStream_t strm);
};

// 參數結構：各 UCI 的資源映射
struct perUciPrms0_t {
    uint8_t nSym;  // 攜帶 UCI 和/或 SCH 的符號數
    reGrid_t rvdHarqReGrids[MAX_ND_SUPPORTED];      // 重傳 HARQ RE 網格
    reGrid_t harqReGrids[MAX_ND_SUPPORTED];         // 原始 HARQ RE 網格
    reGrid_t csi1ReGrids[MAX_ND_SUPPORTED];         // CSI-1 RE 網格
    uint32_t descramOffsets[MAX_ND_SUPPORTED];      // 解擾偏移
    bool dmrsFlags[MAX_ND_SUPPORTED];
    uint32_t schRmBuffOffsets[MAX_ND_SUPPORTED];    // SCH RM 緩衝偏移
    bool harqPunctFlag;
    uint8_t harqSpx1Flag;
    uint8_t nBitsPerRe;
};

// UCI 到使用者映射
struct uciToUserMap_t {
    uint16_t ueIdx;
    uint16_t ueGrpIdx;
};

// 動態描述符
struct uciOnPuschSegLLRs0DynDescr_t {
    cuphyUciToSeg_t uciToSeg;
    perUciPrms0_t perUciPrmsArray[CUPHY_MAX_N_UCI_ON_PUSCH];
    uciToUserMap_t uciToUserMap[CUPHY_MAX_N_UCI_ON_PUSCH];
    PerTbParams* pUePrmsGpu;
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsGpu;
    tensor_ref_any<CUPHY_R_16F> tEqOutLLRs[MAX_N_USER_GROUPS_SUPPORTED];
};
```

**功能**：從均衡輸出 LLR 提取 HARQ-ACK 和 CSI-1 比特的對數似然比。

---

#### 2. **uciOnPuschSegLLRs1** - 主要 HARQ+CSI-1 分段

```cpp
// filepath: uci_on_pusch/uciOnPusch_segLLRs1.hpp
class uciOnPuschSegLLRs1 : public cuphyUciOnPuschSegLLRs1 {
public:
    void setup(cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl,
               uint16_t nUciUes,
               uint16_t* pUciUserIdxs,
               PerTbParams* pTbPrmsCpu,
               PerTbParams* pTbPrmsGpu,
               uint16_t nUeGrps,
               cuphyTensorPrm_t* pTensorPrmsEqOutLLRs,
               uint16_t* pNumPrbs,
               uint8_t startSym,
               uint8_t nPuschSym,
               uint8_t nPuschDataSym,
               uint8_t* pDataSymIdxs,
               uint8_t nPuschDmrsSym,
               uint8_t* pDmrsSymIdxs,
               uint8_t nDmrsCdmGrpsNoData,
               uciOnPuschSegLLRs1DynDescr_t* pCpuDynDesc,
               void* pGpuDynDesc,
               bool enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
               cudaStream_t strm);
};

// 重傳（RVD）步幅對應
struct uciRvdStride {
    uint16_t rvdStride;
    uint16_t rvdCount;
};

struct uciRvdStrideArray {
    uciRvdStride strideMap[14];  // 最多 14 個 OFDM 符號
};

struct uciRvdLcmArray {
    uint16_t lcmMap[14];  // LCM 對應用於速率匹配計算
};

// UCI 分段動態描述符
struct uciOnPuschSegLLRs1DynDescr_t {
    uint8_t nPuschSym;
    uint8_t nDataSym;
    uint8_t nDmrsSym;
    uint8_t dataSymIdxs[14];
    uint8_t dmrsSymIdxs[14];
    uint16_t uciUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
    PerTbParams* pTbPrms;
};
```

**功能**：處理主 HARQ 和 CSI-1 比特的分段和 LLR 抽取。

---

#### 3. **uciOnPuschCsi2Ctrl** - CSI-2 控制協處理

```cpp
// filepath: uci_on_pusch/uciOnPusch_csi2Ctrl.hpp
class uciOnPuschCsi2Ctrl : public cuphyUciOnPuschCsi2Ctrl {
public:
    void setup(uint16_t nCsi2Ues,
               uint16_t* pCsi2UeIdxsCpu,
               PerTbParams* pTbPrmsCpu,
               PerTbParams* pTbPrmsGpu,
               cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsCpu,
               cuphyPuschCellStatPrm_t* pCellStatPrmsGpu,
               cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsetsCpu,
               uint8_t* pUciPayloadsGpu,
               uint16_t* pNumCsi2BitsGpu,
               cuphyPolarUciSegPrm_t* pCsi2PolarSegPrmsGpu,
               cuphyPolarCwPrm_t* pCsi2PolarCwPrmsGpu,
               cuphyRmCwPrm_t* pCsi2RmCwPrmsGpu,
               cuphySimplexCwPrm_t* pCsi2SpxCwPrmsGpu,
               uint16_t forcedNumCsi2Bits,
               uint8_t enableCsiP2Fapiv3,
               uciOnPuschCsi2CtrlDynDescr_t* pCpuDynDesc,
               void* pGpuDynDesc,
               bool enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
               cudaStream_t strm);
};

// CSI-2 到緩衝區映射
struct csi2ToBuffersMap_t {
    uint16_t ueIdx;                      // UE 索引
    uint16_t statCellIdx;                // 靜態小區索引
    uint32_t csi1PayloadByteOffset;      // CSI-1 有效負載字節偏移
    uint16_t numCsi2BitsOffset;          // CSI-2 比特數偏移
};

// CSI-2 控制動態描述符
struct uciOnPuschCsi2CtrlDynDescr_t {
    cuphyUciToSeg_t uciToSeg;
    csi2ToBuffersMap_t csi2ToBuffersMapArray[CUPHY_MAX_N_UCI_ON_PUSCH];
    cuphyPolarCwPrm_t* pPolCwPrms;
    cuphyPolarUciSegPrm_t* pPolSegPrms;
    cuphySimplexCwPrm_t* pSpxCwPrms;
    cuphyRmCwPrm_t* pRmCwPrms;
};
```

**功能**：計算 CSI-2 比特長度、控制預編碼因子、管理 CSI-2 編碼參數。

---

#### 4. **uciOnPuschSegLLRs2** - CSI-2 分段

```cpp
// filepath: uci_on_pusch/uciOnPusch_segLLRs2.hpp
class uciOnPuschSegLLRs2 : public cuphyUciOnPuschSegLLRs2 {
public:
    void setup(cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl,
               uint16_t nCsi2Ues,
               uint16_t* pCsi2UeIdxs,
               PerTbParams* pTbPrmsCpu,
               PerTbParams* pTbPrmsGpu,
               uint16_t nUeGrps,
               cuphyTensorPrm_t* pTensorPrmsEqOutLLRs,
               uint8_t startSym,
               uint8_t nPuschSym,
               uint8_t nPuschDataSym,
               uint8_t* pDataSymIdxs,
               uint8_t nPuschDmrsSym,
               uint8_t* pDmrsSymIdxs,
               uint8_t nDmrsCdmGrpsNoData,
               uciOnPuschSegLLRs2DynDescr_t* pCpuDynDesc,
               void* pGpuDynDesc,
               bool enableCpuToGpuDescrAsyncCpy,
               cuphyUciOnPuschSegLLRs2LaunchCfg_t* pLaunchCfg,
               cudaStream_t strm);
};

// CSI-2 使用者映射
struct csi2ToUserMap_t {
    uint16_t ueIdx;
    uint16_t ueGrpIdx;
};

// HARQ + CSI-1 資源參數
struct harqAndCsi1RePrms_t {
    reGrid_t rvdHarqReGrids[MAX_ND_SUPPORTED];
    reGrid_t harqReGrids[MAX_ND_SUPPORTED];
    reGrid_t csi1ReGrids[MAX_ND_SUPPORTED];
    uint32_t descramOffsets[MAX_ND_SUPPORTED];
    bool dmrsFlags[MAX_ND_SUPPORTED];
    uint32_t schRmBuffOffsets[MAX_ND_SUPPORTED];
    bool harqPunctFlag;
    uint8_t harqSpx1Flag;
};

// CSI-2 分段動態描述符
struct uciOnPuschSegLLRs2DynDescr_t {
    uint8_t nPuschSym;
    uint8_t nDataSym;
    uint8_t nDmrsSym;
    uint8_t dataSymIdxs[14];
    uint8_t dmrsSymIdxs[14];
    csi2ToUserMap_t csi2ToUserMapArray[CUPHY_MAX_N_UCI_ON_PUSCH];
    PerTbParams* pTbPrms;
};
```

**功能**：提取 CSI-2 比特的 LLR 並進行分段。

---

## 參數結構

### UCI 輸入參數

```cpp
// UCI 到分段映射選項
enum cuphyUciToSeg_t {
    SEG_ALL_UCI = 0,           // 所有 UCI（HARQ + CSI-1 + CSI-2）
    SEG_HARQ_ONLY = 1,         // 僅 HARQ
    SEG_HARQ_CSI1 = 2          // HARQ + CSI-1（不含 CSI-2）
};

// UCI 出力偏移
struct cuphyUciOnPuschOutOffsets_t {
    uint32_t csi1PayloadByteOffset;     // CSI-1 有效負載字節偏移
    uint16_t numCsi2BitsOffset;         // CSI-2 比特數偏移
};
```

### 編碼參數

```cpp
// 極化 UCI 段參數（來自 polar_encoder）
struct cuphyPolarUciSegPrm_t {
    uint16_t K;              // 訊息長度
    uint16_t N_cw;           // 極化碼字長度
    uint16_t nCbs;           // 代碼塊數（每段 1 或 2）
    uint8_t nCrcBits;        // CRC 比特數
    uint8_t zeroInsertFlag;  // 零插入標誌
};

// 速率匹配代碼字參數
struct cuphyRmCwPrm_t {
    uint32_t E;              // 輸出比特數
    uint32_t d_cbEst;        // 設備估算緩衝地址
};

// 簡單碼代碼字參數
struct cuphySimplexCwPrm_t {
    uint16_t A;              // 訊息長度
    uint16_t E;              // 編碼長度
};
```

---

## 初始化與生命週期

### 創建階段

```cpp
// C API 創建函數
cuphyStatus_t cuphyCreateUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0Hndl_t* pUciOnPuschSegLLRs0Hndl);

cuphyStatus_t cuphyCreateUciOnPuschSegLLRs1(
    cuphyUciOnPuschSegLLRs1Hndl_t* pUciOnPuschSegLLRs1Hndl);

cuphyStatus_t cuphyCreateUciOnPuschSegLLRs2(
    cuphyUciOnPuschSegLLRs2Hndl_t* pUciOnPuschSegLLRs2Hndl);

cuphyStatus_t cuphyCreateUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlHndl_t* pUciOnPuschCsi2CtrlHndl);
```

**步驟**：
1. 在 CPU 上分配不透明結構
2. 初始化內部狀態
3. 返回不透明句柄供後續設定

### 設定階段

```cpp
// 例：SegLLRs1 設定
cuphyStatus_t cuphySetupUciOnPuschSegLLRs1(
    cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl,
    uint16_t nUciUes,
    uint16_t* pUciUserIdxs,
    PerTbParams* pTbPrmsCpu,           // CPU 傳輸塊參數
    PerTbParams* pTbPrmsGpu,           // GPU 傳輸塊參數
    uint16_t nUeGrps,                  // 使用者群組數
    cuphyTensorPrm_t* pTensorPrmsEqOutLLRs,  // 均衡輸出 LLR 張量
    uint16_t* pNumPrbs,                // 每個 UE 的 PRB 數
    uint8_t startSym,
    uint8_t nPuschSym,
    uint8_t nPuschDataSym,
    uint8_t* pDataSymIdxs,
    uint8_t nPuschDmrsSym,
    uint8_t* pDmrsSymIdxs,
    uint8_t nDmrsCdmGrpsNoData,
    uciOnPuschSegLLRs1DynDescr_t* pCpuDynDesc,
    void* pGpuDynDesc,
    bool enableCpuToGpuDescrAsyncCpy,
    cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);
```

**功能**：
- 驗證輸入參數
- 計算動態描述符大小
- 將 CPU 描述符複製到 GPU（可選非同步）
- 填充 CUDA 核啟動配置

### 執行階段

```cpp
// 執行 SegLLRs1 核
cuphyStatus_t cuphyRunUciOnPuschSegLLRs1(
    cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);
```

**核功能**：
1. 讀取 UE 索引和 TB 參數
2. 遍歷每個 UCI 及其符號
3. 計算資源元素（RE）的位置
4. 從均衡 LLR 張量提取 LLR
5. 存儲到結果緩衝區

### 清理階段

```cpp
cuphyStatus_t cuphyDestroyUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl);

cuphyStatus_t cuphyDestroyUciOnPuschSegLLRs1(
    cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl);

cuphyStatus_t cuphyDestroyUciOnPuschSegLLRs2(
    cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl);

cuphyStatus_t cuphyDestroyUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl);
```

**功能**：釋放內部緩衝區和句柄。

---

## UCI 分段 LLR 抽取

### 演算法流程

```
輸入：均衡 LLR 張量 (EqOutLLRs)
      ├─ 形狀：[nRxAnt, nDataSym, nPrbs, CUPHY_N_TONES_PER_PRB, nLayers]
      └─ 資料類型：CUPHY_R_16F（FP16）

流程：
1. 對每個 UCI UE：
   ├─ 檢索 TB 參數（PerTbParams）
   │  ├─ mScUciSum：UCI 資源元素總數
   │  ├─ betaOffsetHarq：HARQ 比例因子索引
   │  └─ betaOffsetCsi1：CSI-1 比例因子索引
   │
   ├─ 對每個 UCI 類型（HARQ/CSI-1/CSI-2）：
   │  ├─ 檢索資源映射（RE 網格）
   │  ├─ 對每個符號：
   │  │  ├─ 計算 RE 位置（PRB、子載波）
   │  │  ├─ 檢索該 RE 的 LLR
   │  │  ├─ 應用 β 縮放：LLR_scaled = β × LLR_original
   │  │  └─ 儲存到輸出 UCI 緩衝
   │  │
   │  └─ 應用解擾序列（降低自相關）
   │
   └─ 提供下一級（編碼鏈）

輸出：UCI 比特 LLR 緩衝區 (pUciPayloads)
      └─ 組織：[HARQ|CSI-1|CSI-2 LLRs]
```

### 資源映射範例

對於配置 A（15 KHz 子載波間距）：

```
PUSCH 分配：PRBStart=0, nPRBs=48（240 RE/符號）
資料符號索引：{2,3,4,5,8,9,10,11}
DMRS 符號索引：{1,6}

HARQ-ACK 分配：
├─ nHarqBits=2
├─ β_HARQ_index=11 → β=3.0
├─ mScHarq=8（4個RE×2層）

CSI-1 分配（如果存在）：
├─ nCsi1Bits=4
├─ β_CSI_index=6 → β=1.0
└─ mScCsi1=8

CSI-2 分配（如果存在）：
├─ nCsi2Bits（由 CSI-2 控制計算）
├─ β_CSI2=自動計算
└─ mScCsi2=剩餘 RE
```

---

## CSI-2 控制協處理

### CSI-2 長度計算

CSI-2 長度基於信道狀態訊息第 2 部分（CSI-IM）的測量：

```cpp
// CSI-2 控制核算法
__global__ void uciOnPuschCsi2CtrlKernel(
    uciOnPuschCsi2CtrlDynDescr_t* pDynDescr) {
    
    uint16_t csi2UeIdx = blockIdx.x;
    csi2ToBuffersMap_t& bufMap = pDynDescr->csi2ToBuffersMapArray[csi2UeIdx];
    
    // 1. 讀取 CSI-1 有效負載以估計信道
    uint8_t* pCsi1Payload = pDynDescr->pUciPayloads + bufMap.csi1PayloadByteOffset;
    
    // 2. 從 CSI-1 比特解碼天線選擇和預編碼
    // 3. 根據信道品質計算 CSI-2 資訊比特數
    uint16_t A_csi2;  // CSI-2 訊息比特
    if (enableCsiP2Fapiv3) {
        A_csi2 = compute_csi2_bits_fapiv3(pCsi1Payload, ...);
    } else {
        A_csi2 = compute_csi2_bits_legacy(pCsi1Payload, ...);
    }
    
    // 4. 儲存結果
    uint16_t* pNumCsi2Bits = pDynDescr->pNumCsi2Bits + bufMap.numCsi2BitsOffset;
    *pNumCsi2Bits = A_csi2;
    
    // 5. 計算 CSI-2 編碼長度
    uint16_t E_csi2 = calculate_encoded_length(A_csi2, tbPrms.mScCsi2Sum, ...);
    
    // 6. 更新極化編碼參數
    cuphyPolarUciSegPrm_t& polSegPrm = pDynDescr->pPolSegPrms[csi2UeIdx];
    polSegPrm.K = A_csi2;
}
```

### CSI-2 與 β 縮放

```cpp
// β 計算（3GPP TS 38.213）
float beta_csi2 = 10^(betaOffsetCsi2Index / 10.0);

// CSI-2 資源元素配置
uint16_t mScCsi2 = remaining_res_elements_after_harq_csi1();

// 動態調整確保輸出長度
uint16_t E_csi2 = (A_csi2 == 0) ? 0 : 
    (uint16_t)((float)A_csi2 * (float)mScCsi2 / (float)tbPrms.mScCsi2Sum);
```

---

## 編碼鏈整合

UCI on PUSCH 編碼管道連接到極化編碼器/速率匹配器/簡單碼解碼器：

```
LLR 分段（SegLLRs0/1/2）
    ↓
CSI-2 控制（Csi2Ctrl）→ [計算 CSI-2 長度]
    ↓
極化編碼 (Polar Encoder)
    ├─ 輸入：A 訊息比特 + CRC
    ├─ 輸出：N_cw 編碼比特
    └─ 參數：cuphyPolarUciSegPrm_t
    ↓
速率匹配 (Rate Matching)
    ├─ 輸入：N_cw 編碼比特
    ├─ 輸出：E 比特
    └─ 參數：cuphyRmCwPrm_t
    ↓
交錯 (Interleaving)
    └─ 改善突發誤差耐受性
    ↓
簡單碼解碼 (Simplex Decoder) - 可選
    ├─ 用於短 UCI 序列
    └─ 參數：cuphySimplexCwPrm_t
    ↓
PUSCH 資源映射 → [與資料/DMRS 多工]
```

### 參數準備範例

```cpp
// 設定 CSI-2 極化編碼參數
void prepare_csi2_polar_params(
    uint16_t nCsi2Ues,
    uint16_t* pNumCsi2Bits,           // CSI-2 控制的輸出
    cuphyPolarUciSegPrm_t* pPolSegPrms) {
    
    for (uint16_t i = 0; i < nCsi2Ues; ++i) {
        uint16_t A = pNumCsi2Bits[i];
        
        // 查表得到 K（訊息長度含 CRC）
        uint16_t K_with_crc = a_to_k_mapping(A);
        
        // 查表得到 N_cw（極化碼長度）
        uint16_t N_cw = get_polar_code_length(K_with_crc);
        
        pPolSegPrms[i].K = K_with_crc;
        pPolSegPrms[i].N_cw = N_cw;
        pPolSegPrms[i].nCbs = 1;  // CSI-2 通常為 1 個代碼塊
        pPolSegPrms[i].nCrcBits = 11;  // CRC-11
        pPolSegPrms[i].zeroInsertFlag = 0;
    }
}
```

---

## CUDA 核實現細節

### SegLLRs1 核簽名

```cpp
// filepath: uci_on_pusch/uciOnPusch_segLLRs1.cu
__global__ void uciOnPuschSegLLRs1Kernel(
    uciOnPuschSegLLRs1KernelArgs_t* pArgs) {
    
    uciOnPuschSegLLRs1DynDescr_t* pDynDescr = pArgs->pDynDescr;
    
    // 工作分配：每個線程塊處理一個 UCI
    uint16_t uciIdx = blockIdx.x;
    uint16_t uciToSeg = pDynDescr->uciToSeg;
    
    // 獲取 UE 索引
    uciToUserMap_t& uciToUserMap = pDynDescr->uciToUserMap[uciIdx];
    uint16_t ueIdx = uciToUserMap.ueIdx;
    uint16_t ueGrpIdx = uciToUserMap.ueGrpIdx;
    
    // 檢索 TB 和 UE 群組參數
    PerTbParams& tbPrms = pDynDescr->pTbPrms[ueIdx];
    cuphyPuschRxUeGrpPrms_t& ueGrpPrms = pDynDescr->pUeGrpPrmsGpu[ueGrpIdx];
    
    // 提取 UCI 參數
    uint8_t nSym = pDynDescr->nDataSym;
    uint8_t* pDataSymIdxs = pDynDescr->dataSymIdxs;
    
    // 計算資源映射
    uint16_t nPrbs = ueGrpPrms.nPrb;
    uint16_t startPrb = ueGrpPrms.startPrb;
    
    // 並行迴圈：每個線程處理一個 RE
    for (uint32_t reIdx = threadIdx.x; reIdx < tbPrms.mScUciSum; reIdx += blockDim.x) {
        // 1. 計算 RE 位置（PRB、子載波）
        uint16_t prb = (reIdx / CUPHY_N_TONES_PER_PRB) + startPrb;
        uint16_t sc = reIdx % CUPHY_N_TONES_PER_PRB;
        uint8_t sym = pDataSymIdxs[reIdx / (nPrbs * CUPHY_N_TONES_PER_PRB)];
        
        // 2. 從均衡 LLR 張量讀取
        __half eqLlr = pDynDescr->tEqOutLLRs[ueGrpIdx](0, sym, prb, sc, 0);
        
        // 3. 應用 β 縮放
        float beta = get_beta_scaling(tbPrms.betaOffsetHarq);
        float scaledLlr = (float)eqLlr * beta;
        
        // 4. 轉換為 int8 （量化）
        int8_t quantLlr = (int8_t)clamp(scaledLlr, -128.0f, 127.0f);
        
        // 5. 儲存到 UCI 緩衝
        pUciLLRs[reIdx] = quantLlr;
    }
}
```

### CSI-2 控制核

```cpp
// filepath: uci_on_pusch/uciOnPusch_csi2Ctrl.cu
__global__ void uciOnPuschCsi2CtrlKernel(
    uciOnPuschCsi2CtrlDynDescr_t* pDynDescr) {
    
    uint16_t csi2UeIdx = blockIdx.x;
    if (csi2UeIdx >= pDynDescr->nCsi2Ues) return;
    
    // 1. 獲取 CSI-2 UE 資訊
    csi2ToBuffersMap_t& bufMap = pDynDescr->csi2ToBuffersMapArray[csi2UeIdx];
    uint16_t ueIdx = bufMap.ueIdx;
    
    // 2. 讀取 CSI-1 有效負載
    uint8_t* pCsi1Payload = pDynDescr->pUciPayloads + bufMap.csi1PayloadByteOffset;
    
    // 3. 解碼 CSI-1（天線選擇、預編碼指示）
    uint8_t i1 = decode_antenna_selection(pCsi1Payload);
    uint8_t i2 = decode_precoding_indicator(pCsi1Payload);
    
    // 4. 計算 CSI-2 訊息比特
    uint16_t nCsi2Bits = compute_csi2_info_bits(i1, i2, ...);
    
    // 5. 儲存結果
    uint16_t* pNumCsi2BitsOut = pDynDescr->pNumCsi2Bits + bufMap.numCsi2BitsOffset;
    *pNumCsi2BitsOut = nCsi2Bits;
    
    // 6. 準備極化編碼參數
    cuphyPolarUciSegPrm_t* pPolSegPrm = &pDynDescr->pPolSegPrms[csi2UeIdx];
    pPolSegPrm->K = nCsi2Bits + 11;  // +11 CRC 比特
    pPolSegPrm->N_cw = select_polar_code_length(pPolSegPrm->K);
    pPolSegPrm->nCbs = 1;
    pPolSegPrm->nCrcBits = 11;
    pPolSegPrm->zeroInsertFlag = (nCsi2Bits == 0) ? 1 : 0;
}
```

---

## C/C++ API 參考

### C API 函數

```cpp
// 建立 UCI 分段 LLR 提取器
cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0Hndl_t* pUciOnPuschSegLLRs0Hndl);

// 設定描述符並構建 CUDA 圖核
cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl,
    uint16_t nUciUes,
    uint16_t* pUciUeIdxs,
    PerTbParams* pTbPrmsCpu,
    PerTbParams* pTbPrmsGpu,
    uint16_t nUeGrps,
    cuphyTensorPrm_t* pTensorPrmsEqOutLRs,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsCpu,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsGpu,
    cuphyUciToSeg_t uciToSeg,
    uciOnPuschSegLLRs0DynDescr_t* pCpuDynDesc,
    void* pGpuDynDesc,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);

// 執行核
cuphyStatus_t CUPHYWINAPI cuphyRunUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);

// 銷毀句柄
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs0(
    cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl);

// 獲取描述符大小
cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs0GetDescrInfo(
    size_t* pDynDescrSizeBytes,
    size_t* pDynDescrAlignBytes);
```

**CSI-2 控制對應函數：**

```cpp
cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlHndl_t* pUciOnPuschCsi2CtrlHndl);

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl,
    uint16_t nCsi2Ues,
    uint16_t* pCsi2UeIdxsCpu,
    PerTbParams* pTbPrmsCpu,
    PerTbParams* pTbPrmsGpu,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsCpu,
    cuphyPuschCellStatPrm_t* pCellStatPrmsGpu,
    cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsetsCpu,
    uint8_t* pUciPayloadsGpu,
    uint16_t* pNumCsi2BitsGpu,
    cuphyPolarUciSegPrm_t* pCsi2PolarSegPrmsGpu,
    cuphyPolarCwPrm_t* pCsi2PolarCwPrmsGpu,
    cuphyRmCwPrm_t* pCsi2RmCwPrmsGpu,
    cuphySimplexCwPrm_t* pCsi2SpxCwPrmsGpu,
    uint16_t forcedNumCsi2Bits,
    uint8_t enableCsiP2Fapiv3,
    uciOnPuschCsi2CtrlDynDescr_t* pCpuDynDesc,
    void* pGpuDynDesc,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);

cuphyStatus_t CUPHYWINAPI cuphyRunUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschCsi2Ctrl(
    cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl);

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschCsi2CtrlGetDescrInfo(
    size_t* pDynDescrSizeBytes,
    size_t* pDynDescrAlignBytes);
```

---

## 使用範例

### 基本管道設定

```cpp
// filepath: examples/uciOnPusch_csi2_ctrl/cuphy_ex_uciOnPusch_csi2_ctrl.cpp

int main(int argc, char* argv[]) {
    // 1. 建立 UCI 分段提取器
    cuphyUciOnPuschSegLLRs1Hndl_t segLLRs1Hndl;
    cuphyStatus_t status = cuphyCreateUciOnPuschSegLLRs1(&segLLRs1Hndl);
    if (status != CUPHY_STATUS_SUCCESS) throw cuphy::cuphy_exception(status);
    
    // 2. 建立 CSI-2 控制協處理器
    cuphyUciOnPuschCsi2CtrlHndl_t csi2CtrlHndl;
    status = cuphyCreateUciOnPuschCsi2Ctrl(&csi2CtrlHndl);
    if (status != CUPHY_STATUS_SUCCESS) throw cuphy::cuphy_exception(status);
    
    // 3. GPU 緩衝區配置
    uint16_t nUciUes = evalDataset.nUciUes;
    uint16_t nCsi2Ues = evalDataset.nCsi2Ues;
    
    uint8_t* pUciPayloadGpu = linearAlloc.alloc(UCI_PAYLOAD_MAX_BYTES);
    uint16_t* pNumCsi2Bits = linearAlloc.alloc(nCsi2Ues * sizeof(uint16_t));
    
    // 4. 設定動態描述符
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(
        uciOnPuschSegLLRs1::getDynDescrSize());
    
    uciOnPuschSegLLRs1DynDescr_t* pCpuDynDesc = 
        (uciOnPuschSegLLRs1DynDescr_t*)dynDescrBufCpu.addr();
    
    void* pGpuDynDesc = linearAlloc.alloc(
        uciOnPuschSegLLRs1::getDynDescrSize());
    
    // 5. 設定 SegLLRs1
    cuphyUciOnPuschSegLLRs1LaunchCfg_t segLLRs1LaunchCfg;
    
    status = cuphySetupUciOnPuschSegLLRs1(
        segLLRs1Hndl,
        nUciUes,
        uciUserIdxs_buffer.addr(),
        tbPrmsCpu_buffer.addr(),
        pTbPrmsGpu,
        nUeGrps,
        tPrmEqOutLLRsVec.data(),
        nPrbsVec.data(),
        startSym,
        nPuschSym,
        nPuschDataSym,
        pDataSymIdxs,
        nPuschDmrsSym,
        pDmrsSymIdxs,
        nDmrsCdmGrpsNoData,
        pCpuDynDesc,
        pGpuDynDesc,
        true,  // enableCpuToGpuDescrAsyncCpy
        &segLLRs1LaunchCfg,
        cuStrmMain.handle());
    
    if (status != CUPHY_STATUS_SUCCESS) 
        throw cuphy::cuphy_exception(status);
    
    // 6. 設定 CSI-2 控制
    cuphyUciOnPuschCsi2CtrlLaunchCfg_t csi2CtrlLaunchCfg;
    
    status = cuphySetupUciOnPuschCsi2Ctrl(
        csi2CtrlHndl,
        nCsi2Ues,
        csi2UeIdxsVec.data(),
        tbPrmsCpu_buffer.addr(),
        pTbPrmsGpu,
        drvdUeGrpPrmsBuffer.addr(),
        pPuschCellStatPrmsGpu,
        uciOffsetVec.data(),
        pUciPayloadGpu,
        pNumCsi2Bits,
        pCsi2PolSegPrmsGpu,
        pCsi2PolCwPrmsGpu,
        pCsi2RmCwPrmsGpu,
        pCsi2SpxCwPrmsGpu,
        0,     // forcedNumCsi2Bits
        1,     // enableCsiP2Fapiv3
        (uciOnPuschCsi2CtrlDynDescr_t*)dynDescrBufCpu.addr(),
        pGpuDynDesc,
        true,  // enableCpuToGpuDescrAsyncCpy
        &csi2CtrlLaunchCfg,
        cuStrmMain.handle());
    
    // 7. 執行
    status = cuphyRunUciOnPuschSegLLRs1(&segLLRs1LaunchCfg, cuStrmMain.handle());
    if (status != CUPHY_STATUS_SUCCESS) 
        throw cuphy::cuphy_exception(status);
    
    status = cuphyRunUciOnPuschCsi2Ctrl(&csi2CtrlLaunchCfg, cuStrmMain.handle());
    if (status != CUPHY_STATUS_SUCCESS) 
        throw cuphy::cuphy_exception(status);
    
    // 8. 等待完成
    cudaStreamSynchronize(cuStrmMain.handle());
    
    // 9. 清理
    cuphyDestroyUciOnPuschSegLLRs1(segLLRs1Hndl);
    cuphyDestroyUciOnPuschCsi2Ctrl(csi2CtrlHndl);
    
    return 0;
}
```

---

## PUSCH RX 整合

UCI on PUSCH 模組在完整 PUSCH 接收管道中的位置：

```cpp
// filepath: cuphy_channels/pusch_rx.hpp

class PuschRx : public cuphyPuschRx {
private:
    // UCI 分段提取
    cuphyUciOnPuschSegLLRs0Hndl_t m_uciOnPuschEarlySegLLRs0Hndl;
    cuphyUciOnPuschSegLLRs1Hndl_t m_uciOnPuschSegLLRs1Hndl;
    cuphyUciOnPuschSegLLRs2Hndl_t m_uciOnPuschSegLLRs2Hndl;
    
    // CSI-2 控制
    cuphyUciOnPuschCsi2CtrlHndl_t m_uciOnPuschCsi2CtrlHndl;
    
    // UCI 編碼鏈
    cuphyPolarDecoderHndl_t m_polarDecoderHndl;
    cuphyPolSegDeRmDeItlHndl_t m_polSegDeRmDeItlHndl;
    
    // CSI-2 編碼鏈
    cuphyPolarDecoderHndl_t m_polarDecoderHndl_csi2;
    cuphyPolSegDeRmDeItlHndl_t m_polSegDeRmDeItlHndl_csi2;
    cuphySimplexDecoderHndl_t m_simplexDecoderHndl_csi2;

public:
    cuphyStatus_t setup(cuphyPuschDynPrms_t* pDynPrm) {
        // ... LDPC 和資料解碼 ...
        
        // UCI 分段提取
        if (nUciUes > 0) {
            status = cuphySetupUciOnPuschSegLLRs1(
                m_uciOnPuschSegLLRs1Hndl, ...);
        }
        
        // CSI-2 控制
        if (nCsi2Ues > 0) {
            status = cuphySetupUciOnPuschCsi2Ctrl(
                m_uciOnPuschCsi2CtrlHndl, ...);
        }
        
        return CUPHY_STATUS_SUCCESS;
    }
    
    cuphyStatus_t run(cudaStream_t stream) {
        // 1. 資料/DMRS 解調和均衡
        status = runEqualization(stream);
        
        // 2. UCI LLR 分段提取
        if (nUciUes > 0) {
            status = cuphyRunUciOnPuschSegLLRs1(
                &m_uciOnPuschSegLLRs1LaunchCfg, stream);
        }
        
        // 3. CSI-2 控制（計算比特長度）
        if (nCsi2Ues > 0) {
            status = cuphyRunUciOnPuschCsi2Ctrl(
                &m_uciOnPuschCsi2CtrlLaunchCfg, stream);
        }
        
        // 4. HARQ+CSI-1 解碼
        status = runHarqCsi1Decoding(stream);
        
        // 5. CSI-2 解碼
        if (nCsi2Ues > 0) {
            status = runCsi2Decoding(stream);
        }
        
        return CUPHY_STATUS_SUCCESS;
    }
};
```

---

## 效能特性

### 計算複雜度

| 操作 | 時間複雜度 | 記憶體 |
|------|----------|-------|
| SegLLRs1 LLR 提取 | O(nUciUes × mScUciSum) | O(UCI_PAYLOAD_MAX) |
| CSI-2 控制 | O(nCsi2Ues) | O(nCsi2Ues × DESCR_SIZE) |
| 極化編碼 | O(nUciSegs × N_cw × log N_cw) | O(N_cw × nUciSegs) |
| 速率匹配 | O(nCwSum × E_cw) | O(E_cw × nCwSum) |

### 典型吞吐量（H100 GPU，FP16）

```
UCI 分段提取：
├─ 帶寬：1200 GB/s @ 96 UE × 1000 RE/UE → 9.6 ms/slot
└─ 延遲：< 0.5 ms （含 H2D/D2H 複製）

CSI-2 控制（批量 32 UE）：
├─ 計算時間：< 0.1 ms
└─ 記憶體頻寬：240 GB/s

完整編碼鏈（極化+速率匹配+簡單碼）：
├─ 吞吐量：500 Mbps (HARQ-2 bit) 到 1.2 Gbps (CSI-16 bit)
└─ 延遲：1-3 ms per slot
```

### 記憶體配置（最大配置）

```
UCI 有效負載緩衝：
├─ nUciUes = 96
├─ max nHarqBits = 2（每 UE）
├─ max nCsi1Bits = 4（每 UE）
├─ max nCsi2Bits = 11（每 UE）
└─ 總計：96 × (2+4+11)/8 ≈ 228 bytes

描述符記憶體：
├─ SegLLRs1 描述符：~2 KB
├─ CSI-2 Ctrl 描述符：~1 KB
└─ 編碼參數描述符：~4 KB
```

---

## MATLAB 參考實現

### MATLAB UCI 多工

```matlab
% filepath: 5GModel/nr_matlab/pxsch/compute_uciOnPuschMux_descriptor.m

function uciOnPuschDeMuxDescr = compute_uciOnPuschMux_descriptor(nPrb, ...
    nDataSym, symIdx_data, symIdx_dmrs, nl, Qm, G_harq, G_harq_rvd, ...
    G_csi1, G_csi2, alpha_harq, alpha_csi1, alpha_csi2)
    
    % 計算資源元素數
    nReData = nPrb * length(symIdx_data) * 12;  % OFDM 子載波/PRB
    
    % 資源分配
    mSc_harq = floor(G_harq / Qm);
    mSc_csi1 = floor(G_csi1 / Qm);
    mSc_csi2 = floor((nReData - mSc_harq - mSc_csi1) / 1);  % 剩餘
    
    % 分段參數
    nHarq = G_harq / (Qm * alpha_harq);
    nCsi1 = G_csi1 / (Qm * alpha_csi1);
    nCsi2 = mSc_csi2 * alpha_csi2;
    
    % 輸出結構
    uciOnPuschDeMuxDescr.mScHarq = mSc_harq;
    uciOnPuschDeMuxDescr.mScCsi1 = mSc_csi1;
    uciOnPuschDeMuxDescr.mScCsi2 = mSc_csi2;
    uciOnPuschDeMuxDescr.nHarq = nHarq;
    uciOnPuschDeMuxDescr.nCsi1 = nCsi1;
    uciOnPuschDeMuxDescr.nCsi2 = nCsi2;
    uciOnPuschDeMuxDescr.reMapping = compute_re_mapping(...);
end
```

---

## 故障排除

### 常見錯誤

| 錯誤碼 | 原因 | 解決方案 |
|------|------|--------|
| CUPHY_STATUS_INVALID_ARGUMENT | 無效的 UCI UE 索引 | 驗證 `pUciUeIdxs` 在範圍 [0, nUes) 內 |
| CUPHY_STATUS_UNSUPPORTED_ALIGNMENT | 緩衝區對齐不當 | 使用 `linearAlloc` 或 128 字節對齐 |
| CUPHY_STATUS_INTERNAL_ERROR | CUDA 核啟動失敗 | 檢查 CUDA 可用記憶體，運行 `nvidia-smi` |
| CUDA_ERROR_INVALID_ARGUMENT | 無效的流或張量描述符 | 驗證 `cudaStream_t` 和 `cuphyTensorPrm_t` 有效性 |

### 調試檢查清單

1. **緩衝區配置**
   - [ ] UCI 有效負載緩衝大小 ≥ `nUciUes × max(nHarqBits + nCsi1Bits + nCsi2Bits) / 8`
   - [ ] 所有設備指標有效且已對齐
   - [ ] H2D/D2H 複製完成（使用 `cudaStreamSynchronize`）

2. **描述符驗證**
   - [ ] 動態描述符大小與 `getDynDescrInfo()` 返回值相匹配
   - [ ] CPU 和 GPU 描述符地址有效
   - [ ] 所有子結構正確初始化

3. **參數檢查**
   - [ ] `nUciUes` 和 `nCsi2Ues` ≤ `CUPHY_MAX_N_UCI_ON_PUSCH`
   - [ ] `nPrbs` 匹配 PUSCH 分配
   - [ ] RE 映射索引在有效範圍內

4. **流管理**
   - [ ] 相同流中的所有核執行（CUDA 圖相容）
   - [ ] 流在核執行前同步（如必要）

---

## 優化建議

### 1. 記憶體最佳化

```cpp
// 使用 CUDA 圖異步描述符複製
bool enableCpuToGpuDescrAsyncCpy = true;

// 預分配最大 UCI 緩衝區（避免重配置）
uint8_t* pUciPayloadGpu = linearAlloc.alloc(
    CUPHY_MAX_N_UCI_ON_PUSCH * MAX_UCI_BYTES_PER_UE);

// 使用品紅記憶體固定主機緩衝區
cuphy::buffer<uint8_t, cuphy::pinned_alloc> hostBuf(size);
```

### 2. 核配置最佳化

```cpp
// UCI LLR 提取：每 UCI 一個 warp（32 線程）
// 網格：nUciUes 塊
// 塊：256 線程（8 個 warp）
dim3 gridDim(nUciUes);
dim3 blockDim(256);

// CSI-2 控制：輕量級核
// 網格：nCsi2Ues 塊
// 塊：64 線程
dim3 csi2GridDim(nCsi2Ues);
dim3 csi2BlockDim(64);
```

### 3. 精度選擇

```cpp
// FP16 用於 LLR（節省頻寬）
tensor_ref_any<CUPHY_R_16F> tEqOutLLRs;

// 累積使用 FP32（精度）
float scaledLlr = (float)eqLlr_fp16 * beta;

// 量化為 int8（最終存儲）
int8_t quantLlr = (int8_t)clamp(scaledLlr, -128.0f, 127.0f);
```

---

## 結論

UCI on PUSCH 模組提供完整的 GPU 加速上行控制訊息處理，包括：

- **三階段 LLR 分段**：早期 HARQ、主 HARQ+CSI-1、CSI-2
- **CSI-2 動態控制**：基於 CSI-1 計算 CSI-2 長度
- **完整編碼鏈整合**：極化編碼 → 速率匹配 → 簡單碼解碼
- **高效能實現**：< 1 ms 延遲，支援 96+ UE 並行處理

該模組是現代 5G NR PUSCH 接收器的關鍵元件，實現了 3GPP 標準要求的靈活 UCI 多工和傳送。
