# BFC 波束成型係數計算模組

## 概述

BFC（Beam Forming Coefficient）波束成型係數計算模組是 NVIDIA cuPHY 中用於計算 MMSE（Minimum Mean Square Error）波束成型權重係數的 GPU 加速模組。該模組基於信道估計結果和正則化參數，使用高效的 GPU 核心進行矩陣運算，支援多層多用戶的 MU-MIMO 系統。

**主要功能**:
- **MMSE 係數計算**: 計算最小均方誤差波束成型權重
- **多層支持**: 支援 8、16 層的多層傳輸
- **多天線支持**: 支援 64 根基站天線
- **LU 分解**: 使用高效的 LU 矩陣分解計算 Gram 矩陣的逆
- **Gram 矩陣增強**: $G = HH^T + \Lambda$（H 為信道矩陣，Λ 為正則化對角矩陣）

**數學基礎**:
$$C = H^T (HH^T + \Lambda)^{-1} = H^T G^{-1}$$

其中：
- $H$: 信道矩陣 (層數 × 天線數)
- $\Lambda$: 正則化係數對角矩陣，用於改進數值穩定性
- $C$: 輸出波束成型係數矩陣
- $G^{-1}$: Gram 矩陣的逆（通過 LU 分解計算）

---

## 架構概述

### 核心組件

| 組件 | 功能 | 應用場景 |
|------|------|---------|
| **MMSE 係數核心函數** | 實現 MMSE 波束成型權重計算 | 所有波束成型情境 |
| **LU 分解模組** | 使用 Gaussian 消元計算矩陣逆 | 提高計算效率和精度 |
| **Gram 矩陣計算** | 計算 $HH^T + \Lambda$ | 準備分解輸入 |
| **設備管理** | 靜態/動態描述符、批量 memcpy | 多 UE 支持和異構配置 |
| **多核配置** | 支援不同層數和天線數的核選擇 | 靈活配置支持 |

### 檔案結構

```
cuphy/bfc/
├── bfc.cuh              # CUDA 核心算法和矩陣運算
├── bfc.cu               # 主 BFC 核心和核啟動邏輯
├── bfc.hpp              # C++ 介面和類定義
├── bfw_blockFP.cuh      # Block Floating Point 壓縮（可選）
└── testBfc.cpp          # 單元測試

特定應用包裝:
├── cuphy.cpp            # C API 包裝函數
└── examples/bfc/        # BFC 示例代碼
    ├── cuphy_ex_bfc.cpp
    └── datasets.cpp
```

---

## 矩陣計算演算法

### Gram 矩陣計算

**公式**: $G = HH^T + \Lambda$

```cpp
// Gram 矩陣的計算
// 輸入: 信道矩陣 H (層數 × 天線數)，正則化向量 Lambda (層數)
// 輸出: Gram 矩陣 G (層數 × 層數)

template <typename TStorageIn, typename TCompute>
CUDA_BOTH inline void gramMatrixCompute(
    const tensor_ref& H,              // 信道矩陣
    const tensor_ref& Lambda,         // 正則化係數
    tensor_ref& G)                    // 輸出 Gram 矩陣
{
    // G = HH^T
    // 針對每個 PRB group 和 UE group，計算 Gram 矩陣
    // 使用協作群組確保 warp 級別的效率
}
```

### LU 分解與反演

**演算法**: Gaussian 消元 + 回代求解

```cpp
// LU 分解和矩陣求逆
// G 使用 Gaussian 消元分解為 L 和 U
// 然後計算 G^{-1}

template <uint32_t N_LAYERS>
CUDA_BOTH inline void luDecompose(
    tensor_ref& G,                    // Gram 矩陣 (輸入/輸出)
    tensor_ref& Linv)                 // 下三角矩陣逆 (輸出)
{
    // 實現 LU 分解，結果存在 G 中
    // Gram 矩陣在分解過程中被覆蓋
    
    // 前向消元和後向替換
    for(int i = 0; i < N_LAYERS; i++)
    {
        // Pivot 選擇和消元
        // 更新剩餘行
    }
}
```

### MMSE 係數計算

**演算法**: 矩陣乘法 $C = H^T \times G^{-1}$

```cpp
// MMSE 波束成型係數計算
// 計算 C = H^T * inv(G) = H^T * Linv^{-T}

template <typename TStorageIn, typename TStorageOut, typename TCompute,
          uint32_t N_BS_ANTS,  // 基站天線數 (H 矩陣行數)
          uint32_t N_LAYERS>   // 層數 (H 矩陣列數)
__global__ void bfc_mmse_coef_comp_kernel(
    uint32_t Nprb,                    // 資源塊數量
    const_tensor_ref_v0 tH,           // 信道矩陣張量
    const_tensor_ref_v0 tLambda,      // 正則化係數張量
    tensor_ref_v0 tCoef,              // 輸出係數張量
    tensor_ref_v0 tDbg)               // 調試信息 (可選)
{
    // 每個 thread block 處理一個 PRB group
    // 每個 thread group 計算一層
    
    // 1. 從全局內存讀取 H 和 Lambda
    // 2. 計算 Gram 矩陣 G = HH^T + Lambda
    // 3. 執行 LU 分解
    // 4. 計算 H^T * inv(G)
    // 5. 寫入結果到全局內存
}
```

---

## C/C++ 函數介面

### 初始化和清理

```cpp
// 創建 BFC 係數計算對象
cuphyStatus_t CUPHYWINAPI cuphyCreateBfwCoefComp(
    unsigned int         nMaxUeGrps,           // 最大 UE 群組數
    unsigned int         nMaxTotalLayers,      // 最大總層數
    unsigned int         flags,                // 初始化標誌
    cuphyBfwCoefComp_t** pBfwCoefComp);       // 輸出: 對象句柄

// 銷毀 BFC 係數計算對象
cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwCoefComp(
    cuphyBfwCoefComp_t* pBfwCoefComp);

// 初始化靜態參數
cuphyStatus_t CUPHYWINAPI cuphyBfwCoefCompInit(
    cuphyBfwCoefComp_t*  pBfwCoefComp,
    uint8_t              compressBitwidth,     // Block FP 位寬
    float                beta,                 // 波束成型功率歸一化係數
    float                lambda,               // 正則化係數
    uint8_t              bfwPowerNormAlg_selector,
    void*                pStatDescrCpu,        // 靜態描述符 (CPU)
    void*                pStatDescrGpu,        // 靜態描述符 (GPU)
    void*                pDynDescrsCpu,        // 動態描述符陣列 (CPU)
    void*                pDynDescrsGpu,        // 動態描述符陣列 (GPU)
    void*                pHetCfgUeGrpMapCpu,   // 異構配置映射 (CPU)
    void*                pHetCfgUeGrpMapGpu,   // 異構配置映射 (GPU)
    void*                pUeGrpPrmsCpu,        // UE 群組參數 (CPU)
    void*                pUeGrpPrmsGpu,        // UE 群組參數 (GPU)
    void*                pBfLayerPrmsCpu,      // 波束成型層參數 (CPU)
    void*                pBfLayerPrmsGpu,      // 波束成型層參數 (GPU)
    cudaStream_t         strm);
```

### 設置和執行

```cpp
// 設置係數計算的動態參數
cuphyStatus_t CUPHYWINAPI cuphyBfwCoefCompSetupCoefComp(
    cuphyBfwCoefComp_t*            pBfwCoefComp,
    uint16_t                       nUeGrps,      // UE 群組數
    const cuphyBfwUeGrpPrm_t*      pUeGrpPrms,   // UE 群組參數陣列
    bool                           enableCpuToGpuDescrAsyncCpy,
    cuphySrsChEstBuffInfo_t*       pChEstInfo,   // SRS 信道估計緩衝
    uint8_t**                      pBfwCompCoef, // 輸出: 計算結果緩衝
    cuphyBfwCoefCompLaunchCfgs_t*  pLaunchCfgs,  // 核啟動配置
    cudaStream_t                   strm);

// 直接執行單步 BFC 計算（高層 API）
cuphyStatus_t CUPHYWINAPI cuphyBfcCoefCompute(
    unsigned int            nBSAnts,            // 基站天線數
    unsigned int            nLayers,            // 層數
    unsigned int            Nprb,               // 資源塊數量
    cuphyTensorDescriptor_t tDescH,             // H 矩陣描述符
    const void*             HAddr,              // H 矩陣地址
    cuphyTensorDescriptor_t tDescLambda,        // Lambda 描述符
    const void*             lambdaAddr,         // Lambda 地址
    cuphyTensorDescriptor_t tDescCoef,          // 係數描述符
    void*                   coefAddr,           // 係數輸出地址
    cuphyTensorDescriptor_t tDescDbg,           // 調試描述符
    void*                   dbgAddr,            // 調試輸出地址
    cudaStream_t            strm);
```

### 描述符管理

```cpp
// 獲取描述符大小和對齐要求
cuphyStatus_t CUPHYWINAPI cuphyBfwCoefCompGetDescrInfo(
    uint16_t nMaxUeGrps,
    uint16_t nMaxTotalLayers,
    size_t&  statDescrSizeBytes,
    size_t&  statDescrAlignBytes,
    size_t&  dynDescrSizeBytes,
    size_t&  dynDescrAlignBytes,
    size_t&  hetCfgUeGrpMapSizeBytes,
    size_t&  hetCfgUeGrpMapAlignBytes,
    size_t&  ueGrpPrmsSizeBytes,
    size_t&  ueGrpPrmsAlignBytes,
    size_t&  bfLayerPrmsSizeBytes,
    size_t&  bfLayerPrmsAlignBytes);
```

---

## 資料結構

### BFC 計算描述符

```cpp
// 靜態描述符 (初始化一次)
typedef struct _bfwCoefCompStatDescr
{
    float beta;                       // 波束成型功率歸一化係數
    float lambda;                     // 正則化係數
    uint8_t compressBitwidth;        // Block FP 壓縮位寬
    uint8_t bfwPowerNormAlg;         // 功率歸一化演算法選擇
} bfwCoefCompStatDescr_t;

// 動態描述符 (每個 slot/幀更新)
typedef struct _bfwCoefCompDynDescr
{
    uint16_t nUeGrps;                // UE 群組數
    uint16_t nMaxPrbGrp;             // 最大 PRB 群組數
    uint32_t* pHetCfgUeGrpMapGpu;    // 異構配置 UE 群組映射
} bfwCoefCompDynDescr_t;

// UE 群組參數
typedef struct _bfwCoefCompKernelUeGrpPrm
{
    uint8_t nLayers;                 // 層數
    uint16_t nPrbGrp;                // PRB 群組數
    uint16_t maxPrbGrp;              // 最大 PRB 群組數
    cuphyTensorInfo3_t tInfoSrsChEst; // SRS 信道估計張量信息
    uint16_t startPrbGrp;            // 起始 PRB 群組
    uint16_t prbGrpStride;           // PRB 群組跨度
} bfwCoefCompKernelUeGrpPrm_t;

// 波束成型層參數
typedef struct _bfwCoefCompKernelBfLayerPrm
{
    uint8_t ueLayerIdx;              // UE 層索引
    uint16_t startPrbGrpOffset;      // 起始 PRB 群組偏移
    uint16_t prbGrpStride;           // 跨度
    cuphyTensorInfo3_t tInfoSrsChEst; // 信道估計張量
    uint16_t chEstInfoStartPrbGrp;   // 信道估計起始 PRB 群組
    uint16_t startValidPrg;          // 起始有效 PRB 群組
    uint16_t nValidPrg;              // 有效 PRB 群組數
} bfwCoefCompKernelBfLayerPrm_t;

// 核啟動配置
typedef struct _bfwCoefCompLaunchCfg
{
    CUfunction func;                 // CUDA 核函數指針
    uint16_t nMaxPrbGrp;             // 最大 PRB 群組數
    uint16_t nUeGrps;                // UE 群組數
    dim3 gridDim;                    // 網格維度
    dim3 blockDim;                   // 塊維度
} bfwCoefCompLaunchCfg_t;

typedef struct _bfwCoefCompLaunchCfgs
{
    uint32_t nCfgs;                  // 配置數
    bfwCoefCompLaunchCfg_t* pCfgs;   // 配置陣列
} bfwCoefCompLaunchCfgs_t;
```

---

## GPU 核配置

### 核心參數

| 參數 | 值 | 說明 |
|------|-----|------|
| **N_THREADS_PER_WARP** | 32 | 每個 warp 的線程數 |
| **N_THRD_GRPS_PER_THRD_BLK** | 可變 | 每個 thread block 的 thread groups 數 |
| **BLOCK_SIZE** | 256 | 標準 thread block 大小 |
| **SUPPORTED_LAYERS** | 8, 16 | 支援的層數 |
| **MAX_BS_ANTS** | 64 | 最大基站天線數 |

### 核啟動模式

```cpp
// 多核配置選擇
template <typename TStorageIn, typename TStorageOut, typename TCompute>
void bfwCoefCompKernelSelL0(
    bool                          getKernelFuncOnly,
    uint16_t                      nMaxPrbGrp,
    uint16_t                      nUeGrps,
    uint16_t                      nRxAnts,      // 基站天線數
    uint8_t                       nLayers,      // 層數
    cuphyDataType_t              srsChEstType, // 信道估計資料型態
    cuphyDataType_t              lambdaType,   // Lambda 資料型態
    bfwCoefCompKernelSelL0_t&     launchCfg);   // 輸出: 核配置

// 根據天線數和層數選擇合適的核實現
// 16 層: 使用 N_BS_ANTS=64, N_LAYERS=16 的專用核
// 8 層:  使用 N_BS_ANTS=64, N_LAYERS=8 的專用核
```

---

## 效能特性

### 計算複雜度

| 操作 | 複雜度 | 備註 |
|------|--------|------|
| Gram 矩陣計算 | $O(L^2 \times M)$ | L = 層數，M = 天線數 |
| LU 分解 | $O(L^3)$ | L = 層數，使用 Gaussian 消元 |
| 係數計算 | $O(L^2 \times M)$ | 矩陣乘法 |
| 單 PRB 總計算 | $O(L^3 + L^2 \times M)$ | 通常 $L^3$ 項主導 |

### 使用範例

```cpp
// 直接單步計算
cuphyBfcCoefCompute(
    nBSAnts,                         // 基站天線數
    nLayers,                         // 層數
    nPRBs,                           // 資源塊數量
    tDescH,                          // H 矩陣描述符
    d_H,                             // H 矩陣地址
    tDescLambda,                     // Lambda 描述符
    d_Lambda,                        // Lambda 地址
    tDescCoef,                       // 係數描述符
    d_Coef,                          // 係數輸出地址
    tDescDbg,                        // 調試描述符
    d_Dbg,                           // 調試輸出地址
    cuStream);                       // CUDA 流
```

---

## 相關標準參考

- **3GPP TS 38.211**: NR 物理層信道結構
- **3GPP TS 38.212**: NR 多工和信道編碼
- **3GPP TS 38.214**: NR 物理層工作流程
- **NVIDIA cuPHY**: cuphy/bfc/ 目錄
- **MIMO 波束成型理論**: Goldsmith, Jafar, Jindal 等人的相關論文
