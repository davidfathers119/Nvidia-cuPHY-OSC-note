# Channel Estimation (CH_EST)

## 概述

CH_EST（Channel Estimation）是NVIDIA cuPHY庫中用於**PUSCH（Physical Uplink Shared Channel）接收處理**的信道估計組件。實現了多種先進的信道估計算法，包括RKHS（Reproducing Kernel Hilbert Space）方法。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - ch_est](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/ch_est)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構與組件

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `ch_est.hpp` | 信道估計核心類定義（IKernelBuilder、puschRxChEst） |
| `ch_est.cu` | CUDA內核實現 |
| `ch_est_types.hpp` | 數據結構定義 |
| `ch_est_settings.hpp` | 設置和配置 |
| `ch_est_stream.hpp/cpp` | CUDA流管理 |
| `ch_est_graph_mgr.hpp/cpp` | 計算圖管理 |
| `ch_est_null_graphs.hpp` | 空圖實現 |
| `ch_est_config_loader.hpp` | 配置加載器 |
| `ch_est_yaml_loader.cpp` | YAML配置解析 |
| `chest_factory.hpp/cpp` | 工廠模式創建 |
| `ch_est_utils.hpp` | 工具函數 |
| `ch_est_trtengine_pre_post_conversion.hpp/cu` | TensorRT引擎轉換 |
| `trtengine_chest.hpp/cpp` | TensorRT信道估計引擎 |
| `IModule.hpp` | 模塊接口 |
| `IGraph_mgr.hpp` | 圖管理接口 |
| `IStream.hpp` | 流接口 |
| `TensorRT-IModule.md` | TensorRT文檔 |

---

## 主要類

### 1. puschRxChEstKernelBuilder（內核構建器）

**繼承**: `IKernelBuilder`

**功能**: 根據不同的系統配置動態選擇和構建最優的CUDA內核。

#### 公開方法

```cpp
// 構造和初始化
puschRxChEstKernelBuilder();
void init(gsl_lite::span<uint8_t*> ppStatDescrsCpu,
          gsl_lite::span<uint8_t*> ppStatDescrsGpu,
          bool enableCpuToGpuDescrAsyncCpy,
          cudaStream_t strm);

// 構建內核配置
[[nodiscard]] cuphyStatus_t build(
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint8_t maxDmrsMaxLen,
    uint8_t enableDftSOfdm,
    uint8_t chEstAlgo,                    // 信道估計算法選擇
    uint8_t enableUlRxBf,
    uint8_t enablePerPrgChEst,
    uint8_t* pPreEarlyHarqWaitKernelStatusGpu,
    uint8_t* pPostEarlyHarqWaitKernelStatusGpu,
    uint16_t waitTimeOutPreEarlyHarqUs,
    uint16_t waitTimeOutPostEarlyHarqUs,
    bool enableCpuToGpuDescrAsyncCpy,
    gsl_lite::span<uint8_t*> ppDynDescrsCpu,
    gsl_lite::span<uint8_t*> ppDynDescrsGpu,
    pusch::IStartKernels* pStartKernels,
    gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t> launchCfgs,
    uint8_t enableEarlyHarqProc,
    uint8_t enableFrontLoadedDmrsProc,
    uint8_t enableDeviceGraphLaunch,
    CUgraphExec* pSubSlotDeviceGraphExec,
    CUgraphExec* pFullSlotDeviceGraphExec,
    cuphyPuschRxWaitLaunchCfg_t* pWaitKernelLaunchCfgsPreSubSlot,
    cuphyPuschRxWaitLaunchCfg_t* pWaitKernelLaunchCfgsPostSubSlot,
    cuphyPuschRxDglLaunchCfg_t* pDglKernelLaunchCfgsPreSubSlot,
    cuphyPuschRxDglLaunchCfg_t* pDglKernelLaunchCfgsPostSubSlot,
    cudaStream_t strm);
```

#### 三層內核選擇機制

**L2層** - 高級選擇（根據基本參數）
```cpp
template <typename TCompute>
void kernelSelectL2(uint16_t nBSAnts,
                    uint8_t nLayers,
                    uint8_t nDmrsSyms,
                    uint8_t nDmrsGridsPerPrb,
                    uint16_t nTotalDataPrb,
                    uint8_t Nh,
                    uint16_t nUeGrps,
                    uint8_t enableDftSOfdm,
                    uint8_t chEstAlgo,
                    uint8_t enablePerPrgChEst,
                    cuphyDataType_t dataRxType,
                    cuphyDataType_t hEstType,
                    cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**L1層** - 中級選擇（支持模板化）
```cpp
template <typename TStorage, typename TDataRx, typename TCompute>
void kernelSelectL1(uint16_t nBSAnts,
                    uint8_t nLayers,
                    uint8_t nDmrsSyms,
                    uint8_t nDmrsGridsPerPrb,
                    uint16_t nTotalDataPrb,
                    uint8_t Nh,
                    uint16_t nUeGrps,
                    uint8_t enableDftSOfdm,
                    uint8_t chEstAlgo,
                    uint8_t enablePerPrgChEst,
                    cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**L0層** - 具體內核實現選擇
```cpp
template <typename TStorage, typename TDataRx, typename TCompute,
          uint32_t N_LAYERS, uint32_t N_DMRS_GRIDS_PER_PRB, 
          uint32_t N_DMRS_SYMS>
void kernelSelectL0(uint16_t nTotalDataPrb,
                    uint16_t nUeGrps,
                    uint32_t nRxAnt,
                    uint8_t enableDftSOfdm,
                    uint8_t chEstAlgo,
                    uint8_t enablePerPrgChEst,
                    cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

#### 信道估計算法實現

**1. RKHS方法** - Reproducing Kernel Hilbert Space
```cpp
void rkhsKernelSelectL1(uint16_t nTotalDataPrb,
                        cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**2. 分窗口估計**
```cpp
template <typename TStorage, typename TDataRx, typename TCompute,
          uint32_t N_LAYERS, uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
          uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
          uint32_t N_DMRS_SYMS>
void windowedChEst(uint16_t nTotalDataPrb,
                   uint16_t nUeGrps,
                   uint32_t nRxAnt,
                   uint8_t enabelDftSOfdm,
                   cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**3. 小型配置估計**
```cpp
template <typename TStorage, typename TDataRx, typename TCompute,
          uint32_t N_LAYERS, uint32_t N_PRBS,
          uint32_t N_DMRS_GRIDS_PER_PRB, uint32_t N_DMRS_SYMS>
void smallChEst(uint16_t nUeGrps,
                uint32_t nRxAnt,
                uint8_t enabelDftSOfdm,
                cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**4. 多階段估計**
```cpp
template <typename TStorage, typename TDataRx, typename TCompute,
          uint32_t N_LAYERS, uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
          uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
          uint32_t N_DMRS_SYMS>
void multiStageChEst(uint16_t nTotalDataPrb,
                     uint16_t nUeGrps,
                     uint32_t nRxAnt,
                     uint8_t enableDftSOfdm,
                     uint8_t enablePerPrgChEst,
                     cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

**5. 最小二乘估計**
```cpp
template <typename TStorage, typename TDataRx, typename TCompute,
          uint32_t N_LAYERS, uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
          uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
          uint32_t N_DMRS_SYMS>
void lsChEst(uint16_t nTotalDataPrb,
             uint16_t nUeGrps,
             uint32_t nRxAnt,
             uint8_t enableDftSOfdm,
             cuphyPuschRxChEstLaunchCfg_t& launchCfg);
```

---

### 2. puschRxChEst（信道估計模塊）

**繼承**: `IModule`

**功能**: 信道估計的主要實現，管理估計過程的完整生命週期。

#### 公開方法

```cpp
// 構造和設置
puschRxChEst(const cuphyChEstSettings& chEstSettings,
             bool earlyHarqModeEnabled);
puschRxChEst(puschRxChEst const&) = delete;
puschRxChEst& operator=(puschRxChEst const&) = delete;

// 初始化
void init(IKernelBuilder* pKernelBuilder,
          bool enableCpuToGpuDescrAsyncCpy,
          gsl_lite::span<uint8_t*> ppStatDescrsCpu,
          gsl_lite::span<uint8_t*> ppStatDescrsGpu,
          cudaStream_t strm);

// 設置
[[nodiscard]]
cuphyStatus_t setup(
    IKernelBuilder* pKernelBuilder,
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsCpu,
    gsl_lite::span<cuphyPuschRxUeGrpPrms_t> pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint8_t maxDmrsMaxLen,
    uint8_t* pPreEarlyHarqWaitKernelStatusGpu,
    uint8_t* pPostEarlyHarqWaitKernelStatusGpu,
    uint16_t waitTimeOutPreEarlyHarqUs,
    uint16_t waitTimeOutPostEarlyHarqUs,
    bool enableCpuToGpuDescrAsyncCpy,
    gsl_lite::span<uint8_t*> ppDynDescrsCpu,
    gsl_lite::span<uint8_t*> ppDynDescrsGpu,
    uint8_t enableEarlyHarqProc,
    uint8_t enableFrontLoadedDmrsProc,
    uint8_t enableDeviceGraphLaunch,
    CUgraphExec* pSubSlotDeviceGraphExec,
    CUgraphExec* pFullSlotDeviceGraphExec,
    cuphyPuschRxWaitLaunchCfg_t* pWaitKernelLaunchCfgsPreSubSlot,
    cuphyPuschRxWaitLaunchCfg_t* pWaitKernelLaunchCfgsPostSubSlot,
    cuphyPuschRxDglLaunchCfg_t* pDglKernelLaunchCfgsPreSubSlot,
    cuphyPuschRxDglLaunchCfg_t* pDglKernelLaunchCfgsPostSubSlot,
    cudaStream_t strm);

// 獲取描述符信息
static void getDescrInfo(size_t& statDescrSizeBytes,
                         size_t& statDescrAlignBytes,
                         size_t& dynDescrSizeBytes,
                         size_t& dynDescrAlignBytes);

// 設置Early HARQ模式
void setEarlyHarqModeEnabled(bool earlyHarqModeEnabled);
```

#### 圖管理接口

```cpp
// 獲取計算圖管理器
IChestGraphNodes& chestGraph();
IChestStream& chestStream();
IChestSubSlotNodes& earlyHarqGraph();
IChestSubSlotNodes& frontDmrsGraph();
pusch::IStartKernels& startKernels();
```

---

## 核心數據結構

### 靜態描述符 (puschRxChEstStatDescr_t)

```cpp
struct puschRxChEstStatDescr_t final
{
    // 頻域插值係數張量
    puschRxChEstTensorPrm_t<...> tPrmFreqInterpCoefs;
    puschRxChEstTensorPrm_t<...> tPrmFreqInterpCoefs4;
    puschRxChEstTensorPrm_t<...> tPrmFreqInterpCoefsSmall;
    
    // 移位序列
    puschRxChEstTensorPrm_t<...> tPrmShiftSeq;
    puschRxChEstTensorPrm_t<...> tPrmUnShiftSeq;
    puschRxChEstTensorPrm_t<...> tPrmShiftSeq4;
    puschRxChEstTensorPrm_t<...> tPrmUnShiftSeq4;
    
    // 符號接收狀態指針
    const uint32_t* pSymbolRxStatus;
    
    // RKHS參數
    prbRkhsDesc_t prbRkhsDescs[MAX_N_PRBS_SUPPORTED];
    zpRkhsDesc_t zpRkhsDescs[NUM_RKHS_ZP];
};
```

### 動態描述符 (puschRxChEstDynDescr_t)

```cpp
struct puschRxChEstDynDescr_t final
{
    uint8_t chEstTimeInst;                        // 時域實例
    uint8_t dmrsSymPos[N_MAX_DMRS_SYMS];         // DMRS符號位置
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;    // UE群組參數
    
    // 異質配置到UE群組的映射
    uint32_t hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED];
    
    // Early HARQ等待內核狀態
    uint8_t* pPreSubSlotWaitKernelStatusGpu;
    uint8_t* pPostSubSlotWaitKernelStatusGpu;
    uint16_t waitTimeOutPreEarlyHarqUs;
    uint16_t waitTimeOutPostEarlyHarqUs;
    uint64_t mPuschStartTimeNs;                  // 啟動時間參考
    
    // RKHS參數
    rkhsUeGrpPrms_t rkhsUeGrpPrms[MAX_N_USER_GROUPS_SUPPORTED];
    rkhsComputeBlockPrms_t rkhsCompBlockPrms[MAX_N_USER_GROUPS_SUPPORTED];
    
    uint8_t nSymPreSubSlotWaitKernel;
    uint8_t nSymPostSubSlotWaitKernel;
};
```

### RKHS相關結構

**PRB RKHS描述符**
```cpp
struct prbRkhsDesc_t final
{
    uint8_t zpIdx;                          // 零填充索引
    __half sumEigValues;                    // 特徵值之和
    cuphyTensorInfo2_t tInfoEigVecCob;     // 特徵向量基變換
    cuphyTensorInfo2_t tInfoCorr_half_nZpDmrsSc;  // 特徵向量相關
    cuphyTensorInfo3_t tInfoCorr;          // 完整相關矩陣
    cuphyTensorInfo1_t tInfoEigVal;        // 特徵值
    cuphyTensorInfo2_t tInfoInterpCob;     // 插值基變換
};
```

**零填充RKHS描述符**
```cpp
struct zpRkhsDesc_t final
{
    cuphyTensorInfo2_t tInfoZpDmrsScEigenVec;        // 零填充特徵向量
    cuphyTensorInfo2_t tInfoZpInterpVec;            // 插值特徵向量
    cuphyTensorInfo2_t tInfoSecondStageTwiddleFactors; // 旋轉因子
    cuphyTensorInfo1_t tInfoSecondStageFourierPerm;    // 傅里葉排列
};
```

### 計算參數

**焦點相關參數**
```cpp
struct foccPrm_t final
{
    uint8_t layerIdx;  // 層索引
};

struct toccPrm_t final
{
    uint8_t foccBitMask;
    foccPrm_t foccPrms[2];
};

struct gridPrm_t final
{
    uint8_t toccBitMask;
    toccPrm_t toccPrms[2];
};
```

**計算塊通用參數**
```cpp
struct computeBlocksCommonPrms_t final
{
    uint16_t nPrb;                        // PRB數量
    uint8_t zpIdx;                        // 零填充索引
    rkhsNoiseEstMethod_t noiseEstMethod;  // 噪聲估計方法
    float nNoiseMeasurments;              // 噪聲測量總數
    uint16_t noiseRegionFirstIntIdx;      // 噪聲測量首個時間間隔
    uint16_t nNoiseIntsPerFocc;          // 每個FOCC的時間間隔
    uint16_t nNoiseIntsPerGrid;          // 每個網格的時間間隔
};
```

**RKHS計算塊參數**
```cpp
struct __align__(32) rkhsComputeBlockPrms_t
{
    uint16_t ueGrpIdx;                // UE群組索引
    uint16_t startInputPrb;           // 起始輸入PRB
    uint16_t nOutputSc;               // 輸出子載波數
    uint16_t startOutputScInBlock;    // 塊中起始輸出SC
    uint16_t scOffsetIntoChEstBuff;   // 到信道估計緩衝區的偏移
};
```

---

## 噪聲估計方法

```cpp
enum rkhsNoiseEstMethod_t
{
    USE_EMPTY_DMRS_GRID    = 0,  // 使用空DMRS網格
    USE_EMPTY_FOCC         = 1,  // 使用空FOCC
    USE_QUITE_FOCC_REGIONS = 2   // 使用安靜的FOCC區域
};
```

---

## 使用流程

```
1. 創建對象
   ↓
2. 初始化 (init)
   - 設置靜態描述符
   - 配置內核構建器
   ↓
3. 設置 (setup)
   - 配置動態參數
   - 構建計算圖
   ↓
4. 執行估計
   - 啟動CUDA內核
   - 計算信道估計
   ↓
5. 獲取結果
   - 從GPU讀取估計結果
```

---

## 關鍵特性

✅ **多算法支持** - RKHS、最小二乘、多階段、分窗口估計  
✅ **高度參數化** - 支持多層、多天線、多PRB配置  
✅ **Early HARQ支持** - 提前HARQ處理機制  
✅ **Device Graph Launch** - 異步GPU執行  
✅ **TensorRT集成** - 支持AI加速的信道估計  
✅ **計算圖管理** - 靈活的計算流程控制  
✅ **異步描述符複製** - CPU-GPU高效數據傳輸  
✅ **禁止複製** - 安全的資源管理  

---

## 配置和工廠

### 配置加載
```cpp
// YAML配置加載
ch_est_yaml_loader.cpp

// 設置管理
ch_est_settings.hpp
```

### 工廠模式
```cpp
// 使用工廠創建信道估計器
chest_factory.hpp/cpp
```

---

## TensorRT集成

支持使用TensorRT引擎加速信道估計：

```cpp
// TensorRT信道估計引擎
trtengine_chest.hpp/cpp

// 預處理/後處理轉換
ch_est_trtengine_pre_post_conversion.hpp
ch_est_trtengine_trtengine_pre_post_converion.cu
```

---

## 接口定義

### 模塊接口
```cpp
// IModule.hpp - 基礎模塊接口
```

### 圖管理接口
```cpp
// IGraph_mgr.hpp - 計算圖管理接口
```

### 流管理接口
```cpp
// IStream.hpp - CUDA流管理接口
```

---

## 記憶體對齐

某些結構使用對齐優化以提高緩存性能：

```cpp
struct __align__(32) rkhsUeGrpPrms_t      // 32字節對齐
struct __align__(32) rkhsComputeBlockPrms_t  // 32字節對齐
```

---

## 相關宏定義

```cpp
MAX_N_PRBS_SUPPORTED              // 最大PRB數
MAX_N_LAYERS_PUSCH                // PUSCH最大層數
N_MAX_DMRS_SYMS                   // 最大DMRS符號數
MAX_N_USER_GROUPS_SUPPORTED       // 最大UE群組數
NUM_RKHS_ZP                       // RKHS零填充數量
CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST // 最大時域信道估計實例
CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS  // 最大異質配置
```


## 延伸閱讀

- [TensorRT IModule文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/blob/main/cuPHY/src/cuphy/ch_est/TensorRT-IModule.md)
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
