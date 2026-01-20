# CFO/TA 估計器（CFO_TA_EST）

## 概述

CFO_TA_EST是NVIDIA cuPHY庫中用於**PUSCH（Physical Uplink Shared Channel）接收處理**的載波頻率偏移（CFO）和時序進階（TA）估計器組件。

## 文件結構

- **Header File**: `cfo_ta_est.hpp`
- **License**: Apache-2.0
- **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

## 核心概念

### 1. CFO（Carrier Frequency Offset）
- 估計並補償接收信號的載波頻率偏移
- 用於改進PUSCH信道估計的準確性

### 2. TA（Timing Advance）
- 與CFO估計同時進行的時序進階估計

### 3. 多層次內核選擇
- **L2層**：根據系統配置選擇合適的實現
- **L1層**：模板化的中層選擇
- **L0層**：具體的CUDA內核實現

---

## 公開API

### 構造和析構
```cpp
puschRxCfoTaEst()                          // 默認構造
~puschRxCfoTaEst()                         // 析構
puschRxCfoTaEst(const puschRxCfoTaEst&) = delete;  // 禁止複製
```

### 初始化
```cpp
void init(bool enableCpuToGpuDescrAsyncCpy,
          puschRxCfoTaEstStatDescr_t& statDescrCpu,
          void* pStatDescrGpu,
          cudaStream_t strm);
```
**功能**: 初始化CFO估計器和靜態組件描述符
- `enableCpuToGpuDescrAsyncCpy`: 是否啟用異步描述符複製
- `statDescrCpu`: CPU端靜態描述符
- `pStatDescrGpu`: GPU端靜態描述符指針
- `strm`: CUDA流

### 設置和配置
```cpp
cuphyStatus_t setup(cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
                    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
                    float** pFoCompensationBuffers,
                    uint16_t nUeGrps,
                    uint32_t nMaxPrb,
                    bool enableCpuToGpuDescrAsyncCpy,
                    puschRxCfoTaEstDynDescrVec_t& dynDescrVecCpu,
                    void* pDynDescrsGpu,
                    cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                    cudaStream_t strm);
```
**功能**: 設置對象狀態和動態描述符

**參數**:
| 參數 | 類型 | 說明 |
|------|------|------|
| `pDrvdUeGrpPrmsCpu` | `cuphyPuschRxUeGrpPrms_t*` | CPU端UE群組參數 |
| `pDrvdUeGrpPrmsGpu` | `cuphyPuschRxUeGrpPrms_t*` | GPU端UE群組參數 |
| `pFoCompensationBuffers` | `float**` | FO補償緩衝區 |
| `nUeGrps` | `uint16_t` | UE群組數 |
| `nMaxPrb` | `uint32_t` | 最大PRB數 |
| `enableCpuToGpuDescrAsyncCpy` | `bool` | 異步複製開關 |
| `dynDescrVecCpu` | `puschRxCfoTaEstDynDescrVec_t&` | CPU端動態描述符向量 |
| `pDynDescrsGpu` | `void*` | GPU端動態描述符指針 |
| `pLaunchCfgs` | `cuphyPuschRxCfoTaEstLaunchCfgs_t*` | 啟動配置 |
| `strm` | `cudaStream_t` | CUDA流 |

### 獲取描述符信息
```cpp
static void getDescrInfo(size_t& statDescrSizeBytes,
                         size_t& statDescrAlignBytes,
                         size_t& dynDescrSizeBytes,
                         size_t& dynDescrAlignBytes);
```
**功能**: 獲取靜態和動態描述符的大小和對齊信息

---

## 數據結構

### 靜態描述符
```cpp
struct puschRxCfoTaEstStatDescr
{
    // 靜態配置信息
};
typedef struct puschRxCfoTaEstStatDescr puschRxCfoTaEstStatDescr_t;
```

### 動態描述符
```cpp
struct puschRxCfoTaEstDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;    // UE群組參數指針
    uint16_t nUeGrps;                           // UE群組數
    float** pFoCompensationBuffers;             // FO補償緩衝區
};
typedef struct puschRxCfoTaEstDynDescr puschRxCfoTaEstDynDescr_t;
```

### 內核參數
```cpp
typedef struct _puschRxCfoTaEstKernelArgs
{
    puschRxCfoTaEstStatDescr_t* pStatDescr;   // 靜態描述符陣列
    puschRxCfoTaEstDynDescr_t* pDynDescr;     // 動態描述符陣列
} puschRxCfoTaEstKernelArgs_t;
```

### 張量參數
```cpp
template <size_t NDim>
struct puschRxCfoTaEstTensorPrm
{
    void* pAddr;           // 張量地址
    int strides[NDim];     // 步幅信息
};
```

---

## 私有方法

### 三層內核選擇機制

#### L2層選擇
```cpp
void kernelSelectL2(uint16_t nBSAnts,
                    uint8_t nLayers,
                    uint8_t nDmrsAddlnPos,
                    uint16_t nMaxPrb,
                    uint16_t nUeGrps,
                    cuphyDataType_t hEstType,
                    cuphyDataType_t cfoEstType,
                    cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);
```
根據基站天線數、層數、DMRS附加位置等參數選擇合適的實現。

#### L1層選擇（模板化）
```cpp
template <typename TStorage, typename TDataRx, typename TCompute>
void kernelSelectL1(uint16_t nBSAnts,
                    uint8_t nLayers,
                    uint8_t nDmrsAddlnPos,
                    uint16_t nMaxPrb,
                    uint16_t nUeGrps,
                    cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);
```

#### L0層選擇（完整模板化）
```cpp
template <typename TStorage, typename TDataRx, typename TCompute, uint32_t N_TIME_CH_EST>
void kernelSelectL0(uint16_t nBSAnts,
                    uint8_t nLayers,
                    uint16_t nMaxPrb,
                    uint16_t nUeGrps,
                    cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);
```

### CFO/TA低MIMO估計
```cpp
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,
          uint32_t N_LAYERS,
          uint32_t N_TIME_CH_EST>
void cfoTaEstLowMimo(uint16_t nMaxPrb,
                     uint16_t nUeGrps,
                     cuphyPuschRxCfoTaEstLaunchCfg_t& launchCfg);
```

### 內核啟動配置
```cpp
template <uint32_t N_LAYERS,
          uint32_t THRD_GRP_TILE_SIZE,
          uint32_t N_THRD_GRP_TILES_PER_LAYER,
          uint32_t N_PRB_PER_THRD_BLK>
void cfoTaEstLowMimoKernelLaunchGeo(uint16_t nMaxPrb,
                                    uint16_t nUeGrps,
                                    dim3& gridDim,
                                    dim3& blockDim);
```

---

## 使用流程

1. **創建對象**
   ```cpp
   cfo_ta_est::puschRxCfoTaEst cfoEstimator;
   ```

2. **初始化**
   ```cpp
   cfoEstimator.init(enableAsync, statDescr, pStatDescrGpu, stream);
   ```

3. **設置配置**
   ```cpp
   cfoEstimator.setup(pUeGrpParams, pUeGrpParamsGpu, foBuffers, 
                      nGroups, maxPrb, enableAsync, dynDescrs, 
                      pDynDescrsGpu, pLaunchCfgs, stream);
   ```

4. **執行估計**（通過啟動CUDA內核）

---

## 關鍵特性

✅ **支持異步CPU-GPU描述符複製**  
✅ **多層次的內核選擇機制**  
✅ **支持多層MIMO配置**  
✅ **高度參數化的模板設計**  
✅ **靈活的張量訪問方式**  
✅ **禁止複製構造和賦值**（使用delete）

---

## 配置宏

```cpp
#define USE_SPLIT_CFO_TA (CUPHY_ENABLE_SUB_SLOT_PROCESSING)
```
根據是否啟用子時隙處理來切換CFO/TA估計的實現方式。

---

## 相關類型定義

```cpp
using puschRxCfoTaEstDynDescrVec_t = 
    std::array<puschRxCfoTaEstDynDescr_t, CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS>;

using puschRxCfoTaEstKernelArgsArr_t = 
    std::array<puschRxCfoTaEstKernelArgs_t, CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS>;
```
