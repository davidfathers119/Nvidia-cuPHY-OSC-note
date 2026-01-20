# Channel Equalization (CHANNEL_EQ)

## 概述

CHANNEL_EQ（Channel Equalization）是NVIDIA cuPHY庫中用於**PUSCH（Physical Uplink Shared Channel）接收處理**的信道均衡組件。實現了MMSE（最小均方誤差）等化算法，用於信號檢測和軟解映射。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - channel_eq](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/channel_eq)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `channel_eq.hpp` | 信道均衡核心類定義（puschRxChEq） |
| `channel_eq.cu` | CUDA內核實現 |
| `channel_eq_types.cuh` | CUDA數據結構和類型定義 |

---

## 主要類

### puschRxChEq（信道均衡主類）

**繼承**: `cuphyPuschRxChEq`（不透明接口）

**功能**: 管理信道均衡和軟解映射的完整生命週期，支持多種QAM調製和MMSE算法。

#### 公開方法

```cpp
// 構造和析構
puschRxChEq();
~puschRxChEq() = default;
puschRxChEq(puschRxChEq const&) = delete;
puschRxChEq& operator=(puschRxChEq const&) = delete;

// 初始化
cuphyStatus_t init(
    cuphyContext_t ctx,
    cuphyTensorInfo2_t& tInfoDftBluesteinWorkspaceTime,
    cuphyTensorInfo2_t& tInfoDftBluesteinWorkspaceFreq,
    uint cudaDeviceArch,
    uint8_t enableDftSOfdm,
    uint8_t enableDebugEqOutput,
    bool enableCpuToGpuDescrAsyncCpy,
    void** ppStatDescrCpu,
    void** ppStatDescrGpu,
    void** ppIdftStatDescrCpu,
    void** ppIdftStatDescrGpu,
    cudaStream_t strm);

// 係數計算設置
cuphyStatus_t setupCoefCompute(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint16_t nMaxPrb,
    uint8_t enableCfoCorrection,
    uint8_t enablePuschTdi,
    bool enableCpuToGpuDescrAsyncCpy,
    void** ppDynDescrsCpu,
    void** ppDynDescrsGpu,
    cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
    cudaStream_t strm);

// 軟解映射設置
cuphyStatus_t setupSoftDemap(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint16_t nMaxPrb,
    uint8_t enableCfoCorrection,
    uint8_t enablePuschTdi,
    uint16_t symbolBitmask,
    bool enableCpuToGpuDescrAsyncCpy,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
    void* pDynDescrsGpu,
    cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
    cudaStream_t strm);

// 軟解映射（IDFT前處理）
cuphyStatus_t setupSoftDemapIdft(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint16_t nMaxPrb,
    uint cudaDeviceArch,
    uint8_t enableCfoCorrection,
    uint8_t enablePuschTdi,
    uint16_t symbolBitmask,
    bool enableCpuToGpuDescrAsyncCpy,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
    void* pDynDescrsGpu,
    cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
    cudaStream_t strm);

// 軟解映射（IDFT後處理）
cuphyStatus_t setupSoftDemapAfterDft(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
    uint16_t nUeGrps,
    uint16_t nMaxPrb,
    uint8_t enableCfoCorrection,
    uint8_t enablePuschTdi,
    uint16_t symbolBitmask,
    bool enableCpuToGpuDescrAsyncCpy,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
    void* pDynDescrsGpu,
    cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
    cudaStream_t strm);

// 獲取描述符信息
static void getDescrInfo(
    size_t& statDescrSizeBytes,
    size_t& statDescrAlignBytes,
    size_t& idftStatDescrSizeBytes,
    size_t& idftStatDescrAlignBytes,
    size_t& coefCompDynDescrSizeBytes,
    size_t& coefCompDynDescrAlignBytes,
    size_t& softDemapDynDescrSizeBytes,
    size_t& softDemapDynDescrAlignBytes);
```

---

## 核心數據結構

### QAM調製定義

```cpp
enum class QAM_t : uint8_t
{
    QAM_4   = CUPHY_QAM_4,      // 4-QAM (QPSK)
    QAM_16  = CUPHY_QAM_16,     // 16-QAM
    QAM_64  = CUPHY_QAM_64,     // 64-QAM
    QAM_256 = CUPHY_QAM_256     // 256-QAM
};
```

### 靜態描述符

```cpp
struct puschRxChEqStatDescr_t
{
    cudaTextureObject_t demapper_tex;    // 軟解映射查找表紋理
    uint8_t enableDebugEqOutput;         // 調試輸出開關
};

struct puschRxChEqIdftStatDescr_t
{
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceTime;  // 時域工作空間
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceFreq;  // 頻域工作空間
};
```

### 動態描述符

**係數計算**
```cpp
struct puschRxChEqCoefCompDynDescr_t
{
    uint8_t chEqTimeInstIdx;                          // 時域實例索引
    uint32_t hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // 異質配置映射
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;         // UE群組參數
};
```

**軟解映射**
```cpp
struct puschRxChEqSoftDemapDynDescr_t
{
    uint32_t hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // 異質配置映射
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;         // UE群組參數
};
```

### 內核參數結構

**係數計算內核參數**
```cpp
struct puschRxChEqCoefCompKernelArgs_t
{
    puschRxChEqStatDescr_t* pStatDescr;        // 靜態描述符
    puschRxChEqCoefCompDynDescr_t* pDynDescr;  // 動態描述符
};
```

**軟解映射內核參數**
```cpp
struct puschRxChEqSoftDemapKernelArgs_t
{
    puschRxChEqStatDescr_t* pStatDescr;         // 靜態描述符
    puschRxChEqSoftDemapDynDescr_t* pDynDescr;  // 動態描述符
};

struct puschRxChEqSoftDemapIdftKernelArgs_t
{
    puschRxChEqStatDescr_t* pStatDescr;         // 靜態描述符
    puschRxChEqSoftDemapDynDescr_t* pDynDescr;  // 動態描述符
    puschRxChEqIdftStatDescr_t* pIdftStatDescr; // IDFT靜態描述符
};
```

### 張量參數

```cpp
template <size_t NDim>
struct puschRxChEqTensorPrm
{
    void* pAddr;           // 張量基地址
    int strides[NDim];     // 步幅信息
};
template <size_t NDim>
using puschRxChEqTensorPrm_t = puschRxChEqTensorPrm<NDim>;
```

---

## 私有方法 - 係數計算

### 批量係數計算

```cpp
cuphyStatus_t batchEqCoefComp(
    uint32_t chEqInstIdx,
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
    uint16_t nUeGrps,
    uint32_t& nHetCfgs,
    puschRxChEqCoefCompDynDescrVec_t& dynDescrVecCpu);
```

### 內核選擇機制

**L1層選擇**
```cpp
void coefCompKernelSelectL1(
    uint16_t nBSAnts,        // 基站天線數
    uint8_t nLayers,         // 層數
    uint16_t nPrb,          // PRB數
    uint16_t nUeGrps,       // UE群組數
    cuphyDataType_t hEstType,     // 信道估計類型
    cuphyDataType_t coefType,     // 係數類型
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

**L0層選擇（模板化）**
```cpp
template <typename TStorageIn, typename TStorageOut, typename TCompute>
void coefCompKernelSelectL0(
    uint16_t nBSAnts,
    uint8_t nLayers,
    uint16_t nPrb,
    uint16_t nUeGrps,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

### MMSE等化實現

**大規模MIMO**
```cpp
template <typename TStorageIn, typename TStorageOut, typename TCompute,
          uint32_t N_BS_ANTS, uint32_t N_LAYERS>
void eqMmseCoefCompMassiveMimo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);

template <uint32_t N_LAYERS, uint32_t N_THRD_BLK_TONES, uint32_t N_TONES_PER_ITER>
void coefCompMassiveMimoKernelLaunchGeo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);
```

**高MIMO配置**
```cpp
template <typename TStorageIn, typename TStorageOut, typename TCompute,
          uint32_t N_BS_ANTS, uint32_t N_LAYERS>
void eqMmseCoefCompHighMimo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);

template <uint32_t N_THRD_BLK_PER_PRB, uint32_t N_TONES_PER_ITER,
          uint32_t N_THRDS_PER_TONE>
void coefCompHighMimoKernelLaunchGeo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);
```

**低MIMO配置**
```cpp
template <typename TStorageIn, typename TStorageOut, typename TCompute,
          uint32_t N_BS_ANTS, uint32_t N_LAYERS>
void eqMmseCoefCompLowMimo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);

template <uint32_t N_BS_ANTS, uint32_t N_LAYERS, uint32_t N_FREQ_BINS_PER_ITER>
void coefCompLowMimoKernelLaunchGeo(
    uint16_t nPrb,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);
```

---

## 私有方法 - 軟解映射

### 批量軟解映射

```cpp
cuphyStatus_t batchEqSoftDemap(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    uint32_t& nHetCfgs,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);

// IDFT前的軟解映射
cuphyStatus_t batchEqSoftDemapIdft(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    uint cudaDeviceArch,
    uint32_t& nHetCfgs,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);

// IDFT後的軟解映射
cuphyStatus_t batchEqSoftDemapAfterDft(
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    uint32_t& nHetCfgs,
    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);
```

### 軟解映射內核選擇

**L1層選擇**
```cpp
void softDemapKernelSelectL1(
    uint16_t nBSAnts,
    uint8_t nLayers,
    uint8_t Nd,              // QAM等級
    uint16_t nPrb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    cuphyDataType_t coefType,
    cuphyDataType_t dataRxType,
    cuphyDataType_t llrType,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);

// IDFT版本
void softDemapIdftKernelSelectL1(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    uint cudaDeviceArch,
    cuphyDataType_t coefType,
    cuphyDataType_t dataRxType,
    cuphyDataType_t llrType,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);

// IDFT後版本
void softDemapAfterDftKernelSelectL1(
    uint16_t nBSAnts,
    uint8_t nLayers,
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    cuphyDataType_t coefType,
    cuphyDataType_t dataRxType,
    cuphyDataType_t llrType,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

**L0層選擇（模板化）**
```cpp
template <typename TStorageIn, typename TDataRx, typename TStorageOut,
          typename TCompute>
void softDemapKernelSelectL0(
    uint16_t nBSAnts,
    uint8_t nLayers,
    uint8_t Nd,
    uint16_t Nprb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

### 軟解映射實現

**通用MMSE軟解映射**
```cpp
template <typename TStorageIn, typename TDataRx, typename TStorageOut,
          typename TCompute, uint32_t N_BS_ANTS>
void eqMmseSoftDemap(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nLayers,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

**IDFT版本**
```cpp
template <typename TStorageIn, typename TDataRx, typename TStorageOut,
          typename TCompute>
void eqMmseSoftDemapIdft(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    uint cudaDeviceArch,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

**IDFT後版本**
```cpp
template <typename TStorageIn, typename TDataRx, typename TStorageOut,
          typename TCompute, uint32_t N_BS_ANTS, uint32_t N_SYMBS_PER_THRD_BLK>
void eqMmseSoftDemapAfterDft(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    uint16_t symbolBitmask,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

**64比特版本**
```cpp
template <typename TStorageIn, typename TDataRx, typename TStorageOut,
          typename TCompute, uint32_t N_LAYERS, uint32_t N_SYMBS_PER_THRD_BLK>
void eqMmseSoftDemap_64R(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    cuphyPuschRxChEqLaunchCfg_t& launchCfg);
```

### 內核啟動配置

```cpp
void softDemapKernelLaunchGeo(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nLayers,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);

template <uint32_t N_SYMBS_PER_THRD_BLK>
void softDemapAfterDftKernelLaunchGeo(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);

template <uint32_t N_SYMBS_PER_THRD_BLK>
void softDemapKernelLaunchGeo_64R(
    uint8_t Nd,
    uint16_t nPrb,
    uint16_t nUeGrps,
    dim3& gridDim,
    dim3& blockDim);
```

---

## CUDA類型定義（channel_eq_types.cuh）

### 張量引用模板

```cpp
template <typename TElem>
struct tensor_ref
{
    TElem* addr;              // 數據基地址
    const int32_t* strides;   // 步幅

    // 支持1-5維索引操作
    CUDA_BOTH TElem& operator()(int i0);
    CUDA_BOTH TElem& operator()(int i0, int i1);
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2);
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2, int i3);
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2, int i3, int i4);
    
    // 常量版本
    CUDA_BOTH const TElem& operator()(int i0) const;
    // ... 其他維度
};
```

### 塊存儲模板

```cpp
// 1D塊（用於共享內存）
template <typename T, int M>
struct block_1D
{
    T data[M];
    CUDA_BOTH T& operator()(int idx);
};

// 2D塊
template <typename T, int M, int N>
struct block_2D
{
    T data[M * N];
    CUDA_BOTH T& operator()(int m, int n);
};

// 3D塊
template <typename T, int L, int M, int N>
struct block_3D
{
    T data[L * M * N];
    CUDA_BOTH T& operator()(int l, int m, int n);
};
```

### 複數運算

**32位複數（cuComplex）**
```cpp
// 基本運算
static CUDA_BOTH_INLINE cuComplex cuCmul(cuComplex x, cuComplex y);    // 複數乘法
static CUDA_BOTH_INLINE cuComplex cuCma(cuComplex x, cuComplex y, cuComplex a); // 乘加
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x);                // 共軛

// 算術操作
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y);
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y);
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex& x, cuComplex y);
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex& x, float y);
```

**半精度複數（__half2）**
```cpp
static CUDA_INLINE __half2 cuCmul(__half2 x, __half2 y);     // 半精度複數乘法
static CUDA_INLINE __half2 cuCma(__half2 x, __half2 y, __half2 a); // 半精度乘加
```

### 類型轉換

```cpp
template <typename T> CUDA_BOTH_INLINE T cuGet(int);       // 整數轉換
template <typename T> CUDA_BOTH_INLINE T cuGet(float);     // 浮點轉換
template <typename T> CUDA_BOTH_INLINE T cuGet(__half);    // 半精度轉換
template <typename T> CUDA_BOTH_INLINE T cuAbs(T);         // 絕對值
```

### 常量和配置

```cpp
static constexpr uint32_t CUDA_MAX_N_THRDS_PER_BLK = 1024;
static constexpr uint32_t N_THREADS_PER_WARP = 32;

// LLR（對數似然比）限制
static constexpr float LLR_LOW_LIM  = -65504.0f;  // 最小值
static constexpr float LLR_HIGH_LIM =  65504.0f;  // 最大值

// 零強制正則化反値（等效於SNR = 10^(3.6)的對角線MMSE）
static constexpr float INV_ZF_REGULARIZER = 3981.071705534973f;
```

### QAM映射（標籤分發）

```cpp
template <QAM_t>
struct QAMEnumToTagMap;

// 支持的QAM級別（僅偶數QAM，由3GPP支持）
template <>
struct QAMEnumToTagMap<QAM_t::QAM_4>   { struct QAM4Tag {}; };

template <>
struct QAMEnumToTagMap<QAM_t::QAM_16>  { struct QAM16Tag {}; };

template <>
struct QAMEnumToTagMap<QAM_t::QAM_64>  { struct QAM64Tag {}; };

template <>
struct QAMEnumToTagMap<QAM_t::QAM_256> { struct QAM256Tag {}; };
```

---

## 每個線程的頻率賓數計算

```cpp
template <uint32_t N_BS_ANTS, uint32_t N_LAYERS>
constexpr uint32_t getThreadsPerFreqBin()
{
    // 選擇32的除數，確保每個頻率賓完全適配一個warp
    constexpr uint32_t need = std::max(N_BS_ANTS * N_LAYERS, 
                                        N_BS_ANTS + N_LAYERS);
    if constexpr (need <= 1)  return 1;
    else if constexpr (need <= 2) return 2;
    else if constexpr (need <= 4)  return 4;
    else if constexpr (need <= 8)  return 8;
    else if constexpr (need <= 16) return 16;
    else return 32;
}
```

---

## 異質配置管理

### 係數計算異質配置

```cpp
struct puschRxChEqCoefCompHetCfg_t
{
    CUfunction func;        // CUDA函數指針
    uint16_t nMaxPrb;       // 最大PRB數
    uint16_t nUeGrps;       // UE群組數
};
```

### 軟解映射異質配置

```cpp
struct puschRxChEqSoftDemapHetCfg_t
{
    CUfunction func;        // CUDA函數指針
    uint16_t nMaxPrb;       // 最大PRB數
    uint16_t nMaxDataSym;   // 最大數據符號數
    uint16_t nMaxLayers;    // 最大層數
    uint16_t nUeGrps;       // UE群組數
};
```

### 哈希表結構

```cpp
struct puschRxChEqHash_t
{
    std::size_t operator()(const std::tuple<int, int>& comb) const
    {
        return std::get<0>(comb) ^ (std::get<1>(comb) * 17);
    }
};

struct chEqHashVal
{
    CUfunction func;
    int32_t hetCfgIdx;
};

// 哈希表類型
using chEqHashMap_t = std::unordered_map<std::tuple<int, int>, 
                                         chEqHashVal, 
                                         puschRxChEqHash_t>;
using softDemapperMap_t = std::unordered_map<int, chEqHashVal>;
```

---

## 使用流程

```
1. 創建對象
   puschRxChEq eq;
   ↓
2. 初始化 (init)
   - 配置Blustein DFT工作空間
   - 設置紋理對象
   ↓
3. 設置係數計算 (setupCoefCompute)
   - 配置MMSE等化參數
   ↓
4. 設置軟解映射 (setupSoftDemap*)
   - 根據QAM級別配置
   - 支持IDFT前/後處理
   ↓
5. 執行均衡和檢測
   - 計算均衡係數
   - 軟解映射到LLR
   ↓
6. 獲取結果
   - 讀取對數似然比（LLR）
```

---

## 關鍵特性

✅ **多種MIMO配置** - 大規模、高、低MIMO  
✅ **MMSE等化** - 最優檢測算法  
✅ **多QAM支持** - 4/16/64/256-QAM  
✅ **軟解映射** - 生成LLR供後續處理  
✅ **IDFT集成** - 支持時域處理  
✅ **異質配置** - 動態內核選擇  
✅ **CFO補償** - 載波頻率偏移修正  
✅ **TDI支持** - 時間差分干涉測量  
✅ **調試輸出** - 可選的均衡後輸出  
✅ **異步操作** - 支持GPU流同步  

---

## 性能優化

- **三層內核選擇** - L1/L0自適應選擇
- **Warp優化** - 每個Warp內頻率賓對齐
- **共享內存** - 高效局部存儲
- **紋理緩存** - 軟解映射查表加速
- **哈希表緩存** - 避免重複編譯
- **64位優化** - 特定架構版本

---

## 編譯配置

```cpp
// 調試模式（可選）
// #define CUPHY_DEBUG 1

// CUDA設備架構檢測
uint cudaDeviceArch;  // 傳入架構代碼
```

---

## 相關宏定義

```cpp
MAX_N_USER_GROUPS_SUPPORTED        // 最大UE群組數
CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ   // 最大時域均衡實例
CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS  // 最大異質配置數

// 異質配置計數
// nRxAnt配置數 = 14（根據天線和層的組合）
// 時域配置數 = 4
// 最大總數 = 14 * 4 = 56（實際上限為8）
```

---

## 延伸閱讀

- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- CFO估計器 - 前置處理階段
- 信道估計器 - 信道獲取
