# SRS Channel Estimation (cuPHY 上行聲測參考信號信道估計模組)

## 概述

**SRS Channel Estimator (srsChEst)** 是 NVIDIA cuPHY 中用於上行 Sounding Reference Signal (SRS) 信道估計的 GPU 加速模組。SRS 是 5G NR 中的聲測參考信號，用於測量上行鏈路信道質量，並支持波束成形權重計算和信道狀態信息 (CSI) 反饋。該模組支持多個天線端口、多種疊頻組合 (Comb) 配置，以及高級的 RKHS (Reproducing Kernel Hilbert Space) 信道估計演算法。

**核心特性：**
- 支持 OFDM 疊頻組合 (Comb-2/Comb-4)
- 多天線端口支持 (1/2/4/6/8/12 端口)
- 標準濾波器和 RKHS 兩種估計演算法
- FP16 精度優化
- 時延偏移校正
- RB 級 SNR 報告和寬帶統計
- CUDA 圖支持

---

## 軟體架構

### 1. 核心類別

#### `srsChEst`

SRS 信道估計的主要 C++ 類，實現信道估計演算法核心邏輯。

```cpp
class srsChEst : public cuphySrsChEst {
public:
    srsChEst();
    ~srsChEst() = default;
    
    void init(cuphySrsFilterPrms_t* pSrsFilterPrms,
              cuphySrsRkhsPrms_t* pRkhsPrms,
              cuphySrsChEstAlgoType_t chEstAlgo,
              uint8_t chEstToL2NormalizationAlgo,
              float chEstToL2ConstantScaler,
              uint8_t enableDelayOffsetCorrection,
              bool enableCpuToGpuDescrAsyncCpy,
              srsChEstStatDescr_t* pCpuStatDesc,
              void* pGpuStatDesc,
              cudaStream_t strm);
    
    cuphyStatus_t setup(uint16_t nSrsUes,
                        cuphyUeSrsPrm_t* h_srsUePrms,
                        uint16_t nCells,
                        cuphyTensorPrm_t* pTDataRx,
                        cuphySrsCellPrms_t* h_srsCellPrms,
                        float* d_rbSnrBuff,
                        uint32_t* h_rbSnrBuffOffsets,
                        cuphySrsReport_t* d_pSrsReports,
                        cuphySrsChEstBuffInfo_t* h_chEstBuffInfo,
                        void** d_addrsChEstToL2Buff,
                        cuphySrsChEstToL2_t* h_chEstToL2,
                        void* d_workspace,
                        bool enableCpuToGpuDescrAsyncCpy,
                        srsChEstDynDescr_t* pCpuDynDesc,
                        void* pGpuDynDesc,
                        cuphySrsChEstLaunchCfg_t* pLaunchCfg,
                        cudaStream_t strm);
    
    void kernelSelect(srsChEstDynDescr_t* pCpuDynDesc,
                      uint16_t nSrsUes,
                      uint16_t nCompBlocks,
                      uint16_t nRkhsCompBlocks,
                      cuphySrsChEstLaunchCfg_t* pLaunchCfg,
                      cuphySrsChEstNormalizationLaunchCfg_t* pNormalizationLaunchCfg);
    
    static void getDescrInfo(size_t& statDescrSizeBytes,
                             size_t& statDescrAlignBytes,
                             size_t& dynDescrSizeBytes,
                             size_t& dynDescrAlignBytes);
};
```

**主要方法：**
- `init()`: 初始化信道估計器，配置濾波器參數和 RKHS 參數
- `setup()`: 為特定的 UE 和信元配置動態參數
- `kernelSelect()`: 選擇適當的 CUDA 核心和啟動配置
- `getDescrInfo()`: 獲取描述符大小和對齐要求

### 2. 數據結構

#### 靜態描述符 (srsChEstStatDescr_t)

```cpp
struct srsChEstStatDescr_t {
    // FOCC (Frequency Orthogonal Comb) 查表
    tensor_ref_any<CUPHY_C_16F> tFocc_table;
    tensor_ref_any<CUPHY_C_16F> tFocc_comb2_table;
    tensor_ref_any<CUPHY_C_16F> tFocc_comb4_table;
    
    // 濾波器係數 (Comb-2 配置)
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts4_wide;
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts8_wide;
    
    // 濾波器係數 (Comb-4 配置)
    tensor_ref_any<CUPHY_C_16F> tW_comb4_nPorts1_wide;
    tensor_ref_any<CUPHY_C_16F> tW_comb4_nPorts2_wide;
    tensor_ref_any<CUPHY_C_16F> tW_comb4_nPorts4_wide;
    
    // 窄帶濾波器變體
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F> tW_comb2_nPorts2_narrow;
    tensor_ref_any<CUPHY_C_16F> tW_comb4_nPorts1_narrow;
    tensor_ref_any<CUPHY_C_16F> tW_comb4_nPorts2_narrow;
    
    // 雜訊估計去偏參數
    float noisEstDebias_comb2_nPorts1;
    float noisEstDebias_comb2_nPorts2;
    float noisEstDebias_comb4_nPorts1;
    float noisEstDebias_comb4_nPorts2;
    
    // RKHS 參數
    rkhsGridDesc_t rkhsGridDescs[NUM_RKHS_GRIDS];
    
    // 配置參數
    uint8_t chEstToL2NormalizationAlgo;
    float chEstToL2ConstantScaler;
    uint8_t enableDelayOffsetCorrection;
};
```

#### 動態描述符 (srsChEstDynDescr_t)

```cpp
struct srsChEstDynDescr_t {
    // UE 級描述符 (每個 UE 最多 CUPHY_SRS_MAX_N_USERS)
    ueDescr_t ueDescrs[CUPHY_SRS_MAX_N_USERS];
    
    // UE 組描述符 (相同配置的 UE 分組)
    ueGroupDescr_t ueGroupDescrs[CUPHY_SRS_MAX_N_USERS];
    
    // 信元描述符 (每個信元最多 MAX_N_SRS_CELL)
    cellDescr_t cellDescrs[MAX_N_SRS_CELL];
    
    // 計算塊描述符 (疊頻估計)
    compBlockDescr_t compBlockDescrs[MAX_N_COMP_BLOCKS];
    
    // RKHS 計算塊描述符
    rkhsCompBlockDescriptor_t rkhsCompBlockDescrs[MAX_N_SRS_RKHS_COMP_BLOCKS];
    
    int nSrsUes;  // 有效 UE 數量
};
```

#### UE 描述符 (ueDescr_t)

```cpp
struct ueDescr_t {
    // SRS 接收符號 (按 PRB 分組、天線、子載波組織)
    tensor_ref_any<CUPHY_C_16F> tSrsRxSymbols;
    
    // 輸出信道估計 (可選 FP32 或 FP16 精度)
    #ifdef ASIM_CUPHY_SRS_OUTPUT_FP32
        tensor_ref_any<CUPHY_C_32F> tChEstBuff;
        tensor_ref_any<CUPHY_C_32F> tChEstToL2;
    #else
        tensor_ref_any<CUPHY_C_16F> tChEstBuff;
        tensor_ref_any<CUPHY_C_16I> tChEstToL2;
    #endif
    
    // SRS 參數
    uint8_t combSize;  // 疊頻大小 (2 或 4)
    uint16_t nSrsSubcarriers;  // SRS 子載波數
    uint16_t nRxAntennas;  // 接收天線數
    
    // 報告指針
    cuphySrsReport_t* pUeSrsReport;
    
    // L2 標準化參數
    uint16_t nRxAntSrsL2;
    uint16_t nPrbGrpsL2;
    uint8_t nAntPortsL2;
    uint16_t prgIdxMappingL2[CUPHY_SRS_MAX_N_PRGS_SUPPORTED];
    uint8_t portIdxMappingL2[MAX_N_ANT_PORTS];
};
```

---

## SRS 信道估計演算法

### 1. 基本流程

**步驟 1: FOCC 乘以濾波器**

將接收的 SRS 符號乘以 FOCC 序列和估計濾波器：

$$\tilde{H}_{fc}[k,n] = \sum_{i=0}^{L-1} h_i \cdot \text{FOCC}(k_i) \cdot y(k_i, n)$$

其中：
- $h_i$ 是估計濾波器係數
- FOCC 是頻率正交組合序列
- $y(k_i, n)$ 是接收的 SRS 符號
- $k$ 是子載波索引，$n$ 是天線索引

**步驟 2: 子載波聚合**

沿著 SRS 子載波聚合估計值：

$$H_{est}[f,n] = \sum_{k \in \text{SRS}} \tilde{H}_{fc}[k,n]$$

**步驟 3: 雜訊估計與去偏**

使用空子載波或空 FOCC 時間槽估計雜訊：

$$\hat{\sigma}^2 = \frac{1}{K_{\text{empty}}} \sum_{k \in \text{empty}} |\tilde{H}_{fc}[k,n]|^2$$

應用去偏校正因子以提高估計精度。

### 2. 疊頻配置

#### Comb-2 (2 子載波間隔)

```
SRS 子載波: |X|_|X|_|X|_|X|_|X|_
           0 2 4 6 8 ...
```

- 使用 `nPorts1/2/4/8_wide` 濾波器
- 適合中等頻道相干帶寬

#### Comb-4 (4 子載波間隔)

```
SRS 子載波: |X|_|_|_|X|_|_|_|X|_
           0     4     8    ...
```

- 使用 `nPorts1/2/4/6/12_wide` 濾波器
- 適合寬頻道變化

### 3. 時延偏移校正

當 `enableDelayOffsetCorrection = 1` 時，在頻域應用相位旋轉以補償時延偏移：

$$y'(k,n) = e^{j2\pi \Delta \tau k} \cdot y(k,n)$$

其中 $\Delta \tau$ 是估計的時延偏移。

校正相位在共享記憶體中預計算為 LUT：

```cuda
__half2 phase_conj = sh_phaseLUT[portIdx * (nSrsScBlock + 1) + inputScIdx];
inputSignal = complex_mul(phase_conj, inputSignal);
```

---

## CUDA 核心實現

### 主要核心函數

#### srsChEstKernel

標準 SRS 信道估計核心 (疊頻方法)

```cu
__global__ void srsChEstKernel(srsChEstStatDescr_t* pStatDescr,
                               srsChEstDynDescr_t* pDynDescr)
{
    // 塊索引對應計算塊
    const uint32_t compBlockIdx = blockIdx.x;
    compBlockDescr_t& compBlockDescr = 
        pDynDescr->compBlockDescrs[compBlockIdx];
    
    // 獲取 UE 組和 UE 信息
    uint16_t ueGroupIdx = compBlockDescr.ueGroupIdx;
    ueGroupDescr_t& ueGroupDescr = 
        pDynDescr->ueGroupDescrs[ueGroupIdx];
    
    // 第一個 UE 決定疊頻大小和天線端口
    uint16_t firstUeInBlockIdx = ueGroupDescr.ueIdxs[0];
    uint8_t combSize = pDynDescr->ueDescrs[firstUeInBlockIdx].combSize;
    uint8_t nAntPorts = ueGroupDescr.nAntPorts;
    
    // 調用模板特化的內核
    if(combSize == 2 && nAntPorts == 1)
        srsChEstKernelInner<2, 1>(pStatDescr, pDynDescr);
    else if(combSize == 2 && nAntPorts == 2)
        srsChEstKernelInner<2, 2>(pStatDescr, pDynDescr);
    // ... 其他組合
    else if(combSize == 4 && nAntPorts == 12)
        srsChEstKernelInner<4, 12>(pStatDescr, pDynDescr);
}
```

#### srsChEstKernelInner 設備函數

每個 (combSize, nAntPorts) 組合的特化版本

```cu
template <int combSize, int nAntPorts>
__device__ __forceinline__ void srsChEstKernelInner(
    srsChEstStatDescr_t* pStatDescr,
    srsChEstDynDescr_t* pDynDescr)
{
    // 線程塊級別同步
    cg::thread_block thisThrdBlk = cg::this_thread_block();
    cg::thread_block_tile<32> tile = 
        cg::tiled_partition<32>(thisThrdBlk);
    
    // 載入濾波器參數到共享記憶體
    // ...
    
    // 步驟 1: 載入 SRS 符號並乘以 FOCC
    for(int inputScIdx = 0; inputScIdx < nSrsScBlock; inputScIdx++) {
        __half2 inputSignal = *srs;
        
        // 時延偏移校正
        if(correctDelayOffsetFlag == 1) {
            __half2 phase_conj = 
                sh_phaseLUT[portIdx * (nSrsScBlock + 1) + inputScIdx];
            inputSignal = complex_mul(phase_conj, inputSignal);
        }
        
        // 與 FOCC 和濾波器係數相乘
        const __half2* foccBase = 
            sh_focc_table + foccIdx * (FOCC_LENGTH + 1);
        const __half2 focc = foccBase[inputScIdx % FOCC_LENGTH];
        
        est = __hcmadd(complex_conjmul(*w, focc), inputSignal, est);
        w++;
        srs++;
    }
    
    // 步驟 2: 寫入估計結果
    sh_Hest[i] = est;
}
```

#### srsRkhsChEstKernel

RKHS 信道估計核心

```cu
__global__ void srsRkhsChEstKernel(srsChEstStatDescr_t* pStatDescr,
                                   srsChEstDynDescr_t* pDynDescr)
{
    // 線程指數
    uint16_t THREAD_IDX = threadIdx.x;
    uint16_t BLOCK_IDX = blockIdx.x;
    uint8_t WARP_IDX = THREAD_IDX / 32;
    uint8_t LANE_IDX = THREAD_IDX % 32;
    
    // 協作組
    cg::thread_block thisThrdBlk = cg::this_thread_block();
    cg::thread_block_tile<32> tile = 
        cg::tiled_partition<32>(thisThrdBlk);
    
    // 獲取 RKHS 計算塊信息
    rkhsCompBlockDescriptor_t& compBlock = 
        pDynDescr->rkhsCompBlockDescrs[BLOCK_IDX];
    
    uint16_t ueIdx = compBlock.ueIdx;
    uint8_t combIdx = compBlock.combIdx;
    
    // RKHS 特定計算
    // ...
}
```

### 核心啟動配置

**標準疊頻核心：**
```cuda
dim3 gridDim(nCompBlocks);  // X: 計算塊數
dim3 blockDim(SRS_CHEST_BLOCK_SZ);  // 每塊執行緒數 (~256-512)
```

**RKHS 核心：**
```cuda
dim3 gridDim(nRkhsCompBlocks);
dim3 blockDim(1024);
```

### 共享記憶體使用

```cpp
// 標準疊頻核心
size_t sharedMemBytes = 
    (nMaxSrsScBlock * nMaxSrsScBlock * sizeof(__half2)) +  // W_{wide,narrow}
    (max_nPorts * sizeof(float)) +                          // phaseRamp
    (max_nPorts * sizeof(__half2)) +                        // avgScCorr
    ((13 * 12) * sizeof(__half2)) +                         // focc_table
    (max_nPorts * N_PRB_PER_COMP_BLK * sizeof(float)) +     // avgSignalEnergyPrb
    (max_nPorts * max_nSrsScBlock * sizeof(float)) +        // avgSignalEnergySc
    (max_nPorts * sizeof(float)) +                          // avgSignalEnergy
    (max_nPorts * sizeof(uint32_t)) +                       // ueBlockCntr
    (max_nPorts * (SRS_CHEST_BLOCK_SZ / 32) * sizeof(__half2)) +  // tile_avgScCorr
    ((max_nSrsScBlock + 1) * 12 * sizeof(__half2));         // phaseTable
```

---

## 數據流

### 輸入張量

1. **SRS 接收符號** (`tSrsRxSymbols`)
   - 維度: [PRB_GROUPS] x [ANTENNAS] x [SRS_SUBCARRIERS]
   - 類型: `CUPHY_C_16F` (複數 FP16)
   - 佈局: 行優先

2. **天線端口** (`nAntPorts`)
   - 支援: 1, 2, 4, 6, 8, 12

3. **疊頻配置** (`combSize`)
   - 值: 2 或 4

### 輸出張量

1. **信道估計** (`tChEstBuff`)
   - 維度: [PRB_GROUPS] x [ANTENNAS] x [UE_ANTENNAS]
   - 類型: `CUPHY_C_16F` 或 `CUPHY_C_32F`

2. **標準化信道估計** (`tChEstToL2`)
   - 用於 L2 標準化
   - 可選 FP16 或 FP32 精度

3. **報告** (`srsReport`)
   - 寬帶 SNR、相關係數等

4. **RB SNR** (`rbSnrBuffer`)
   - 每個 RB 的信噪比
   - 輸出形狀: [273 PRBs] x [nSrsUes]

---

## C API 接口

### `cuphyCreateSrsChEst()`

創建並初始化 SRS 信道估計器

```cpp
cuphyStatus_t cuphyCreateSrsChEst(
    cuphySrsChEstHndl_t* pSrsChEstHndl,
    cuphySrsFilterPrms_t* pSrsFilterPrms,
    cuphySrsRkhsPrms_t* pRkhsPrms,
    cuphySrsChEstAlgoType_t chEstAlgo,
    uint8_t chEstToL2NormalizationAlgo,
    float chEstToL2ConstantScaler,
    uint8_t enableDelayOffsetCorrection,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    void* pCpuStatDesc,
    void* pGpuStatDesc,
    cudaStream_t strm);
```

**參數：**
- `pSrsChEstHndl`: 返回的估計器句柄
- `pSrsFilterPrms`: SRS 濾波器參數
- `pRkhsPrms`: RKHS 參數 (如需要)
- `chEstAlgo`: 估計演算法選擇
  - `SRS_CH_EST_ALGO_TYPE_STANDARD`: 標準濾波器
  - `SRS_CH_EST_ALGO_TYPE_RKHS`: RKHS 方法
- `chEstToL2NormalizationAlgo`: L2 標準化類型
- `chEstToL2ConstantScaler`: L2 縮放因子
- `enableDelayOffsetCorrection`: 時延偏移校正開關
- `enableCpuToGpuDescrAsyncCpy`: 非同步描述符複製

### `cuphySetupSrsChEst()`

為特定 UE 和信元配置估計器

```cpp
cuphyStatus_t cuphySetupSrsChEst(
    cuphySrsChEstHndl_t srsChEstHndl,
    uint16_t nSrsUes,
    cuphyUeSrsPrm_t* h_srsUePrms,
    uint16_t nCell,
    cuphyTensorPrm_t* pTDataRx,
    cuphySrsCellPrms_t* h_srsCellPrms,
    float* d_rbSnrBuff,
    uint32_t* h_rbSnrBuffOffsets,
    cuphySrsReport_t* d_pSrsReports,
    cuphySrsChEstBuffInfo_t* h_chEstBuffInfo,
    void** d_addrsChEstToL2Buff,
    cuphySrsChEstToL2_t* h_chEstToL2,
    void* d_workspace,
    bool enableCpuToGpuDescrAsyncCpy,
    srsChEstDynDescr_t* pCpuDynDesc,
    void* pGpuDynDesc,
    cuphySrsChEstLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);
```

### `cuphySrsChEstGetDescrInfo()`

獲取描述符大小和對齐要求

```cpp
cuphyStatus_t cuphySrsChEstGetDescrInfo(
    size_t* statDescrSizeBytes,
    size_t* statDescrAlignBytes,
    size_t* dynDescrSizeBytes,
    size_t* dynDescrAlignBytes);
```

---

## 使用範例

### 基本 SRS 信道估計

```cpp
// 1. 初始化上下文
cuphyContext_t ctx;
cuphyCreateContext(&ctx, 0);  // GPU 0

// 2. 準備 SRS 濾波器參數
cuphySrsFilterPrms_t srsFilterPrms;
// ... 載入濾波器係數

// 3. 準備 RKHS 參數 (如需要)
cuphySrsRkhsPrms_t rkhsPrms;
// ... 配置 RKHS 參數

// 4. 獲取描述符大小
size_t statDescrSizeBytes, statDescrAlignBytes;
size_t dynDescrSizeBytes, dynDescrAlignBytes;
cuphySrsChEstGetDescrInfo(&statDescrSizeBytes, &statDescrAlignBytes,
                          &dynDescrSizeBytes, &dynDescrAlignBytes);

// 5. 分配描述符緩衝區
cuphy::buffer<uint8_t, cuphy::pinned_alloc> 
    statDescrBufCpu(statDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::device_alloc> 
    statDescrBufGpu(statDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::pinned_alloc> 
    dynDescrBufCpu(dynDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::device_alloc> 
    dynDescrBufGpu(dynDescrSizeBytes);

// 6. 創建 SRS 信道估計器
cuphySrsChEstHndl_t srsChEstHndl;
cuphyCreateSrsChEst(&srsChEstHndl,
                    &srsFilterPrms,
                    &rkhsPrms,
                    SRS_CH_EST_ALGO_TYPE_STANDARD,
                    0,  // normalization algo
                    1.0f,  // constant scaler
                    1,  // enable delay offset correction
                    0,  // enable async copy
                    statDescrBufCpu.addr(),
                    statDescrBufGpu.addr(),
                    cuStrm);

// 7. 複製靜態描述符到 GPU
cudaMemcpyAsync(statDescrBufGpu.addr(),
                statDescrBufCpu.addr(),
                statDescrSizeBytes,
                cudaMemcpyHostToDevice,
                cuStrm);

// 8. 準備 UE 和信元參數
std::vector<cuphyUeSrsPrm_t> ueSrsPrms(nSrsUes);
std::vector<cuphySrsCellPrms_t> cellPrms(nCells);
// ... 填充參數

// 9. 分配輸出緩衝區
std::vector<float*> chEstToL2Vec(nSrsUes);
for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {
    size_t maxChEstSize = 273 * 128 * 4 * sizeof(float2);
    cudaMalloc(&chEstToL2Vec[ueIdx], maxChEstSize);
}

// 10. 配置估計器
cuphySetupSrsChEst(srsChEstHndl,
                   nSrsUes,
                   ueSrsPrms.data(),
                   nCells,
                   pTDataRx,
                   cellPrms.data(),
                   d_rbSnrBuff,
                   h_rbSnrBuffOffsets,
                   d_srsReports,
                   h_chEstBuffInfo,
                   chEstToL2Vec.data(),
                   h_chEstToL2,
                   d_workspace,
                   false,
                   (srsChEstDynDescr_t*)dynDescrBufCpu.addr(),
                   dynDescrBufGpu.addr(),
                   &launchCfg,
                   cuStrm);

// 11. 執行信道估計
// 使用 CUDA 驅動 API 啟動核心
cuLaunchKernel(launchCfg.kernelNodeParamsDriver.func,
               launchCfg.kernelNodeParamsDriver.gridDimX,
               launchCfg.kernelNodeParamsDriver.gridDimY,
               launchCfg.kernelNodeParamsDriver.gridDimZ,
               launchCfg.kernelNodeParamsDriver.blockDimX,
               launchCfg.kernelNodeParamsDriver.blockDimY,
               launchCfg.kernelNodeParamsDriver.blockDimZ,
               launchCfg.kernelNodeParamsDriver.sharedMemBytes,
               cuStrm,
               launchCfg.kernelArgs,
               NULL);

// 12. 同步並獲取結果
cudaStreamSynchronize(cuStrm);

// 13. 清理
for(auto ptr : chEstToL2Vec) cudaFree(ptr);
cuphyDestroySrsChEst(srsChEstHndl);
cuphyDestroyContext(ctx);
```

---

## Python 綁定接口

### `PySrsChannelEstimator`

Python 用戶可透過 pybind11 包裝使用 SRS 信道估計：

```python
from pyaerial.phy5g.algorithms import SrsChannelEstimator

# 初始化
estimator = SrsChannelEstimator(
    chest_algo_idx=0,  # 標準演算法
    enable_delay_offset_correction=1,
    chest_params=chest_params_dict,
    cuda_stream=cuda_stream
)

# 執行估計
ch_ests, rb_snrs, srs_reports = estimator.estimate(
    rx_data=rx_data,  # CuPy 複數陣列
    num_srs_ues=num_ues,
    num_srs_cells=num_cells,
    num_prb_grps=num_prb_groups,
    start_prb_grp=start_prb_group,
    srs_cell_prms=cell_prms,
    srs_ue_prms=ue_prms
)

# 獲取輔助信息
rb_snr_buffer = estimator.get_rb_snr_buffer()  # [273 PRBs, num_ues]
srs_reports = estimator.get_srs_report()  # 寬帶統計
```

---

## 效能特性

### 計算複雜度

**每個 SRS 傳輸的運算數：**
- 乘法: O(L_SRS × L_filter × nAntennas)
- L_SRS: SRS 子載波數 (~264)
- L_filter: 濾波器長度 (~6-12)
- nAntennas: 天線數 (1-12)

### GPU 資源使用

- **暫存器**: ~120-180 個 (含共享記憶體尋址)
- **共享記憶體**: 8-16 KB (FOCC、濾波器、臨時結果)
- **線程塊大小**: 256-512 執行緒

### 記憶體帶寬

- **讀取**: SRS 符號 + 濾波器係數
- **寫入**: 信道估計 + RB SNR 值
- **典型帶寬**: 50-100 GB/s

---

## 故障排除

### 常見問題

**Q: 信道估計 SNR 較低？**
- A: 驗證 SRS 接收功率水平
- 檢查濾波器係數是否正確載入
- 確認天線端口和疊頻配置

**Q: 時延偏移校正效果不理想？**
- A: 調整 `enableDelayOffsetCorrection` 參數
- 驗證時延估計精度
- 檢查 LUT 預計算

**Q: 記憶體不足？**
- A: 減小批次大小
- 使用流式處理分塊數據
- 啟用 L2 標準化以減少輸出大小

---

## 參考資源

- [NVIDIA cuPHY 文檔](https://docs.nvidia.com/networking/aerial/)
- [3GPP TS 38.211 - NR 物理層](https://www.3gpp.org)
- [3GPP TS 38.214 - NR 物理層流程](https://www.3gpp.org)
- 相關模組: `srs_tx`, `soft_demapper`, `rate_matching`

