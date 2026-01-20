# NVIDIA cuPHY Simplex 解碼器（Simplex Decoder）模組

## 概述

Simplex 解碼器是 NVIDIA Aerial cuPHY 的核心誤差更正編碼（Error Correction Code, ECC）模組，專門用於解碼 Simplex 碼和經由擴展 Hamming 碼衍生的碼字。該模組在 5G NR 物理層中用於 PUSCH（Physical Uplink Shared Channel）的 HARQ（混合自動重傳請求）和 CSI（通道狀態信息）解碼。

**主要功能：**
- Simplex 碼解碼（K=1，K=2 支持）
- QAM 調制適應（Qm 變量）
- 置信度評估（Confidence Level）
- DTX（不傳送）偵測
- GPU 優化的平行處理

---

## 架構概述

### 核心組件

#### 1. **SimplexDecoder 主類**

```cpp
class SimplexDecoder : public cuphySimplexDecoder
{
public:
    // 構造函數
    SimplexDecoder();
    
    // 設置解碼器
    void setup(uint16_t                        nCws,      // 碼字數
               cuphySimplexCwPrm_t*            pCwPrmsCpu,      // CPU 碼字參數
               cuphySimplexCwPrm_t*            pCwPrmsGpu,      // GPU 碼字參數
               bool                            enableCpuToGpuDescrAsyncCpy,
               simplexDecoderDynDescr_t*       pCpuDynDesc,
               void*                           pGpuDynDesc,
               cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,
               cudaStream_t                    strm);
    
    // 取得描述符信息
    static void getDescrInfo(size_t& dynDescrSizeBytes, 
                            size_t& dynDescrAlignBytes);

private:
    uint8_t CW_PER_BLOCK_;  // 每塊 4 個碼字
    simplexDecoderKernelArgs_t m_kernelArgs;
    void kernelSelect(uint16_t nCws, 
                     cuphySimplexDecoderLaunchCfg_t* pLaunchCfg);
};
```

#### 2. **C API 函數**

```cpp
// 創建解碼器
cuphyStatus_t cuphyCreateSimplexDecoder(
    cuphySimplexDecoderHndl_t* pHndl);

// 設置解碼器
cuphyStatus_t cuphySetupSimplexDecoder(
    cuphySimplexDecoderHndl_t       simplexDecoderHndl,
    uint16_t                        nCws,
    cuphySimplexCwPrm_t*            pCwPrmsCpu,
    cuphySimplexCwPrm_t*            pCwPrmsGpu,
    uint8_t                         enableCpuToGpuDescrAsyncCpy,
    void*                           pCpuDynDesc,
    void*                           pGpuDynDesc,
    cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,
    cudaStream_t                    strm);

// 取得描述符信息
cuphyStatus_t cuphySimplexDecoderGetDescrInfo(
    size_t* pDynDescrSizeBytes,
    size_t* pDynDescrAlignBytes);

// 摧毀解碼器
cuphyStatus_t cuphyDestroySimplexDecoder(
    cuphySimplexDecoderHndl_t simplexDecoderHndl);
```

---

## 資料結構

### 碼字參數結構

```cpp
struct cuphySimplexCwPrm_t
{
    __half*     d_LLRs;           // 對數似然比輸入（GPU）
    uint32_t*   d_cbEst;          // 估計碼字（GPU）
    float*      d_noiseVar;       // 噪聲方差（GPU）
    
    uint8_t*    d_DTXStatus;      // DTX 狀態（GPU）
    
    uint32_t    E;                // 速率匹配後的比特數
    uint8_t     K;                // Simplex 參數（K=1 或 K=2）
    uint8_t     nBitsPerQam;      // QAM 調制的比特數（Qm）
    
    uint8_t     en_DTXest;        // 啟用 DTX 估計
    float       DTXthreshold;     // DTX 閾值
    uint8_t     exitFlag;         // 退出標誌
};
```

### 動態描述符

```cpp
struct simplexDecoderDynDescr
{
    cuphySimplexCwPrm_t* pCwPrmsGpu;   // 指向 GPU 碼字參數
    uint16_t             nCws;          // 碼字數
};
typedef struct simplexDecoderDynDescr simplexDecoderDynDescr_t;
```

### 核心參數結構

```cpp
typedef struct
{
    simplexDecoderDynDescr_t* pDynDescr;  // 動態描述符
} simplexDecoderKernelArgs_t;
```

---

## CUDA 核心函數

### 主解碼核心

```cpp
// Simplex 解碼主核心
template <typename T>
__global__ void simplex_decoder_kernel(simplexDecoderDynDescr_t* pDynDescr)
{
    // 1. 提取參數
    uint16_t   numCW                 = pDynDescr->nCws;
    cuphySimplexCwPrm_t* pCwPrmsGpu  = pDynDescr->pCwPrmsGpu;
    
    // 2. 建立協作組
    cg::thread_block cta = cg::this_thread_block();
    
    // 3. 共享內存配置
#ifdef USE_ASYNC
    __shared__ T shmem[SIMPLEX_DECODER_MAX_E*2];  // 雙緩衝
    T *s_in[2] = {&shmem[0], &shmem[SIMPLEX_DECODER_MAX_E]};
#else
    __shared__ T shmem[SIMPLEX_DECODER_MAX_E];
    T *s_in = &shmem[0];
#endif
    
    // 4. 使用 cuda::memcpy_async 異步傳輸
    // 5. 網格步幅遍歷碼字
    // 6. DTX 狀態更新
}
```

### Simplex 解碼核心函數

```cpp
template <typename T>
__device__ void simplex_decode_core(
    int cw,                              // 碼字索引
    cg::thread_block &cta,               // 協作組
    const T* __restrict__ s_in,          // LLR 輸入
    uint32_t * __restrict__ x_out,       // 估計碼字輸出
    uint8_t K,                           // Simplex K 參數
    uint32_t E,                          // 速率匹配比特數
    uint8_t Qm,                          // QAM 比特
    float* __restrict__ confLevelFloat   // 置信度輸出
)
{
    // 1. 初始化累加變量
    const T llrscale = static_cast<T>(1.0) / static_cast<T>(SIMPLEX_DECODER_MAX_E);
    T c0Sum = 0.;
    T c1Sum = 0.;
    T c2Sum = 0.;
    uint16_t counter = 0;
    float totalPwr = 0.0;
    
    // 2. 創建 warp 級別的瓦片
    cg::thread_block_tile<SIMPLEX_DECODER_TPB> tile = 
        cg::tiled_partition<SIMPLEX_DECODER_TPB>(cta);
    
    // 3. 根據 K 值分支解碼
    if (K == 1)
    {
        // 單碼字解碼：根據符號判決
        // 累加所有 LLR
        // 計算置信度 = |sum| / sqrt(power)
    }
    else // K == 2
    {
        // 雙碼字解碼：根據 Qm 比較最大值
        // 3 路表決（Qm=1）或 2 路表決（Qm=2）
    }
}
```

---

## 算法流程

### Simplex 解碼流程

#### 步驟 1: 初始化

```
創建解碼器 → 分配 GPU 內存 → 設置描述符 → 選擇核心配置
```

#### 步驟 2: 數據準備

```
CPU 碼字參數 → 複製到 GPU → 更新動態描述符 → 異步複製到 GPU
```

#### 步驟 3: 核心執行

```
網格配置: gridDim = (nCws + 4 - 1) / 4 塊，每塊 32 線程
塊配置: blockDim = 32 線程

每個塊處理 CW_PER_BLOCK = 4 個碼字
```

#### 步驟 4: 解碼算法（K=1 情況）

```
1. 協作累加所有 LLR 值
   
   sum = Σ(LLR[i]) for i = 0 to E-1
   
2. 符號判決
   
   estimated_bit = (sum < 0) ? 1 : 0
   
3. 置信度計算
   
   confLevel = |sum| * SIMPLEX_DECODER_MAX_E / counter / sqrt(totalPwr/counter)
```

#### 步驟 5: 解碼算法（K=2 情況）

```
1. 根據 QAM 比特分組（Qm）
   
   對 Qm=1: 3 路表決（對應 3 個 QAM 象限）
   - c0Sum: 累加 LLR[0, 3, 6, ...]
   - c1Sum: 累加 LLR[1, 4, 7, ...]
   - c2Sum: 累加 LLR[2, 5, 8, ...]
   
   對 Qm=2: 2 路表決（對應 2 個比特平面）
   
2. 最大值選擇
   
   estimated_bits = argmax(|c0Sum|, |c1Sum|, |c2Sum|)
   
3. 置信度計算
   
   confLevel = (cost_max - cost_min) * SIMPLEX_DECODER_MAX_E / counter / sqrt(totalPwr)
              * (3.0 / 2.0) for K=2
```

### DTX 檢測

```cpp
// DTX 估計邏輯
if (en_DTXest & CUPHY_DTX_EN)
{
    float confLevelThr = CUPHY_DTX_THRESHOLD_ADJ_SIMPLEX_DECODER * DTXthreshold;
    
    if (noiseVar <= CUPHY_NOISE_REGULARIZER_THRESH)
    {
        // 噪聲過低 → DTX 
        d_DTXEst = 1;
    }
    else if (confLevelFloat < confLevelThr)
    {
        // 置信度低 → DTX
        d_DTXEst = 1;
    }
    else
    {
        // 置信度高 → 有效傳輸
        d_DTXEst = 0;
    }
}
```

---

## 性能特性

### GPU 優化

| 項目 | 數值 | 說明 |
|------|------|------|
| 線程配置 | 32 線程/塊 | `SIMPLEX_DECODER_TPB` |
| 每塊碼字 | 4 | `CW_PER_BLOCK_` |
| 最大 E | 1536 位 | `SIMPLEX_DECODER_MAX_E` |
| 共享內存 | 雙緩衝 | 無阻塞預取（可選） |

### 計算複雜度

- **每碼字操作**：O(E) 其中 E 是速率匹配比特數
- **協作累加**：O(log warp_size) = O(5) 用於 warp 歸約
- **吞吐量**：每時鐘多個碼字（網格步幅）

### 內存帶寬

- **輸入**：E × sizeof(__half) 字節
- **參數**：每碼字 32 字節
- **輸出**：每碼字 4 字節（1 個 uint32_t）

---

## 使用示例

### C++ 示例：基本解碼

```cpp
#include "cuphy.h"
#include "cuphy.hpp"

// 初始化
cuphy::context ctx;
cuphySimplexDecoderHndl_t decoder;
cuphyCreateSimplexDecoder(&decoder);

// 分配內存
size_t dynDescrSizeBytes, dynDescrAlignBytes;
cuphySimplexDecoderGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);

cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrGpu(dynDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCpu(dynDescrSizeBytes);

// 準備碼字參數
uint16_t nCws = 10;
std::vector<cuphySimplexCwPrm_t> cwPrmsCpu(nCws);
cuphy::buffer<cuphySimplexCwPrm_t, cuphy::device_alloc> cwPrmsGpu(nCws);

// 填充參數
for(int i = 0; i < nCws; ++i)
{
    cwPrmsCpu[i].E = 1024;           // 速率匹配比特
    cwPrmsCpu[i].K = 1;              // Simplex K=1
    cwPrmsCpu[i].nBitsPerQam = 2;    // QPSK
    cwPrmsCpu[i].en_DTXest = 1;      // 啟用 DTX 檢測
    cwPrmsCpu[i].DTXthreshold = 0.5;
    
    // 分配 LLR 和輸出內存
    cudaMalloc(&cwPrmsCpu[i].d_LLRs, cwPrmsCpu[i].E * sizeof(__half));
    cudaMalloc(&cwPrmsCpu[i].d_cbEst, sizeof(uint32_t));
    cudaMalloc(&cwPrmsCpu[i].d_noiseVar, sizeof(float));
    cudaMalloc(&cwPrmsCpu[i].d_DTXStatus, sizeof(uint8_t));
}

// 複製到 GPU
cudaMemcpy(cwPrmsGpu.addr(), cwPrmsCpu.data(), nCws * sizeof(cuphySimplexCwPrm_t),
           cudaMemcpyHostToDevice);

// 設置啟動配置
cuphySimplexDecoderLaunchCfg_t launchCfg;
cuphySetupSimplexDecoder(decoder, nCws, cwPrmsCpu.data(), cwPrmsGpu.addr(),
                        false, dynDescrCpu.addr(), dynDescrGpu.addr(),
                        &launchCfg, 0);

// 複製描述符到 GPU
cudaMemcpy(dynDescrGpu.addr(), dynDescrCpu.addr(), dynDescrSizeBytes,
          cudaMemcpyHostToDevice);

// 啟動核心
const CUDA_KERNEL_NODE_PARAMS& kernelParams = launchCfg.kernelNodeParamsDriver;
cuLaunchKernel(kernelParams.func,
              kernelParams.gridDimX, kernelParams.gridDimY, kernelParams.gridDimZ,
              kernelParams.blockDimX, kernelParams.blockDimY, kernelParams.blockDimZ,
              kernelParams.sharedMemBytes, 0,
              kernelParams.kernelParams, nullptr);

cudaDeviceSynchronize();

// 檢索結果
std::vector<uint32_t> decodedBits(nCws);
for(int i = 0; i < nCws; ++i)
{
    cudaMemcpy(&decodedBits[i], cwPrmsCpu[i].d_cbEst, sizeof(uint32_t),
              cudaMemcpyDeviceToHost);
}

// 清理
cuphyDestroySimplexDecoder(decoder);
```

### Python 示例（通過 PyAerial）

```python
from cuphy import SimplexDecoder
import numpy as np

# 建立解碼器
decoder = SimplexDecoder()

# 準備 LLR 數據
E = 1024  # 速率匹配比特
nCws = 10 # 碼字數
llrs = np.random.randn(nCws, E).astype(np.float16)

# 設置參數
params = {
    'K': 1,           # Simplex K=1
    'Qm': 2,          # QPSK
    'E': E,
    'en_DTXest': True,
    'DTXthreshold': 0.5
}

# 執行解碼
decoded = decoder.decode(llrs, **params)
print(f"Decoded bits: {decoded}")
```

---

## 完整工作流程

### 初始化階段

```
1. cuphyCreateSimplexDecoder()
   └─ 建立 SimplexDecoder 實例
   
2. cuphySimplexDecoderGetDescrInfo()
   └─ 查詢動態描述符大小
   
3. 分配 CPU/GPU 描述符緩衝區
```

### 配置階段

```
1. 填充 cuphySimplexCwPrm_t 結構
   ├─ 分配 GPU 內存
   ├─ 複製 LLR 數據
   └─ 設置參數（K, E, Qm）
   
2. 複製碼字參數到 GPU
   
3. cuphySetupSimplexDecoder()
   ├─ 填充動態描述符
   ├─ 選擇核心（kernelSelect）
   ├─ 計算網格/塊維度
   └─ 填充啟動配置
```

### 執行階段

```
1. 選擇性複製描述符到 GPU（非同步）
   
2. 通過 CUDA 驅動程序 API 啟動核心
   ├─ cuLaunchKernel() 或
   └─ CUDA 圖形節點
   
3. 同步流完成
```

### 輸出檢索階段

```
1. 從 GPU 複製解碼比特
   
2. 從 GPU 複製 DTX 狀態
   
3. 從 GPU 複製置信度指標
```

---

## 配置參數

### DTX 檢測參數

```cpp
// 常量
#define CUPHY_DTX_EN = 0x01              // DTX 檢測啟用標誌
#define CUPHY_DTX_THRESHOLD_ADJ_SIMPLEX_DECODER = 1.1
#define CUPHY_NOISE_REGULARIZER = 1.0e-3

// 結構欄位
cuphySimplexCwPrm_t.en_DTXest        // 啟用 DTX 估計
cuphySimplexCwPrm_t.DTXthreshold     // DTX 決策閾值
```

### 核心常量

```cpp
const int SIMPLEX_DECODER_MAX_E = 1536;    // 最大 E 值
const int SIMPLEX_DECODER_TPB = 32;        // 線程每塊
```

---

## 應用場景

### 1. HARQ 解碼

```cpp
// HARQ-ACK/NACK 解碼
cuphySimplexCwPrm_t cwPrm;
cwPrm.K = 1;              // 單比特決策
cwPrm.E = harqRmBits;     // 根據 HARQ 配置
decoder.setup(...);
// 執行解碼以得到 ACK/NACK 結果
```

### 2. CSI 反饋解碼

```cpp
// CSI 第 2 部分（相位反饋）
cuphySimplexCwPrm_t cwPrm;
cwPrm.K = 2;              // 多比特決策
cwPrm.Qm = 1 or 2;        // 根據反饋配置
cwPrm.E = csiRmBits;
decoder.setup(...);
// 執行解碼以得到相位調制碼字
```

### 3. DTX 偵測示例

```cpp
// 啟用 DTX 檢測用於節能
cwPrm.en_DTXest = CUPHY_DTX_EN;
cwPrm.DTXthreshold = adaptiveThreshold;  // 自適應閾值

// 解碼後檢查 DTX 狀態
uint8_t dtxStatus;
cudaMemcpy(&dtxStatus, cwPrm.d_DTXStatus, sizeof(uint8_t), 
          cudaMemcpyDeviceToHost);

if (dtxStatus == CUPHY_FAPI_DTX)
{
    // 偵測到 DTX - 未傳送
    processNoTransmission();
}
else
{
    // 有效傳輸 - 使用解碼結果
    processDecoded();
}
```

---

## 故障排除

### 常見問題

1. **解碼結果不正確**
   - 檢查 LLR 數據精度（應為 FP16）
   - 驗證速率匹配比特數 E 正確性
   - 確認 K 和 Qm 參數符合實際配置

2. **性能不理想**
   - 確保多個碼字批次（批量大小 > 4）
   - 驗證核心配置選擇（線程和塊大小）
   - 檢查內存帶寬利用率

3. **DTX 檢測問題**
   - 調整 DTXthreshold 參數
   - 驗證噪聲方差估計準確性
   - 檢查 CUPHY_NOISE_REGULARIZER 設置

### 調試技巧

```cpp
// 驗證核心參數
printf("E=%d, K=%d, Qm=%d, nCws=%d\n",
       cwPrm.E, cwPrm.K, cwPrm.nBitsPerQam, nCws);

// 檢查 DTX 狀態
uint8_t dtxStatus;
cudaMemcpy(&dtxStatus, cwPrm.d_DTXStatus, sizeof(uint8_t),
          cudaMemcpyDeviceToHost);
printf("DTX Status: %d\n", dtxStatus);

// 驗證置信度值
// （通過額外的核心修改或單獨的診斷核心）

// 使用 cuda-memcheck 檢測內存錯誤
// $ cuda-memcheck ./simplex_decoder_app
```

---

## 參考文獻

- NVIDIA Aerial cuPHY 5G PHY 層加速文檔
- 3GPP TS 38.212 - NR; Multiplexing and channel coding
- 3GPP TS 38.213 - NR; Physical layer procedures for control
- CUDA C++ Programming Guide - Cooperative Groups
- Information Theory and Coding Theory 參考文獻
