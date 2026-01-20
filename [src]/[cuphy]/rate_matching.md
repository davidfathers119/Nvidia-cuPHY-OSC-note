# NVIDIA cuPHY 速率匹配 (Rate Matching) 模組

## 概述

速率匹配 (Rate Matching) 是 5G NR 物理層處理中的關鍵組件，負責將LDPC編碼後的比特重新組織成適配實際傳輸資源的位數。該模組支持 PUSCH (Physical Uplink Shared Channel) 接收端和 DLSCH (Downlink Shared Channel) 傳輸端的速率匹配和反速率匹配操作。

**主要功能：**
- PUSCH 反速率匹配 (De-Rate Matching)：將接收到的LLR恢復到碼塊尺寸
- DLSCH 速率匹配 (Rate Matching)：調整編碼比特以適配傳輸資源
- 支持LDPC編碼後的位排列和選擇
- 支持冗餘版本 (RV) 為 0, 1, 2, 3

---

## 架構概述

### 核心組件

#### 1. **PUSCH 反速率匹配類** - `puschRxRateMatch`

```cpp
class puschRxRateMatch : public cuphyPuschRxRateMatch {
public:
    // 初始化和配置
    void init(int rmFPconfig, int descramblingOn);
    void setup(uint16_t nSchUes, uint16_t* pSchUserIdxsCpu, 
               const PerTbParams* pTbPrmsCpu, ...);
    
    // 描述符管理
    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);
    
private:
    CUfunction m_kernelFunc;                    // 主CUDA核心函數
    CUfunction m_resetBufferKernelFunc;         // 緩衝區重置核心
    int m_descramblingOn;                       // 是否啟用解擾
    int m_rmFPconfig;                           // FP精度配置
};
```

#### 2. **DLSCH 速率匹配類** - `dlRateMatching`

```cpp
class dlRateMatching {
    // DLSCH 速率匹配主核心
    // 處理PDSCH數據的速率匹配和層映射
    // 支持多個傳輸塊的批處理
};
```

---

## 資料結構

### PUSCH 反速率匹配描述符

```cpp
struct puschRxRateMatchDescr {
    const void* llr_vec_in[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];    // 輸入LLR向量
    uint16_t schUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];      // 用戶索引
    void** out;                         // 反速率匹配輸出LLR
    const PerTbParams* tbPrmsArray;     // 傳輸塊參數數組
    int descramblingOn;                 // 解擾標誌
};

struct puschRxRateMatchLaunchGeo {
    dim3 gridDim;                       // 網格尺寸
    dim3 blockDim;                      // 線程塊尺寸
    uint32_t shMemBytes;                // 共享內存大小
};
```

### DLSCH 速率匹配描述符

```cpp
struct dlRateMatchingDescr {
    const uint32_t* d_rate_matching_input;      // LDPC編碼器輸出
    uint32_t* d_rate_matching_output;           // 速率匹配輸出
    uint32_t* d_restructure_rate_matching_output; // 重組輸出
    
    const uint32_t* d_Er_array;                 // 每碼塊匹配長度E_r
    const uint32_t* d_k0_array;                 // 起始位置k0
    const uint32_t* d_TB_start_offset_array;    // TB起始偏移
    
    bool enable_scrambling;             // 啟用擾碼
    bool enable_layer_mapping;          // 啟用層映射
    
    uint32_t num_TBs;                   // 傳輸塊數
    uint32_t cmax;                      // 碼塊數
    uint32_t emax;                      // 最大匹配比特數
    const PdschPerTbParams* cfg_workspace;  // TB配置
};
```

### 傳輸塊參數

```cpp
struct PerTbParams {
    uint32_t K;                         // 碼塊大小 (系統比特數)
    uint32_t Kd;                        // Kd = K - 2*Zc (奇偶校驗比特數)
    uint32_t N;                         // LDPC碼長
    uint32_t F;                         // 填充比特數
    uint32_t Ncb;                       // 循環緩衝大小 = min(N, Nref)
    uint32_t Qm;                        // 調變階數 (1, 2, 4, 6, 8)
    uint32_t rv;                        // 冗餘版本 (0-3)
    uint32_t num_CBs;                   // 碼塊數量
    uint32_t isDataPresent;             // 數據是否存在
    uint32_t bg;                        // 基圖 (1 或 2)
    uint32_t Zc;                        // 基礎矩陣提升因子
};
```

---

## CUDA 核心函數

### PUSCH 反速率匹配核心

```cpp
template <typename T_IN, typename T_OUT>
__global__ void de_rate_matching_global2(puschRxRateMatchDescr_t* pRmDesc) {
    // 參數化大小：96 線程/塊, 佔有 10 個塊/多處理器
    
    // 1. 檢索傳輸塊和用戶索引
    uint32_t tbIdx = blockIdx.z;
    uint16_t ueIdx = rmDesc.schUserIdxs[tbIdx];
    const PerTbParams& tbPrms = rmDesc.tbPrmsArray[ueIdx];
    
    // 2. 對每個QAM階數進行反速率匹配
    // 支持的Qm值: 1 (BPSK), 2 (QPSK), 4 (16QAM), 6 (64QAM), 8 (256QAM)
    
    // 3. 計算反速率匹配索引
    // 根據 3GPP TS 38.212 Sec. 5.4.2.1
    uint32_t inIdx = kIdx * Qm + jIdx;
    uint32_t outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb);
}

template <typename T_OUT>
__global__ void de_rate_matching_reset_buffer(puschRxRateMatchDescr_t* pRmDesc) {
    // 重置輸出緩衝區為零
    // 確保未使用的位置被初始化
}
```

### DLSCH 速率匹配核心

```cpp
__global__ void dl_rate_matching(dlRateMatchingDescr_t* p_desc) {
    // 處理多個傳輸塊：速率匹配、擾碼和層映射
    
    dlRateMatchingDescr_t& desc = *p_desc;
    uint32_t TB_id = blockIdx.y;
    uint32_t CB_id = blockIdx.x;
    
    const PdschPerTbParams* TB_params = &desc.cfg_workspace[TB_id];
    
    // 1. 從LDPC編碼器輸出中讀取碼塊
    const uint32_t* ldpc_encoder_output = desc.d_rate_matching_input;
    
    // 2. 執行速率匹配：
    //    - 位選擇 (子塊交錯後)
    //    - 根據E_r截斷或重複
    //    - 擾碼 (如果啟用)
    
    // 3. 執行層映射 (如果啟用)
    //    - 將調製符號映射到多個層
}

__global__ void restructure_rate_matching_output(dlRateMatchingDescr_t* p_desc) {
    // 重組速率匹配輸出以滿足內存佈局要求
}
```

---

## 算法流程

### PUSCH 反速率匹配算法

反速率匹配遵循 3GPP TS 38.212 第 5.4.2 節規範：

#### 步驟 1: 計算起始位置 k0

```
For redundancy version rv:
    Zc = lifting factor
    Ncb = min(N, Nref)
    
    if rv == 0: k0 = (0)
    if rv == 1: k0 = (17 * Ncb / (50 * Zc)) * Zc
    if rv == 2: k0 = (25 * Ncb / (50 * Zc)) * Zc
    if rv == 3: k0 = (43 * Ncb / (50 * Zc)) * Zc
```

#### 步驟 2: 計算反速率匹配索引

```
For each output index k:
    // 循環選擇方法
    outIdx = (k0 + k) mod (Ncb - F)
    
    if outIdx >= Kd:
        input is from parity bits at outIdx - Kd
    else:
        input is from systematic bits at outIdx
```

#### 步驟 3: 提取和LLR映射

```
for j in 0 to Qm-1:           // 調變階數
    for k in 0 to E/Qm-1:     // 匹配長度
        inIdx = k * Qm + j
        llrOut[inIdx] = llrIn[outIdx]  // 從計算的outIdx讀取
        
        // 應用LLR鉗制
        llrOut[inIdx] = clamp(llrOut[inIdx], LLR_MIN, LLR_MAX)
```

### DLSCH 速率匹配算法

DLSCH 速率匹配分為三個主要階段：

#### 階段 1: 子塊交錯

根據 3GPP TS 38.212 第 5.4.1.1 節進行P矩陣交錯：

```
// 構建交錯索引
for n in 0 to (N-1):
    i = floor(32 * n / N)
    J[n] = P[i] * (N / 32) + mod(n, N/32)

// 應用交錯
y = d[J]
```

#### 階段 2: 位選擇

```
if E >= N:              // 重複模式
    for k in 0 to (E-1):
        e[k] = y[mod(k, N)]
        
else:                   // 截斷/穿孔模式
    for k in 0 to (E-1):
        outIdx = derate_match_fast_calc_modulo(k, Kd, F, k0, Ncb)
        e[k] = y[outIdx]
```

#### 階段 3: 擾碼和層映射

```
if enable_scrambling:
    // 使用初始值c_init進行擾碼序列生成
    scrambling_sequence = generate_gold_sequence(c_init, E)
    e_scrambled = e XOR scrambling_sequence

if enable_layer_mapping:
    // 將調製符號映射到nl層
    for layer in 0 to nl-1:
        for symbol in 0 to symbols_per_layer-1:
            tx_buffer[layer, symbol] = e_scrambled[...]
```

---

## 浮點精度配置 (FP Configuration)

PUSCH 反速率匹配支持4種浮點配置：

| FPconfig | 輸入類型 | 輸出類型 | 使用場景 |
|----------|---------|---------|--------|
| 0 | FP32 | FP32 | 參考實現，最高精度 |
| 1 | FP16 | FP32 | 輸入節省帶寬 |
| 2 | FP32 | FP16 | 輸出節省內存 |
| 3 | FP16 | FP16 | 完全低精度，最快速度 |

核心函數根據配置動態選擇：

```cpp
case 0: kernel = de_rate_matching_global2<float, float>;
case 1: kernel = de_rate_matching_global2<__half, float>;
case 2: kernel = de_rate_matching_global2<float, __half>;
case 3: kernel = de_rate_matching_global2<__half, __half>;
```

---

## Python 綁定接口

### LDPC 速率匹配 Python 類

```python
class LdpcRateMatch:
    def __init__(self, 
                 enable_scrambling=True,
                 num_dl_bwp_prbs=273,
                 max_num_code_blocks=152,
                 max_num_tbs=128,
                 cuda_stream=None):
        """初始化LDPC速率匹配"""
    
    def rate_match(self,
                   *,
                   coded_blocks: List[Array],      # LDPC編碼後的碼塊
                   tb_sizes: List[int],            # TB大小
                   code_rates: List[float],        # 編碼率
                   rate_match_lens: List[int],     # 匹配長度E_r
                   mod_orders: List[int],          # QAM階數
                   num_layers: List[int],          # 層數
                   redundancy_versions: List[int], # 冗餘版本
                   cinits: List[int]) -> List[Array]:  # 擾碼初值
        """
        執行LDPC速率匹配
        遵循 TS 38.212 Sec. 5.4.2.1
        """
    
    def rm_mod_layer_map(self,
                        *,
                        coded_blocks: List[Array],
                        tx_buffer: Array,
                        pdsch_configs: List[PdschConfig],
                        csi_rs_configs: Optional[List] = None):
        """速率匹配、調變和層映射"""
```

### LDPC 反速率匹配 Python 類

```python
class LdpcDeRateMatch:
    def __init__(self,
                 enable_scrambling=True,
                 cuda_stream=None,
                 fp_config=3):
        """初始化LDPC反速率匹配"""
    
    def derate_match(self,
                     *,
                     input_llrs: List[Array],              # 接收器LLR
                     pusch_configs: Optional[List] = None,
                     tb_sizes: Optional[List[int]] = None,
                     code_rates: Optional[List[float]] = None,
                     rate_match_lengths: Optional[List[int]] = None,
                     mod_orders: Optional[List[int]] = None,
                     num_layers: Optional[List[int]] = None,
                     redundancy_versions: Optional[List[int]] = None,
                     cinits: Optional[List[int]] = None,
                     ue_grp_idx: Optional[List[int]] = None) -> List[Array]:
        """
        執行LDPC反速率匹配
        - 恢復碼塊LLR
        - 支持多個UE和UE組
        - 支持HDArQ冗餘版本
        """
```

---

## 使用示例

### C++ API 使用

```cpp
// 1. 創建反速率匹配對象
cuphyPuschRxRateMatchHndl_t puschRmHndl;
cuphyCreatePuschRxRateMatch(&puschRmHndl, 
                            FPconfig = 3,     // FP16 in/out
                            descramblingOn = 1);

// 2. 獲取描述符信息
size_t dynDescrSizeBytes, dynDescrAlignBytes;
cuphyPuschRxRateMatchGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes);

// 3. 分配描述符緩衝區
cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

// 4. 設置速率匹配
cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;
cuphySetupPuschRxRateMatch(puschRmHndl,
                           nSchUes,
                           pSchUserIdxsCpu,
                           pTbPrmsCpu,
                           pTbPrmsGpu,
                           pTPrmRmIn,
                           pTPrmCdm1RmIn,
                           ppRmOut,
                           dynDescrBufCpu.addr(),
                           dynDescrBufGpu.addr(),
                           enableCpuToGpuDescrAsyncCpy,
                           &puschRmLaunchCfg,
                           cuStrmMain.handle());

// 5. 啟動核心
const CUDA_KERNEL_NODE_PARAMS& kernelNodeParams = puschRmLaunchCfg.kernelNodeParamsDriver;
cuLaunchKernel(kernelNodeParams.func,
               kernelNodeParams.gridDimX, kernelNodeParams.gridDimY, kernelNodeParams.gridDimZ,
               kernelNodeParams.blockDimX, kernelNodeParams.blockDimY, kernelNodeParams.blockDimZ,
               kernelNodeParams.sharedMemBytes,
               static_cast<CUstream>(cuStrmMain.handle()),
               kernelNodeParams.kernelParams,
               kernelNodeParams.extra);
```

### Python API 使用

```python
import numpy as np
from aerial.phy5g.ldpc import LdpcRateMatch

# 初始化速率匹配
rm = LdpcRateMatch(enable_scrambling=True, 
                   num_dl_bwp_prbs=273)

# 準備輸入數據
coded_blocks = [np.random.randint(0, 2, (N, C), dtype=np.uint8) for _ in range(num_tbs)]
tb_sizes = [7680, 7680]              # 傳輸塊大小
code_rates = [0.5, 0.5]              # 編碼率
rate_match_lens = [10920, 10920]     # 匹配長度
mod_orders = [4, 4]                  # 16-QAM
num_layers = [2, 2]                  # 2層
redundancy_versions = [0, 0]         # RV0
cinits = [0x1234, 0x5678]            # 擾碼初值

# 執行速率匹配
rm_output = rm.rate_match(
    coded_blocks=coded_blocks,
    tb_sizes=tb_sizes,
    code_rates=code_rates,
    rate_match_lens=rate_match_lens,
    mod_orders=mod_orders,
    num_layers=num_layers,
    redundancy_versions=redundancy_versions,
    cinits=cinits
)
```

---

## 性能特性

### GPU 優化

- **線程並行性**：96 線程/塊配置以最大化GPU利用率
- **佔有率**：10 個活動塊/多處理器，平衡資源利用
- **共享內存**：最小化共享內存以實現更高的佔有率
- **合作組**：支持高效的線程塊級約化

### 計算複雜度

- **反速率匹配**：O(E) 其中 E 是匹配比特數
- **循環模運算**：快速非迭代方法避免重複除法
  ```cpp
  outIdx = derate_match_fast_calc_modulo(inIdx, Kd, F, k0, Ncb)
  // 使用預計算查找表優化
  ```

### 內存訪問模式

- **連續讀取**：從LLR張量進行對齐的連續讀取
- **準隨機寫入**：根據反速率匹配索引進行準隨機寫入
- **緩存優化**：充分利用GPU L1/L2緩存

---

## 3GPP 標準合規性

該實現完全符合以下3GPP標準：

| 標準 | 章節 | 描述 |
|------|------|------|
| TS 38.212 | 5.4.1 | 下行速率匹配 (極軟碼、極碼、LDPC) |
| TS 38.212 | 5.4.2 | 上行反速率匹配 (LDPC) |
| TS 38.211 | 6.3.1 | PUSCH 物理通道定義 |
| TS 38.213 | 6.1 | PUSCH 傳輸配置 |

### 主要算法合規性

1. **子塊交錯 (Sub-block Interleaving)**
   - 使用32x32排列矩陣 P
   - 適用於所有LDPC基圖 (BG1, BG2)

2. **位選擇 (Bit Selection)**
   - 支持截斷 (Puncturing): E < N
   - 支持重複 (Repetition): E > N
   - 根據編碼率動態選擇

3. **循環移位 (Circular Shift)**
   - 基於冗餘版本 (RV0-3)
   - 計算起始位置 k0
   - 支持HARQ重傳

---

## 測試和驗證

### 單元測試框架

```cpp
class RateMatchingTest: public ::testing::TestWithParam<TestParams> {
    void basicTest() {
        // CPU和GPU實現比較
        // 驗證輸出匹配度
        // 性能基準測試
    }
};

TEST_P(RateMatchingTest, IDENTICAL_TB_CONFIGS) {
    ASSERT_TRUE(gpu_mismatch == 0);
}
```

### 測試向量支持

- HDF5 格式測試向量
- 多TB、多CB配置
- 不同QAM階數 (BPSK 到 256-QAM)
- 所有冗餘版本 (RV0-3)

---

## API 參考

### PUSCH 反速率匹配 API

```cpp
// 獲取描述符信息
cuphyStatus_t cuphyPuschRxRateMatchGetDescrInfo(
    size_t* pDescrSizeBytes, 
    size_t* pDescrAlignBytes);

// 創建反速率匹配對象
cuphyStatus_t cuphyCreatePuschRxRateMatch(
    cuphyPuschRxRateMatchHndl_t* pPuschRxRateMatchHndl,
    int FPconfig,
    int descramblingOn);

// 設置反速率匹配
cuphyStatus_t cuphySetupPuschRxRateMatch(
    cuphyPuschRxRateMatchHndl_t puschRxRateMatchHndl,
    uint16_t nSchUes,
    uint16_t* pSchUserIdxsCpu,
    const PerTbParams* pTbPrmsCpu,
    const PerTbParams* pTbPrmsGpu,
    cuphyTensorPrm_t* pTPrmRmIn,
    cuphyTensorPrm_t* pTPrmCdm1RmIn,
    void** ppRmOut,
    void* pCpuDesc,
    void* pGpuDesc,
    uint8_t enableCpuToGpuDescrAsyncCpy,
    cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,
    cudaStream_t strm);

// 摧毀反速率匹配對象
cuphyStatus_t cuphyDestroyPuschRxRateMatch(
    cuphyPuschRxRateMatchHndl_t* pPuschRxRateMatchHndl);
```

### DLSCH 速率匹配 API

```cpp
// 獲取描述符信息
cuphyStatus_t cuphyDlRateMatchingGetDescrInfo(
    size_t* pDescrSizeBytes, 
    size_t* pDescrAlignBytes);

// 設置速率匹配
cuphyStatus_t cuphySetupDlRateMatching(
    cuphyDlRateMatchingHndl_t rmHandle,
    cuphyPdschStatusOut_t* pStatusOut,
    const uint32_t* dInputBits,
    uint32_t* dRmOutput,
    uint32_t* dModOutput,
    const uint16_t* reMapping,
    uint16_t nPrbBwp,
    uint32_t numTbs,
    uint32_t numLayers,
    uint8_t enableScrambling);
```

---

## 診斷和故障排除

### 常見問題

1. **LLR 飽和或下溢**
   - 檢查 LLR_CLAMP_MIN/MAX 設置
   - 驗證輸入LLR範圍

2. **索引計算錯誤**
   - 確認 k0 計算與冗餘版本一致
   - 驗證 Ncb 和 Kd 參數

3. **性能瓶頸**
   - 分析內存帶寬使用率
   - 檢查線程塊占有率
   - 考慮使用FP16精度配置

### 調試工具

- NVIDIA Nsight Systems：分析內核執行
- NVIDIA Nsight Compute：詳細性能指標
- cuda-memcheck：檢測內存錯誤

---

## 相關模組集成

速率匹配在 cuPHY 接收器和傳輸器管道中的位置：

**上行通道 (PUSCH 接收)：**
```
接收數據 → 均衡化 → 反速率匹配 → LDPC 解碼 → CRC 檢查 → 數據輸出
           (此處)         ↓
                    從符號恢復LLR
                    從碼塊恢復尺寸
```

**下行通道 (PDSCH 傳輸)：**
```
CRC 編碼 → LDPC 編碼 → 速率匹配 → 調變 → 層映射 → 傳輸
                      (此處)
                     按資源大小調整
                     循環移位應用
```

---

## 參考文獻

- 3GPP TS 38.212 V18.x.x - NR; Multiplexing and channel coding
- NVIDIA cuPHY 文檔
- NVIDIA CUDA 編程指南
- Aerial cuPHY-SDAP 開源實現
