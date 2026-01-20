# 通道狀態信息參考信號 (CSIRS)

## 概述

CSIRS (Channel State Information Reference Signal) 是NVIDIA cuPHY庫中用於**5G下行鏈路通道狀態信息參考信號生成**的組件。它根據3GPP標準生成時頻域CSI-RS信號，支持多種碼分複用(CDM)模式和預編碼波束賦形。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - CSIRS](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/csirs)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `csirs.hpp` | CSIRS參數結構和API定義 |
| `csirs.cuh` | CUDA設備代碼 - 常數表和CDM序列 |
| `csirs_tf_signal.cu` | CUDA內核實現 - 加擾和信號生成 |

---

## CSIRS參數結構

### CsirsParams

```cpp
typedef struct _CsirsParams
{
    uint16_t startRb;                           // 起始RB (< 273)
    uint16_t nRb;                               // RB數量
    uint8_t  row;                               // 符號位置表行號 (1-18)
    uint8_t  li[2];                             // 時域位置 L0和L1
    uint8_t  seqIndexCount;                     // CDM序列索引計數 (1,2,4,8)
    uint8_t  genEvenRB;                         // rho=0.5時的生成開關
    uint8_t  idxSlotInFrame;                    // 幀內時隙索引
    uint16_t scrambId;                          // 加擾ID
    uint8_t  ki[CUPHY_CSIRS_MAX_KI_INDEX_LENGTH];  // k0,k1,k2,k3,k4,k5
    cuphyCsiType_t csiType;                     // CSI類型 (0:TRS, 1:NZP, 2:ZP)
    cuphyCdmType_t cdmType;                     // CDM類型
    float    alpha;                             // X=1時為rho，否則為2*rho
    float    rho;                               // 密度 (0.5, 1, 3)
    float    beta;                              // 功率縮放因子
    uint16_t cell_index;                        // 小區索引
    uint8_t  enablePrcdBf;                      // 預編碼開關
    uint16_t pmwPrmIdx;                         // 預編碼矩陣索引
    uint8_t  nTxAnt;                            // 發射天線數
} CsirsParams;
```

**參數說明**:

| 參數 | 說明 | 範圍 |
|------|------|------|
| `startRb` | 資源起始RB位置 | 0-272 |
| `nRb` | 資源佔用RB數 | 1-273 |
| `row` | 符號位置表行 | 1-18 |
| `li[2]` | OFDM符號時域位置 | 0-13 |
| `seqIndexCount` | CDM維度因子 | 1,2,4,8 |
| `rho` | 資源密度 | 0.5, 1.0, 3.0 |
| `alpha` | CDM分離因子 | rho 或 2*rho |
| `beta` | 功率歸一化 | 取決於天線數 |
| `cdmType` | CDM配置 | 0:noCDM, 1:FD-CDM2, 2:FD2-TD2, 3:FD2-TD4 |

---

## 符號位置表

### CsirsSymbLocRow 結構

```cpp
struct CsirsSymbLocRow
{
    uint8_t  lenKBarLBar;           // K-Bar/L-Bar對數量
    uint8_t  lenKPrime;             // K' 維度 (1 or 2)
    uint8_t  lenLPrime;             // L' 維度
    uint8_t  nCdmGroup;             // CDM組數
    uint8_t  kIndices[];            // K索引表
    uint8_t  kOffsets[];            // K偏移
    uint8_t  lIndices[];            // L索引表
    uint8_t  lOffsets[];            // L偏移
    uint8_t  cdmGroupIndex[];       // CDM組索引
};
```

**表格配置** (18行):

| 行 | 密度 | K-Bar數 | L-Bar數 | K' | L' | CDM |
|----|------|---------|---------|----|----|-----|
| 1  | 1/2  | 3       | 1       | 2  | 1  | -   |
| 2  | 1    | 1       | 1       | 1  | 1  | -   |
| 3  | 1    | 1       | 2       | 2  | 1  | -   |
| 4  | 1    | 2       | 2       | 2  | 1  | -   |
| 5  | 3/2  | 2       | 2       | 2  | 1  | -   |
| 6  | 1    | 4       | 2       | 2  | 1  | FD2 |
| 7  | 1    | 4       | 2       | 2  | 1  | FD2 |
| 8  | 1    | 2       | 2       | 2  | 2  | TD2 |
| ... | ... | ... | ... | ... | ... | ... |

---

## CDM配置和序列表

### CDM類型定義

```cpp
typedef enum {
    CUPHY_CDM_TYPE_NOCDM      = 0,  // 無CDM
    CUPHY_CDM_TYPE_FD_CDM2    = 1,  // 頻域 2-port CDM
    CUPHY_CDM_TYPE_FD2_TD2    = 2,  // 頻域2×時域2 4-port CDM
    CUPHY_CDM_TYPE_FD2_TD4    = 3,  // 頻域2×時域4 8-port CDM
} cuphyCdmType_t;
```

### Wf/Wt 序列表

```cpp
// 常數內存序列表: constSeqTableCsirs[CDM類型][序列索引][Wf/Wt][值]
// 每個CDM類型的W序列
__constant__ int8_t constSeqTableCsirs[MAX_CDM_TYPE][CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT][2][4];

// 示例配置
NO_CDM:         // seqIndexCount=1
    {{{1}, {1}}}

CDM2_FD:        // seqIndexCount=2
    {{{1, 1}, {1}},
     {{1, -1}, {1}}}

CDM4_FD2_TD2:   // seqIndexCount=4
    {{{1, 1}, {1, 1}},
     {{1, -1}, {1, 1}},
     {{1, 1}, {1, -1}},
     {{1, -1}, {1, -1}}}

CDM8_FD2_TD4:   // seqIndexCount=8
    {{{1, 1}, {1, 1, 1, 1}},
     {{1, -1}, {1, 1, 1, 1}},
     {{1, 1}, {1, -1, 1, -1}},
     {{1, -1}, {1, -1, 1, -1}},
     {{1, 1}, {1, 1, -1, -1}},
     {{1, -1}, {1, 1, -1, -1}},
     {{1, 1}, {1, -1, -1, 1}},
     {{1, -1}, {1, -1, -1, 1}}}
```

---

## CUDA內核

### 1. 加擾序列生成

**內核**: `genScramblingKernel`

**功能**: 為每個符號生成Gold加擾序列

```cpp
__global__ void genScramblingKernel(CsirsParams* csirs_params,
                                    int num_params,
                                    uint8_t* d_scrambling_seq)
{
    CsirsParams& params = csirs_params[blockIdx.x];  // 參數索引
    const int symbol = blockIdx.y;                   // 符號索引
    
    // Gold加擾初值計算
    uint32_t c_init_scrambling = ((1 << 10) * 
                                  (OFDM_SYMBOLS_PER_SLOT * params.idxSlotInFrame + 
                                   symbol + 1) * 
                                  (2 * params.scrambId + 1) + 
                                  params.scrambId) & 0x7FFFFFFF;
    
    // 線程ID到位置映射
    int tid = threadIdx.x;
    if(tid < max_scrambling_elements)  // max=52 (1664/32)
    {
        // 並行生成32位的Gold序列
        uint32_t val = gold32(c_init_scrambling, tid * 32);
        scrambling_seq[tid] = val;
    }
}
```

**網格配置**:
- `gridDim`: (numParams, OFDM_SYMBOLS_PER_SLOT, 1)
- `blockDim`: (64, 1, 1)
- `sharedMem`: 0

**計算步驟**:

1. **加擾初值計算**
   ```
   c_init = (2^10 * (14 * slotIdx + symbolIdx + 1) * (2*id_csi_rs + 1) + id_csi_rs) mod 2^31
   ```

2. **Gold序列生成** (使用LFSR)
   ```
   x1(n) = x1(n-1) ^ x1(n-3) ^ x1(n-4) ^ x1(n-6) ^ x1(n-7) ^ x1(n-9) ^ ...
   x2(n) = x2(n-1) ^ x2(n-2) ^ x2(n-3) ^ x2(n-4) ^ x2(n-6) ^ x2(n-9) ^ ...
   c(n) = x1(n) ^ x2(n)
   ```

3. **並行化**
   - 每個線程生成32比特
   - 共生成 1664 比特 (273*2*3)

### 2. 時頻域信號生成

**內核**: `genCsirsTfSignalKernel<SignalType>`

**功能**: 為每個CSI-RS資源生成時頻域復信號

```cpp
template <typename SignalType>
__global__ void genCsirsTfSignalKernel(SignalType** tfSignalArray,
                                       CsirsParams* csirsParams,
                                       int numParams,
                                       uint32_t* offsets,
                                       uint32_t total_offsets,
                                       uint8_t* goldSeq,
                                       uint16_t* numReInFreqDimArray,
                                       cuphyCsirsPmWOneLayer_t* d_pmw_params)
{
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(globalIndex >= total_offsets)
        return;
    
    // 查找CSI-RS參數
    int paramNum = 0;
    while(globalIndex >= offsets[paramNum + 1])
        paramNum++;
    
    // 本地索引計算
    int localIndex = globalIndex - offsets[paramNum];
    
    // 獲取行數據
    CsirsParams& params = csirsParams[paramNum];
    int rowNum = params.row;  // 1-indexed
    CsirsSymbLocRow& rowData = constRowDataCsirs[rowNum - 1];
}
```

**網格配置**:
- `gridDim`: (ceil(total_offsets / 128), 1, 1)
- `blockDim`: (128, 1, 1)
- `sharedMem`: 0

### 3. 輔助函數

**函數**: `csirsTfSignalGenHelper`

**功能**: 根據CDM類型生成CSI-RS符號

```cpp
template <typename SignalType, int SEQ_INDEX, int SHIFT>
__device__ void csirsTfSignalGenHelper(uint lenLKBarPrime,
                                       uint lenKPrime,
                                       CsirsParams& params,
                                       uint idxRB,
                                       int8_t* seqTable,
                                       uint8_t* goldSeq,
                                       CsirsSymbLocRow& rowData,
                                       uint16_t numReInFreqDim,
                                       SignalType* tfSignal,
                                       cuphyCsirsPmWOneLayer_t* d_pmw_params)
{
    // K-Bar/L-Bar索引計算
    uint idxKBarLBar = lenLKBarPrime >> SHIFT;
    uint lenLKPrime = lenLKBarPrime & (SEQ_INDEX - 1);
    
    // K'和L'提取
    const uint lenKPrime_minus_one = ((lenKPrime - 1) & 0x1);
    const uint lPrime = lenLKPrime >> lenKPrime_minus_one;
    const uint kPrime = lenLKPrime & lenKPrime_minus_one;
}
```

---

## 信號生成算法

### 基本流程

```
CSI-RS生成流程
    ↓
[步驟1] 初始化參數
    ├─ 讀取行表數據
    ├─ 計算時頻位置
    └─ 檢查密度門限
    ↓
[步驟2] 計算頻率位置
    ├─ k_bar = ki[rowData.kIndices[idx]] + rowData.kOffsets[idx]
    ├─ k = k_bar + k_prime + RB_idx * 12
    └─ m_prime = floor(RB_idx * alpha) + k_prime + floor(k_bar * rho / 12)
    ↓
[步驟3] 計算時間位置
    ├─ l_bar = li[rowData.lIndices[idx]] + rowData.lOffsets[idx]
    └─ l = l_bar + l_prime
    ↓
[步驟4] 獲取加擾序列
    ├─ Gold序列: goldSeq[l * Ng_elements + (2*m_prime >> 3)]
    └─ 提取比特: (goldSeq[idx] >> (2*m_prime & 7)) & 0x3
    ↓
[步驟5] 生成信號
    ├─ 對每個序列索引 s (0 to seqIndexCount-1):
    │  ├─ w_f = seqTable[s*2*4 + k_prime]     (Wf值)
    │  ├─ w_t = seqTable[s*2*4 + 4 + l_prime] (Wt值)
    │  ├─ 實部: I = beta * w_f * w_t * sqrt(0.5) * (1 - 2*bit0)
    │  └─ 虛部: Q = beta * w_f * w_t * sqrt(0.5) * (1 - 2*bit1)
    ├─ 天線索引: p = (cdmGroupIndex * seqIndexCount + s) % nTxAnt
    └─ 寫入: tfSignal[k + l * numReInFreqDim + p * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim]
    ↓
[步驟6] 可選:預編碼波束賦形
    └─ 如果enablePrcdBf:
       ├─ pmw_matrix = d_pmw_params[pmwPrmIdx].matrix
       └─ tfSignal[...] = __hcmadd(signal, pmw_matrix, 0)
    ↓
輸出: 時頻域CSI-RS信號
```

---

## 密度處理

### Rho = 0.5 特殊情況

```cpp
// 當密度為0.5時，需要交替RB生成
if(params.rho == 0.5f)
{
    bool isEvenRB = (idxRB & 1) == 0;
    
    // 根據genEvenRB標誌選擇
    if((params.genEvenRB && !isEvenRB) || (!params.genEvenRB && isEvenRB))
        return;  // 跳過此RB
}

// rho=0.5: 配置1 - 偶數RB
// rho=0.5: 配置2 - 奇數RB
```

---

## 配置常數

```cpp
// 尺寸限制
const int Ng = 273 * 2 * 3;              // 1638個生成元素
const int Ng_elements = 1664 / 8;        // 208字節 (1664比特 / 8)
const int OFDM_SYMBOLS_PER_SLOT = 14;    // 每時隙符號數

// 表長度
const int CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH = 18;  // 行數
const int CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT = 8;            // 最大CDM維度
const int CUPHY_CSIRS_MAX_KI_INDEX_LENGTH = 6;            // ki陣列長度
const int CUPHY_N_TONES_PER_PRB = 12;                     // 每RB子載波數

// 資源計算
const uint32_t MAX_WORDS_PER_SYMBOL = ceil(Ng / 32);      // Gold序列字數
```

---

## 預編碼波束賦形

### 預編碼矩陣結構

```cpp
struct cuphyCsirsPmWOneLayer
{
    uint8_t nPorts;                  // 預編碼端口數
    __half2 matrix[nPorts][OFDM_SYMBOLS_PER_SLOT][numReInFreqDim];  // 預編碼矩陣
};
typedef struct cuphyCsirsPmWOneLayer cuphyCsirsPmWOneLayer_t;
```

### 預編碼應用

```cpp
if(enablePrcdBf)
{
    const cuphyCsirsPmWOneLayer_t& pmw = d_pmw_params[pmwPrmIdx];
    uint8_t nPorts = pmw.nPorts;
    
    SignalType zeroValue = make_complex<SignalType>::create(0, 0);
    
    // 對每個端口應用預編碼
    for(int idx = 0; idx < nPorts; idx++)
    {
        // 復數乘法: signal * matrix[idx]
        tfSignal[k + l * numReInFreqDim + idx * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim] =
            __hcmadd(signal, pmw.matrix[idx], zeroValue);
    }
}
else
{
    // 無預編碼: 直接寫入天線對應位置
    tfSignal[k + l * numReInFreqDim + p * OFDM_SYMBOLS_PER_SLOT * numReInFreqDim] = signal;
}
```

---

## API函數

### 內核選擇函數

```cpp
void kernelSelectGenScrambling(cuphyGenScramblingLaunchCfg_t* pLaunchCfg,
                               uint32_t numParams)
```

**功能**: 設置加擾序列生成內核

**參數**:
- `pLaunchCfg`: 啟動配置結構
- `numParams`: CSI-RS參數集數量

**內核配置**:
```
gridDim:  (numParams, OFDM_SYMBOLS_PER_SLOT)
blockDim: (64)
sharedMem: 0
```

### CSI-RS信號生成內核選擇

```cpp
void kernelSelectGenCsirsTfSignal(cuphyGenCsirsTfSignalLaunchCfg_t* pLaunchCfg)
```

**功能**: 設置時頻域信號生成內核

**內核配置**:
```
gridDim:  (ceil(totalNumThreadsLB / 128))
blockDim: (128)
sharedMem: 0
```

---

## 執行流程

### 完整處理流程

```
應用層 API 調用
    ↓
[調用1] cuphyGenCsirsScramblingSeq()
    ├─ kernelSelectGenScrambling()
    ├─ 啟動: genScramblingKernel<<<(numParams, 14), 64>>>
    └─ 輸出: Gold加擾序列
    ↓
[調用2] cuphyGenCsirsTfSignal()
    ├─ kernelSelectGenCsirsTfSignal()
    ├─ 啟動: genCsirsTfSignalKernel<<<gridDim, 128>>>
    └─ 輸出: 時頻域CSI-RS信號
    ↓
完成: 所有小區的CSI-RS信號準備完成
```

---

## 支持的場景

### CSI-RS類型

| 類型 | 值 | 說明 | 支持 |
|------|-----|------|------|
| TRS | 0 | 追蹤參考信號 | - |
| CSI-RS NZP | 1 | 非零功率CSI-RS | ✅ |
| CSI-RS ZP | 2 | 零功率CSI-RS | 開發中 |

### CDM配置

| CDM類型 | 端口數 | 頻域 | 時域 | 用途 |
|---------|--------|------|------|------|
| noCDM | 1-32 | - | - | 單端口配置 |
| FD-CDM2 | 2 | 2 | 1 | 2端口頻域CDM |
| FD2-TD2 | 4 | 2 | 2 | 4端口時頻CDM |
| FD2-TD4 | 8 | 2 | 4 | 8端口時頻CDM |

### 支持的密度

- **rho = 0.5**: 半密度 (配置1或2)
- **rho = 1.0**: 全密度
- **rho = 3.0**: 超密度

---

## 3GPP標準映射

實現遵循3GPP TS 38.211標準:

| 組件 | 標準 | 說明 |
|------|------|------|
| 加擾序列 | 38.211 Sec 5.2.1 | Gold序列生成 |
| 符號映射 | 38.211 Sec 7.4.1.5 | CSI-RS資源位置 |
| CDM應用 | 38.211 Sec 7.4.1.5.3 | 碼分複用配置 |
| 預編碼 | 38.211 Sec 7.3.1 | 波束賦形矩陣 |

---

## 性能優化

✅ **常數內存表** - 符號位置和序列表
✅ **共享表存取** - 所有線程快速訪問
✅ **並行化生成** - 多參數/符號同時處理
✅ **密集計算** - Warp級操作優化
✅ **半精度支持** - `__half2`複數運算

---

## 延伸閱讀

- 3GPP TS 38.211 - 物理通道和調製
- 3GPP TS 38.212 - 多路復用和編碼
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- CSI-RS設計和應用
- 波束管理和反饋機制
