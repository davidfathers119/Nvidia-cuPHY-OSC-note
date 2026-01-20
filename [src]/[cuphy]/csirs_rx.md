# CSI-RS接收提取 (CSIRS_RX)

## 概述

CSIRS_RX (Channel State Information Reference Signal Reception) 是NVIDIA cuPHY庫中用於**5G上行鏈路CSI-RS接收和通道估計**的組件。它從接收信號中提取CSI-RS參考信號，進行去加擾和相關性計算，用於推導通道狀態信息。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - CSIRS_RX](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/csirs_rx)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `csirs_rx.hpp` | CSI-RS接收參數結構和API定義 |
| `csirs_rx.cuh` | CUDA設備代碼 - 常數表和符號位置 |
| `csirs_rx.cu` | CUDA內核實現 - 通道估計 |

---

## CSI-RS接收參數結構

### CsirsRxParams

```cpp
typedef struct _CsirsRxParams
{
    uint16_t startRb;                           // 起始RB (< 273)
    uint16_t nRb;                               // RB數量
    uint8_t  row;                               // 符號位置表行號 (1-18)
    uint8_t  freqDensity;                       // 頻率密度配置
    uint8_t  li[2];                             // 時域位置 L0和L1
    uint8_t  seqIndexCount;                     // CDM序列索引計數 (1,2,4,8)
    uint8_t  genEvenRB;                         // rho=0.5時的配置選擇
    uint8_t  idxSlotInFrame;                    // 幀內時隙索引
    uint16_t scrambId;                          // 加擾ID
    uint8_t  ki[CUPHY_CSIRS_MAX_KI_INDEX_LENGTH];  // k0-k5 參考位置
    cuphyCsiType_t csiType;                     // CSI類型 (0:TRS, 1:NZP, 2:ZP)
    cuphyCdmType_t cdmType;                     // CDM類型
    float    alpha;                             // X=1時為rho，否則為2*rho
    float    rho;                               // 密度 (0.5, 1, 3)
    float    beta;                              // 功率歸一化因子
    uint16_t cell_index;                        // 小區索引
    uint8_t  enablePrcdBf;                      // 預編碼開關
    uint16_t pmwPrmIdx;                         // 預編碼矩陣索引
} CsirsRxParams;
```

### CsirsRxUeParams

```cpp
typedef struct _CsirsRxUeParams
{
    uint16_t cellIdx;                          // 小區索引
    uint32_t csirsRxParamsStartIdx;            // CSI-RS參數起始索引
    uint32_t csirsRxChEstBufStartIdx;          // 通道估計緩衝起始索引
    uint16_t nCsirs;                           // 此UE的CSI-RS數量
    uint8_t  nRxAnt;                           // 接收天線數
    uint16_t nRxRe;                            // 接收RE數量 (每符號)
} CsirsRxUeParams;
```

**參數說明**:

| 參數 | 說明 | 範圍 |
|------|------|------|
| `nCsirs` | 每UE的CSI-RS資源數 | 1-8 |
| `nRxAnt` | 接收天線數 | 1-64 |
| `nRxRe` | 每符號接收資源元素數 | 0-3276 (273*12) |

---

## CUDA內核

### 主內核：csirsRxChEstKernel

**功能**: 從接收信號中提取CSI-RS、去加擾、計算通道估計

```cpp
__global__ void csirsRxChEstKernel(float2** inputArray,          // 接收信號
                                   float2** chEstArray,          // 輸出通道估計
                                   CsirsRxParams* csirsRxParams,     // CSI-RS配置
                                   CsirsRxUeParams* csirsRxUeParams) // UE配置
{
    int ueIdx = blockIdx.x;           // UE索引
    int csirsRxIdxinUe = blockIdx.y;  // 該UE內的CSI-RS索引
    
    CsirsRxUeParams ueParam = csirsRxUeParams[ueIdx];
    if(csirsRxIdxinUe >= ueParam.nCsirs)
        return;
    
    // 獲取CSI-RS參數
    CsirsRxParams csirsRxParam = 
        csirsRxParams[ueParam.csirsRxParamsStartIdx + csirsRxIdxinUe];
}
```

**網格配置**:
- `gridDim`: (numUes, maxNumRrcParamPerUe, 1)
- `blockDim`: (maxNumPrbPerUe, 1, 1)
- `sharedMem`: 0

---

## 通道估計算法

### 計算流程

```
接收信號 (多個天線, 多個符號)
    ↓
[步驟1] 初始化參數
    ├─ 讀取行表數據 (lenKBarLBar, lenKPrime, lenLPrime, numPorts)
    ├─ 計算RB索引 (startRb + threadIdx.x)
    └─ 密度檢查 (rho=0.5時的RB篩選)
    ↓
[步驟2] 建立相關性積累器
    └─ xcor[kPrime][lPrime][port/kBarLBar][antenna] = complex storage
    ↓
[步驟3] 遍歷所有CSI-RS位置
    ├─ for idxKBarLBar in [0, lenKBarLBar)
    │  ├─ for idxLPrime in [0, lenLPrime)
    │  │  └─ for idxKPrime in [0, lenKPrime)
    │  │     ├─ 計算頻率位置: k = kBar + kPrime + RB_idx * 12
    │  │     ├─ 計算時間位置: l = lBar + lPrime
    │  │     └─ 計算加擾索引: mPrime = floor(RB_idx*alpha) + kPrime + floor(kBar*rho/12)
    │  └─ 對每個CDM序列 (s = 0 to seqIndexCount-1):
    │     ├─ 獲取Wf, Wt CDM係數
    │     ├─ 計算Gold加擾序列
    │     ├─ 反調制: a = (1/beta) * Wf * Wt * sqrt(0.5) * (1 - 2*bit)
    │     └─ 對每個接收天線:
    │        ├─ 提取接收樣本: r[k, l, antenna]
    │        ├─ 共軛: a*
    │        ├─ 複數乘法: xcor = r * a*
    │        └─ 積累到緩衝
    ↓
[步驟4] 通道估計計算
    ├─ 特殊情況 (Row 1 或高密度):
    │  └─ 直接輸出相關值
    └─ 一般情況:
       ├─ 對每個端口求和所有相關值
       ├─ 平均: chEst = sum(xcor) / count
       └─ 寫入: chEst[rbIdx + port*nRb + antenna*nRb*numPorts]
    ↓
輸出: 通道估計係數
```

---

## 核心計算步驟

### 1. 位置計算

```cpp
// 頻率域位置
uint k = kBar + idxKPrime + rbIdx * CUPHY_N_TONES_PER_PRB;

// 時間域位置
uint l = lBar + idxLPrime;

// 加擾序列索引
uint mPrime = floorf(rbIdx * alpha) + idxKPrime + floorf(kBar * rho / 12.0f);
```

### 2. 信號反調制

```cpp
// 計算加擾序列初值
uint32_t c_init_scrambling = ((1 << 10) * 
                              (OFDM_SYMBOLS_PER_SLOT * idxSlotInFrame + l + 1) * 
                              (2 * scrambId + 1) + 
                              scrambId) & 0x7FFFFFFF;

// 獲取Gold序列值
uint32_t val = gold32n(c_init_scrambling, 2 * mPrime);

// CDM加權係數
int wf = seqTable[s*2*4 + idxKPrime];      // 頻域CDM
int wt = seqTable[s*2*4 + 1*4 + idxLPrime]; // 時域CDM

// 反調制信號
float2 a;
a.x = (1.0f / beta) * wf * wt * sqrt(0.5f) * (1.0f - 2.0f * (val & 0x1));
a.y = (1.0f / beta) * wf * wt * sqrt(0.5f) * (1.0f - 2.0f * ((val >> 1) & 0x1));
```

### 3. 相關性計算

```cpp
for(int antIdx = 0; antIdx < nRxAnt; antIdx++)
{
    // 提取接收信號
    float2 input = dataRx[k + nRxRe * l + nRxRe * OFDM_SYMBOLS_PER_SLOT * antIdx];
    
    // 共軛
    float2 a_conj = cuConjf(a);
    
    // 複數乘法：input * conj(a)
    float2 temp;
    temp.x = input.x * a_conj.x - input.y * a_conj.y;
    temp.y = input.x * a_conj.y + input.y * a_conj.x;
    
    // 積累到相關性存儲
    if(row == 1)
    {
        xcor[idxKPrime][idxLPrime][idxKBarLBar][antIdx] = temp;
    }
    else
    {
        uint jj = rowData.cdmGroupIndex[idxKBarLBar];
        uint p = jj * seqIndexCount + s;
        xcor[idxKPrime][idxLPrime][p][antIdx] = temp;
    }
}
```

### 4. 通道估計輸出

#### Row 1 特殊情況

```cpp
if(row == 1)
{
    for(int idxKBarLBar = 0; idxKBarLBar < 3; idxKBarLBar++)
    {
        for(int antIdx = 0; antIdx < nRxAnt; antIdx++)
        {
            chEst[chEstRbIdx + idxKBarLBar + antIdx * nRb] = 
                xcor[0][0][idxKBarLBar][antIdx];
        }
    }
}
```

#### 一般情況 (CDM平均)

```cpp
else
{
    for(int idxPort = 0; idxPort < numPorts; idxPort++)
    {
        for(int idxAnt = 0; idxAnt < nRxAnt; idxAnt++)
        {
            float2 sum = {0.0f, 0.0f};
            int count = 0;
            
            // 對所有K'和L'求和
            for(int idxLPrime = 0; idxLPrime < lenLPrime; idxLPrime++)
            {
                for(int idxKPrime = 0; idxKPrime < lenKPrime; idxKPrime++)
                {
                    sum.x += xcor[idxKPrime][idxLPrime][idxPort][idxAnt].x;
                    sum.y += xcor[idxKPrime][idxLPrime][idxPort][idxAnt].y;
                    count++;
                }
            }
            
            // 平均
            sum.x /= count;
            sum.y /= count;
            
            // 存儲估計值
            chEst[chEstRbIdx + idxPort*nRb + idxAnt*nRb*numPorts] = sum;
        }
    }
}
```

---

## 密度處理

### Rho = 0.5 交替RB

```cpp
bool isEvenRB = (rbIdx & 1) == 0;

if(rho == 0.5f)
{
    // 根據genEvenRB配置選擇
    if((genEvenRB && !isEvenRB) || (!genEvenRB && isEvenRB))
        return;  // 跳過此RB
}
```

### Row 1 與高密度

```cpp
uint16_t chEstRbIdx = threadIdx.x;

if((row == 1) && (freqDensity == 3))
{
    chEstRbIdx = chEstRbIdx * 3;  // 跳躍3個RB
    nRb = nRb * 3;
}
else if(rho == 0.5f)
{
    chEstRbIdx = chEstRbIdx >> 1;  // 密度折半
    nRb = nRb >> 1;
}
```

---

## 共享內存結構

### 相關性積累器

```cpp
// 本地數組：存儲所有相關值
float2 xcor[2][4][32][4];  // 維度: [kPrime][lPrime][port/kBarLBar][antenna]

// 大小: 2 * 4 * 32 * 4 * 2 * sizeof(float) = 2048字節
// 可在所有1024個線程下共享內存內使用 (通常>5KB可用)
```

---

## 配置常數

```cpp
// 表尺寸
const int CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH = 18;  // 行數
const int CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT = 8;            // 最大CDM維度
const int CUPHY_CSIRS_MAX_KI_INDEX_LENGTH = 6;            // ki陣列長度

// 時頻參數
const int OFDM_SYMBOLS_PER_SLOT = 14;        // 每時隙符號
const int CUPHY_N_TONES_PER_PRB = 12;        // 每RB子載波

// 符號位置表行特性
struct CsirsSymbLocRow
{
    uint8_t  lenKBarLBar;      // 不同K-Bar/L-Bar組合數 (1-32)
    uint8_t  lenKPrime;        // K'維度 (1或2)
    uint8_t  lenLPrime;        // L'維度 (1-4)
    uint8_t  numPorts;         // 端口數 (1-32)
    uint8_t  kIndices[];       // K索引表
    uint8_t  kOffsets[];       // K偏移
    uint8_t  lIndices[];       // L索引表
    uint8_t  lOffsets[];       // L偏移
    uint8_t  cdmGroupIndex[];  // CDM組映射
};
```

---

## API函數

### 內核選擇函數

```cpp
cuphyStatus_t cuphyCsirsRxKernelSelect(
    cuphyCsirsRxChEstLaunchCfg_t* pCsirsRxChEstLaunchCfg,
    int numUes,
    int maxNumRrcParamPerUe,
    int maxNumPrbPerUe)
```

**功能**: 設置CSI-RS接收通道估計內核

**參數**:
- `pCsirsRxChEstLaunchCfg`: 啟動配置結構
- `numUes`: 活躍UE數量
- `maxNumRrcParamPerUe`: 每UE最大CSI-RS配置數
- `maxNumPrbPerUe`: 每UE最大RB數

**內核選擇函數**:

```cpp
void kernelSelectCsirsRxChEst(cuphyCsirsRxChEstLaunchCfg_t* pLaunchCfg,
                              int numUes,
                              int maxNumRrcParamPerUe,
                              int maxNumPrbPerUe)
```

**內核配置**:
```
blockDim: (maxNumPrbPerUe, 1, 1)  // 每個線程處理一個RB
gridDim:  (numUes, maxNumRrcParamPerUe, 1)
sharedMem: 0 (局部陣列)
```

---

## 處理流程

### 完整接收流程

```
接收信號到達
    ↓
[調用] cuphyCsirsRxKernelSelect()
    ├─ 設置啟動配置
    └─ kernelSelectCsirsRxChEst()
    ↓
啟動內核: csirsRxChEstKernel<<<(numUes, maxNumRrcParamPerUe), maxNumPrbPerUe>>>
    ↓
[內核執行流程]
    ├─ 每個線程塊處理1個UE + 1個CSI-RS配置
    ├─ 每個線程處理1個RB (0 to maxNumPrbPerUe-1)
    ├─ 每個線程:
    │  ├─ 積累該RB所有CSI-RS位置的相關值
    │  ├─ 根據行配置進行特殊處理或CDM平均
    │  └─ 寫入通道估計
    └─ 同步完成
    ↓
輸出: 所有UE的通道估計係數
```

---

## 數據佈局

### 輸入信號 (dataRx)

```
dataRx[k + l * nRxRe + antenna * OFDM_SYMBOLS_PER_SLOT * nRxRe]

其中:
- k: 子載波索引 (0-3275 for 273 RBs)
- l: OFDM符號索引 (0-13)
- antenna: 天線索引 (0-nRxAnt-1)
- nRxRe: 每符號RE總數
```

### 輸出通道估計 (chEst)

#### Row 1 格式

```
chEst[rbIdx + kBarIdx + antenna * nRb]

其中:
- rbIdx: RB索引
- kBarIdx: kBar索引 (0-2)
- antenna: 天線索引
```

#### 一般格式

```
chEst[rbIdx + port*nRb + antenna*nRb*numPorts]

其中:
- rbIdx: RB索引 (物理層CDM平均後)
- port: CDM端口索引
- antenna: 天線索引
```

---

## 支持的配置

### CDM類型

| CDM類型 | 序列數 | 頻域 | 時域 | numPorts |
|---------|--------|------|------|----------|
| noCDM | 1 | - | - | 1-32 |
| FD-CDM2 | 2 | 2 | 1 | 2 |
| FD2-TD2 | 4 | 2 | 2 | 4 |
| FD2-TD4 | 8 | 2 | 4 | 8 |

### 符號位置行

18行配置支持各種密度、天線端口和頻率分佈

---

## 計算複雜度

### 每個線程的計算

- **循環層級**: 4層 (KBarLBar × LPrime × KPrime × SeqIndexCount)
- **內部操作**:
  - Gold序列生成: 1次
  - CDM加權: 2次 (wf, wt)
  - 反調制: 2次操作
  - 複數乘法: 1次 (每天線)
  - 積累求和: numPorts × nRxAnt次

### 總體複雜度

- **每UE每CSI-RS**: O(nRb × lenKBarLBar × lenLPrime × lenKPrime × seqIndexCount × nRxAnt)
- **典型**: ~1-10ms (取決於參數)

---

## 3GPP標準映射

實現遵循3GPP TS 38.211標準:

| 組件 | 標準 | 說明 |
|------|------|------|
| CSI-RS提取 | 38.211 Sec 7.4.1.5 | 資源位置計算 |
| 去加擾 | 38.211 Sec 5.2.1 | Gold序列 |
| CDM應用 | 38.211 Sec 7.4.1.5.3 | 碼分複用反演 |
| 通道估計 | 38.215 Sec 5.2.2 | LS估計 |

---

## 性能特點

✅ **局部積累** - 使用寄存器/局部內存
✅ **並行化** - 多RB同時處理
✅ **無同步開銷** - 每線程獨立
✅ **內存高效** - 無全局同步
✅ **靈活配置** - 支持18種符號位置

---

## 延伸閱讀

- 3GPP TS 38.211 - 物理通道
- 3GPP TS 38.215 - 物理層測量
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- 通道估計理論
- MIMO接收機設計
