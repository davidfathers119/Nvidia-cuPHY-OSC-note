# 循環冗餘校驗 (CRC)

## 概述

CRC (Cyclic Redundancy Check) 是NVIDIA cuPHY庫中用於**5G信號檢測和驗證**的組件。它實現了對傳輸塊(Transport Block)和碼塊(Code Block)的CRC計算、編碼和解碼。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - CRC](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/crc)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `crc.hpp` | CRC基礎類和工具函數定義 |
| `crc.cuh` | CUDA設備代碼 - LUT和多項式計算 |
| `crc.cu` | 主CUDA內核實現 |
| `crc_encode.hpp` | PDSCH (下行鏈路) CRC編碼描述符 |
| `crc_decode.hpp` | PUSCH (上行鏈路) CRC解碼描述符 |
| `CRC_256_LUTS.h` | 預計算CRC查找表 |

---

## 標準3GPP多項式

### CRC多項式定義

```cpp
const uint32_t G_CRC_24_A = 0x01864CFB;  // 傳輸塊CRC多項式
const uint32_t G_CRC_24_B = 0x01800063;  // 碼塊CRC多項式 (多個CB時)
const uint32_t G_CRC_24_C = 0x01B2B117;  // UCI CRC多項式
const uint32_t G_CRC_16   = 0x11021;     // 小碼塊CRC多項式 (K <= 3840)
const uint16_t G_CRC_11   = 0x0E21;      // 其他應用
const uint8_t  G_CRC_6    = 0x61;        // 小應用
```

**特性**:
- CRC-24-A: 標準24位多項式(TB級)
- CRC-24-B: 多CB場景的24位多項式
- CRC-16: 優化小碼塊(≤3840比特)的16位多項式
- MSB隱含為1(隱含度數)

---

## 配置常數

```cpp
// 碼塊大小限制
const uint32_t MAX_SMALL_A_BITS            = 3824;  // 小碼塊最大比特數
const uint32_t MAX_CB_BIT_SIZE_FOR_CRC16   = 3840;  // CRC-16適用上限
const uint32_t MAX_A_BITS                  = 8424;  // 最大信息比特數
const uint32_t MAX_BYTES_PER_CODE_BLOCK    = 1056;  // 最大碼塊字節數(8448比特)
const uint32_t MAX_WORDS_PER_CODE_BLOCK    = MAX_BYTES_PER_CODE_BLOCK / sizeof(uint32_t);

// CRC長度
const uint32_t SMALL_L_BITS  = 16;   // 小碼塊CRC長度(比特)
const uint32_t LARGE_L_BITS  = 24;   // 標準CRC長度(比特)

// 內核配置
const uint32_t WARP_SIZE           = 32;
const uint32_t GLOBAL_BLOCK_SIZE   = 128;  // 線程塊大小
const uint32_t MAX_WORDS_PER_TRANSPORT_BLOCK = 8320;  // 最大傳輸塊字數
```

---

## 上行鏈路 (PUSCH) CRC解碼

### puschRxCrcDecode 類

```cpp
class puschRxCrcDecode : public cuphyPuschRxCrcDecode
{
public:
    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);
    void init(int reverseBytes);
    void setup(uint16_t                          nSchUes,
               uint16_t*                        pSchUserIdxsCpu,
               uint32_t*                        pOutputCBCRCs,
               uint8_t*                         pOutputTBs,
               const uint32_t*                  pInputCodeBlocks,
               uint32_t*                        pOutputTBCRCs,
               const PerTbParams*               pTbPrmsCpu,
               const PerTbParams*               pTbPrmsGpu,
               void*                            pCpuDesc,
               void*                            pGpuDesc,
               uint8_t                          enableCpuToGpuDescrAsyncCpy,
               cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
               cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
               cudaStream_t                     strm);

private:
    bool      m_reverseBytes;      // 位元組逆序開關
    CUfunction m_cbCrcKernelFunc;  // 碼塊CRC內核
    CUfunction m_tbCrcKernelFunc;  // 傳輸塊CRC內核
};
```

### 下行鏈路 (PDSCH) CRC編碼

```cpp
// 編碼描述符
struct crcEncodeDescr
{
    uint32_t*              d_cbCRCs;               // 碼塊CRC輸出
    uint32_t*              d_tbCRCs;               // 傳輸塊CRC輸出
    const uint32_t*        d_inputTransportBlocks; // 傳輸塊輸入
    uint8_t*               d_codeBlocks;           // 碼塊輸出
    const PdschPerTbParams* d_tbPrmsArray;        // TB參數
    bool                   reverseBytes;           // 位元組反轉
    bool                   codeBlocksOnly;         // 僅CB CRC模式
};

// 準備描述符 (預處理)
struct prepareCrcEncodeDescr
{
    uint32_t           offset[PDSCH_MAX_UES_PER_CELL_GROUP];
    const uint32_t*    d_inputOrigTBs;  // 原始TB地址
    uint32_t*          d_inputTBs;      // 準備後TB緩衝區
    uint32_t*          d_inputTBsTM;    // 測試模式TB緩衝區
    const PdschPerTbParams* d_tbPrmsArray;
};
```

---

## CUDA內核

### 上行鏈路 - 碼塊CRC計算

**內核**: `crcUplinkPuschCodeBlocksKernel`

**功能**: 計算每個碼塊的CRC並組裝傳輸塊

```cpp
__global__ void crcUplinkPuschCodeBlocksKernel(const __grid_constant__ 
                                               puschRxCrcDecodeDescr_t desc)
{
    // 輸入參數
    uint32_t*          outputCBCRCs    = desc.pOutputCBCRCs;
    uint8_t*           outputTBs       = desc.pOutputTBs;
    const uint32_t*    inputCodeBlocks = desc.pInputCodeBlocks;
    uint32_t*          outputTBCRCs    = desc.pOutputTBCRCs;
    const PerTbParams* tbPrmsArray     = desc.pTbPrmsArray;
    bool               reverseBytes    = desc.reverseBytes;
    
    // 線程映射
    uint16_t ueIdx = desc.schUserIdxs[blockIdx.y];  // UE索引
    uint32_t codeBlockIdx = blockIdx.x;             // CB索引
}
```

**算法流程**:

1. **CRC多項式選擇**
   ```cpp
   // 根據碼塊大小選擇多項式
   uint32_t crcPolyBitSize = (K - F) > (MAX_SMALL_A_BITS + SMALL_L_BITS) || 
                             num_CBs > 1 ? LARGE_L_BITS : SMALL_L_BITS;
   ```

2. **多項式計算** (使用查找表加速)
   ```cpp
   // CRC-16用於小碼塊 (K-F <= 3840)
   if(codeBlockDataByteSize <= MAX_CB_BYTE_SIZE_FOR_CRC16)
   {
       crc ^= mulModCRCPolyLUT<uint16_t, 16>(inVal,
                                              G_CRC_16_P_LUT[(tid)],
                                              G_CRC_16_256_LUT,
                                              *POLY_16);
   }
   else
   {
       // CRC-24用於大碼塊
       crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal,
                                              G_CRC_24_A_P_LUT[(tid)],
                                              G_CRC_24_A_256_LUT,
                                              *POLY_A);
   }
   ```

3. **傳輸塊組裝** (去分段)
   ```cpp
   // 將每個碼塊數據寫回到TB緩衝區
   tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid]     = (uint8_t)inVal & 0xFF;
   tb[codeBlockIdx * codeBlockDataByteSize + 4 * tid + 1] = (uint8_t)(inVal >> 8) & 0xFF;
   // ... 更多字節
   ```

### 上行鏈路 - 傳輸塊CRC計算

**內核**: `crcUplinkPuschTransportBlockKernel`

**功能**: 計算組合後傳輸塊的CRC(僅當多個CB時)

```cpp
__global__ void crcUplinkPuschTransportBlockKernel(const __grid_constant__ 
                                                   puschRxCrcDecodeDescr_t desc)
{
    // 輸入: 由CB CRC內核輸出的組裝TB
    const uint32_t* inputTBs     = (uint32_t*)desc.pOutputTBs;
    uint32_t*       outputTBCRCs = desc.pOutputTBCRCs;
    
    // 兩級規約(Two-level reduction)
    // Level 1: 每個線程塊的Warp規約
    crc = xorReductionWarpShared<uint32_t>(crc, shmemBuf);
    
    // Level 2: 原子XOR規約到TB CRC
    if(threadIdx.x == 0)
    {
        atomicXor(&outputTBCRCs[blockIdx.y], crc);
    }
}
```

### 下行鏈路 - 碼塊CRC編碼

**內核**: `crcDownlinkPdschCodeBlocksKernel`

**功能**: 從傳輸塊分段生成碼塊並計算CRC

```cpp
__global__ void crcDownlinkPdschCodeBlocksKernel(const __grid_constant__ 
                                                 crcEncodeDescr_t desc)
{
    // 分段操作
    uint32_t cbShiftBits = (32 - (size % 32)) % 32;
    
    // 異步內存複製(CUDA 11.1+)
    auto group = cg::this_thread_block();
    cg::memcpy_async(group, shmemBuf + MAX_WORDS_PER_CODE_BLOCK,
                     CB_input_words, sizeof(uint32_t) * CB_data_word_size);
    cg::wait(group);
    
    // CRC計算和填充處理
    while(tid < size)
    {
        inVal = shmemBuf[tid];
        
        if(reverseBytes)
        {
            inVal = __brev(inVal);      // 位反轉
            inVal = swap<32>(inVal);    // 字節交換
        }
        
        // 多項式計算
        crc ^= mulModCRCPolyLUT<uint32_t, 24>(inVal, tabVal,
                                               G_CRC_24_B_256_LUT, *POLY_B);
    }
}
```

### 下行鏈路 - 傳輸塊CRC編碼

**內核**: `crcDownlinkPdschTransportBlockKernel`

**功能**: 計算多個CB時的TB級CRC

### 緩衝區準備內核

**內核**: `prepare_crc_buffers`

**功能**: 預處理PDSCH輸入以滿足CRC內核要求

**特性**:
- 32位對齊
- 位反轉 (`__brev()`)
- 字節順序調整
- 填充處理
- 支持主機固定內存直接訪問

```cpp
__global__ void prepare_crc_buffers(prepareCrcEncodeDescr_t* p_desc)
{
    // 位反轉和字節順序調整
    uint32_t value = __byte_perm(temp_value, 0, 0x0123);
    d_inputTBs[total_src_tb_offset + CB_element_offset] = __brev(value);
}
```

---

## 多項式計算函數

### 模GF(2)多項式乘法

#### 基本乘法

```cpp
template <typename T, uint32_t size>
__device__ T mulModPoly(T a, T b, T poly)
{
    T prod;
#pragma unroll
    for(int i = 0; i < size; i++)
    {
        prod ^= (b & 1) ? a : 0;
        a = (a << 1) ^ ((a & (1 << (size - 1))) ? poly : 0);
        b >>= 1;
    }
    return prod;
}
```

#### 基於查找表的乘法

```cpp
template <typename T, uint32_t size>
__device__ uint32_t mulModCRCPolyLUT(uint32_t a,
                                     T        b,
                                     uint32_t LUT[4][256],  // 預計算表
                                     T        poly)
{
    T prod = 0;
    
    // 從LUT中4個字節的表項組合結果
    T crc = LUT[3][static_cast<uint8_t>(a)]         ^
            LUT[2][static_cast<uint8_t>(a >> 8)]    ^
            LUT[1][static_cast<uint8_t>(a >> 16)]   ^
            LUT[0][static_cast<uint8_t>(a >> 24)];
    
    // 用多項式b對結果執行多項式乘法
#pragma unroll
    for(int i = 0; i < size; i++)
    {
        prod ^= (crc & 1) ? b : 0;
        b = (b << 1) ^ (b & (1 << (size - 1)) ? poly : 0);
        crc >>= 1;
    }
    return prod;
}
```

### 高級多項式乘法變體

#### 小端24位多項式乘法

```cpp
template <typename uintCRC_t, int uintCRCBitLength>
CUDA_BOTH uintCRC_t mulModCRCPoly32_1LR(const uint32_t   a,
                                        const uintCRC_t* table,
                                        uintCRC_t        poly,
                                        int              msB = 3)
{
    // 逐字節處理，支持部分比特寬度的查找表
}
```

---

## 數據規約函數

### Warp級XOR規約

```cpp
template <typename uintCRC_t>
__inline__ __device__ uintCRC_t warpReduceSum(uintCRC_t val)
{
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val ^= __shfl_down_sync(FULL_MASK, val, offset, WARP_SIZE);
    return val;
}
```

### 共享內存XOR規約

```cpp
template <typename uintCRC_t>
__device__ inline uintCRC_t xorReductionWarpShared(uintCRC_t  input,
                                                   uintCRC_t* shared)
{
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;
    
    // 第1層: 每個Warp規約
    input = warpReduceSum<uintCRC_t>(input);
    
    if(lane == 0) 
        shared[wid] = input;
    
    __syncthreads();
    
    // 第2層: 第一個Warp跨所有部分結果
    input = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;
    
    if(wid == 0)
        input = warpReduceSum<uintCRC_t>(input);
    
    return input;
}
```

---

## 查找表(LUT)結構

### CRC多項式查找表

```cpp
// 設備常數內存中的預計算表
__constant__ uint32_t G_CRC_24_A_256_LUT[4][256];
__constant__ uint32_t G_CRC_24_B_256_LUT[4][256];
__constant__ uint32_t G_CRC_16_256_LUT[4][256];

// 多項式冪表(用於流式計算)
__constant__ uint32_t G_CRC_24_A_P_LUT[LARGE_LUT_SIZE];  // 大小: 8425
__constant__ uint32_t G_CRC_24_B_P_LUT[LARGE_LUT_SIZE];
__constant__ uint16_t G_CRC_16_P_LUT[SMALL_LUT_SIZE];    // 大小: 3825
```

### LUT大小

```cpp
const uint32_t G_CRC_24_A_P_LUT_SIZE = 8425;  // CRC-24-A多項式表大小
const uint32_t G_CRC_24_B_P_LUT_SIZE = 8425;  // CRC-24-B多項式表大小
const uint32_t G_CRC_16_P_LUT_SIZE   = 3825;  // CRC-16多項式表大小
const uint32_t G_CRC_16_256_LUT_SIZE = 4 * 256;
const uint32_t G_CRC_24_A_256_LUT_SIZE = 4 * 256;
const uint32_t G_CRC_24_B_256_LUT_SIZE = 4 * 256;
```

---

## 應用流程

### 上行鏈路 (PUSCH) 解碼

```
輸入碼塊序列 (多個UE)
        ↓
[內核1] crcUplinkPuschCodeBlocksKernel
  ├─ 計算每個CB的CRC-24-B或CRC-16
  ├─ 去分段(CB → TB)
  └─ 輸出: CB CRC, 組裝TB
        ↓
[內核2] crcUplinkPuschTransportBlockKernel (可選，僅多CB)
  ├─ 計算TB級CRC-24-A
  └─ 使用原子XOR規約
        ↓
輸出: TB CRC, 傳輸塊數據
```

### 下行鏈路 (PDSCH) 編碼

```
傳輸塊序列 (多個UE)
        ↓
[內核0] prepare_crc_buffers (預處理)
  ├─ 32位對齊
  ├─ 位反轉(__brev)
  ├─ 字節順序調整
  └─ 填充處理
        ↓
[內核1] crcDownlinkPdschCodeBlocksKernel
  ├─ 分段(TB → CB)
  ├─ 計算CB CRC-24-B或CRC-16
  └─ 輸出: 碼塊 + CRC
        ↓
[內核2] crcDownlinkPdschTransportBlockKernel (可選，多CB)
  ├─ 計算TB級CRC-24-A
  └─ 原子XOR規約
        ↓
輸出: 編碼碼塊序列
```

---

## 網格和塊配置

### 碼塊CRC內核

```cpp
dim3 gCBSize(maxNCBsPerTB,      // gridDimX = 每TB最大CB數
             nTBs,              // gridDimY = TB數
             1);
dim3 blockDim(GLOBAL_BLOCK_SIZE, // 128個線程/塊
              1, 1);
uint32_t shmem_size = sizeof(uint32_t) * WARP_SIZE;
```

### 傳輸塊CRC內核

```cpp
uint32_t gridSizeTBX = (tbSize + GLOBAL_BLOCK_SIZE - 1) / GLOBAL_BLOCK_SIZE;
dim3 gTBSize(gridSizeTBX,       // gridDimX = TB字數/(塊大小)
             nTBs,              // gridDimY = TB數
             1);
uint32_t shmem_size = sizeof(uint32_t) * WARP_SIZE;
```

---

## 性能特性

✅ **預計算LUT** - 快速多項式計算
✅ **Warp級規約** - 無全線程塊同步
✅ **共享內存優化** - 兩級XOR規約
✅ **異步內存複製** - CUDA 11.1+支持
✅ **位反轉優化** - 使用`__brev()`內置函數
✅ **字節順序處理** - `__byte_perm()`硬件支持
✅ **原子操作** - 無鎖TB CRC規約

---

## 3GPP規範應用

該實現遵循3GPP TS 38.212標準:

| 參數 | 說明 |
|------|------|
| **CRC-24-A** | 傳輸塊級CRC(上行/下行) |
| **CRC-24-B** | 碼塊級CRC(多CB時) |
| **CRC-16** | 小碼塊優化(K ≤ 3840) |
| **CRC-11** | UCI(上行控制信息) |
| **CRC-6** | RNTI掩碼 |

---

## 延伸閱讀

- 3GPP TS 38.212 - CRC規範
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- CRC理論與有限域算術
- 多項式哈希和LUT優化技術
