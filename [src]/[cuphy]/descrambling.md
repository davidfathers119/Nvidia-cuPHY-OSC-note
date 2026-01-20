# 去加擾 (Descrambling)

## 概述

Descrambling是NVIDIA cuPHY庫中用於**5G信號去加擾處理**的組件。它實現了Gold序列生成和對數似然比(LLR)信號的去加擾操作。去加擾是接收端解調的基本步驟，用於移除發射端施加的加擾序列。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - Descrambling](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/descrambling)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `descrambling.hpp` | 常數定義和命名空間 |
| `descrambling.cuh` | CUDA設備代碼 - Gold序列生成 |
| `descrambling.cu` | CUDA內核實現 - 去加擾處理 |
| `testDescrambling.cpp` | 測試程序 |

---

## 核心常數和配置

```cpp
namespace descrambling {
    // LUT配置
    constexpr uint32_t BITS_PROCESSED_PER_LUT_ENTRY = 32;
    constexpr uint32_t BITS_PROCESSED_PER_LUT_ENTRY_MASK = 0xFFFFFFFF;
    
    // LFSR多項式 (Galois表示)
    constexpr uint32_t POLY_1        = 0x80000009;  // x^31 + x^3 + 1
    constexpr uint32_t POLY_2        = 0x8000000F;  // x^31 + x^3 + x^2 + x + 1
    constexpr uint32_t POLY_1_GMASK  = 0x00000009;
    constexpr uint32_t POLY_2_GMASK  = 0x0000000F;
    
    // 內核配置
    constexpr uint32_t GLOBAL_BLOCK_SIZE = 512;  // 線程塊大小
    constexpr uint32_t WARP_SIZE         = 32;   // Warp大小
    constexpr uint32_t WORD_SIZE         = 32;   // 字大小(比特)
    
    // 3GPP規範
    constexpr uint32_t NC = 1600;  // 跳過比特數
}
```

---

## Gold序列生成

### 理論基礎

Gold序列由兩個線性反饋移位寄存器(LFSR)生成：

```
Gold序列 = LFSR1 XOR LFSR2

LFSR1多項式: x^31 + x^3 + 1
LFSR2多項式: x^31 + x^3 + x^2 + x + 1
```

### Gold序列函數

#### 基本函數：gold32

```cpp
__device__ inline uint32_t gold32(uint32_t seed2, uint32_t n)
```

**功能**: 計算Gold序列中第n個字(32比特)

**參數**:
- `seed2`: LFSR2初值(種子)
- `n`: 字索引 (n*32即比特索引)

**算法**:
```
1. 初始化:
   - state1: 從GOLD_1_SEQ_LUT[n/32]查表 (LFSR1預計算)
   - state2: __brev(seed2) >> 1 (反轉種子)
   
2. 多項式運算:
   - state2 = polyBMulHigh31(state2)
   - prod2 = mulModPoly31LUT(state2, GOLD_2_32_P_LUT[n/32], POLY_2)
   
3. Galois LFSR運算:
   - fstate2 = galois31MaskLFSRWord(prod2)
   
4. 輸出:
   - output2 = fibonacciLFSR2_1bit(fstate2)
   - return output1 XOR output2
```

#### 帶偏移：gold32n

```cpp
__device__ inline uint32_t gold32n(uint32_t seed2, uint32_t n)
```

**功能**: 計算Gold序列中任意比特n開始的32比特

**特點**: 支持比特級精度(n可以不是32的倍數)

**實現**:
```
1. 計算基礎序列:
   - fstate1 = GOLD_1_SEQ_LUT[n/32] & 0x7FFFFFFF
   - fstate2 = galois31MaskLFSRWord(prod2)
   
2. 移位對齐:
   - seq1f = fibonacciLFSR1_n32(fstate1) >> (n % 32)
   - output2 = fibonacciLFSR2_n32(fstate2) >> (n % 32)
   
3. 跨字邊界處理:
   - seq1f |= (n % 32) ? (fstate1 << (32 - (n % 32))) : 0
   - output2 |= (n % 32) ? (fstate2 << (32 - (n % 32))) : 0
   
4. 返回XOR結果
```

---

## LFSR實現

### Fibonacci LFSR - 多項式1

```cpp
CUDA_BOTH inline uint32_t fibonacciLFSR1_n32(uint32_t& state)
```

**多項式**: x^31 + x^3 + 1

**特性**: 32比特專用、優化實現

**遞推關係**:
```
s(b, i+1) = s(b+1, i)           (b < 30)
s(30, i+1) = s(31, i) XOR s(0, i) XOR s(3, i)
s(31, i+1) = 0

輸出: r(b, 32) = s(b+1, 0) (對b = 0-26)
```

### Fibonacci LFSR - 多項式2

```cpp
CUDA_BOTH inline uint32_t fibonacciLFSR2_n32(uint32_t& state)
```

**多項式**: x^31 + x^3 + x^2 + x + 1

**特性**: 4項反饋（比多項式1更複雜）

**遞推關係**:
```
s(b, i+1) = s(b+1, i)                                    (b < 30)
s(30, i+1) = s(31, i) XOR s(0, i) XOR s(1, i) XOR s(2, i) XOR s(3, i)
s(31, i+1) = 0
```

### Galois LFSR

```cpp
__device__ inline uint32_t galois31MaskLFSRWord(uint32_t state)
```

**功能**: Galois型LFSR，31比特輸出

**特性**: 使用`__brev()`位反轉指令優化

**算法**:
```
1. 位反轉: rev_state = __brev(state)
2. 提取高28比特: ((rev_state >> 1) & 0xFFFFFFF)
3. 計算反饋位並填充上層比特
4. 返回31比特結果
```

---

## 多項式乘法

### 簡單乘法

```cpp
CUDA_BOTH inline uint32_t mulModPoly31LUT(uint32_t a,
                                          uint32_t b,
                                          uint32_t poly)
```

**功能**: 31比特模GF(2)多項式乘法

**實現**:
```cpp
uint32_t prod = 0;
uint32_t crc = a ^ (a >= POLY_2) * POLY_2;

for(int i = 0; i < 31; i++)
{
    prod ^= (crc & 1) * b;
    b = (b << 1) ^ (b & (1 << 30) ? poly : 0);
    crc >>= 1;
}
return prod;
```

### 高位乘積

```cpp
CUDA_BOTH inline uint32_t polyBMulHigh31(uint32_t a)
```

**功能**: 快速計算 a × POLY_2 的高31比特

**實現**:
```cpp
uint32_t prodHi = (a >> 30) ^ (a >> 29) ^ (a >> 28) ^ a;
return prodHi;
```

---

## 去加擾內核

### 主內核：descrambleKernel

```cpp
__global__ void descrambleKernel(float*          llrs,              // LLR輸入/輸出
                                 uint32_t        size,              // 總大小
                                 const uint32_t* tbBoundaryArray,   // TB邊界
                                 const uint32_t* cinitArray)        // C_init初值
```

**功能**: 對所有傳輸塊的LLR進行去加擾

**網格配置**:
- `gridDim.x`: maxNCodeBlocks (每TB最大碼塊數)
- `gridDim.y`: nTBs (傳輸塊數)
- `blockDim.x`: GLOBAL_BLOCK_SIZE (512)
- `sharedMem`: (blockDim.x / WARP_SIZE) × sizeof(uint32_t)

### 算法流程

```
輸入: 加擾的LLR序列
    ↓
[步驟1] 初始化
    ├─ myTBBase = tbBoundaryArray[blockIdx.y]
    ├─ myTBEnd = tbBoundaryArray[blockIdx.y + 1]
    ├─ myCinit = cinitArray[blockIdx.y]
    └─ tid = blockIdx.x * blockDim.x + threadIdx.x
    ↓
[步驟2] 生成Gold序列
    ├─ 分批加載: 每個Warp (32線程) 生成1個字 (32比特)
    ├─ sharedSeq[warpIdx] = gold32(myCinit, startBit + warpIdx*32)
    └─ 同步共享內存
    ↓
[步驟3] 並行去加擾
    ├─ for t = tid + myTBBase; t < myTBEnd; t += gridDim.x * blockDim.x:
    │  ├─ seq = sharedSeq[threadIdx.x / WARP_SIZE]
    │  ├─ s = (seq >> (threadIdx.x % WARP_SIZE)) & 1  // 提取該比特
    │  ├─ sn = (s + 1) & 0x1                           // 邏輯非
    │  ├─ llrs[t] = -llrs[t] * s + llrs[t] * sn       // 根據加擾比特改變符號
    │  └─ 寫回結果
    ↓
輸出: 去加擾的LLR序列
```

### 符號改變邏輯

```
s = 0: llrs[t] = -llrs[t] * 0 + llrs[t] * 1 = llrs[t]    (無改變)
s = 1: llrs[t] = -llrs[t] * 1 + llrs[t] * 0 = -llrs[t]   (反轉符號)
```

---

## 類接口

### cuphyDescramble 類

```cpp
class cuphyDescramble
{
public:
    cuphyStatus_t loadParams(const uint32_t* tbBoundaryArray,
                             const uint32_t* cinitArray,
                             uint32_t nTBs,
                             uint32_t maxNCodeBlocks);
    
    cuphyStatus_t loadInput(float* llrs);
    
    cuphyStatus_t launch(float* llrs = nullptr,
                        bool timeIt = false,
                        uint32_t NRUNS = 10000,
                        cudaStream_t strm = 0);
    
    cuphyStatus_t storeOutput(float* llrs);
    
    void cleanup();

private:
    unique_device_ptr<float>    d_llrs_;
    unique_device_ptr<uint32_t> d_tbBoundaryArray_;
    unique_device_ptr<uint32_t> d_cinitArray_;
    uint32_t maxNCodeBlocks_;
    uint32_t nTBs_;
    uint32_t totalSize_;
};
```

---

## API函數

### 初始化

```cpp
void cuphyDescrambleInit(void** descrambleEnv)
```

**功能**: 創建descramble對象

**使用**:
```cpp
void* descrambleEnv;
cuphyDescrambleInit(&descrambleEnv);
```

### 清理

```cpp
void cuphyDescrambleCleanUp(void** descrambleEnv)
```

**功能**: 銷毀descramble對象

### 加載參數

```cpp
cuphyStatus_t cuphyDescrambleLoadParams(void** descrambleEnv,
                                        uint32_t nTBs,
                                        uint32_t maxNCodeBlocks,
                                        const uint32_t* tbBoundaryArray,
                                        const uint32_t* cinitArray)
```

**功能**: 設置傳輸塊邊界和初值

**參數**:
- `tbBoundaryArray`: TB邊界偏移數組 (大小: nTBs+1)
- `cinitArray`: 每個TB的C_init初值 (大小: nTBs)

### 加載輸入

```cpp
cuphyStatus_t cuphyDescrambleLoadInput(void** descrambleEnv, float* llrs)
```

**功能**: 加載LLR數據到GPU

### 執行去加擾

```cpp
cuphyStatus_t cuphyDescramble(void** descrambleEnv,
                              float* d_llrs,
                              bool timeIt = false,
                              uint32_t NRUNS = 1,
                              cudaStream_t strm = 0)
```

**功能**: 執行去加擾內核

### 輸出結果

```cpp
cuphyStatus_t cuphyDescrambleStoreOutput(void** descrambleEnv, float* llrs)
```

**功能**: 複製結果回主機

### 一體化接口

```cpp
cuphyStatus_t cuphyDescrambleAllParams(float* llrs,
                                       const uint32_t* tbBoundaryArray,
                                       const uint32_t* cinitArray,
                                       uint32_t nTBs,
                                       uint32_t maxNCodeBlocks,
                                       int timeIt = 0,
                                       uint32_t NRUNS = 1,
                                       cudaStream_t stream = 0)
```

**功能**: 一次性調用完整流程

---

## C_init計算

Gold序列初值計算遵循3GPP TS 38.211標準：

```
c_init = (N_ID^(cell) * 2^30 + n_rnti * 2^14 + (n_s mod 4) * 2^12 + k) mod 2^31

其中:
- N_ID^(cell): 小區ID
- n_rnti: RNTI (無線電網絡臨時標識)
- n_s: 時隙號
- k: 子通道索引
```

---

## 查找表

### GOLD_1_SEQ_LUT

```cpp
__constant__ uint32_t GOLD_1_SEQ_LUT[]
```

**用途**: LFSR1預計算序列表

**大小**: (MAX_TB_SIZE / 32) 字

**內容**: 所有可能的LFSR1輸出狀態

### GOLD_2_32_P_LUT

```cpp
__constant__ uint32_t GOLD_2_32_P_LUT[]
```

**用途**: LFSR2多項式乘法加速表

**大小**: (MAX_TB_SIZE / 32) 字

---

## 共享內存優化

### 批量Gold序列生成

```
共享內存佈局:
┌──────────────────────────────┐
│  Warp 0: gold32(myCinit, 0)  │ → sharedSeq[0]
│  Warp 1: gold32(myCinit, 32) │ → sharedSeq[1]
│  ...                          │
│  Warp N: gold32(myCinit, N*32)│ → sharedSeq[N]
└──────────────────────────────┘

使用:
  seq = sharedSeq[threadIdx.x / WARP_SIZE]
  s = (seq >> (threadIdx.x % WARP_SIZE)) & 1
```

**優勢**:
- 避免每個線程單獨生成序列
- 提高Gold序列生成吞吐量
- 減少寄存器使用

---

## 性能特性

✅ **Warp級並行化** - 32線程共享1個Gold序列字
✅ **共享內存優化** - 批量生成減少計算
✅ **無全局同步** - 只有線程塊內同步
✅ **預計算LUT** - 加速多項式運算
✅ **bit級精度** - 支持任意比特索引

---

## 計算複雜度

### 每個LLR處理

- **Gold序列提取**: O(1) 共享內存訪問
- **符號改變**: O(1) 數學運算
- **內存訪問**: 1次讀 + 1次寫

### 總體性能

- **吞吐量**: 每秒 ~GB/s (取決於GPU)
- **延遲**: ~ms (對於GB級LLR)

---

## 3GPP標準映射

實現遵循3GPP TS 38.211標準:

| 組件 | 標準 | 說明 |
|------|------|------|
| Gold序列 | 38.211 Sec 5.2.1 | 加擾序列生成 |
| LFSR1 | 38.211 Sec 5.2.1.1 | x^31 + x^3 + 1 |
| LFSR2 | 38.211 Sec 5.2.1.2 | x^31 + x^3 + x^2 + x + 1 |
| C_init | 38.211 Sec 5.2.1 | 序列初值計算 |
| Nc值 | 38.211 Sec 5.2.1 | 跳過1600比特 |

---

## 使用示例

### 基本用法

```cpp
// 初始化
void* descrambleEnv;
cuphyDescrambleInit(&descrambleEnv);

// 設置參數
uint32_t tbBoundaryArray[] = {0, 1000, 2500};  // 3個TB
uint32_t cinitArray[] = {0x12345678, 0x87654321};
cuphyDescrambleLoadParams(&descrambleEnv, 2, 1, 
                          tbBoundaryArray, cinitArray);

// 加載輸入
float* h_llrs = ...;  // 主機LLR
cuphyDescrambleLoadInput(&descrambleEnv, h_llrs);

// 執行
cuphyDescramble(&descrambleEnv, nullptr);

// 輸出
float* result = ...;
cuphyDescrambleStoreOutput(&descrambleEnv, result);

// 清理
cuphyDescrambleCleanUp(&descrambleEnv);
```

### 一體化用法

```cpp
cuphyStatus_t status = cuphyDescrambleAllParams(
    llrs,
    tbBoundaryArray,
    cinitArray,
    nTBs,
    maxNCodeBlocks,
    false,  // timeIt
    1,      // NRUNS
    stream
);
```

---

## 延伸閱讀

- 3GPP TS 38.211 - 物理層過程
- 3GPP TS 38.212 - 多路復用和編碼
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- Gold序列和LFSR理論
- GF(2)多項式運算
