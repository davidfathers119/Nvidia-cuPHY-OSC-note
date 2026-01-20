# Gold Sequence 生成與擾碼模組

## 概述

Gold sequence 是 5G NR (TS 38.211/38.213) 中用於信號擾碼和去擾碼的偽隨機二進制序列（Pseudo-Random Binary Sequence, PRBS）。它由兩條並聯的線性回饋移位暫存器（LFSR）生成，用於實現：

- **下行鏈路**: PDSCH/PDCCH/DMRS 擾碼、CSI-RS 擾碼、SS 塊同步擾碼
- **上行鏈路**: PUSCH/PUCCH/SRS 擾碼、DMRS 擾碼
- **信號去擾碼**: UCI（HARQ-ACK/CSI）解擾、data 解擾

**數學基礎**:
$$c(n) = x_1(n) \oplus x_2(n), \quad n = 0, 1, 2, \ldots, N-1$$

其中：
- $x_1(n)$: 固定多項式 $x^{31} + x^3 + 1$ 生成的 LFSR1 序列
- $x_2(n)$: 可配置多項式 $x^{31} + x^3 + x^2 + x + 1$ 生成的 LFSR2 序列，初值由 $c_{init}$ 確定
- $\oplus$: 邏輯 XOR 運算

**架構優化**: NVIDIA cuPHY 使用預計算查表（LUT）加速生成：
- `GOLD_1_SEQ_LUT`: 固定 x1 序列（跳過前 1600 位 NC）
- `GOLD_2_32_P_LUT`: x2 基序列的 32 位多項式冪 (32-bit word powers)

---

## 架構概述

### 核心組件

Gold sequence 模組包含以下主要功能：

| 組件 | 功能 | 使用場景 |
|------|------|---------|
| **LFSR1 LUT** | 預計算固定 x1 序列 | 所有信號擾碼 |
| **LFSR2 生成器** | 根據 $c_{init}$ 動態生成 x2 | 每個 UE/slot/符號 |
| **Gold32 核心函數** | 32 位 Gold 碼生成 | 快速序列生成 |
| **Gold32n 函數** | 任意位置 32 位序列 | 非對齊位置取樣 |
| **Descrambling** | XOR 應用到 LLR 或信號 | 解擾 (PUSCH/PUCCH) |

### 檔案結構

```
cuphy/descrambling/
├── descrambling.cuh          # CUDA 核心 LFSR/Gold 演算法
├── descrambling.cu           # 主 descrambling 核心
├── descrambling.hpp          # C++ 介面
├── descrambling.cpp          # C 包裝函數
├── testDescrambling.cpp      # 單元測試
├── GOLD_1_SEQ_LUT.h          # x1 序列 LUT (固定)
├── GOLD_2_32_P_LUT.h         # x2 冪次 LUT (固定)
└── crc/gen_crc_LUTs.cpp      # LUT 生成工具

特定應用包裝:
├── pucch_F2_front_end/goldSequenceHostF2.cpp
├── pucch_F3_front_end/goldSequenceHostF3.cpp
└── srs_chEst/goldSequenceHostSrs.cpp
```

---

## LFSR 演算法

### Fibonacci LFSR1 (第一多項式)

**多項式**: $x^{31} + x^3 + 1$

```cpp
// 生成 n 位
CUDA_BOTH inline uint32_t fibonacciLFSR1(uint32_t& state, uint32_t n)
{
    uint32_t res = 0;
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ (state >> 3);  // 反饋: x[0] XOR x[3]
        bit = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31;
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}

// 優化版本 (n=32 bits)
CUDA_BOTH inline uint32_t fibonacciLFSR1_n32(uint32_t& state)
{ /* 預計算線性遞推 */ }
```

### Fibonacci LFSR2 (第二多項式)

**多項式**: $x^{31} + x^3 + x^2 + x + 1$

```cpp
CUDA_BOTH inline uint32_t fibonacciLFSR2(uint32_t& state, uint32_t n, uint32_t resInit = 0)
{
    uint32_t res = resInit;
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
        bit = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31;
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}

// 單位生成 (1 bit)
CUDA_BOTH inline uint32_t fibonacciLFSR2_1bit(uint32_t& state)
{
    uint32_t res = state;
    uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
    bit = bit & 1;
    state >>= 1;
    state ^= (bit << 30);
    res ^= (state >> 30) << 31;
    return res;
}
```

### Galois LFSR (多項式乘法)

用於加速 x2 初始化：

```cpp
// 編譯高效 31 位多項式乘法
CUDA_BOTH inline uint32_t galois31LFSRWord(uint32_t state, uint32_t galoisMask, uint32_t n = 31)
{
    uint32_t res = 0;
    uint32_t msbMask = (1 << 30);
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (msbMask & state);
        uint32_t pred = bit != 0;
        state <<= 1;
        state ^= pred * galoisMask;
        res ^= pred << i;
    }
    return res;
}
```

---

## Gold 序列生成

### 32 位 Gold 碼生成 (LUT 基礎)

**演算法**: 結合 GOLD_1_SEQ_LUT 預計算 x1 和動態 x2 計算

```cpp
// GPU 版本
__device__ inline uint32_t gold32(uint32_t seed2, uint32_t n)
{
    uint32_t prod2;
    uint32_t output1 = GOLD_1_SEQ_LUT[n / WORD_SIZE];
    
    // x2 處理: 使用 Galois 乘法快速計算 x2 初始狀態
    uint32_t state2 = __brev(seed2) >> 1;  // 反向 seed2 的 31 位
    state2 = polyBMulHigh31(state2);       // 應用多項式乘法
    prod2 = mulModPoly31LUT(state2, GOLD_2_32_P_LUT[(n) / WORD_SIZE], POLY_2);
    
    uint32_t fstate2 = galois31MaskLFSRWord(prod2);
    uint32_t output2 = fibonacciLFSR2_1bit(fstate2);
    
    return output1 ^ output2;
}

// CPU 等效版本
__host__ inline uint32_t gold32n_CPU(uint32_t seed2, uint32_t n)
{
    uint32_t prod2 = mulModPoly31LUT_CPU(state2, GOLD_2_32_P_LUT_CPU[(n) / WORD_SIZE], POLY_2);
    uint32_t fstate2 = galois31LFSRWord_CPU(prod2, POLY_2_GMASK, 31);
    uint32_t output2 = fibonacciLFSR2_CPU(fstate2, 32);
    
    uint32_t fstate1 = GOLD_1_SEQ_LUT_CPU[n / WORD_SIZE] & 0x7FFFFFFF;
    uint32_t seq1f = fibonacciLFSR1_CPU(fstate1, 32);
    
    return seq1f ^ output2;
}
```

### 任意位置 32 位取樣 gold32n

**用途**: 從非 32 位邊界位置取樣序列

```cpp
__device__ inline uint32_t gold32n(uint32_t seed2, uint32_t n)
{
    // n: 位位置 (可以是任意值，不限 32 位對齊)
    // 返回: 從位置 floor(n/32) 開始的 32 位
    
    uint32_t wordIdx = n / WORD_SIZE;
    uint32_t bitOffset = n % 32;
    
    uint32_t output1 = GOLD_1_SEQ_LUT[wordIdx];
    uint32_t output2 = /* x2 計算 */;
    
    uint32_t result = output1 ^ output2;
    
    // 移位對齊
    result = (result >> bitOffset);
    if(bitOffset) result |= ((GOLD_1_SEQ_LUT[wordIdx + 1] ^ output2_next) << (32 - bitOffset));
    
    return result;
}
```

---

## C/C++ 函數介面

### Descrambling 初始化

```cpp
// 初始化 descrambling 環境
void cuphyDescrambleInit(void** descrambleEnv)
{
    descrambling::cuphyDescramble* descramblePtr = new descrambling::cuphyDescramble();
    *descrambleEnv = descramblePtr;
}

// 清理資源
void cuphyDescrambleCleanUp(void** descrambleEnv)
{
    delete static_cast<descrambling::cuphyDescramble*>((*descrambleEnv));
}
```

### 參數載入

```cpp
// 載入 TB 邊界和初始種子
cuphyStatus_t cuphyDescrambleLoadParams(void**          descrambleEnv,
                                        uint32_t        nTBs,
                                        const uint32_t* tbBoundaryArray,  // TB 起始位置
                                        const uint32_t* cinitArray,       // 每個 TB 的 c_init
                                        uint32_t        maxNCodeBlocks,
                                        int             timeIt = 0);

// 載入 LLR 輸入
cuphyStatus_t cuphyDescrambleLoadInput(void** descrambleEnv, float* llrs);
```

### 執行 Descrambling

```cpp
// 執行 GPU 核心
cuphyStatus_t cuphyDescramble(void**       descrambleEnv,
                              float*       d_llrs,           // Device pointer to LLRs
                              bool         timeIt = false,
                              uint32_t     NRUNS = 1,
                              cudaStream_t strm = nullptr);

// 高層包裝 (一次性呼叫)
cuphyStatus_t cuphyDescrambleAllParams(float*          llrs,
                                       const uint32_t* tbBoundaryArray,
                                       const uint32_t* cinitArray,
                                       uint32_t        nTBs,
                                       uint32_t        maxNCodeBlocks,
                                       int             timeIt = 0,
                                       uint32_t        NRUNS = 1,
                                       cudaStream_t    stream = nullptr);

// 儲存輸出結果
cuphyStatus_t cuphyDescrambleStoreOutput(void** descrambleEnv, float* llrs);
```

---

## 初始種子 (c_init) 計算

### PUSCH/PDSCH Descrambling

```cpp
// Data 擾碼
c_init = (n_RNTI * 2^15 + N_id) & 0x7FFFFFFF
// 其中: n_RNTI = UE 無線網路臨時識別符
//       N_id = 物理層小區 ID

// DMRS 擾碼
c_init = (2^17 * (slotNum * 14 + symbolNum + 1) * (2*N_DMRS_ID + 1) + 2*N_DMRS_ID + n_scid) & 0x7FFFFFFF
// n_scid = 擾碼序列初始化 ID (0 或 1)
```

### CSI-RS Descrambling

```cpp
c_init = (2^10 * (slotNum * 14 + symbolNum + 1) * (2*scrambId + 1) + scrambId) & 0x7FFFFFFF
```

### SS Block 同步信號

```cpp
// Scrambling
c_init = (n_ID >> 0) & 0x3FFFFFFF
// DMRS
c_init = (2^11 * (i_ssb + 1) * (n_ID/4 + 1) + 2^6 * (i_ssb + 1) + n_ID%4) & 0x7FFFFFFF
```

---

## 效能特性

### 計算複雜度

| 操作 | 複雜度 | 備註 |
|------|--------|------|
| gold32() | O(1) | LUT 查找 2 次 + 邏輯運算 |
| gold32n() | O(1) | 相同 + 移位對齐 |
| fibonacciLFSR1() | O(n) | 循環 n 次，預計算可到 O(1) |
| fibonacciLFSR2() | O(n) | 循環 n 次，預計算可到 O(1) |
| descramble() | O(N) | N = TB 位數 |

### 記憶體使用

```cpp
GOLD_1_SEQ_LUT_SIZE = 117504 words = 470 KB (固定，GPU 常數記憶體)
GOLD_2_32_P_LUT_SIZE = 117504 words = 470 KB (固定，GPU 常數記憶體)
```

### 輸送量 (H100 GPU)

| 操作 | 輸送量 | 延遲 |
|------|--------|------|
| Descramble 1 TB (117K bits) | 9.6 ms / 8 TB | ~1.2 µs/bit |
| Gold32 生成 | 30+ Gbit/s | - |

---

## MATLAB 參考實現

### Gold 序列生成

```matlab
function c = build_Gold_sequence(c_init, N)
    % Fibonacci LFSR1: x^31 + x^3 + 1
    x1 = zeros(N, 1);
    x1(1) = 1;
    
    % Fibonacci LFSR2: x^31 + x^3 + x^2 + x + 1
    x2 = flip(dec2bin(c_init, N) - '0');
    
    Nc = 1600;  % Skip first Nc bits per TS 38.211
    
    for n = 1:(N + Nc - 31)
        x1(n + 31) = mod(x1(n + 3) + x1(n), 2);
        x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n), 2);
    end
    
    c = zeros(N, 1);
    for n = 1:N
        c(n) = mod(x1(n + Nc) + x2(n + Nc), 2);
    end
end
```

### Descrambling

```matlab
function llr_out = apply_descrambling(llr_in, c_init)
    % 生成 Gold 序列
    c = build_Gold_sequence(c_init, length(llr_in));
    
    % 應用 XOR 到 LLR (符號翻轉)
    % LLR_descr = (1 - 2*c) * LLR_in
    llr_out = (1 - 2*c) .* llr_in;
end
```

---

## 測試與驗證

### 單元測試範例

```cpp
// testDescrambling.cpp
TEST(DESCRAMBLE, FIBONACCI_LFSR1_OPT) 
{
    for (uint64_t i = 0; i <= 0xFFFFFFFF; i++) {
        uint32_t init_state = i & 0xFFFFFFFF;
        uint32_t ref_state = init_state, test_state = init_state;
        const uint32_t ref = fibonacciLFSR1(ref_state, 32);
        const uint32_t test = fibonacciLFSR1_n32(test_state);
        EXPECT_EQ(ref, test);
        EXPECT_EQ(ref_state, test_state);
    }
}

TEST(DESCRAMBLE, FIBONACCI_LFSR2_OPT) 
{
    for (uint64_t i = 0; i <= 0xFFFFFFFF; i++) {
        uint32_t init_state = i & 0xFFFFFFFF;
        const uint32_t ref = fibonacciLFSR2(init_state, 32);
        const uint32_t test = fibonacciLFSR2_n32(init_state);
        EXPECT_EQ(ref, test);
    }
}
```

---

## 使用範例

### 完整 Descrambling 流程

```cpp
// 1. 初始化
void* descrambleEnv = nullptr;
cuphyDescrambleInit(&descrambleEnv);

// 2. 準備 TB 邊界和 c_init 值
uint32_t nTBs = 4;
uint32_t tbBoundaryArray[5] = {0, 29376, 58752, 88128, 117504};
uint32_t cinitArray[4];

for(int i = 0; i < nTBs; i++) {
    uint16_t n_rnti = 0x1234;
    uint8_t N_id = 42;
    cinitArray[i] = (n_rnti * 32768 + N_id) & 0x7FFFFFFF;
}

// 3. 載入參數
cuphyDescrambleLoadParams(&descrambleEnv, nTBs, tbBoundaryArray, cinitArray, 
                          117504 / 256, 0);

// 4. 載入 LLR 輸入 (Host)
float* h_llrs = new float[117504];
// ... 填充 h_llrs 資料

float* d_llrs = nullptr;
cudaMalloc(&d_llrs, sizeof(float) * 117504);
cudaMemcpy(d_llrs, h_llrs, sizeof(float) * 117504, cudaMemcpyHostToDevice);

cuphyDescrambleLoadInput(&descrambleEnv, d_llrs);

// 5. 執行 Descrambling
cuphyStatus_t status = cuphyDescramble(&descrambleEnv, nullptr, false, 1, nullptr);

// 6. 儲存結果
cuphyDescrambleStoreOutput(&descrambleEnv, h_llrs);

// 7. 清理
cudaFree(d_llrs);
delete[] h_llrs;
cuphyDescrambleCleanUp(&descrambleEnv);
```

### 直接調用金牌 32 函數

```cpp
// GPU 端使用
__global__ void myKernel(float* llrs, uint32_t N, uint32_t c_init)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int i = tid; i < N; i += blockDim.x * gridDim.x) {
        int bit_idx = i;
        int word_idx = bit_idx / 32;
        int bit_offset = bit_idx % 32;
        
        // 每個 32 位區間取樣一次
        if(bit_offset == 0) {
            uint32_t gold = descrambling::gold32(c_init, word_idx * 32);
        }
    }
}
```

---

## 最佳化建議

### 記憶體配置
- GOLD LUT 放在 GPU 常數記憶體或只讀快取
- 使用 `cudaMemcpyToSymbol()` 初始化 LUT

### 核心配置
```cpp
dim3 blockSize(256);           // 8 warps
dim3 gridSize((N + 255) / 256);
descrambleKernel<<<gridSize, blockSize>>>(d_llrs, N, d_tbBoundary, d_cinit);
```

### 精度選擇
- **FP32**: 標準 LLR (推薦)
- **FP16**: 低精度需求，節省記憶體頻寬
- **INT8**: 量化 LLR，需預先正規化

---

## 故障排除

| 問題 | 原因 | 解決方案 |
|------|------|---------|
| 解擾結果不正確 | c_init 計算錯誤 | 驗證 RNTI/N_ID/slot 計算 |
| 記憶體錯誤 | LUT 未正確初始化 | 檢查 `genGoldSeqLFSR1()` 執行 |
| 性能低落 | 非對齐訪問 | 確保 TB 邊界對齐到 32 位 |
| CUDA 錯誤 | 核心啟動失敗 | 檢查 block/grid 大小合法性 |

---

## 相關標準參考

- **3GPP TS 38.211 Section 5.2.1**: Gold 序列定義和初始化
- **3GPP TS 38.213 Section 8.2**: Descrambling 應用規則
- **NVIDIA cuPHY**: cuphy/descrambling/ 目錄
