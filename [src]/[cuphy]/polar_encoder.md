# NVIDIA cuPHY Polar Encoder

**File Location**: `cuPHY/src/cuphy/polar_encoder/`
**Main Files**: `polar_encoder.hpp`, `polar_encoder.cu`, `polar_encoder.cuh`
**Purpose**: High-performance CUDA kernels for encoding polar codes in 5G NR uplink control information (UCI) transmissions and downlink PDCCH/PDSCH signaling

---

## 概述 (Overview)

Polar Encoder 組件負責在 5G NR 物理層上行和下行傳輸中編碼極碼。這包括：

- **UCI on PUSCH**: HARQ-ACK、CSI-Part 1/2、SR (Scheduling Request)
- **PDCCH**: 物理下行控制信道
- **SSB PBCH**: 同步信號塊中的 PBCH (物理廣播信道)

該實現支持：

- **Polar 編碼** (Kronecker 乘積構造)
- **CRC 編碼** (24 位或 11 位)
- **速率匹配** (Repetition、Shortening、Puncturing)
- **比特交織** (Information bits 和 Coded bits)
- **多個數據塊** (DCI、SSB、UCI 段)
- **靈活的發送比特長度**

---

## 極碼編碼基礎

### 編碼過程

```
信息比特: u (K 位)
    ↓
CRC 編碼: u || CRC (K_cw 位，包括 CRC)
    ↓
凍結比特設置: 在特定位置設置凍結比特 = 0
    ↓
極碼編碼 (Kronecker 乘積): v = (u || frozen) · G_N (N 位)
    ↓
速率匹配 (Puncturing/Repetition): 恢復到 E 位 (傳送長度)
    ↓
比特交織: 最終發送比特
    ↓
傳送比特: d (E 位)
```

### 核心參數

```cpp
N       = 極碼長度 (2^n，其中 n = 5-10，即 32-1024)
K       = 信息位數 (含 CRC)
E       = 發送比特數 (速率匹配後)
nCrcBits = CRC 位數 (24 或 11)
K_cw    = 碼字信息位數 = K + nCrcBits
```

---

## 核心數據結構

### `cuphyPolarCbPrm_t` - 碼塊參數

```cpp
struct cuphyPolarCbPrm_t {
    uint32_t N_cw;              // 碼字長度
    uint16_t n_cw;              // log2(N_cw)
    uint32_t K_cw;              // 信息位數 (含 CRC)
};
```

### 極碼編碼信息表

#### 1. POLAR_ENC_INFO_BIT_INTERLEAVER_IDX[] - 信息比特交織表

```
大小: 最多 512 個索引

作用: 重新排列信息比特位置以改善性能
機制: 
  - 優先排列位置: 
    * 位置 0, 2, 4, 7, 9, 14, 19, 20, ...
  - 用於信息比特選擇
  - 提高編碼性能 (特別是在 SC 解碼中)
```

#### 2. POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[] - 編碼比特交織表

```
大小: N (最多 1024)

作用: 編碼比特的最終交織
機制:
  - 映射編碼比特到傳送位置
  - 用於速率匹配和調制編碼適配
```

#### 3. POLAR_DEPTH_TABLE[][] - 凍結比特設置表

```
結構: [N_row][32] 的表格
  - 行: 不同的凍結比特集合
  - 列: 凍結比特位置

用途: 快速查詢凍結比特位置
索引計算: 基於 K_cw 和 N_cw 的比值
```

---

## 編碼演算法

### 步驟 1: CRC 編碼

```cpp
// 信息比特: u[0..K-1]
// CRC 多項式: TS 38.212 定義

crc_bits = compute_crc(u, K);
// crc_bits 是 24 位或 11 位

u_crc[0..K-1] = u;                    // 複製信息比特
u_crc[K..K+nCrcBits-1] = crc_bits;   // 附加 CRC
```

### 步驟 2: 凍結比特設置

```cpp
// 根據碼字長度 N_cw 和信息位數 K_cw 查表

frozen_bits = POLAR_DEPTH_TABLE[table_idx];  // 凍結比特位置

c[0..N_cw-1] = unknown;  // 初始化

// 設置信息比特位置
for (int i = 0; i < K_cw; i++) {
    info_bit_pos = info_bit_interleaver[i];
    c[info_bit_pos] = u_crc[i];
}

// 設置凍結比特
for (int i = 0; i < (N_cw - K_cw); i++) {
    frozen_bit_pos = frozen_bits[i];
    c[frozen_bit_pos] = 0;  // 凍結比特 = 0
}
```

### 步驟 3: 極碼樹編碼 (Kronecker 乘積)

```cpp
// 實現 d = c · G_N，其中 G_N 是生成矩陣

__device__ void encode(uint32_t nInfoBits, 
                       uint32_t nCodedBits, 
                       uint32_t nTxBits,
                       uint8_t const* pInfoBits, 
                       uint32_t* pSmem, 
                       uint8_t* pCodedBits, 
                       uint32_t procModeBmsk)
{
    // 1. 準備凍結比特和信息比特
    //    - 複製信息比特到適當位置
    //    - 設置凍結比特 = 0
    
    // 2. 執行樹編碼
    //    使用共享內存進行級聯 (butterfly) 操作
    
    for (int stage = 0; stage < n; stage++) {
        // Butterfly 操作
        //   input[2i]   ----\
        //                     +---> output[i]   (XOR)
        //   input[2i+1] ----/
        
        // 執行 XOR 操作
        // 類似於 FFT 的樹狀計算結構
    }
    
    // 3. 輸出編碼比特到全局內存
    // pCodedBits[0..nCodedBits-1] = 編碼比特
}
```

**樹編碼結構** (Butterfly Network):

```
層級 0:    層級 1:    層級 2:
u[0]─┐      ┌───┐     ┌───┐
     ├─XOR─┤   ├─XOR─┤   │
u[1]─┘      └───┘     └───┘
u[2]─┐      ┌───┐     ┌───┐
     ├─XOR─┤   ├─XOR─┤   │
u[3]─┘      └───┘     └───┘
...
```

**時間複雜度**: $O(N \log N)$

### 步驟 4: 速率匹配

```cpp
// 從編碼比特 (N 位) 映射到傳送比特 (E 位)

void polarRateMatch(uint32_t nCodedBits,      // N
                    uint32_t nTxBits,         // E
                    uint8_t* pCodedBits,      // 輸入: 編碼比特
                    uint8_t* pTxBits)         // 輸出: 發送比特
{
    if (E < N) {
        // 速率匹配: 縮短或打孔
        // 選擇 E 個比特進行發送
        
        // 方法 1: 打孔 (Puncturing)
        // 從末尾移除比特
        for (int i = 0; i < E; i++) {
            pTxBits[i] = pCodedBits[i];  // 發送前 E 個
        }
        
        // 方法 2: 特定位置打孔 (依據 3GPP)
        // 使用打孔圖案
    } else if (E > N) {
        // 重複 (Repetition)
        for (int i = 0; i < E; i++) {
            pTxBits[i] = pCodedBits[i % N];  // 重複發送
        }
    } else {
        // E == N: 無需速率匹配
        memcpy(pTxBits, pCodedBits, N);
    }
}
```

### 步驟 5: 比特交織

```cpp
// 應用編碼比特交織表

void bitInterleaving(uint32_t nTxBits,
                     uint8_t* pTxBits)
{
    // 讀取交織表
    uint8_t perm[nTxBits];
    for (int i = 0; i < nTxBits; i++) {
        perm[POLAR_ENC_CODED_BIT_INTERLEAVER_IDX[i]] = pTxBits[i];
    }
    
    // 寫回
    memcpy(pTxBits, perm, nTxBits);
}
```

---

## 主要 CUDA 內核

### 1. `encodeRateMatchKernel()` - 極碼編碼和速率匹配

```cpp
__global__ void encodeRateMatchKernel(
    uint32_t nInfoBits,         // 信息比特數
    uint32_t nCodedBits,        // 編碼比特數
    uint32_t nTxBits,           // 發送比特數
    uint8_t const* pInfoBits,   // 輸入: 信息比特 (GPU)
    uint32_t* pCodedBits,       // 輸出: 編碼比特 (可選，用於調試)
    uint8_t* pTxBits,           // 輸出: 發送比特 (GPU)
    uint32_t procModeBmsk       // 處理模式位掩碼
)
```

**執行模型**:
- gridDim = (1, 1, 1)
- blockDim = (N_THRDS_PER_TILE, N_MAX_THRD_TILES, 1)
  * N_THRDS_PER_TILE = 32 (threads per CUDA warp)
  * N_MAX_THRD_TILES = 32 (max tile groups per block)
- 共 32×32 = 1024 threads per block

**流程**:
```
1. 加載信息比特到共享內存
2. 執行編碼:
   - CRC 計算
   - 凍結比特設置
   - Butterfly 樹編碼
3. 速率匹配:
   - 打孔或重複
4. 比特交織
5. 寫入發送比特到全局內存
```

### 2. `encodeRateMatchMultipleDCIsKernel()` - 多個 DCI 編碼

```cpp
__global__ void encodeRateMatchMultipleDCIsKernel(
    uint32_t num_DCIs,          // DCI 個數
    cuphyPolEncDCIPrm_t* pDCIPrms,  // DCI 參數陣列
    uint8_t const* pDCIBits,    // 所有 DCI 比特 (連續存儲)
    uint8_t* pEncodedDCIBits    // 編碼輸出
)
```

**特點**:
- gridDim.x = num_DCIs (每個 block 處理一個 DCI)
- 支持可變長度 DCI
- 支持不同長度的編碼和速率匹配

### 3. `encodeRateMatchMultipleSSBsKernel()` - 多個 SSB 編碼

```cpp
__global__ void encodeRateMatchMultipleSSBsKernel(
    uint16_t num_SSBs,          // SSB 個數
    cuphyPolEncSSBPrm_t* pSSBPrms,  // SSB 參數陣列
    uint8_t const* pSSBBits,    // SSB 比特
    uint8_t* pEncodedSSBBits    // 編碼輸出
)
```

---

## 優化技巧

### 1. 共享內存優化

```cpp
// 共享內存分配 (per block)
__shared__ uint32_t shMemBuf[SHARED_MEM_SIZE];

// 用途:
// - 存儲信息比特
// - 編碼過程中的中間結果
// - Butterfly 樹的工作空間

// 優勢:
// - 快速訪問 (vs 全局內存)
// - 支持 block 內的線程協作
```

### 2. Warp 級合作

```cpp
// 使用 Cooperative Groups 實現高效同步

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
auto g = cg::this_thread_block();

// Warp 級同步
g.sync();

// Tile 級同步 (多個 warps)
auto tile = cg::tiled_partition<32>(g);
```

### 3. 比特級操作優化

```cpp
// 批量處理比特
// 使用 uint32_t 每次處理 32 位

__device__ __forceinline__ uint32_t butterflyCombine(
    uint32_t x0, uint32_t x1)
{
    // 執行 XOR 操作 (32 比特並行)
    return x0 ^ x1;
}
```

### 4. 內存訪問模式

```
線程訪問模式 (Coalesced):
┌─────────────────────────────────────┐
│ Thread 0 → Memory[0]                │
│ Thread 1 → Memory[1]                │
│ Thread 2 → Memory[2]                │
│ ...                                 │
│ Thread 31 → Memory[31]              │
└─────────────────────────────────────┘
單次 128B 內存事務

相比之下，非對齊訪問會導致多次事務。
```

---

## 數據結構詳解

### 凍結比特設置表

**結構** (POLAR_DEPTH_TABLE):

```cpp
static __device__ __constant__ int8_t POLAR_DEPTH_TABLE[][32] = 
{
    // 表 0: N_cw = 32, K_cw = 17
    { -1,  -1,  -1,   1,  5,  -1,  -1,  -1,  ... },
    
    // 表 1: N_cw = 32, K_cw = 18
    { -1,  -1,  -1,   3,  5,  -1,  -1,  -1,  ... },
    
    // ...
    
    // 表 N-1: N_cw = 1024, K_cw = ...
    { -1,  -1,  -1,  30, 28,  -1,  -1,  -1,  ... },
};
```

**元素含義**:
- -1: 無效位置或佔位符
- 非負整數: 凍結比特在編碼序列中的位置

**查詢方法**:

```cpp
// 根據 K_cw 和 N_cw 計算表索引
int table_idx = compute_table_index(K_cw, N_cw);
// 從表中讀取凍結比特位置
```

---

## 設置函數

### `cuphyPolarEncRateMatch()`

執行單個極碼編碼和速率匹配

```cpp
cuphyStatus_t CUPHYWINAPI cuphyPolarEncRateMatch(
    uint32_t                nInfoBits,      // 信息比特數
    uint32_t                nTxBits,        // 發送比特數
    uint8_t const*          pInfoBits,      // 輸入信息比特 (GPU/主機)
    uint8_t*                pCodedBits,     // 輸出編碼比特 (可選)
    uint8_t*                pTxBits,        // 輸出發送比特 (GPU)
    uint32_t                procModeBmsk,   // 處理模式
    cudaStream_t            strm            // CUDA 流
)
```

**流程**:
1. 確定 N_cw (編碼長度)
2. 選擇內核 (基於 procModeBmsk)
3. 啟動內核進行編碼和速率匹配
4. 返回狀態

### `kernelSelectEncodeRateMatchMultiDCIs()`

為多 DCI 編碼配置內核

```cpp
cuphyStatus_t CUPHYWINAPI kernelSelectEncodeRateMatchMultiDCIs(
    cuphyEncoderRateMatchMultiDCILaunchCfg_t* pLaunchCfg,
    uint32_t num_DCIs
)
```

**配置**:
```cpp
gridDim.x = num_DCIs;
blockDim = (N_THRDS_PER_TILE, N_MAX_THRD_TILES);
```

### `kernelSelectEncodeRateMatchMultiSSBs()`

為多 SSB 編碼配置內核

```cpp
cuphyStatus_t CUPHYWINAPI kernelSelectEncodeRateMatchMultiSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pLaunchCfg,
    uint16_t num_SSBs
)
```

---

## 3GPP 標準映射

| 功能 | 標準 | 說明 |
|------|------|------|
| 極碼結構 | TS 38.212 § 5.3.1 | 極碼編碼定義 |
| 凍結比特 | TS 38.212 Table 5.3.1.1 | 凍結比特位置 |
| CRC | TS 38.212 § 5.1 | CRC 編碼多項式 |
| UCI 極碼 | TS 38.212 § 5.3 | UCI 編碼參數 |
| PDCCH 極碼 | TS 38.212 § 7.3 | PDCCH 編碼 |
| 速率匹配 | TS 38.212 § 5.4.1 | 匹配和交織 |

---

## 使用示例

### 基本編碼

```cpp
// 參數設置
uint32_t nInfoBits = 100;       // 100 信息比特
uint32_t nTxBits = 256;         // 256 發送比特

// 分配緩衝區 (GPU)
uint8_t* pInfoBits_d;
uint8_t* pTxBits_d;

cudaMalloc(&pInfoBits_d, (nInfoBits + 7) / 8);
cudaMalloc(&pTxBits_d, (nTxBits + 7) / 8);

// 複製輸入
uint8_t* pInfoBits_h = new uint8_t[(nInfoBits + 7) / 8];
// ... 填充 pInfoBits_h ...
cudaMemcpyAsync(pInfoBits_d, pInfoBits_h, (nInfoBits + 7) / 8, 
                cudaMemcpyHostToDevice, stream);

// 執行編碼
cuphyStatus_t status = cuphyPolarEncRateMatch(
    nInfoBits,
    nTxBits,
    pInfoBits_d,
    nullptr,        // 不輸出編碼比特
    pTxBits_d,
    0,              // 處理模式
    stream
);

// 複製結果
uint8_t* pTxBits_h = new uint8_t[(nTxBits + 7) / 8];
cudaMemcpyAsync(pTxBits_h, pTxBits_d, (nTxBits + 7) / 8,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

// ... 使用 pTxBits_h ...

// 清理
cudaFree(pInfoBits_d);
cudaFree(pTxBits_d);
delete[] pInfoBits_h;
delete[] pTxBits_h;
```

### 多 DCI 編碼

```cpp
// 準備多個 DCI
std::vector<cuphyPolEncDCIPrm_t> dciPrms(num_DCIs);
std::vector<uint8_t*> dciBits_d(num_DCIs);
std::vector<uint8_t*> encodedBits_d(num_DCIs);

// 配置每個 DCI
for (int i = 0; i < num_DCIs; i++) {
    dciPrms[i].nInfoBits = 40;      // 每個 DCI 40 信息比特
    dciPrms[i].nTxBits = 128;       // 128 發送比特
    
    // 分配緩衝區
    cudaMalloc(&dciBits_d[i], 5);
    cudaMalloc(&encodedBits_d[i], 16);
}

// 複製參數到 GPU
cuphyPolEncDCIPrm_t* dciPrms_d;
cudaMalloc(&dciPrms_d, num_DCIs * sizeof(cuphyPolEncDCIPrm_t));
cudaMemcpyAsync(dciPrms_d, dciPrms.data(), 
                num_DCIs * sizeof(cuphyPolEncDCIPrm_t),
                cudaMemcpyHostToDevice, stream);

// 設置內核
cuphyEncoderRateMatchMultiDCILaunchCfg_t launchCfg;
kernelSelectEncodeRateMatchMultiDCIs(&launchCfg, num_DCIs);

// 啟動內核
// ... 使用 cuLaunchKernel() 或 CUDA 圖
```

---

## 性能分析

### 計算複雜度

```
時間複雜度: O(N log N)
  - N = 編碼長度
  - Butterfly 樹有 log N 級
  - 每級 N 個操作

空間複雜度: O(N)
  - 共享內存: ~4KB 到 ~16KB
  - 取決於 N 和線程配置
```

### 吞吐量

```
單個編碼:
  N = 1024 bits, E = 256 bits
  時間: ~1-5 μs (依賴於 GPU)
  
批量編碼 (多 DCI/SSB):
  100 個 DCI: ~50-200 μs (完全管道化)
  
```

### 內存帶寬

```
讀: 輸入信息比特
    E = 256 bits → 32 bytes (最多)
    
寫: 輸出發送比特
    E = 256 bits → 32 bytes (最多)
    
共享: 中間結果
    N = 1024 bits → 128 bytes
```

---

## 調試和驗證

### 啟用調試輸出

```cpp
// 在 polar_encoder.cu 中
#define ENABLE_DEBUG

// 編譯時輸出調試信息
// printf() 語句將在內核執行中打印
```

### 與 MATLAB 參考對比

```
1. 準備測試向量 (MATLAB):
   - 隨機信息比特
   - 已知編碼輸出
   
2. 運行 GPU 編碼器:
   - 加載信息比特
   - 執行編碼
   - 複製結果
   
3. 比對:
   - 編碼比特
   - 發送比特
   - 中間 CRC 值
```

### 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|--------|
| 編碼輸出不正確 | K_cw 或 N_cw 參數錯誤 | 驗證參數查表 |
| 內存不對齐 | 輸入/輸出緩衝區未對齐 | 使用 32 位對齊 |
| 性能低下 | 內存訪問未合併 | 檢查訪問模式 |
| 內核超時 | 塊大小過大 | 減小 blockDim |

---

## 整體數據流

```
UCI 信息 (主機)
    ↓
分段、添加 CRC (主機或 GPU)
    ↓ [複製到 GPU]
    ↓ (GPU 內存中)
┌──────────────────────────────┐
│ Polar Encoder Kernel         │
├──────────────────────────────┤
│ 1. 凍結比特設置              │
│    ├─ 查表: K_cw, N_cw       │
│    ├─ 設置信息比特位置      │
│    └─ 設置凍結比特 = 0      │
│                              │
│ 2. Butterfly 樹編碼         │
│    ├─ Stage 0-log(N)         │
│    ├─ 級聯 XOR 操作         │
│    └─ 輸出: 編碼比特        │
│                              │
│ 3. 速率匹配                 │
│    ├─ 計算打孔/重複位置     │
│    └─ 輸出: 匹配比特        │
│                              │
│ 4. 比特交織                 │
│    ├─ 應用交織表            │
│    └─ 輸出: 發送比特        │
└──────────────────────────────┘
    ↓ (GPU 內存中的編碼比特)
    ↓ [複製回主機]
    ↓
調制、波束賦型、發送
```

---

## 最佳實踐

### 1. 內存管理

```cpp
// 使用固定內存進行主機-設備傳輸
cudaHostAlloc(&hostBuf, size, cudaHostAllocDefault);

// 異步複製
cudaMemcpyAsync(d_buf, h_buf, size, 
                cudaMemcpyHostToDevice, stream);

// 流同步而不是全局同步
cudaStreamSynchronize(stream);  // 優於 cudaDeviceSynchronize()
```

### 2. 內核啟動

```cpp
// 最大化佔用率
// blockDim = (32, 32) → 1024 threads
// gridDim = (1, 1) 或更多 (取決於數據)

// 啟用 L1 緩存 (針對共享內存優化)
cudaFuncSetAttribute(encodeRateMatchKernel,
                    cudaFuncAttributePreferredSharedMemoryCardinality,
                    cudaSharedMemBankSizeFourByte);
```

### 3. 流水線化

```cpp
// 將多個編碼操作流水線化
for (int i = 0; i < num_batches; i++) {
    // 複製輸入 (流 0)
    cudaMemcpyAsync(d_inBuf_i, h_inBuf_i, size, 
                    cudaMemcpyHostToDevice, stream_0);
    
    // 編碼 (流 1，使用先前批次的數據)
    launchKernel(..., stream_1);
    
    // 複製輸出 (流 2)
    cudaMemcpyAsync(h_outBuf_i, d_outBuf_i, size,
                    cudaMemcpyDeviceToHost, stream_2);
}
```

### 4. 監控性能

```cpp
// 使用 CUDA 事件測量時間
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
launchKernel(...);
cudaEventRecord(stop, stream);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Kernel time: %.2f ms\n", milliseconds);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

---

## 總結

NVIDIA cuPHY Polar Encoder 提供了高效、可擴展的極碼編碼實現，適用於 5G NR 下行和上行控制信息。通過優化的內核設計、共享內存管理和批量處理支持，它實現了出色的吞吐量和延遲性能。該組件是 5G 物理層發送管道的關鍵部分。
