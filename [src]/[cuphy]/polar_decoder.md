# NVIDIA cuPHY Polar Decoder

**File Location**: `cuPHY/src/cuphy/polar_decoder/`
**Main Files**: `polar_decoder.hpp`, `polar_decoder.cu`
**Purpose**: High-performance CUDA kernels for decoding polar codes (Successive Cancellation List decoder with CRC-aided pruning) in 5G NR uplink PUSCH UCI (Uplink Control Information) reception

---

## 概述 (Overview)

Polar Decoder 組件負責在 5G NR 物理層上行傳輸中解碼極碼編碼的控制信息。這是通過 Successive Cancellation List (SCL) 解碼器實現的，具有 CRC 輔助決策和動態列表大小自適應。

該實現支持：

- **Successive Cancellation (SC) 和 Successive Cancellation List (SCL)** 解碼
- **CRC 輔助決策** (CA-SCL: CRC-Aided SCL)
- **動態列表大小** (1, 2, 4, 8)
- **Fast-SSC (Simplified SC)** 優化的樹結構
- **UCI on PUSCH** (HARQ-ACK、CSI、SR 等)
- **多個 UCI 段和碼字** (多個 UE、多個層)

---

## 極碼基礎

### 極碼結構 (Polar Code Structure)

5G NR 極碼基於 **Kronecker 乘積** 的遞歸構造：

```
Generator Matrix: G_N = (1, 0) ⊗ G_{N/2}
                       (1, 1)

最小規模: G_2 = [1 0]
              [1 1]

完整矩陣: G_N = G_2 ⊗ G_2 ⊗ ... ⊗ G_2 (n 次)
其中 N = 2^n
```

### 編碼過程

```
信息比特: u (K 位)
冗餘比特: (N - K) 位
編碼輸出: d = u · G_N (N 位)
```

### 解碼樹結構

極碼解碼器使用遞歸樹結構，從根到葉：

```
         D (N bits) - Root
        /           \
    D_0 (N/2)     D_1 (N/2)  - Level n-1
    /    \         /     \
 ...    ...      ...     ...   - Level n-2
 |       |        |      |
 u_0   u_1  ...  u_{N-2} u_{N-1}  - Leaf (信息比特)
```

**樹深度**: $\log_2(N)$
- N = 512 → depth = 9
- N = 1024 → depth = 10

### CRC (Cyclic Redundancy Check)

UCI 極碼使用 CRC 進行錯誤檢測和 SCL 列表剪枝：

```
信息位: K 位
├─ 實際數據: K - nCrcBits 位
└─ CRC: nCrcBits 位

在極碼編碼前進行 CRC 編碼
在解碼時用於檢驗和列表剪枝
```

---

## 核心數據結構

### `cuphyPolarUciSegPrm_t` - UCI 段參數

```cpp
struct cuphyPolarUciSegPrm_t {
    uint32_t K_cw;                  // 碼字信息位數 (含 CRC)
    uint32_t N_cw;                  // 碼字長度
    uint16_t nCbs;                  // 該段中的碼字數 (1 or 2)
    uint16_t nCrcBits;              // CRC 比特數
    uint8_t  cbIdxWithinUciSeg;     // 碼字在段內索引 (0 or 1)
    uint8_t  zeroInsertFlag;        // 是否在第一個碼字起始處插入 0
    uint32_t* pUciSegEst;           // 估計的 UCI 段指針 (GPU)
};
typedef struct cuphyPolarUciSegPrm_t cuphyPolarUciSegPrm_t;
```

### `cuphyPolarCwPrm_t` - 碼字參數

```cpp
struct cuphyPolarCwPrm_t {
    uint32_t N_cw;                  // 碼字長度 (64-1024)
    uint32_t K_cw;                  // 信息位數 (含 CRC)
    uint32_t A_cw;                  // 有效數據位 = K_cw - nCrcBits
    uint16_t nCrcBits;              // CRC 比特數 (0 or 24)
    uint8_t  cbIdxWithinUciSeg;     // 碼字在 UCI 段內的索引
    uint8_t  zeroInsertFlag;        // 零插入標誌
    uint32_t* pCwTreeTypes;         // 樹結點類型數組 (GPU 指針)
    uint32_t* pCbEst;               // 估計碼字 (硬決策) (GPU 指針)
    uint8_t*  pCrcStatus;           // CRC 檢驗結果 (GPU 指針)
    uint8_t   exitFlag;             // 退出標誌 (提前終止)
};
typedef struct cuphyPolarCwPrm_t cuphyPolarCwPrm_t;
```

### `polarDecoderDynDescr_t` - 動態描述符

```cpp
struct polarDecoderDynDescr {
    __half**           cwTreeLLRsAddrs;       // 碼字樹 LLR 地址陣列 (GPU 指針)
    cuphyPolarCwPrm_t* pCwPrmsGpu;            // 碼字參數 (GPU)
    uint32_t**         polCbEstAddrs;         // 估計碼字地址陣列 (GPU)
    bool**             listPolScratchAddrs;   // 列表解碼臨時緩衝區地址 (GPU)
    uint8_t*           pPolCrcErrorFlags;     // CRC 錯誤標誌 (GPU)
};
typedef struct polarDecoderDynDescr polarDecoderDynDescr_t;
```

### 樹結點類型 (`POLAR_TREE_TYPES`)

Fast-SSC 優化中使用的結點類型：

```
Type 0: 通用結點 (需要遞歸計算)
Type 1: R0 結點 (右子樹全為凍結比特)
        → 全部冷凍，無需計算
Type 2: R1 結點 (右子樹全為信息比特，左子樹全為凍結比特)
        → 直接決策
Type 3: R2 結點 (兩個子節點都是 R0 或 R1)
        → 快速計算
```

---

## 演算法: Successive Cancellation List (SCL) 解碼

### 步驟 1: 初始化

```cpp
// 初始化列表
num_paths = 1;  // 開始時只有 1 條路徑

for (int bit_idx = 0; bit_idx < N; bit_idx++) {
    pathPrime[0] = 0;  // 初始路徑
    pathMetrics[0] = 0;  // 初始路徑度量 = 0
}
```

### 步驟 2: 自下而上遍歷 (Bottom-up)

對於每個比特位置 (從 0 到 N-1)：

```
如果 bit_idx 是凍結比特:
    1. 從右子樹讀取估計: u[bit_idx] = 0 (固定值)
    2. 更新路徑度量: pathMetrics[p] += log(1 + exp(-u[bit_idx] * LLR))

否則 (信息比特):
    1. 計算 LLR: llr_bit = LLR[bit_idx]
    2. 分支測試:
        對於每條現有路徑 p:
            a) 測試 u = 0: llr_0 = LLR[bit_idx]
            b) 測試 u = 1: llr_1 = LLR[bit_idx] - Inf (等於反轉 LLR)
            c) 更新路徑度量
    
    3. 列表管理:
        - 合併所有候選 (現有 num_paths × 2)
        - 按路徑度量排序
        - 保留頂部 list_size 條路徑
        - 更新 num_paths
```

### 步驟 3: 自上而下遍歷 (Top-down)

完成 SC 過程後，對每條保留的路徑：

```
對於每個比特位置 (從 0 到 N-1):
    計算該位的硬決策: 
    d[i] = 決策(LLR[i], 父節點決策)
```

### 步驟 4: CRC 檢驗和決策

```cpp
for (int path = 0; path < num_paths; path++) {
    // 提取信息位 (移除 CRC 和冗餘比特)
    extract_info_bits(decoded[path], codeword[path]);
    
    // 計算 CRC
    crc_computed = compute_crc(info_bits, K - nCrcBits);
    crc_received = info_bits[K - nCrcBits : K];
    
    // 檢驗
    if (crc_computed == crc_received) {
        best_path = path;
        crc_error_flag = 0;
        break;  // 找到正確解
    }
}

// 如果所有路徑 CRC 都失敗
if (no_valid_crc_found) {
    best_path = 最小路徑度量的路徑;
    crc_error_flag = 1;
}
```

---

## 關鍵 CUDA 內核

### 1. `polarDecoderKernel()` - 單路徑解碼 (列表大小 = 1)

```cpp
__global__ void polarDecoderKernel(polarDecoderDynDescr_t* pDynDescr)
```

**目的**: 當列表大小為 1 時的高效直接 SC 解碼

**執行模型**:
- gridDim.x = 碼字數
- blockDim.x = 32
- 每個 block 對應一個碼字

**主要函數調用**:
1. `singlePolarDecoder()` - 執行完整 SC 過程
2. `updateCRCstatus()` - 檢驗 CRC 並更新狀態

### 2. `listPolarDecoderKernel<TILE_SZ, LIST_SZ>()` - 多路徑列表解碼

```cpp
template<uint32_t TILE_SZ, uint32_t LIST_SZ = 8>
__launch_bounds__(1024, 1)
static __global__ void listPolarDecoderKernel(polarDecoderDynDescr_t* pDynDescr)
```

**目的**: 處理 CRC 錯誤時的列表解碼 (回退方案)

**執行模型**:
- TILE_SZ = blockDim.x / LIST_SZ (每條路徑的線程數)
- 通常 blockDim = 32, TILE_SZ = 4 (LIST_SZ = 8)
- 或 blockDim = 32, TILE_SZ = 8-16 (LIST_SZ = 2-4)

**流程**:
```
1. 先執行單路徑解碼 (SC)
2. 檢驗 CRC
3. 如果 CRC 通過 → 返回
4. 如果 CRC 失敗 → 執行列表解碼
   a) 重新計算樹 LLR
   b) 維護多條路徑
   c) 根據 CRC 篩選路徑
```

### 3. `singlePolarDecoder()` - 核心 SC 解碼

```cpp
__device__ __forceinline__ void singlePolarDecoder(polarDecoderDynDescr_t* pDynDescr)
```

**關鍵步驟**:

1. **樹 LLR 初始化**
   ```cpp
   // 讀取接收的碼字 LLR (已進行速率匹配恢復)
   cwTreeLLR[0 : N] = 接收 LLR[0 : N];
   ```

2. **自下而上樹遍歷 (編碼樹)**
   ```cpp
   for (int stage = DEPTH - 1; stage >= 0; stage--) {
       for (int idx = 0 to N / 2^(stage+1)) {
           // 根據樹類型選擇計算:
           tree_type = get_type(stage, N, idx, treeTypes);
           
           if (tree_type == R0) {
               // 凍結: 無需計算
           } else if (tree_type == R1) {
               // 信息位: 直接決策
               estimate[idx] = (LLR[idx] < 0) ? 1 : 0;
           } else if (tree_type == R2 || GENERIC) {
               // 合併兩子樹
               // 輸入在位置 [idx*sz, idx*sz + 2*sz)
               // 輸出在位置 [idx*sz, idx*sz + sz)
               
               if (is_frozen) {
                   // F_kernel: LLR 合併 (凍結位)
                   F_kernel(out, in, sz);
               } else {
                   // 組合 F 和 G
                   F_kernel(temp, in, sz);  // 先計算 F
                   決策 = (temp[0] < 0) ? 1 : 0;
                   G_kernel(out, temp, 決策, sz);  // 再計算 G
               }
           }
       }
   }
   ```

3. **自上而下樹遍歷 (決策樹)**
   ```cpp
   for (int stage = 0; stage < DEPTH; stage++) {
       for (int idx = 0 to N / 2^stage) {
           // H_kernel: 結合父決策和子樹 LLR
           H_kernel(決策, ...);
       }
   }
   ```

4. **提取信息位**
   ```cpp
   // 移除冗餘和 CRC，保留 A_cw 有效比特
   extract_info_bits(codeword_estimate, info_bits);
   ```

### 4. F_kernel() - LLR 合併 (框-加)

```cpp
__device__ __forceinline__ void F_kernel(__half* llrOut, __half* llrIn, 
                                         int sz, const thread_group& grp)
{
    __half* a = llrIn;           // 第一個子樹 LLR
    __half* b = &llrIn[sz];      // 第二個子樹 LLR

    for (int i = grp.thread_rank(); i < sz; i += grp.size()) {
        // 框-加 (box-plus) 運算 (最小和或精確)
        __half minAbs = __hmin(__habs(a[i]), __habs(b[i]));
        __half sgn = __hsign(a[i]) * __hsign(b[i]);
        
        llrOut[i] = sgn * minAbs;  // 或使用更精確的公式
    }
}
```

**框-加運算** ($\oplus$):
```
LLR_1 ⊕ LLR_2 = 2 * tanh^{-1}(tanh(LLR_1/2) * tanh(LLR_2/2))
                = sign(LLR_1 * LLR_2) * min(|LLR_1|, |LLR_2|)  [近似，最小和]
                ≈ sign(LLR_1 * LLR_2) * min(|LLR_1|, |LLR_2|) + 
                  log(1 + exp(-|LLR_1| - |LLR_2|)) - 
                  log(1 + exp(-|LLR_1 - LLR_2|))  [精確]
```

### 5. G_kernel() - LLR 更新 (偽後驗)

```cpp
__device__ __forceinline__ void G_kernel(__half* llrOut, __half* llrIn, 
                                         bool* est, int32_t sz)
{
    __half* a = llrIn;           // 來自頂層 LLR
    __half* b = &llrIn[sz];      // 來自底層 LLR

    for (int32_t i = threadIdx.x; i < sz; i += blockDim.x) {
        __half u = static_cast<__half>(1 - 2 * est[i]);  // u = +1 or -1
        llrOut[i] = b[i] + u * a[i];  // 更新下層 LLR
    }
}
```

**G_kernel 計算** ($L_u$):
```
L_u(u | y) = (-1)^{u} * L_{top} + L_{bottom}
           = (1 - 2u) * L_{top} + L_{bottom}

其中:
- u: 硬決策 (來自上層)
- L_{top}: 來自上層的 LLR
- L_{bottom}: 來自下層的 LLR
```

### 6. H_kernel() - 硬決策合併 (比特級)

```cpp
__device__ __forceinline__ void H_kernel(uint32_t* bitsOut,
                                         const uint32_t* bitsIn0,
                                         const int in0IdxOffset,
                                         uint32_t in0ArraySz,
                                         const uint32_t* bitsIn1,
                                         uint32_t in1ArraySz,
                                         int sz,
                                         const thread_group& grp)
{
    // 按位級存儲: 每 32 個比特存儲在一個 uint32_t 中
    
    if (sz == 1) {
        // 基本情況: 1 比特
        // bitsOut[0] = bitsIn0[0] XOR bitsIn1[0]
    } else {
        // 遞歸情況: sz > 1
        // 合併上層決策和下層決策
        // bitsOut[i] = bitsIn0[i] XOR bitsIn1[i]  (逐字級異或)
    }
}
```

**H_kernel 邏輯** (比特級異或):
```
上層決策: u_0 (sz 比特)
下層決策: u_1 (sz 比特)
輸出: u (sz 比特) = u_0 XOR u_1
```

---

## 列表解碼特定優化

### 路徑管理

**路徑表示**:
```cpp
// 每條路徑存儲:
- pathMetric[p]      // 累積度量 (路徑可能性)
- pathDecision[p][*] // 位決策 (跨所有比特)
- pathLLR[p][*]      // 樹 LLR (用於下一迭代)
```

**列表管理操作**:
```cpp
// 合併和排序
std::vector<Path> candidates;
for (int p = 0; p < num_paths; p++) {
    candidates.push_back({pathMetric[p], 0, ...});      // 決策 0
    candidates.push_back({pathMetric[p] + llr_cost, 1, ...});  // 決策 1
}
std::sort(candidates.begin(), candidates.end());
num_paths = min(list_size, candidates.size());
for (int p = 0; p < num_paths; p++) {
    pathMetric[p] = candidates[p].metric;
    pathDecision[p] = candidates[p].decision;
}
```

### 共享內存優化 (列表解碼)

```cpp
// 動態共享內存分配 (每個 block)
int dyn_shared_sz = 0;
dyn_shared_sz += LIST_SZ * N_MAX_POLAR_DEPTH * sizeof(int16_t);  
    // 鏈表指針
dyn_shared_sz += LIST_SZ * sizeof(__half);  
    // 路徑度量
dyn_shared_sz += LIST_SZ * 2 * (N_MAX_WORDS + BCO) * sizeof(uint32_t);
    // 臨時比特緩衝區 (複製用)
dyn_shared_sz += LIST_SZ * (N_MAX_WORDS + BCO) * sizeof(uint32_t);
    // 估計碼字 (每階段)

// BCO = 3: 銀行衝突偏移量
```

**銀行衝突避免**:
- 為每條路徑添加 3 字填充
- 確保不同路徑訪問不同銀行
- 提高共享內存吞吐量

---

## 設置函數

### `cuphyCreatePolarDecoder()`

創建 Polar Decoder 對象

```cpp
cuphyStatus_t CUPHYWINAPI cuphyCreatePolarDecoder(
    cuphyPolarDecoderHndl_t* pPolarDecoderHndl  // 輸出: Decoder 句柄
)
```

### `cuphySetupPolarDecoder()`

配置解碼器並準備內核啟動

```cpp
cuphyStatus_t CUPHYWINAPI cuphySetupPolarDecoder(
    cuphyPolarDecoderHndl_t       polarDecoderHndl,      // Decoder 句柄
    uint16_t                      nPolCws,                // 碼字數
    __half**                      pCwTreeLLRsAddrs,       // 碼字樹 LLR 地址
    cuphyPolarCwPrm_t*            pCwPrmsGpu,             // GPU 碼字參數
    cuphyPolarCwPrm_t*            pCwPrmsCpu,             // CPU 碼字參數
    uint32_t**                    pPolCbEstAddrs,         // 估計碼字地址
    bool**                        pListPolScratchAddrs,   // 列表解碼臨時緩衝區
    uint8_t                       nPolLists,              // 列表大小 (1, 2, 4, 8)
    uint8_t*                      pPolCrcErrorFlags,      // CRC 錯誤標誌
    bool                          enableCpuToGpuDescrAsyncCpy,  // 異步複製
    polarDecoderDynDescr_t*       pCpuDynDesc,            // CPU 描述符
    void*                         pGpuDynDesc,            // GPU 描述符
    cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,             // 啟動配置
    cudaStream_t                  strm                    // 流
)
```

**內核選擇邏輯**:
```cpp
if (nPolLists == 1) {
    kernelFunc = polarDecoderKernel;  // 直接 SC
} else if (nPolLists == 2) {
    kernelFunc = listPolarDecoderKernel<16, 2>;  // 列表, tile_sz=16
} else if (nPolLists == 4) {
    kernelFunc = listPolarDecoderKernel<8, 4>;   // 列表, tile_sz=8
} else if (nPolLists == 8) {
    kernelFunc = listPolarDecoderKernel<4, 8>;   // 列表, tile_sz=4
}
```

**啟動配置**:
```cpp
gridDim.x = nPolCws;        // 每個碼字一個 block
blockDim.x = 32;            // 固定 block 大小

// 動態共享內存計算
if (nPolLists > 1) {
    shared_mem = LIST_SZ * N_MAX_POLAR_DEPTH * sizeof(int16_t) +
                 LIST_SZ * sizeof(__half) +
                 LIST_SZ * 2 * (N_MAX_WORDS + BCO) * sizeof(uint32_t) +
                 LIST_SZ * (N_MAX_WORDS + BCO) * sizeof(uint32_t);
} else {
    shared_mem = 5 * N_MAX_CODED_BITS / 2 * sizeof(__half);
}
```

### `cuphyDestroyPolarDecoder()`

銷毀 Decoder 對象

```cpp
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolarDecoder(
    cuphyPolarDecoderHndl_t polarDecoderHndl
)
```

---

## 常數 (編譯時已知)

### 極碼樹參數

```cpp
namespace polar_decoder {
    static constexpr int N_MAX_POLAR_DEPTH = 10;           // 最大樹深度
    static constexpr int N_MAX_CODED_BITS = 1024;          // 最大碼字長度 (2^10)
    static constexpr int WORD_LENGTH = 32;                 // uint32_t 位寬
    static constexpr int N_MAX_WORDS = 1024 / 32;          // 32 字
    static constexpr int BCO = 3;                          // 銀行衝突偏移量
};
```

### POLAR_DEPTH[] - 樹結點深度

設備常量，存儲每個比特對應的樹深度 (用於遍歷順序):

```cpp
__device__ __constant__ uint8_t POLAR_DEPTH[1024];
// POLAR_DEPTH[i] = 比特 i 在樹中的深度
// 例: 
// POLAR_DEPTH[0] = 10    (最深, 最後處理)
// POLAR_DEPTH[1] = 9
// POLAR_DEPTH[511] = 1
```

### 樹類型查表

```cpp
// 預計算的樹結點類型
// 根據 Fast-SSC 優化進行分類
uint8_t treeTypes[2 * N];  // 索引樹結點
```

---

## 3GPP 標準映射

實現遵循以下 3GPP 規範：

| 功能 | 標準 | 說明 |
|------|------|------|
| 極碼結構 | TS 38.212 § 5.3.1 | 極碼編碼和凍結比特設置 |
| 凍結比特 | TS 38.212 Table 5.3.1.1-1 | 凍結比特位置 |
| UCI 極碼 | TS 38.212 § 5.3 | UCI 段和碼字配置 |
| CRC | TS 38.212 § 5.1 | 24 位 CRC 多項式 |
| 速率匹配 | TS 38.212 § 5.4.1 | 匹配和交織 |
| 調制編碼 | TS 38.211 § 6.3 | QPSK、16QAM 等 |

---

## 數據流程

```
接收信號 (設備)
    ↓
解調、匹配濾波、同步
    ↓
速率匹配恢復 (恢復編碼 LLR → 碼字 LLR)
    ↓ [GPU 內存中的碼字樹 LLR]
┌─────────────────────────┐
│ Polar Decoder Kernel    │
├─────────────────────────┤
│ 1. SC 解碼 (列表大小=1) │
│    ├─ 樹遍歷            │
│    ├─ LLR 合併          │
│    ├─ 硬決策            │
│    └─ 提取信息位        │
│                         │
│ 2. CRC 檢驗             │
│    ├─ 如果 CRC OK → 輸出 │
│    └─ 如果 CRC 失敗     │
│       └─ 執行列表解碼   │
│          ├─ 多路徑 SC   │
│          ├─ 路徑管理    │
│          ├─ 列表剪枝    │
│          └─ 選擇最佳    │
└─────────────────────────┘
    ↓
估計的 UCI 比特 (設備內存)
    ↓ [複製回主機]
    ↓
后處理、解分段 (UCI 段 → 各類信息)
    ↓
HARQ-ACK、CSI、SR 決策
```

---

## 使用示例

### 基本設置和執行

```cpp
// 1. 創建 Decoder
cuphyPolarDecoderHndl_t decoderHndl;
cuphyCreatePolarDecoder(&decoderHndl);

// 2. 準備參數 (主機)
std::vector<cuphyPolarCwPrm_t> cwPrmsCpu(nPolCws);
for (int i = 0; i < nPolCws; i++) {
    cwPrmsCpu[i].N_cw = N;              // 1024
    cwPrmsCpu[i].K_cw = K;              // 例: 140
    cwPrmsCpu[i].nCrcBits = 24;
    cwPrmsCpu[i].A_cw = K - 24;
    // ... 其他參數
}

// 3. 分配 GPU 內存
std::vector<__half*> cwTreeLLRsAddrs(nPolCws);
std::vector<uint32_t*> cbEstAddrs(nPolCws);
for (int i = 0; i < nPolCws; i++) {
    cudaMalloc(&cwTreeLLRsAddrs[i], N * sizeof(__half));
    cudaMalloc(&cbEstAddrs[i], (N + 31) / 32 * sizeof(uint32_t));
}

// 4. 複製參數到 GPU
cuphyPolarCwPrm_t* cwPrmsGpu;
cudaMalloc(&cwPrmsGpu, nPolCws * sizeof(cuphyPolarCwPrm_t));
cudaMemcpyAsync(cwPrmsGpu, cwPrmsCpu.data(), 
                nPolCws * sizeof(cuphyPolarCwPrm_t),
                cudaMemcpyHostToDevice, stream);

// 5. 設置 Decoder
cuphyPolarDecoderLaunchCfg_t launchCfg;
cuphySetupPolarDecoder(decoderHndl, nPolCws, 
                      cwTreeLLRsAddrs.data(), 
                      cwPrmsGpu, cwPrmsCpu.data(),
                      cbEstAddrs.data(),
                      nullptr,  // no list scratch for list_sz=1
                      1,         // list size
                      crcErrorFlags,
                      false,     // no async copy
                      cpuDesc, gpuDesc,
                      &launchCfg, stream);

// 6. 執行內核
cuLaunchKernel(launchCfg.kernelNodeParamsDriver.func, ...);

// 7. 檢索結果
uint32_t* estimateHost = new uint32_t[N / 32];
cudaMemcpyAsync(estimateHost, cbEstAddrs[0], 
                (N + 31) / 32 * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);

// 8. 銷毀
cuphyDestroyPolarDecoder(decoderHndl);
```

### 列表解碼 (多路徑)

```cpp
// 步驟 4-5 改變: 使用列表大小 > 1

// 分配列表臨時緩衝區
std::vector<bool*> listScratchAddrs(nPolCws);
for (int i = 0; i < nPolCws; i++) {
    size_t scratchSize = sizeof(bool) * 2 * N * list_size;
    cudaMalloc(&listScratchAddrs[i], scratchSize);
}

// 設置 (列表大小 = 4)
cuphySetupPolarDecoder(decoderHndl, nPolCws, 
                      cwTreeLLRsAddrs.data(), 
                      cwPrmsGpu, cwPrmsCpu.data(),
                      cbEstAddrs.data(),
                      listScratchAddrs.data(),  // 提供臨時緩衝區
                      4,         // list size = 4
                      crcErrorFlags,
                      false,
                      cpuDesc, gpuDesc,
                      &launchCfg, stream);

// ... 其他與單路徑解碼相同
```

---

## 性能特性

### 計算複雜度

**SC 解碼 (列表大小 = 1)**:
```
時間複雜度: O(N * log(N))
  - N 個比特
  - 每個比特最多 log(N) 級別的樹遍歷

空間複雜度: O(N * log(N))
  - 樹存儲 (LLR 和決策)
```

**SCL 解碼 (列表大小 = L)**:
```
時間複雜度: O(N * log(N) * L * log(L))
  - 額外的路徑管理和排序

空間複雜度: O(N * log(N) * L)
  - L 條路徑的樹副本
```

### GPU 資源

**每個 Block**:
- 線程數: 32
- 共享內存 (列表=1): ~5KB
- 共享內存 (列表=8): ~100KB

**整體**:
- 吞吐量: N 個碼字並行 (gridDim.x = N)
- 內存帶寬: 主要受樹 LLR 訪問限制

---

## 最佳實踐和優化提示

### 1. 碼字長度選擇

- **N = 512**: 深度 9，較小的共享內存，適合高列表大小
- **N = 1024**: 深度 10，需要更多資源，但吞吐量好
- 根據 UCI 信息量和 SNR 選擇

### 2. 列表大小選擇

```cpp
列表大小 | 場景 | 優勢
---------|------|-----
1        | 高 SNR | 最快，低延遲
2        | 中 SNR | 平衡
4        | 低 SNR | 更好的覆蓋
8        | 極低 SNR | 最佳可靠性，但延遲最高
```

### 3. 批量處理

- 分組 UCI 段和碼字
- 最大化 GPU 利用率
- 考慮各個碼字的長度差異

### 4. 內存管理

- 預分配 GPU 內存 (避免頻繁重分配)
- 使用固定內存用於主機-設備數據傳輸
- 考慮使用 CUDA 圖進行多幀優化

---

## 調試和驗證

### 常見問題

| 問題 | 原因 | 解決方案 |
|------|------|--------|
| 低解碼性能 | 列表大小不合適 | 調整 list_size |
| CRC 錯誤率高 | LLR 值不准確 | 檢查速率匹配和 LLR 計算 |
| 內核超時 | 共享內存不足 | 使用較小的列表大小或 N |
| 內存訪問衝突 | 銀行衝突 | 已通過 BCO 優化 |

### 測試方法

1. **功能驗證**:
   ```cpp
   // 與 MATLAB 參考實現對比
   // 測試各種 N, K, SNR 組合
   ```

2. **性能測試**:
   ```cpp
   // 測量吞吐量、延遲、功耗
   // 比較不同列表大小
   ```

3. **覆蓋測試**:
   ```cpp
   // 遍歷所有碼字長度 (64-1024)
   // 所有列表大小 (1, 2, 4, 8)
   ```

---

## 總結

NVIDIA cuPHY Polar Decoder 是一個高度優化的 CUDA 實現，支持完整的 5G NR UCI 解碼流程。通過組合 SC/SCL 算法、CRC 輔助決策和 GPU 並行化，它實現了出色的性能和可靠性。該組件是 PUSCH 接收管道中的關鍵部分，直接影響上行鏈路吞吐量和可靠性。
