# 錯誤更正 (Error Correction - LDPC 解碼器)

## 概述

Error Correction 組件是 NVIDIA cuPHY 庫中用於**5G LDPC 編碼和解碼**的核心模塊。它實現了NR (新無線電) 標準的 LDPC 解碼器,支持多種解碼算法和優化實現,是5G接收端信號處理的關鍵模塊。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - Error Correction](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/error_correction)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `ldpc.hpp` | 解碼器類定義和API |
| `ldpc.cpp` | LDPC解碼器實現 |
| `ldpc.cuh` | CUDA輔助函數和設備代碼 |
| `ldpc2.hpp` | LDPC2 算法系列定義 |
| `ldpc2_*.cuh/.cu` | 各種算法實現 (25+文件) |
| `nrLDPC.cuh` | NR標準LDPC支持 |
| `ldpc_encode.cu` | LDPC編碼器 |

---

## 核心數據結構

### LDPC_output_t

```cpp
typedef tensor_ref_contig_2D<CUPHY_R_32U> LDPC_output_t;

// 用途: 連續2D張量 (32比特字)
// 用於存儲解碼輸出比特
// 尺寸: [num_words_per_codeword, num_codewords]
```

### cuphyLDPCDecodeConfigDesc_t

```cpp
struct cuphyLDPCDecodeConfigDesc_t {
    uint32_t        llr_type;           // LLR數據類型 (CUPHY_R_16F, CUPHY_R_32F)
    uint8_t         BG;                 // LDPC基圖 (1 或 2)
    uint8_t         Z;                  // 升縮因子
    uint16_t        num_parity_nodes;   // 校驗節點數
    uint16_t        Kb;                 // 信息節點數
    uint32_t        max_iterations;     // 最大迭代次數
    uint32_t        flags;              // 配置標誌
    int16_t         algo;               // 算法索引 (0=自動選擇)
    union {
        __half2     f16x2;              // FP16 歸一化值
        float       f32;                // FP32 歸一化值
    } norm;
    void*           workspace;          // 工作區指針
    float           clamp_value;        // LLR截幅值
};
```

### LDPC_kernel_params

```cpp
struct LDPC_kernel_params {
    const char*     input_llr;          // 輸入LLR地址
    char*           out;                // 輸出比特地址
    int             input_llr_stride_elements;  // 輸入步長 (元素)
    int             output_stride_words;        // 輸出步長 (字)
    int             max_iterations;     // 最大迭代次數
    int             outputs_per_codeword;       // 每碼字輸出字數
    word_t          norm;               // 歸一化值
    void*           workspace;          // 工作區
    int             Z;                  // 升縮因子
    int             z2, z4, z8, z16;    // Z的倍數
    int             mbz8, mbz16;        // m*Z的倍數
    int             num_parity_nodes;   // 校驗節點數
    int             num_var_nodes;      // 可變節點數
    int             K;                  // 碼字長度 (比特)
    int             Kb;                 // 信息比特數
    int             KbZ;                // Kb*Z
    int             Z_var;              // Z * num_var_nodes
    int             Z_var_szelem;       // Z_var * 元素大小
    int             num_codewords;      // 碼字數
    float           clamp_value;        // LLR截幅值
};
```

---

## 解碼器類 (ldpc::decoder)

### 初始化

```cpp
class ldpc::decoder {
public:
    explicit decoder(const cuphy_i::context& ctx);
    
    // 獲取設備信息
    int index() const;                  // 設備索引
    uint64_t compute_cap() const;       // 計算能力
    int max_shmem_per_block_optin() const;  // 最大共享內存
    int sm_count() const;               // SM數量
};
```

### 主要方法

#### decode() - 張量介面

```cpp
[[nodiscard]]
cuphyStatus_t decode(const tensor_pair& tDst,
                     const_tensor_pair& tLLR,
                     const cuphy_optional<tensor_pair>& optSoftOutputs,
                     const cuphyLDPCDecodeConfigDesc_t& config,
                     cudaStream_t strm);
```

**功能**: 使用張量描述符進行LDPC解碼

**參數**:
- `tDst`: 輸出比特 (LDPC_output_t)
- `tLLR`: 輸入LLR張量對 (host/device)
- `optSoftOutputs`: 可選軟輸出 (APP值)
- `config`: 解碼配置
- `strm`: CUDA流

**返回**: 成功/失敗狀態

#### decode_tb() - 傳輸塊介面

```cpp
[[nodiscard]]
cuphyStatus_t decode_tb(const cuphyLDPCDecodeDesc_t& decodeDesc,
                        cudaStream_t strm);
```

**功能**: 批量解碼傳輸塊

**参數**:
- `decodeDesc`: TB解碼描述符
- `strm`: CUDA流

#### workspace_size()

```cpp
[[nodiscard]]
std::pair<bool, size_t> workspace_size(
    const cuphyLDPCDecodeConfigDesc_t& config,
    int numCodeWords) const;
```

**功能**: 查詢工作區大小

**返回**: (是否成功, 大小)

#### choose_algo()

```cpp
[[nodiscard]]
int choose_algo(const cuphyLDPCDecodeConfigDesc_t& config) const;
```

**功能**: 根據配置選擇最優解碼算法

**選擇標準**:
1. 代碼長度 (Z值)
2. LLR數據類型 (FP16 vs FP32)
3. GPU計算能力
4. 吞吐量/延遲偏好

#### set_normalization()

```cpp
[[nodiscard]]
static cuphyStatus_t set_normalization(cuphyLDPCDecodeConfigDesc_t& config);
```

**功能**: 設置最小和歸一化因子

**基圖相關**: 不同基圖有不同的歸一化表

---

## 支持的LDPC算法

### 算法列表

| 算法ID | 名稱 | 特點 |
|--------|------|------|
| 26 | REG_INDEX_FP_DESC_DYN | 通用FP16解碼 (動態描述符) |
| 27 | REG_INDEX_FP_X2_DESC_DYN | 雙碼字/CTA x2版本 |
| 28 | REG_INDEX_FP_DESC_DYN_SM80 | SM80優化 |
| 29 | REG_INDEX_FP_DESC_DYN_ROW_DEP | 行依賴優化 |
| 30 | REG_INDEX_FP_DP_DESC_DYN_ROW_DEP | 雙精度版本 |
| 31 | REG_INDEX_FP_DESC_DYN_ROW_DEP_SM80 | SM80行依賴 |
| 32 | REG_INDEX_FP_DESC_DYN_ROW_DEP_SM86 | SM86優化 |
| 33 | REG_INDEX_FP_DESC_DYN_SMALL | 小Z值優化 (Z ≤ 50) |
| 34 | SHM_INDEX_FP_DESC_DYN | 共享內存優化 |
| 35 | SPLIT_INDEX_FP_X2_DESC_DYN | 分割x2算法 |
| 37 | SPLIT_INDEX_FP_X2_DESC_DYN_SM86 | SM86分割x2 |
| 38 | REG_INDEX_FP_DESC_DYN_ROW_DEP_SM90 | SM90優化 |
| 39 | SPLIT_INDEX_FP_X2_DESC_DYN_SM90 | SM90分割x2 (高共享內存) |
| 40 | REG_BOX_PLUS | Box-Plus近似算法 (Blackwell) |

### 計算能力映射

```cpp
// SM70 (Volta) / SM75 (Turing)
↓ (自動選擇) ↓
- 若 Z ≤ 50: REG_INDEX_FP_DESC_DYN_SMALL
- 若 FP32: REG_INDEX_FP_DESC_DYN
- 若 FP16 + 吞吐量: SPLIT_INDEX_FP_X2_DESC_DYN
- 若 FP16 + 延遲: REG_INDEX_FP_DESC_DYN_ROW_DEP

// SM80 (Ampere) / SM86 / SM90 (Hopper) / SM100 (Blackwell)
↓ (進行算法選擇) ↓
- 根據碼率和配置選擇x2或x1版本
- 對Blackwell: 優先使用 REG_BOX_PLUS
```

---

## LDPC基圖和配置

### 基圖(BG)參數

#### LDPC基圖1 (BG1) - 5G NR USCH和PDSCH

| 參數 | BG1 |
|------|-----|
| 信息節點 (Kb) | 22 |
| 最大校驗節點 | 46 |
| 可變節點數 | 22 + m (m = 校驗節點數) |
| 代碼率範圍 | 1/3 ~ 8/9 |

#### LDPC基圖2 (BG2) - 5G NR PDSCH

| 參數 | BG2 |
|------|-----|
| 信息節點 (Kb) | 6, 8, 9, 10 |
| 最大校驗節點 | 42 |
| 可變節點數 | 10 + m (m = 校驗節點數) |
| 代碼率範圍 | 1/5 ~ 8/9 |

### 升縮因子(Z)支持

```
Z值集合:
Set 0: a=2,  j∈[0,7]  → Z ∈ {2, 4, 8, 16, 32, 64, 128, 256}
Set 1: a=3,  j∈[0,7]  → Z ∈ {3, 6, 12, 24, 48, 96, 192, 384}
Set 2: a=5,  j∈[0,6]  → Z ∈ {5, 10, 20, 40, 80, 160, 320}
Set 3: a=7,  j∈[0,5]  → Z ∈ {7, 14, 28, 56, 112, 224}
Set 4: a=9,  j∈[0,5]  → Z ∈ {9, 18, 36, 72, 144, 288}
Set 5: a=11, j∈[0,5]  → Z ∈ {11, 22, 44, 88, 176, 352}
Set 6: a=13, j∈[0,4]  → Z ∈ {13, 26, 52, 104, 208}
Set 7: a=15, j∈[0,4]  → Z ∈ {15, 30, 60, 120, 240}

有效範圍: 2 ≤ Z ≤ 384
```

---

## 解碼算法實現

### 算法架構分類

#### 1. 寄存器索引 (REG_INDEX) 算法

```cpp
// 特徵:
- 使用寄存器存儲變量節點值
- 索引式訪問校驗節點
- 適合中等碼率
- 較低共享內存需求

// 變體:
REG_INDEX_FP_DESC_DYN      : 基礎版本
REG_INDEX_FP_X2_DESC_DYN   : 雙碼字版本
REG_INDEX_FP_DESC_DYN_SMALL: 小Z優化
REG_INDEX_FP_DESC_DYN_ROW_DEP: 行依賴版本
```

#### 2. 分割x2 (SPLIT_INDEX_FP_X2) 算法

```cpp
// 特徵:
- 每CTA處理2個碼字
- 分割內存訪問模式
- 高吞吐量
- 需要較大共享內存

// 變體:
SPLIT_INDEX_FP_X2_DESC_DYN    : 通用版本
SPLIT_INDEX_FP_X2_DESC_DYN_SM86: SM86優化
SPLIT_INDEX_FP_X2_DESC_DYN_SM90: SM90優化 (最多40個校驗節點)
```

#### 3. 共享內存 (SHM_INDEX_FP) 算法

```cpp
// 特徵:
- 將變量節點值存儲在共享內存
- 最小化全局內存訪問
- 適合低Z值
- 需要較大共享內存
```

#### 4. Box-Plus算法 (Blackwell SM100)

```cpp
// 特徵:
- 新型近似算法
- 替代min-sum算法
- 更簡單的計算
- 更好的訪問模式
```

### 最小和 (Min-Sum) 更新

```cpp
// 標準min-sum:
Q_ij(t+1) = R_ij(t) - ∑_{k≠j} γ * sign(P_ik(t)) * min(|P_ik(t)|)

// 歸一化參數表:
g_min_sum_norm_BG1_Z384[47]  // BG1歸一化因子
g_min_sum_norm_BG2_Z384[43]  // BG2歸一化因子

// 根據校驗節點數m應用不同的歸一化因子
- BG1: m∈[4, 46], 校驗節點索引 → 歸一化值
- BG2: m∈[4, 42], 校驗節點索引 → 歸一化值
```

---

## 設備代碼實現

### LFSR支持函數

```cpp
// 升縮因子判斷
int set_from_Z(int Z)  // 返回Z所在集合 [0-7] 或 -1

// 例:
set_from_Z(128) = 0   // 2^7
set_from_Z(192) = 1   // 3*64
set_from_Z(320) = 2   // 5*64
```

### LLR數據類型操作

```cpp
// 數據類型檢查
template<typename TLLR>
CUDA_INLINE bool is_neg(const TLLR& llr);

template<>
CUDA_INLINE bool is_neg<__half>(const __half& llr)
{
    return __hlt(llr, 0);  // CUDA half比較
}

// 數據類型轉換
template<typename TLLR>
CUDA_INLINE float to_float(const TLLR& llr);

template<>
CUDA_INLINE float to_float(const __half& llr)
{
    return __half2float(llr);
}
```

### 硬判決輸出

```cpp
template<cuphyDataType_t TLLREnum>
__device__ void cta_write_hard_decision(LDPC_output_t tOutput,
                                        int codeWordIdx,
                                        int K,
                                        const LLR_t* srcAPP)
{
    // 根據LLR符號生成硬判決比特
    // 輸出: K比特字到 tOutput[K/32, codeWordIdx]
    
    // 算法:
    // 1. 每個warp處理32個LLR值
    // 2. 判決: bit = (llr < 0) ? 1 : 0
    // 3. 使用ballot_sync收集warp內比特
    // 4. 寫入tOutput
}
```

---

## 工作流程

### 完整解碼流程

```
輸入: LLR序列 (浮點值) + 配置
    ↓
[步驟1] 參數驗證
    ├─ BG檢查 (1或2)
    ├─ Z有效性檢查
    ├─ LLR類型檢查 (FP16/FP32)
    └─ 迭代次數檢查
    ↓
[步驟2] 歸一化設置
    ├─ 根據BG和m查表
    └─ 轉換為適當數據類型 (FP16/FP32)
    ↓
[步驟3] 算法選擇
    ├─ 檢查代碼長度
    ├─ 檢查LLR類型
    ├─ 查詢GPU計算能力
    └─ 應用吞吐量標誌
    ↓
[步驟4] 工作區分配
    ├─ 計算所需大小
    └─ 分配GPU內存
    ↓
[步驟5] 內核啟動
    ├─ 網格/塊配置 (取決於算法)
    ├─ 共享內存配置
    ├─ LLR加載
    ├─ 迭代解碼:
    │  ├─ 可變到校驗消息
    │  ├─ 校驗到可變消息
    │  └─ 估計消息
    └─ 硬判決輸出
    ↓
[步驟6] 後處理
    ├─ 可選軟輸出生成
    ├─ 結果驗證
    └─ 返回到主機
    ↓
輸出: 解碼比特 (LDPC_output_t) + 軟輸出 (可選)
```

---

## 配置常數

```cpp
// 支持的LLR類型
CUPHY_R_16F = 16-bit float (半精度)
CUPHY_R_32F = 32-bit float (單精度)

// 基圖
BG = 1  // LDPC基圖1 (低碼率)
BG = 2  // LDPC基圖2 (高碼率)

// 標誌
CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT = 0x01  // 優先吞吐量
CUPHY_LDPC_DECODE_USE_RESERVED_ALGO = 0x02  // 使用保留算法

// 計算能力
CC_7_0  = (7 << 32)      // Volta
CC_7_5  = (7 << 32) + 5  // Turing
CC_8_0  = (8 << 32)      // Ampere
CC_8_6  = (8 << 32) + 6  // Ampere (A10M)
CC_8_9  = (8 << 32) + 9  // Ampere (A6000)
CC_9_0  = (9 << 32)      // Hopper (H100)
CC_10_0 = (10 << 32)     // Blackwell (B100)

// 最大值
MAX_DL_LAYERS_PER_TB = 4
MAX_ENCODED_CODE_BLOCK_BIT_SIZE = 25344
```

---

## 使用示例

### 基本解碼

```cpp
// 1. 創建解碼器
ldpc::decoder decoder(context);

// 2. 配置參數
cuphyLDPCDecodeConfigDesc_t config{};
config.llr_type = CUPHY_R_16F;       // 半精度
config.BG = 1;                        // 基圖1
config.Z = 256;                       // 升縮因子
config.num_parity_nodes = 20;         // 20個校驗節點
config.Kb = 22;                       // 信息節點
config.max_iterations = 10;           // 10次迭代
config.algo = 0;                      // 自動選擇

// 3. 設置歸一化
decoder.set_normalization(config);

// 4. 準備輸入
// tLLR: LLR張量 (num_codewords維度)
// tDst: 輸出比特 (32比特字)

// 5. 執行解碼
cuphyStatus_t status = decoder.decode(
    tDst,           // 輸出
    tLLR,           // 輸入LLR
    optSoftOutputs, // 軟輸出 (可選)
    config,
    stream
);

if(status != CUPHY_STATUS_SUCCESS) {
    // 錯誤處理
}
```

### 批量傳輸塊解碼

```cpp
// 定義多個TB
cuphyLDPCDecodeDesc_t decodeDesc{};
decodeDesc.num_tbs = 3;                    // 3個TB

// 為每個TB設置配置
for(int i = 0; i < 3; i++) {
    decodeDesc.llr_input[i].addr = d_llr_ptrs[i];
    decodeDesc.llr_input[i].num_codewords = num_cw[i];
    // ... 其他配置
}

// 執行批量解碼
cuphyStatus_t status = decoder.decode_tb(decodeDesc, stream);
```

---

## 性能特性

✅ **多GPU支持** - 自動選擇最優算法
✅ **精度靈活性** - FP16 (高速) / FP32 (高精度)
✅ **動態配置** - 根據碼率自適應
✅ **雙碼字處理** - x2算法提高吞吐量
✅ **共享內存優化** - 最小化全局內存帶寬
✅ **行依賴** - 利用反饋提高收斂
✅ **軟輸出** - APP值支持後級處理

---

## 計算複雜度

### 每迭代複雜度

| 操作 | 複雜度 |
|------|--------|
| 可變→校驗 | O(m*Z) - 1次求和 |
| 校驗→可變 | O(m*Z*m) - min-sum計算 |
| 估計 | O(K+m*Z) |
| 總計/迭代 | O(m*Z*(K+m)) |

### 典型性能

| 配置 | GPU | 吞吐量 | 延遲 |
|------|-----|--------|------|
| BG1, m=20, Z=256, FP16 | H100 | ~GB/s | ~ms |
| BG2, m=30, Z=384, FP16 | H100 | ~GB/s | ~ms |
| 低碼率 | A100 | ~GB/s | ~ms |

---

## 3GPP標準映射

實現遵循3GPP TS 38.212標準:

| 組件 | 標準 | 說明 |
|------|------|------|
| LDPC基圖 | TS 38.212 5.3.2 | 基圖1和基圖2 |
| 碼率匹配 | TS 38.212 5.4.2 | 跳孔和打孔 |
| 加擾 | TS 38.211 5.2.1 | Gold序列 |
| 升縮因子 | TS 38.212 5.3.1 | Z值集合定義 |

---

## 延伸閱讀

- 3GPP TS 38.212 - 多路復用和編碼
- NVIDIA LDPC解碼優化白皮書
- [cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- LDPC理論和應用文獻
- CUDA性能優化指南
