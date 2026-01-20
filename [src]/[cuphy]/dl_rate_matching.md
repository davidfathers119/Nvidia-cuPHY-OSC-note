# 下行速率匹配 (Downlink Rate Matching)

## 概述

Downlink Rate Matching (DL Rate Matching) 是NVIDIA cuPHY庫中用於**5G下行PDSCH速率匹配和調制映射**的核心組件。它實現了LDPC編碼比特的速率匹配、加擾和多層映射，是5G接收端信號處理的關鍵環節。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - DL Rate Matching](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/dl_rate_matching)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `dl_rate_matching.hpp` | 數據結構定義和API |
| `dl_rate_matching.cuh` | CUDA輔助函數 - 參數驗證 |
| `dl_rate_matching.cu` | CUDA內核實現 - 速率匹配和調制 |
| `testDlRateMatching.cpp` | 測試程序 |

---

## 核心數據結構

### dlRateMatchingDescr_t

```cpp
struct dlRateMatchingDescr_t {
    // 輸入/輸出指針
    const uint32_t* d_rate_matching_input;      // LDPC編碼器輸出
    uint32_t* d_rate_matching_output;            // 速率匹配輸出 (可選)
    uint32_t* d_restructure_rate_matching_output; // 重構輸出
    
    // 數組指針
    const uint32_t* d_Er_array;                  // Er值數組
    const uint32_t* d_k0_array;                  // k0起始位置數組
    const uint32_t* d_TB_start_offset_array;     // TB起始偏移
    
    // 配置參數
    bool enable_scrambling;                      // 是否啟用加擾
    bool enable_layer_mapping;                   // 是否啟用多層映射
    
    // 維度信息
    uint32_t num_TBs;                            // 傳輸塊數
    uint32_t cmax;                               // 最大碼塊數
    uint32_t emax;                               // 最大速率匹配比特數
    uint32_t max_bits_per_layer;                 // 每層最大比特數
    uint32_t num_layers;                         // 層數
    
    // 工作區指針
    const PdschPerTbParams* cfg_workspace;       // TB參數配置
    
    // 調制融合擴展
    PdschDmrsParams* d_params;                   // DMRS參數
    uint16_t* temp_xtf_re_map;                   // RE映射表
    uint16_t max_PRB_BWP;                        // 最大PRB帶寬 (≤273)
    
    PdschUeGrpParams* d_ue_grp_params;           // UE組參數
};
```

### PdschPerTbParams

```cpp
struct PdschPerTbParams {
    uint32_t rv;              // 冗余版本 [0, 3]
    uint32_t Qm;              // 調制階數 (4, 16, 64, 256)
    uint32_t bg;              // LDPC基圖 (1 or 2)
    uint32_t Nl;              // 層數 [1, MAX_DL_LAYERS_PER_TB]
    uint32_t num_CBs;         // 碼塊數
    uint32_t Zc;              // 升縮因子
    uint32_t N;               // LDPC編碼長度
    uint32_t Ncb;             // 循環緩衝區長度
    uint32_t K;               // 信息比特數
    uint32_t G;               // 速率匹配比特數
    uint32_t F;               // 填充比特數
    uint32_t cinit;           // 加擾種子
};
```

---

## API函數

### TB參數設置

```cpp
cuphyStatus_t cuphySetTBParams(PdschPerTbParams* tb_params_struct,
                               uint32_t cfg_rv,      // 冗余版本
                               uint32_t cfg_Qm,      // 調制階數
                               uint32_t cfg_bg,      // LDPC基圖
                               uint32_t cfg_Nl,      // 層數
                               uint32_t cfg_num_CBs, // 碼塊數
                               uint32_t cfg_Zc,      // 升縮因子
                               uint32_t cfg_G,       // 速率匹配比特數
                               uint32_t cfg_F,       // 填充比特數
                               uint32_t cfg_cinit,   // 加擾種子
                               uint32_t cfg_Nref)    // 參考長度
```

**功能**: 設置並驗證每個TB的參數

**參數驗證**:
```cpp
- cfg_rv: 必須在 [0, 3] 範圍內
- cfg_Qm: 必須是 CUPHY_QAM_4, CUPHY_QAM_16, CUPHY_QAM_64, CUPHY_QAM_256
- cfg_bg: 必須是 1 或 2
- cfg_Nl: 必須在 [1, MAX_DL_LAYERS_PER_TB] 範圍內 (最多4層)
```

### k0計算

```cpp
int compute_k0(int rv, int bg_num, int Ncb, int Zc)
```

**功能**: 根據冗余版本計算速率匹配起始位置k0

**3GPP標準映射** (TS 38.212 Table 5.4.2.1-2):
```
BG=1:  enumerator[0] = {0, 17, 33, 56}
BG=2:  enumerator[1] = {0, 13, 25, 43}

計算公式:
k0 = floor(enumerator[rv] * Ncb / (UNPUNCTURED_VAR_NODES * Zc)) * Zc
```

### Er計算

```cpp
void compute_rate_matching_length(uint32_t Er[],    // 輸出Er值
                                   int C,             // 碼塊數
                                   int Qm,            // 調制階數
                                   int Nl,            // 層數
                                   int G,             // 速率匹配比特
                                   uint32_t& Emax,    // 最大Er
                                   bool& updated_Emax,// Emax是否更新
                                   bool no_rate_matching = false,
                                   int word_size = 32)
```

**功能**: 計算每個碼塊的速率匹配長度

**Er值計算** (TS 38.212 Section 5.4.2.1):
```cpp
modulo_C = (G / (Nl * Qm)) % C;      // 碼塊分割點
CB_split = C - modulo_C;              // 分割後碼塊ID

Er[0] = CB_split;                     // 分割ID
Er[1] = Nl * Qm * floor(G / (Nl * Qm * C));  // 最小Er

max_Er = Er[1] + (modulo_C == 0 ? 0 : Nl * Qm)
```

---

## 主內核實現

### 速率匹配內核

#### dl_rate_matching (獨立模式)

```cpp
__global__ void dl_rate_matching(dlRateMatchingDescr_t* p_desc)
```

**功能**: 執行純速率匹配 (不包括調制)

**網格配置**:
- `gridDim.x`: cmax (最大碼塊數)
- `gridDim.y`: num_TBs (傳輸塊數)
- `blockDim.x`: 288 (線程數)

**共享內存**:
```
如果enable_scrambling:   rounded_Er_elements + 1 個uint32_t
如果enable_layer_mapping: MAX_DL_LAYERS_PER_TB * rounded_Er_elements

總大小: (scrambling_shmem + layer_mapping_shmem) * sizeof(uint32_t)
```

#### fused_dl_rm_and_modulation (融合模式)

```cpp
template<bool precoding>
__global__ void fused_dl_rm_and_modulation(dlRateMatchingDescr_t* p_desc)
```

**功能**: 速率匹配 + 調制融合處理

**特點**: 支持可選的波束賦形預編碼

#### restructure_rate_matching_output

```cpp
__global__ void restructure_rate_matching_output(dlRateMatchingDescr_t* p_desc)
```

**功能**: 重構速率匹配輸出 (層映射專用)

**網格配置**:
- `blockDim.x`: 256
- `gridDim.x`: cmax
- `gridDim.y`: num_TBs

---

## 速率匹配算法

### 比特選擇與交織

#### 步驟1: 讀取LDPC編碼比特

```
對於每個碼塊CB和索引 index = 0 to CB_Er-1:
1. 計算讀取索引: read_index = index + k0 (第一遍)
2. 檢查是否需要跳過填充比特
3. 從CB_ldpc_encoder_output讀取比特
```

#### 步驟2: 比特交織 (Bit Interleaving)

```
根據調制階數Qm進行交織:

QAM4 (Qm=2):
  final_index = EdivQm_bit_id * 2 + EdivQm_block_id

QAM16 (Qm=4):
  final_index = EdivQm_bit_id * 4 + EdivQm_block_id

QAM64 (Qm=6):
  final_index = EdivQm_bit_id * 6 + EdivQm_block_id

QAM256 (Qm=8):
  final_index = EdivQm_bit_id * 8 + EdivQm_block_id
```

#### 步驟3: 加擾 (XOR with Gold序列)

```cpp
if (enable_scrambling) {
    uint32_t scrambling_bit = CB_scrambling_vals[scrambling_index] >> scrambling_bit_pos;
    bit_read ^= scrambling_bit;
}
```

### 緩衝區情況

#### 情況A: Kd ≥ k0 (無繞過)

```
<------------ N比特 ------------>
|-------|--------|--------|-------|
  k0     Kd   Kd+F       N
           ↑
        第一遍訪問範圍: [k0, Kd) 和 [Kd+F, N)
```

#### 情況B: k0 ≥ (Kd + F) (繞過到緩衝區開始)

```
<------------ N比特 ------------>
|-------|--------|--------|-------|
  0      Kd    Kd+F      k0
                         ↑
第一遍訪問: [k0, N) 
後續遍數: [0, Kd) 和 [Kd+F, N)
```

---

## 調制映射

### QAM符號映射

#### QAM4 (QPSK)

```cpp
float reciprocal_sqrt2 = 0.707106781186547f * beta_qam;

// 查表映射 (rev_qam_16_long)
tmp_val.x = ((qam_value & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
tmp_val.y = ((qam_value & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
```

#### QAM16

```cpp
__shared__ __half shmem_qam_16[8];
shmem_qam_16[i] = (__half)(rev_qam_16_long[i] * beta_qam);

// 索引映射 (map_index_6bits)
masked_index = (index & 0x1) | ((index & 0x4) >> 1) | ((index & 0x10) >> 2);
```

#### QAM64

```cpp
__shared__ __half shmem_qam_64[8];
shmem_qam_64[i] = (__half)(rev_qam_64[i] * beta_qam);

// 6比特映射
x_index = map_index_6bits(qam_value);
y_index = map_index_6bits(qam_value >> 1);
```

#### QAM256

```cpp
__shared__ __half shmem_qam_256[16];
shmem_qam_256[i] = (__half)(rev_qam_256[i] * beta_qam);

// 8比特映射
x_index = map_index_8bits(qam_value);
y_index = map_index_8bits(qam_value >> 1);
```

### 層映射 (Layer Mapping)

```cpp
if (enable_layer_mapping) {
    // 將碼塊比特映射到各層
    for (int layer = 0; layer < Nl; layer++) {
        int layer_id = port_ids_array[layer] + 8*n_scid + 16*nlAbove16;
        
        // 每層分配 per_layer_Er / Qm 個符號
        int symbols_per_layer = per_layer_Er / Qm;
        
        // 層內符號交錯映射
        for (int symbol = 0; symbol < symbols_per_layer; symbol++) {
            output[layer * padded_bits_per_layer + symbol*Qm : (symbol+1)*Qm];
        }
    }
}
```

### 波束賦形預編碼 (可選)

```cpp
if (precoding) {
    for (int antenna_port = 0; antenna_port < Np; antenna_port++) {
        __half2 matCoeff = pmW[layer * Np + antenna_port];
        __half2 precoded = mult_half2(qam_symbol, matCoeff);
        atomicAdd(&modulation_output[output_index + antenna_port_offset], precoded);
    }
} else {
    modulation_output[output_index] = qam_symbol;
}
```

---

## 複雜算法細節

### 跨word邊界比特提取

```cpp
// 當比特位置跨越32比特邊界時
int tmp_final_index = final_index + (gold32_CB_start_output & ELEMENT_MASK);
int CB_scrambling_index = tmp_final_index >> ELEMENT_BITS;  // word索引
int CB_scrambling_bit = tmp_final_index & ELEMENT_MASK;      // 比特位置

// 從兩個相鄰words提取
uint32_t scrambling_val = CB_scrambling_vals[CB_scrambling_index];
uint32_t next_scrambling = CB_scrambling_vals[CB_scrambling_index + 1];

// 移位並組合
scrambling_bit = (scrambling_val >> CB_scrambling_bit) & 0x1;
```

### QAM64特殊處理

```cpp
// QAM64使用6比特編碼,不能對齐32比特邊界
// 特殊處理跨word讀取

if (i_mod_3 == 0) {
    // 讀6比特符號
    read_val from dl_rm_shmem[i];
    // 可能需要從next element讀取部分比特
    qam_value |= ((dl_rm_shmem[i + 1] & 0x0FU) << 2);
} else if (i_mod_3 == 1) {
    read_val >>= 4;
    read_val |= ((dl_rm_shmem[i+1] & 0x03U) << 28);
} else if (i_mod_3 == 2) {
    read_val >>= 2;
}
```

### 層映射索引計算

```cpp
// 多層模式下,計算QAM符號在層內的位置
int tmp_div = EdivQm_bit_id * reciprocal_Nl;  // 商
uint32_t tmp_layer = EdivQm_bit_id - tmp_div * Nl;  // 層索引

int index_in_layer = tmp_div * Qm + EdivQm_block_id;
int final_layer_index = tmp_layer * emax + index_in_layer;

// 寫入層映射共享內存
atomicOr(&dl_rm_shmem[(final_layer_index >> ELEMENT_BITS)], layered_bit_read);
```

---

## 性能優化

### 共享內存優化

✅ **動態共享內存** - 根據emax和層數動態分配
✅ **原子操作最小化** - 使用atomicOr合併32比特操作
✅ **Warp級優化** - 利用NVIDIA特有指令(__brev, __popc)

### 融合內核優化

✅ **單遍內核** - 速率匹配 + 調制融合,減少內存往返
✅ **共享內存預編碼矩陣** - 避免全局內存訪問延遲
✅ **模板特化** - Tnl參數編譯時已知,允許編譯器優化

### 特殊情況快速路徑

```cpp
// QAM256 special_case優化:
bool special_case = (only_first_pass && (k0 == 0) && CB_Er_alignment);

if (special_case) {
    // 避免比特級原子操作,直接操作4比特塊
    // 減少原子操作次數至1/4
}
```

---

## 工作流程

### 完整流程

```
輸入: LDPC編碼比特序列 (d_rate_matching_input)
    ↓
[步驟1] 參數驗證與初始化
    ├─ cuphySetTBParams() - 驗證所有TB參數
    ├─ compute_k0() - 計算每個TB的k0
    └─ compute_rate_matching_length() - 計算Er值
    ↓
[步驟2] 工作區準備
    ├─ 複製k0數組到GPU
    ├─ 複製TB_start_offset數組到GPU
    └─ 複製Er數組到GPU
    ↓
[步驟3] 速率匹配內核 (若enable_layer_mapping=false)
    ├─ 網格: (cmax, num_TBs)
    ├─ 塊: 288線程
    ├─ 共享: 根據emax和加擾配置
    │
    ├─[子步驟a] 加擾序列生成 (若enable_scrambling)
    │   └─ gold32(cinit, CB_start + bit_offset)
    │
    ├─[子步驟b] 比特選擇 (循環緩衝區)
    │   ├─ 第一遍: [k0, Kd) ∪ [Kd+F, N)
    │   └─ 後續遍: [0, Kd) ∪ [Kd+F, N)
    │
    └─[子步驟c] 比特交織和加擾
        └─ atomicOr(&shmem[final_index >> 5], bit << (final_index & 31))
    ↓
[步驟4] 調制內核 (若enable_layer_mapping=true)
    ├─ 融合內核: fused_dl_rm_and_modulation
    │
    ├─[調制選擇]
    │   ├─ QAM4_work_*
    │   ├─ QAM16_work_*
    │   ├─ QAM64_work_*
    │   └─ QAM256_work_*
    │
    ├─[預編碼應用] (若enablePrcdBf=1)
    │   ├─ 加載預編碼矩陣到共享內存
    │   └─ 複數乘法: mult_half2()
    │
    └─[層映射]
        └─ 多層原子加法到調制輸出
    ↓
[步驟5] 重構內核 (若restructure_kernel)
    ├─ 網格: (cmax, num_TBs)
    ├─ 塊: 256線程
    │
    └─ 重新組織層映射輸出為線性順序
    ↓
輸出: 調制符號或速率匹配比特
```

---

## 配置常數

```cpp
// 調制相關
CUPHY_QAM_4   = 2  // QPSK
CUPHY_QAM_16  = 4  // 16-QAM
CUPHY_QAM_64  = 6  // 64-QAM
CUPHY_QAM_256 = 8  // 256-QAM

// 內核配置
GLOBAL_BLOCK_SIZE = 288        // 線程數
MAX_DL_LAYERS_PER_TB = 4       // 最大層數
MAX_DL_PORTS = 8               // 最大天線端口
OFDM_SYMBOLS_PER_SLOT = 14     // 時隙符號數
CUPHY_N_TONES_PER_PRB = 12     // 每PRB子載波數

// LDPC相關
CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES = 66
CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES = 50
MAX_ENCODED_CODE_BLOCK_BIT_SIZE = 25344

// 位操作
ELEMENT_SIZE = 32   // 字大小(比特)
ELEMENT_BITS = 5    // log2(32)
ELEMENT_MASK = 31   // 0x1F
```

---

## 3GPP標準映射

實現完全遵循3GPP TS 38.212標準:

| 組件 | 標準 | 說明 |
|------|------|------|
| 速率匹配 | TS 38.212 5.4.2.1 | 比特選擇與交織 |
| k0計算 | TS 38.212 Table 5.4.2.1-2 | 冗余版本映射 |
| Er計算 | TS 38.212 5.4.2.1 | 速率匹配長度 |
| 加擾 | TS 38.211 5.2.1 | Gold序列 |
| 層映射 | TS 38.212 5.3.1 | MIMO多層映射 |
| 預編碼 | TS 38.202 5.3 | 波束賦形 |

---

## 使用示例

### 基本設置

```cpp
// 1. 創建工作區和參數
PdschPerTbParams h_params[num_TBs];
uint32_t h_workspace[num_TBs * 4];  // k0, TB_start, Er

// 2. 設置每個TB的參數
for (int i = 0; i < num_TBs; i++) {
    cuphySetTBParams(&h_params[i],
        rv = 0,                    // 冗余版本0
        Qm = CUPHY_QAM_16,        // 16-QAM
        bg = 1,                    // 基圖1
        Nl = 2,                    // 2層MIMO
        num_CBs = 8,              // 8個碼塊
        Zc = 256,                 // 升縮因子
        G = 10000,                // 速率匹配比特數
        F = 40,                   // 填充比特數
        cinit = 0x12345678,       // 加擾種子
        Nref = 66*256             // 參考長度
    );
}

// 3. 複製參數到GPU
cudaMemcpy(d_params, h_params, sizeof(PdschPerTbParams)*num_TBs, 
           cudaMemcpyHostToDevice);
cudaMemcpy(d_workspace, h_workspace, num_TBs*4*sizeof(uint32_t),
           cudaMemcpyHostToDevice);
```

### 速率匹配調用

```cpp
cuphySetupDlRateMatching(
    launchConfig,
    status,
    d_ldpc_output,              // LDPC編碼輸出
    d_rate_matching_output,     // 速率匹配輸出
    d_restructure_output,       // 重構輸出
    d_modulation_output,        // 調制符號輸出
    d_xtf_re_map,              // RE映射表
    max_PRB_BWP,               // 最大PRB數
    num_TBs,
    num_layers,
    enable_scrambling = 1,
    enable_layer_mapping = 1,  // 啟用層映射
    enable_modulation = 1,     // 啟用調制
    precoding = 0,             // 無預編碼
    restructure_kernel = 1,    // 運行重構內核
    batching = 1,
    h_workspace, d_workspace,
    h_params, d_params,
    d_dmrs_params,
    d_ue_grp_params,
    cpu_desc, gpu_desc,
    enable_desc_async_copy = 1,
    stream
);

// 執行內核
cuphyLaunchDlRateMatching(launchConfig, stream);
```

---

## 性能特性

✅ **全GPU加速** - CUDA內核完全實現
✅ **多層MIMO** - 支持最多4層並行處理
✅ **波束賦形** - 可選預編碼支持
✅ **CSI-RS感知** - 支持CSI-RS預處理
✅ **融合設計** - 速率匹配+調制單遍內核
✅ **動態配置** - 根據TB參數自適應共享內存

---

## 計算複雜度

### 每比特操作

| 操作 | 複雜度 |
|------|--------|
| 比特選擇 | O(1) - 單次內存訪問 |
| 加擾 | O(1) - XOR操作 |
| 調制映射 | O(1) - 表查詢 |
| 預編碼 | O(Nl*Np) - 複數乘法 |

### 總吞吐量

- **非融合**: ~GB/s (受內存往返限制)
- **融合**: ~2-3x加速 (單遍+緩存)
- **預編碼**: +30%開銷 (原子操作)

---

## 延伸閱讀

- 3GPP TS 38.212 - 多路復用和編碼
- 3GPP TS 38.211 - 物理層過程
- NVIDIA CUDA Best Practices Guide
- [cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- LDPC碼相關文獻
- MIMO預編碼理論
