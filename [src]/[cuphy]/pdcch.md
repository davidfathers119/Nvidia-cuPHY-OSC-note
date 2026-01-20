# NVIDIA cuPHY PDCCH (Physical Downlink Control Channel) Embedding

**File Location**: `cuPHY/src/cuphy/pdcch/`
**Main File**: `embed_pdcch_tf_signal.cu`
**Purpose**: High-performance CUDA kernel for mapping PDCCH DCI (Downlink Control Information) payloads to time-frequency grid symbols in 5G NR downlink transmission

---

## 概述 (Overview)

PDCCH Embedding 組件負責在 5G NR 物理層下行傳輸中，將編碼/速率匹配的 DCI 比特映射到複數域的 QPSK 星座點，並嵌入到時間-頻率資源網格中。該實現支持：

- **多個 Coreset** (控制資源集) 和 DCI
- **靈活的聚合等級** (1, 2, 4, 8, 16)
- **Bundle 映射** (非交錯和交錯模式)
- **DMRS** (Demodulation Reference Signal) 生成
- **波束賦型** (Beamforming) 支持

---

## 核心概念

### PDCCH 資源結構

```
Coreset (Control Resource Set)
├─ Frequency Domain: 連續的 RB (Resource Block)
├─ Time Domain: 1-3 OFDM 符號
├─ Bundle: 資源分組單位
│  ├─ Bundle Size: 2, 3, or 6 RB
│  └─ Bundle Configuration: Interleaved or Non-Interleaved
└─ CCE (Control Channel Element): 邏輯資源單位
   └─ Aggregation Level: 1-16 CCE per DCI
```

### 關鍵術語

| 術語 | 說明 |
|------|------|
| **Coreset** | 控制信道資源集，具有固定的時頻位置 |
| **DCI** | 下行控制信息，傳輸調度/功率等信息 |
| **CCE** | 控制信道元素，PDCCH 的邏輯資源 |
| **Bundle** | RB 分組，支持聚合和交錯映射 |
| **Aggregation Level** | DCI 占用的 CCE 數量 (1, 2, 4, 8, 16) |
| **DMRS** | 解調參考信號，用於信道估計 |
| **RE** | Resource Element，12 個子載波 × 1 OFDM 符號 |

---

## 主要數據結構

### `PdcchParams` - Coreset 配置

Coreset 層級的參數，定義時頻資源和 DCI 配置：

```cpp
struct PdcchParams {
    // 頻域配置
    uint64_t        freq_domain_resource;     // 64 位 RB 分配掩碼
    uint32_t        rb_coreset;               // Coreset 包含的 RB 數
    uint32_t        coreset_map;              // 移位後的頻域掩碼
    
    // 時域配置
    uint32_t        n_sym;                    // Coreset 占用的 OFDM 符號數 (1-3)
    uint32_t        n_f;                      // 總帶寬的 RB 數 × 12
    uint32_t        slot_number;              // 時隙編號
    uint32_t        start_sym;                // 起始符號位置
    uint32_t        start_rb;                 // 起始 RB 位置
    
    // Bundle 配置
    uint32_t        bundle_size;              // Bundle 大小 (2, 3 or 6)
    uint32_t        interleaver_size;        // 交錯器大小 (2, 3 or 6)
    bool            interleaved;              // 是否使用交錯模式
    uint32_t        shift_index;              // 交錯移位索引
    
    // 邏輯 CCE 配置
    uint32_t        n_CCE;                    // 邏輯 CCE 數量 = count_set_bits(freq_domain_resource) × n_sym
    
    // DCI 配置
    uint32_t        dciStartIdx;              // DCI 開始索引
    uint32_t        num_dl_dci;               // DCI 數量
    
    // 型式
    uint32_t        coreset_type;             // 0 = 分佈式, 1 = 本地化
    uint32_t        testModel;                // 測試模式標誌 (0 = 禁用)
    
    // 槽位
    void*           slotBufferAddr;           // 輸出時頻信號緩衝區
};
```

### `cuphyPdcchDciPrm_t` - DCI 參數

單個 DCI 的配置：

```cpp
struct cuphyPdcchDciPrm_t {
    // 編碼參數
    uint32_t        Npayload;                 // DCI 有效載荷位數
    uint32_t        rntiCrc;                  // RNTI 用於 CRC 加擾
    
    // 映射參數
    uint32_t        cce_index;                // 起始 CCE 索引
    uint32_t        aggr_level;               // 聚合等級 (1, 2, 4, 8, 16)
    uint32_t        dmrs_id;                  // DMRS ID (0-65535)
    
    // 功率參數
    float           beta_qam;                 // QAM 符號功率縮放
    float           beta_dmrs;                // DMRS 功率縮放
    
    // 波束賦型參數
    uint8_t         enablePrcdBf;             // 波束賦型啟用標誌
    uint16_t        pmwPrmIdx;                // 波束矩陣索引
    
    // 導出參數 (在內核中計算)
    uint32_t        rntiBits;                 // RNTI (16 位)
};
```

---

## 演算法: CRC 和加擾

### 1. CRC 計算: `pdcchAddCrc()`

DCI 有效載荷的 24 位 CRC 計算和加擾

```cpp
void pdcchAddCrc(uint8_t* dci_payload,              // 輸入有效載荷
                 uint32_t& dci_crc,                 // 輸出 CRC (24 位)
                 const uint32_t rnti_crc,           // RNTI 用於加擾
                 const uint32_t payload_bits)       // 有效載荷位數
{
    // 步驟 1: 計算 CRC
    // CRC 多項式: 0x01B2B117 (24 位)
    // 初始值: 0
    // 包括 24 位 1 的前置
    dci_crc = computePdcchCRC<uint32_t, 24>(
        dci_payload,
        payload_bits,
        G_CRC_24_C,           // 0x01B2B117
        0,                    // initVal
        1,                    // stride
        false                 // 不包括 CRC 1 (已在函數中處理)
    );
    
    // 步驟 2: 用 RNTI 加擾 CRC 的 16 LSB
    dci_crc ^= (rnti_crc & 0x0FFFFU);
}
```

**CRC 計算流程**:

```
Input: DCI Payload (Npayload bits)
    |
    v
Prepend: 24 bits of 1's (0xFFFFFF)
    |
    v
Shift through CRC polynomial: 0x01B2B117
    |
    v
Output: 24-bit CRC
    |
    v
XOR with RNTI[15:0]:  CRC ^= (RNTI & 0xFFFF)
    |
    v
Result: Scrambled CRC for DCI
```

**3GPP 標準**: TS 38.212 § 5.1

### 2. 加擾: Gold 序列生成

#### `pdcchGenGoldSeq()` - Gold 序列生成

Gold 序列用於 PDCCH 比特加擾和 DMRS 序列生成

```cpp
void pdcchGenGoldSeq(uint32_t c_init,     // 初始化值
                     uint32_t len,        // 序列長度 (位)
                     uint32_t* x1,        // LFSR x1 輸出 (長度 ≥ 32+len-1)
                     uint32_t* x2,        // LFSR x2 輸出 (長度 ≥ 32+len-1)
                     uint32_t* c)         // 輸出 Gold 序列 (打包為 uint32_t)
{
    // 步驟 1: 初始化 x1 和 x2 LFSR
    x1[0] = 1;                                      // x1(0) = 1
    for(int i = 0; i < 32; i++) {
        x1[i] = 0;                                  // 其餘 x1(1..31) = 0
        x2[i] = (c_init >> i) & 0x1;                // x2(0..31) = c_init 位
    }
    
    // 步驟 2: 運行 LFSR Nc + len - 31 次迭代
    for(int i = 0; i < Nc + len - 31; i++) {
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2;       // x1(n+31) = x1(n+3) + x1(n) mod 2
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2;  // x2 反饋
    }
    
    // 步驟 3: 合併 x1 和 x2，打包為 uint32_t
    for(int i = 0; i < len; i += 32) {
        uint32_t val = 0;
        for(int offset = 0; offset < 32 && (i + offset) < len; offset++) {
            uint32_t bit = (x1[i + offset + Nc] + x2[i + offset + Nc]) % 2;
            val |= (bit << (31 - offset));
        }
        c[i >> 5] = val;
    }
}
```

**LFSR 多項式**:
- **x1**: $x^{31} + x^3 + 1$ (Fibonacci 型)
- **x2**: $x^{31} + x^3 + x^2 + x + 1$ (反饋 x2(n) = x2(n+3) + x2(n+2) + x2(n+1) + x2(n))

**常數**: Nc = 1600 (預運行迭代次數)

#### `genPdcchScramSeq()` - PDCCH 加擾序列

```cpp
void genPdcchScramSeq(uint32_t dmrs_id,           // DMRS ID (16 位)
                      uint32_t rnti_bits,         // RNTI (16 位)
                      uint32_t Nscram,            // 序列長度
                      uint32_t* x_scramSeq)       // 輸出序列
{
    // c_init = (RNTI << 16) | DMRS_ID, masked to 31 bits
    uint32_t c_init = ((rnti_bits << 16) + dmrs_id) & 0x7fffffffU;
    
    // 生成 Gold 序列
    uint32_t x1[Nc + Nscram];
    uint32_t x2[Nc + Nscram];
    
    pdcchGenGoldSeq(c_init, Nscram, x1, x2, x_scramSeq);
}
```

---

## DMRS 生成: `generate_dmrs()` 內核函數

生成 PDCCH DMRS 符號和放置在時頻網格上

```cpp
template <typename TComplex, typename Block>
__device__ void generate_dmrs(
    TComplex* __restrict__ dmrs_seqs,          // 輸出 DMRS 序列 [3 × n_rb]
    uint32_t* __restrict__ gold_seqs,          // Gold 序列 [6 × n_rb / 32]
    uint32_t dmrs_id,                          // DMRS ID
    uint32_t n_rb,                             // DMRS RB 數 (n_rb = 6 * aggr_level / n_sym)
    uint32_t start_rb,                         // 起始 RB
    float   beta_dmrs,                         // DMRS 功率縮放
    uint32_t slot_number,                      // 時隙編號
    uint32_t n_sym,                            // Coreset 符號數 (1-3)
    uint32_t start_sym,                        // Coreset 起始符號
    Block&   block,                            // 協作線程組
    uint32_t coreset_type)                     // Coreset 型式 (0/1)
```

**步驟**:

1. **計算 c_init**
   ```cpp
   c_init = ((1 << 17) × (14 × slot_number + t + 1) × (2 × dmrs_id + 1) + (2 × dmrs_id)) & ~(1 << 31)
   ```
   其中 t = start_sym + symbol_id

2. **生成 Gold 序列**
   - 每個線程計算一個 32 位 Gold 序列字
   - 存儲到共享內存

3. **QPSK 調制和縮放**
   - 從 Gold 序列提取 2 位
   - 映射到 QPSK 星座 (DMRS = QPSK)
   - 乘以 beta_dmrs 縮放因子

4. **結果**
   - DMRS_seqs[i] = complex(real, imag) × beta_dmrs
   - 3 個 DMRS 符號 per RB (位置 1, 5, 9)

---

## Bundle 映射: `compute_map()` 內核函數

計算邏輯 Bundle 到物理 Bundle 的映射

```cpp
__device__ inline void compute_map(
    uint32_t  bundles_per_level,               // Bundle/聚合等級 = 6 / bundle_size
    uint32_t  aggr_level,                      // 聚合等級 (1-16)
    uint32_t  cce_index,                       // 起始 CCE 索引
    bool      interleaved,                     // 交錯模式標誌
    uint32_t  interleaver_size,                // 交錯器大小 (2, 3, or 6)
    uint32_t  shift_index,                     // 交錯移位
    uint32_t  C,                               // 交錯參數 = n_CCE × 6 / (bundle_size × interleaver_size)
    uint32_t  n_sym,                           // 符號數
    uint32_t  n_CCEs,                          // 總 CCE 數
    uint8_t*  log_to_physical_map,             // Coreset bit 到物理編號映射
    uint16_t* phy_bundles)                     // 輸出物理 Bundle (已排序)
```

**映射流程**:

1. **計算邏輯 Bundle 編號**
   ```cpp
   log_bundle = bundles_per_level × (cce_index + j) + (i % bundles_per_level)
   ```

2. **應用交錯 (如果啟用)**
   ```cpp
   if (interleaved) {
       c = log_bundle / interleaver_size
       r = log_bundle - c × interleaver_size
       log_bundle = (r × C + c + shift_index) % N_bundle
   }
   ```

3. **映射到物理 Bundle**
   - 使用 Coreset 頻域掩碼將邏輯位映射到物理位置

4. **排序物理 Bundle**
   - 使用 Bitonic 排序 (交錯模式)
   - 確保遞增順序

---

## 主內核: `genPdcchTfSignalKernel()`

將 PDCCH DCI 和 DMRS 映射到時頻網格

```cpp
template <typename TComplex>
__global__ void genPdcchTfSignalKernel(
    uint8_t* __restrict__ d_x_tx_addr,         // 輸入加擾 DCI 比特
    uint32_t* __restrict__ d_x_scramSeq_addr,  // 加擾序列
    uint32_t num_coresets,                     // Coreset 數量
    PdcchParams* __restrict__ coreset_params,  // Coreset 配置
    cuphyPdcchDciPrm_t* __restrict__ params,   // DCI 參數
    cuphyPdcchPmWOneLayer_t* __restrict__ pmw_params)  // 波束矩陣
```

**執行模型**:
- gridDim.x = max_n_sym (最多 3)
- gridDim.y = num_DCIs
- blockDim.x = 128
- 每個 block: 1 DCI × 1 符號

**內核邏輯**:

### 步驟 1: Coreset 查詢
```cpp
for (int i = 0; i < num_coresets; i++) {
    if (DCI_id in coreset i's DCI range) {
        coreset_idx = i;
        break;
    }
}
```

### 步驟 2: 初始化共享內存
```cpp
__shared__ TComplex shmem_dmrs_seqs[(coreset_rb × 6 × 3)];      // DMRS 序列
__shared__ uint32_t shmem_gold_seqs[...];                       // Gold 序列
__shared__ uint8_t log_to_physical_map[64];                     // 映射表
__shared__ uint16_t phy_bundles[64];                            // 物理 Bundle
```

### 步驟 3: 生成 DMRS
```cpp
generate_dmrs(shmem_dmrs_seqs, shmem_gold_seqs, 
              dmrs_id, coreset_rb × 6, start_rb, beta_dmrs,
              slot_number, n_sym, start_sym, block, coreset_type);
```

### 步驟 4: 計算 Bundle 映射
```cpp
compute_map(bundles_per_level, aggr_level, cce_index,
            interleaved, interleaver_size, shift_index, C,
            n_sym, n_CCEs, &log_to_physical_map[0], &phy_bundles[0]);
```

### 步驟 5: 映射資源元素 (RE)
每個線程映射一個 RE:

```cpp
for (int tid = threadIdx.x; tid < total_n_REs; tid += blockDim.x) {
    // 計算 RE 位置
    int contiguous_res_chunk_id = tid / (12 × contiguous_rbs);
    int phy_bundle_id = phy_bundles[contiguous_res_chunk_id];
    int g_w_idx = pdcch_start_freq + (symbol_id + start_sym) × n_f 
                  + phy_bundle_id × contiguous_rbs × 12 
                  + (tid % (12 × contiguous_rbs));
    
    TComplex val;
    
    if ((tid & 0x3) == 1) {
        // 位置 1, 5, 9 (DMRS 位置)
        int dmrs_idx = phy_bundle_id × contiguous_rbs × 3 
                     + (((tid % (12 × contiguous_rbs)) - 1) >> 2);
        val = shmem_dmrs_seqs[dmrs_idx];
    } else {
        // QAM 位置 (9 QAM + 3 DMRS per RB)
        
        // 比特索引計算
        int qam_idx = (tid == 0) ? symbol_id × n_qam_per_sym 
                                 : (tid - ((tid-1)/4 + 1)) + symbol_id × n_qam_per_sym;
        int bit_idx = 2 × qam_idx;
        
        // 提取比特
        int x_tx_x = (d_x_tx[bit_idx / 8] >> (bit_idx % 8)) & 0x1;
        int x_tx_y = (d_x_tx[bit_idx / 8] >> ((bit_idx+1) % 8)) & 0x1;
        
        // 提取加擾序列
        uint32_t scram_val = d_x_scramSeq[bit_idx >> 5];
        int x = (x_tx_x + (scram_val >> (31 - (bit_idx & 0x1F)))) & 0x1;
        int y = (x_tx_y + (scram_val >> (31 - ((bit_idx+1) & 0x1F)))) & 0x1;
        
        // QPSK 調制: 1/sqrt(2) × (1 - 2x) 和 1/sqrt(2) × (1 - 2y)
        val.x = 0.70710678f × (1 - 2×x) × beta_qam;
        val.y = 0.70710678f × (1 - 2×y) × beta_qam;
    }
    
    // 應用波束賦型或直接寫入
    if (enablePrcdBf) {
        for (int idx = 0; idx < nPorts; idx++) {
            tf_signal[g_w_idx + offset_per_port×idx] = 
                __hcmadd(val, pmw_params[pmwPrmIdx].matrix[idx], zeroValue);
        }
    } else {
        tf_signal[g_w_idx] = val;
    }
}
```

---

## 加擾序列生成內核: `genScramblingSeqKernel()`

生成 DCI 比特加擾序列

```cpp
__global__ void genScramblingSeqKernel(
    uint32_t* __restrict__ d_x_scramSeq_addr,
    cuphyPdcchDciPrm_t* __restrict__ params)
```

**執行模型**:
- gridDim.x = num_DCIs
- blockDim.x = 64

**邏輯**:

```cpp
const int DCI_id = blockIdx.x;
const int tid = threadIdx.x;

// 計算 c_init
uint32_t c_init = ((params[DCI_id].rntiBits << 16) + params[DCI_id].dmrs_id) & 0x7fffffffU;

// 每個 DCI 的加擾序列長度
int tx_bits = 2 × 9 × 6 × params[DCI_id].aggr_level;  // 每個聚合等級 108 位
int max_elements = tx_bits / 32 + ((tx_bits % 32) != 0 ? 1 : 0);

// 生成 Gold 序列
if (tid < max_elements) {
    uint32_t offset = DCI_id × (CUPHY_PDCCH_MAX_TX_BITS_PER_DCI / 32);
    d_x_scramSeq_addr[offset + tid] = __brev(gold32(c_init, tid × 32));
    // __brev() 反轉比特順序以匹配硬體期望
}
```

---

## 準備函數: `cuphyPdcchPipelinePrepare()`

在主機端準備 PDCCH 管道：CRC、加擾、比特反轉

```cpp
cuphyStatus_t cuphyPdcchPipelinePrepare(
    void* h_input_w_crc_addr,                      // 輸出: 含 CRC 的 DCI
    cuphyTensorDescriptor_t h_input_w_crc_desc,
    const void* h_input_addr,                      // 輸入: 原始 DCI
    cuphyTensorDescriptor_t h_input_desc,
    int num_coresets,
    int num_dcis,
    PdcchParams* params,                           // 輸出: 更新的參數
    cuphyPdcchDciPrm_t* h_dci_params,
    uint8_t* h_dci_tm_info,                        // 輸出: 測試模式標誌
    cuphyEncoderRateMatchMultiDCILaunchCfg_t* pEncdRMLaunchCfg,
    cuphyGenScramblingSeqLaunchCfg_t* pScrmSeqLaunchCfg,
    cuphyGenPdcchTfSgnlLaunchCfg_t* pTfSignalLaunchCfg,
    cudaStream_t stream
);
```

**主要操作**:

### 1. 推導 Coreset 參數
```cpp
// rb_coreset: 最右邊設置位的位置
params[coreset_idx].rb_coreset = 64 - find_rightmost_bit(freq_domain_resource);

// n_CCE: 邏輯 CCE 總數
params[coreset_idx].n_CCE = count_set_bits(freq_domain_resource) × n_sym;

// coreset_map: 頻域掩碼 (移位)
params[coreset_idx].coreset_map = freq_domain_resource >> (64 - rb_coreset);

// bundle_size: 非交錯模式下設為 6
if (!params[coreset_idx].interleaved) {
    params[coreset_idx].bundle_size = 6;
}
```

### 2. 為每個 DCI 計算 CRC
```cpp
for (int i = 0; i < num_dl_dci; i++) {
    uint32_t payload_crc = 0;
    
    if (!coreset_cell_in_testing_mode) {
        pdcchAddCrc(h_input_addr, payload_crc, rnti_crc, payload_bits);
    }
    
    // 比特反轉
    pdcchReverseBitInByte(h_input_addr, h_input_w_crc, nCrcOutByte, payload_bits);
    
    // 追加 CRC 比特 (反序)
    for (int j = 0; j < 24; j++) {
        uint8_t val = (payload_crc >> (24 - j - 1)) & 0x1;
        // 寫入適當的位置
        ...
    }
}
```

### 3. 設置內核參數
```cpp
kernelSelectEncodeRateMatchMultiDCIs(pEncdRMLaunchCfg, num_dcis);
kernelSelectGenScramblingSeq(pScrmSeqLaunchCfg, num_dcis);
kernelSelectGenTfSignal(pTfSignalLaunchCfg, num_dcis, num_coresets, params);
```

---

## 內核啟動配置

### `kernelSelectGenScramblingSeq()`

```cpp
void kernelSelectGenScramblingSeq(cuphyGenScramblingSeqLaunchCfg_t* pLaunchCfg,
                                  uint32_t num_DCIs)
{
    // 內核: genScramblingSeqKernel
    gridDim = (num_DCIs, 1, 1)
    blockDim = (64, 1, 1)
    sharedMemBytes = 0
}
```

### `kernelSelectGenTfSignal()`

```cpp
void kernelSelectGenTfSignal(cuphyGenPdcchTfSgnlLaunchCfg_t* pLaunchCfg,
                             uint32_t num_DCIs,
                             int num_coresets,
                             PdcchParams* h_coreset_params)
{
    // 內核: genPdcchTfSignalKernel<__half2>
    
    // 獲取最大 Coreset 大小
    uint32_t max_rb_coreset = max(rb_coreset for all coresets);
    uint32_t max_n_sym = max(n_sym for all coresets);
    
    gridDim = (max_n_sym, num_DCIs, 1)
    blockDim = (128, 1, 1)
    
    // 共享內存大小
    s_dmrs_seqs_size = sizeof(__half2) × (max_rb_coreset × 6 × 3)
    s_gold_seqs_size = sizeof(uint32_t) × (ceil(max_rb_coreset × 6 × 6 / 32) + 2)
    sharedMemBytes = s_dmrs_seqs_size + s_gold_seqs_size
}
```

---

## 整體流程圖

```
DCI 有效載荷 (主機)
    |
    v
計算 CRC (24 位)
    |
    v
CRC 用 RNTI 加擾
    |
    v
比特反轉 (逐字節)
    |
    v
追加 CRC 比特
    |
    v
DCI + CRC 緩衝區 (主機內存)
    |
    v [複製到設備]
    |
    v (設備)
編碼 + 速率匹配 (polar_encoder 內核)
    |
    v
加擾 DCI 比特 (genScramblingSeqKernel)
    |
    v
生成 Gold 序列 (動態)
    |
    v
QPSK 調制 + 功率縮放
    |
    v
映射到時頻網格 (genPdcchTfSignalKernel)
    ├─ DMRS 位置 (1, 5, 9 in RB)
    └─ QAM 位置 (0, 2-4, 6-8, 10-11 in RB)
    |
    v [可選: 波束賦型]
    |
    v
時頻信號輸出 (設備內存)
```

---

## 資源元素 (RE) 映射

### PDCCH RE 佈局 per RB

每個 RB 有 12 個子載波 × 1 OFDM 符號 = 12 RE

DMRS 位置: **1, 5, 9** (3 DMRS RE)
QAM 位置: **0, 2-4, 6-8, 10-11** (9 QAM RE)

```
RE Index:  0   1   2  3  4   5   6  7  8   9  10 11
          QAM DMRS QAM QAM QAM DMRS QAM QAM QAM DMRS QAM QAM
```

### Bundle 配置

| Bundle Size | n_sym | RB/Bundle | CCE/Bundle |
|------------|-------|-----------|-----------|
| 2 | 1 | 2 | 4 |
| 2 | 2 | 1 | 2 |
| 2 | 3 | 1 | 2 |
| 3 | 1 | 3 | 6 |
| 3 | 2 | 2 | 4 |
| 3 | 3 | 1 | 3 |
| 6 | 1 | 6 | 12 |
| 6 | 2 | 3 | 6 |
| 6 | 3 | 2 | 4 |

---

## 3GPP 標準映射

實現遵循以下 3GPP 規範：

| 功能 | 標準 | 說明 |
|------|------|------|
| PDCCH 結構 | TS 38.211 § 7.3 | Coreset、Bundle、RE 映射 |
| 編碼 | TS 38.212 § 5.2 | 極碼編碼 |
| 加擾 | TS 38.211 § 7.3.2 | Gold 序列加擾 |
| DMRS | TS 38.211 § 7.4 | Demodulation 參考信號 |
| 速率匹配 | TS 38.212 § 5.4 | 重複、穿孔 |
| 交錯 | TS 38.211 § 7.3.3 | Bundle 交錯配置 |

---

## 性能特性

### 並行化
- 每個 DCI × 符號組合 = 1 個 thread block
- 每個 thread = 1 個 RE
- 無 block 間依賴關係

### 共享內存優化
- DMRS LUT: 快速查詢
- Gold 序列: 批量計算
- Bundle 映射: 緩存

### 動態共享內存
- max_rb_coreset × 6 × 3 × sizeof(__half2) (DMRS)
- ceil(max_rb_coreset × 6 × 6 / 32) × sizeof(uint32_t) (Gold seq)

---

## 總結

PDCCH Embedding 組件實現了 5G NR 物理層下行控制信道的完整處理流程，從 DCI 有效載荷到時頻網格符號。通過高度優化的 CUDA 內核，它支持多個 Coreset、靈活的聚合等級、交錯映射和波束賦型，是 PDSCH 傳輸管道中不可或缺的部分。
