# NVIDIA cuPHY PDSCH DMRS (Demodulation Reference Signal)

**File Location**: `cuPHY/src/cuphy/pdsch_dmrs/`
**Main Files**: `pdsch_dmrs.hpp`, `pdsch_dmrs.cu`
**Purpose**: High-performance CUDA kernels for generating and embedding PDSCH demodulation reference signals (DMRS) and CSI-RS (Channel State Information Reference Signal) preprocessing in 5G NR downlink transmission

---

## 概述 (Overview)

PDSCH DMRS 組件負責在 5G NR 物理層下行傳輸中生成和映射解調參考信號。這些信號用於接收端的信道估計和解調。該實現支持：

- **Type-I DMRS** (密集型和稀疏型)
- **多層 MIMO** (最多 8 層)
- **波束賦型** (Precoding)
- **多個 Coreset 和 Cell**
- **CSI-RS 資源避免**

---

## 核心概念

### DMRS 型式

**Type-I DMRS** (3GPP TS 38.211 § 7.4):
- 密集型: CDM_grps = 2 (每個符號 6 DMRS RE per RB)
- 稀疏型: CDM_grps = 1 (每個符號 3 DMRS RE per RB)

### 資源配置

```
PDSCH 時頻資源
├─ Time: 1-14 個 OFDM 符號/時隙
│  ├─ DMRS 符號位置 (1-4 個)
│  └─ 資料符號位置 (其餘)
├─ Frequency: 連續 PRB (Resource Block)
│  ├─ Type 0: 位圖配置 (非連續)
│  └─ Type 1: 連續配置
└─ Layers: 1-8 層 (MIMO)
   └─ Port IDs: 1000-1011
```

### DMRS 埠和 RE 位置

**Type-I DMRS RE 位置** (per RB, 12 RE):
- **REG (Resource Element Group)**: 2 RE 相鄰
- **CDM (Code Division Multiplexing)**: 通過 OCC (Orthogonal Cover Code)

#### Port ID 編碼 (8 位)

```
Port ID = [tOCC | fOCC | delta]
          [2-3] [1]    [0]

- tOCC (Temporal OCC): 0 or 1
- fOCC (Frequency OCC): 0 or 1
- delta (Spacing): 0 or 1
```

---

## 主要數據結構

### `PdschDmrsParams` - DMRS 參數

```cpp
struct PdschDmrsParams {
    // 基本參數
    int ueGrp_idx;                      // UE 組索引
    uint32_t slot_number;               // 時隙編號
    int cell_id;                        // 物理 Cell ID
    
    // 時頻配置
    uint64_t data_sym_loc;              // 資料符號位置 (4 位 × 最多 14 個)
    uint16_t dmrs_sym_loc;              // DMRS 符號位置 (4 位 × 最多 4 個)
    uint8_t num_dmrs_symbols;           // DMRS 符號數
    uint8_t num_data_symbols;           // 資料符號數
    
    // DMRS 配置
    uint32_t dmrs_scid;                 // DMRS 加擾 ID (0-65535)
    uint8_t n_scid;                     // 加擾 Cell ID (0-1)
    uint8_t dmrsCdmGrpsNoData1;         // CDM = 1 標誌
    
    // 頻域配置
    uint32_t start_Rb;                  // 起始 RB
    uint32_t num_Rbs;                   // RB 數量
    uint8_t resourceAlloc;              // 資源配置型式 (0 or 1)
    uint32_t rbBitmap[MAX_RBMASK_UINT32_ELEMENTS];  // RB 位圖 (型式 0)
    uint32_t ref_point;                 // 參考點 (0 = CRB, 1 = PRB)
    uint32_t BWP_start_PRB;             // BWP 起始 PRB
    uint32_t num_BWP_PRBs;              // BWP PRB 數
    
    // 功率參數
    float beta_dmrs;                    // DMRS 功率縮放
    float beta_qam;                     // QAM 功率縮放
    
    // MIMO 配置
    uint32_t num_layers;                // 層數 (1-8)
    uint8_t port_ids[MAX_DL_LAYERS];    // 端口 ID (最多 8 層)
    uint16_t Np;                        // 天線端口數 (波束賦型用)
    __half2 pmW[MAX_DL_LAYERS * MAX_DL_PORTS];  // 波束矩陣
    
    // 波束賦型
    uint8_t enablePrcdBf;               // 波束賦型啟用標誌
    
    // 輸出配置
    void* cell_output_tensor_addr;      // 輸出時頻信號緩衝區
    uint32_t cell_index_in_cell_group;  // Cell 組內索引
    
    // DMRS 序列配置 (內部使用)
    uint32_t symbol_number;             // 符號開始位置
    
#ifdef ENABLE_32DL
    uint8_t nlAbove16;                  // 層數 > 16 的標誌
#endif
};
```

### `pdschDmrsDescr_t` - DMRS 描述符

```cpp
struct pdschDmrsDescr {
    PdschDmrsParams* dmrs_params;        // DMRS 參數陣列 (設備指針)
    int num_TBs;                         // 傳輸塊數量
};
typedef struct pdschDmrsDescr pdschDmrsDescr_t;
```

### `PdschUeGrpParams` - UE 組參數

```cpp
struct PdschUeGrpParams {
    int tb_idx;                                        // 傳輸塊索引
    uint32_t cumulative_skipped_REs[OFDM_SYMBOLS_PER_SLOT];  // 跳過的 RE 計數
};
```

---

## 演算法: DMRS 生成

### 步驟 1: c_init 計算

DMRS 金序列初始化值：

```cpp
uint32_t c_init = (
    (1 << 17) * (slot_number * OFDM_SYMBOLS_PER_SLOT + symbol_loc + 1) * (double_nid + 1) +
    double_nid + n_scid
) & 0x7FFFFFFFU;

其中:
- double_nid = dmrs_scid << 1
- symbol_loc = DMRS 符號位置 (0-基)
```

**3GPP 標準**: TS 38.211 Table 7.4.1.1.2-1

### 步驟 2: Gold 序列生成

通過 gold32() 函數生成偽隨機序列：

```cpp
uint32_t gold_seq = gold32(c_init, gold_index << 5);
// gold_index 以 32 位為單位
```

### 步驟 3: QPSK 調制

從金序列提取 2 位進行 QPSK 調制：

```cpp
uint32_t gold_value = ((gold_seq >> bit_offset) & 0x3U);

// 映射:
// gold_value = 0: (1, 1) / sqrt(2)
// gold_value = 1: (-1, 1) / sqrt(2)
// gold_value = 2: (1, -1) / sqrt(2)
// gold_value = 3: (-1, -1) / sqrt(2)

Tscalar positive_scramble_seq = 0.707106781186547f * beta_dmrs;

if (gold_value == 0)      -> (+1, +1)
else if (gold_value == 1) -> (-1, +1)
else if (gold_value == 2) -> (+1, -1)
else if (gold_value == 3) -> (-1, -1)
```

### 步驟 4: OCC (正交覆蓋碼) 應用

根據端口 ID 應用時頻正交覆蓋碼：

```cpp
// Port ID 結構: [tOCC | fOCC | delta]
uint8_t fOCC_flag = (port_idx & 0x1U);          // 頻率 OCC
uint8_t tOCC_flag = (port_idx >> 2) & 0x1U;     // 時間 OCC
uint8_t delta = (port_idx >> 1) & 0x1U;         // 間隔

// 頻率 OCC: fOCC_flag = 1 時，交替 RE 反轉
if ((fOCC_flag == 1) && ((tid_x & 0x1U) == 0x1U)) {
    symbol_to_read = -symbol_to_read;  // 反轉 I/Q
}

// 時間 OCC: tOCC_flag = 1 時，交替符號反轉
if (((symbol_id & 0x1) == 1) && (tOCC_flag == 1)) {
    symbol_to_read = -symbol_to_read;
}

// Delta: 確定寫入位置 (偶數或奇數 RE)
if (delta == 0) {
    symbol_to_write_0 = symbol_to_read;
    symbol_to_write_1 = 0;  // 奇數 RE 為 0
} else {
    symbol_to_write_0 = 0;
    symbol_to_write_1 = symbol_to_read;
}
```

---

## 主內核: `fused_dmrs<>()`

生成 PDSCH DMRS 並映射到時頻網格

```cpp
template<bool enablePrecoding, typename Tcomplex>
__global__ void fused_dmrs(pdschDmrsDescr_t* p_desc)
```

**執行模型**:
- gridDim.x = ceil(num_allocated_RB × 12 / 256)
- gridDim.y = num_TBs × num_layers
- blockDim.x = 256
- 每個線程: 處理 2 個 DMRS RE (配對)

### 內核邏輯

#### 步驟 1: Gold 序列預計算

```cpp
// 共享內存中計算 Gold 序列
for (int i = threadIdx.x; i < num_dmrs_symbols * gold_elements_one_dmrs; i += blockDim.x) {
    int dmrs_symbol = i >> 4;  // 每個 DMRS 符號 16 個元素 (256 / 16)
    int offset = i - dmrs_symbol * gold_elements_one_dmrs;
    
    uint32_t c_init = (
        (1 << 17) * (slot_number * 14 + symbol_loc[dmrs_symbol] + 1) * 
        (double_nid + 1) + double_nid + n_scid
    ) & 0x7FFFFFFFU;
    
    shmem_gold_seqs[i] = gold32(c_init, (blockIdx.x * 16 + offset) << 5);
}
__syncthreads();
```

#### 步驟 2: 確定啟用線程

資源配置型式 (Type 0 or 1):

```cpp
bool tid_enable = false;

if (resourceAlloc == 1) {
    // Type 1: 連續 RB
    tid_enable = (new_tidx < num_Rbs * 6) && (new_tidx >= 0);
} else {
    // Type 0: 位圖配置
    int tid_rb = ...;  // 計算 RB 索引
    if ((tid_rb < MAX_RBMASK_BYTE_SIZE*8) && (0 <= tid_rb)) {
        tid_enable = (0 != (rbBitmap[tid_rb >> 5] & (1L << (tid_rb & 0x1F))));
    }
}
```

#### 步驟 3: 處理每個 DMRS 符號

```cpp
if (tid_enable) {
    for (int symbol_id = 0; symbol_id < num_dmrs_symbols; symbol_id++) {
        // 從共享內存提取 2 位 Gold 序列
        int shmem_index = symbol_id * 16 + (threadIdx.x >> 4);
        int shmem_bit_offset = (threadIdx.x & 0xF) << 1;
        uint32_t gold_value = ((shmem_gold_seqs[shmem_index] >> shmem_bit_offset) & 0x3U);
        
        // QPSK 調制
        Tcomplex scrambled_val;
        if (gold_value == 0)      scrambled_val = (+A, +A);
        else if (gold_value == 1) scrambled_val = (-A, +A);
        else if (gold_value == 2) scrambled_val = (+A, -A);
        else                       scrambled_val = (-A, -A);
        
        // 處理每一層
        for (int layer_id = 0; layer_id < num_layers; ++layer_id) {
            Tcomplex symbol_to_read = scrambled_val;
            uint8_t port_idx = port_ids[layer_id];
            
            // 應用 OCC
            uint8_t fOCC_flag = (port_idx & 0x1U);
            uint8_t tOCC_flag = (port_idx >> 2) & 0x1U;
            uint8_t delta = (port_idx >> 1) & 0x1U;
            
            if ((fOCC_flag == 1) && ((tid_x & 0x1U) == 0x1U))
                symbol_to_read = -symbol_to_read;
            
            if (((symbol_id & 0x1) == 1) && (tOCC_flag == 1))
                symbol_to_read = -symbol_to_read;
            
            // 準備寫入值 (考慮 delta)
            Tcomplex symbol_to_write_0, symbol_to_write_1;
            if (delta == 0) {
                symbol_to_write_0 = symbol_to_read;
                symbol_to_write_1 = 0;
            } else {
                symbol_to_write_0 = 0;
                symbol_to_write_1 = symbol_to_read;
            }
            
            // 寫入時頻網格
            if (!enablePrecoding) {
                // 直接寫入
                int layer = port_idx + (n_scid << 3);
                uint32_t output_index = (12 * 273 * (14 * layer + symbol_loc[symbol_id])) +
                                       (12 * start_Rb) + (new_tidx << 1);
                
                if (!dmrsCdmGrpsNoData1) {
                    dmrs_output[output_index] = symbol_to_write_0;
                    dmrs_output[output_index + 1] = symbol_to_write_1;
                } else {
                    dmrs_output[output_index + delta] = symbol_to_read;
                }
            } else if (enablePrecoding) {
                if (!enablePrcdBf) {
                    // 原子操作累加
                    atomicAdd(&dmrs_output[output_index + delta], symbol_to_read);
                } else {
                    // 波束賦型: 累加波束矩陣乘積
                    for (int out_port = 0; out_port < Np; ++out_port) {
                        __half2 matCoeff = pmW[layer_id * Np + out_port];
                        dmrs_temp_out[out_port] = __hcmadd(symbol_to_write_*, 
                                                           matCoeff, 
                                                           dmrs_temp_out[out_port]);
                    }
                }
            }
        }
    }
}
```

---

## DMRS 參數更新: `cuphyUpdatePdschDmrsParams()`

在主機端從高階配置計算 DMRS 參數

```cpp
cuphyStatus_t CUPHYWINAPI cuphyUpdatePdschDmrsParams(
    PdschDmrsParams * h_dmrs_params,          // 輸出: DMRS 參數
    cuphyPdschDynPrms_t* dyn_params,          // 輸入: 動態參數
    const cuphyPdschStatPrms_t* static_params,// 輸入: 靜態參數
    PdschUeGrpParams* pdsch_ue_group_params   // 輸出: UE 組參數
)
```

### 主要計算

1. **符號位置編碼**
   ```cpp
   // 4 位編碼每個符號位置
   data_sym_loc |= ((symbol_index & 0xF) << (data_symbol * 4));
   dmrs_sym_loc |= ((symbol_index & 0xF) << (dmrs_symbol * 4));
   ```

2. **DMRS 功率計算**
   ```cpp
   h_dmrs_params[TB_id].beta_dmrs = sqrt(dmrs_cdm_grps) * ue->beta_dmrs;
   ```

3. **端口 ID 提取**
   ```cpp
   // 從位圖提取端口 ID
   uint32_t dmrs_ports_bitmask = ue->dmrsPortBmsk;
   for (int i = 0; i < num_layers; i++) {
       h_dmrs_params[TB_id].port_ids[i] = __builtin_ctz(dmrs_ports_bitmask);
       dmrs_ports_bitmask ^= (1 << port_ids[i]);
   }
   ```

4. **波束矩陣複製** (如果啟用)
   ```cpp
   memcpy(h_dmrs_params[TB_id].pmW,
          dyn_params->pCellGrpDynPrm->pPmwPrms[ue->pmwPrmIdx].matrix,
          sizeof(__half2) * num_layers * Np);
   ```

---

## CSI-RS 預處理內核

### 1. `genCsirsReMap()` - CSI-RS 資源映射

為每個 CSI-RS 參數生成資源元素位置映射

```cpp
__global__ void genCsirsReMap(pdschCsirsPrepDescr_t* p_desc)
```

**功能**:
- 根據 CSI-RS 型式計算資源元素 (RE) 位置
- 標記時頻網格中的 CSI-RS RE
- 每個線程映射一個 CSI-RS RE

**演算法**:
```
對於每個 CSI-RS 參數:
  1. 根據行編號查詢 CSI-RS RE 模式
  2. 計算 RB、kBar、lBar (基礎位置)
  3. 添加 kPrime、lPrime (偏移)
  4. 計算最終 k、l 位置
  5. 在 RE 映射中標記 (設為 1)
```

### 2. `postProcessCsirsReMap()` - CSI-RS RE 避免

後處理 CSI-RS 映射，計算跳過的 RE

```cpp
__global__ void postProcessCsirsReMap(pdschCsirsPrepDescr_t* p_desc)
```

**功能**:
- 對於每個資料符號，計算被 CSI-RS 占用的 RE 數
- 生成 RE 位置重新映射 (避免 CSI-RS RE)
- 累積每個 UE 組跳過的 RE 計數

**執行模型**:
- gridDim.x = num_ue_groups
- blockDim.x = 32 × OFDM_SYMBOLS_PER_SLOT
- 每個 warp (32 線程) 處理 1 個符號

**支持的資源配置**:
- **Type 0**: 位圖配置 - 處理非連續 RB
- **Type 1**: 連續配置 - 處理連續 RB

### 3. `zero_memset_kernel()` - 內存初始化

快速清零 RE 映射緩衝區

```cpp
__global__ void zero_memset_kernel(pdschCsirsPrepDescr_t* p_desc)
```

**最佳化**:
- 每線程寫 uint4 (16 字節)
- 處理對齐邊界
- 最後處理餘數字節

---

## 設置函數

### `cuphySetupPdschDmrs()`

配置 PDSCH DMRS 內核執行

```cpp
cuphyStatus_t CUPHYWINAPI cuphySetupPdschDmrs(
    cuphyPdschDmrsLaunchConfig_t pdschDmrsLaunchConfig,
    PdschDmrsParams * dmrs_params,
    int num_TBs,
    uint8_t enable_precoding,
    cuphyTensorDescriptor_t dmrs_output_desc,
    void* dmrs_output_addr,
    void* cpu_desc,
    void* gpu_desc,
    uint8_t enable_desc_async_copy,
    cudaStream_t strm
)
```

**內核選擇**:
- 根據 enable_precoding 選擇模板特殊化
- `fused_dmrs<false, __half2>()`: 無波束賦型
- `fused_dmrs<true, __half2>()`: 帶波束賦型

**啟動配置**:
```
threads = 256
blockDim = (256, 1, 1)
gridDim.x = ceil(max_Nf / 2 / 256)  其中 max_Nf = 273 × 12
gridDim.y = num_TBs
```

### `cuphySetupPdschCsirsPreprocessing()`

配置 CSI-RS 預處理內核

```cpp
cuphyStatus_t CUPHYWINAPI cuphySetupPdschCsirsPreprocessing(
    cuphyPdschCsirsPrepLaunchConfig_t pdschCsirsPrepLaunchConfig,
    void* re_map_array_addr,
    cuphyCsirsRrcDynPrm_t* d_params,
    size_t numParams,
    uint32_t total_offsets,
    uint32_t* d_offsets,
    uint32_t* d_cellIndex,
    uint16_t num_ue_groups,
    PdschUeGrpParams* d_ue_grp_params,
    PdschDmrsParams* d_dmrs_params,
    uint16_t max_BWP,
    uint16_t num_cells,
    void* cpu_desc,
    void* gpu_desc,
    uint8_t enable_desc_async_copy,
    cudaStream_t stream
)
```

**配置 3 個內核**:

1. **genCsirsReMap()**
   - blockDim = 128
   - gridDim.x = ceil(total_offsets / 128)

2. **postProcessCsirsReMap()**
   - blockDim = 32 × 14
   - gridDim.x = num_ue_groups

3. **zero_memset_kernel()**
   - blockDim = 1024
   - gridDim.x = ceil(buffer_size / (1024 × sizeof(uint4)))

---

## 整體流程圖

```
PDSCH 配置 (主機)
    |
    v
cuphyUpdatePdschDmrsParams()
    ├─ 計算符號位置
    ├─ 計算功率參數
    ├─ 提取端口 ID
    └─ 複製波束矩陣
    |
    v [複製參數到設備]
    |
    v (設備)
fused_dmrs<enablePrecoding>() 內核
    ├─ Gold 序列生成
    ├─ QPSK 調制
    ├─ OCC 應用
    ├─ 波束賦型 (可選)
    └─ 寫入時頻網格
    |
    v
時頻信號 + DMRS (設備內存)
    
CSI-RS 配置 (並行)
    |
    v
zero_memset_kernel()
    └─ 初始化 RE 映射緩衝區
    |
    v
genCsirsReMap()
    ├─ 根據 CSI-RS 型式生成位置
    ├─ 標記 CSI-RS RE
    └─ 按符號組織
    |
    v
postProcessCsirsReMap()
    ├─ 避免 CSI-RS RE
    ├─ 計算 RE 重新映射
    ├─ 累積跳過計數
    └─ 支持 Type 0/1 配置
    |
    v
RE 映射 + 跳過計數 (設備內存)
```

---

## 資源元素 (RE) 位置

### DMRS RE 映射

每個 PDSCH RB 有 12 個 RE (12 子載波 × 1 符號)

**Type-I DMRS** (CDM_grps = 2):
```
RE 位置:  0   1   2   3   4   5   6   7   8   9  10  11
         QAM DMRS QAM QAM QAM DMRS QAM QAM QAM DMRS QAM QAM
         
DMRS 位置: 1, 5, 9 (共 3 個 DMRS RE)
QAM 位置:  0, 2-4, 6-8, 10-11 (共 9 個 QAM RE)
```

**Type-I DMRS** (CDM_grps = 1 - 密集):
```
RE 位置:  0   1   2   3   4   5   6   7   8   9  10  11
         DMRS DMRS QAM QAM QAM DMRS DMRS QAM QAM QAM DMRS DMRS
         
DMRS 位置: 0-1, 5-6, 10-11 (共 6 個 DMRS RE)
QAM 位置:  2-4, 7-9 (共 6 個 QAM RE)
```

---

## 3GPP 標準映射

實現遵循以下 3GPP 規範：

| 功能 | 標準 | 說明 |
|------|------|------|
| DMRS 結構 | TS 38.211 § 7.4 | Type-I DMRS 定義 |
| c_init 計算 | TS 38.211 Table 7.4.1.1.2-1 | 金序列初始化 |
| Port ID | TS 38.211 § 7.4.1 | 端口編號方案 |
| OCC | TS 38.211 § 7.4.1 | 正交覆蓋碼 |
| CSI-RS | TS 38.211 § 7.4.2 | 信道狀態信息參考信號 |
| RE 避免 | TS 38.212 § 6.4 | 速率匹配中的 RE 跳過 |

---

## 性能特性

### 並行化策略

- **線程級**: 每線程 = 2 個 DMRS RE 對
- **Block 級**: 256 線程/block
- **Grid 級**: 多個 Block 並行處理不同 RB 和 TB

### 共享內存優化

- Gold 序列預計算存儲在共享內存
- 減少全局內存訪問
- warp 級 cooperative 操作

### 原子操作 (波束賦型模式)

- 用於多層累加
- 確保正確的浮點結果
- 性能開銷可接受 (DMRS 為稀疏 RE)

---

## 使用示例

### 基本 DMRS 生成

```cpp
// 1. 準備 DMRS 參數
PdschDmrsParams dmrs_params[num_TBs];

// 2. 更新參數
cuphyUpdatePdschDmrsParams(h_dmrs_params, dyn_params, static_params, ue_grp_params);

// 3. 複製到設備
cudaMemcpy(d_dmrs_params, h_dmrs_params, sizeof(PdschDmrsParams) * num_TBs, 
           cudaMemcpyHostToDevice);

// 4. 設置內核
cuphySetupPdschDmrs(&launch_cfg, d_dmrs_params, num_TBs, 
                     enable_precoding, ..., stream);

// 5. 啟動內核
cuLaunchKernel(launch_cfg.m_kernelNodeParams.func, ...);
```

### CSI-RS 預處理

```cpp
// 1. 初始化 RE 映射緩衝區
cudaMalloc(&re_map_buffer, buffer_size);

// 2. 設置 CSI-RS 預處理
cuphySetupPdschCsirsPreprocessing(
    &csirs_launch_cfg,
    re_map_buffer,
    d_csi_params,
    num_csi_params,
    ...
);

// 3. 執行 3 個內核
for (int i = 0; i < 3; i++) {
    cuLaunchKernel(csirs_launch_cfg.m_kernelNodeParams[i].func, ...);
}

// 4. RE 避免已自動應用於速率匹配
```

---

## 總結

PDSCH DMRS 組件實現了 5G NR 物理層下行解調參考信號的完整處理，包括生成、映射和波束賦型支持。通過高度優化的 CUDA 內核，它支持多層 MIMO、CSI-RS 資源避免和靈活的資源配置，是 PDSCH 傳輸管道中關鍵的組成部分。
