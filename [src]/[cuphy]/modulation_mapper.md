# NVIDIA cuPHY Modulation Mapper

**File Location**: `cuPHY/src/cuphy/modulation_mapper/`
**Main Files**: `modulation_mapper.hpp`, `modulation_mapper.cu`
**Purpose**: High-performance CUDA kernel for mapping binary bits to QAM constellation symbols in 5G PDSCH downlink transmission

---

## 概述 (Overview)

Modulation Mapper 是 5G 物理層下行傳輸鏈的關鍵組件，負責將編碼和速率匹配後的比特流映射到複數域的 QAM 正交點。該實現支持多種調制方案並針對 GPU 執行進行了高度優化。

**主要功能**:
- 支持多種 QAM 調制方案 (QPSK/QPSK-4QAM, 16QAM, 64QAM, 256QAM)
- 並行處理多個傳輸塊 (Transport Block)
- 支持靈活的內存佈局和步長張量
- 星座歸一化係數支持
- 硬件加速比特提取和正交映射

---

## 核心數據結構

### `modulationDescr_t`

調制映射描述符，包含所有必要參數

```cpp
struct modulationDescr {
    PdschDmrsParams*      d_params;              // 設備指針: PDSCH DMRS 參數
    const uint32_t*       modulation_input;      // 設備指針: 輸入比特流
    const PdschPerTbParams* workspace;           // 設備指針: 每個TB的參數
    __half2*              modulation_output;     // 設備指針: 輸出調制符號
    int                   max_bits_per_layer;    // 最大比特數/層
};
typedef struct modulationDescr modulationDescr_t;
```

**字段說明**:

| 字段 | 類型 | 說明 |
|------|------|------|
| `d_params` | PdschDmrsParams* | 指向 PDSCH DMRS 參數 (包含 beta_qam 歸一化因子) |
| `modulation_input` | const uint32_t* | 輸入比特，打包為 uint32_t 字 (1 字 = 32 位) |
| `workspace` | const PdschPerTbParams* | 每個 TB 的配置 (G、Qm、layer 計數) |
| `modulation_output` | __half2* | 輸出複數符號，使用 __half2 (FP16 複數對) |
| `max_bits_per_layer` | int | 單層最大比特數，用於索引計算 |

---

## QAM 正交點定義

### 常量內存中的 LUT (Look-Up Table)

系統在常量設備內存中存儲預計算的正交點幅度值：

#### QPSK 4-QAM

```cpp
// 通過 QAM_traits<2>::A = 1/sqrt(2) = 0.707106781186547 計算
// 符號: 00 -> (+A, +A), 01 -> (+A, -A), 10 -> (-A, +A), 11 -> (-A, -A)
```

**幅度計算**: $A = \frac{1}{\sqrt{2}}$

#### 16-QAM

```cpp
__device__ __constant__ float rev_qam_16[4] = {
    0.316227766,      // 1/sqrt(10) for inner constellation points
    -0.316227766,
    0.948683298,      // 3/sqrt(10) for outer constellation points
    -0.948683298,
};

__device__ __constant__ float rev_qam_16_long[8] = {
    0.316227766, -0.316227766,    // Inner level (1x)
    0.316227766, -0.316227766,    // Duplicate for I and Q
    0.948683298, -0.948683298,    // Outer level (3x)
    0.948683298, -0.948683298,    // Duplicate
};
```

**正交點**: 4x4 網格
- 內層 (2x2): ±0.316
- 外層 (2x2): ±0.949
- **幅度**: $A = \frac{1}{\sqrt{10}}$，外層點為 3A

#### 64-QAM

```cpp
__device__ __constant__ float rev_qam_64[8] = {
    0.462910049886276,       // 1/sqrt(10.67)
    -0.462910049886276,
    0.77151674981046,        // 3/sqrt(10.67)
    -0.77151674981046,
    0.154303349962092,       // 1/(6*sqrt(10.67))
    -0.154303349962092,
    1.08012344973464,        // 5/sqrt(10.67)
    -1.08012344973464,
};
```

**正交點**: 8x8 網格
- 7 種不同的幅度級別
- **幅度**: $A = \frac{1}{\sqrt{42}}$

#### 256-QAM

```cpp
__device__ __constant__ float rev_qam_256[16] = {
    0.383482494, -0.383482494,      // Levels ±1, ±3, ±5, ±7
    0.843661488, -0.843661488,
    0.230089497, -0.230089497,
    0.997054486, -0.997054486,
    // ... (8 more entries)
};
```

**正交點**: 16x16 網格
- 各層約 ±1, ±3, ±5, ±7、±9、±11、±13、±15
- **幅度**: $A = \frac{1}{\sqrt{170}}$

### 正交點歸一化

所有星座點使用 `beta_qam` 進行縮放：

```cpp
normalized_symbol = symbol * params[blockIdx.y].beta_qam
```

其中 `beta_qam` 在 PdschDmrsParams 中設置，用於功率控制。

---

## bit提取機制

### `extract_bits()` - 通用bit提取

從輸入bits中提取指定調制階的bit pair

```cpp
template <int TLOG2_QAM>
__device__ uint32_t extract_bits(const uint32_t* bits, int symbolIndex)
{
    const uint32_t MASK        = (1 << TLOG2_QAM) - 1;     // 比特掩碼
    const int      BIT_IDX     = symbolIndex * TLOG2_QAM;  // 位位置
    const int      WORD_IDX    = BIT_IDX / BITS_PER_WORD;  // uint32_t 字索引
    const int      WORD_OFFSET = BIT_IDX % BITS_PER_WORD;  // 字內偏移
    
    uint32_t value = ((bits[WORD_IDX] >> WORD_OFFSET) & MASK);
    return value;
}
```

**參數**:
- `TLOG2_QAM`: 調制階的對數 (1=BPSK, 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM)
- `bits`: 輸入比特緩衝區 (uint32_t 字)
- `symbolIndex`: 符號索引 (第幾個調制符號)

**返回**: 用於該符號的 TLOG2_QAM 比特

### `extract_bits<6>()` - 64-QAM 特殊化

64-QAM 使用 6 位/符號，某些符號跨越 32 位字邊界。特殊化版本使用漏斗移位 (funnel shift) 進行高效跨邊界提取：

```cpp
template <>
__device__ uint32_t extract_bits<6>(const uint32_t* bits, int symbolIndex)
{
    const int      LOG2_QAM    = 6;
    const uint32_t MASK        = (1 << LOG2_QAM) - 1;
    const int      BIT_IDX     = symbolIndex * LOG2_QAM;
    const int      WORD_IDX    = BIT_IDX / 32;
    const int      WORD_OFFSET = BIT_IDX % 32;
    
    // 使用漏斗移位實現跨邊界提取
    uint32_t value = MASK & __funnelshift_r(bits[WORD_IDX],
                                            bits[WORD_IDX + 1],
                                            WORD_OFFSET);
    return value;
}
```

**邊界情況** (每 16 個符號出現):
- 符號 5: bit_offset = 30，需要從 2 個字讀取
- 符號 10: bit_offset = 28，需要從 2 個字讀取

---

## 正交映射函數

### `bits_to_symbol<>()` - 模板化映射

通用的比特-到-符號映射器，根據 QAM 階級特殊化

#### BPSK (1 位)

```cpp
template <typename TOut> struct bits_to_symbol<1, TOut>
{
    __device__ static void map(TOut* dst, uint32_t bits, const mod_table_none&)
    {
        *dst = make_complex<TOut>::create(
            (0 == bits) ? QAM_traits<1>::A : -QAM_traits<1>::A,
            (0 == bits) ? QAM_traits<1>::A : -QAM_traits<1>::A
        );
    }
};
```

映射:
- Bit 0 → (+1, +1) × A / √2
- Bit 1 → (-1, -1) × A / √2

#### QPSK (2 位)

```cpp
template <typename TOut> struct bits_to_symbol<2, TOut>
{
    __device__ static void map(TOut* dst, uint32_t bits, const mod_table_none&)
    {
        *dst = make_complex<TOut>::create(
            (0 == (bits & 0x1)) ? QAM_traits<2>::A : -QAM_traits<2>::A,
            (0 == (bits & 0x2)) ? QAM_traits<2>::A : -QAM_traits<2>::A
        );
    }
};
```

映射:
- Bits [1:0]: I_bit=bit[0], Q_bit=bit[1]
- 4 種正交點

#### 16-QAM (4 位)

```cpp
template <typename TOut> struct bits_to_symbol<4, TOut>
{
    __device__ static void map(TOut* dst, uint32_t bits, const mod_table_QAM16<TOut>& tbl)
    {
        *dst = make_complex<TOut>::create(
            tbl.qam16_values[bits & 0x05],           // 提取 bits[2:0] 中的 I 比特
            tbl.qam16_values[(bits >> 1) & 0x05]     // 提取 bits[3:1] 中的 Q 比特
        );
    }
};
```

**比特分配**:
- I 軸: bits [0, 2]
- Q 軸: bits [1, 3]
- 索引計算使用掩碼提取交替比特

#### 64-QAM (6 位)

```cpp
template <typename TOut> struct bits_to_symbol<6, TOut>
{
    __device__ static void map(TOut* dst, uint32_t bits, const mod_table_QAM64<TOut>& tbl)
    {
        int x_index = map_index_6bits(bits);
        int y_index = map_index_6bits(bits >> 1);
        
        *dst = make_complex<TOut>::create(
            tbl.qam64_values[x_index],
            tbl.qam64_values[y_index]
        );
    }
};
```

**比特分配**:
- I 軸: map_index_6bits(bits[5:0])
- Q 軸: map_index_6bits(bits[5:1])

#### 256-QAM (8 位)

```cpp
template <typename TOut> struct bits_to_symbol<8, TOut>
{
    __device__ static void map(TOut* dst, uint32_t bits, const mod_table_QAM256<TOut>& tbl)
    {
        int x_index = map_index_8bits(bits);
        int y_index = map_index_8bits(bits >> 1);
        
        *dst = make_complex<TOut>::create(
            tbl.qam256_values[x_index],
            tbl.qam256_values[y_index]
        );
    }
};
```

---

## 索引映射函數

### `map_index_6bits()` - 64-QAM I/Q 映射

```cpp
__device__ __inline__ uint32_t map_index_6bits(uint32_t index) {
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                            ((index & 0x10) >> 2);
    return masked_index;  // 返回 3 位索引 [0-7]
}
```

**比特重新排列**:
- 輸入 6 位: [b5 b4 b3 b2 b1 b0]
- 輸出 3 位: [b4 b2 b0]

### `map_index_8bits()` - 256-QAM I/Q 映射

```cpp
__device__ __inline__ uint32_t map_index_8bits(uint32_t index) {
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                            ((index & 0x10) >> 2) | ((index & 0x40) >> 3);
    return masked_index;  // 返回 4 位索引 [0-15]
}
```

**比特重新排列**:
- 輸入 8 位: [b7 b6 b5 b4 b3 b2 b1 b0]
- 輸出 4 位: [b6 b4 b2 b0]

---

## 調制映射內核

### 簡化內核: `modulation_mapper()`

為 PDSCH 傳輸鏈設計的快速路徑內核

```cpp
__global__ void modulation_mapper(modulationDescr_t* p_desc) {
    modulationDescr_t& desc = *p_desc;
    const PdschDmrsParams* params = desc.d_params;
    const uint32_t* modulation_input = desc.modulation_input;
    const struct PdschPerTbParams* workspace = desc.workspace;
    __half2* modulation_output = desc.modulation_output;
    int max_bits_per_layer = desc.max_bits_per_layer;
    
    if (modulation_input == nullptr) return;
    
    int TB_id = blockIdx.y;
    int modulation_order = workspace[TB_id].Qm;
    
    // 根據調制階選擇映射函數
    if (modulation_order == CUPHY_QAM_4) {
        modulation_QPSK(params, modulation_input, modulation_output, 
                       workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_16) {
        modulation_16QAM(params, modulation_input, modulation_output,
                        workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_64) {
        modulation_64QAM(params, modulation_input, modulation_output,
                        workspace, max_bits_per_layer);
    } else if (modulation_order == CUPHY_QAM_256) {
        modulation_256QAM(params, modulation_input, modulation_output,
                         workspace, max_bits_per_layer);
    }
}
```

**執行模型**:
- gridDimX: 符號數量/256
- gridDimY: 傳輸塊數量 (TB_id in blockIdx.y)
- blockDim: (256, 1, 1)
- 每個線程處理 1 個符號

#### QAM 特定實現

##### `modulation_QPSK()`

```cpp
__device__ void modulation_QPSK(const PdschDmrsParams * __restrict__ params,
                                const uint32_t* __restrict__ modulation_input,
                                __half2 * __restrict__ modulation_output,
                                const struct PdschPerTbParams * workspace,
                                int max_bits_per_layer) {
    float reciprocal_sqrt2 = 0.707106781186547f * params[blockIdx.y].beta_qam;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = (workspace[blockIdx.y].G >> 1);  // G/2
    
    if (tid >= num_symbols) return;
    
    int output_index = tid;
    uint32_t bit_values;
    
    if (params[blockIdx.y].num_Rbs != 0) {
        // 完整 PDSCH 配置: 計算資源元素位置
        output_index = output_index_calc(...);
        bit_values = input_index_calc<CUPHY_QAM_4>(...);
    } else {
        // 簡化模式: 連續符號
        bit_values = flat_input_index_calc_4QAM(modulation_input);
    }
    
    __half2 tmp_val;
    tmp_val.x = ((bit_values & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
    tmp_val.y = ((bit_values & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
    
    modulation_output[output_index] = tmp_val;
}
```

**關鍵優化**:
- 直接計算 QPSK 星座 (無 LUT)
- FP32 計算然後轉換為 FP16
- 自動向量化為 __half2 (複數對)

##### `modulation_64QAM()` 和 `modulation_256QAM()`

使用共享內存 LUT 加速星座查詢：

```cpp
__device__ void modulation_64QAM(const PdschDmrsParams * __restrict__ params,
                                 const uint32_t* __restrict__ modulation_input,
                                 __half2 * __restrict__ modulation_output,
                                 const struct PdschPerTbParams * workspace,
                                 int max_bits_per_layer) {
    // 從常量內存加載到共享內存
    __shared__ __half shmem_qam_64[8];
    if (threadIdx.x < 8) {
        shmem_qam_64[threadIdx.x] = (__half)(rev_qam_64[threadIdx.x] * 
                                             params[blockIdx.y].beta_qam);
    }
    __syncthreads();
    
    // 線程級計算
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_symbols = (workspace[blockIdx.y].G / CUPHY_QAM_64);
    
    if (tid >= num_symbols) return;
    
    uint32_t bit_values = ...;  // 提取 6 位
    int x_index = map_index_6bits(bit_values);
    int y_index = map_index_6bits(bit_values >> 1);
    
    modulation_output[output_index] = make_complex<__half2>::create(
        shmem_qam_64[x_index],
        shmem_qam_64[y_index]
    );
}
```

---

## 平坦輸入索引計算

用於簡化模式（無 DMRS 參數）的高效比特索引函數

### `flat_input_index_calc_4QAM()`

```cpp
__device__ uint32_t flat_input_index_calc_4QAM(const uint32_t* __restrict__ modulation_input) {
    // 針對 blockDim.x=256 優化的硬編碼
    int input_index = (blockIdx.x << 4) + (threadIdx.x >> 4);  // 每個線程 2 位
    int symbol_start_bit = ((threadIdx.x & 0xF) << 1);
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}
```

計算邏輯:
- 每 4 個符號 = 1 個 uint32_t 字
- 256 個線程，每個線程 2 位
- blockIdx.x << 4 = blockIdx.x * 16 個字 / CTA

### `flat_input_index_calc_16QAM()`

```cpp
__device__ uint32_t flat_input_index_calc_16QAM(const uint32_t* __restrict__ modulation_input) {
    int input_index = (blockIdx.x << 5) + (threadIdx.x >> 3);
    int symbol_start_bit = ((threadIdx.x & 0x7) << 2);  // 每個線程 4 位
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}
```

### `flat_input_index_calc_64QAM()`

```cpp
__device__ uint32_t flat_input_index_calc_64QAM(const uint32_t* __restrict__ modulation_input) {
    const int element_size = sizeof(uint32_t) * 8;
    int input_index = blockIdx.x * blockDim.x * CUPHY_QAM_64 / element_size + 
                     threadIdx.x * CUPHY_QAM_64 / element_size;
    int symbol_start_bit = (threadIdx.x * CUPHY_QAM_64) % element_size;
    
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    
    // 處理跨邊界情況 (6 位符號在 32 位字邊界上)
    if (symbol_start_bit == 28) {  // 讀 2 位
        bit_values &= 0x0FU;
        bit_values |= ((modulation_input[input_index + 1] & 0x03U) << 4);
    } else if (symbol_start_bit == 30) {  // 讀 4 位
        bit_values &= 0x03U;
        bit_values |= ((modulation_input[input_index + 1] & 0x0FU) << 2);
    }
    return bit_values;
}
```

### `flat_input_index_calc_256QAM()`

```cpp
__device__ uint32_t flat_input_index_calc_256QAM(const uint32_t* __restrict__ modulation_input) {
    int input_index = (blockIdx.x << 6) + (threadIdx.x >> 2);
    int symbol_start_bit = ((threadIdx.x & 0x3) << 3);  // 每個線程 8 位
    uint32_t bit_values = (modulation_input[input_index] >> symbol_start_bit);
    return bit_values;
}
```

---

## 設置函數: `cuphySetupModulation()`

配置並準備調制映射內核執行

```cpp
cuphyStatus_t CUPHYWINAPI cuphySetupModulation(
    cuphyModulationLaunchConfig_t modulationLaunchConfig,
    PdschDmrsParams * d_params,
    const cuphyTensorDescriptor_t input_desc,       // 未使用
    const void* modulation_input,
    int max_num_symbols,
    int max_bits_per_layer,
    int num_TBs,
    PdschPerTbParams* workspace,                    // G、Qm 字段
    cuphyTensorDescriptor_t output_desc,            // 未使用
    void* modulation_output,
    void* cpu_desc,                                 // 主機描述符
    void* gpu_desc,                                 // 設備描述符
    uint8_t enable_desc_async_copy,
    cudaStream_t strm
);
```

**參數說明**:

| 參數 | 說明 |
|------|------|
| `modulationLaunchConfig` | 內核啟動配置 (輸出) |
| `d_params` | PDSCH DMRS 參數 (包含 beta_qam) |
| `modulation_input` | 輸入比特流 |
| `max_num_symbols` | 最大符號數/CTA |
| `max_bits_per_layer` | 單層最大比特 |
| `num_TBs` | 傳輸塊數量 |
| `workspace` | 每個 TB 的 G (速率匹配比特) 和 Qm (調制階) |
| `modulation_output` | 輸出調制符號 |
| `cpu_desc` / `gpu_desc` | 主機/設備描述符 (已預分配) |
| `enable_desc_async_copy` | 1 = 異步複製到設備，0 = 同步 |
| `strm` | CUDA 流 |

**內部流程**:

```
1. 填充 CPU 描述符
   ├─ d_params (設備指針)
   ├─ modulation_input (設備指針)
   ├─ workspace (設備指針)
   ├─ modulation_output (設備指針)
   └─ max_bits_per_layer

2. 可選的異步描述符複製到 GPU

3. 獲取內核函數指針
   └─ 通過 cudaGetFuncBySymbol() 查詢 modulation_mapper

4. 配置執行參數
   ├─ blockDimX = min(256, max_num_symbols)
   ├─ blockDimY/Z = 1
   ├─ gridDimX = ceil(max_num_symbols / blockDimX)
   ├─ gridDimY = num_TBs
   ├─ gridDimZ = 1
   └─ sharedMemBytes = 0 (或 LUT 大小)

5. 返回成功
```

**返回值**: `CUPHY_STATUS_SUCCESS` 或錯誤代碼

---

## 通用 Utility 內核: `sym_mod_util<>()`

用於離線或示例程序的簡化符號調制

```cpp
template <unsigned int THREADS_PER_CTA, typename TOut>
__global__ void sym_mod_util(
    const tensor_layout_any tDstLayout,
    TOut*                   dst,
    const tensor_layout_any tSrcLayout,
    const uint32_t*         src,
    int                     log2_QAM,
    int                     symbolsPerBatch
);
```

**功能**:
- 通用 QAM 調制 (BPSK/QPSK/16QAM/64QAM/256QAM)
- 支持任意張量佈局
- 共享內存中的星座 LUT
- 批處理模式以提高緩存效率

**執行配置**:
- gridDim.x = NUM_COLUMNS (張量列數)
- blockDim.x = THREADS_PER_CTA (通常 32)

**內核邏輯**:

```cpp
// 1. 初始化共享內存 LUT
__shared__ mod_table_QAM16<TOut>  tbl_QAM16;
__shared__ mod_table_QAM64<TOut>  tbl_QAM64;
__shared__ mod_table_QAM256<TOut> tbl_QAM256;

tbl_QAM16.init();   // 線程 0-7 從常量內存加載
tbl_QAM64.init();   // 線程 0-7 從常量內存加載
tbl_QAM256.init();  // 線程 0-15 從常量內存加載
__syncthreads();

// 2. 批量讀取比特到共享內存
__shared__ uint32_t block_src[THREADS_PER_CTA * WORDS_PER_THREAD + 1];
block_copy_N_sync_check<uint32_t, WORDS_PER_THREAD>(
    block_src, blockAddr, SRC_COL_END
);

// 3. 根據 log2_QAM 分派到適當的調制函數
switch(log2_QAM) {
    case 2:  mod_QPSK_t::modulate (...);  break;
    case 4:  mod_QAM16_t::modulate (...); break;
    case 6:  mod_QAM64_t::modulate (...); break;
    case 8:  mod_QAM256_t::modulate(...); break;
}

// 4. 批次循環直到列完成
```

---

## `symbol_modulate()` C++ Wrapper

簡化的 C++ 接口

```cpp
namespace cuphy_i {

cuphyStatus_t symbol_modulate(
    const tensor_desc& tSym,        // 輸出符號張量
    void*              pSym,
    const tensor_desc& tBits,       // 輸入比特張量 (CUPHY_BIT)
    const void*        pBits,
    int                log2_QAM,    // log2(調制階)
    cudaStream_t       strm
);

}  // namespace cuphy_i
```

**功能**:
- 自動張量佈局轉換
- 類型檢查 (支持 CUPHY_C_32F 和 CUPHY_C_16F 輸出)
- 內核啟動和錯誤檢查

---

## 內存佈局和訪問模式

### 輸入比特佈局

比特以 `uint32_t` 字打包：
- 字 0: bits [0:31]
- 字 1: bits [32:63]
- 每個符號占用 log2_QAM 位

**範例 (64-QAM, 6 位/符號)**:
```
Word 0: [S5 S5 S5 S4 S4 S4 S3 S3 S3 S2 S2 S2 S1 S1 S1 S0 S0 S0]
        [63-58] [57-52] [51-46] [45-40] [39-34] [33-28] [27-22] [21-16] [15-10] [9-4]  [3:0 跨越 Word1]
```

### 輸出符號佈局

對於簡化模式 (num_Rbs == 0):
- 符號按順序存儲為 `__half2` (複數 FP16 對)
- 連續內存，無步長

對於完整 PDSCH 模式:
- 三維張量: [273 × 12 × 14 × num_layers × num_TBs]
  - 273 × 12 = 3276 (一整個帶寬的資源元素)
  - 14 = OFDM 符號/時隙
  - num_layers = MIMO 層
  - num_TBs = 傳輸塊

---

## 性能特性

### 優化策略

1. **常量內存 LUT**
   - 星座點存儲在常量內存中 (L1 緩存)
   - 自動廣播到所有線程

2. **共享內存優化**
   - 星座 LUT 複製到共享內存 (更快訪問)
   - 批量比特加載到共享內存

3. **線程級並行化**
   - 每個線程 = 1 個符號
   - 無線程間同步 (除 __syncthreads)

4. **比特提取優化**
   - 硬編碼的索引計算 (特定 blockDim.x = 256)
   - 64-QAM 特殊化 (漏斗移位)

5. **FP16 向量化**
   - __half2 (複數對) 在單指令中處理
   - 減少內存帶寬需求

### 典型吞吐量

- **QPSK**: 256 符號/CTA, 1 次迭代
- **16-QAM**: 256 符號/CTA
- **64-QAM**: 256 符號/CTA (需要 2 字邊界檢查)
- **256-QAM**: 256 符號/CTA

---

## 3GPP 標準映射

實現遵循以下 3GPP 規範：

| 調制方案 | 3GPP 標準 | 星座 | 歸一化 |
|---------|---------|------|--------|
| QPSK | TS 38.211 § 5.1.3 | 4 點 | 1/√2 |
| 16-QAM | TS 38.211 § 5.1.4 | 16 點 | 1/√10 |
| 64-QAM | TS 38.211 § 5.1.5 | 64 點 | 1/√42 |
| 256-QAM | TS 38.211 § 5.1.6 | 256 點 | 1/√170 |

歸一化係數 beta_qam 在 PDSCH 信道估計期間計算，用於功率控制。

---

## 使用示例

### 簡化模式 (無 DMRS)

```cpp
// 準備輸入比特 (已打包為 uint32_t 字)
uint32_t* d_bits = ...;           // 設備內存

// 準備輸出符號緩衝區
__half2* d_symbols = ...;          // 設備內存

// 配置參數
int num_symbols = 1024;
int num_bits = num_symbols * 6;    // 64-QAM: 6 位/符號
int num_words = div_round_up(num_bits, 32);

// 設置描述符
modulationDescr_t cpu_desc = {};
gpu_desc = nullptr;  // 分配的指針
cpu_desc.d_params = nullptr;       // 簡化模式
cpu_desc.modulation_input = d_bits;
cpu_desc.modulation_output = d_symbols;
cpu_desc.max_bits_per_layer = num_bits;
cpu_desc.workspace = &tb_params;

// 複製到設備
cudaMemcpy(d_gpu_desc, &cpu_desc, sizeof(modulationDescr_t), cudaMemcpyHostToDevice);

// 啟動內核
modulation_mapper<<<grid, block>>>(d_gpu_desc);
```

### 完整 PDSCH 模式 (帶 DMRS)

```cpp
// 配置 PDSCH 參數
PdschDmrsParams dmrs_params = {
    .num_Rbs = 50,
    .start_Rb = 0,
    .num_data_symbols = 12,
    .num_dmrs_symbols = 2,
    .symbol_number = 2,
    .beta_qam = 1.0f,               // 功率縮放
    .num_layers = 2,
    .port_ids = {0, 1},
    .n_scid = 0
};

// 工作空間
PdschPerTbParams tb_params = {
    .G = 5000,                      // 速率匹配比特數
    .Qm = CUPHY_QAM_64              // 64-QAM
};

// 設置
cuphyModulationLaunchConfig_t launch_config;
cuphySetupModulation(
    &launch_config,
    d_dmrs_params,
    tensor_desc_input,      // 未使用
    d_bits,
    max_num_symbols,        // 256
    max_bits_per_layer,     // 5000
    1,                      // 1 TB
    d_tb_params,
    tensor_desc_output,     // 未使用
    d_symbols,
    &cpu_desc, d_gpu_desc,
    1,                      // 異步複製
    stream
);

// 啟動
cuLaunchKernel(
    launch_config.m_kernelNodeParams.func,
    launch_config.m_kernelNodeParams.gridDimX,
    launch_config.m_kernelNodeParams.gridDimY,
    launch_config.m_kernelNodeParams.gridDimZ,
    launch_config.m_kernelNodeParams.blockDimX,
    launch_config.m_kernelNodeParams.blockDimY,
    launch_config.m_kernelNodeParams.blockDimZ,
    launch_config.m_kernelNodeParams.sharedMemBytes,
    stream,
    launch_config.m_kernelNodeParams.kernelParams,
    launch_config.m_kernelNodeParams.extra
);
```

---

## 總結

Modulation Mapper 是一個高度優化的 CUDA 內核，實現了 5G NR 物理層調制映射的所有必要功能。通過利用 GPU 的并行性、常量內存優化和高效的比特提取算法，它實現了高吞吐量和低延遲的調制性能。支持從簡化的獨立模式到完整的 PDSCH 集成管道的多種執行模式。
