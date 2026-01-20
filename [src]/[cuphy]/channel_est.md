# Channel Estimation - MMSE 1D Time-Frequency (CHANNEL_EST)

## 概述

CHANNEL_EST是NVIDIA cuPHY庫中的**MMSE（最小均方誤差）一維時頻信道估計**專用組件。實現了高度優化的CUDA內核，用於在時間和頻域上同時進行信道插值估計。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - channel_est](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/channel_est)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `channel_est.hpp` | 公開API定義 |
| `channel_est.cu` | 高度優化的CUDA內核實現 |

---

## 公開API

### mmse_1D_time_frequency()

```cpp
namespace channel_est
{
void mmse_1D_time_frequency(tensor_pair&       tDst,
                            const_tensor_pair& tSymbols,
                            const_tensor_pair& tFreqFilters,
                            const_tensor_pair& tTimeFilters,
                            const_tensor_pair& tFreqIndices,
                            const_tensor_pair& tTimeIndices,
                            cudaStream_t       strm);
}
```

**功能**: 執行MMSE一維時頻信道估計

**參數**:
| 參數 | 類型 | 說明 |
|------|------|------|
| `tDst` | `tensor_pair&` | 輸出插值信道估計結果張量（4D: 頻率×時間×天線×UE） |
| `tSymbols` | `const_tensor_pair&` | 輸入DMRS符號張量（3D: 頻率×時間×天線） |
| `tFreqFilters` | `const_tensor_pair&` | 頻域濾波器係數（3D: 濾波器×PRB×UE） |
| `tTimeFilters` | `const_tensor_pair&` | 時域濾波器係數（3D: 濾波器×時間×UE） |
| `tFreqIndices` | `const_tensor_pair&` | 頻域DMRS索引（2D: PRB×UE） |
| `tTimeIndices` | `const_tensor_pair&` | 時域DMRS索引（2D: 時間位置×UE） |
| `strm` | `cudaStream_t` | CUDA流 |

**支持的數據類型**:
```cpp
case CUPHY_C_16F:   // 16位複數浮點
case CUPHY_C_32F:   // 32位複數浮點
```

---

## CUDA內核詳解

### mmse_1D_time_frequency_kernel

```cpp
template <typename TStorage,
          typename TCompute,
          int BLOCK_TILE_SIZE_X,    // 塊瓦片大小X（通常4-8）
          int BLOCK_TILE_SIZE_Y,    // 塊瓦片大小Y（通常4）
          int WARP_TILE_SIZE_X,     // Warp瓦片大小X（通常4）
          int WARP_TILE_SIZE_Y,     // Warp瓦片大小Y（通常8）
          int NFP,                   // 每個PRB的濾波器數（32）
          int NT>                    // 時域濾波器數（14）
__global__ void mmse_1D_time_frequency_kernel(
    tensor_ref<complex_storage_t, 4> tHinterp,           // 輸出：插值信道
    tensor_ref<const complex_storage_t, 3> tY,           // 輸入：DMRS符號
    tensor_ref<const int16_t, 2> tDMRS_index_freq,      // 頻域索引
    tensor_ref<const int16_t, 2> tDMRS_index_time,      // 時域索引
    tensor_ref<const TStorage, 3> tW_freq,              // 頻域濾波器
    tensor_ref<const TStorage, 3> tW_time,              // 時域濾波器
    int* debug);                                         // 調試計數器
```

**內核組織**:
- **Grid Dimension**: numUEs（每個UE一個線程塊）
- **Block Dimension**: BLOCK_TILE_SIZE_X × BLOCK_TILE_SIZE_Y × 32 線程

**計算流程**:

#### 1. 線程索引計算

```
THREAD_LANE = threadIdx.x % 32          // Warp內線程索引 [0-31]
WARP_IDX = threadIdx.x / 32             // 塊內Warp索引
THREAD_TILE_X/Y = 根據WARP內位置計算    // 線程瓦片位置
```

#### 2. 三重嵌套迴圈結構

**迴圈1: 天線塊**（NUM_ANT_BLK = NUM_ANT / BLOCK_TILE_SIZE_X）
- 加載Hp（DMRS符號）數據塊
- 預取下一個天線塊的數據

**迴圈2: 頻率塊**（NUM_FREQ_BLK = 3）
- 將頻域濾波器Wfreq加載到共享內存
- 執行矩陣乘法：C = Hp × Wfreq^T

**迴圈3: 時間濾波**
- 應用時域濾波器Wtime
- 使用Warp shuffle操作累加結果

#### 3. 矩陣乘法優化

```
計算格式: Hinterp = (Hp × Wfreq) × Wtime

Hp矩陣:
- 尺寸: BLOCK_SIZE_THREADS_X × 32（共享內存中轉置存儲）
- 來源: 接收符號張量

Wfreq矩陣:
- 尺寸: 32 × 96（NFP=32, 3個頻率塊×32）
- 來源: 頻域濾波器張量
- 存儲: 96×NFP共享內存

Wtime向量:
- 尺寸: 14（NT=14時間濾波器）
- 來源: 時域濾波器張量
- 存儲: 寄存器中
```

#### 4. Warp Shuffle操作

```cpp
// 沿Warp累加結果
Hinterp.x += __shfl_down_sync(0xFFFFFFFF, Hinterp.x, WARP_TILE_SIZE_Y * 2);
Hinterp.x += __shfl_down_sync(0xFFFFFFFF, Hinterp.x, WARP_TILE_SIZE_Y * 1);
```

使用Warp內洗牌操作高效地沿著warp方向進行數據匯聚。

#### 5. 輸出寫回

```cpp
if(0 == THREAD_TILE_X)  // 確保只有特定線程寫入
{
    tHinterp(freq, time, ant, ue) = type_convert<TComplexStorage>(Hinterp);
}
```

---

## 張量引用模板

### tensor_ref

```cpp
template <typename TElem, int NDim>
struct tensor_ref
{
    TElem* addr;              // 張量基地址
    int dim[NDim];            // 各維度大小
    int strides[NDim];        // 各維度步幅

    CUDA_BOTH tensor_ref(tensor_pair& tp);
    CUDA_BOTH tensor_ref(const_tensor_pair& tp);
    
    // 多維索引訪問（支持1-5維）
    CUDA_BOTH int offset(int i0) const;
    CUDA_BOTH int offset(int i0, int i1) const;
    CUDA_BOTH int offset(int i0, int i1, int i2) const;
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const;
    CUDA_BOTH int offset(int i0, int i1, int i2, int i3, int i4) const;
    
    // 操作符重載
    CUDA_BOTH TElem& operator()(int i0);
    CUDA_BOTH TElem& operator()(int i0, int i1);
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2);
    // ... 其他維度
};
```

### block_1D & block_2D

```cpp
// 1D塊（用於寄存器或共享內存存儲）
template <typename T, int M>
struct block_1D
{
    T data[M];
    CUDA_BOTH T& operator[](int idx) { return data[idx]; }
};

// 2D塊（用於共享內存2D矩陣存儲）
template <typename T, int M, int N>
struct block_2D
{
    T data[M * N];
    CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
};
```

---

## 數據加載優化

### load_1D - 1D數據加載

```cpp
template <typename Tdst, typename Tsrc>
CUDA_INLINE void load_1D(Tdst* dst, const Tsrc* src, int num)
{
    // 所有線程並行加載，每次跳過blockDim.x個元素
    for(int i = threadIdx.x; i < num; i += blockDim.x)
    {
        dst[i] = type_convert<Tdst>(src[i]);
    }
}
```

### load_2D_transpose - 2D轉置加載

```cpp
template <typename Tdst, typename Tsrc>
CUDA_INLINE void load_2D_transpose(Tdst* dst,
                                   const Tsrc* src,
                                   int srcM, int srcN,
                                   int srcStride)
{
    // 加載並在飛行中轉置矩陣
    // 源矩陣: srcM × srcN (步幅為srcStride)
    // 目標矩陣: srcN × srcM (緊密排列)
}
```

---

## 共享內存使用

### 典型配置（NFP=32, NT=14）

```cpp
__shared__ block_2D<TComplexCompute, 32, 128> Hp_block;              // 32×128 複數計算類型
__shared__ block_2D<TCompute, 96, NFP> Wfreq_ue_block;              // 96×32 實數存儲類型
__shared__ block_2D<TComplexStorage, 32, 56> Hinterp_out;           // 32×56 複數存儲類型
```

**共享內存大小計算**:
```
Hp_block: 32×128×8 bytes (複數) = 32KB
Wfreq_ue_block: 96×32×4 bytes (實數) = 12.3KB
Hinterp_out: 32×56×8 bytes (複數) = 14.3KB
總計: ~60KB（在1024執行緒內為典型值）
```

---

## 類型轉換和輔助函數

### type_convert<>

```cpp
template <typename Tdst, typename Tsrc>
Tdst type_convert(const Tsrc& src);
// 支持 int, float, complex, half 類型轉換
```

### 複數操作（make_complex）

```cpp
template <typename T>
struct make_complex;

// 示例: make_complex<cuComplex>::create(x, y);
```

### 調試函數

```cpp
// 打印共享內存塊
template <typename T>
__device__ void print_block(const T* t, const char* name, 
                            const char* fmt, int M, int N = 1);

// 轉儲整個共享內存區域
template <typename T>
__device__ void dump_shared_mem(const char* desc, T* shmem, int N);
```

---

## 性能特性

### 優化策略

| 策略 | 詳情 |
|------|------|
| **Warp瓦片化** | 將工作分為4×8的Warp瓦片，確保整個warp協作 |
| **共享內存緩衝** | 雙緩衝技術預取下一塊數據 |
| **Warp Shuffle** | 使用`__shfl_down_sync`替代全同步 |
| **寄存器優化** | 在寄存器中保持過濾器係數 |
| **協作加載** | 所有線程並行加載，減少等待 |
| **轉置存儲** | Hp矩陣轉置存儲以提高緩存局部性 |

### 計算特性

```
理論峰值:
- 每個UE的計算量: ~1130496 FLOPS（根據代碼註釋）
- 記憶體訪問: 高度優化的模式
- 共享內存帶寬: 充分利用
```

---

## 啟動配置

### 默認配置

```cpp
const int WARP_TILE_SIZE_X  = 4;
const int WARP_TILE_SIZE_Y  = 8;
const int BLOCK_TILE_SIZE_X = 4;
const int BLOCK_TILE_SIZE_Y = 4;
const int NFP               = 32;  // 頻域濾波器數
const int NT                = 14;  // 時域濾波器數

// 啟動配置
dim3 gridDim(numUEs);
dim3 blockDim(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y * 32);
// blockDim.x = 4 × 4 × 32 = 512 線程
```

### 計算的維度

```
BLOCK_SIZE_THREADS_X = 4 × 4 = 16
BLOCK_SIZE_THREADS_Y = 4 × 8 = 32
THREADS_PER_WARP = 32
LOAD_FREQS_PER_WARP = 32 / 16 = 2
```

---

## 執行流程

```
1. 啟動內核
   ├─ 每個UE一個線程塊
   └─ 512個線程/塊
   
2. 加載過濾器係數
   ├─ 時域過濾器→寄存器（14個係數）
   └─ 頻域過濾器→共享內存（按需）

3. 三重嵌套迴圈
   ├─ 天線塊迴圈
   │  └─ 預取下一天線塊
   ├─ 頻率塊迴圈
   │  ├─ 加載Wfreq到共享內存
   │  ├─ 同步等待
   │  └─ 矩陣乘法
   └─ 時間過濾
       ├─ Warp Shuffle累加
       └─ 寫入全局內存

4. 同步和完成
   ├─ 線程塊同步（__syncthreads）
   └─ 全局內存寫入完成
```

---

## 數據流

```
輸入:
  tSymbols (Y)      ← DMRS符號
  tFreqFilters      ← 頻域MMSE濾波器
  tTimeFilters      ← 時域MMSE濾波器
  tFreqIndices      ← 頻域DMRS位置
  tTimeIndices      ← 時域DMRS位置

處理:
  Y × Wfreq^T × Wtime = Hinterp

輸出:
  tDst (Hinterp)    ← 插值信道估計
```

---

## 調試支持

### 調試輸出

```cpp
// 條件編譯
#if CUPHY_DEBUG
    // 調試打印函數啟用
#endif

// 可選的調試計數器
int* debugInt = nullptr;  // 追踪讀/寫操作數
if(debug) atomicAdd(debug + 0, 2);  // 記錄讀操作
if(debug) atomicAdd(debug + 1, 2);  // 記錄寫操作
```

### 性能測量代碼

```cpp
#if 0  // 條件編譯
    // WARMUP運行
    // 性能迴圈（1000次反覆）
    // CUDA事件計時
    // 帶寬和GFLOPS計算
    
    // 性能指標:
    // BW = 1000.0f * 1.0e-9 * IO_BYTES_PER_ITER / elapsed_ms
    // GFLOPS = (1000.0 * 1.0e-9 * 1130496 * numUEs) / elapsed_ms
#endif
```

---

## 限制和注意事項

1. **固定尺寸**: 當前實現针對特定的濾波器大小進行了硬編碼優化
   - NFP = 32（頻域濾波器數）
   - NT = 14（時域濾波器數）

2. **數據類型**: 只支持複數浮點（16位或32位）

3. **對齐要求**: 張量步幅必須正確對齐以獲得最佳性能

4. **共享內存**: 總共使用約60KB共享內存

5. **線程數**: 預期使用512個線程/塊

---

## 延伸閱讀

- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- MMSE估計理論
- CUDA優化指南
