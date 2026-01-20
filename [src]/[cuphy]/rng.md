# NVIDIA cuPHY 隨機數生成 (Random Number Generator) 模組

## 概述

隨機數生成 (RNG) 模組是 NVIDIA Aerial cuPHY 的核心工具組件，負責在 GPU 上高效地生成各種分佈的隨機數。該模組支持正態分佈 (Normal Distribution)、均勻分佈 (Uniform Distribution) 和隨機比特生成，適用於信號處理、擾碼生成和測試環境中的噪聲模擬等應用。

**主要功能：**
- 正態分佈隨機數生成（平均值和標準差可配置）
- 均勻分佈隨機數生成（範圍可配置）
- 隨機比特序列生成
- 多數據類型支持（FP16, FP32, FP64, complex）
- 多維張量支持（最多 5 維）

---

## 架構概述

### 核心組件

#### 1. **RNG 主類** - `cuphy_i::rng`

```cpp
namespace cuphy_i {
    class rng : public cuphyRNG {
    public:
        // 構造函數
        rng(unsigned long long seed, cudaStream_t s);
        
        // 生成正態分佈隨機數
        cuphyStatus_t normal(const tensor_desc& t,
                            void* p,
                            const cuphyVariant_t& mean,
                            const cuphyVariant_t& stddev,
                            cudaStream_t strm);
        
        // 生成均勻分佈隨機數
        cuphyStatus_t uniform(const tensor_desc& t,
                             void* p,
                             const cuphyVariant_t& min_v,
                             const cuphyVariant_t& max_v,
                             cudaStream_t strm);
    
    private:
        // 隨機狀態儲存在設備內存中
        cuphy_i::unique_device_ptr<curandState> randStates_;
    };
}
```

#### 2. **Python 包裝類** - `rng`

```cpp
class rng {
public:
    // 構造函數
    rng(unsigned long long seed = 0, 
        unsigned int flags = 0, 
        cudaStream_t strm = 0);
    
    // 生成正態分佈
    template <typename T, typename TVal>
    void normal(T& t, TVal mean, TVal stddev, cudaStream_t strm = 0);
    
    // 生成均勻分佈
    template <typename T, typename TVal>
    void uniform(T& t, TVal min_val, TVal max_val, cudaStream_t strm = 0);

private:
    unique_rng_ptr rng_;
};
```

---

## 資料結構

### 隨機狀態

```cpp
// cuRAND 隨機狀態結構（由 CUDA 提供）
typedef struct {
    uint32_t d[5];      // 內部狀態
    int boxmuller_flag;
    float boxmuller_gauss;
} curandState_t;
```

### 配置參數

```cpp
struct RNGConfiguration {
    unsigned long long seed;     // 種子值
    cudaStream_t stream;         // CUDA 流
    unsigned int flags;          // 配置標誌
    int numThreads = 1024;       // 每塊線程數
};
```

---

## CUDA 核心函數

### 初始化核心

```cpp
// 初始化每個線程的隨機狀態
__global__ void cuphy_rng_init(unsigned long long seed,
                               curandState* s) {
    // 每個線程都獲得相同的種子，但不同的序列號
    curand_init(seed,          // 種子
                threadIdx.x,   // 序列號
                0,             // 偏移
                s + threadIdx.x); // 狀態地址
}
```

### 張量隨機數生成核心

```cpp
// 張量隨機數生成模板核心
template <typename TOut, typename TRand, 
          template <typename> class TGenerator>
__global__ void tensor_rng_kernel(curandState* s,
                                  tensor_layout_any layoutDst,
                                  TOut* dst,
                                  TRand scale,
                                  TRand offset) {
    // 1. 從全局內存檢索隨機狀態
    unsigned int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    curandState randState = s[idx];
    
    // 2. 定義類型特性
    typedef TGenerator<TRand> generator_t;        // 生成器
    typedef rng_value_adjust<TRand> adjustor_t;   // 調整器
    typedef rng_value_cast<TOut> cast_t;          // 類型轉換
    
    // 3. 迴圈遍歷目標張量（最多 5 維）
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4) {
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3) {
            for(grid_stride_index<2> it2; 
                it2 < layoutDst.dimensions[2]; 
                it2.next()) {
                for(grid_stride_index<1> it1; 
                    it1 < layoutDst.dimensions[1]; 
                    it1.next()) {
                    for(grid_stride_index<0> it0; 
                        it0 < layoutDst.dimensions[0]; 
                        it0.next()) {
                        // 生成隨機值
                        TRand val = generator_t::generate(randState);
                        // 縮放和偏移
                        TRand val_adj = adjustor_t::apply(val, scale, offset);
                        // 轉換並儲存
                        dst[out_idx] = cast_t::cast(val_adj);
                    }
                }
            }
        }
    }
    
    // 4. 寫回隨機狀態到全局內存
    s[idx] = randState;
}
```

### 隨機比特生成核心

```cpp
__global__ void tensor_rng_kernel_bits(curandState* s,
                                       tensor_layout_any layoutDst,
                                       uint32_t* dst,
                                       int dim0Bits) {
    // 與上述類似，但為比特張量特化
    // 處理 CUPHY_BIT 類型張量
    // 將 32 位字轉換為比特
}
```

---

## 算法流程

### 正態分佈生成

#### 步驟 1: 初始化隨機狀態

```
seed → curand_init() → curandState[threadId]
```

#### 步驟 2: Box-Muller 轉換（cuRAND 內部實現）

cuRAND 使用 Box-Muller 轉換將均勻分佈轉換為正態分佈：

```
U1, U2 ~ Uniform(0, 1)
Z0 = sqrt(-2 * ln(U1)) * cos(2π * U2)
Z1 = sqrt(-2 * ln(U1)) * sin(2π * U2)
```

#### 步驟 3: 應用縮放和偏移

```
X = mean + stddev * Z
其中 Z 是標準正態分佈值 (μ=0, σ=1)
```

### 均勻分佈生成

#### 步驟 1: 生成 [0,1] 均勻隨機數

```cpp
// FP32
float u = curand_uniform(&randState);    // 返回 [0, 1) 均勻值

// FP64
double u = curand_uniform_double(&randState);

// Complex (FP32)
cuComplex c = curand_uniform(&randState);  // 返回 (u_real, u_imag)
```

#### 步驟 2: 應用範圍轉換

```
對於標量：
    output = min_val + (max_val - min_val) * u

對於複數：
    output_real = min_real + (max_real - min_real) * u_real
    output_imag = min_imag + (max_imag - min_imag) * u_imag
```

### 隨機比特生成

#### 步驟 1: 使用 curand() 生成 32 位無符號整數

```cpp
uint32_t bits = curand(&randState);  // 生成 32 位隨機數
```

#### 步驟 2: 比特打包到張量

```
每個 uint32_t 對應 32 個比特
比特排列：LSB 優先
```

---

## 支持的數據類型

### 實數類型

| cuPHY 類型 | C++ 類型 | cuRAND 函數 | 精度 |
|----------|---------|-----------|------|
| CUPHY_R_16F | __half | N/A | FP16 |
| CUPHY_R_32F | float | curand_uniform() | FP32 |
| CUPHY_R_64F | double | curand_uniform_double() | FP64 |
| CUPHY_R_8I | int8_t | curand_uniform() | 轉換後 |
| CUPHY_R_8U | uint8_t | curand_uniform() | 轉換後 |

### 複數類型

| cuPHY 類型 | C++ 類型 | 元素類型 | 維度 |
|----------|---------|--------|------|
| CUPHY_C_32F | cuComplex | float | 2D |
| CUPHY_C_64F | cuDoubleComplex | double | 2D |
| CUPHY_C_16F | N/A | __half | 2D |
| CUPHY_C_8I | char2 | int8_t | 2D |
| CUPHY_C_8U | uchar2 | uint8_t | 2D |

### 特殊類型

| 類型 | 描述 |
|------|------|
| CUPHY_BIT | 隨機比特序列 |

---

## 性能特性

### GPU 優化

- **線程配置**：1024 線程/塊
- **佔有率**：最大化多處理器利用率
- **內存訪問**：合併內存訪問模式，連續寫入
- **協作組**：支持 32 位至 64 位隨機數的高效生成

### 計算複雜度

- **初始化時間**：O(num_threads)
- **生成時間**：O(N) 其中 N 是元素總數
- **每個元素的操作**：~1 cuRAND 函數調用 + 1-2 個算術操作

### 內存帶寬

- **狀態內存**：每個線程 20 字節（curandState 結構）
- **輸出帶寬**：取決於數據類型（FP32: 4 字節/元素）

---

## API 參考

### C API

```cpp
// 創建隨機數生成器
cuphyStatus_t cuphyCreateRandomNumberGenerator(
    cuphyRNG_t* pRNG,
    unsigned long long seed,
    unsigned int flags,
    cudaStream_t strm);

// 摧毀隨機數生成器
cuphyStatus_t cuphyDestroyRandomNumberGenerator(cuphyRNG_t rng);

// 生成正態分佈隨機數
cuphyStatus_t cuphyRandomNormal(
    cuphyRNG_t rng,
    cuphyTensorDescriptor_t tDst,
    void* pDst,
    const cuphyVariant_t* mean,
    const cuphyVariant_t* stddev,
    cudaStream_t strm);

// 生成均勻分佈隨機數
cuphyStatus_t cuphyRandomUniform(
    cuphyRNG_t rng,
    cuphyTensorDescriptor_t tDst,
    void* pDst,
    const cuphyVariant_t* minValue,
    const cuphyVariant_t* maxValue,
    cudaStream_t strm);
```

### C++ API

```cpp
namespace cuphy {
    class rng {
    public:
        // 構造函數
        rng(unsigned long long seed = 0, 
            unsigned int flags = 0, 
            cudaStream_t strm = 0);
        
        // 生成正態分佈
        template <typename T, typename TVal>
        void normal(T& t, TVal mean, TVal stddev, 
                   cudaStream_t strm = 0);
        
        // 生成均勻分佈
        template <typename T, typename TVal>
        void uniform(T& t, TVal min_val, TVal max_val, 
                    cudaStream_t strm = 0);
    };
}
```

---

## 使用示例

### C++ 示例 1: 生成正態分佈

```cpp
// 建立 RNG 對象
cuphy::rng rng_gen(0xDEADBEEF);  // 使用自定義種子

// 分配張量（1024 個 FP32 元素）
cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc> tensor(1024);

// 生成平均值為 0，標準差為 1 的正態分佈
rng_gen.normal(tensor, 0.0f, 1.0f, 0);

// 等待完成
cudaStreamSynchronize(0);

// 訪問數據
float value = tensor(0);
```

### C++ 示例 2: 生成複數均勻分佈

```cpp
// 建立 RNG 對象
cuphy::rng rng_gen;

// 分配複數張量（256 個 complex64 元素）
cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> 
    complex_tensor(256);

// 生成 [-1, 1] 複數均勻分佈
cupy_gen.uniform(complex_tensor, 
                 cupy::cuComplex(-1.0f, -1.0f),
                 cupy::cuComplex(1.0f, 1.0f), 0);
```

### C 示例：多張量生成

```c
// 創建 RNG
cuphyRNG_t rng;
cuphyCreateRandomNumberGenerator(&rng, 12345, 0, 0);

// 建立張量描述符
cuphyTensorDescriptor_t tensor_desc;
cuphyCreateTensorDescriptor(&tensor_desc);

// 設置 1D 張量，1024 個 FP32 元素
cuphyTensorPrm_t tensor_params = {
    .type = CUPHY_R_32F,
    .numDimensions = 1,
    .dimensions = {1024, 1, 1, 1, 1}
};
cuphySetTensorDescriptor(tensor_desc, &tensor_params);

// 分配輸出內存
float* d_output;
cudaMalloc(&d_output, 1024 * sizeof(float));

// 生成隨機數
cuphyVariant_t mean = {.f32 = 5.0f};
cuphyVariant_t stddev = {.f32 = 2.0f};
cuphyRandomNormal(rng, tensor_desc, d_output, 
                  &mean, &stddev, 0);

// 清理
cuphyDestroyRandomNumberGenerator(rng);
cuphyDestroyTensorDescriptor(tensor_desc);
cudaFree(d_output);
```

### Python 示例 (通過 PyAerial)

```python
import numpy as np
from cuphy import rng, typed_tensor

# 建立 RNG 對象
rng_gen = rng(seed=42)

# 分配張量（2D: 128x128）
tensor = typed_tensor((128, 128), dtype='float32')

# 生成正態分佈 N(mean=10, std=2)
rng_gen.normal(tensor, mean=10.0, stddev=2.0)

# 訪問結果
print(f"Mean: {tensor.mean()}")
print(f"Std: {tensor.std()}")
```

---

## 隨機數生成質量特性

### 統計性能

| 分佈 | 均值誤差 | 方差誤差 | 通過測試 |
|------|--------|--------|--------|
| Normal(0, 1) | < 0.1% | < 0.1% | ✓ |
| Normal(0, 50) | < 0.1% | < 0.1% | ✓ |
| Uniform[0, 1] | < 0.1% | < 0.1% | ✓ |

### 週期性和獨立性

- **週期**：cuRAND Tausworthe 生成器 > 2^127
- **獨立序列**：每個線程獨立序列 (sequence number)
- **低相關性**：符合 NIST 隨機性測試

---

## 應用場景

### 1. 信號處理中的噪聲生成

```cpp
// 生成高斯噪聲用於模擬 AWGN 通道
cuphy::rng rng_gen;
cuphy::typed_tensor<CUPHY_C_32F, cuphy::device_alloc> 
    noise(num_samples);

// 根據 SNR 計算標準差
float noise_std = sqrt(signal_power / SNR_linear);
rng_gen.normal(noise, 0.0f, noise_std);
```

### 2. 初始化和隨機取樣

```cpp
// 隨機初始化權重矩陣
cuphy::rng rng_gen;
cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> 
    weights(rows, cols);

// Xavier 初始化
float limit = sqrt(6.0f / (rows + cols));
rng_gen.uniform(weights, -limit, limit);
```

### 3. 蒙特卡羅模擬

```cpp
// 蒙特卡羅方法估計 π
int num_samples = 1000000;
cuphy::typed_tensor<CUPHY_R_32F, cuphy::device_alloc> 
    x(num_samples), y(num_samples);

cuphy::rng rng_gen;
rng_gen.uniform(x, 0.0f, 1.0f);
rng_gen.uniform(y, 0.0f, 1.0f);
// 後續計算距離並計算比例
```

---

## 異步操作和流支持

RNG 支持 CUDA 流進行異步操作：

```cpp
// 建立多個 CUDA 流
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 在不同流上生成隨機數
cuphy::rng rng1(seed1, 0, stream1);
cuphy::rng rng2(seed2, 0, stream2);

// 非阻塞性操作
rng1.normal(tensor1, mean, stddev, stream1);
rng2.uniform(tensor2, min_val, max_val, stream2);

// 等待兩個流完成
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
```

---

## 故障排除

### 常見問題

1. **隨機數始終相同**
   - 檢查是否使用相同的種子
   - 確認流已同步

2. **性能不理想**
   - 增加張量大小以更好地利用 GPU
   - 確保線程配置正確（1024 線程/塊）

3. **數據類型不支持**
   - 檢查是否使用支持的 cuPHY 類型
   - 對不支持的類型使用類型轉換

### 調試技巧

```cpp
// 驗證隨機數統計特性
std::vector<float> h_data(size);
cudaMemcpy(h_data.data(), d_data, size * sizeof(float), 
          cudaMemcpyDeviceToHost);

float mean = 0, var = 0;
for(auto& v : h_data) mean += v;
mean /= size;

for(auto& v : h_data) var += (v - mean) * (v - mean);
var /= size;

printf("Mean: %f (expected: %f)\n", mean, expected_mean);
printf("Variance: %f (expected: %f)\n", var, expected_var);
```

---

## 參考文獻

- NVIDIA CUDA Toolkit Documentation - cuRAND
- NVIDIA GPU Computing Toolkit - Parallel Programming Guide
- Numerical Recipes: The Art of Scientific Computing
- NIST Special Publication 800-22 - Statistical Test Suite for RNG
