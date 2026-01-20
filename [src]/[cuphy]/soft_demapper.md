# Soft Demapper (cuPHY 軟判決解映模組)

## 概述

**Soft Demapper** 是 NVIDIA cuPHY 中的 GPU 加速軟判決解映模組，負責將接收到的調制符號轉換為 Log-Likelihood Ratios (LLRs)，用於後續的信道解碼。該模組支援 BPSK、QPSK、16QAM、64QAM 及 256QAM 調制格式，提供高效的 LLR 計算以最大化解碼效能。

**核心特性：**
- 支持多種 QAM 調制格式 (BPSK/QPSK/16QAM/64QAM/256QAM)
- 紋理記憶體 (Texture Memory) 加速的 LLR 查表
- FP16/FP32 混合精度計算
- 模板特化優化編譯
- 直接計算與查表混合實現

---

## 軟體架構

### 1. 核心類別

#### `soft_demapper_context`

儲存軟解映所需的唯讀資源，特別是 QAM 查表紋理。

```cpp
namespace cuphy_i {
    class soft_demapper_context {
    public:
        soft_demapper_context();
        const mipmapped_texture& QAM_tex() const { return QAMtex_; }
    private:
        mipmapped_texture QAMtex_;
    };
}
```

**初始化流程：**
- 建立多層級紋理 (4 層級) 用於不同 QAM 格式
- Layer 0: QAM256_table (32x32 紋理)
- Layer 1: QAM64_table
- Layer 2: QAM16_table  
- Layer 3: QAM4_table
- 使用 `cudaFilterModeLinear` 進行線性插值

#### `cuphy_i::soft_demap()` 函數

主要公開 API，用於執行軟解映運算。

```cpp
cuphyStatus_t soft_demap(context&     ctx,
                         tensor_desc& tLLR,
                         void*        pLLR,
                         tensor_desc& tSym,
                         const void*  pSym,
                         int          log2_QAM,
                         float        noiseVariance,
                         cudaStream_t strm);
```

**參數說明：**
- `ctx`: cuPHY 上下文，包含軟解映資源
- `tLLR`: 輸出 LLR 張量描述符
- `pLLR`: 輸出 LLR 資料指針
- `tSym`: 輸入符號張量描述符
- `pSym`: 輸入符號資料指针
- `log2_QAM`: QAM 位寬 (1/2/4/6/8)
- `noiseVariance`: 複數雜訊功率 (QAM 域)
- `strm`: CUDA 流

**返回狀態：**
- `CUPHY_STATUS_SUCCESS`: 執行成功
- `CUPHY_STATUS_UNSUPPORTED_CONFIG`: 不支持的資料型別組合
- `CUPHY_STATUS_UNSUPPORTED_LAYOUT`: 不支持的張量佈局
- `CUPHY_STATUS_SIZE_MISMATCH`: LLR 輸出空間不足
- `CUPHY_STATUS_INTERNAL_ERROR`: CUDA 執行錯誤

---

## QAM 特性

軟解映模組使用 `QAM_traits` 模板結構儲存各 QAM 格式的調制參數：

### QAM 格式參數

```cpp
template <> struct QAM_traits<2>  // BPSK
{
    static constexpr int    bits     = 1;
    static constexpr int    PAM_bits = 1;
    static constexpr int    N        = 4;
    static constexpr double A        = 0.707107;  // 1/sqrt(2)
    static constexpr float  m        = /* slope */;
    static constexpr float  b        = /* intercept */;
    static constexpr float  LEVEL    = 3.0f;      // 紋理 LOD
};

template <> struct QAM_traits<4>  // QPSK
{
    static constexpr int    bits     = 2;
    static constexpr int    PAM_bits = 1;
    static constexpr int    N        = 4;
    static constexpr double A        = 0.707107;
    // ...
};

template <> struct QAM_traits<16>  // 16QAM
{
    static constexpr int    bits     = 4;
    static constexpr int    PAM_bits = 2;
    static constexpr int    N        = 8;
    static constexpr double A        = 0.316228;  // 1/sqrt(10)
    // ...
};

template <> struct QAM_traits<64>  // 64QAM
{
    static constexpr int    bits     = 6;
    static constexpr int    PAM_bits = 3;
    static constexpr int    N        = 16;
    static constexpr double A        = 0.154303;  // 1/sqrt(42)
    // ...
};

template <> struct QAM_traits<256>  // 256QAM
{
    static constexpr int    bits     = 8;
    static constexpr int    PAM_bits = 4;
    static constexpr int    N        = 32;
    static constexpr double A        = 0.076696;  // 1/sqrt(170)
    // ...
};
```

**參數說明：**
- `bits`: 每符號比特數
- `PAM_bits`: PAM (同軸幅度調制) 子成分比特數
- `N`: 紋理查表大小
- `A`: 調制歸一化因子
- `m, b`: 符號到紋理座標的線性變換參數
- `LEVEL`: 紋理金字塔級別 (LOD)

---

## 軟解映演算法

### 1. PAM 分解

複數 QAM 符號被分解為同相 (I) 和正交相 (Q) 兩個 PAM 分量：

$$\text{Symbol} = I + jQ$$

每個分量獨立計算 LLR，最後組合成完整的 QAM LLR。

### 2. LLR 計算

#### BPSK (1 位)

直接計算：
$$\text{LLR} = 2 \cdot A \cdot \text{PAM\_noise\_var\_inv} \cdot (I + Q)$$

**C++ 實現：**
```cpp
template <>
struct LLR_BPSK<float, __half> {
    __device__ static void symbol_to_LLR_group(
        TLLRGroup& grp, 
        const symbol_t& sym,
        noise_t PAMnoiseVarInv)
    {
        grp.f[0] = PAMnoiseVarInv * 2 * QAM_traits<2>::A * (sym.x + sym.y);
    }
};
```

#### QPSK (2 位)

分別計算 I 和 Q 分量的 LLR：
$$\text{LLR}[0] = 2 \cdot A \cdot \text{PAM\_noise\_var\_inv} \cdot I$$
$$\text{LLR}[1] = 2 \cdot A \cdot \text{PAM\_noise\_var\_inv} \cdot Q$$

**FP16 實現：**
```cpp
template <>
struct LLR_QPSK<float, __half> {
    __device__ static void symbol_to_LLR_group(
        TLLRGroup& grp,
        const symbol_t& sym,
        noise_t PAMnoiseVarInv)
    {
        __half2 A2 = __float2half2_rn(2 * QAM_traits<2>::A);
        grp.f16x2[0] = __hmul2(
            __hmul2(PAMnoiseVarInv, A2),
            __floats2half2_rn(sym.x, sym.y)
        );
    }
};
```

#### 高階 QAM (16QAM/64QAM/256QAM)

使用紋理記憶體查表加速計算：

```cpp
template <typename TSymbolScalar, typename TLLR, int QAM>
struct soft_demapper {
    __device__ static void symbol_to_LLR_group(
        TLLRGroup& grp,
        const symbol_t& sym,
        noise_t PAMnoiseVarInv,
        cudaTextureObject_t texObj)
    {
        tex_result_t res_I, res_Q;
        float2 t;
        t = symbol_to_tex_coords(sym, 
                                 QAM_traits<QAM>::m,
                                 QAM_traits<QAM>::b);
        tex_1D_lod_ptx(res_I, texObj, t.x, 
                       QAM_traits<QAM>::LEVEL);
        tex_1D_lod_ptx(res_Q, texObj, t.y, 
                       QAM_traits<QAM>::LEVEL);
        swizzle_LLRs(grp, res_I, res_Q);
        apply_noise(grp, PAMnoiseVarInv);
    }
};
```

### 3. 雜訊調整

LLR 按雜訊方差進行縮放：

$$\text{LLR\_final} = \text{LLR\_base} \times \text{PAM\_noise\_var\_inv}$$

其中 PAM 雜訊方差是 QAM 雜訊方差的一半：
$$\text{PAM\_noise\_var} = \text{QAM\_noise\_var} / 2$$

---

## CUDA 核心實現

### 軟解映核心函數

```cpp
template <typename TSymbol, typename TLLR>
__global__ void soft_demapper_kernel(
    cudaTextureObject_t texObj,
    float noiseInv,
    int QAM_bits,
    tensor_ref_t_contig_2D<TLLR> tLLR,
    tensor_ref_t_contig_2D<const TSymbol> tSym)
{
    typedef typename scalar_from_complex<TSymbol>::type 
        symbol_scalar_t;
    typedef soft_demapper::soft_demapper_any<
        symbol_scalar_t, TLLR> soft_demapper_t;
    
    int SYMBOL_IDX = (blockIdx.x * blockDim.x) + threadIdx.x;
    int COLUMN_IDX = blockIdx.y;
    
    if(SYMBOL_IDX >= tSym.layout().dimensions[0])
        return;
    
    TSymbol softEst = tSym({SYMBOL_IDX, COLUMN_IDX});
    
    // LLR 輸出結構 (最多 8 個 LLR)
    llr_group_t grp;
    
    // PAM_variance = QAM_variance / 2
    // 1/PAM_variance = 2 * 1/QAM_variance = 2 * inv_QAM_variance
    soft_demapper_t::symbol_to_LLR_group(
        grp,
        softEst,
        noise_type_map_t::scale(noiseInv, 2.0f),
        QAM_bits,
        texObj);
    
    // 寫入輸出記憶體
    grp.write(tLLR.addr() + 
              tLLR.layout().offset({SYMBOL_IDX * QAM_bits, COLUMN_IDX}),
              QAM_bits);
}
```

### 核心啟動配置

```cpp
dim3 blkDim(1024);  // 每塊 1024 執行緒
dim3 grdDim(div_round_up(NUM_SYMBOLS, 1024), NUM_COL);
```

- 每個執行緒處理一個符號
- X 維度：符號數量
- Y 維度：列數量 (多個天線/層)

---

## 資料型別支持

### 支持的輸入/輸出組合

| LLR 類型 | 符號類型 | 狀態 |
|---------|--------|-----|
| FP16    | FP16 複數 | ✅ 支持 |
| FP16    | FP32 複數 | ✅ 支持 |
| FP32    | FP16 複數 | ✅ 支持 |
| FP32    | FP32 複數 | ✅ 支持 |

### 型別映射

```cpp
template <> struct noise_type_map<float> {
    typedef float type;
    static __device__ float create(float f) { return f; }
    static __device__ float create(__half h) 
    { return __half2float(h); }
    static __device__ float scale(float f, float s)
    { return f * s; }
    static __device__ float scale(__half h, float s)
    { return (s * __half2float(h)); }
};

template <> struct noise_type_map<__half> {
    typedef __half2 type;
    static __device__ __half2 create(float f)
    { return __float2half2_rn(f); }
    static __device__ __half2 create(__half h)
    { return __half2half2(h); }
    static __device__ __half2 scale(float f, float s)
    { return __float2half2_rn(f * s); }
};
```

---

## LLR 輸出結構

### `LLR_group` 聯合體

針對不同精度的 LLR 群組存儲：

#### FP32 (8 個 LLR)

```cpp
template <> union LLR_group<float, 8> {
    float   f[8];
    float4  f4[2];
    
    __device__ void write(void* dst) {
        float4* f4dst = static_cast<float4*>(dst);
        f4dst[0] = f4[0];
        f4dst[1] = f4[1];
    }
    
    __device__ float& operator[](int i) { return f[i]; }
};
```

#### FP16 (8 個 LLR / 4 對)

```cpp
template <> union LLR_group<__half, 8> {
    __half  f16[1];
    __half2 f16x2[4];
    uint4   ui32_4;
    uint2   ui32_2[2];
    
    __device__ void write(void* dst) {
        *((uint4*)(dst)) = ui32_4;
    }
    
    __device__ float operator[](int i) {
        return (0 == (i % 2)) ? 
            __low2float(f16x2[i/2]) : 
            __high2float(f16x2[i/2]);
    }
};
```

### 紋理結果交錯

從紋理記憶體獲得的 I/Q 分量結果需要進行交錯 (swizzle) 排列：

```cpp
inline __device__
void swizzle_LLRs(LLR_group<__half, 4>& LLR_grp,
                  const tex_result_v4<__half>& res_I,
                  const tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] 
    // --> [ Q.a.lo I.a.lo ]
    LLR_grp.ui32_2.x = 
        uint32_permute<0x5410>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] 
    // --> [ Q.a.hi I.a.hi ]
    LLR_grp.ui32_2.y = 
        uint32_permute<0x7632>(res_I.a.u32, res_Q.a.u32);
}
```

---

## C API 接口

### `cuphyDemodulateSymbol()`

公開 C API 用於軟解映：

```cpp
cuphyStatus_t cuphyDemodulateSymbol(
    cuphyContext_t context,
    cuphyTensorDescriptor_t tLLR,
    void* pLLR,
    cuphyTensorDescriptor_t tSym,
    const void* pSym,
    int log2_QAM,
    float noiseVariance,
    cudaStream_t strm)
{
    cuphy_i::context& ctx = 
        static_cast<cuphy_i::context&>(*context);
    return cuphy_i::soft_demap(ctx,
                               tLLRDesc,
                               pLLR,
                               tSymDesc,
                               pSym,
                               log2_QAM,
                               noiseVariance,
                               strm);
}
```

### 張量描述符要求

- **輸入符號張量**: 2D 連續記憶體，類型為 CUPHY_C_16F 或 CUPHY_C_32F
- **輸出 LLR 張量**: 2D 連續記憶體，類型為 CUPHY_R_16F 或 CUPHY_R_32F

---

## 使用範例

### 基本軟解映

```cpp
// 初始化 cuPHY 上下文
cuphyContext_t ctx;
cuphyCreateContext(&ctx, 0);  // GPU 0

// 準備輸入符號 (複數)
std::vector<cuFloatComplex> symbols(1024);
// ... 填充符號資料

// 準備輸出 LLR
std::vector<float> llr_output(1024 * 2);  // QPSK: 2 bits/symbol

// 建立張量描述符
cuphyTensorDescriptor_t tSym, tLLR;
cuphyCreateTensorDescriptor(&tSym);
cuphyCreateTensorDescriptor(&tLLR);

// 設定張量
int64_t sym_dims[] = {1024, 1};
cuphySetTensorDescriptor(tSym, 2, sym_dims, 
                         CUPHY_C_32F, NULL);

int64_t llr_dims[] = {1024 * 2, 1};
cuphySetTensorDescriptor(tLLR, 2, llr_dims,
                         CUPHY_R_32F, NULL);

// 執行軟解映
float noise_variance = 2.0f;
int qam_bits = 2;  // QPSK: 2
cuphyDemodulateSymbol(ctx, 
                      tLLR, llr_output.data(),
                      tSym, symbols.data(),
                      qam_bits,
                      noise_variance,
                      0);

// 清理
cuphyDestroyTensorDescriptor(tSym);
cuphyDestroyTensorDescriptor(tLLR);
cuphyDestroyContext(ctx);
```

### 16QAM 軟解映

```cpp
// 配置
int num_symbols = 256;
int log2_qam = 4;  // 16QAM
float noise_var = 1.0f;

// 符號張量: 256 個 16QAM 符號
int64_t sym_shape[] = {256, 1};
// LLR 張量: 256 * 4 = 1024 個 LLR
int64_t llr_shape[] = {1024, 1};

cuphyDemodulateSymbol(ctx,
                      tLLR, llr_data,
                      tSym, sym_data,
                      log2_qam,
                      noise_var,
                      stream);
```

---

## 紋理查表最佳化

### 查表方法

對於高階 QAM (16/64/256)，軟解映使用 CUDA 紋理記憶體的線性插值功能：

1. **座標變換**: 符號值 → 紋理座標 `[0, 1]`
2. **紋理查詢**: 使用 1D 紋理進行 LLR 值查詢
3. **金字塔級別選擇**:
   - QPSK: LOD = 3.0
   - 16QAM: LOD = 1.0
   - 64QAM: LOD = 1.0
   - 256QAM: LOD = 0.0

### 紋理配置

```cpp
const cudaTextureDesc s_MipmappedTexDesc = {
    { cudaAddressModeClamp, cudaAddressModeClamp, 
      cudaAddressModeClamp},
    cudaFilterModeLinear,
    cudaReadModeElementType,
    0,
    {0.0f, 0.0f, 0.0f, 0.0f},
    1,  // normalizedCoords
    0,
    cudaFilterModePoint,
    0.0f,
    0.0f,
    3.0f,  // maxMipmapLevelClamp
    0
};
```

---

## 簡化軟解映

針對某些應用場景，模組提供簡化版本 `soft_demapper_simplified`，使用近似 LLR 計算以提高效能：

```cpp
template <typename TSymbolScalar, typename TLLR>
struct soft_demapper_simplified {
    __device__ static void symbol_to_LLR_group(
        TLLRGroup& grp,
        const symbol_t& sym,
        noise_t PAMnoiseVarInv,
        int nBits)
    {
        // 簡化的 LLR 計算邏輯
        // 用於實時性要求高的場景
    }
};
```

---

## 效能特性

### 計算複雜度

**每符號操作數：**
- BPSK: 1 加法 + 1 乘法 + 1 轉換
- QPSK: 2 加法 + 2 乘法 + 1 轉換
- 16QAM: 1 紋理讀取 + 4 交錯 + 調度
- 64QAM: 1 紋理讀取 + 6 交錯 + 調度
- 256QAM: 1 紋理讀取 + 8 交錯 + 調度

### 記憶體帶寬

- **輸入**: 每符號 8 字節 (複數 FP32) 或 4 字節 (複數 FP16)
- **輸出**: 每比特 4 字節 (FP32 LLR) 或 2 字節 (FP16 LLR)
- **紋理**: 多層級紋理金字塔快取存取

### 佔用資源

- **暫存器**: ~80-120 個 (含紋理座標計算)
- **共享記憶體**: 0 (純全域記憶體操作)
- **紋理快取**: 高效率 (L1/L2 快取幫助)

---

## 測試基礎設施

### 內部測試核心

```cpp
template <int QAM>
__global__ void test_soft_demapper_kernel(
    cudaTextureObject_t dmTexObj,
    const cuFloatComplex* symbols,
    size_t symbolCount,
    __half2* LLR,
    float noiseVarInv)
{
    typedef soft_demapper::QAM_traits<QAM> 
        QAM_traits_t;
    typedef soft_demapper::soft_demapper<float, __half, QAM>
        soft_demapper_t;
    typedef soft_demapper::LLR_group<__half, 
        QAM_traits_t::bits> llr_group_t;
    
    llr_group_t llr_grp;
    
    int symbolIdx = 
        (blockIdx.x * blockDim.x) + threadIdx.x;
    if(symbolIdx >= symbolCount) return;
    
    __half2 PAMnoiseVarInv = 
        __float2half2_rn(noiseVarInv * 2.0f);
    
    soft_demapper_t::symbol_to_LLR_group(
        llr_grp,
        symbols[symbolIdx],
        PAMnoiseVarInv,
        dmTexObj);
    
    llr_grp.write(LLR + 
        (symbolIdx * (QAM_traits_t::bits / 2)));
}
```

### 單元測試

```cpp
TEST(SoftDemapper, BPSK) {
    do_soft_demapper_test<2>(256, 
        1 * soft_demapper::QAM_traits<2>::A,
        2.0f, PAM2_pos, 2);
}

TEST(SoftDemapper, QPSK) {
    do_soft_demapper_test<4>(256,
        2 * soft_demapper::QAM_traits<4>::A,
        2.0f, PAM2_pos, 2);
}

TEST(SoftDemapper, QAM16) {
    do_soft_demapper_test<16>(256,
        4 * soft_demapper::QAM_traits<16>::A,
        2.0f, PAM4_pos, 4);
}

TEST(SoftDemapper, QAM64) {
    do_soft_demapper_test<64>(256,
        8 * soft_demapper::QAM_traits<64>::A,
        2.0f, PAM8_pos, 8);
}

TEST(SoftDemapper, QAM256) {
    do_soft_demapper_test<256>(256,
        16 * soft_demapper::QAM_traits<256>::A,
        2.0f, PAM16_pos, 16);
}
```

---

## 故障排除

### 常見問題

**Q: 輸出 LLR 全為 0？**
- A: 檢查 `noiseVariance` 參數是否正確設置
- 確認張量描述符 "連續性要求 (contiguous requirement)

**Q: 計算結果不匹配 MATLAB 參考？**
- A: 驗證 PAM 雜訊方差計算 (應為 QAM_var/2)
- 檢查紋理插值設置 (normalizedCoords = 1)

**Q: 低精度 (FP16) 下精度下降？**
- A: 在涉及大數值乘法時可考慮 FP32 中間計算
- 調整雜訊方差預縮放以避免溢出

**Q: GPU 記憶體不足？**
- A: 減小批次大小或使用流來分塊處理
- 考慮啟用統一記憶體 (Unified Memory)

---

## 應用場景

### 1. 5G NR PUSCH 接收

軟解映是 PUSCH (Physical Uplink Shared Channel) 接收鏈的核心：

```
OFDM 解調 → 信道等化 → 軟解映 → 信道解碼 → 資訊提取
```

### 2. 實時 LLR 計算

在 5G 物理層信號處理中，軟解映直接作用於等化後的符號生成 LLR 供 LDPC 解碼器使用。

### 3. 混合精度推理

結合 FP16 輸入符號和 FP32 輸出 LLR，在保證精度的同時降低記憶體帶寬。

---

## 參考資源

- [NVIDIA cuPHY 文檔](https://docs.nvidia.com/networking/aerial/)
- [5G NR 3GPP TS 38.211 (物理層)][](https://www.3gpp.org)
- [CUDA 紋理記憶體最佳化](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#texture-memory)
- 相關模組: `channel_eq`, `rate_matching`, `ldpc_decoder`

