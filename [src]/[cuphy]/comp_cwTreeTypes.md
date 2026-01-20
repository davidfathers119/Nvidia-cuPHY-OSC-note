# Codeword Tree Types Computation (COMP_CWTREE_TYPES)

## 概述

COMP_CWTREE_TYPES是NVIDIA cuPHY庫中用於**Polar編碼**的碼字樹類型計算組件。它用於根據Polar編碼配置動態計算信息位和奇偶校驗位的樹結構類型。

**來源**: [NVIDIA Aerial CUDA Accelerated RAN - comp_cwTreeTypes](https://github.com/NVIDIA/aerial-cuda-accelerated-ran/tree/main/cuPHY/src/cuphy/comp_cwTreeTypes)

**License**: Apache-2.0 | **Copyright**: NVIDIA CORPORATION & AFFILIATES (2025)

---

## 文件結構

### 核心文件

| 文件名 | 功能描述 |
|--------|---------|
| `comp_cwTreeTypes.hpp` | 類定義和API |
| `comp_cwTreeTypes.cu` | CUDA內核實現和輔助函數 |

---

## 主要類

### compCwTreeTypes

**繼承**: `cuphyCompCwTreeTypes`（不透明接口）

**功能**: 計算Polar碼字的樹類型信息，用於解碼和編碼過程中的邏輯決策。

#### 公開方法

```cpp
class compCwTreeTypes : public cuphyCompCwTreeTypes
{
public:
    compCwTreeTypes() = default;
    ~compCwTreeTypes() = default;
    compCwTreeTypes(compCwTreeTypes const&) = delete;
    compCwTreeTypes& operator=(compCwTreeTypes const&) = delete;

    // 設置對象狀態和動態描述符
    void setup(
        uint16_t nPolUciSegs,                                // Polar UCI段數
        const cuphyPolarUciSegPrm_t* pPolUciSegPrmsCpu,     // CPU端UCI段參數
        const cuphyPolarUciSegPrm_t* pPolUciSegPrmsGpu,     // GPU端UCI段參數
        uint8_t** pCwTreeTypesAddrs,                        // 碼字樹類型地址
        compCwTreeTypesDynDescr_t* pCpuDynDesc,            // CPU動態描述符
        void* pGpuDynDesc,                                  // GPU動態描述符
        uint8_t enableCpuToGpuDescrAsyncCpy,               // 異步複製開關
        cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,      // 啟動配置
        cudaStream_t strm);                                 // CUDA流

    // 內核選擇
    void kernelSelect(
        uint16_t nPolUciSegs,
        const cuphyPolarUciSegPrm_t* pPolUciSegPrmsCpu,
        cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg);

    // 獲取描述符信息
    static void getDescrInfo(
        size_t& dynDescrSizeBytes,
        size_t& dynDescrAlignBytes);

private:
    compCwTreeTypesKernelArgs_t m_kernelArgs;
};
```

---

## 數據結構

### 動態描述符

```cpp
struct compCwTypesDynDescr
{
    uint8_t** pCwTreeTypesAddrs;                    // 碼字樹類型地址指針
    const cuphyPolarUciSegPrm_t* pPolarUciSegPrms; // Polar UCI段參數
};
typedef struct compCwTypesDynDescr compCwTreeTypesDynDescr_t;
```

### 內核參數

```cpp
typedef struct
{
    compCwTreeTypesDynDescr_t* pDynDescr;
} compCwTreeTypesKernelArgs_t;
```

---

## 核心CUDA內核

### compCwTreeTypesKernel

```cpp
static __global__ void compCwTreeTypesKernel(compCwTreeTypesDynDescr_t* pDynDescr)
```

**功能**: 為每個Polar UCI段計算碼字樹類型

**輸入參數**（通過pDynDescr獲取）:
- `E_cw`: 傳輸比特數
- `K_cw`: 信息比特數
- `N_cw`: 編碼比特數（碼長）
- `n_cw`: log2(N_cw)
- `exitFlag`: 提前退出標誌

**計算流程**:

#### 1. 禁止區間計算

根據Polar編碼標準，計算需要被跳過的可靠性序列索引區間（最多3個）：

```cpp
int16_t intervalStart[3];
int16_t intervalEnd[3];
```

計算基於：
- 塊長度：`blkLen = nCodedBits / 32`
- 塊數：`nBlks = (nCodedBits - nTxBits) / blkLen`
- 剩餘比特：`nRemBits = (nCodedBits - nTxBits) - (nBlks * blkLen)`

**查找表**:
```cpp
POLAR_REL_SEQ_FORBID_IDXS_FWD  // 前向禁止索引表
POLAR_REL_SEQ_FORBID_IDXS_BWD  // 反向禁止索引表
```

#### 2. 流壓縮（Stream Compaction）

```cpp
strmCompactionHelper<N_THRDS_PER_TILE>(
    thisThrdBlk,
    thisThrdTile,
    pred,
    nActiveThrdTiles,
    pTileStartOffsets,
    thrdOffset);
```

**作用**: 從完整可靠性序列中移除禁止索引

使用Warp級排他性掃描（Warp-level exclusive scan）：

```cpp
uint32_t warpLevelExclusiveScan(bool pred)
{
    uint32_t validRelSeqBmsk = __ballot_sync(FULL_WARP_ACTIVE_BMSK, pred);
    return __popc(validRelSeqBmsk & __lanemask_lt());
}
```

#### 3. 比特類型設置

設置每個比特的類型（0=凍結、1=信息、2=奇偶校驗）：

```cpp
// 信息比特和奇偶校驗比特設置為1和2
if(thrdIdxInBlk < (nInfoBits + nPc))
{
    int16_t infoBitIdx = pRelSeqIdxsPruned[prunedRelSeqStartIdx + thrdIdxInBlk];
    pCwBitTypes[infoBitIdx] = 1;  // 信息比特
    if(thrdIdxInBlk >= (nInfoBits + wmFlag))
    {
        pCwBitTypes[infoBitIdx] = 2;  // 奇偶校驗比特
    }
}
```

#### 4. 樹類型遞歸計算

```cpp
for(int32_t s = 1; s < n; s++)
{
    int32_t stgSize = 1 << (n - s);  // 當前階段大小
    
    // 獲取兩個子節點
    int8_t childNodeA = pCwTreeTypes[childStartIdx + 2 * thrdIdxInBlk];
    int8_t childNodeB = pCwTreeTypes[childStartIdx + 2 * thrdIdxInBlk + 1];
    
    // 根據子節點類型計算父節點類型
    if(childNodeA == 0 && childNodeB == 0)
    {
        pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 0;  // 兩個都是凍結
    }
    else if(childNodeA == 1 && childNodeB == 1)
    {
        pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 1;  // 兩個都是信息
    }
    else
    {
        pCwTreeTypes[cwTypeStartIdx + thrdIdxInBlk] = 3;  // 混合類型
    }
}
```

**樹類型定義**:
- `0`: 兩個子節點都是凍結（SPC節點）
- `1`: 兩個子節點都是信息（Information节点）
- `2`: 保留/未使用
- `3`: 兩個子節點的類型不同（混合節點）

---

## 查找表和常數

### Polar權重度量（Weight Metric）LUT

```cpp
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_32[16];
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_64[64];
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_128[256];
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_256[1024];
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_512[4096];
static __device__ __constant__ uint16_t POLAR_WM_ARRAY_1024[8192];

static __device__ __constant__ uint16_t const* POLAR_WM_LUT_PTR[] =
{
    POLAR_WM_ARRAY_32,
    POLAR_WM_ARRAY_64,
    POLAR_WM_ARRAY_128,
    POLAR_WM_ARRAY_256,
    POLAR_WM_ARRAY_512,
    POLAR_WM_ARRAY_1024
};
```

**用途**: 用於計算位的可靠性排序中的最小權重匹配

### 可靠性序列索引（Reliability Sequence Index）LUT

```cpp
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_32[32];
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_64[64];
// ... 其他大小
static __device__ __constant__ uint16_t POLAR_REL_SEQ_IDXS_1024[1024];

static __device__ __constant__ uint16_t const* POLAR_REL_SEQ_IDXS_LUT_PTR[] =
{
    POLAR_REL_SEQ_IDXS_32,
    POLAR_REL_SEQ_IDXS_64,
    POLAR_REL_SEQ_IDXS_128,
    POLAR_REL_SEQ_IDXS_256,
    POLAR_REL_SEQ_IDXS_512,
    POLAR_REL_SEQ_IDXS_1024
};
```

**用途**: 存儲按可靠性排序的Polar編碼位的索引

### 禁止索引表

```cpp
static __device__ __constant__ int8_t POLAR_REL_SEQ_FORBID_IDXS_FWD[][32];
static __device__ __constant__ int8_t POLAR_REL_SEQ_FORBID_IDXS_BWD[][32];
```

**用途**: 根據速率匹配情況指示哪些可靠性序列索引應被禁止

---

## 輔助函數

### Warp級操作

```cpp
// 獲取小於當前線程的活躍線程掩碼
__device__ __forceinline__ uint32_t __lanemask_lt()
{
    uint32_t mask;
    asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
    return mask;
}

// Warp級排他性掃描
__device__ __forceinline__ uint32_t warpLevelExclusiveScan(bool pred)
{
    uint32_t validRelSeqBmsk = __ballot_sync(FULL_WARP_ACTIVE_BMSK, pred);
    return __popc(validRelSeqBmsk & __lanemask_lt());
}
```

### 流壓縮幫助

```cpp
template <uint32_t N_THRDS_PER_TILE>
__device__ void strmCompactionHelper(
    thread_block const& thisThrdBlk,
    thread_block_tile<N_THRDS_PER_TILE> const& thisThrdTile,
    bool pred,
    uint32_t nActiveThrdTiles,
    int32_t* pTileStartOffset,
    int32_t& thrdOffset);
```

**功能**: 計算每個線程的輸出位置用於流壓縮

### Warp級最小值縮減

```cpp
__device__ __forceinline__ void warpLevelReduceMinIdx(int32_t& val, int32_t& idx)
```

**功能**: 沿Warp找最小值及其索引，保持順序以處理平局

### 塊級最小值縮減

```cpp
__inline__ __device__ int32_t blockReduceMinIdx(
    thread_block const& thisThrdBlk,
    const int16_t* idxArray,
    const uint16_t* valArray,
    const uint32_t startRange,
    const uint32_t endRange);
```

**功能**: 跨整個線程塊找最小值，用於WM（Weight Metric）計算

---

## 共享內存佈局

```cpp
// 共享內存分配
__shared__ uint8_t smemBlk[N_SMEM_ELEMS];

uint32_t* pSmem = reinterpret_cast<uint32_t*>(smemBlk);
int32_t* pTileStartOffsets = reinterpret_cast<int32_t*>(pSmem);
int16_t* pRelSeqIdxsPruned = reinterpret_cast<int16_t*>(
    &pTileStartOffsets[N_MAX_THRD_TILES]);
int8_t* pCwBitTypes = reinterpret_cast<int8_t*>(
    &pRelSeqIdxsPruned[REL_SEQ_IDX_BUF_LEN]);
int8_t* pCwTreeTypes = reinterpret_cast<int8_t*>(
    &pCwBitTypes[N_MAX_CODED_BITS]);
```

**使用**:
- `pTileStartOffsets`: 流壓縮中每個Tile的起始偏移
- `pRelSeqIdxsPruned`: 修剪後的可靠性序列索引
- `pCwBitTypes`: 每個比特的類型（凍結/信息/奇偶）
- `pCwTreeTypes`: 樹節點類型

---

## 配置和常數

```cpp
// 編碼長度範圍
static constexpr uint32_t N_MIN_CODED_BITS = 32;   // 最小碼字長度
static constexpr uint32_t N_MAX_CODED_BITS = 1024; // 最大碼字長度

// 線程配置
static constexpr uint32_t N_THRDS_PER_WARP = 32;
static constexpr uint32_t N_THRDS_PER_TILE = N_THRDS_PER_WARP;
static constexpr uint32_t FULL_WARP_ACTIVE_BMSK = 0xFFFFFFFF;

// 最大樹節點數
static constexpr uint32_t N_MAX_TREE_TYPES = 2 * N_MAX_CODED_BITS - 2;

// 奇偶校驗位配置
// 對於 18 <= K <= 25 的信息位：
// - nPc = 3（3個奇偶校驗位）
// - wmFlag = (E - K + 3) > 192 ? 1 : 0
```

---

## 3GPP規範應用

該實現遵循3GPP標準中的Polar編碼：

1. **可靠性排序**: 根據Weight Metric對所有比特進行排序
2. **凍結比特**: 最不可靠的比特被設置為固定值（通常為0）
3. **信息比特**: 可靠性最高的比特用於傳輸信息
4. **奇偶校驗比特**: 用於提高編碼效率
5. **速率匹配**: 根據傳輸比特數E進行截斷或重複

---

## 內核啟動配置

```cpp
dim3 gridDim(nPolUciSegs);  // 每個UCI段一個線程塊
dim3 blockDim(max_N_cw);    // 線程塊大小 = 最大碼長
```

---

## 執行流程

```
1. 初始化
   ├─ 計算禁止比特區間
   └─ 分配共享內存

2. 流壓縮
   ├─ 過濾禁止索引
   └─ 計算線程偏移

3. 比特類型設置
   ├─ 標記信息比特
   ├─ 標記奇偶校驗比特
   └─ 其餘為凍結比特

4. 樹類型計算（遞歸）
   ├─ 階段0：從比特類型初始化
   ├─ 階段1-n：根據子節點遞歸計算
   └─ 樹類型：{0=SPC, 1=Info, 2=Reserved, 3=Mixed}

5. 輸出寫回
   └─ 將樹類型存儲到全局內存
```

---

## 性能特性

✅ **並行化流壓縮** - 使用Warp級掃描
✅ **共享內存優化** - 局部存儲重用
✅ **遞歸樹計算** - 高效的層級計算
✅ **向量化操作** - 充分利用SIMD
✅ **非同步支持** - CPU-GPU描述符異步複製

---

## 相關配置宏

```cpp
CUPHY_POLAR_ENC_MAX_INFO_BITS   // 最大信息比特數
CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS  // 異質配置最大數
```

---

## 延伸閱讀

- 3GPP TS 38.212 - Polar編碼規範
- [NVIDIA cuPHY官方文檔](https://github.com/NVIDIA/aerial-cuda-accelerated-ran)
- Polar編碼理論與實現
