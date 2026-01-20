# PUSCH RSSI and RSRP Measurement

## 概述 (Overview)

PUSCH RSSI (Received Signal Strength Indicator) 和 RSRP (Reference Signal Received Power) 是上行鏈路 PHY 層測量的重要指標，用於評估接收信號強度、SINR 計算、DTX 檢測和功率控制。cuPHY 實現提供高性能的 GPU 加速計算，支持全時隙和子時隙（早期 HARQ）處理。

## 檔案結構 (File Structure)

```
cuPHY/src/cuphy/pusch_rssi/
├── pusch_rssi.hpp      # 頭文件 - 類定義和接口
├── pusch_rssi.cu       # CUDA 核函數實現
└── (集成在 cuPHY 框架中)

pyaerial/pybind11/
├── pycuphy_rsrp.hpp    # Python 綁定頭文件
└── pycuphy_rsrp.cpp    # Python 綁定實現
```

## 核心功能 (Core Functions)

### 1. 主類: `puschRxRssi`

```cpp
namespace puschRx_rssi {
    class puschRxRssi : public cuphyPuschRxRssi
    {
    public:
        // RSSI 測量設置
        cuphyStatus_t setupRssiMeas(
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
            uint16_t nUeGrps,
            uint32_t nMaxPrb,
            uint8_t dmrsSymbolIdx,
            uint8_t enableCpuToGpuDescrAsyncCpy,
            puschRxRssiDynDescrVec_t& dynDescrVecCpu,
            void* pDynDescrsGpu,
            cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
            cudaStream_t strm
        );
        
        // RSRP 測量設置
        cuphyStatus_t setupRsrpMeas(
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
            uint16_t nUeGrps,
            uint32_t nMaxPrb,
            uint8_t dmrsSymbolIdx,
            uint8_t enableCpuToGpuDescrAsyncCpy,
            puschRxRsrpDynDescrVec_t& dynDescrVecCpu,
            void* pDynDescrsGpu,
            cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
            cudaStream_t strm
        );
        
        static void getDescrInfo(
            size_t& rssiDynDescrSizeBytes,
            size_t& rssiDynDescrAlignBytes,
            size_t& rsrpDynDescrSizeBytes,
            size_t& rsrpDynDescrAlignBytes
        );
    };
}
```

### 2. RSSI 測量核函數: `rssiMeasKernel`

```cuda
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK,
          uint32_t N_ITER_PER_THRD_BLK,
          uint8_t  DMRS_SYMBOL_IDX>
__global__ void rssiMeasKernel(puschRxRssiDynDescr_t* pDynDescr)
```

**功能:**
- 測量所有分配 PRB 上的接收信號強度
- 支持多符號和多接收天線聚合
- 生成 RSSI_Full（完整 RSSI）和 RSSI（平均 RSSI）

### 3. RSRP 測量核函數: `rsrpMeasKernel`

```cuda
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t THRD_GRP_TILE_SIZE,
          uint32_t N_THRD_GRP_TILES_PER_THRD_BLK,
          uint32_t N_SC_PER_THRD_BLK_ITER,
          uint32_t N_ITER_PER_THRD_BLK,
          uint8_t  DMRS_SYMBOL_IDX>
__global__ void rsrpMeasKernel_v1/v2(puschRxRsrpDynDescr_t* pDynDescr)
```

**功能:**
- 計算通道估計功率（Reference Signal Received Power）
- 平均化所有 PRB、接收天線、時間域估計
- 計算前置和後置均衡 SINR
- 計算後置均衡噪聲方差

## 數據流 (Data Flow)

### 輸入張量 (Input Tensors - RSSI)

| 張量 | 形狀 | 類型 | 說明 |
|------|------|------|------|
| `tDataRx` | (NF, ND, N_BS_ANTS) | 複數 | 接收信號 |
| `tSymbLoc` | (N_SYMB,) | 整數 | 符號位置索引 |

### 輸入張量 (Input Tensors - RSRP)

| 張量 | 形狀 | 類型 | 說明 |
|------|------|------|------|
| `tHEst` | (N_BS_ANTS, N_LAYERS, NF, NH) | 複數 | 通道估計值 |
| `tNoiseIntfVarPreEq` | (N_UE_GRP,) | 實數 | 均衡前噪聲功率 (dB) |

### 輸出張量 (Output Tensors)

| 張量 | 形狀 | 類型 | 說明 |
|------|------|------|------|
| `tRssiFull` | (N_SYMB, N_UE_GRP) | 實數 | 完整 RSSI (dB)，每符號值 |
| `tRssi` | (N_UE_GRP,) | 實數 | 平均 RSSI (dB) |
| `tRsrp` | (N_UE,) 或 (N_UE_GRP,) | 實數 | RSRP (dB) |
| `tSinrPreEq` | (N_UE,) | 實數 | 均衡前 SINR (dB) |
| `tSinrPostEq` | (N_UE,) | 實數 | 均衡後 SINR (dB) |
| `tNoiseVarPostEq` | (N_UE,) | 實數 | 均衡後噪聲功率 (dB) |

## 算法流程 (Algorithm Flow)

### RSSI 計算

**步驟 1: 功率積累**

對於每個分配符號位置和接收天線，計算接收信號功率：

$$P_{rssi} = \sum_{f} |x_f|^2$$

其中 $x_f$ 是頻率子載波 $f$ 的接收複數信號。

**步驟 2: 符號內聚合**

在每個符號內聚合功率：

$$P_{rssi,sym} = \sum_{RxAnt} \sum_{PRB} P_{rssi}(PRB, RxAnt)$$

**步驟 3: 轉換為 dB**

$$RSSI_{dB} = 10 \log_{10}(P_{rssi,lin})$$

### RSRP 計算

**步驟 1: 通道估計功率計算**

對於每個接收天線、層和子載波：

$$P_{rsrp} = |H_{est}|^2$$

**步驟 2: 符號聚合**

對所有分配的 PRB、接收天線、時間估計值和層求平均：

$$RSRP_{lin} = \frac{1}{N_{PRB} \times N_{RxAnt} \times N_{TimeChEst}} \sum P_{rsrp}$$

**步驟 3: 轉換為 dB**

$$RSRP_{dB} = 10 \log_{10}(RSRP_{lin})$$

### SINR 計算

**步驟 4: 預均衡 SINR**

$$SINR_{preEq,dB} = RSRP_{dB} - NoiseVar_{preEq,dB}$$

其中 $NoiseVar_{preEq,dB}$ 來自噪聲干擾估計模組。

**步驟 5: 後均衡 SINR（可選）**

在頻域均衡後計算：

$$SINR_{postEq,dB} = RSRP_{dB} - NoiseVar_{postEq,dB}$$

## 關鍵參數 (Key Parameters)

### 動態描述符

```cpp
// RSSI 描述符
struct puschRxRssiDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;  // UE 組參數
};

// RSRP 描述符
struct puschRxRsrpDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;  // UE 組參數
};
```

### 核函數配置

| 參數 | 說明 |
|------|------|
| `THRD_GRP_TILE_SIZE` | 線程組磁貼大小（通常 32，對應 warp） |
| `N_THRD_GRP_TILES_PER_THRD_BLK` | 每塊線程組磁貼數量（典型值 4 → 128 線程/塊） |
| `N_SC_PER_THRD_BLK_ITER` | 每次迭代處理的子載波數 |
| `N_ITER_PER_THRD_BLK` | 每塊的迭代次數 |
| `DMRS_SYMBOL_IDX` | 處理的 DMRS 符號索引 |

### 符號位置配置

| 模式 | 說明 |
|------|------|
| `CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS` | 全時隙處理 (14 個符號) |
| `CUPHY_PUSCH_RSSI_EST_FIRST_DMRS` | 子時隙處理 (早期 HARQ) |

### 支援的數據類型

| 配置 | DataRx | RssiFull | Rssi | RSRP | 說明 |
|------|--------|----------|------|------|------|
| 主要 | C32F | R32F | R32F | R32F | 32 位浮點 |

## 並行化策略 (Parallelization Strategy)

### 網格和塊配置

**RSSI 核函數:**
- **X 維度**: 跨越子載波（16-128 線程/塊）
- **Y 維度**: UE 組
- **Z 維度**: 不使用

**RSRP 核函數:**
- **X 維度**: 跨越子載波
- **Y 維度**: 層/UE 索引
- **Z 維度**: 不使用

### 共享記憶體使用

```cuda
// RSSI 核函數
- 符號位置存儲
- 臨時功率累計緩衝

// RSRP 核函數
- 通道估計矩陣（共享記憶體副本）
- 層/RxAnt 映射表
```

### 同步和歸約

- **Warp 級別**: 使用 `cg::reduce()` 進行高效歸約
- **塊級別**: 使用 `atomicAdd()` 進行全局記憶體聚合
- **線程同步**: `__syncthreads()` 用於塊內協調

## 測量指標 (Measurement Metrics)

### RSSI (Received Signal Strength Indicator)

**定義:**
- 接收信號總功率（所有分配的 PRB、符號、天線求和）
- FAPIv3 表 3-16: "RSSI 報告為跨所有天線求和的總接收功率"

**單位:**
- 線性功率 (Watts)
- dB 轉換: $RSSI_{dB} = 10 \log_{10}(P_{lin})$

**時間粒度:**
- 全時隙: 整個時隙的平均值
- 子時隙: 早期 HARQ 的部分時隙值

### RSRP (Reference Signal Received Power)

**定義:**
- DMRS 參考信號的平均功率
- 計算: $RSRP = E[|H_{est}|^2]$
- 不包括 DMRS 功率因子（amplitude normalization）

**計算維度:**
- 平均化: 分配 PRB 數量、接收天線、時間領域估計
- 求和: 層（不平均化層）

**質量指標:**
- 可靠的通道質量估計
- 用於功率控制和自適應調制

### SINR (Signal-to-Interference-plus-Noise Ratio)

**預均衡 SINR:**
$$SINR_{preEq,dB} = RSRP_{dB} - N0_{preEq,dB}$$

**後均衡 SINR:**
$$SINR_{postEq,dB} = RSRP_{dB} - N0_{postEq,dB}$$

其中 $N0$ 包含噪聲和干擾功率。

## 輸出格式 (Output Formats)

### FAPI 相容性

根據 3GPP FAPIv3-v4：

```cpp
// 測量輸出
struct {
    uint32_t RSSI;           // [0:1280] => [-128:0.1:0] dBm
    uint32_t RSSI_ehq;       // 早期 HARQ RSSI
    uint32_t RSRP;           // [0:1280] => [-128:0.1:0] dBm
    uint32_t RSRP_ehq;       // 早期 HARQ RSRP
    int16_t  sinrdB;         // SINR (dB)
    int16_t  sinr_postEq_dB; // 後置均衡 SINR (dB)
} measurements;
```

### 校準和偏移

**增益校準:**
- O-RU dBm 值對應基帶 0 dB 功率
- 通常通過 O-RU 規格/校準獲得
- 應用於 RSSI 和 RSRP 報告

## Python 綁定 (Python Binding)

### RsrpEstimator 類

```python
class RsrpEstimator:
    def estimate(
        self,
        rx_slot: Array,           # 接收信號
        channel_est: List[Array], # 通道估計
        ree_diag_inv: List[Array],# 均衡器係數逆矩陣
        noise_var_pre_eq: Array   # 預均衡噪聲
    ) -> Tuple[Array, Array, Array]:
        # 返回: rsrp, sinr_pre_eq, sinr_post_eq
```

## 性能特性 (Performance Characteristics)

### 優化技術

1. **合作組歸約**: 高效的 warp 級別並行歸約
2. **全局記憶體合併**: 連貫的記憶體訪問模式
3. **指令級並行**: 操作重疊隱藏記憶體延遲
4. **多核心利用**: 每塊多個線程組並行執行

### 記憶體需求

**動態分配:**
- RSSI 描述符向量: ~數 KB
- RSRP 描述符向量: ~數 KB
- 輸出張量:
  - RSSI: $N_{UeGrp} \times 4$ 字節
  - RSRP: $N_{Ue} \times 4$ 字節
  - SINR: $N_{Ue} \times 8$ 字節 (preEq + postEq)

## 支援的模式 (Supported Modes)

| 模式 | 符號覆蓋 | 用途 | 說明 |
|------|---------|------|------|
| 全時隙 | 全 14 符號 | 完整測量 | 標準上行鏈路報告 |
| 子時隙 | 部分符號 | 早期 HARQ | 快速反饋，前 N 個 DMRS 符號 |

## 集成與使用 (Integration and Usage)

### 在 PUSCH 接收管道中的角色

```
接收信號 → 通道估計 → RSSI/RSRP 測量 → SINR 報告
                     ↓
                  功率控制
                  
通道估計 → 噪聲干擾估計 → RSSI/RSRP 測量 → 均衡
                                ↓
                            後置 SINR 計算
```

### 關鍵 API 函數

```cpp
// 創建句柄
cuphyStatus_t cuphyCreatePuschRxRssi(
    cuphyPuschRxRssiHndl_t* pHandle
);

// 設置 RSSI 測量
cuphyStatus_t cuphySetupPuschRxRssi(
    cuphyPuschRxRssiHndl_t handle,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrms,
    // ... 其他參數
);

// 設置 RSRP 測量
cuphyStatus_t cuphySetupPuschRxRsrp(
    cuphyPuschRxRssiHndl_t handle,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrms,
    // ... 其他參數
);

// 銷毀
cuphyStatus_t cuphyDestroyPuschRxRssi(
    cuphyPuschRxRssiHndl_t handle
);
```

## 診斷和調試 (Diagnostics and Debugging)

### 可用的調試信息

啟用 `ENABLE_DEBUG` 時提供:

- 接收信號樣本
- 通道估計值
- 功率累計中間值
- 最終 RSSI/RSRP 計算結果
- 線程塊和線程索引信息

### 常見配置

```cpp
// 調試輸出
#define ENABLE_DEBUG

// 性能分析
#define ENABLE_PROFILING

// 內存追蹤
#define ENABLE_MEMTRACE
```

## 相關檔案 (Related Files)

- [pusch_noise_intf_est.md](pusch_noise_intf_est.md) - 噪聲干擾估計
- [pusch_receiver.md](pusch_receiver.md) - PUSCH 接收管道總體架構
- [ch_est.md](ch_est.md) - 通道估計
- [cfo_ta_est.md](cfo_ta_est.md) - 頻偏和時間提前估計

## 參考文獻 (References)

- 3GPP TS 38.211: NR 物理通道和調製
- 3GPP TS 38.214: NR 物理層程序進行上行傳輸
- 5G FAPI 規格: 測量和指標定義
- cuPHY API 文檔: RSSI/RSRP 測量接口

## 版本信息 (Version Information)

- **SPDX-License-Identifier**: Apache-2.0
- **SPDX-FileCopyrightText**: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
- **實現語言**: CUDA C++
- **支援計算能力**: SM 7.0 及以上
