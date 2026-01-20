# PUSCH Noise and Interference Estimation

## 概述 (Overview)

PUSCH (Physical Uplink Shared Channel) 噪聲與干擾估計是 cuPHY 上行鏈路接收管道中的關鍵元件，負責估計接收信號中的噪聲和干擾功率。這些估計用於頻域均衡、SINR 計算、DTX 檢測等後續處理步驟。

## 檔案結構 (File Structure)

```
cuPHY/src/cuphy/pusch_noise_intf_est/
├── pusch_noise_intf_est.hpp      # 頭文件 - 類定義和接口
├── pusch_noise_intf_est.cu       # CUDA 核函數實現
└── (集成在 cuPHY 框架中)

pyaerial/pybind11/
├── pycuphy_noise_intf_est.hpp    # Python 綁定頭文件
└── pycuphy_noise_intf_est.cpp    # Python 綁定實現
```

## 核心功能 (Core Functions)

### 1. 主類: `puschRxNoiseIntfEst`

```cpp
namespace pusch_noise_intf_est {
    class puschRxNoiseIntfEst : public cuphyPuschRxNoiseIntfEst
    {
    public:
        // 設置和執行
        cuphyStatus_t setup(
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsCpu,
            cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrmsGpu,
            uint16_t nUeGrps,
            uint32_t nMaxPrb,
            uint8_t enableDftSOfdm,
            uint8_t isEarlyHarq,
            cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
            cudaStream_t strm,
            uint8_t subSlotStageIdx
        );
        
        // 描述符信息
        static void getDescrInfo(
            size_t& dynDescrSizeBytes, 
            size_t& dynDescrAlignBytes
        );
    };
}
```

### 2. 核函數: `noiseIntfEstKernel`

```cuda
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_DMRS_GRIDS_PER_PRB,
          uint32_t N_PRB_PER_THRD_BLK,
          uint8_t  DMRS_SYMBOL_IDX>
__global__ void noiseIntfEstKernel(
    const __grid_constant__ puschRxNoiseIntfEstDynDescr_t dynDescr
)
```

**核函數職責:**
- 計算 DMRS 符號位置的噪聲干擾功率
- 計算噪聲協方差矩陣 (Noise Covariance Matrix)
- 進行 Cholesky 因式分解和矩陣求逆
- 計算線性最小均方誤差 (LMMSE) 等化器係數

## 數據流 (Data Flow)

### 輸入張量 (Input Tensors)

| 張量 | 形狀 | 類型 | 說明 |
|------|------|------|------|
| `tDataRx` | (NF, ND, N_BS_ANTS) | 複數 | 接收信號，NF=頻率子載波數，ND=數據符號數，N_BS_ANTS=基站天線數 |
| `tHEst` | (N_BS_ANTS, N_LAYERS, NF, NH) | 複數 | 通道估計值 |
| `tDmrsSymbol` | (NF_DMRS, N_LAYERS) | 複數 | DMRS 參考信號 |

### 輸出張量 (Output Tensors)

| 張量 | 形狀 | 類型 | 說明 |
|------|------|------|------|
| `tNoiseVarPreEq` | (N_UE_GRP,) 或 (N_UE,) | 實數 | 均衡前噪聲功率 (dB) |
| `tLwInv` | (N_BS_ANTS, N_BS_ANTS, N_PRB) | 複數 | Cholesky 因式分解逆矩陣 |
| `tInfoNoiseIntfEstInterCtaSyncCnt` | (N_UE_GRP,) | 無符號整數 | 線程塊同步計數器 |

## 算法流程 (Algorithm Flow)

### 步驟 1: 噪聲干擾功率估計

對於每個 DMRS 符號位置，估計接收信號與通道估計預測信號的差異:

$$r_{tilde} = r_{received} - H_{est} \cdot x_{dmrs}$$

其中:
- $r_{received}$ = 接收信號
- $H_{est}$ = 通道估計
- $x_{dmrs}$ = DMRS 參考信號

### 步驟 2: 功率計算

計算噪聲干擾功率：

$$P_{noise} = \frac{1}{N_{DMRS}} \sum |r_{tilde}|^2$$

應用修正因子補償通道插值濾波器濾除的噪聲:

$$P_{noise,dB} = 10 \log_{10}(P_{noise}) + 0.5 \text{ dB}$$

其中 0.5 dB 是實驗確定的修正值。

### 步驟 3: 噪聲協方差矩陣估計

對於每個 PRB，計算接收天線間的噪聲協方差矩陣:

$$R_{ww} = E[r_{tilde} \cdot r_{tilde}^H]$$

尺寸為 $(N_{RxAnt} \times N_{RxAnt})$

### 步驟 4: 矩陣求逆

當 `compCovFlag` 設置時，進行:

1. **Cholesky 因式分解**: $R_{ww} = L \cdot L^H$
2. **逆矩陣計算**: $R_{ww}^{-1} = (L^{-1})^H \cdot L^{-1}$
3. **結果儲存**: 用於後續 LMMSE 均衡

### 步驟 5: 自適應正則化 (可選)

支援 MMSE-IRC (干擾拒絕組合) 演算法:

$$\rho_{adaptive} = \frac{(T-2)/T \cdot \text{trace}(R_{ww}^2) + (\text{trace}(R_{ww}))^2}{(T+2)(\text{trace}(R_{ww}^2) - (\text{trace}(R_{ww}))^2/N_{Ant})}$$

其中 $T = N_{DMRS\_SC} \times N_{DMRS\_Symbols}$

## 關鍵參數 (Key Parameters)

### 動態描述符 (Dynamic Descriptor)

```cpp
struct puschRxNoiseIntfEstDynDescr
{
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;      // UE 組參數
    bool                     compCovFlag;         // 計算協方差標誌
    uint8_t                  subSlotStageIndex;   // 子時隙階段索引
};
```

### 核函數配置

| 參數 | 說明 |
|------|------|
| `N_PRB_PER_THRD_BLK` | 每個線程塊處理的 PRB 數量 |
| `N_DMRS_GRIDS_PER_PRB` | 每個 PRB 的 DMRS 網格數量 (通常為 2) |
| `DMRS_SYMBOL_IDX` | 處理的 DMRS 符號索引 |

### 支援的數據類型組合

| DataRx | ChEst | NoiseVarPreEq | LwInv | 說明 |
|--------|-------|---------------|-------|------|
| C16F | C32F | R32F | C32F | 主要配置 |
| C32F | C32F | R32F | C32F | 高精度模式 |

其中 C = Complex, R = Real, 16F = 16 位浮點, 32F = 32 位浮點

## 並行化策略 (Parallelization Strategy)

### 網格和塊配置

- **網格維度**: 沿著 UE 組和 PRB 進行分解
- **塊維度**: 處理頻率子載波和接收天線
- **共享記憶體**:
  - `nShRwwElems`: 噪聲協方差矩陣元素
  - `nShTxDmrsElems`: DMRS 參考信號
  - `nShNoiseIntfEstElems`: 噪聲干擾估計中間結果

### 線程同步

- 使用 `__syncthreads()` 同步線程塊內線程
- 使用合作組 (Cooperative Groups) 進行 warp 級別歸約
- CTA 級別同步用於矩陣求逆操作

## Python 綁定 (Python Binding)

### PyNoiseIntfEstimator 類

```python
class PyNoiseIntfEstimator:
    def estimate(
        self,
        chEst: List[CudaArray],      # 通道估計
        puschParams: PuschParams      # PUSCH 參數
    ) -> List[CudaArray]:            # 返回 Lw_inv 矩陣
        # 運行估計
        # 返回線性加權逆矩陣列表
```

### 使用例子

```python
# 創建估計器
noise_intf_est = pycuphy.NoiseIntfEstimator(cuda_stream)

# 設置參數和運行估計
lw_inv = noise_intf_est.estimate(channel_est, pusch_params)

# 獲取輸出
noise_var_pre_eq = noise_intf_est.get_info_noise_var_pre_eq()
```

## 性能特性 (Performance Characteristics)

### 優化技術

1. **共享記憶體優化**: 最大化 L1 快取使用和頻寬
2. **合作組歸約**: 高效的 warp 級別並行歸約
3. **DFT-Spread-OFDM 支持**: 針對 SC-FDM 信號的特殊核函數
4. **子時隙處理**: 早期 HARQ 反饋支持

### 記憶體需求

對於最大配置:

- **動態描述符**: ~幾 KB (取決於 UE 組數量)
- **臨時張量**: 
  - 噪聲協方差矩陣: $N_{RxAnt}^2 \times N_{PRB} \times 4$ 字節 (32 位浮點)
  - 噪聲功率: $N_{UeGrp} \times 4$ 字節

## 集成與使用 (Integration and Usage)

### 在 PUSCH 接收管道中的角色

```
接收信號 → 通道估計 → 噪聲干擾估計 → 均衡 → 解調解碼
                          ↓
                    - 噪聲功率
                    - 協方差矩陣
                    - SINR 計算
```

### 關鍵 API 函數

```cpp
// 創建句柄
cuphyStatus_t cuphyCreatePuschRxNoiseIntfEst(
    cuphyPuschRxNoiseIntfEstHndl_t* pHandle
);

// 設置和執行
cuphyStatus_t cuphySetupPuschRxNoiseIntfEst(
    cuphyPuschRxNoiseIntfEstHndl_t handle,
    cuphyPuschRxUeGrpPrms_t* pUeGrpPrms,
    // ... 其他參數
);

// 銷毀
cuphyStatus_t cuphyDestroyPuschRxNoiseIntfEst(
    cuphyPuschRxNoiseIntfEstHndl_t handle
);
```

## 支援的模式 (Supported Modes)

| 模式 | 說明 |
|------|------|
| 全時隙處理 | 處理整個 14 符號的時隙 |
| 子時隙處理 | 早期 HARQ 反饋，處理部分符號 |
| DFT-S-OFDM | SC-FDM 信號支持 (標誌: `enableDftSOfdm`) |
| 多天線配置 | 支持可變接收天線數量 |

## 診斷和調試 (Diagnostics and Debugging)

### 可用的調試信息

啟用 `ENABLE_DEBUG` 時提供:

- DMRS 符號信息
- 接收信號和通道估計值
- 中間計算結果 (噪聲干擾估計、接收的導頻)
- 矩陣運算信息 (Cholesky 分解、求逆)

### 常見設置

```cpp
// 調試輸出
#define ENABLE_DEBUG

// 性能分析
#define ENABLE_PROFILING

// 特殊處理模式
#define ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
```

## 相關文件 (Related Files)

- [ch_est.md](ch_est.md) - 通道估計模組
- [pusch_receiver.md](pusch_receiver.md) - PUSCH 接收管道總體架構
- [Gold_sequence.md](Gold_sequence.md) - DMRS 生成
- [cfo_ta_est.md](cfo_ta_est.md) - 頻偏和時間提前估計

## 參考文獻 (References)

- 3GPP TS 38.211: NR 物理通道和調製
- 3GPP TS 38.214: NR 物理層程序進行上行傳輸
- cuPHY API 文檔: PUSCH 噪聲干擾估計接口

## 版本信息 (Version Information)

- **SPDX-License-Identifier**: Apache-2.0
- **SPDX-FileCopyrightText**: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
