# SRS Transmission (cuPHY 上行聲測參考信號發送模組)

## 概述

**SRS Transmitter (SrsTx)** 是 NVIDIA cuPHY 中用於上行 Sounding Reference Signal (SRS) 發送的 GPU 加速模組。SRS 是 5G NR 標準中定義的參考信號，由基站用於測量上行鏈路信道品質，並用於自適應波束成形、功率控制和信道狀態反饋。該模組支持多個天線端口、複雜的頻率跳轉配置、Zadoff-Chu 序列生成以及低峰均比 (Low PAPR) 降低技術。

**核心特性：**
- 支持多個天線端口 (1/2/4/6/8/12)
- Zadoff-Chu 序列動態生成
- Low PAPR 表查詢和序列組合
- 疊頻配置支持 (Comb-2/Comb-4)
- 周期、半持久和非周期 SRS 資源類型
- 循環移位、頻率跳轉和組序列跳躍
- FP16 混合精度優化
- CUDA 圖支持

---

## 軟體架構

### 1. 核心類別

#### `SrsTx`

SRS 發送的主要 C++ 類，實現 SRS 訊號生成和 GPU 執行。

```cpp
class SrsTx : public cuphySrsTx {
public:
    static constexpr int Ng = 273*2*3;  // Max subcarrier coverage
    
    enum Component {
        SRSTX_PARAMS       = 0,      // Parameter buffer
        SRSTX_TENSOR_ADDR  = 1,      // Output tensor address array
        N_SRSTX_COMPONENTS = 2
    };
    
    explicit SrsTx(cuphySrsTxStatPrms_t const* pStatPrms);
    ~SrsTx();
    
    // Configuration
    const cuphySrsTxStatPrms_t* static_params{};
    const cuphySrsTxDynPrms_t*  dynamic_params{};
    
    // Parameter expansion and execution
    cuphyStatus_t expandParameters(cuphySrsTxDynPrms_t* dyn_params, 
                                   cudaStream_t cuda_strm);
    void setKernelParams();
    cuphyStatus_t run(const cudaStream_t& cuda_strm) const;
    
    // Configuration validation and utilities
    cuphyStatus_t checkConfig(SrsTxParams* params, int numSrsTxParams);
    template <fmtlog::LogLevel log_level>
    void printSrsTxConfig(const cuphySrsTxStatPrms_t& static_params,
                          const cuphySrsTxDynPrms_t& dyn_params);
    
    const void* getMemoryTracker();
    CUgraph *GetGraph() { return &m_graph; }

private:
    // 工作區管理
    void updateWorkspaceOffsets(int nSrsTxParams);
    void updateWorkspacePtrs();
    
    // CUDA 圖管理
    void createGraph();
    void updateGraph();
};
```

**主要方法：**
- `expandParameters()`: 將動態參數轉換為 GPU 可使用的格式
- `setKernelParams()`: 配置核心啟動參數
- `run()`: 執行 SRS 發送核心
- `checkConfig()`: 驗證參數有效性

### 2. 數據結構

#### SrsTx 參數結構 (SrsTxParams)

```cpp
struct SrsTxParams {
    uint8_t nAntPorts;              // 天線端口數
    uint16_t nSrsSubcarriers;       // SRS 子載波數
    uint8_t combSize;               // 疊頻大小 (2 或 4)
    uint8_t combOffset;             // 疊頻偏移
    uint8_t cyclicShift;            // 循環移位 (0-11)
    uint8_t frequencyPosition;      // 頻域位置 (0-67)
    uint16_t frequencyShift;        // 頻域移位
    uint8_t frequencyHopping;       // 頻率跳轉模式 (0-3)
    uint8_t resourceType;           // 資源類型 (0:非周期, 1:半持久, 2:周期)
    uint16_t Tsrs;                  // SRS 周期 (時隙)
    uint16_t Toffset;               // 時隙偏移
    uint8_t groupOrSequenceHopping; // 序列跳躍開關
    uint16_t idxSlotInFrame;        // 幀內時隙索引
    uint16_t idxFrame;              // 幀索引
    uint16_t nSlotsPerFrame;        // 每幀時隙數
    uint16_t nSymbsPerSlot;         // 每時隙符號數
};
```

#### 靜態參數 (cuphySrsTxStatPrms_t)

```cpp
typedef struct _cuphySrsTxStatPrms {
    uint16_t nMaxSrsUes;            // 最大 SRS UE 數
    uint16_t nSlotsPerFrame;        // 每幀時隙數
    uint16_t nSymbsPerSlot;         // 每時隙符號數
    cuphyTracker_t* pOutInfo;       // 記憶體足跡追蹤
} cuphySrsTxStatPrms_t;
```

#### 動態參數 (cuphySrsTxDynPrms_t)

```cpp
typedef struct _cuphySrsTxDynPrms {
    cudaStream_t cuStream;          // CUDA 串流
    CUgraph *chan_graph;            // 計算圖指針
    uint16_t nSrsUes;               // 活躍 UE 數
    cuphyUeSrsTxPrm_t* pUeSrsTxPrms;// UE 參數陣列
    cuphyTensorPrm_t* pTDataSrsTx;  // 輸出張量參數
} cuphySrsTxDynPrms_t;
```

---

## SRS 發送演算法

### 1. 序列生成流程

#### 步驟 1: Zadoff-Chu 基序列生成

對於每個 UE，生成基 ZC 序列：

$$r_u(m) = e^{-j\pi u m(m+1) / M_{ZC}}$$

其中：
- $u$ 是序列索引 (0-29)
- $m$ 是樣本索引 (0 至 $M_{ZC}-1$)
- $M_{ZC}$ 是 ZC 序列長度 (通常為 137-1019)

#### 步驟 2: Low PAPR 表查詢

應用 Low PAPR 覆蓋碼以降低峰均比：

$$r'_{u,v}(m) = r_u(m) \cdot c_{u,v}(m)$$

其中 $c_{u,v}$ 來自預計算的查表 (srsLowPaprTable0 或 srsLowPaprTable1)

#### 步驟 3: FOCC 乘以並應用循環移位

應用頻率正交組合碼 (FOCC) 進行頻域處理：

$$s(k,l) = r'_{u,v}(k) \cdot \text{FOCC}(k) \cdot e^{j2\pi n_{cs} k / 12}$$

其中：
- $n_{cs}$ 是循環移位參數
- 對於四天線端口，某些端口使用 $n_{cs} + 6$ 偏移

#### 步驟 4: 頻率跳轉和嵌入

根據跳轉模式動態調整起始子載波，嵌入到時頻資源網格中。

### 2. 工作空間管理

**工作空間佈局：**

```
Offset 0:
┌─────────────────────────────────┐
│ SrsTxParams[N_UEs]              │ (每個 UE 一個結構)
├─────────────────────────────────┤
│ __half2* h_ue_tensor_addr[N_UEs]│ (輸出張量指針數組)
├─────────────────────────────────┤
│ Padding/Alignment               │
└─────────────────────────────────┘
```

**大小計算：**

```cpp
workspace_offsets[1] = round_up(nSrsUes * sizeof(SrsTxParams), 64);
workspace_offsets[2] = workspace_offsets[1] + 
                       round_up(nSrsUes * sizeof(__half2*), 64);
```

---

## CUDA 核心實現

### 主要核心函數

#### `genSrsTx` 核心

生成所有 UE 的 SRS 訊號的主核心。

```cu
__global__ void genSrsTx(SrsTxParams* srstx_params, 
                         __half2** tfSignalArray)
{
    // 塊對應一個 UE，執行緒處理子載波
    int ueIdx = blockIdx.x;
    if(ueIdx >= blockDim.x) return;
    
    SrsTxParams& param = srstx_params[ueIdx];
    __half2* output = tfSignalArray[ueIdx];
    
    // 步驟 1: 生成 Zadoff-Chu 序列
    uint16_t u = computeSeqIndex(param.sequenceId, 
                                 param.groupOrSequenceHopping);
    uint16_t v = computeGroupIndex(param.sequenceId);
    
    // 步驟 2: 查詢 Low PAPR 表並應用
    for(int sc = threadIdx.x; sc < param.nSrsSubcarriers; 
        sc += blockDim.x) {
        float2 zc_val = computeZcElement(u, v, sc);
        
        // 應用循環移位和 FOCC
        float phase = 2.0f * M_PI * param.cyclicShift * sc / 12.0f;
        zc_val = rotatePhasor(zc_val, phase);
        
        // 嵌入到時頻網格
        embedIntoGrid(output, sc, zc_val, param);
    }
}
```

#### Zadoff-Chu 元素計算

```cu
__device__ __inline__ float2 computeZcElement(uint16_t u, uint16_t v, 
                                               uint16_t m)
{
    // Zadoff-Chu 序列：exp(-j*pi*u*m*(m+1)/M)
    float M = 1024.0f;  // 典型序列長度
    float angle = -M_PI * u * m * (m + 1) / M;
    
    float2 result;
    __sincosf(angle, &result.y, &result.x);
    
    // 應用 Low PAPR 覆蓋碼
    int tableSel = (v == 0) ? 0 : 1;
    float2 cover = getLowPaprCode(tableSel, u, m);
    
    return complexMul(result, cover);
}
```

### 核心啟動配置

**標準 SRS TX 核心：**

```cuda
dim3 gridDim(nSrsUes);           // 每個 UE 一個塊
dim3 blockDim(256);               // 每塊 256 執行緒
size_t sharedMemBytes = 16*1024;  // 共享記憶體

cuLaunchKernel(genSrsTx, 
               gridDim.x, gridDim.y, gridDim.z,
               blockDim.x, blockDim.y, blockDim.z,
               sharedMemBytes,
               cuStream,
               kernelArgs,
               NULL);
```

### 共享記憶體使用

```cpp
// 共享記憶體佈局 (~16KB)
__shared__ float2 sh_zcTable[1024];      // ZC 查表
__shared__ float2 sh_lowPaprTable[512];  // Low PAPR 代碼
__shared__ float  sh_phaseRamp[256];     // 相位旋轉
__shared__ __half2 sh_output[512];       // 中間輸出
```

---

## 數據流

### 輸入參數

1. **SRS 參數** (SrsTxParams)
   - 天線端口、疊頻配置、序列索引
   - 時隙和幀信息
   - 循環移位和頻率跳轉設置

2. **靜態參數** (cuphySrsTxStatPrms_t)
   - 最大 UE 數
   - 時隙和符號配置

3. **動態參數** (cuphySrsTxDynPrms_t)
   - 活躍 UE 數
   - 當前時隙和幀索引
   - CUDA 串流

### 輸出張量

1. **SRS 發送訊號** (tfSignalArray)
   - 維度: [MAX_N_PRBS] x [OFDM_SYMBOLS_PER_SLOT] x [NUM_TX_ANT]
   - 類型: `CUPHY_C_16F` (複數 FP16)
   - 佈局: 緊湊行優先

### 記憶體大小估計

```
每個 UE 的輸出大小 = 
  273 PRBs * 12 子載波 * 14 符號 * 4 天線 * 4 bytes (FP16)
  ≈ 5.7 MB

最大總記憶體 (64 UEs):
  64 * 5.7 MB ≈ 365 MB
```

---

## C API 接口

### `cuphyCreateSrsTx()`

創建並初始化 SRS 發送器

```cpp
cuphyStatus_t cuphyCreateSrsTx(
    cuphySrsTxHndl_t* pSrsTxHndl,
    cuphySrsTxStatPrms_t const* pStatPrms)
{
    if((pSrsTxHndl == nullptr) || (pStatPrms == nullptr)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    auto* new_pipeline = new (std::nothrow) SrsTx(pStatPrms);
    if(new_pipeline == nullptr) {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    *pSrsTxHndl = new_pipeline;
    return CUPHY_STATUS_SUCCESS;
}
```

### `cuphySetupSrsTx()`

配置動態參數

```cpp
cuphyStatus_t cuphySetupSrsTx(
    cuphySrsTxHndl_t srsTxHndl, 
    cuphySrsTxDynPrms_t* pDynPrms)
{
    if((pDynPrms == nullptr) || (srsTxHndl == nullptr)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    SrsTx* pipeline = static_cast<SrsTx*>(srsTxHndl);
    pDynPrms->chan_graph = pipeline->GetGraph();
    return pipeline->expandParameters(pDynPrms, pDynPrms->cuStream);
}
```

### `cuphyRunSrsTx()`

執行 SRS 發送

```cpp
cuphyStatus_t cuphyRunSrsTx(cuphySrsTxHndl_t srsTxHndl)
{
    if(srsTxHndl == nullptr) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    SrsTx* pipeline = static_cast<SrsTx*>(srsTxHndl);
    return pipeline->run(pipeline->dynamic_params->cuStream);
}
```

### `cuphyDestroySrsTx()`

銷毀 SRS 發送器

```cpp
cuphyStatus_t cuphyDestroySrsTx(cuphySrsTxHndl_t srsTxHndl)
{
    if(srsTxHndl == nullptr) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    SrsTx* pipeline = static_cast<SrsTx*>(srsTxHndl);
    delete pipeline;
    return CUPHY_STATUS_SUCCESS;
}
```

---

## 使用範例

### 基本 SRS 發送

```cpp
// 1. 初始化靜態參數
cuphySrsTxStatPrms_t statPrms{};
statPrms.nMaxSrsUes = 16;
statPrms.nSlotsPerFrame = 10;
statPrms.nSymbsPerSlot = 14;

cuphy::cuphy_tracker tracker;
statPrms.pOutInfo = &tracker;

// 2. 創建發送器
cuphySrsTxHndl_t srsTxHndl;
cuphyCreateSrsTx(&srsTxHndl, &statPrms);

// 3. 分配輸出緩衝區
std::vector<__half2*> txBuffers(statPrms.nMaxSrsUes);
for(int i = 0; i < statPrms.nMaxSrsUes; ++i) {
    size_t bufSize = 273 * 12 * 14 * sizeof(__half2);
    cudaMalloc(&txBuffers[i], bufSize);
}

// 4. 準備 UE 參數
std::vector<cuphyUeSrsTxPrm_t> uePrms(nSrsUes);
for(int i = 0; i < nSrsUes; ++i) {
    uePrms[i].nAntPorts = 4;
    uePrms[i].combSize = 2;
    uePrms[i].cyclicShift = i % 12;
    uePrms[i].sequenceId = i;
    // ... 其他參數
}

// 5. 準備張量參數
std::vector<cuphyTensorPrm_t> txTensorPrms(nSrsUes);
for(int i = 0; i < nSrsUes; ++i) {
    txTensorPrms[i].pAddr = txBuffers[i];
    txTensorPrms[i].desc = createTensorDesc(273, 12, 14);
}

// 6. 配置動態參數
cuphySrsTxDynPrms_t dynPrms{};
dynPrms.cuStream = cuStream;
dynPrms.chan_graph = nullptr;
dynPrms.nSrsUes = nSrsUes;
dynPrms.pUeSrsTxPrms = uePrms.data();
dynPrms.pTDataSrsTx = txTensorPrms.data();

// 7. 設置並執行
cuphySetupSrsTx(srsTxHndl, &dynPrms);
cuphyRunSrsTx(srsTxHndl);

// 8. 同步並回收
cudaStreamSynchronize(cuStream);
for(auto ptr : txBuffers) cudaFree(ptr);
cuphyDestroySrsTx(srsTxHndl);
```

---

## Python 綁定接口

### `PySrsTx`

Python 用戶可透過 pybind11 包裝使用 SRS 發送：

```python
from pyaerial.phy5g.srs import SrsTx

# 初始化
tx = SrsTx(
    num_max_srs_ues=16,
    num_slot_per_frame=10,
    num_symb_per_slot=14,
    cuda_stream=cuda_stream_handle
)

# 執行 SRS 發送
tx_signals = tx.run(
    slot_index=0,
    frame_index=0,
    srs_configs=srs_config_list  # 每個 UE 一個配置
)

# 結果: List[CuPy Array]
# 每個陣列: [273 PRBs, 12 subcarriers, 14 symbols, 4 TX ants]
# 數據類型: complex64 (FP32)
```

**配置物件結構：**

```python
class SrsConfig:
    def __init__(self):
        self.nAntPorts = 4
        self.combSize = 2
        self.cyclicShift = 0
        self.sequenceId = 0
        self.frequencyPosition = 0
        self.frequencyShift = 0
        self.frequencyHopping = 0
        self.resourceType = 0  # 非周期
        self.Tsrs = 80  # 周期
        self.Toffset = 0
```

---

## 效能特性

### 計算複雜度

**每個 UE 的運算數：**
- ZC 序列生成: ~1024 複數乘法
- Low PAPR 應用: 表查詢 (O(1))
- 嵌入: ~3000 複數乘加

**總體複雜度:** O(nUes × nSubcarriers × nSymbols × nAntPorts)

### GPU 資源使用

- **暫存器**: ~64-96 個 (含地址計算)
- **共享記憶體**: 14-16 KB (LUT + 臨時值)
- **全局記憶體**: 100-200 GB/s 峰值帶寬

### 執行時間估計

**在 A100 GPU 上：**
- 16 UE，1 時隙: ~50-100 μs
- 64 UE，1 時隙: ~200-400 μs
- 理論最大吞吐: 1000+ UEs/ms

---

## 故障排除

### 常見問題

**Q: 發送訊號功率太低？**
- A: 驗證 Low PAPR 表是否正確載入
- 檢查循環移位是否正確應用
- 確認天線端口配置

**Q: 峰均比仍然很高？**
- A: 驗證 Low PAPR 表查詢邏輯
- 檢查 u/v 索引計算
- 確認 FOCC 應用位置

**Q: 記憶體不足？**
- A: 減少最大 UE 數
- 使用流式處理分塊數據
- 優化輸出緩衝區大小

**Q: 性能低於預期？**
- A: 檢查共享記憶體競爭
- 驗證核心佔用率
- 考慮使用 CUDA 圖優化

---

## 相關模組整合

### 與 SRS 接收的互操作

SRS 發送生成的訊號由接收端 (`srs_chEst`) 用於信道估計：

```
UE (SrsTx) → 無線通道 → Base Station (SrsRx) → ChEst
```

### 配合波束成形迴路

SRS 發送支持基站反饋的波束成形設置：

```cpp
// 從信道估計接收反饋
chEst -> SNR 計算
     -> 波束選擇
     -> 發送 PMI (預編碼矩陣指示)
     
// UE 在下一個 SRS 時隙應用新的配置
```

---

## 參考資源

- [NVIDIA cuPHY 文檔](https://docs.nvidia.com/networking/aerial/)
- [3GPP TS 38.211 - NR 物理層 (SRS 定義)](https://www.3gpp.org)
- [3GPP TS 38.214 - NR 物理層流程](https://www.3gpp.org)
- 相關模組: `srs_chEst`, `srs_rx`, `bfw_tx`

---

## 附錄：Low PAPR 表結構

### srsLowPaprTable0

標準 Low PAPR 覆蓋碼表，大小為 [30][1024]

```matlab
% MATLAB 驗證
for u = 0:29
    for m = 0:1023
        alpha = pi * u * m * (m+1) / 1024;
        low_papr(u+1, m+1) = exp(-j * alpha);
    end
end
```

### srsLowPaprTable1

交替 Low PAPR 表，用於 v=1 序列

```matlab
for u = 0:29
    d_prime = nextPrime(u);
    q_bar = d_prime * (u + 1) / 31;
    for m = 0:1023
        q = floor(q_bar + 0.5);
        low_papr_1(u+1, m+1) = exp(-j * pi * q * m * (m+1) / d_prime);
    end
end
```

