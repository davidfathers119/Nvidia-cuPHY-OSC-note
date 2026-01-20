# NVIDIA cuPHY 同步信號（SS）模組文檔

## 1. 概述

同步信號（SS）模組是NVIDIA cuPHY的核心下行鏈路元件，負責物理層同步和系統資訊播送。該模組實現Primary/Secondary Synchronization Signal（PSS/SSS）檢測、PBCH（Physical Broadcast Channel）解調和Master Information Block（MIB）提取，支持多波束掃描和假設測試。

**核心功能：**
- Primary/Secondary Synchronization Signal（PSS/SSS）生成與檢測
- PBCH編碼、調制與映射
- Master Information Block（MIB）CRC驗證與提取
- 多波束掃描與假設排序
- 物理層同步和時序對齊
- Cell ID識別（NID計算）
- GPU加速信號處理

**應用場景：** 下行鏈路接收、小區搜索、系統資訊獲取、波束對齊

## 2. 架構設計

### 2.1 核心類與組件

#### 2.1.1 SsbTx類（SSB傳輸）

```cpp
class SsbTx : public cuphySsbTx {
    enum Component {
        SSB_PER_CELL_PARAMS = 0,      // Per-cell參數
        PER_SS_BLOCK_PARAMS = 1,      // Per-SSB參數
        SSB_INPUT_W_CRC = 2,          // PBCH輸入+CRC
        CELL_OUTPUT_ADDR = 3,         // Cell輸出地址
        SSB_PRECODING_MATRIX = 4,     // 預編碼矩陣
        N_SSB_COMPONENTS = 5
    };
    
    // Static constraints
    static const int Nc = 1600;               // PN序列起始索引
    static const uint32_t G_CRC_24_C = 0x01B2B117; // CRC24多項式
    
    // 核心方法
    SsbTx(const cuphySsbStatPrms_t* cfg_static_params);
    ~SsbTx();
    
    cuphyStatus_t expandParameters(cuphySsbDynPrms_t* dyn_params, cudaStream_t cuda_strm);
    cuphyStatus_t setup(cuphySsbDynPrms_t* dyn_params);
    cuphyStatus_t run(const cudaStream_t& cuda_strm);
    
    cuphyStatus_t preparePBCH(uint32_t* h_x_mib,
                              const cuphyPerSsBlockDynPrms_t* h_ssb_params,
                              const cuphyPerCellSsbDynPrms_t* h_per_cell_params,
                              cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                              cuphySsbMapperLaunchCfg_t* pSsbMapperLaunchCfg);
    
    void writeDbgBufSynch(cudaStream_t cuStream);
};
```

#### 2.1.2 PHY PBCH聚合類（下行鏈路集成）

```cpp
class PhyPbchAggr : public PhyChannel {
public:
    // 初始化與銷毀
    PhyPbchAggr(phydriver_handle _pdh, GpuDevice* _gDev,
                cudaStream_t _s_channel, MpsCtx* _mpsCtx);
    ~PhyPbchAggr();
    
    // 流程控制
    int createPhyObj();
    int setup(const std::vector<DLOutputBuffer*>& aggr_dlbuf,
              const std::vector<Cell*>& aggr_cell_list);
    int run();
    int callback();
    
    // 參數檢索
    slot_command_api::pbch_group_params* getDynParams();

private:
    cuphySsbTxHndl_t handle;
    std::vector<uint16_t> cell_id_list;
    
    cuphySsbStatPrms_t ssbStatParams;
    cuphySsbDynPrms_t ssbDynPrms;
    cuphySsbDataIn_t DataIn;
    cuphySsbDataOut_t DataOut;
    
    static constexpr int MAX_MIB_BITS = 24;  // 24位MIB有效載荷
    static constexpr int MAX_MIB_BYTES = 3;
};
```

### 2.2 參數結構體

#### 2.2.1 靜態參數（cuphySsbStatPrms_t）

```cpp
struct cuphySsbStatPrms_t {
    // 預設參數集
    uint16_t nMaxSSBs;              // 最大SSB數量
    uint16_t nMaxCells;             // 最大小區數
    uint16_t nMaxSlotsPerFrame;     // 每幀最大槽數
    uint16_t nSymbolsPerSlot;       // 每槽符號數
    uint16_t nSubcarriersPerSlot;   // 每槽子載波數
    
    // 處理模式選擇
    uint64_t procModeBmsk;          // 處理模式位掩碼
    
    // 工作區配置
    cuphy::kernelDescrs workspace;  // 多組件工作區
};
```

#### 2.2.2 動態參數（cuphySsbDynPrms_t）

```cpp
struct cuphySsbDynPrms_t {
    // 運行時參數
    int nCells;                     // 活躍小區數
    int nSSBlocks;                  // 活躍SSB數
    
    // 數據指針
    cuphySsbDataIn_t* pDataIn;      // MIB輸入指針
    cuphySsbDataOut_t* pDataOut;    // 輸出緩衝區指針
    
    // Per-cell參數
    cuphyPerCellSsbDynPrms_t* pPerCellParams;
    
    // Per-SSB參數
    cuphyPerSsBlockDynPrms_t* pPerSsBlockParams;
    
    // CUDA流
    cudaStream_t cuStream;
};
```

#### 2.2.3 Per-SSB參數（cuphyPerSsBlockDynPrms_t）

```cpp
struct cuphyPerSsBlockDynPrms_t {
    uint16_t blockIndex;            // SSB索引 (0 - L_max)
    uint8_t  t0;                    // OFDM符號索引
    uint16_t f0;                    // 子載波索引
    float    beta_pss;              // PSS縮放因子
    float    beta_sss;              // SSS縮放因子
    uint16_t NID;                   // 物理小區ID (0-1007)
    uint16_t nHF;                   // 半幀索引 (0或1)
    uint16_t L_max;                 // PBCH週期內最大SSB數 (4/8/64)
    uint8_t  enablePrcdBf;          // 預編碼波束成型使能
};
```

#### 2.2.4 Per-Cell參數（cuphyPerCellSsbDynPrms_t）

```cpp
struct cuphyPerCellSsbDynPrms_t {
    uint16_t cellIndex;             // 小區索引
    uint16_t nF;                    // FFT子載波數（768/512等）
    uint16_t slotBufferIdx;         // 時頻域緩衝區索引
    uint8_t SFN;                    // 系統幀號
    uint16_t k_SSB;                 // SSB子載波偏移
    uint8_t ss_pbch_multiple_carriers;
    uint8_t multiple_cells_pbch;
};
```

## 3. 算法詳解

### 3.1 PSS/SSS生成與檢測

#### 3.1.1 Primary Synchronization Signal（PSS）

```matlab
% PSS生成算法（MATLAB參考）
function d_pss = gen_pss(N_id2)
    % N_id2 = NID % 3 (0, 1, 2)
    % PSS基序列：127位M序列
    
    load('pss_x_seq.mat');  % 預先計算的M序列表
    
    % PSS結構：127個子載波
    % 位置：第0和第2 OFDM符號，子載波56-182
    d_pss = x_pss(:, N_id2 + 1);  % 128 x 1複數向量
end

% PSS相關性檢測
pss_rx = received_signal(pss_indices);  % 127個接收樣本
for N_id2 = 0:2
    pss_tx = gen_pss(N_id2);
    pss_corr(N_id2 + 1) = abs(sum(pss_rx .* conj(pss_tx)));
end
[max_corr, N_id2_est] = max(pss_corr);
pss_metric = max_corr / sqrt(sum(abs(pss_rx).^2));  % 歸一化相關性
```

#### 3.1.2 Secondary Synchronization Signal（SSS）

```matlab
% SSS生成算法
function d_sss = gen_sss(N_id1, N_id2)
    % N_id1 = (NID - N_id2) / 3 (0-335)
    % N_id2 = NID % 3 (0, 1, 2)
    
    load('sss_x_seq.mat');  % 預先計算的Gold序列
    
    % SSS參數計算
    m0 = 15 * floor(N_id1/112) + 5 * N_id2;
    m1 = mod(N_id1, 112);
    
    % 索引排列
    idx0 = mod((0:126) + m0, 127);
    idx1 = mod((0:126) + m1, 127);
    
    % Gold序列生成
    d_sss = x0(idx0 + 1) .* x1(idx1 + 1);  % 127 x 1複數向量
end

% SSS相關性檢測（N_id1估計）
sss_rx = received_signal(sss_indices);  % 127個接收樣本
for N_id1 = 0:335
    sss_tx = gen_sss(N_id1, N_id2_est);
    sss_corr(N_id1 + 1) = abs(sum(sss_rx .* conj(sss_tx)));
end
[max_corr, N_id1_est] = max(sss_corr);
NID = N_id1_est * 3 + N_id2_est;  % 物理小區ID
```

#### 3.1.3 Cell ID識別

```cpp
// CUDA核心實現
__global__ void cellIdDetectionKernel(
    __half2* rx_signal,           // 接收信號
    float* pss_corr_out,          // PSS相關性輸出
    float* sss_corr_out,          // SSS相關性輸出
    uint16_t* cell_id_out)        // Cell ID輸出
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    // PSS檢測（3個假設）
    if (tid < 3) {
        float corr = 0.0f;
        for (int i = 56; i < 183; i += WARP_SIZE) {
            if (i + tid < 183) {
                __half2 pss_ref = PSS_LUT[tid * 127 + (i - 56)];
                __half2 rx = rx_signal[bid * 768 + i];
                corr += __half2float(pss_ref.x) * __half2float(rx.x) +
                        __half2float(pss_ref.y) * __half2float(rx.y);
            }
        }
        pss_corr_out[bid * 3 + tid] = corr;
    }
    
    // 選擇最大PSS
    __syncthreads();
    if (tid == 0) {
        float max_pss = max({pss_corr_out[bid * 3], 
                            pss_corr_out[bid * 3 + 1],
                            pss_corr_out[bid * 3 + 2]});
    }
    
    // SSS檢測（336個假設）
    if (tid < 336) {
        float corr = 0.0f;
        for (int i = 56; i < 183; i++) {
            __half2 sss_ref = SSS_LUT[tid * 127 + (i - 56)];
            __half2 rx = rx_signal[bid * 768 + i + 240 * 2];
            corr += __half2float(sss_ref.x) * __half2float(rx.x) +
                    __half2float(sss_ref.y) * __half2float(rx.y);
        }
        sss_corr_out[bid * 336 + tid] = corr;
    }
}
```

### 3.2 PBCH編碼與調制

#### 3.2.1 MIB有效載荷結構

```cpp
// MIB字段映射（24位）
struct MIBPayload {
    uint8_t sfn_msb : 8;           // [7:0] SFN MSB (高8位)
    uint8_t sfn_lsb : 2;           // [9:8] SFN LSB (低2位)
    uint8_t dmrs_type_a_pos : 1;   // [10] DMRS位置
    uint8_t pdcch_config_sib : 8;  // [18:11] PDCCH配置
    uint8_t cell_barred : 1;       // [19] Cell禁用
    uint8_t intra_freq_resel : 1;  // [20] 頻內重選
    uint8_t spare : 3;             // [23:21] 備用
};

// MIB編碼流程
int pbchGenPayload_per_SSB(uint32_t* x_mib,
                           uint32_t* x_payload,
                           uint8_t SSB_block_index,
                           const cuphyPerCellSsbDynPrms_t* cell_params)
{
    // 1. MIB提取
    uint32_t a_bar = x_mib[0] << (32 - CUPHY_SSB_N_MIB_BITS);
    
    // 2. SSB塊索引編碼
    uint8_t a0_3 = (SSB_block_index & 0xF) >> 2;  // 高2位
    uint8_t a5_7 = (SSB_block_index & 0x3);       // 低2位
    
    // 3. 有效載荷組合
    a_bar |= (a0_3 << 4);
    a_bar |= ((cell_params->nHF & 0x1) << 3);
    a_bar |= (a5_7 & 0x7);
    
    // 4. Payload生成（32位用滿）
    uint32_t v = (a0_3 & 6) >> 1;
    uint32_t M = (1 << (v + 1));  // 映射次數
    uint32_t len = v * M + 32;    // 有效載荷長度
    
    *x_payload = a_bar;
    return 0;
}
```

#### 3.2.2 Polar編碼與速率匹配

```cpp
// Polar編碼參數
struct PolarEncParams {
    uint16_t K = 32;        // 信息位數（固定）
    uint16_t E = 864;       // 編碼輸出位數（固定）
    uint16_t N = 512;       // Polar碼長度
    uint8_t nCrcBits = 24;  // CRC24位數
};

// 核心流程：CRC → Polar編碼 → 速率匹配
cuphyStatus_t cuphyRunPolarEncRateMatchSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRmSSBCfg,
    uint8_t const* pInfoBits,    // 32位MIB+CRC
    uint8_t* pCodedBits,         // 編碼比特輸出
    uint8_t* pTxBits,            // 速率匹配輸出（864位）
    uint16_t nSSBs,              // SSB數
    cudaStream_t strm)
{
    // CRC計算（24位）
    uint32_t crc = computeCRC_24C(pInfoBits, 24, G_CRC_24_C);
    
    // Polar編碼（32 → 512位）
    polarEncode(pInfoBits, pCodedBits, crc);
    
    // 速率匹配（512 → 864位）
    rateMatch(pCodedBits, pTxBits, 512, 864, SSB_RM_TABLE);
    
    return CUPHY_STATUS_SUCCESS;
}
```

### 3.3 PBCH QPSK調制與映射

#### 3.3.1 Scrambling序列

```cpp
// PBCH加擾序列生成
void pbch_scrambling_sequence(uint8_t* x_scrambled,
                              const uint8_t* x_bits,
                              uint8_t n_hf,
                              uint16_t N_id,
                              uint8_t L_max,
                              uint8_t block_idx)
{
    // 加擾序列初始化值
    uint32_t c_init = N_id * 65536 +          // 小區ID
                     (n_hf & 0x1) * 32768 +  // 半幀
                     ((block_idx >> 1) & 0x1) * 16384;
    
    // Gold序列生成
    uint32_t x1_reg = 0x54D;     // M序列1初值
    uint32_t x2_reg = c_init;    // M序列2初值
    
    for (int i = 0; i < 864; i++) {
        uint32_t bit_x1 = lsb_m_seq(x1_reg, 31);
        uint32_t bit_x2 = lsb_m_seq(x2_reg, 31);
        x_scrambled[i] = (x_bits[i] + bit_x1 + bit_x2) & 0x1;
    }
}
```

#### 3.3.2 PBCH時頻映射CUDA核心

```cuda
// PBCH調制與映射核心
template <typename TComplex>
__global__ void ssbModTfSigKernel(
    TComplex** d_tfSignal,
    const uint8_t* d_x_tx,
    const cuphyPerSsBlockDynPrms_t* d_ssb_params,
    const cuphyPerCellSsbDynPrms_t* d_per_cell_params,
    const cuphyPmWOneLayer_t* d_pmw_params)
{
    int ssb_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const uint8_t t0 = d_ssb_params[ssb_idx].t0;
    const uint16_t f0 = d_ssb_params[ssb_idx].f0;
    const float beta_pss = d_ssb_params[ssb_idx].beta_pss;
    const float beta_sss = d_ssb_params[ssb_idx].beta_sss;
    const uint16_t NID = d_ssb_params[ssb_idx].NID;
    const uint16_t nF = d_per_cell_params[cell_index].nF;
    
    // PBCH映射（第1、2、4符號）
    if (tid < CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS / 2) {
        // QPSK調制：2個加擾比特 → 1個複數符號
        uint8_t bit_i = d_x_tx[tid * 2];
        uint8_t bit_q = d_x_tx[tid * 2 + 1];
        
        __half2 qam = {(bit_i == 0) ? 0.707f : -0.707f,
                       (bit_q == 0) ? 0.707f : -0.707f};
        qam = __hmul2(qam, {beta_sss, beta_sss});
        
        // 資源映射
        int freq_idx = 56 + (tid % 120);
        int sym_idx;
        if (tid < 120) sym_idx = 1;        // 第2符號
        else if (tid < 240) sym_idx = 2;   // 第3符號
        else sym_idx = 3;                  // 第4符號
        
        tf_signal[ssb_idx][t0 * nF + f0 + freq_idx + sym_idx * nF] = qam;
    }
    
    // PSS/SSS映射（第0、2符號）
    if (tid < CUPHY_SSB_N_SS_SEQ_BITS) {
        uint8_t pss_bit = SSB_PSS_X[(tid + 43 * (NID % 3)) % 127];
        float pss_sym = (pss_bit == 0) ? 1.0f : -1.0f;
        pss_sym *= beta_pss;
        
        tf_signal[ssb_idx][t0 * nF + f0 + 56 + tid].x = pss_sym;
        tf_signal[ssb_idx][(t0 + 2) * nF + f0 + 56 + tid].x = pss_sym;
    }
}
```

### 3.4 波束成型與預編碼

#### 3.4.1 預編碼矩陣應用

```cpp
// 預編碼矩陣結構
struct cuphyPmWOneLayer_t {
    __half2 matrix[4][4];      // 最多4x4天線端口配置
    uint8_t numPorts;          // 有效端口數
    uint8_t precodingType;     // 預編碼類型
};

// 預編碼應用
if (enablePrcdBf) {
    for (int port_idx = 0; port_idx < nPorts; port_idx++) {
        // 預編碼權重應用
        __half2 precoded_sym = __hcmadd(qam_symbol,
                                        pmw_params.matrix[tid][port_idx],
                                        zero_value);
        
        // 輸出到對應天線端口
        tf_signal_per_port[port_idx][resource_elem] = precoded_sym;
    }
} else {
    // 未預編碼直接映射
    tf_signal[0][resource_elem] = qam_symbol;
}
```

## 4. C/C++ API參考

### 4.1 主要API函數

#### 4.1.1 SSB初始化與銷毀

```cpp
// 創建SSB TX對象
cuphyStatus_t CUPHYWINAPI cuphyCreateSsbTx(
    const cuphySsbStatPrms_t* pStatPrms,
    cuphySsbTxHndl_t* pSsbTxHndl,
    cudaStream_t cuStream = cudaStreamDefault);

// 銷毀SSB TX對象
cuphyStatus_t CUPHYWINAPI cuphyDestroySsbTx(
    cuphySsbTxHndl_t ssbTxHndl);

// 設置SSB參數
cuphyStatus_t CUPHYWINAPI cuphySetupSsbTx(
    cuphySsbTxHndl_t ssbTxHndl,
    cuphySsbDynPrms_t* pDynPrms,
    cudaStream_t cuStream = cudaStreamDefault);
```

#### 4.1.2 SSB執行

```cpp
// 執行SSB處理
cuphyStatus_t CUPHYWINAPI cuphyRunSsbTx(
    cuphySsbTxHndl_t ssbTxHndl,
    uint64_t procModeBmsk);  // 處理模式選擇

// 執行SSB對應核心（Mapper）
cuphyStatus_t CUPHYWINAPI cuphyRunSsbMapper(
    const uint8_t* d_x_tx,
    __half2** d_tfSignal,
    const cuphyPerSsBlockDynPrms_t* d_ssb_params,
    const cuphyPerCellSsbDynPrms_t* d_per_cell_params,
    const cuphyPmWOneLayer_t* d_pmw_params,
    const cuphySsbMapperLaunchCfg_t* pSsbMapperCfg,
    cudaStream_t stream);
```

#### 4.1.3 Polar編碼與速率匹配

```cpp
// 執行多SSB的Polar編碼與速率匹配
cuphyStatus_t CUPHYWINAPI cuphyRunPolarEncRateMatchSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRmSSBCfg,
    const uint8_t* pInfoBits,    // MIB + CRC（每SSB32+24位）
    uint8_t* pCodedBits,         // 編碼輸出（每SSB512位）
    uint8_t* pTxBits,            // 速率匹配輸出（每SSB864位）
    uint16_t nSSBs,              // SSB數
    cudaStream_t strm);

// 核心選擇（自動根據SSB數選擇最優配置）
cuphyStatus_t CUPHYWINAPI cuphySSBsKernelSelect(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
    cuphySsbMapperLaunchCfg_t* pSsbMapperLaunchCfg,
    uint16_t nSSBs);
```

### 4.2 使用流程

```cpp
// 1. 初始化
cuphySsbStatPrms_t statPrms = {
    .nMaxSSBs = 8,
    .nMaxCells = 4,
    .nMaxSlotsPerFrame = 10,
    .nSymbolsPerSlot = 14,
    .nSubcarriersPerSlot = 768
};

cuphySsbTxHndl_t ssbHandle;
cuphyCreateSsbTx(&statPrms, &ssbHandle, stream);

// 2. 準備動態參數
cuphyPerSsBlockDynPrms_t ssb_params[8];
cuphyPerCellSsbDynPrms_t cell_params[4];
uint32_t mib_payload[8];  // 每SSB MIB

// ... 填充參數 ...

cuphySsbDynPrms_t dynPrms = {
    .nCells = 4,
    .nSSBlocks = 8,
    .pPerSsBlockParams = ssb_params,
    .pPerCellParams = cell_params,
    .pDataIn = &data_in,
    .cuStream = stream
};

// 3. 設置
cuphySetupSsbTx(ssbHandle, &dynPrms, stream);

// 4. 執行
cuphyRunSsbTx(ssbHandle, CUPHY_PROC_MODE_GPU_ONLY);

// 5. 清理
cuphyDestroySsbTx(ssbHandle);
```

## 5. Python綁定與使用

### 5.1 Python API

```python
from pycuphy import SsbTx

# 初始化參數
ssb_static_params = {
    'nMaxSSBs': 8,
    'nMaxCells': 4,
    'nMaxSlotsPerFrame': 10,
    'nSymbolsPerSlot': 14
}

# 創建SSB TX對象
ssb_tx = SsbTx(cuda_stream_handle)

# 準備動態參數
ssb_params = [
    {
        'blockIndex': i,
        't0': 2,
        'f0': 0,
        'beta_pss': 1.0,
        'beta_sss': 1.0,
        'NID': 100,
        'nHF': 0,
        'L_max': 8,
        'enablePrcdBf': False
    } for i in range(8)
]

# 設置並執行
ssb_tx.setup(ssb_params)
output = ssb_tx.run()

# output: 時頻域SSB信號 (複數)
```

## 6. MATLAB參考實現

### 6.1 SSB生成

```matlab
function ssb_out = genSsb(mib_payload, ssb_params, carrier, Xtf)
    % 生成完整SSB塊（PSS/SSS/PBCH）
    
    Lmax = ssb_params.Lmax;
    blockIdx = ssb_params.blockIndex;
    beta_pss = ssb_params.beta_pss;
    beta_sss = ssb_params.beta_sss;
    NID = ssb_params.NID;
    
    % PSS生成與映射
    pss_sym = gen_pss(NID % 3);
    Xtf(57:183, 1) = beta_pss * pss_sym;
    
    % SSS生成與映射
    sss_sym = gen_sss((NID - NID%3)/3, NID % 3);
    Xtf(57:183, 3) = beta_sss * sss_sym;
    
    % PBCH編碼與調制
    pbch_bits = pbch_encode(mib_payload, NID, Lmax, blockIdx);
    pbch_qam = qpsk_modulate(pbch_bits);
    
    % PBCH映射
    pbch_re_idx = nrPBCHIndices(NID);
    pbch_idx_local = pbch_re_idx + [56, 240*2];  % 第2、4符號
    Xtf(pbch_idx_local) = beta_sss * pbch_qam;
    
    % DMRS生成與映射
    dmrs = build_pbch_dmrs(Lmax, blockIdx, 0, NID);
    dmrs_re_idx = nrPBCHDMRSIndices(NID);
    dmrs_idx_local = dmrs_re_idx + [56, 240*2];
    Xtf(dmrs_idx_local) = beta_sss * dmrs;
    
    ssb_out = Xtf;
end

function [pbch_bits] = pbch_encode(mib, N_id, L_max, block_idx)
    % MIB編碼流程：CRC → Polar編碼 → 速率匹配 → 加擾
    
    % CRC計算
    crc_bits = add_CRC_LUT(mib, '24C');  % 32位MIB+24位CRC = 56位
    
    % Polar編碼（32信息位 → 512編碼位）
    [encoded, N] = polar_encode(crc_bits, 32, 512);
    
    % 速率匹配（512 → 864位）
    pbch_rm = polar_rate_match(encoded, 512, 32, 864);
    
    % 加擾（使用N_id、半幀、塊索引）
    pbch_bits = pbch_scrambling(pbch_rm, 864, N_id, L_max, block_idx);
end
```

### 6.2 SSB檢測

```matlab
function [cell_id, pbch_payload] = detSsb(received_ssb, carrier)
    % SSB檢測：PSS → SSS → PBCH解調
    
    nAnt = size(received_ssb, 3);
    Xtf_ss = received_ssb(57:183, 1:4, :);  % 提取240x4 SSB資源
    
    % 1. PSS檢測 (3個假設)
    pss_corr = [];
    for N_id2 = 0:2
        pss_ref = gen_pss(N_id2);
        pss_rx = Xtf_ss(1:127, 1, 1);
        pss_corr(N_id2+1) = abs(sum(pss_rx .* conj(pss_ref)));
    end
    [~, N_id2_est] = max(pss_corr);
    N_id2_est = N_id2_est - 1;
    
    % 2. SSS檢測 (336個假設)
    sss_corr = [];
    for N_id1 = 0:335
        sss_ref = gen_sss(N_id1, N_id2_est);
        sss_rx = Xtf_ss(1:127, 3, 1);
        sss_corr(N_id1+1) = abs(sum(sss_rx .* conj(sss_ref)));
    end
    [~, N_id1_est] = max(sss_corr);
    N_id1_est = N_id1_est - 1;
    
    cell_id = N_id1_est * 3 + N_id2_est;
    
    % 3. PBCH解調與CRC
    pbch_rx_raw = Xtf_ss(:, [2, 3, 4], 1);
    pbch_bits_eq = pbch_demod(pbch_rx_raw, cell_id);
    
    % 加擾移除
    pbch_bits = pbch_descrambling(pbch_bits_eq, cell_id);
    
    % 速率匹配逆變換
    pbch_decoded = polar_rate_unmatch(pbch_bits, 864, 512);
    
    % Polar解碼
    pbch_payload = polar_decode(pbch_decoded, 32);
    
    % CRC驗證
    [mib, crc_ok] = verify_CRC_24C(pbch_payload);
    
    if ~crc_ok
        error('PBCH CRC驗證失敗');
    end
end
```

## 7. 性能特性

### 7.1 計算復雜度

| 操作 | 輸入大小 | 輸出大小 | 計算複雜度 |
|-----|---------|---------|-----------|
| PSS相關 | 127×N_ant | 3 | $O(127 \times N_{ant})$ |
| SSS相關 | 127×N_ant×336 | 336 | $O(127 \times 336 \times N_{ant})$ |
| Polar編碼 | 32位×N_SSB | 512位×N_SSB | $O(512 \log(512) \times N_{SSB})$ |
| 速率匹配 | 512位×N_SSB | 864位×N_SSB | $O(864 \times N_{SSB})$ |
| QPSK調制 | 864位×N_SSB | 432符號×N_SSB | $O(432 \times N_{SSB})$ |

### 7.2 記憶體需求

```
靜態工作區：
  - PSS/SSS參考表：~64 KB (預先計算)
  - Polar編碼LUT：~256 KB
  - 加擾序列LUT：~8 KB

動態工作區（per-slot）：
  - 編碼緩衝區：512 × N_SSB 位
  - 速率匹配緩衝區：864 × N_SSB 位
  - 輸出時頻域：768 × 14 × N_cells × 8 字節 (複數FP32)
  
總計: ~50 MB (8個SSB, 4小區)
```

### 7.3 延遲特性

```
典型延遲 (A100 GPU, 單小區):
  - PSS相關: ~10-15 µs
  - SSS相關: ~50-80 µs
  - Polar編碼: ~20-30 µs
  - SSB映射: ~30-50 µs
  ────────────────────
  總時間: ~110-175 µs
```

## 8. 故障排除

### 8.1 常見問題

**問題：PBCH CRC驗證失敗**
- 檢查MIB有效載荷格式（24位有效）
- 驗證加擾序列初始化（N_id、nHF、blockIdx）
- 確認速率匹配表版本3GPP TS 38.212相符

**問題：SSB信號功率異常**
- 驗證beta_pss與beta_sss值 (通常0.5-1.0)
- 檢查輸出緩衝區對齊 (4字節邊界)
- 確認CUDA圖同步完成

**問題：Cell ID檢測失敗**
- 增加相關性計算精度（使用FP32而非FP16）
- 檢查多天線合併邏輯
- 驗證信噪比≥0 dB

## 9. 優化建議

### 9.1 GPU加速技術

1. **協作組優化**
   - PSS相關使用WARP_SIZE=32
   - SSS相關使用分塊相關策略

2. **共享記憶體優化**
   - PSS/SSS參考表駐留共享記憶體
   - 銀行衝突避免 (padding align)

3. **非結構化記憶體訪問**
   - PBCH映射使用合併寫入
   - 預編碼權重預取

### 9.2 多小區處理

```cpp
// 並行處理4個小區
dim3 grid(4, 8);    // 4小區 × 8 SSB
dim3 block(32);     // 32線程
ssbModTfSigKernel<__half2><<<grid, block, shared_mem>>>(
    d_tfSignal, d_x_tx, d_ssb_params, d_per_cell_params, d_pmw_params);
```

## 10. 參考文獻

- 3GPP TS 38.211 V16.5.0 (2021-03) - NR物理層框架
- 3GPP TS 38.212 V16.5.0 (2021-03) - NR多路復用與通道編碼
- NVIDIA cuPHY API文檔
- MATLAB Communications Toolbox 5G標準實現
