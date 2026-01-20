# NVIDIA cuPHY LDPC (Low-Density Parity-Check) API Layer

**File Location**: `cuPHY/src/cuphy/ldpc/`
**Main Files**: `ldpc_api.hpp`, `ldpc_api.cpp`
**Purpose**: High-level C++ API wrapper for LDPC encoding and decoding operations

---

## 概述 (Overview)

LDPC API 層提供高級別的 C++ 接口，用於 5G NR 信道編碼的 LDPC 編碼和解碼操作。它建立在底層 error_correction LDPC 實現之上，提供以下功能：

- **Encoding Support**: 模板化 LDPC 編碼函數，支持多種數據類型
- **Decoding Configuration**: 靈活的配置結構和類，用於 LDPC 解碼
- **Transport Block Management**: 描述符集合用於管理多個傳輸塊
- **Tensor Interfaces**: 支持步長 (strided) 張量的靈活數據佈局

---

## 檔案結構 (File Structure)

### ldpc_api.hpp (Header)
- LDPC 編碼和解碼 C++ 包裝類
- 配置結構定義
- 描述符管理類

### ldpc_api.cpp (Implementation)
- 包裝類的實現
- 描述符管理邏輯
- CUDA API 調用轉發

---

## 數據結構 (Data Structures)

### 1. `LDPC_decode_config`

C++ 包裝類，對應底層 `cuphyLDPCDecodeConfigDesc_t`

```cpp
class LDPC_decode_config final : public cuphyLDPCDecodeConfigDesc_t
{
public:
    explicit LDPC_decode_config(
        cuphyDataType_t llr_type_in = CUPHY_R_16F,        // LLR 數據類型
        int16_t         num_parity_nodes_in = 4,          // 奇偶校驗節點數
        int16_t         Z_in = 384,                        // 提升因子
        int16_t         max_iterations_in = 10,           // 最大迭代次數
        float           clamp_value_in = 32.0f,           // 鉗位值
        int16_t         Kb_in = 22,                        // 信息節點數
        float           norm_in = 0.8125f,                // 歸一化因子
        uint32_t        flags_in = 0,                      // 標誌位
        int16_t         BG_in = 1,                         // 基圖 (1 or 2)
        int16_t         algo_in = 0,                       // 算法選擇 (0=自動)
        void*           workspace_in = nullptr             // 工作空間
    );
    
    [[nodiscard]]
    float get_norm() const;                                // 獲取歸一化值
};
```

**參數說明**:
- `llr_type_in`: LLR 數據類型 (CUPHY_R_16F=FP16, CUPHY_R_32F=FP32)
- `num_parity_nodes_in`: 檢查節點數量 (4-46 for BG1, 4-42 for BG2)
- `Z_in`: 提升因子 (2-384)
- `max_iterations_in`: LDPC 迭代器最大迭代次數
- `Kb_in`: 信息變量節點數 (22 for BG1, 6-10 for BG2)
- `norm_in`: Min-Sum 歸一化因子 (通常 0.75-0.85)
- `BG_in`: LDPC 基圖選擇 (1 or 2)
- `algo_in`: 算法選擇 (0 = 自動選擇最合適的算法)

---

### 2. `LDPC_decode_desc`

管理單個傳輸塊的 LDPC 解碼描述符

```cpp
class LDPC_decode_desc final : public cuphyLDPCDecodeDesc_t
{
public:
    LDPC_decode_desc();
    explicit LDPC_decode_desc(const cuphyLDPCDecodeConfigDesc_t& config_in);
    
    // 不需要軟輸出的傳輸塊
    void add_tensor_as_tb(
        const tensor_desc& llrTensorDesc,
        void*              llrAddr,
        const tensor_desc& decodeTensorDesc,
        void*              decodeAddr
    );
    
    // 需要軟輸出的傳輸塊
    void add_tensor_as_tb(
        const tensor_desc& llrTensorDesc,
        void*              llrAddr,
        const tensor_desc& decodeTensorDesc,
        void*              decodeAddr,
        const tensor_desc& softOutputsTensorDesc,
        void*              softOutputsAddr
    );
    
    void reset();                                          // 重置傳輸塊計數
    [[nodiscard]] bool has_config(
        const int16_t BG_,
        const int Z_,
        const int parity_nodes
    ) const;
    [[nodiscard]] bool is_full() const;                    // 檢查是否已滿
};
```

**主要功能**:
- 管理 LLR 輸入、解碼輸出和軟輸出張量
- 支持步長 (strided) 張量用於靈活的內存佈局
- 最多支持 `CUPHY_LDPC_DECODE_DESC_MAX_TB` 個傳輸塊

**add_tensor_as_tb() 說明**:
- LLR 輸入: 對數似然比輸入，形狀 [num_codewords, stride_elements]
- 解碼輸出: 解碼位輸出，以 uint32_t 字為單位
- 軟輸出 (可選): 軟決策值輸出

---

### 3. `LDPC_decode_desc_set`

管理多個 LDPC 解碼描述符集合

```cpp
class LDPC_decode_desc_set final
{
public:
    LDPC_decode_desc_set();
    
    LDPC_decode_desc& operator[](size_t idx);             // 索引訪問
    [[nodiscard]] unsigned int count() const;             // 獲取計數
    
    // 查找或創建匹配配置的描述符
    [[nodiscard]]
    LDPC_decode_desc& find(int16_t BG, int Z, int num_parity);
    
    void resize(size_t maxSize);                          // 調整最大大小
    void reset();                                         // 重置所有描述符
};
```

**find() 邏輯**:
1. 遍歷現有描述符查找匹配配置
2. 如果找到未滿的描述符，返回該描述符
3. 否則，創建新描述符並返回
4. 如果達到最大大小，拋出異常

---

### 4. `LDPC_decode_tensor_params`

張量接口的 LDPC 解碼參數集合

```cpp
struct LDPC_decode_tensor_params final
{
    LDPC_decode_tensor_params(
        const cuphyLDPCDecodeConfigDesc_t& cfg,
        cuphyTensorDescriptor_t            dst_desc_,
        void*                              dst_addr_,
        cuphyTensorDescriptor_t            LLR_desc_,
        const void*                        LLR_addr_,
        cuphyTensorDescriptor_t            softOut_desc_ = nullptr,
        void*                              softOut_addr_ = nullptr
    );
    
    cuphyLDPCDecodeConfigDesc_t   config;                 // LDPC 配置
    cuphyTensorDescriptor_t       dst_desc;               // 解碼輸出描述符
    void*                         dst_addr;               // 解碼輸出地址
    cuphyTensorDescriptor_t       LLR_desc;               // LLR 輸入描述符
    const void*                   LLR_addr;               // LLR 輸入地址
    cuphyTensorDescriptor_t       softOutputs_desc;       // 軟輸出描述符 (可選)
    void*                         softOutputs_addr;       // 軟輸出地址
};
```

---

## LDPC_decoder 類 (Main Decoder Class)

### 構造函數

```cpp
class LDPC_decoder final
{
public:
    explicit LDPC_decoder(context& ctx, unsigned int flags = 0);
```

**參數**:
- `ctx`: cuPHY 上下文
- `flags`: 控制標誌 (未來擴展)

---

### 主要方法

#### 1. `get_workspace_size()`

計算解碼所需的工作空間大小

```cpp
[[nodiscard]]
size_t get_workspace_size(
    const cuphyLDPCDecodeConfigDesc_t& cfg,
    int                                numCodeWords
) const;
```

**用途**: 確定分配多少設備內存用於中間計算

**參數**:
- `cfg`: LDPC 配置
- `numCodeWords`: 同時解碼的代碼字數

**返回值**: 工作空間大小 (字節)

---

#### 2. `decode()` - 張量接口

使用張量描述符進行 LDPC 解碼

```cpp
void decode(
    const LDPC_decode_tensor_params& params,
    cudaStream_t                     strm = nullptr
) const;
```

**參數**:
- `params`: 包含所有張量參數的結構體
- `strm`: CUDA 流 (nullptr = 使用默認流)

**用途**: 通用的張量接口，支持任意步長和數據佈局

---

#### 3. `decode()` - 傳輸塊接口

使用描述符集合進行 LDPC 解碼

```cpp
void decode(
    const cuphyLDPCDecodeDesc_t& desc,
    cudaStream_t                 strm = nullptr
) const;
```

**參數**:
- `desc`: 包含多個傳輸塊的描述符
- `strm`: CUDA 流

**用途**: 針對優化的傳輸塊批處理解碼

---

#### 4. `set_normalization()`

設置 Min-Sum 解碼器的歸一化因子

```cpp
void set_normalization(cuphyLDPCDecodeConfigDesc_t& config) const;
```

**功能**:
- 根據基圖 (BG)、提升因子 (Z) 和其他參數自動設置歸一化因子
- 修改提供的配置結構

---

#### 5. `get_launch_config()`

獲取 LDPC 內核啟動配置

```cpp
void get_launch_config(cuphyLDPCDecodeLaunchConfig_t& cfg) const;
```

**返回**: 包含網格/塊維度和其他內核啟動信息的結構體

---

#### 6. `handle()`

獲取底層解碼器句柄

```cpp
[[nodiscard]] cuphyLDPCDecoder_t handle() const;
```

**用途**: 直接訪問底層 C API

---

## LDPC_encode() 模板函數

高級別的 LDPC 編碼函數

```cpp
template <class TDst, class TSrc>
void ldpc_encode(
    TDst&        dst,                    // 輸出張量
    TSrc&        src,                    // 輸入信息比特
    int          BG,                     // 基圖 (1 or 2)
    int          Z,                      // 提升因子
    bool         puncture = false,       // 穿孔標誌
    int          maxParityNodes = 0,     // 最大奇偶校驗節點
    int          rv = 0,                 // 冗餘版本
    cudaStream_t strm = nullptr          // CUDA 流
);
```

**功能**:
1. 分配設備和主機描述符
2. 分配工作空間
3. 設置編碼配置
4. 啟動 LDPC 編碼內核
5. 同步流以確保完成

**參數說明**:

| 參數 | 類型 | 說明 |
|-----|------|------|
| `dst` | 張量 | 編碼輸出 (編碼比特) |
| `src` | 張量 | 輸入信息比特 |
| `BG` | int | LDPC 基圖選擇 (1 or 2) |
| `Z` | int | 提升因子 (2-384) |
| `puncture` | bool | 是否穿孔輸出比特 |
| `maxParityNodes` | int | 最大奇偶校驗節點數 |
| `rv` | int | 冗餘版本 (率匹配) |
| `strm` | cudaStream_t | CUDA 流句柄 |

**內部流程**:

```cpp
// 1. 獲取描述符大小信息
cuphyLDPCEncodeGetDescrInfo(&desc_size, &alloc_size, max_UEs, &workspace_size);

// 2. 分配設備和主機內存
d_ldpc_desc = make_unique_device<uint8_t>(desc_size);
h_ldpc_desc = make_unique_pinned<uint8_t>(desc_size);
d_workspace = make_unique_device<uint8_t>(workspace_size);
h_workspace = make_unique_pinned<uint8_t>(workspace_size);

// 3. 設置編碼配置
cuphySetupLDPCEncode(
    &launchConfig,
    src.desc().handle(), src.addr(),
    dst.desc().handle(), dst.addr(),
    BG, Z,
    puncture, maxParityNodes, rv,
    ...,
    h_workspace.get(), d_workspace.get(),
    h_ldpc_desc.get(), d_ldpc_desc.get(),
    1,  // do async copy during setup
    strm
);

// 4. 啟動內核
cuLaunchKernel(
    kernelNodeParams.func,
    kernelNodeParams.gridDimX/Y/Z,
    kernelNodeParams.blockDimX/Y/Z,
    kernelNodeParams.sharedMemBytes,
    strm,
    kernelNodeParams.kernelParams,
    kernelNodeParams.extra
);

// 5. 同步流
cudaStreamSynchronize(strm);
```

**內核啟動參數**:
- 網格維度: (gridDimX, gridDimY, gridDimZ)
- 塊維度: (blockDimX, blockDimY, blockDimZ)
- 共享內存: sharedMemBytes 字節

---

## LDPC_encode() 流程圖

```
Input Information Bits
         |
         v
+------------------+
|  ldpc_encode()   |
|  C++ Template    |
+------------------+
         |
         +-----> Allocate Descriptors
         |
         +-----> Allocate Workspace
         |
         +-----> Setup LDPC Config
         |           - BG, Z, RV
         |           - Puncture pattern
         |
         +-----> Launch LDPC Kernel
         |           - Device execution
         |           - Parallel encoding
         |
         +-----> Synchronize Stream
         |
         v
    Encoded Bits
   (Output Codeword)
```

---

## LDPC_decoder 使用示例

### 基本配置和解碼

```cpp
// 1. 創建解碼器
cuphy::context ctx;
cuphy::LDPC_decoder decoder(ctx);

// 2. 創建配置
cuphy::LDPC_decode_config config(
    CUPHY_R_16F,           // FP16 LLR 類型
    12,                    // 12 奇偶校驗節點
    384,                   // 提升因子 384
    10,                    // 最大 10 次迭代
    32.0f,                 // 鉗位值
    22,                    // Kb = 22 (BG1)
    0.8125f,               // 歸一化因子
    0,                     // 無特殊標誌
    1                      // 基圖 1
);

// 3. 設置自動歸一化
decoder.set_normalization(config);

// 4. 準備張量
// llr_tensor: [num_codewords, num_llr_elements]
// decode_tensor: [num_codewords, num_decode_words]
// soft_tensor (可選): [num_codewords, num_llr_elements]

cuphy::LDPC_decode_tensor_params params(
    config,
    decode_tensor.desc().handle(), decode_out_addr,
    llr_tensor.desc().handle(), llr_in_addr,
    soft_tensor.desc().handle(), soft_out_addr
);

// 5. 執行解碼
decoder.decode(params, stream);
```

### 多傳輸塊解碼

```cpp
cuphy::LDPC_decode_desc_set desc_set;
desc_set.resize(10);  // 最多 10 個描述符

// 添加傳輸塊 1
auto& desc1 = desc_set.find(1, 384, 12);  // BG=1, Z=384, parity=12
desc1.add_tensor_as_tb(llr_desc1, llr_addr1, decode_desc1, decode_addr1);

// 添加傳輸塊 2
auto& desc2 = desc_set.find(1, 384, 12);
desc2.add_tensor_as_tb(llr_desc2, llr_addr2, decode_desc2, decode_addr2);

// 執行批量解碼
for(unsigned int i = 0; i < desc_set.count(); ++i) {
    decoder.decode(desc_set[i], stream);
}
```

---

## 3GPP 標準映射

LDPC API 層實現了 3GPP TS 38.212 標準的下列部分：

| 功能 | 標準 | 說明 |
|------|------|------|
| 基圖選擇 | TS 38.212 Section 5.3.2 | BG1 (1472 列) vs BG2 (1024 列) |
| 提升因子 | TS 38.212 Section 5.3.2 | Z ∈ {2,4,6,8,...,384} |
| 編碼 | TS 38.212 Section 5.3 | LDPC 編碼過程 |
| 解碼 | TS 38.212 Section 5.3.4 | 最小和、規範化最小和 |
| 率匹配 | TS 38.212 Section 5.4 | 冗餘版本 (RV) 支持 |

---

## 關鍵特性

1. **靈活的數據類型**: 支持 FP16 和 FP32 LLR 表示
2. **自動算法選擇**: 根據 GPU 架構和配置參數自動選擇最合適算法
3. **批處理支持**: 通過描述符集合進行多傳輸塊批處理
4. **步長張量**: 支持非連續內存佈局
5. **軟輸出選項**: 可選的軟決策值輸出
6. **工作空間管理**: 精確的內存需求計算

---

## 性能特性

- **吞吐量優化**: 多種算法變體針對不同 GPU 特性優化
- **内存效率**: 共享內存優化減少全局內存訪問
- **批處理**: 多傳輸塊的高效處理
- **流支持**: 與其他操作異步執行

---

## 錯誤處理

所有函數調用被異常包裝。失敗時拋出 `cuphy_fn_exception`:

```cpp
try {
    decoder.decode(params, stream);
} catch (const cuphy::cuphy_fn_exception& e) {
    std::cerr << "LDPC decode failed: " << e.what() << std::endl;
}
```

常見的失敗原因：
- 無效的配置 (Z、BG、奇偶校驗節點)
- 不足的工作空間
- CUDA 內存分配失敗
- 非法的流操作

---

## 依賴關係

```
ldpc_api.hpp/cpp
    |
    +---> ldpc.hpp/cpp/cuh (error_correction layer)
    |
    +---> cuphy.hpp/h (Core API)
    |
    +---> CUDA Runtime
    |
    +---> NVIDIA GPU Driver
```

---

## 總結

LDPC API 層提供了簡潔而強大的 C++ 接口，用於在 NVIDIA GPU 上進行高性能的 LDPC 編碼和解碼。通過提供靈活的配置選項、自動算法選擇和批處理支持，它可以有效地用於 5G NR 系統中的信道編碼操作。
