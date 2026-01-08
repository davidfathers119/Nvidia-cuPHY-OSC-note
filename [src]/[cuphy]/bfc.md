# BFC

BFC = BlueField-Converged (或者 BlueField Controller)
指的是 搭載 BlueField DPU 的收發控制層
用來負責 前傳接口、時間同步、封包處理，在 Aerial L1 與 RU 之間當橋樑。
它跑在 NVIDIA BlueField SmartNIC/DPU 上，
負責處理 O-RAN/eCPRI 前傳、封包加速，以及時間同步。它與 cuPHY-CP 與 cuPHY 配合，形成完整的 L1 布建架構。
nvIPC 是 CPU↔CPU 的 IPC，而 BFC 則是 NIC/Timing↔L1 的收包發包入口。

