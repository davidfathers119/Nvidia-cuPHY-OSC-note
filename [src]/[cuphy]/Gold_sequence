# Gold_sequence

Gold sequence 是由兩條線性回饋移位暫存器 (LFSR) XOR 組合而成的偽隨機碼序列，用來讓無線訊號看起來像噪音，避免不同 UE 或不同符號互相干擾。

c(n) = x1(n) XOR x2(n)

其中GOLD_1_SEQ_LUT_CPU.h 所存的是 x1 序列（LFSR1），**不會改變**，因此這個檔案可以預先算好、放 LUT

而GOLD_2_32_P_LUT_CPU.h 存的是 x2 序列（LFSR2）對應不同 seed / UE / slot ，他是變數，它決定每個 UE、每個時機的 Gold code 都不同

總結:

**Gold sequence 是用兩條 LFSR 序列 XOR 運算而成，因此 NVIDIA 分別把 固定不變的 x1 序列存成一個 LUT，再把 可根據 UE/RNTI/slot 改變的 x2 base 序列拆成另一個 LUT。
這樣一來生成 Gold code 的時候，只需要從兩組表中取數並 XOR，就可以快速得到不同 UE/不同時間的擾碼序列。**
