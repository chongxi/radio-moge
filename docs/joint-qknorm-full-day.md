# 4×2048 joint adapter + QK norm：seed43 全量训练

状态：已于2026-10-28 23:36 UTC启动单次预检→全量训练流程（wrapper16168）。
执行版本9c9fd29278740a85b7908fb4b574acdd24a691bc；3次可丢弃GPU预检已通过；23:45 UTC自动启动正式训练进程16388，当前仍在数据初始化。
用户授权新架构训练，并要求记录与训练好的旧版 naive 的差异。
实际版本、启动、W&B与停止时间以共享实验目录的 plan/launch/status/provenance 为准。

## 新旧架构

旧版已有多头 attention、四层 RADIO 特征融合及 summary 的空间读取。
旧版参照是约两天训练后选定的156000步检查点，53,400,580参数。
本次采用筛选中的 attention_qknorm_s43，221,740,868参数，约为旧版4.15倍。

| 项目 | 训练好的旧版 | 新版 |
|---|---|---|
| 主干与目标 | 冻结 RADIO、MoGe encoder/decoder | 相同冻结模型与权重版本 |
| RADIO输入 | 四个spatial tap：7/15/23/31（0-based）和summary | 相同 |
| tap融合 | 独立投影，四个可学习全局softmax权重求和 | 保留，不是新增逐图动态融合 |
| 主Transformer | 2层×1280，16头，每头80维，只处理spatial | 4层×2048，32头，每头64维 |
| summary交互 | 独立width512、8-head cross-attention读取spatial，含summary skip | spatial、一个summary token、4个register共同self-attention，四层中双向交互 |
| 位置编码 | spatial上加固定2D sine/cosine absolute编码 | Q/K使用2D RoPE，summary/register旋转角为0 |
| QK norm | pre-LN，投影后没有per-head QK norm | 保留pre-LN，Q/K投影后各做per-head LayerNorm，再RoPE |
| attention输出门控 | 无额外gate | 输入经Linear→sigmoid，得到每token、每head的输出gate |
| register | 无 | 4个可学习全局辅助token，参与交互，不作为最终输出 |
| FFN | Linear→GELU→Linear | SwiGLU门控FFN |
| LayerScale | 可学习，初值1e-5 | 可学习，初值0.01 |
| 输出接口 | 1024维spatial和1024维summary，交给冻结MoGe decoder | 相同接口，分别由spatial/summary token经LN+Linear输出 |

每个新block：pre-LN→QKV→QK LayerNorm→2D RoPE→SDPA→sigmoid head gate→投影→LayerScale残差，
再pre-LN→SwiGLU→LayerScale残差。QK norm不处理V，也不强制最终输出为单位范数。
没有启用额外summary-only readout、局部卷积、flow matching或LoRA。

## RADIO异常通道清理：spatial沿用，summary按用户要求新增

函数：`zero_dominant_channels`。对每张图、每个spatial tap分别计算：

`a[c] = mean_{h,w} abs(feature[c,h,w])`

若 `a[c] > 8 * median_c(a[c])`，将该通道所有空间位置置零。
这是动态、逐图、逐tap的channel规则，不是固定channel黑名单，也不是按token范数删点。
只置零，不减少维度；不原地修改共享RADIO特征。四个tap在各自投影之前执行。
旧全量训练与架构筛选只清理spatial，summary/class未清理。
用户随后要求所有送入network的RADIO特征都先清理：本次新增summary清理。
RADIO summary是两个原始1280维token串接；分别在每个token内部，按绝对坐标幅值
超过该token通道中位数8倍置零，再串接送入summary_input。不能跨两个token混算中位数。
MoGe教师目标不做清理；原始RADIO缓存不修改，register为adapter参数，不属于RADIO输入。
所有实际消费的RADIO输入都经过清理：四个tap到各自投影，summary到summary投影。
最后一层spatial字段未单独消费，不存在额外未清理的旁路。训练/评估/恢复均由同一adapter forward执行。
高幅值不自动等于无用噪声；summary规则是本次新增假设，预检记录掩码比例与能量变化。
之前短训结果证明的是未清理summary的版本，不能宣称新增规则已经带来收益。
阈值8并未在本轮重新搜索；未证明该阈值最优。当前日志不声称有固定被删除channel编号。

## 损失

| 权重 | 旧版全量训练 | 本次 |
|---|---:|---:|
| spatial teacher-RMS-normalized MSE | 1 | 1 |
| summary teacher-RMS-normalized MSE | 1 | 1 |
| spatial 1−cosine | 0.1 | 1 |
| summary 1−cosine | 0 | 1 |
| 深度/法线/平面残差等几何训练损失 | 0 | 0 |

MSE约束幅值和方向；RMS只从训练数据拟合，128个microbatch×4=512张采样视图。输出特征本身保留幅值。
cosine、幅值比和深度质量分别报告，不能以cosine代替几何验证。

## 全量训练计划

- 使用已验证的796,478张七来源训练池、固定1,268张选择验证集（116张NYU GT）。
- 沿用旧版来源权重：ARKitScenes17.5%、Hypersim15%、NYU7.5%、COCO10%、
  TartanAir20%、Structured3D15%、IRS15%。带替换采样，池大小不等于实际已见独立图数。
- 从头初始化，seed43，fresh AdamW；不续训1,000步筛选checkpoint，不加载旧adapter。
  adapter初始化单独固定seed43，避免冻结主干加载消耗RNG改变初始参数；数据采样seed43。
- 从RGB在线提取特征，保留数据增强与480×640/640×480；不使用固定1,792图缓存训练。
- BF16，有效batch16（micro4×累积4），AdamW、weight decay0.01、clip1。
- 峰值LR3e-4沿用新架构筛选；长训warmup1,000步，120,000步cosine horizon/cap，floor为峰值5%。
  旧全量训练是LR3e-5、batch32；这些也属于实验差异，结果不能全部归因于结构。
- 首阶段累计trainer wall time最多86,400秒或120,000步，先到即停；包含加载、评估、保存。
  最后评估/保存可以越过截止时间，但不再更新参数。没有自动延长或第二阶段。
- 每20步训练日志，每1,000步保存latest，初始/每2,000步/最终评估1,268图。
  分别记录cosine全图均值、七来源宏平均和每来源值、幅值比、teacher depth与NYU GT。
- best仍按macro teacher-depth AbsRel选取，独立跟踪cosine，最后保留latest与selected。
- 新W&B run、新目录和版本；保留全部旧数据、模型、viewer，不关闭GPU。

## 已有证据与边界

同一224图复测面板，1,000步QK norm：seed42 cosine为0.917512/0.812422；
seed43为0.918232/0.812730。NYU AbsRel：0.088440→0.082710；平面RMS：0.025018→0.023658。
采用43，但小样本差异不证明它在长训一定更好；面板已经反复查看。

旧版156k在相同缓存输入/教师目标上只读复测：cosine0.942834/0.892400；
新短训QK两seed均值0.917872/0.812576，尚未超过训练好的旧版。
同为从零1,000步、seed42时，新架构比旧架构高0.024740/0.027747。
这是完整结构同一步数收益，不是QK单因素收益，也不证明实际秒数更快。

网页：https://buildhub-nano-test--p7869.pub.astera.work/
共享实验目录：`/shared/radio-moge/chongxi/data-expansion-attention-20261022/adapter-qknorm2048-full-s43-20261028/`。

## 启动验证

本地与H200远端44项CPU测试通过，ruff通过。真实GPU预检检查全部RADIO投影输入已清理、
原始特征与MoGe目标未修改、冻结主干无梯度、adapter梯度有限、参数更新非零。
预检结果与正式训练状态以共享目录preflight.json、status.json、monitor.json为准；
CPU测试或预检通过本身不代表正式训练已经完成更新。

预检结果：全部输入清理路径与原始特征/教师目标不变已由forward hooks验证；冻结主干无梯度。
12个采样视图的两个summary token分别平均置零2.17/2.33个通道；移除平方幅值比例均值34.38%/12.05%。
预检只检查数值和执行正确性，不证明清理有益；三次更新不计入正式训练。显存峰值9.71GiB。
