/* SunnyNet 稳定 C ABI.
 *
 * 设计约束:
 *   - 只经由本头文件暴露的符号属于稳定接口, 其签名在 ABI 版本内不改动。
 *     内部实现 (sunnynet.h / rwkv7.h / recur.h) 不在此承诺范围内。
 *   - 符号带 sn_abi_ 前缀, 与内部符号隔离, 避免重名与误用。
 *   - 所有句柄为不透明指针; 调用方不依赖任何结构体布局。
 *   - 新增能力走新函数 + 能力位, 不改旧函数签名。破坏性变更必须提升
 *     SN_ABI_VERSION_MAJOR。
 *
 * 版本语义: 主版本不同即不兼容; 主版本相同、次版本更高表示向后兼容的功能新增
 * (旧调用方仍可工作)。
 */
#ifndef SN_ABI_H
#define SN_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SN_ABI_VERSION_MAJOR 1
#define SN_ABI_VERSION_MINOR 0
#define SN_ABI_VERSION_PATCH 0

/* (major << 16) | (minor << 8) | patch */
#define SN_ABI_VERSION \
    ((SN_ABI_VERSION_MAJOR << 16) | (SN_ABI_VERSION_MINOR << 8) | \
     SN_ABI_VERSION_PATCH)

/* 能力位: 调用方按位查询, 未知位应忽略. */
#define SN_CAP_RECUR_SCAN   (1u << 0)  /* sn_abi_recur_scan */
#define SN_CAP_RECUR_STEP   (1u << 1)  /* sn_abi_recur_step */
#define SN_CAP_QUANT_I8     (1u << 2)  /* sn_abi_quant_i8 / dequant */
#define SN_CAP_RWKV7        (1u << 3)  /* sn_abi_rwkv7_load */
#define SN_CAP_THREADS      (1u << 4)  /* sn_abi_set_threads */

/* 错误码: 0 为成功, 负数为失败. */
#define SN_OK              0
#define SN_E_INVAL        -1   /* 参数非法 (空指针/负维度等) */
#define SN_E_NOMEM        -2   /* 分配失败 */
#define SN_E_IO           -3   /* 打开/读取失败 */
#define SN_E_FORMAT       -4   /* 魔数或版本不匹配 */
#define SN_E_UNSUPPORTED  -5   /* 当前构建不支持该能力 */

typedef int32_t sn_status;   /* 固定宽度, 避免 int 宽度差异 */

/* ---------------------------------------------------------------- 版本 */
/* 返回 SN_ABI_VERSION. */
int32_t sn_abi_version(void);

/* 返回能力位掩码. */
uint32_t sn_abi_caps(void);

/* 检查调用方编译期主版本是否与本库兼容. 返回 SN_OK 或 SN_E_UNSUPPORTED. */
sn_status sn_abi_check(int32_t caller_major);

/* ------------------------------------------------------------ 运行时信息 */
/* 写入库的版本字符串 (如 "1.0.0"), 不申请内存; buf 为 NULL 时返回所需长度. */
int32_t sn_abi_version_string(char* buf, int32_t buflen);

/* 写入支持的后端名 ("c"), 便于调用方做兼容判断. */
int32_t sn_abi_backend_string(char* buf, int32_t buflen);

/* ------------------------------------------------------------- 线程控制 */
sn_status sn_abi_set_threads(int32_t n);
int32_t sn_abi_get_threads(void);

/* --------------------------------------------------------- 一阶递推内核 */
/* 单步: a, b, s 均为 (B, N) 连续 fp32; s 原地更新.
 * gated 非 0 时 b 语义为 u, 且 b_new = (1 - a) * u. */
sn_status sn_abi_recur_step(const float* a, const float* b, float* s,
                            int32_t B, int32_t N, int32_t gated);

/* 整段扫描: a, b, out 为 (B, T, N); s0, send 为 (B, N). */
sn_status sn_abi_recur_scan(const float* a, const float* b, const float* s0,
                            float* out, float* send,
                            int32_t B, int32_t T, int32_t N, int32_t gated);

/* ------------------------------------------------------------ 量化原语 */
sn_status sn_abi_quant_i8(const float* x, int8_t* q, float* scale,
                          int32_t B, int32_t N, int32_t block);
sn_status sn_abi_dequant_i8(const int8_t* q, const float* scale, float* x,
                            int32_t B, int32_t N, int32_t block);

/* --------------------------------------------------------- RWKV-7 句柄 */
/* 不透明句柄. 生命周期由调用方管理. */
typedef struct SnAbiRwkv7 SnAbiRwkv7;

SnAbiRwkv7* sn_abi_rwkv7_load(const char* blob_path);

/* 返回 SN_OK 或错误码; logits 缓冲由调用方提供, 长度须 >= sn_abi_rwkv7_vocab */
sn_status sn_abi_rwkv7_step(SnAbiRwkv7* m, int32_t token, float* logits);
sn_status sn_abi_rwkv7_prefill(SnAbiRwkv7* m, const int32_t* tokens,
                               int32_t n, float* logits);
sn_status sn_abi_rwkv7_reset(SnAbiRwkv7* m);
void      sn_abi_rwkv7_free(SnAbiRwkv7* m);
int32_t   sn_abi_rwkv7_vocab(const SnAbiRwkv7* m);

#ifdef __cplusplus
}
#endif
#endif /* SN_ABI_H */