//! ggml.zig - 安全封装层（模块化入口）
//!
//! 提供 ggml C API 的类型安全 Zig 封装。
//! 所有分配类操作返回 `!*T` 错误联合，纯计算操作返回 `*T`。
//! 使用 `opaque {}` 类型包装不透明指针。
//!
//! 模块结构：
//! - c.zig:      原始 C API 导入和类型枚举
//! - context.zig: ggml_context 封装
//! - tensor.zig:  ggml_tensor 封装
//! - graph.zig:   ggml_cgraph 封装
//! - backend.zig: Backend 与 Gallocr 封装
//! - ops.zig:     计算图操作函数
//! - quantize.zig: 量化操作函数（quantize_chunk, quantize_init 等）
//! - utils.zig:   工具函数（版本、CPU 特性等）

const std = @import("std");

pub const GraphMeasureInfo = @import("graph.zig").GraphMeasureInfo;
pub const measureGraph = @import("graph.zig").measureGraph;
pub const measureGraphDetailed = @import("graph.zig").measureGraphDetailed;

// ============================================================================
// 重新导出所有子模块
// ============================================================================

pub const c = @import("c.zig").c;
pub const Type = @import("c.zig").Type;
pub const Prec = @import("c.zig").Prec;
pub const GgufValueType = @import("c.zig").GgufValueType;
pub const GgufValue = @import("c.zig").GgufValue;
pub const PoolOp = @import("c.zig").PoolOp;
pub const ScaleMode = @import("c.zig").ScaleMode;
pub const ScaleFlag = @import("c.zig").ScaleFlag;
pub const n_tasks_max = @import("c.zig").n_tasks_max;

pub const Context = @import("context.zig").Context;
pub const Tensor = @import("tensor.zig").Tensor;
pub const CGraph = @import("graph.zig").CGraph;

pub const Backend = @import("backend.zig").Backend;
pub const BackendBufferType = @import("backend.zig").BackendBufferType;
pub const Scheduler = @import("backend.zig").Scheduler;
pub const Gallocr = @import("backend.zig").Gallocr;
pub const backendCpuInit = @import("backend.zig").backendCpuInit;
pub const backendCpuBufferType = @import("backend.zig").backendCpuBufferType;
pub const backendInitByType = @import("backend.zig").backendInitByType;
pub const backendInitBest = @import("backend.zig").backendInitBest;
pub const backendGetDefaultBufferType = @import("backend.zig").backendGetDefaultBufferType;
pub const backendAllocCtxTensors = @import("backend.zig").backendAllocCtxTensors;
pub const backendAllocCtxTensorsFromBuft = @import("backend.zig").backendAllocCtxTensorsFromBuft;
pub const backendFree = @import("backend.zig").backendFree;
pub const backendCpuSetNThreads = @import("backend.zig").backendCpuSetNThreads;
pub const backendGraphCompute = @import("backend.zig").backendGraphCompute;
pub const backendTensorGet = @import("backend.zig").backendTensorGet;
pub const loadBackends = @import("backend.zig").loadBackends;
pub const setInput = @import("backend.zig").setInput;
pub const backendBuftIsHost = @import("backend.zig").backendBuftIsHost;
pub const backendTensorSet = @import("backend.zig").backendTensorSet;
pub const DeviceType = @import("backend.zig").DeviceType;
pub const detectBestBackend = @import("backend.zig").detectBestBackend;
pub const backendName = @import("backend.zig").backendName;
pub const backendIsGpu = @import("backend.zig").backendIsGpu;
pub const logAvailableBackends = @import("backend.zig").logAvailableBackends;

/// backend 子模块命名空间
pub const backend = @import("backend.zig");

pub const ThreadPool = @import("threadpool.zig").ThreadPool;

pub const gguf = @import("gguf.zig");

pub const mulMat = @import("ops.zig").mulMat;
pub const mulMatSetPrec = @import("ops.zig").mulMatSetPrec;
pub const flashAttnExtSetPrec = @import("ops.zig").flashAttnExtSetPrec;
pub const mul = @import("ops.zig").mul;
pub const add = @import("ops.zig").add;
pub const neg = @import("ops.zig").neg;
pub const exp = @import("ops.zig").exp;
pub const cpy = @import("ops.zig").cpy;
pub const cast = @import("ops.zig").cast;
pub const rmsNorm = @import("ops.zig").rmsNorm;
pub const l2Norm = @import("ops.zig").l2Norm;
pub const ropeExt = @import("ops.zig").ropeExt;
pub const ropeMulti = @import("ops.zig").ropeMulti;
pub const scale = @import("ops.zig").scale;
pub const softMax = @import("ops.zig").softMax;
pub const softMaxExt = @import("ops.zig").softMaxExt;
pub const flashAttnExt = @import("ops.zig").flashAttnExt;
pub const diagMaskInf = @import("ops.zig").diagMaskInf;
pub const silu = @import("ops.zig").silu;
pub const gelu = @import("ops.zig").gelu;
pub const geluErf = @import("ops.zig").geluErf;
pub const geluQuick = @import("ops.zig").geluQuick;
pub const sqr = @import("ops.zig").sqr;
pub const tanh = @import("ops.zig").tanh;
pub const relu = @import("ops.zig").relu;
pub const sigmoid = @import("ops.zig").sigmoid;
pub const softplus = @import("ops.zig").softplus;
pub const permute = @import("ops.zig").permute;
pub const cont = @import("ops.zig").cont;
pub const gatedDeltaNet = @import("ops.zig").gatedDeltaNet;
pub const clamp = @import("ops.zig").clamp;
pub const dequantizeRow = @import("ops.zig").dequantizeRow;
pub const dequantizeTensor = @import("ops.zig").dequantizeTensor;

pub const cont2d = @import("ops.zig").cont2d;
pub const cont4d = @import("ops.zig").cont4d;
pub const reshape2d = @import("ops.zig").reshape2d;
pub const reshape3d = @import("ops.zig").reshape3d;
pub const reshape4d = @import("ops.zig").reshape4d;
pub const repeat = @import("ops.zig").repeat;
pub const repeat4d = @import("ops.zig").repeat4d;
pub const transpose = @import("ops.zig").transpose;
pub const concat = @import("ops.zig").concat;
pub const getRows = @import("ops.zig").getRows;
pub const dupTensor = @import("ops.zig").dupTensor;
pub const conv1d = @import("ops.zig").conv1d;
pub const ssmConv = @import("ops.zig").ssmConv;
pub const ssmScan = @import("ops.zig").ssmScan;
pub const sumRows = @import("ops.zig").sumRows;
pub const setOutput = @import("ops.zig").setOutput;

pub const arange = @import("ops.zig").arange;
pub const fill = @import("ops.zig").fill;
pub const interpolate = @import("ops.zig").interpolate;
pub const padReflect1d = @import("ops.zig").padReflect1d;
pub const roll = @import("ops.zig").roll;
pub const timestepEmbedding = @import("ops.zig").timestepEmbedding;
pub const pool1d = @import("ops.zig").pool1d;
pub const getRelPos = @import("ops.zig").getRelPos;
pub const addRelPos = @import("ops.zig").addRelPos;
pub const addRelPosInplace = @import("ops.zig").addRelPosInplace;
pub const mapCustom1 = @import("ops.zig").mapCustom1;
pub const mapCustom2 = @import("ops.zig").mapCustom2;
pub const mapCustom3 = @import("ops.zig").mapCustom3;
pub const custom4d = @import("ops.zig").custom4d;

// ============================================================================
// 量化 API
// ============================================================================

pub const quantize = @import("quantize.zig");
pub const quantizeInit = quantize.quantizeInit;
pub const quantizeFree = quantize.quantizeFree;
pub const quantizeRequiresImatrix = quantize.quantizeRequiresImatrix;
pub const quantizeChunk = quantize.quantizeChunk;
pub const quantizeTensor = quantize.quantizeTensor;
pub const quantizedSize = quantize.quantizedSize;

pub const version = @import("utils.zig").version;
pub const cpuNThreads = @import("utils.zig").cpuNThreads;
pub const CpuFeatures = @import("utils.zig").CpuFeatures;
pub const recommendedThreads = @import("utils.zig").recommendedThreads;
pub const LogLevel = @import("utils.zig").LogLevel;
pub const logSet = @import("utils.zig").logSet;
pub const logSetCallback = @import("utils.zig").logSetCallback;

// ============================================================================
// 测试（集成测试，需要 ggml context）
// ============================================================================

const testing = std.testing;

test "ggml version" {
    const v = version();
    try testing.expect(v.major > 0);
}

test "ggml type sizes" {
    try testing.expect(@sizeOf(Type) == @sizeOf(c_uint));
    try testing.expect(@sizeOf(c.ggml_type) == @sizeOf(c_int));
}
