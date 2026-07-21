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
pub const UnaryOp = @import("c.zig").UnaryOp;
pub const GluOp = @import("c.zig").GluOp;
pub const SortOrder = @import("c.zig").SortOrder;
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

// ============================================================================
// 计算图操作（ops.zig）
// ============================================================================

const ops = @import("ops.zig");

// 矩阵运算
pub const mulMat = ops.mulMat;
pub const mulMatSetPrec = ops.mulMatSetPrec;
pub const mulMatId = ops.mulMatId;
pub const outProd = ops.outProd;
pub const mul = ops.mul;
pub const add = ops.add;
pub const addInplace = ops.addInplace;
pub const addId = ops.addId;
pub const addCast = ops.addCast;
pub const sub = ops.sub;
pub const subInplace = ops.subInplace;
pub const div = ops.div;
pub const divInplace = ops.divInplace;
pub const neg = ops.neg;
pub const negInplace = ops.negInplace;
pub const exp = ops.exp;
pub const expInplace = ops.expInplace;
pub const sqrt = ops.sqrt;
pub const log = ops.log;
pub const abs = ops.abs;
pub const sin = ops.sin;
pub const cos = ops.cos;
pub const cpy = ops.cpy;
pub const cast = ops.cast;

// 归一化与激活
pub const rmsNorm = ops.rmsNorm;
pub const rmsNormInplace = ops.rmsNormInplace;
pub const rmsNormBack = ops.rmsNormBack;
pub const norm = ops.norm;
pub const normInplace = ops.normInplace;
pub const groupNorm = ops.groupNorm;
pub const groupNormInplace = ops.groupNormInplace;
pub const l2Norm = ops.l2Norm;
pub const l2NormInplace = ops.l2NormInplace;
pub const ropeExt = ops.ropeExt;
pub const ropeExtInplace = ops.ropeExtInplace;
pub const ropeExtBack = ops.ropeExtBack;
pub const rope = ops.rope;
pub const ropeInplace = ops.ropeInplace;
pub const ropeMulti = ops.ropeMulti;
pub const ropeMultiInplace = ops.ropeMultiInplace;
pub const ropeMultiBack = ops.ropeMultiBack;
pub const ropeYarnCorrDims = ops.ropeYarnCorrDims;
pub const scale = ops.scale;
pub const scaleInplace = ops.scaleInplace;
pub const scaleBias = ops.scaleBias;
pub const scaleBiasInplace = ops.scaleBiasInplace;
pub const softMax = ops.softMax;
pub const softMaxInplace = ops.softMaxInplace;
pub const softMaxExt = ops.softMaxExt;
pub const softMaxExtInplace = ops.softMaxExtInplace;
pub const softMaxExtBack = ops.softMaxExtBack;
pub const softMaxExtBackInplace = ops.softMaxExtBackInplace;
pub const softMaxAddSinks = ops.softMaxAddSinks;
pub const flashAttnExt = ops.flashAttnExt;
pub const flashAttnExtSetPrec = ops.flashAttnExtSetPrec;
pub const flashAttnExtAddSinks = ops.flashAttnExtAddSinks;
pub const flashAttnBack = ops.flashAttnBack;
pub const diagMaskInf = ops.diagMaskInf;
pub const diagMaskInfInplace = ops.diagMaskInfInplace;
pub const diagMaskZero = ops.diagMaskZero;
pub const diagMaskZeroInplace = ops.diagMaskZeroInplace;
pub const silu = ops.silu;
pub const siluInplace = ops.siluInplace;
pub const siluBack = ops.siluBack;
pub const sigmoid = ops.sigmoid;
pub const sigmoidInplace = ops.sigmoidInplace;
pub const softplus = ops.softplus;
pub const softplusInplace = ops.softplusInplace;
pub const gelu = ops.gelu;
pub const geluInplace = ops.geluInplace;
pub const geluErf = ops.geluErf;
pub const geluErfInplace = ops.geluErfInplace;
pub const geluQuick = ops.geluQuick;
pub const geluQuickInplace = ops.geluQuickInplace;
pub const sqr = ops.sqr;
pub const sqrInplace = ops.sqrInplace;
pub const sqrtInplace = ops.sqrtInplace;
pub const logInplace = ops.logInplace;
pub const tanh = ops.tanh;
pub const tanhInplace = ops.tanhInplace;
pub const relu = ops.relu;
pub const reluInplace = ops.reluInplace;
pub const step = ops.step;
pub const stepInplace = ops.stepInplace;
pub const elu = ops.elu;
pub const eluInplace = ops.eluInplace;
pub const leakyRelu = ops.leakyRelu;
pub const hardsigmoid = ops.hardsigmoid;
pub const hardswish = ops.hardswish;
pub const clamp = ops.clamp;
pub const absInplace = ops.absInplace;
pub const sgn = ops.sgn;
pub const sgnInplace = ops.sgnInplace;
pub const expm1 = ops.expm1;
pub const expm1Inplace = ops.expm1Inplace;
pub const sinInplace = ops.sinInplace;
pub const cosInplace = ops.cosInplace;
pub const floor = ops.floor;
pub const floorInplace = ops.floorInplace;
pub const ceil = ops.ceil;
pub const ceilInplace = ops.ceilInplace;
pub const round = ops.round;
pub const roundInplace = ops.roundInplace;
pub const trunc = ops.trunc;
pub const truncInplace = ops.truncInplace;

// GLU 变体
pub const swigluSplit = ops.swigluSplit;
pub const swiglu = ops.swiglu;
pub const swigluSwapped = ops.swigluSwapped;
pub const swigluOai = ops.swigluOai;
pub const gegluSplit = ops.gegluSplit;
pub const geglu = ops.geglu;
pub const gegluErf = ops.gegluErf;
pub const gegluErfSplit = ops.gegluErfSplit;
pub const gegluErfSwapped = ops.gegluErfSwapped;
pub const gegluQuick = ops.gegluQuick;
pub const gegluQuickSplit = ops.gegluQuickSplit;
pub const gegluQuickSwapped = ops.gegluQuickSwapped;
pub const gegluSwapped = ops.gegluSwapped;
pub const regluSplit = ops.regluSplit;
pub const reglu = ops.reglu;
pub const regluSwapped = ops.regluSwapped;
pub const gluSplit = ops.gluSplit;
pub const glu = ops.glu;
pub const xielu = ops.xielu;

// 张量操作
pub const permute = ops.permute;
pub const cont = ops.cont;
pub const cont1d = ops.cont1d;
pub const cont2d = ops.cont2d;
pub const cont3d = ops.cont3d;
pub const cont4d = ops.cont4d;
pub const dup = ops.dup;
pub const dupInplace = ops.dupInplace;
pub const reshape = ops.reshape;
pub const reshape1d = ops.reshape1d;
pub const reshape2d = ops.reshape2d;
pub const reshape3d = ops.reshape3d;
pub const reshape4d = ops.reshape4d;
pub const repeat = ops.repeat;
pub const repeat4d = ops.repeat4d;
pub const repeatBack = ops.repeatBack;
pub const transpose = ops.transpose;
pub const concat = ops.concat;
pub const getRows = ops.getRows;
pub const getRowsBack = ops.getRowsBack;
pub const setRows = ops.setRows;
pub const dupTensor = ops.dupTensor;
pub const diag = ops.diag;
pub const tri = ops.tri;

// 卷积与 SSM
pub const ssmConv = ops.ssmConv;
pub const ssmScan = ops.ssmScan;
pub const gatedDeltaNet = ops.gatedDeltaNet;
pub const gatedLinearAttn = ops.gatedLinearAttn;
pub const conv1d = ops.conv1d;
pub const conv1dPh = ops.conv1dPh;
pub const conv1dDw = ops.conv1dDw;
pub const conv1dDwPh = ops.conv1dDwPh;
pub const convTranspose1d = ops.convTranspose1d;
pub const conv2d = ops.conv2d;
pub const conv2dSkP0 = ops.conv2dSkP0;
pub const conv2dS1Ph = ops.conv2dS1Ph;
pub const conv2dDw = ops.conv2dDw;
pub const conv2dDirect = ops.conv2dDirect;
pub const conv2dDwDirect = ops.conv2dDwDirect;
pub const convTranspose2dP0 = ops.convTranspose2dP0;
pub const conv3d = ops.conv3d;
pub const conv3dDirect = ops.conv3dDirect;
pub const im2col = ops.im2col;
pub const im2colBack = ops.im2colBack;
pub const im2col3d = ops.im2col3d;
pub const col2im1d = ops.col2im1d;

// 输出/输入设置
pub const setOutput = ops.setOutput;
pub const setParam = ops.setParam;
pub const mulInplace = ops.mulInplace;
pub const fillInplace = ops.fillInplace;

// 归约操作
pub const sumRows = ops.sumRows;
pub const sum = ops.sum;
pub const cumsum = ops.cumsum;
pub const mean = ops.mean;
pub const argmax = ops.argmax;
pub const topK = ops.topK;
pub const argsort = ops.argsort;
pub const argsortTopK = ops.argsortTopK;
pub const countEqual = ops.countEqual;
pub const crossEntropyLoss = ops.crossEntropyLoss;
pub const crossEntropyLossBack = ops.crossEntropyLossBack;

// 张量生成与填充
pub const arange = ops.arange;
pub const fill = ops.fill;
pub const set = ops.set;
pub const set1d = ops.set1d;
pub const set2d = ops.set2d;
pub const setInplace = ops.setInplace;
pub const set1dInplace = ops.set1dInplace;
pub const set2dInplace = ops.set2dInplace;
pub const add1 = ops.add1;
pub const add1Inplace = ops.add1Inplace;
pub const acc = ops.acc;
pub const accInplace = ops.accInplace;

// 插值与缩放
pub const interpolate = ops.interpolate;
pub const upscale = ops.upscale;
pub const upscaleExt = ops.upscaleExt;

// 填充操作
pub const pad = ops.pad;
pub const padExt = ops.padExt;
pub const padCircular = ops.padCircular;
pub const padExtCircular = ops.padExtCircular;
pub const padReflect1d = ops.padReflect1d;

// 滚动
pub const roll = ops.roll;

// 时间步嵌入
pub const timestepEmbedding = ops.timestepEmbedding;

// 池化
pub const pool1d = ops.pool1d;
pub const pool2d = ops.pool2d;
pub const pool2dBack = ops.pool2dBack;

// 相对位置编码
pub const getRelPos = ops.getRelPos;
pub const addRelPos = ops.addRelPos;
pub const addRelPosInplace = ops.addRelPosInplace;

// 自定义算子
pub const mapCustom1 = ops.mapCustom1;
pub const mapCustom1Inplace = ops.mapCustom1Inplace;
pub const mapCustom2 = ops.mapCustom2;
pub const mapCustom2Inplace = ops.mapCustom2Inplace;
pub const mapCustom3 = ops.mapCustom3;
pub const mapCustom3Inplace = ops.mapCustom3Inplace;
pub const custom4d = ops.custom4d;
pub const customInplace = ops.customInplace;

// 图构建辅助
pub const buildForwardSelect = ops.buildForwardSelect;
pub const unary = ops.unary;
pub const unaryInplace = ops.unaryInplace;

// 张量创建与视图
pub const newTensor = ops.newTensor;
pub const newTensor1d = ops.newTensor1d;
pub const newTensor2d = ops.newTensor2d;
pub const newTensor3d = ops.newTensor3d;
pub const newTensor4d = ops.newTensor4d;
pub const view1d = ops.view1d;
pub const view2d = ops.view2d;
pub const view3d = ops.view3d;
pub const view4d = ops.view4d;
pub const newBuffer = ops.newBuffer;
pub const viewTensor = ops.viewTensor;
pub const getFirstTensor = ops.getFirstTensor;
pub const getNextTensor = ops.getNextTensor;
pub const getTensor = ops.getTensor;
pub const unravelIndex = ops.unravelIndex;
pub const getUnaryOp = ops.getUnaryOp;
pub const getGluOp = ops.getGluOp;

// 特殊模型算子
pub const winPart = ops.winPart;
pub const winUnpart = ops.winUnpart;
pub const solveTri = ops.solveTri;
pub const lightningIndexer = ops.lightningIndexer;
pub const rwkvWkv6 = ops.rwkvWkv6;
pub const rwkvWkv7 = ops.rwkvWkv7;
pub const dsv4HcPre = ops.dsv4HcPre;
pub const dsv4HcPost = ops.dsv4HcPost;
pub const dsv4HcComb = ops.dsv4HcComb;
pub const optStepAdamw = ops.optStepAdamw;
pub const optStepSgd = ops.optStepSgd;

// 量化/反量化
pub const dequantizeRow = ops.dequantizeRow;
pub const dequantizeTensor = ops.dequantizeTensor;

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
