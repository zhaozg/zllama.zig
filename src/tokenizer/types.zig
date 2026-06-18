//! 分词器类型定义（兼容层）
//!
//! 大部分类型已迁移到 src/vocab.zig，此文件保留向后兼容的类型别名
//! 以及编码/解码模块所需的特定类型。

const vocab = @import("vocab");

// 重新导出 vocab 类型
pub const TokenType = vocab.TokenType;
pub const TokenAttr = vocab.TokenAttr;
pub const VocabType = vocab.VocabType;
pub const PreType = vocab.PreType;
pub const SpecialTokenIds = vocab.SpecialTokenIds;

// 向后兼容别名
pub const TokenizerModel = VocabType;
pub const PreTokenizerType = PreType;
pub const SpecialTokens = SpecialTokenIds;

/// 词表条目类型（编码/解码模块需要）
pub const VocabEntry = union(enum) {
    normal: []const u8,
    byte: u8,
};

/// 分词器配置（向后兼容，推荐直接使用 vocab.Vocab 的配置字段）
pub const TokenizerConfig = struct {
    model: TokenizerModel = .unknown,
    pre_type: PreTokenizerType = .default,
    add_space_prefix: bool = false,
    add_bos: bool = false,
    add_eos: bool = false,
    add_sep: bool = false,
    ignore_merges: bool = false,
    clean_spaces: bool = false,
    remove_extra_whitespaces: bool = false,
    escape_whitespaces: bool = true,
    treat_whitespace_as_suffix: bool = false,

    /// 从 Vocab 构建配置
    pub fn fromVocab(v: *const vocab.Vocab) TokenizerConfig {
        return TokenizerConfig{
            .model = v.getType(),
            .pre_type = v.getPreType(),
            .add_space_prefix = v.getAddSpacePrefix(),
            .add_bos = v.getAddBos(),
            .add_eos = v.getAddEos(),
            .add_sep = v.getAddSep(),
            .ignore_merges = v.getIgnoreMerges(),
            .clean_spaces = v.getCleanSpaces(),
            .remove_extra_whitespaces = v.getRemoveExtraWhitespaces(),
            .escape_whitespaces = v.getEscapeWhitespaces(),
            .treat_whitespace_as_suffix = v.getTreatWhitespaceAsSuffix(),
        };
    }
};
