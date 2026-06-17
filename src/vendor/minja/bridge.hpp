/*
 * C ABI bridge for minja.hpp
 *
 * Provides C-compatible wrappers around the C++ minja::chat_template class
 * so that Zig can call into it via @cImport.
 *
 * Reference: deps/chat-template.cpp
 */

#ifndef MINJA_BRIDGE_H
#define MINJA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

/// Opaque handle to a minja::chat_template instance.
typedef struct minja_chat_template_s minja_chat_template_t;

/// Create a chat_template from a Jinja source string.
/// source: Jinja template string (e.g., from GGUF tokenizer.chat_template)
/// bos_token: beginning-of-sequence token string
/// eos_token: end-of-sequence token string
/// Returns NULL on parse failure.
minja_chat_template_t *minja_chat_template_create(
    const char *source,
    const char *bos_token,
    const char *eos_token);

/// Destroy a chat_template instance.
void minja_chat_template_free(minja_chat_template_t *tmpl);

/// Apply the template to messages and optional tools.
/// messages_json: JSON array of message objects (role, content, ...)
/// tools_json: JSON array of tool definitions, or NULL/empty for none
/// add_generation_prompt: whether to append the generation prompt marker
/// Returns a heap-allocated string (caller must free with minja_free_string).
/// Returns NULL on error.
char *minja_chat_template_apply(
    const minja_chat_template_t *tmpl,
    const char *messages_json,
    const char *tools_json,
    bool add_generation_prompt);

/// Free a string returned by minja_chat_template_apply.
void minja_free_string(char *s);

/// Render a raw Jinja template with a JSON context (no message normalization).
/// template_str: Jinja template string
/// context_json: JSON object with template variables
/// Returns a heap-allocated string (caller must free with minja_free_string).
char *minja_render(
    const char *template_str,
    const char *context_json);

#ifdef __cplusplus
}
#endif

#endif /* MINJA_BRIDGE_H */
