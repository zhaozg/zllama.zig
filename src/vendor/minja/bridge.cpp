/*
 * C ABI bridge implementation for minja.hpp
 *
 * Wraps minja::chat_template and minja::Parser::parse/render in C-callable
 * functions. This is compiled as C++ and linked into the Zig binary.
 */

#include "bridge.hpp"

#include "chat-template.hpp"
#include "minja.hpp"

#include <string>
#include <stdexcept>

using json = nlohmann::ordered_json;

struct minja_chat_template_s {
    minja::chat_template tmpl;
    std::string last_error;

    minja_chat_template_s(const std::string &source,
                          const std::string &bos_token,
                          const std::string &eos_token)
        : tmpl(source, bos_token, eos_token) {}
};

extern "C" {

minja_chat_template_t *minja_chat_template_create(
    const char *source,
    const char *bos_token,
    const char *eos_token)
{
    if (!source || !bos_token || !eos_token) {
        return nullptr;
    }
    try {
        auto *handle = new minja_chat_template_s(source, bos_token, eos_token);
        return handle;
    } catch (const std::exception &e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void minja_chat_template_free(minja_chat_template_t *tmpl) {
    delete tmpl;
}

char *minja_chat_template_apply(
    const minja_chat_template_t *tmpl,
    const char *messages_json,
    const char *tools_json,
    bool add_generation_prompt)
{
    if (!tmpl || !messages_json) {
        return nullptr;
    }
    try {
        json messages;
        if (messages_json[0] != '\0') {
            messages = json::parse(messages_json);
        } else {
            messages = json::array();
        }

        json tools;
        if (tools_json && tools_json[0] != '\0') {
            tools = json::parse(tools_json);
        }

        minja::chat_template_inputs inputs;
        inputs.messages = messages;
        inputs.tools = tools;
        inputs.add_generation_prompt = add_generation_prompt;
        // Match llama.cpp default: enable_thinking=true for all models.
        // Templates that don't use this variable simply ignore it.
        inputs.extra_context = json::object({{"enable_thinking", true}});

        std::string result = tmpl->tmpl.apply(inputs);

        // Copy to C string
        char *cstr = new char[result.size() + 1];
        std::memcpy(cstr, result.c_str(), result.size() + 1);
        return cstr;
    } catch (const std::exception &e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

void minja_free_string(char *s) {
    delete[] s;
}

char *minja_render(
    const char *template_str,
    const char *context_json)
{
    if (!template_str) {
        return nullptr;
    }
    try {
        auto tmpl = minja::Parser::parse(template_str, {
            /* .trim_blocks = */ true,
            /* .lstrip_blocks = */ true,
            /* .keep_trailing_newline = */ false,
        });

        json context;
        if (context_json && context_json[0] != '\0') {
            context = json::parse(context_json);
        }

        auto ctx = minja::Context::make(minja::Value(context));
        std::string result = tmpl->render(ctx);

        char *cstr = new char[result.size() + 1];
        std::memcpy(cstr, result.c_str(), result.size() + 1);
        return cstr;
    } catch (const std::exception &e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

} // extern "C"
