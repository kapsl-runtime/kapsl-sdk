//! Chat-template detection and prompt rendering for supported model families.

/// One turn of a conversation, as sent by an OpenAI-style chat client.
///
/// `role` is matched case-insensitively against `system`, `user`, and
/// `assistant`; anything else is rendered as a user turn, since no template
/// family here has a slot for arbitrary roles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

impl ChatTurn {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    fn is_role(&self, expected: &str) -> bool {
        self.role.trim().eq_ignore_ascii_case(expected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatPromptTemplate {
    Boundary {
        prefix: String,
        suffix: String,
    },
    Gemma,
    Qwen,
    DeepSeek,
    ChatMl,
    Llama2,
    Llama3 {
        bos_token: String,
        think_suffix: String,
    },
}

impl ChatPromptTemplate {
    pub fn render(&self, prompt: &str) -> String {
        if prompt.trim().is_empty() || prompt_is_explicitly_formatted(prompt) {
            return prompt.to_string();
        }

        let prompt = prompt.trim();
        match self {
            Self::Boundary { prefix, suffix } => format!("{prefix}{prompt}{suffix}"),
            Self::Gemma => {
                format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
            }
            Self::Qwen => format!(
                "<|im_start|>system\n{QWEN_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n\
                 <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            ),
            Self::DeepSeek => format!("<｜User｜>{prompt}<｜Assistant｜><think>\n"),
            Self::ChatMl => {
                format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
            }
            Self::Llama2 => format!("[INST] {prompt} [/INST]"),
            Self::Llama3 {
                bos_token,
                think_suffix,
            } => format!(
                "{bos_token}<|start_header_id|>user<|end_header_id|>\n\n\
                 {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\
                 {think_suffix}"
            ),
        }
    }

    /// Render a full multi-turn conversation.
    ///
    /// `render` only ever produces a single user turn, which loses every prior
    /// message when a chat client sends real history. This walks the turns and
    /// emits the same per-family markers for each one.
    ///
    /// When the final turn is an assistant message it is treated as a prefill:
    /// the turn's content is left open for the model to continue instead of
    /// being closed and followed by a fresh generation prompt.
    pub fn render_chat(&self, turns: &[ChatTurn]) -> String {
        if turns.is_empty() {
            return String::new();
        }

        // A single user turn is exactly what `render` already handles, and some
        // callers depend on its pass-through of pre-formatted prompts.
        if turns.len() == 1 && turns[0].is_role("user") {
            return self.render(&turns[0].content);
        }

        let prefill = turns
            .last()
            .filter(|turn| turn.is_role("assistant"))
            .map(|turn| turn.content.trim());
        let body = if prefill.is_some() {
            &turns[..turns.len() - 1]
        } else {
            turns
        };

        match self {
            Self::Gemma => render_gemma_chat(body, prefill),
            Self::Qwen => render_chatml_chat(body, prefill, Some(QWEN_DEFAULT_SYSTEM_PROMPT)),
            Self::DeepSeek => render_deepseek_chat(body, prefill),
            Self::ChatMl => render_chatml_chat(body, prefill, None),
            Self::Llama2 => render_llama2_chat(body, prefill),
            Self::Llama3 {
                bos_token,
                think_suffix,
            } => render_llama3_chat(body, prefill, bos_token, think_suffix),
            // `Boundary` is a single prefix/suffix pair with no per-turn
            // structure, so the best available rendering is a plain transcript
            // inside the boundary.
            Self::Boundary { prefix, suffix } => {
                let transcript = render_plain_transcript(body);
                match prefill {
                    Some(prefill) => format!("{prefix}{transcript}{suffix}{prefill}"),
                    None => format!("{prefix}{transcript}{suffix}"),
                }
            }
        }
    }
}

pub const QWEN_DEFAULT_SYSTEM_PROMPT: &str =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Gemma has no system role: its official template rejects one. Folding the
/// system text into the first user turn is the conventional workaround.
fn render_gemma_chat(turns: &[ChatTurn], prefill: Option<&str>) -> String {
    let (system, rest) = split_leading_system(turns);
    let mut out = String::new();
    let mut pending_system = system;

    for turn in rest {
        if turn.is_role("assistant") {
            out.push_str("<start_of_turn>model\n");
            out.push_str(turn.content.trim());
            out.push_str("<end_of_turn>\n");
            continue;
        }
        out.push_str("<start_of_turn>user\n");
        if let Some(system) = pending_system.take() {
            out.push_str(system);
            out.push_str("\n\n");
        }
        out.push_str(turn.content.trim());
        out.push_str("<end_of_turn>\n");
    }

    // A system-only conversation still needs somewhere to put the system text.
    if let Some(system) = pending_system {
        out.push_str("<start_of_turn>user\n");
        out.push_str(system);
        out.push_str("<end_of_turn>\n");
    }

    out.push_str("<start_of_turn>model\n");
    if let Some(prefill) = prefill {
        out.push_str(prefill);
    }
    out
}

fn render_chatml_chat(
    turns: &[ChatTurn],
    prefill: Option<&str>,
    default_system_prompt: Option<&str>,
) -> String {
    let mut out = String::new();
    if let Some(system) =
        default_system_prompt.filter(|_| turns.first().is_none_or(|turn| !turn.is_role("system")))
    {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    for turn in turns {
        let role = if turn.is_role("system") {
            "system"
        } else if turn.is_role("assistant") {
            "assistant"
        } else {
            "user"
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(turn.content.trim());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    if let Some(prefill) = prefill {
        out.push_str(prefill);
    }
    out
}

fn render_deepseek_chat(turns: &[ChatTurn], prefill: Option<&str>) -> String {
    let (system, turns) = split_leading_system(turns);
    let mut out = system.unwrap_or_default().to_string();

    for turn in turns {
        if turn.is_role("assistant") {
            out.push_str("<｜Assistant｜>");
            out.push_str(turn.content.trim());
            out.push_str("<｜end▁of▁sentence｜>");
        } else {
            out.push_str("<｜User｜>");
            out.push_str(turn.content.trim());
        }
    }

    out.push_str("<｜Assistant｜>");
    if let Some(prefill) = prefill {
        out.push_str(prefill);
    } else {
        out.push_str("<think>\n");
    }
    out
}

/// Llama 2 wraps the system prompt in `<<SYS>>` inside the first `[INST]`
/// block, then alternates `[INST] user [/INST] assistant`.
fn render_llama2_chat(turns: &[ChatTurn], prefill: Option<&str>) -> String {
    let (system, rest) = split_leading_system(turns);
    let mut out = String::new();
    let mut pending_system = system;
    let mut open_instruction = false;

    for turn in rest {
        if turn.is_role("assistant") {
            if open_instruction {
                out.push(' ');
                out.push_str(turn.content.trim());
                out.push(' ');
                open_instruction = false;
            }
            continue;
        }
        out.push_str("[INST] ");
        if let Some(system) = pending_system.take() {
            out.push_str("<<SYS>>\n");
            out.push_str(system);
            out.push_str("\n<</SYS>>\n\n");
        }
        out.push_str(turn.content.trim());
        out.push_str(" [/INST]");
        open_instruction = true;
    }

    if let Some(system) = pending_system {
        out.push_str("[INST] <<SYS>>\n");
        out.push_str(system);
        out.push_str("\n<</SYS>>\n\n [/INST]");
    }

    if let Some(prefill) = prefill {
        out.push(' ');
        out.push_str(prefill);
    }
    out
}

fn render_llama3_chat(
    turns: &[ChatTurn],
    prefill: Option<&str>,
    bos_token: &str,
    think_suffix: &str,
) -> String {
    let mut out = String::from(bos_token);
    for turn in turns {
        let role = if turn.is_role("system") {
            "system"
        } else if turn.is_role("assistant") {
            "assistant"
        } else {
            "user"
        };
        out.push_str("<|start_header_id|>");
        out.push_str(role);
        out.push_str("<|end_header_id|>\n\n");
        out.push_str(turn.content.trim());
        out.push_str("<|eot_id|>");
    }
    out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    match prefill {
        Some(prefill) => out.push_str(prefill),
        None => out.push_str(think_suffix),
    }
    out
}

fn render_plain_transcript(turns: &[ChatTurn]) -> String {
    let mut out = String::new();
    for turn in turns {
        out.push_str(turn.role.trim());
        out.push_str(": ");
        out.push_str(turn.content.trim());
        out.push('\n');
    }
    out
}

/// Split off a leading system turn. Templates with no system slot need that
/// text hoisted somewhere else; any later system turns fall through and are
/// rendered as ordinary turns rather than being silently dropped.
fn split_leading_system(turns: &[ChatTurn]) -> (Option<&str>, &[ChatTurn]) {
    match turns.split_first() {
        Some((first, rest)) if first.is_role("system") => (Some(first.content.trim()), rest),
        _ => (None, turns),
    }
}

pub fn prompt_is_explicitly_formatted(prompt: &str) -> bool {
    const CHAT_MARKERS: &[&str] = &[
        "<start_of_turn>",
        "<end_of_turn>",
        "<|im_start|>",
        "<|im_end|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "[INST]",
        "[/INST]",
        "### Instruction:",
        "### Response:",
    ];

    CHAT_MARKERS.iter().any(|marker| prompt.contains(marker))
}

pub fn chat_template_from_explicit_name(name: &str) -> Option<ChatPromptTemplate> {
    let lower = name.trim().to_ascii_lowercase();
    match lower.as_str() {
        "" | "none" | "raw" | "completion" | "false" => None,
        "gemma" | "gemma2" | "gemma-2" | "gemma3" | "gemma-3" => Some(ChatPromptTemplate::Gemma),
        "qwen" | "qwen2" | "qwen2.5" | "qwen3" => Some(ChatPromptTemplate::Qwen),
        "deepseek" | "deepseek-r1" | "deepseek-v2" | "deepseek-v3" => {
            Some(ChatPromptTemplate::DeepSeek)
        }
        "chatml" | "gpt" | "openai" => Some(ChatPromptTemplate::ChatMl),
        "llama2" | "llama-2" => Some(ChatPromptTemplate::Llama2),
        "llama3" | "llama-3" | "llama3.1" | "llama-3.1" | "llama3.2" | "llama-3.2" | "llama3.3"
        | "llama-3.3" => Some(ChatPromptTemplate::Llama3 {
            bos_token: String::new(),
            think_suffix: String::new(),
        }),
        _ => None,
    }
}

/// Recognize the structural family of a Hugging Face tokenizer chat template.
///
/// We deliberately do not execute arbitrary Jinja here. The tokenizer templates
/// used by common model families can contain custom functions and filters that
/// are not portable outside Transformers. Detecting their role markers keeps
/// ONNX prompting model-specific without pretending that generic ChatML and
/// Qwen's default-system variant are identical.
pub fn chat_template_from_template_source(template: &str) -> Option<ChatPromptTemplate> {
    if template.contains("<start_of_turn>") && template.contains("<end_of_turn>") {
        return Some(ChatPromptTemplate::Gemma);
    }
    if template.contains("<|im_start|>") && template.contains("<|im_end|>") {
        return Some(if template.contains(QWEN_DEFAULT_SYSTEM_PROMPT) {
            ChatPromptTemplate::Qwen
        } else {
            ChatPromptTemplate::ChatMl
        });
    }
    if template.contains("<｜User｜>") && template.contains("<｜Assistant｜>") {
        return Some(ChatPromptTemplate::DeepSeek);
    }
    if template.contains("<|start_header_id|>")
        && template.contains("<|end_header_id|>")
        && template.contains("<|eot_id|>")
    {
        return Some(ChatPromptTemplate::Llama3 {
            bos_token: String::new(),
            think_suffix: if template.contains("<think>") {
                "<think>\n".to_string()
            } else {
                String::new()
            },
        });
    }
    if template.contains("[INST]") && template.contains("[/INST]") {
        return Some(ChatPromptTemplate::Llama2);
    }
    None
}

pub fn chat_template_from_model_identifiers<'a, I>(identifiers: I) -> Option<ChatPromptTemplate>
where
    I: IntoIterator<Item = &'a str>,
{
    for identifier in identifiers {
        let lower = identifier.to_ascii_lowercase();
        if lower.contains("gemma") {
            return Some(ChatPromptTemplate::Gemma);
        }
        if lower.contains("deepseek") {
            return Some(ChatPromptTemplate::DeepSeek);
        }
        if lower.contains("qwen") {
            return Some(ChatPromptTemplate::Qwen);
        }
        if lower.contains("llama-3") || lower.contains("llama3") {
            return Some(ChatPromptTemplate::Llama3 {
                bos_token: String::new(),
                think_suffix: String::new(),
            });
        }
        if lower.contains("llama-2") || lower.contains("llama2") {
            return Some(ChatPromptTemplate::Llama2);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{
        chat_template_from_explicit_name, chat_template_from_model_identifiers,
        chat_template_from_template_source, prompt_is_explicitly_formatted, ChatPromptTemplate,
        ChatTurn, QWEN_DEFAULT_SYSTEM_PROMPT,
    };

    fn turns(pairs: &[(&str, &str)]) -> Vec<ChatTurn> {
        pairs
            .iter()
            .map(|(role, content)| ChatTurn::new(*role, *content))
            .collect()
    }

    #[test]
    fn single_user_turn_matches_single_turn_render() {
        for template in [
            ChatPromptTemplate::Gemma,
            ChatPromptTemplate::Qwen,
            ChatPromptTemplate::DeepSeek,
            ChatPromptTemplate::ChatMl,
            ChatPromptTemplate::Llama2,
            ChatPromptTemplate::Llama3 {
                bos_token: String::new(),
                think_suffix: String::new(),
            },
        ] {
            assert_eq!(
                template.render_chat(&turns(&[("user", "hello")])),
                template.render("hello"),
                "{template:?} multi-turn render diverged on a single user turn"
            );
        }
    }

    #[test]
    fn qwen_adds_its_default_system_prompt_only_when_missing() {
        let rendered = ChatPromptTemplate::Qwen.render_chat(&turns(&[("user", "who are you?")]));
        assert_eq!(
            rendered,
            format!(
                "<|im_start|>system\n{QWEN_DEFAULT_SYSTEM_PROMPT}<|im_end|>\n\
                 <|im_start|>user\nwho are you?<|im_end|>\n\
                 <|im_start|>assistant\n"
            )
        );

        let rendered = ChatPromptTemplate::Qwen.render_chat(&turns(&[
            ("system", "Use a custom identity."),
            ("user", "who are you?"),
        ]));
        assert!(rendered.starts_with(
            "<|im_start|>system\nUse a custom identity.<|im_end|>\n<|im_start|>user\n"
        ));
        assert!(!rendered.contains(QWEN_DEFAULT_SYSTEM_PROMPT));
    }

    #[test]
    fn deepseek_uses_its_native_role_tokens_and_reasoning_prefill() {
        let rendered = ChatPromptTemplate::DeepSeek.render_chat(&turns(&[
            ("system", "Be concise."),
            ("user", "hello"),
            ("assistant", "Hi."),
            ("user", "again"),
        ]));
        assert_eq!(
            rendered,
            "Be concise.<｜User｜>hello<｜Assistant｜>Hi.<｜end▁of▁sentence｜>\
             <｜User｜>again<｜Assistant｜><think>\n"
        );
    }

    #[test]
    fn chatml_renders_every_turn_in_order() {
        let rendered = ChatPromptTemplate::ChatMl.render_chat(&turns(&[
            ("system", "Be terse."),
            ("user", "hi"),
            ("assistant", "hey"),
            ("user", "again"),
        ]));
        assert_eq!(
            rendered,
            "<|im_start|>system\nBe terse.<|im_end|>\n\
             <|im_start|>user\nhi<|im_end|>\n\
             <|im_start|>assistant\nhey<|im_end|>\n\
             <|im_start|>user\nagain<|im_end|>\n\
             <|im_start|>assistant\n"
        );
    }

    #[test]
    fn gemma_folds_system_into_the_first_user_turn() {
        let rendered = ChatPromptTemplate::Gemma
            .render_chat(&turns(&[("system", "Be terse."), ("user", "hi")]));
        assert_eq!(
            rendered,
            "<start_of_turn>user\nBe terse.\n\nhi<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn gemma_maps_assistant_turns_to_the_model_role() {
        let rendered = ChatPromptTemplate::Gemma.render_chat(&turns(&[
            ("user", "hi"),
            ("assistant", "hey"),
            ("user", "again"),
        ]));
        assert!(
            rendered.contains("<start_of_turn>model\nhey<end_of_turn>"),
            "assistant turn should render as Gemma's model role: {rendered}"
        );
    }

    #[test]
    fn llama2_wraps_system_in_the_first_instruction_block() {
        let rendered = ChatPromptTemplate::Llama2.render_chat(&turns(&[
            ("system", "Be terse."),
            ("user", "hi"),
            ("assistant", "hey"),
            ("user", "again"),
        ]));
        assert_eq!(
            rendered,
            "[INST] <<SYS>>\nBe terse.\n<</SYS>>\n\nhi [/INST] hey [INST] again [/INST]"
        );
    }

    #[test]
    fn llama3_emits_a_header_per_turn_and_opens_the_assistant_turn() {
        let rendered = ChatPromptTemplate::Llama3 {
            bos_token: "<|begin_of_text|>".to_string(),
            think_suffix: String::new(),
        }
        .render_chat(&turns(&[("system", "Be terse."), ("user", "hi")]));
        assert_eq!(
            rendered,
            "<|begin_of_text|>\
             <|start_header_id|>system<|end_header_id|>\n\nBe terse.<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn trailing_assistant_turn_is_left_open_as_a_prefill() {
        let rendered = ChatPromptTemplate::ChatMl.render_chat(&turns(&[
            ("user", "count to three"),
            ("assistant", "1, 2,"),
        ]));
        assert!(
            rendered.ends_with("<|im_start|>assistant\n1, 2,"),
            "prefill should stay open for continuation: {rendered}"
        );
        assert_eq!(
            rendered.matches("<|im_start|>assistant").count(),
            1,
            "prefill should not also emit a fresh generation prompt: {rendered}"
        );
    }

    #[test]
    fn empty_conversation_renders_empty() {
        assert_eq!(ChatPromptTemplate::ChatMl.render_chat(&[]), "");
    }

    #[test]
    fn unknown_roles_fall_back_to_user() {
        let rendered = ChatPromptTemplate::ChatMl
            .render_chat(&turns(&[("tool", "some tool output"), ("user", "hi")]));
        assert!(
            rendered.starts_with("<|im_start|>user\nsome tool output<|im_end|>"),
            "unknown role should render as a user turn: {rendered}"
        );
    }

    #[test]
    fn raw_user_prompt_is_not_marked_as_formatted() {
        assert!(!prompt_is_explicitly_formatted(
            "What is the relationship between a start of turn and an end of turn in conversation?"
        ));
    }

    #[test]
    fn known_chat_prompts_are_marked_as_formatted() {
        assert!(prompt_is_explicitly_formatted(
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        ));
        assert!(prompt_is_explicitly_formatted(
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        ));
        assert!(prompt_is_explicitly_formatted(
            "<｜User｜>Hello<｜Assistant｜><think>\n"
        ));
        assert!(prompt_is_explicitly_formatted("<s>[INST] Hello [/INST]"));
    }

    #[test]
    fn gemma_template_wraps_raw_prompt() {
        assert_eq!(
            ChatPromptTemplate::Gemma.render("  Hello  "),
            "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n"
        );
    }

    #[test]
    fn explicit_names_cover_common_chat_families() {
        assert_eq!(
            chat_template_from_explicit_name("gemma"),
            Some(ChatPromptTemplate::Gemma)
        );
        assert_eq!(
            chat_template_from_explicit_name("gpt"),
            Some(ChatPromptTemplate::ChatMl)
        );
        assert_eq!(
            chat_template_from_explicit_name("qwen2.5"),
            Some(ChatPromptTemplate::Qwen)
        );
        assert_eq!(
            chat_template_from_explicit_name("deepseek-r1"),
            Some(ChatPromptTemplate::DeepSeek)
        );
        assert_eq!(
            chat_template_from_explicit_name("llama2"),
            Some(ChatPromptTemplate::Llama2)
        );
        assert!(matches!(
            chat_template_from_explicit_name("llama3"),
            Some(ChatPromptTemplate::Llama3 { .. })
        ));
    }

    #[test]
    fn embedded_template_detection_preserves_model_family_differences() {
        let qwen = format!(
            "{{% for message in messages %}}<|im_start|>{{{{ message.role }}}}\n\
             {QWEN_DEFAULT_SYSTEM_PROMPT}<|im_end|>{{% endfor %}}"
        );
        assert_eq!(
            chat_template_from_template_source(&qwen),
            Some(ChatPromptTemplate::Qwen)
        );
        assert_eq!(
            chat_template_from_template_source(
                "{% for message in messages %}<|im_start|>{{ message.role }}<|im_end|>{% endfor %}"
            ),
            Some(ChatPromptTemplate::ChatMl)
        );
        assert_eq!(
            chat_template_from_template_source("<start_of_turn>{{ role }}<end_of_turn>"),
            Some(ChatPromptTemplate::Gemma)
        );
        assert!(matches!(
            chat_template_from_template_source(
                "<|start_header_id|>{{ role }}<|end_header_id|>{{ text }}<|eot_id|>"
            ),
            Some(ChatPromptTemplate::Llama3 { .. })
        ));
        assert_eq!(
            chat_template_from_template_source("[INST] {{ prompt }} [/INST]"),
            Some(ChatPromptTemplate::Llama2)
        );
        assert_eq!(
            chat_template_from_template_source(
                "{{ bos_token }}<｜User｜>{{ prompt }}<｜Assistant｜><think>"
            ),
            Some(ChatPromptTemplate::DeepSeek)
        );
    }

    #[test]
    fn auto_detection_avoids_generic_gpt_or_llama_base_names() {
        assert_eq!(chat_template_from_model_identifiers(["gpt2"]), None);
        assert_eq!(
            chat_template_from_model_identifiers(["LlamaForCausalLM"]),
            None
        );
        assert_eq!(
            chat_template_from_model_identifiers(["GemmaForCausalLM"]),
            Some(ChatPromptTemplate::Gemma)
        );
        assert_eq!(
            chat_template_from_model_identifiers(["Qwen2ForCausalLM"]),
            Some(ChatPromptTemplate::Qwen)
        );
        assert_eq!(
            chat_template_from_model_identifiers([
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "Qwen2ForCausalLM",
            ]),
            Some(ChatPromptTemplate::DeepSeek)
        );
    }
}
