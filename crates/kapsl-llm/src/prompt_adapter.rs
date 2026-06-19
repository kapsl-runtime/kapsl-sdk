#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatPromptTemplate {
    Boundary {
        prefix: String,
        suffix: String,
    },
    Gemma,
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
        "chatml" | "qwen" | "qwen2" | "qwen2.5" | "qwen3" | "deepseek" | "gpt" | "openai" => {
            Some(ChatPromptTemplate::ChatMl)
        }
        "llama2" | "llama-2" => Some(ChatPromptTemplate::Llama2),
        "llama3" | "llama-3" | "llama3.1" | "llama-3.1" | "llama3.2" | "llama-3.2" | "llama3.3"
        | "llama-3.3" => Some(ChatPromptTemplate::Llama3 {
            bos_token: String::new(),
            think_suffix: String::new(),
        }),
        _ => None,
    }
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
        if lower.contains("qwen") || lower.contains("deepseek") {
            return Some(ChatPromptTemplate::ChatMl);
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
        prompt_is_explicitly_formatted, ChatPromptTemplate,
    };

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
            chat_template_from_explicit_name("llama2"),
            Some(ChatPromptTemplate::Llama2)
        );
        assert!(matches!(
            chat_template_from_explicit_name("llama3"),
            Some(ChatPromptTemplate::Llama3 { .. })
        ));
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
    }
}
