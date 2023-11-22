VICUNA_END_OF_SYSTEM_PROMPT_SEQUENCE = "USER:"
VICUNA_END_OF_USER_PROMPT_SEQUENCE = "ASSISTANT:"


def fmt_vicuna_input(
    prompt: str, response: str, space_before_prompt: bool = True
) -> str:
    return (
        f"A chat between a curious user and an artificial intelligence assistant. "
        f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
        f"{VICUNA_END_OF_SYSTEM_PROMPT_SEQUENCE}{' ' if space_before_prompt else ''}{prompt} {VICUNA_END_OF_USER_PROMPT_SEQUENCE} {response}"
    )
