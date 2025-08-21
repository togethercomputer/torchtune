from typing import Any, Callable, Optional, Union

from torchtune.datasets import chat_dataset
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def open_thoughts_3_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "open-thoughts/OpenThoughts3-1.2M",
    # column_map: Optional[dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:

    return chat_dataset(
        tokenizer=tokenizer,
        source=source,
        conversation_column="conversations",
        conversation_style="sharegpt",
        train_on_input=train_on_input,
        new_system_prompt=new_system_prompt,
        packed=packed,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs)
