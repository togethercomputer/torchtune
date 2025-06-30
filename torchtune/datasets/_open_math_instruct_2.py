from typing import Any, Callable, Optional, Union

from torchtune.data import InputOutputToMessages
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


def open_math_instruct_2_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "nvidia/OpenMathInstruct-2",
    column_map: Optional[dict[str, str]] = None,
    train_on_input: bool = False,
    new_system_prompt: Optional[str] = None,
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
        OpenMathInstruct-2 dataset, follows the structure of C4 Grammar dataset.
    """
    message_transform = InputOutputToMessages(
        train_on_input=train_on_input,
        column_map=column_map,
        new_system_prompt=new_system_prompt,
    )
    ds = SFTDataset(
        source=source,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(ds, max_seq_len=tokenizer.max_seq_len)
    return ds
