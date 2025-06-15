import json
import re
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from src.utils import logger
import torch
from src.metrics import GenrePredictorInterface
from transformers import StaticCache
import gc



def parse_model_response(response: str) -> int:
    try:
        match = re.search(r'\{[^}]*"predict"\s*:\s*(0|1)[^}]*\}', response)
        if match:
            data = json.loads(match.group(0))
            return int(data['predict'])
    except Exception as e:
        print(f"Parsing error: {e}")
    raise ValueError("Could not parse prediction from model response.")


class ZeroShotClassifier(GenrePredictorInterface):
    def __init__(self, model, tokenizer, genres, prompt_template, device="cuda", max_lyrics_length=300):
        self.model = model
        self.tokenizer = tokenizer
        self.genres = genres  # список всех возможных жанров
        self.device = device
        self.max_lyrics_length = max_lyrics_length
        self.prompt_template = prompt_template
        
    def _make_prompts(self, lyrics: str) -> list[str]:
        truncated = lyrics[:self.max_lyrics_length].replace('\n', ' ').replace('"', "'")
        prompts = [self.prompt_template % (truncated, genre) for genre in self.genres]
        return prompts 

    def _parse_response(self, response: str) -> int:
        try:
            match = re.search(r'\{[^}]*"predict"\s*:\s*(0|1)[^}]*\}', response)
            if match:
                data = json.loads(match.group(0))
                return int(data["predict"])
        except Exception as e:
            print(f"Parse error: {e}")
        return 0  # fallback to 0 if anything goes wrong
    

def make_prompts(lyrics: str, genres: list, prompt: str) -> list[str]:
    truncated = lyrics[:300].replace('\n', ' ').replace('"', "'")
    prompts = [prompt % (truncated, genre) for genre in genres]
    return prompts

def parse_response(response: str) -> int:
    try:
        match = re.search(r'\{[^}]*"predict"\s*:\s*(0|1)[^}]*\}', response)
        if match:
            data = json.loads(match.group(0))
            return int(data["predict"])
    except Exception as e:
        print(f"Parse error: {e}")
    return 0  # fallback to 0 if anything goes wrong


def generate_with_prefix_cache_batch(
    model,
    tokenizer,
    prefix_past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    postfix_texts: List[str],
    device: str = "cuda",
    max_new_tokens: int = 128,
) -> List[str]:
    model.eval()

    batch_size = len(postfix_texts)

    # 1. Токенизируем все postfix'ы, паддим
    inputs = tokenizer(
        postfix_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).to(device)

    input_ids = inputs["input_ids"]  # [B, L]

    # 2. Клонируем prefix_past_key_values на батч
    static_cache = StaticCache.from_legacy_cache(prefix_past_key_values)

    # Повторение кеша для каждого элемента в батче
    batched_cache = static_cache.expand(batch_size)
    
    # 3. Подготовка
    generated = [[] for _ in range(batch_size)]
    is_finished = [False] * batch_size

    # 4. Генерация токенов
    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=batched_cache,
            use_cache=True,
        )
        logits = outputs.logits[:, -1, :]  # [B, vocab_size]
        batched_cache = outputs.past_key_values

        next_token_ids = torch.argmax(logits, dim=-1)  # [B]

        for i, token_id in enumerate(next_token_ids.tolist()):
            if not is_finished[i]:
                if token_id == tokenizer.eos_token_id:
                    is_finished[i] = True
                else:
                    generated[i].append(token_id)

        if all(is_finished):
            break

        # Обновляем input_ids: shape [B, 1]
        input_ids = next_token_ids.unsqueeze(1)

    # 5. Декодирование результатов
    return [
        tokenizer.decode(g, skip_special_tokens=True)
        for g in generated
    ]


def generate_with_prefix_cache(
    model,
    tokenizer,
    prefix_past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    postfix_text: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Генерирует продолжение для postfix_text, используя заранее посчитанный
    past_key_values для общего префикса.

    Args:
        model: AutoModelForCausalLM с поддержкой use_cache.
        tokenizer: соответствующий AutoTokenizer.
        prefix_past_key_values: past_key_values, полученный один раз для префикса.
        postfix_text: текст, который следует токенизировать и «досчитать».
        device: device для тензоров.
        max_new_tokens: сколько токенов генерировать.

    Returns:
        generated_ids: тензор [1, L] с ID сгенерированных токенов.
        full_sequence_ids: список всех ID (postfix + сгенерированных).
    """
    model.eval()

    # 1) Токенизируем сразу postfix, но без учёта префикса в input_ids
    inputs = tokenizer(
        postfix_text,
        return_tensors="pt",
        add_special_tokens=False  # важно, чтобы не двойных BOS/EOS
    ).to(device)
    input_ids = inputs["input_ids"]

    # 2) Устанавливаем past = префикс‑кеш
    past = prefix_past_key_values

    generated_ids = []

    # 3) Генерируем по одному токену
    for _ in range(max_new_tokens):
        # forward с use_cache=True и подставленным past
        outputs = model(
            input_ids=input_ids,
            past_key_values=past,
            use_cache=True,
        )
        # logits последнего шага: [1, 1, vocab_size]
        logits = outputs.logits[:, -1, :]

        # обновляем кеш
        past = outputs.past_key_values

        # выбираем следующий токен режимом greedy
        next_token_id = torch.argmax(logits, dim=-1)  # shape [1]

        # Опционально можно досрочно выйти, увидев EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id.item())

        # готовим input_ids для следующего шага: только что предсказанный токен
        input_ids = next_token_id.unsqueeze(-1)       # shape [1,1]


    return tokenizer.decode(generated_ids)


def batch_generate_with_prefix_cache(
    model,
    tokenizer,
    prefix_past_key_values,
    postfix_ids: torch.Tensor,           # [B, T]
    postfix_mask: torch.Tensor,          # [B, T]
    device: str = "cuda",
    max_new_tokens: int = 256,
    debug = False
) -> List[str]:
    model.eval()

    input_ids = postfix_ids.to(device)
    postfix_mask = postfix_mask.to(device)
    batch_size = input_ids.size(0)

    past = prefix_past_key_values
    generated_ids = [[] for _ in range(batch_size)]
    finished = [False] * batch_size

    # Определяем длину префикса (по количеству key/value слоёв и их размерности)
    prefix_len = past[0][0].size(-2)  # [num_layers][0=key or 1=value][B, H, prefix_len, D]

    with torch.inference_mode():
        for step in range(max_new_tokens):
            model_inputs = {
                "input_ids": input_ids,
                "past_key_values": past,
                "use_cache": True
            }

            if step == 0:
                # Добавляем attention_mask на первый шаг: [B, prefix_len + postfix_len]
                # Префиксная часть (всё единицы, потому что они уже прошли)
                prefix_attention_mask = torch.ones((batch_size, prefix_len), dtype=torch.long, device=device)
                full_attention_mask = torch.cat([prefix_attention_mask, postfix_mask], dim=1)  # [B, prefix + postfix]
                model_inputs["attention_mask"] = full_attention_mask

            outputs = model(**model_inputs)
            logits = outputs.logits[:, -1, :]                # [B, V]
            next_tokens = torch.argmax(logits, dim=-1)       # [B]

            del input_ids, past
            past = outputs.past_key_values
            del outputs
            input_ids = next_tokens.unsqueeze(-1)            # [B, 1]

            for i in range(batch_size):
                if not finished[i]:
                    token = next_tokens[i].item()
                    if token == tokenizer.eos_token_id:
                        finished[i] = True
                    else:
                        generated_ids[i].append(token)

            if all(finished):
                break

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]


class ZeroShotClassifierV1(ZeroShotClassifier):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        genres: List[str],
        prompt_template: str,
        use_kv_cache: bool = True,
        device: str = "cuda",
        max_lyrics_length: int = 300,
        batch_size: int = 2
    ):
        super().__init__(model, tokenizer, genres, prompt_template, device, max_lyrics_length)
        self.use_kv_cache = use_kv_cache
        self.batch_size = batch_size

        # Prepare the prefix tokens once
        prefix_template = prompt_template.split("%s")[0]
        tokenized = tokenizer(
            prefix_template,
            return_tensors="pt",
            add_special_tokens=False
        )
        self.prefix_ids = tokenized["input_ids"].to(self.device)[0]
        self.prefix_len = self.prefix_ids.size(0)
        logger.info(f"Initialized classifier with prefix_len={self.prefix_len}")

    def _build_prompts_and_map(
        self, lyrics_list: List[str]
    ) -> Tuple[List[str], List[int]]:
        prompts, idx_map = [], []
        for idx, txt in enumerate(lyrics_list):
            trunc = txt[: self.max_lyrics_length]
            for _g in self.genres:
                prompts.append(self.prompt_template % (trunc, _g))
                idx_map.append(idx)
        return prompts, idx_map

    def predict(
        self,
        batch: Dict,
        enable_thinking: bool = False,
        debug: bool = False
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        lyrics = [f["lyrics"] for f in batch["features"]]
        all_prompts, idx_map = self._build_prompts_and_map(lyrics)
        if debug:
            logger.info(f"Total prompts: {len(all_prompts)}")

        if not all_prompts:
            return np.zeros((0, len(self.genres))), [], []

        # Apply chat template
        full_texts = []
        for p in all_prompts:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                do_sample=False
            )
            full_texts.append(text)
        if debug:
            logger.info(f"Example full_text: {full_texts[0]}")

        # Tokenize batch
        tok = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        ).to(self.device)
        input_ids, attention_mask = tok.input_ids, tok.attention_mask
        if debug:
            logger.info(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}")

        generated = []

        # With KV cache
        if self.use_kv_cache:
            for i in range(0, input_ids.size(0), self.batch_size):
                batch_ids = input_ids[i : i + self.batch_size]
                batch_mask = attention_mask[i : i + self.batch_size]
                if debug:
                    logger.info(f"Processing batch {i}..{i+self.batch_size}, batch_size={batch_ids.size(0)}")

                postfix_ids = batch_ids[:, self.prefix_len :]
                postfix_mask = batch_mask[:, self.prefix_len :]
                if debug:
                    sample_tok = self.tokenizer.decode(postfix_ids[0], skip_special_tokens=False)
                    logger.info(f"Postfix sample: {sample_tok}")

                prefix_batch = self.prefix_ids.unsqueeze(0).repeat(postfix_ids.size(0), 1)
                with torch.no_grad():
                    out = self.model(
                        input_ids=prefix_batch,
                        use_cache=True
                    )
                    past = out.past_key_values
                    if debug:
                        logger.info("Obtained past_key_values from prefix run")

                cont = batch_generate_with_prefix_cache(
                    self.model,
                    self.tokenizer,
                    past,
                    postfix_ids,
                    postfix_mask,
                    device=self.device,
                    max_new_tokens=256,
                    debug=debug
                )
                generated.extend(cont)
                if debug:
                    logger.info(f"Generated {len(cont)} continuations for batch starting at {i}. Generated example: {cont}")

        # Without KV cache
        else:
            for i in range(0, len(full_texts), self.batch_size):
                sub = full_texts[i : i + self.batch_size]
                if debug:
                    logger.info(f"Generating batch (no KV) {i}..{i+self.batch_size}")
                inp = self.tokenizer(
                    sub,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                with torch.no_grad():
                    outs = self.model.generate(
                        **inp,
                        max_new_tokens=1024,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                for j in range(len(sub)):
                    start = inp["input_ids"][j].size(0)
                    txt = self.tokenizer.decode(outs[j][start:], skip_special_tokens=True).strip()
                    generated.append(txt)
                if debug:
                    logger.info(f"Generated {len(sub)} outputs (no KV) for batch starting at {i}")

        # Build prediction matrix
        B, G = len(lyrics), len(self.genres)
        preds = np.zeros((B, G), dtype=np.int32)
        for idx, text in enumerate(generated):
            i = idx_map[idx]
            g = idx % G
            try:
                preds[i, g] = self._parse_response(text)
            except Exception as e:
                logger.warning(f"Parse failed on '{text}': {e}")

        if debug:
            logger.info(f"Final predictions matrix shape: {preds.shape}")
        return preds, generated, full_texts
