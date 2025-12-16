# -*- coding: utf-8 -*-
"""
Translation Service
提供免费的中文->英文翻译服务封装，优先调用开源/免费在线接口，失败时回退到内置小词典。

- 首选：LibreTranslate 社区实例（无需密钥）
- 兜底：MyMemory 免费接口
- 最终兜底：内置医学超声相关术语小词典 + 简单规则

注意：本模块不在前端展示翻译，仅用于后端自动补全英文字段。
"""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class TranslationResult:
    text: str
    provider: str  # 'libre' | 'mymemory' | 'fallback'


class TranslationService:
    """轻量翻译服务，包含简单的内存缓存与多级回退。"""

    def __init__(self) -> None:
        # 简单的内存缓存：key -> (translated, ts)
        self._cache: dict[str, tuple[str, float]] = {}
        self._ttl_seconds = 24 * 3600

    # -----------------
    # 公共方法
    # -----------------
    def translate_text(self, text: str, source: str = 'zh', target: str = 'en') -> TranslationResult:
        text = (text or '').strip()
        if not text:
            return TranslationResult(text='', provider='fallback')

        key = self._cache_key(text, source, target)
        cached = self._cache.get(key)
        if cached and (time.time() - cached[1] < self._ttl_seconds):
            return TranslationResult(text=cached[0], provider='cache')

        # 1) LibreTranslate（社区实例）
        try:
            res = self._translate_via_libre(text, source, target)
            if res:
                self._cache[key] = (res, time.time())
                return TranslationResult(text=res, provider='libre')
        except Exception:
            pass

        # 2) MyMemory 免费接口
        try:
            res = self._translate_via_mymemory(text, source, target)
            if res:
                self._cache[key] = (res, time.time())
                return TranslationResult(text=res, provider='mymemory')
        except Exception:
            pass

        # 3) 内置兜底词典与规则
        fallback = self._fallback_translate(text)
        self._cache[key] = (fallback, time.time())
        return TranslationResult(text=fallback, provider='fallback')

    # -----------------
    # 具体提供者
    # -----------------
    def _translate_via_libre(self, text: str, source: str, target: str) -> Optional[str]:
        url_candidates = [
            # Argos OpenTech 的公共实例（可能限流）
            'https://translate.argosopentech.com/translate',
            # 官方公共实例
            'https://libretranslate.com/translate'
        ]
        payload = {
            'q': text,
            'source': source,
            'target': target,
            'format': 'text'
        }
        headers = {'Content-Type': 'application/json'}
        for url in url_candidates:
            try:
                r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=2.0)
                if r.status_code == 200:
                    data = r.json()
                    translated = (data.get('translatedText') or '').strip()
                    if translated:
                        return translated
            except Exception:
                continue
        return None

    def _translate_via_mymemory(self, text: str, source: str, target: str) -> Optional[str]:
        # 语言对映射
        langpair = f"{self._to_mymemory_lang(source)}|{self._to_mymemory_lang(target)}"
        url = f"https://api.mymemory.translated.net/get?q={requests.utils.quote(text)}&langpair={langpair}"
        r = requests.get(url, timeout=2.0)
        if r.status_code == 200:
            data = r.json()
            translated = ((data.get('responseData') or {}).get('translatedText') or '').strip()
            if translated:
                return translated
        return None

    # -----------------
    # 兜底逻辑
    # -----------------
    def _fallback_translate(self, text: str) -> str:
        # 特定规则：否定检测
        if '未检测到' in text:
            return 'not detected'

        # 简易术语词典（可按需扩充）
        dictionary = {
            '皮肤': 'Skin',
            '皮肤层': 'Skin layer',
            '皮下组织': 'Subcutaneous tissue',
            '皮下脂肪': 'Subcutaneous fat',
            '脂肪层': 'Fat layer',
            '乳腺': 'Mammary gland',
            '乳腺层': 'Breast gland layer',
            '乳腺组织': 'Breast tissue',
            '乳房': 'Breast',
            '肿块': 'Mass',
            '结节': 'Nodule',
            '血管': 'Vessel',
            '淋巴结': 'Lymph node',
            '肌肉': 'Muscle',
            '肌肉层': 'Muscle layer',
            '脂肪': 'Fat',
            '腺体': 'Gland',
            '腺体层': 'Gland layer',
            '腺体组织': 'Glandular tissue',
            '筋膜': 'Fascia',
            '筋膜层': 'Fascia layer',
            '囊肿': 'Cyst',
        }

        # 直接命中词典
        if text in dictionary:
            return dictionary[text]

        # 规则：常见后缀映射（面向短术语/层名）
        suffix_map = {
            '层': ' layer',
            '组织': ' tissue',
        }
        for suffix, en_suffix in suffix_map.items():
            if text.endswith(suffix):
                base = text[:-len(suffix)]
                # 优先用完整中文词典的最长子串匹配
                longest_match = ''
                longest_translation = ''
                for zh_term, en_term in dictionary.items():
                    if zh_term and zh_term in base and len(zh_term) > len(longest_match):
                        longest_match = zh_term
                        longest_translation = en_term
                if longest_translation:
                    return f"{longest_translation}{en_suffix}"
                # 次选：如果base本身有词典翻译
                if base in dictionary:
                    return f"{dictionary[base]}{en_suffix}"
                # 最终兜底：直接英文后缀
                return f"{base}{en_suffix}"

        # 若无匹配，保持原文（避免返回空字符串）
        return text

    # -----------------
    # 工具方法
    # -----------------
    def _cache_key(self, text: str, source: str, target: str) -> str:
        raw = f"{source}->{target}:{text}"
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()

    def _to_mymemory_lang(self, code: str) -> str:
        mapping = {
            'zh': 'zh-CN',
            'en': 'en-US'
        }
        return mapping.get(code, code)


_service_singleton: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = TranslationService()
    return _service_singleton