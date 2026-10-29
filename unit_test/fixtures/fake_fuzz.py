"""
fake_fuzz.py
============
測試用的「固定、可重現」菜單比對分數，取代 rapidfuzz。

為什麼：ocr_labeler 與真的 autolabeling.py 都是呼叫 process.extractOne(文字, 菜單, scorer=fuzz.token_set_ratio)。
要驗證的是「兩邊的邏輯一樣」（正規化、門檻判斷、shape 組裝），不是 rapidfuzz 的數值；
用同一個固定的假分數，兩邊的結果才能逐位元組比對，而且在任何機器上都得到同樣的標準答案。
"""
import difflib


class FakeProcess:
    @staticmethod
    def extractOne(query, choices, scorer=None, **kwargs):
        best = None
        for i, c in enumerate(choices):
            score = 100.0 * difflib.SequenceMatcher(None, query, c).ratio()
            if best is None or score > best[1]:
                best = (c, score, i)
        return best
