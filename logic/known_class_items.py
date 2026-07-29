# logic/known_class_items.py
"""
不需要 OCR、光靠 mmrotate 分類就能得知身份的品項(醬料包等)。
與需要送 PaddleOCR 才能得知身份的一般貼紙 (order_sticker) 做出區隔,
是「身份解析策略」層級的分野,不只是穩定度門檻的分野。
"""

# cls_name -> 對應到菜單/訂單比對用的品項名稱
# 同一包醬料的正反面被視為同一品項
CLASS_TO_ITEM_NAME = {
    "sesame_front": "ごまドレッシング",
    "sesame_back": "ごまドレッシング",
    "wafusauce_front": "和風ドレッング",
    "wafusauce_back": "和風ドレッング",
    "cream_front": "マヨネーズ",
    "cream_back": "マヨネーズ",
}

# 需要「連續多幀確認同一品項」才建立成正式追蹤物件的類別
STABLE_CONFIRM_CLASSES = set(CLASS_TO_ITEM_NAME.keys())


def resolve_item_name(cls_name: str) -> str | None:
    """回傳這個 cls_name 對應的品項名稱;不是「類別即身份」的類別回傳 None"""
    return CLASS_TO_ITEM_NAME.get(cls_name)


def is_known_class_item(cls_name: str) -> bool:
    return cls_name in CLASS_TO_ITEM_NAME