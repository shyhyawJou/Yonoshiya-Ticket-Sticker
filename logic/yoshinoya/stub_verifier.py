"""
stub_verifier.py
================
客戶聯（stub / small_order_ticket）與主傳票（order_ticket）交叉比對工具。
純函式，僅產生驗證結果與 log，絕不影響主訂單完成狀態或發送 NG 告警。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from logic.yoshinoya.stub_parser import StubParse


@dataclass
class StubVerification:
    ticket_no_match: Optional[bool]
    items_match: bool
    missing_in_stub: Dict[str, int] = field(default_factory=dict)
    extra_in_stub: Dict[str, int] = field(default_factory=dict)
    qty_unconfirmed_in_stub: Tuple[str, ...] = ()

    @property
    def matched(self) -> bool:
        return (self.ticket_no_match is not False) and self.items_match


def _digits(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return d.lstrip("0") or ("0" if d else None)


def verify_stub(
    parse: StubParse,
    main_order_number: Optional[str],
    main_items: Sequence[Tuple[str, Optional[int]]],
) -> StubVerification:
    a, b = _digits(parse.ticket_no), _digits(main_order_number)
    if a is None or b is None:
        no_match: Optional[bool] = None
    else:
        no_match = a == b

    main_cnt: Counter = Counter()
    for name, qty in main_items or []:
        if name and qty:
            main_cnt[name] += int(qty)

    stub_cnt: Counter = Counter()
    unconfirmed = []
    for it in parse.items:
        if not it.name:
            continue
        if it.quantity is None:
            unconfirmed.append(it.name)
            stub_cnt[it.name] += 1
        else:
            stub_cnt[it.name] += int(it.quantity)

    missing = dict(main_cnt - stub_cnt)
    extra = dict(stub_cnt - main_cnt)
    items_ok = not missing and not extra and bool(stub_cnt or not main_cnt)

    return StubVerification(
        ticket_no_match=no_match,
        items_match=items_ok,
        missing_in_stub=missing,
        extra_in_stub=extra,
        qty_unconfirmed_in_stub=tuple(unconfirmed),
    )
