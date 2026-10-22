"""Validated per-object text detector choices, shared by NUC and DLA."""
ALIASES = {"ticket": "order_ticket", "sticker": "order_sticker"}
BACKENDS = {"cv_table", "cv_ink", "det"}


def validate_routes(routes):
    result = {}
    for name, value in (routes or {}).items():
        if value not in BACKENDS:
            raise ValueError(f"ocr_routes.{name}: {value!r}; expected {sorted(BACKENDS)}")
        result[ALIASES.get(name, name)] = value
    return result


def backend_for(routes, object_type):
    return routes.get(ALIASES.get(object_type, object_type), "det")
