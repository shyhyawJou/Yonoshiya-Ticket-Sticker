from pathlib import Path
import sys
import shutil
import pytest
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from profile_config import load_profile_config, select_profile, merge_settings
from config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_priority_and_isolation(tmp_path):
    shutil.copytree(ROOT/'tasks/ocr/yoshinoya', tmp_path/'tasks/ocr/yoshinoya')
    shutil.copytree(ROOT/'tasks/ocr/suzumo', tmp_path/'tasks/ocr/suzumo')
    (tmp_path/'runtime.yaml').write_text('profile: yoshinoya')
    assert select_profile(tmp_path)[0] == 'yoshinoya'
    (tmp_path/'runtime.local.yaml').write_text('profile: suzumo')
    assert select_profile(tmp_path)[0] == 'suzumo'
    assert select_profile(tmp_path, 'yoshinoya')[0] == 'yoshinoya'
    with pytest.raises(ValueError):
        select_profile(tmp_path, '../ocr')


def test_runtime_local_rejects_operational_settings(tmp_path):
    shutil.copytree(ROOT/'tasks/ocr/yoshinoya', tmp_path/'tasks/ocr/yoshinoya')
    (tmp_path/'runtime.yaml').write_text('profile: yoshinoya')
    (tmp_path/'runtime.local.yaml').write_text(
        'profile: yoshinoya\nocr_routes:\n  order_ticket: cv_table\n'
    )
    with pytest.raises(ValueError, match='只能包含 profile'):
        select_profile(tmp_path)


def test_profile_pairings():
    y = load_config(ROOT/'tasks/ocr/yoshinoya/config.yaml')
    s = load_config(ROOT/'tasks/ocr/suzumo/config.yaml')
    assert y.ticket_parser == 'legacy' and y.ocr_routes['order_ticket'] == 'det'
    assert s.ticket_parser == 'table' and s.ocr_routes['order_ticket'] == 'cv_table'
    assert s.object_class_items['chopsticks'] == '箸'
    assert 'chopsticks' not in y.object_class_items
    assert 'サラダ・味噌汁セ' in s.menus_ticket
    assert not y.vocab_ticket
    assert Path(y.runtime.model.object_det).name == 'mmr_11cls_v01_01.dla'


def test_override_preserves_other_route():
    cfg = load_config(ROOT/'tasks/ocr/suzumo/config.yaml',
                      {'ocr_routes':{'order_ticket':'det'}})
    assert cfg.ocr_routes == {'order_ticket':'det','order_sticker':'det'}


def test_merge_does_not_mutate():
    base = {'camera':{'gain':1,'exposure':2}}
    result = merge_settings(base, {'camera':{'gain':3}})
    assert result['camera'] == {'gain':3,'exposure':2}
    assert base['camera']['gain'] == 1


def test_profile_loader_without_legacy_files(tmp_path):
    for brand in ('yoshinoya', 'suzumo'):
        shutil.copytree(ROOT/'tasks/ocr'/brand, tmp_path/'tasks/ocr'/brand)
    (tmp_path/'runtime.yaml').write_text('profile: yoshinoya')
    cfg = load_profile_config(tmp_path)
    assert cfg.profile_name == 'yoshinoya'
    assert cfg.ocr_routes['order_ticket'] == 'det'
    assert Path(cfg.source_config_path).parent.name == 'yoshinoya'


def test_legacy_camera_override_is_migrated_and_local_is_profile_only(tmp_path):
    """舊版 local override 應一次性遷移，之後只留下品牌選擇。

    舊版設計會把 UI／MQTT 調整後的相機參數寫到
    ``runtime.local.yaml -> profiles.<brand>.camera_params``，並在啟動時
    覆寫品牌 ``config.yaml``。這會形成兩份設定來源，造成同仁直接修改
    品牌 config 後，相機仍使用 local 舊值。

    新版的遷移政策是：

    1. 將舊 local 內的 camera_params 合併回對應品牌 config。這裡保留
       12345.0，是因為它在升級前就是實際生效值；升級不應突然改變現場
       相機曝光。
    2. 不遷移 ocr_routes 等其他 override。所有非品牌選擇設定都必須回到
       品牌 config 維護，所以測試確認 Yoshinoya 仍使用 config 原有的 det。
    3. 將 runtime.local.yaml 正規化成只剩 profile，避免未來再次出現隱藏
       覆寫來源。
    4. 保留一份帶時間戳的備份，讓被停用的舊 override 仍可供人工追查。

    測試使用 tmp_path 的完整品牌設定副本，確保遷移只操作隔離的測試資料，
    不會修改專案內真正的 Yoshinoya／Suzumo config。
    """
    for brand in ('yoshinoya', 'suzumo'):
        shutil.copytree(ROOT/'tasks/ocr'/brand, tmp_path/'tasks/ocr'/brand)
    (tmp_path/'runtime.yaml').write_text('profile: yoshinoya')
    (tmp_path/'runtime.local.yaml').write_text(
        'profile: yoshinoya\n'
        'profiles:\n'
        '  yoshinoya:\n'
        '    camera_params:\n'
        '      hik:\n'
        '        ExposureTime:\n'
        '          value: 12345.0\n'
        '    ocr_routes:\n'
        '      order_ticket: cv_table\n'
    )

    cfg = load_profile_config(tmp_path)

    assert cfg.camera_params['hik']['ExposureTime']['value'] == 12345.0
    assert cfg.ocr_routes['order_ticket'] == 'det'
    assert yaml.safe_load((tmp_path/'runtime.local.yaml').read_text()) == {
        'profile': 'yoshinoya'
    }
    assert list(tmp_path.glob('runtime.local.yaml.bak-*'))


@pytest.mark.parametrize('brand', ['yoshinoya', 'suzumo'])
def test_diagnostic_models_follow_brand(monkeypatch, brand):
    from tools.ocr_backend import OcrBackend
    calls = []
    monkeypatch.setattr(OcrBackend, '_engine_class', lambda *args: lambda *paths: calls.append(paths))
    backend = OcrBackend(prefer='dla', profile=brand)
    assert calls[0][-1] == backend.cfg.runtime.model.text
    assert Path(calls[0][-1]).parent.name == brand
    assert all(Path(p).exists() for p in calls[0])
    backend = OcrBackend(prefer='onnx', profile=brand)
    assert all(Path(p).parent.name == brand for p in calls[1])


def test_probe_uses_brand_parser():
    from profile_config import load_profile_config
    from tools.ocr_probe import make_ticket_parser
    for brand, detailed in [('yoshinoya', False), ('suzumo', True)]:
        cfg = load_profile_config(ROOT, brand)
        parser = make_ticket_parser(cfg.vocab_ticket or cfg.menus_ticket, cfg)
        assert hasattr(parser, 'parse_detailed') == detailed
