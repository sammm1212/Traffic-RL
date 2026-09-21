"""Protocol and analysis checks for the independent frozen-policy evaluation."""

import hashlib
import json
from pathlib import Path

import pytest

import evaluate_final as final
from src.experiments.generalisation_demand import SCENARIOS, demand_bytes, prepare_demand


def test_registered_protocol_and_checkpoint():
    value = final.protocol()
    assert value['primary_candidate']['minimum_green_seconds'] == 15
    assert value['supporting_exploratory_minimum_green_seconds'] == [10, 20, 30]
    assert value['seeds'] == list(range(4000, 4030))
    assert value['checkpoint_sha256'] == final.EXPECTED_HASH
    assert hashlib.sha256(final.CHECKPOINT.read_bytes()).hexdigest() == final.EXPECTED_HASH


def test_seeds_reserved_for_final_and_smoke():
    assert final.selected_seeds(4000, 30, False) == list(range(4000, 4030))
    assert final.selected_seeds(3000, 1, True) == [3000]
    with pytest.raises(ValueError):
        final.selected_seeds(4000, 1, True)
    with pytest.raises(ValueError):
        final.selected_seeds(4030, 1, False)


def test_deterministic_demand_is_preserved(tmp_path):
    route = tmp_path / 'route.rou.xml'
    prepare_demand(route, SCENARIOS['changing'], 3000, 300)
    original = route.read_bytes()
    assert original == demand_bytes(SCENARIOS['changing'], 3000, 300)
    prepare_demand(route, SCENARIOS['changing'], 3000, 300)
    assert route.read_bytes() == original
    with pytest.raises(ValueError):
        prepare_demand(route, SCENARIOS['changing'], 3001, 300)


def row(seed, controller, value, route_hash='same'):
    return {'scenario': 'balanced', 'seed': seed, 'controller': controller,
            'demand_sha256': route_hash, 'vehicles_completed': value,
            'mean_waiting_time': 4.0, 'mean_queue_length': 3.0,
            'executed_direction_changes': 5, 'transition_seconds': 20,
            'ns_green_seconds': 140, 'ew_green_seconds': 140}


def test_primary_pairing_and_statistics():
    rows = []
    for seed, delta in zip((4000,4001,4002),(1,2,3)):
        rows.extend((row(seed,'fixed_time',10),row(seed,'dqn_15',10+delta)))
    pairs, summary = final.paired(rows,'fixed_time',('dqn_15',),final.PRIMARY,'prespecified_primary')
    completed = next(x for x in summary if x['metric']=='vehicles_completed')
    assert [r['paired_difference'] for r in pairs if r['metric']=='vehicles_completed'] == [1,2,3]
    assert completed['n'] == 3 and completed['mean'] == 2 and completed['sample_sd'] == 1
    assert completed['ci_95_lower'] < 0 < completed['ci_95_upper']
    assert all(x['analysis']=='prespecified_primary' for x in pairs+summary)
    rows[-1]['demand_sha256']='changed'
    with pytest.raises(ValueError,match='route mismatch'):
        final.paired(rows,'fixed_time',('dqn_15',),final.PRIMARY,'prespecified_primary')


def test_supporting_labels_and_direct_primary_comparison():
    rows=[row(4000,'fixed_time',10),row(4000,'dqn_15',12),row(4000,'dqn_20',11)]
    pairs, summary=final.paired(rows,'dqn_15',('dqn_20',),final.SUPPORTING,'supporting_vs_primary')
    completed=next(x for x in pairs if x['metric']=='vehicles_completed')
    assert completed['paired_difference']==-1
    assert all(x['analysis']=='supporting_vs_primary' for x in pairs+summary)


def test_audit_rejects_duplicate_and_incomplete_groups(tmp_path):
    route=tmp_path/'route.rou.xml';route.write_bytes(b'example')
    base={**row(4000,'fixed_time',10,final.digest(route)),
          'demand_file':str(route),'scheduled_vehicles':20,'inserted_vehicles':15,
          'vehicles_remaining':10,'episode_seconds':300,'ns_green_seconds':130,
          'ew_green_seconds':130,'yellow_seconds':30,'all_red_seconds':10}
    with pytest.raises(ValueError,match='duplicate'):
        final.audit([base,base],['balanced'],[4000],False)
    with pytest.raises(ValueError,match='coverage incomplete'):
        final.audit([base],['balanced'],[4000],True)
    assert final.audit([base],['balanced'],[4000],False)['status']=='partial'


def test_incomplete_report_has_no_primary_claim(tmp_path):
    final.save_outputs(tmp_path,[],['balanced'],[4000],False)
    assert 'Incomplete evaluation' in (tmp_path/'report.md').read_text()
    assert not (tmp_path/'primary_summary.csv').exists()
    assert json.loads((tmp_path/'validation_report.json').read_text())['status']=='partial'


def test_report_only_preserves_raw_episode_file(tmp_path):
    route=tmp_path/'route.rou.xml';route.write_bytes(b'example')
    episode=tmp_path/'episodes.csv';episode.write_bytes(b'original raw results\n')
    item={**row(4000,'fixed_time',10,final.digest(route)),
          'demand_file':str(route),'scheduled_vehicles':20,'inserted_vehicles':15,
          'vehicles_remaining':10,'episode_seconds':300,'ns_green_seconds':130,
          'ew_green_seconds':130,'yellow_seconds':30,'all_red_seconds':10}
    final.save_outputs(tmp_path,[item],['balanced'],[4000],False,report_only=True)
    assert episode.read_bytes()==b'original raw results\n'
