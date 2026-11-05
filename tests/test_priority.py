from ratatosk.gpuqueue import P_TUNE
from ratatosk.priority import Rank, rank_for
from ratatosk.settings import Route

PLAIN, RP, COMFY = Route(11434, "llm", "plain"), Route(11435, "llm", "rp"), Route(8188, "comfy")


def test_first_matching_line_wins():
    rules = [{"client": "192.168.1.5"}, {"door": 11434}]
    assert rank_for(rules, PLAIN, "192.168.1.5", "x") == Rank(0, False, "192.168.1.5")
    assert rank_for(rules, PLAIN, "10.0.0.9", "x") == Rank(1, False, "door 11434")


def test_agent_matches_by_substring_ignoring_case_and_address_alone_matches_any_agent():
    rules = [{"client": "192.168.1.5", "agent": "node-fetch", "label": "SillyTavern"}, {"client": "192.168.1.5"}]
    assert rank_for(rules, PLAIN, "192.168.1.5", "Mozilla NODE-FETCH/3").label == "SillyTavern"
    assert rank_for(rules, PLAIN, "192.168.1.5", "curl").priority == 1


def test_client_ranges():
    rules = [{"client": "192.168.1.0/24"}]
    assert rank_for(rules, PLAIN, "192.168.1.77", "").priority == 0
    assert rank_for(rules, PLAIN, "192.168.2.77", "").priority == 1  # falls to everything else


def test_door_line_matches_by_port_including_comfy():
    rules = [{"door": 8188}, {"door": 11435}]
    assert rank_for(rules, COMFY, "x", "").priority == 0 and rank_for(rules, RP, "x", "").priority == 1


def test_everything_else_comes_after_all_lines_and_tuning_after_that():
    rules = [{"door": 1}, {"door": 2}]
    rank = rank_for(rules, PLAIN, "10.0.0.1", "")
    assert rank == Rank(2, False, "Everything else")
    assert P_TUNE > rank.priority


def test_drop_lines_outrank_every_normal_line_and_preempt():
    rules = [{"door": 11435}, {"client": "192.168.1.5", "drop": True, "label": "me"}]
    drop = rank_for(rules, PLAIN, "192.168.1.5", "")
    assert drop.preempt and drop.label == "me"
    assert drop.priority < rank_for(rules, RP, "x", "").priority


def test_a_bad_address_from_the_server_never_matches_or_crashes():
    assert rank_for([{"client": "192.168.1.5"}], PLAIN, None, "").priority == 1
    assert rank_for([{"client": "192.168.1.5"}], PLAIN, "garbage", "").priority == 1
