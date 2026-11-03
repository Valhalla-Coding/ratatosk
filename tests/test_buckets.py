from ratatosk.buckets import BUCKETS, needed_ctx, pick_bucket


def test_buckets_are_the_spec_sizes():
    assert BUCKETS == (8192, 16384, 32768, 65536, 131072)


def test_needed_ctx_adds_a_quarter_headroom():
    assert needed_ctx(8000, 1024) == 11280


def test_nothing_loaded_picks_smallest_fitting():
    assert pick_bucket(11280, None, [], 0) == 16384


def test_moves_up_when_it_does_not_fit():
    assert pick_bucket(20000, 16384, [], 0) == 32768


def test_stays_when_it_fits_and_recent_needs_do_not_allow_a_shrink():
    assert pick_bucket(11280, 32768, [(990, 11280)], 1000) == 32768


def test_moves_down_two_buckets_when_everything_recent_fits():
    assert pick_bucket(5000, 32768, [(900, 7000), (950, 3000)], 1000) == 8192


def test_one_big_recent_need_blocks_the_shrink():
    assert pick_bucket(5000, 32768, [(950, 20000)], 1000) == 32768


def test_needs_older_than_ten_minutes_are_ignored():
    assert pick_bucket(5000, 32768, [(0, 20000)], 1000) == 8192


def test_too_big_returns_none():
    assert pick_bucket(200000, None, [], 0) is None
    assert pick_bucket(40000, 16384, [], 0, limit=32768) is None
