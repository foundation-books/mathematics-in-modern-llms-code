"""Behavioral checks for the relocated six-job resource example."""
import pytest
import inference_schedule as schedule


def test_schedule_witnesses_and_fixed_logical_draws():
    assert schedule.run_checks()['times'] == {1: 6, 2: 3, 3: 2}


def test_premature_check_cancellation_and_repeated_jobs_reject():
    for slots, workers in [
        ((('G1', 'C1'), ('G2', 'C2'), ('G3', 'C3')), 2),
        ((('G1',), ('C1',)), 1),
        ((('G1', 'G1'),), 2),
        (schedule.SCHEDULES[3], 2),
    ]:
        with pytest.raises(ValueError):
            schedule.validate_schedule(slots, workers)
