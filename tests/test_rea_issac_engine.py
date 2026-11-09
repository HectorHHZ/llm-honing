import copy
import unittest

from integrations.rea_issac.engine import SOURCE, VERSION, Action, Engine, replay
from integrations.rea_issac.policy import Baseline


class EngineContracts(unittest.TestCase):
    def test_seed_reproducibility_and_variation(self):
        self.assertEqual(Engine(7).digest(), Engine(7).digest())
        self.assertNotEqual(Engine(7).digest(), Engine(8).digest())

    def test_recovered_defaults_are_used(self):
        e = Engine(7)
        self.assertEqual(e.player["health"], SOURCE["player"]["health"])
        self.assertEqual(e.player["bombs"], SOURCE["player"]["bombs"])
        self.assertEqual(e.player["radius"], SOURCE["player"]["radius"])

    def test_invalid_inputs_do_not_advance_the_simulation(self):
        engine = Engine(7)
        before = engine.digest()
        for invalid in (
            {"move_x": True},
            {"move_x": 2},
            {"move_y": 0.5},
            {"shoot_x": float("nan")},
            {"bomb": 1},
            {"teleport": True},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                engine.step(invalid)
            self.assertEqual(engine.digest(), before)
        for invalid in (0, 301, -1, True, 1.5):
            with self.assertRaises(ValueError):
                engine.step({}, invalid)
        with self.assertRaises(ValueError):
            engine.step(Action(move_x=100))
        self.assertEqual(engine.digest(), before)

    def test_diagonal_input_is_normalized(self):
        straight, diagonal = Engine(7), Engine(7)
        x, y = straight.player["x"], straight.player["y"]
        straight.step({"move_x": 1}, 3)
        diagonal.step({"move_x": 1, "move_y": 1}, 3)
        from math import hypot

        self.assertAlmostEqual(
            hypot(straight.player["x"] - x, straight.player["y"] - y),
            hypot(diagonal.player["x"] - x, diagonal.player["y"] - y),
        )

    def test_walls_block_movement_away_from_doorways(self):
        e = Engine(7)
        e.player.update(x=65, y=70)
        e.step({"move_x": -1, "move_y": -1}, 120)
        self.assertEqual(e.current, "0,0")
        self.assertGreaterEqual(e.player["x"], 42)
        self.assertGreaterEqual(e.player["y"], 50)

    def test_locked_door_then_clear_and_transition(self):
        e = Engine(7)
        combat = next(room for room in e.rooms.values() if room["kind"] == "combat")
        e.current = combat["id"]
        combat["visited"] = True
        door = e.observe()["doors"][0]
        p = e.player
        p["x"], p["y"] = door["x"], door["y"]
        direction = {"north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0)}[door["direction"]]
        action = {"move_x": direction[0], "move_y": direction[1]}
        e.step(action, 1)
        self.assertEqual(e.current, combat["id"])
        for enemy in combat["enemies"]:
            enemy["health"] = 0
        e.step(action, 1)
        self.assertEqual(e.current, door["destination"])

    def test_shots_hit_and_kill(self):
        e = Engine(7)
        enemy = e._enemy("spitter", 400, 208)
        enemy["next_shot"] = 10000
        e.room["enemies"] = [enemy]
        e.room["cleared"] = False
        e.step({"shoot_x": 1}, 40)
        self.assertEqual(len(e.room["enemies"]), 0)
        self.assertEqual(e.metrics["kills"], 1)
        self.assertGreaterEqual(e.metrics["hits"], 3)
        self.assertTrue(e.room["cleared"])

    def test_contact_damage_has_an_invulnerability_window(self):
        e = Engine(7)
        enemy = e._enemy("chaser", 320, 208)
        enemy["speed"] = 0
        e.room["enemies"] = [enemy]
        e.room["cleared"] = False
        health = e.player["health"]
        e.step({}, 20)
        self.assertEqual(e.player["health"], health - 1)
        e.step({}, 20)
        self.assertEqual(e.player["health"], health - 2)

    def test_bomb_fuse_and_environment_damage(self):
        e = Engine(7)
        enemy = e._enemy("spitter", 390, 208)
        enemy["next_shot"] = 10000
        e.room["enemies"] = [enemy]
        e.room["cleared"] = False
        e.room["rocks"] = [[335, 200, 30, 30]]
        e.step({"move_x": -1, "bomb": True}, 40)
        e.step({"bomb": True}, 20)
        self.assertEqual(e.player["bombs"], 0)
        self.assertFalse(e.bombs)
        self.assertFalse(e.room["rocks"])
        self.assertEqual(e.metrics["kills"], 1)
        self.assertEqual(e.player["health"], SOURCE["player"]["health"])

    def test_public_observation_is_detached_and_hides_unvisited_room_contents(self):
        e = Engine(7)
        obs = e.observe()
        obs["player"]["health"] = 0
        self.assertGreater(e.player["health"], 0)
        self.assertNotIn("rng", obs)
        self.assertNotIn("rooms", obs)
        self.assertTrue(all("enemies" not in room and "pickups" not in room for room in obs["map"]))
        self.assertTrue(all(room["kind"] == "unknown" for room in obs["map"] if not room["visited"]))

    def test_replay_reconstructs_exact_state_and_rejects_tampering(self):
        e = Engine(29, 90)
        actions = [{"action": {"move_x": 1, "shoot_y": -1}, "frames": 3} for _ in range(30)]
        for entry in actions:
            e.step(entry["action"], entry["frames"])
        record = {"engine_version": VERSION, "seed": 29, "max_ticks": 90, "actions": actions, "final_hash": e.digest()}
        self.assertEqual(replay(record).digest(), e.digest())
        altered = copy.deepcopy(record)
        altered["actions"][0]["action"]["move_x"] = -1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            replay(altered)
        altered = copy.deepcopy(record)
        altered["engine_version"] = "other"
        with self.assertRaises(ValueError):
            replay(altered)

    def test_terminal_state_is_absorbing(self):
        e = Engine(7, 10)
        e.step({}, 30)
        self.assertEqual(e.status, "truncated")
        before = e.digest()
        e.step({"move_x": 1}, 30)
        self.assertEqual(e.digest(), before)

    def test_rule_baseline_can_finish_a_complete_run(self):
        e, baseline = Engine(11, 6000), Baseline()
        while e.status == "running":
            e.step(baseline.act(e.observe()), 3)
        self.assertEqual(e.status, "won")
        self.assertGreater(e.metrics["kills"], 0)


if __name__ == "__main__":
    unittest.main()
