"""Unit check of the rewards added on 24 Oct 2026 (lit_t2t, ctl_correct_weight). Run from v3/."""
import common as C
import literature_rewards as L

L.register_all(C)
S, E = C.solution_start, C.solution_end


def mk(ans, n):
    return [{"content": "x" * n + S + ans + E}]


prompts = [[{"content": "q"}]] * 4
comp = [mk("5", 100), mk("5", 3500), mk("6", 100), mk("6", 7000)]
gt = ["5"] * 4
# one group, pass rate 0.5: correct short ~1.0, correct half-length 0.95, wrong short ~0.0, wrong full length 0.1
print("lit_t2t", [round(x, 3) for x in C.REWARD_REGISTRY["lit_t2t"](prompts, comp, gt)])
print("ctl_correct_weight", C.REWARD_REGISTRY["ctl_correct_weight"](prompts, comp, gt))
print("cosine", [round(x, 3) for x in C.REWARD_REGISTRY["lit_cosine_length"](prompts, comp, gt)])
