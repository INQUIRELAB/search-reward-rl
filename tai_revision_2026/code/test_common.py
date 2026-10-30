import sys; sys.path.insert(0, "v3"); import common as C
cases = [(r"\frac{1}{2}", "0.5"), ("0.5", r"\frac{1}{2}"), (r"\boxed{12}", "12"), ("x^2+1", "1+x^{2}"), ("(1,2)", "(1, 2)"), ("3,000", "3000"), ("12.0", "12"), ("13", "12"), (r"\sqrt{2}", "1.414")]
for p, g in cases:
    print(f"{p!r:20} vs {g!r:16} numeric={C.numeric_match(p,g)!s:5} math={C.math_match(p,g)}")
print("letter", C.letter_match("B.", "B"), C.letter_match("Option C", "C"), C.letter_match("A", "B"))
resp = "<thinking>\n1+1=2\n2*3=6\n</thinking>\n<solution>\boxed{6}</solution>"
print("extract", C.extract_solution(resp), C.extract_solution_flexible("The answer is \boxed{7}.", C.TASKS["math"]))
print("check_answer", C.check_answer([[]], [[{"role":"assistant","content":resp}]], ["6"]))
