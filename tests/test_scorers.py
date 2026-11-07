import json
from pathlib import Path

import pytest

from llm_eval.client import LLMClient
from llm_eval.datasets import load_dataset
from llm_eval.models import Example
from llm_eval.providers.mock import MockProvider
from llm_eval.scorers import ScoringContext, available_scorers, get_scorer
from llm_eval.scorers.extract import extract_answer, extract_code, extract_json
from llm_eval.scorers.llm_judge import parse_verdict

ROOT = Path(__file__).resolve().parents[1]
CTX = ScoringContext()


def test_registry():
    assert set(available_scorers()) == {"exact", "regex", "json_schema", "unit_test", "llm_judge"}
    with pytest.raises(ValueError):
        get_scorer("nope")


# ------------------------------------------------------------------ extraction


def test_extractors():
    assert extract_answer("thinking...\nAnswer: 42\n") == "42"
    assert extract_answer("**Answer:** knight") == "knight"
    assert extract_answer("just 7") == "just 7"
    text = "x\n```sql\nSELECT 1;\n```\n```python\ndef f(): pass\n```"
    assert extract_code(text, "python") == "def f(): pass"
    assert extract_code(text, "sql") == "SELECT 1;"
    assert extract_code("no fences") == "no fences"
    assert extract_json('Sure! ```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here: {"b": [1, 2]} done') == {"b": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("nothing here")


# ------------------------------------------------------------------ deterministic scorers


async def test_exact():
    s = get_scorer("exact")
    ex = Example(id="e", prompt="p", expected="Friday")
    assert (await s.score(ex, "Let me think.\nAnswer: friday.", CTX)).score == 1.0
    assert (await s.score(ex, "Answer: Monday", CTX)).score == 0.0
    multi = Example(id="e", prompt="p", expected=["1", "one"])
    assert (await s.score(multi, "Answer: One", CTX)).passed
    assert not s.applicable(Example(id="e", prompt="p"))


async def test_regex():
    s = get_scorer("regex")
    ex = Example(
        id="e",
        prompt="p",
        scorer_args={"regex": {"pattern": r"answer:\s*5\b", "ignore_case": True}},
    )
    assert (await s.score(ex, "ANSWER: 5", CTX)).passed
    assert not (await s.score(ex, "Answer: 50", CTX)).passed
    neg = Example(id="e", prompt="p", scorer_args={"regex": {"pattern": "TODO", "negate": True}})
    assert (await s.score(neg, "done", CTX)).passed
    bad = Example(id="e", prompt="p", scorer_args={"regex": {"pattern": "("}})
    assert (await s.score(bad, "x", CTX)).error
    assert not s.applicable(Example(id="e", prompt="p"))


async def test_json_schema():
    s = get_scorer("json_schema")
    [ex] = load_dataset(ROOT / "datasets" / "coding.jsonl", tags=["json"])
    good = '```json\n{"name": "Ann", "age": 30, "emails": ["a@x.io"]}\n```'
    assert (await s.score(ex, good, CTX)).passed
    r = await s.score(ex, '{"name": "Ann", "age": -1, "emails": []}', CTX)
    assert not r.passed and len(r.details["errors"]) == 2
    assert (await s.score(ex, "no json", CTX)).details["reason"] == "no parseable JSON"


# ------------------------------------------------------------------ unit tests (subprocess)

REFERENCE_SOLUTIONS = {
    "py-fizzbuzz": """
def fizzbuzz(n):
    return "FizzBuzz" if n % 15 == 0 else "Fizz" if n % 3 == 0 else "Buzz" if n % 5 == 0 else str(n)
""",
    "py-palindrome": """
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]
""",
    "py-two-sum": """
def two_sum(nums, target):
    seen = {}
    for j, x in enumerate(nums):
        if target - x in seen:
            return (seen[target - x], j)
        seen[x] = j
""",
    "py-flatten": """
def flatten(xs):
    out = []
    for x in xs:
        out.extend(flatten(x) if isinstance(x, list) else [x])
    return out
""",
    "py-roman": """
def to_roman(n):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),
            (10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = ''
    for v, s in vals:
        while n >= v:
            out += s; n -= v
    return out
""",
    "py-lru": """
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cap, self.d = capacity, OrderedDict()
    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)
        return self.d[key]
    def put(self, key, value):
        self.d[key] = value
        self.d.move_to_end(key)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
""",
    "sql-top-salary": """SELECT dept, name FROM employees e
WHERE salary = (SELECT MAX(salary) FROM employees WHERE dept = e.dept) ORDER BY dept;""",
    "sql-avg-dept": """SELECT dept, AVG(salary) FROM employees GROUP BY dept
HAVING AVG(salary) > 90;""",
    "sql-no-orders": """SELECT c.name FROM customers c LEFT JOIN orders o ON o.customer_id = c.id
WHERE o.id IS NULL;""",
}


@pytest.mark.parametrize("example_id", sorted(REFERENCE_SOLUTIONS))
async def test_reference_solutions_pass_dataset_tests(example_id):
    examples = {e.id: e for e in load_dataset(ROOT / "datasets" / "coding.jsonl")}
    ex = examples[example_id]
    lang = ex.scorer_args["unit_test"]["language"]
    output = f"Here you go:\n```{lang}\n{REFERENCE_SOLUTIONS[example_id].strip()}\n```"
    result = await get_scorer("unit_test").score(ex, output, CTX)
    assert result.passed, result.details


async def test_unit_test_partial_failure_and_timeout():
    s = get_scorer("unit_test")
    ex = Example(
        id="e",
        prompt="p",
        scorer_args={"unit_test": {"tests": ["assert f() == 1", "assert f() == 2"], "timeout": 2}},
    )
    r = await s.score(ex, "def f(): return 1", CTX)
    assert r.score == 0.5 and not r.passed and r.details["failures"][0]["test"] == "assert f() == 2"
    r = await s.score(ex, "def f(:", CTX)
    assert r.score == 0.0 and "failed to load" in r.details["reason"]
    r = await s.score(ex, "while True: pass", CTX)
    assert r.score == 0.0 and "timeout" in r.details["reason"]


async def test_unit_test_sql_wrong_and_invalid():
    s = get_scorer("unit_test")
    ex = {e.id: e for e in load_dataset(ROOT / "datasets" / "coding.jsonl")}["sql-no-orders"]
    assert not (await s.score(ex, "SELECT name FROM customers;", CTX)).passed
    r = await s.score(ex, "SELEC nonsense", CTX)
    assert not r.passed and "OperationalError" in r.details["reason"]


# ------------------------------------------------------------------ LLM judge


def test_parse_verdict():
    v = parse_verdict('{"correctness": 5, "clarity": 4, "safety": 5, "rationale": "ok"}')
    assert v["clarity"] == 4
    for bad in ['{"correctness": 9, "clarity": 4, "safety": 5}', '{"clarity": 4}', "[1]"]:
        with pytest.raises(ValueError):
            parse_verdict(bad)


async def test_llm_judge_with_mock(settings, pricing):
    def judge(request):
        assert "<reference>\n42\n</reference>" in request.prompt
        good = "<response>\n42\n</response>" in request.prompt
        score = 5 if good else 1
        return json.dumps({"correctness": score, "clarity": score, "safety": 5, "rationale": "r"})

    mock = MockProvider(settings, pricing, responses=judge)
    client = LLMClient(settings, pricing=pricing, use_cache=False, providers={"mock": mock})
    ctx = ScoringContext(client=client, settings=settings)
    s = get_scorer("llm_judge")
    ex = Example(id="e", prompt="6*7?", expected="42")

    good = await s.score(ex, "42", ctx)
    assert good.score == 1.0 and good.passed and good.details["judge"] == "mock:judge"
    bad = await s.score(ex, "41", ctx)
    assert bad.score == pytest.approx((7 / 3 - 1) / 4) and not bad.passed


async def test_llm_judge_unparseable_is_error_not_zero(settings, pricing):
    mock = MockProvider(settings, pricing, responses=lambda r: "I think it's great!")
    client = LLMClient(settings, pricing=pricing, use_cache=False, providers={"mock": mock})
    r = await get_scorer("llm_judge").score(
        Example(id="e", prompt="p"), "x", ScoringContext(client=client)
    )
    assert r.error and "judge failed" in r.error
