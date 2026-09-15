"""Build the frozen BF-Ladder test: 10 skill levels x 10 problems, 5 test cases each.

Run once locally: writes bf_ladder_v1.json. Never edit or regenerate it after models
have been scored on it, and never train on it.

Every problem carries a reference Brainfuck program that is verified on its 5 test
cases plus 200 extra random inputs, so every problem is known to be solvable.

Deliberately different from training:
  - wording ("Write a Brainfuck program that does the following.")
  - parameter values (shifts 5/6/7, separators + = and ., letters k m p u w b d g h v)
  - task mix (if/else, count/filter, multi-digit numbers, comparisons were never trained)
"""
import json
import random
import string

from bf_gym.config import BFConfig
from bf_gym.interpreter import run_bf

STRICT = BFConfig(wrap_pointer=False, max_steps=2_000_000)
TESTS_PER_PROBLEM = 5
EXTRA_CHECKS = 200


class Asm:
    """Tracks the pointer statically; every loop body must return to its start cell."""

    def __init__(self):
        self.pos, self.code = 0, []

    def at(self, cell, s=""):
        d = cell - self.pos
        self.code.append(">" * d if d >= 0 else "<" * -d)
        self.pos = cell
        self.code.append(s)
        return self

    def out(self):
        return "".join(self.code)


def print_text(a, cell, text):
    """Print text using a scratch cell that starts at 0; leaves it at 0."""
    cur = 0
    for ch in text:
        d = ord(ch) - cur
        a.at(cell, ("+" * d if d >= 0 else "-" * -d) + ".")
        cur = ord(ch)
    a.at(cell, "[-]")


def copy(a, src, dst, tmp):
    a.at(src, "[-"); a.at(dst, "+"); a.at(tmp, "+"); a.at(src, "]")
    a.at(tmp, "[-"); a.at(src, "+"); a.at(tmp, "]")


DIVMOD = "[->-[>+>>]>[+[-<+>]>+>>]<<<<<]"   # n d -> 0 d-n%d n%d n/d, pointer back on n


def print_num(a, p):
    """Print the value in cell p (0-99) in decimal. Uses cells p..p+3, clears them."""
    a.at(p + 1, "+" * 10)
    a.at(p, DIVMOD)
    a.at(p + 3, "["); a.at(p + 3, "+" * 48 + "."); a.at(p + 3, "[-]"); a.at(p + 3, "]")
    a.at(p + 2, "+" * 48 + "."); a.at(p + 2, "[-]")
    a.at(p + 1, "[-]")


def const_code(text):
    a = Asm()
    print_text(a, 0, text)
    return a.out()


# ---------------------------------------------------------------------------
# input samplers
# ---------------------------------------------------------------------------

def letters(lo, hi, alphabet=string.ascii_lowercase, empty_ok=False):
    def f(r):
        if empty_ok and r.random() < 0.15:
            return ""
        return "".join(r.choice(alphabet) for _ in range(r.randint(lo, hi)))
    return f


def one_of(alphabet):
    return lambda r: r.choice(alphabet)


def digit_pair(cond):
    pairs = [(x, y) for x in range(10) for y in range(10) if cond(x, y)]
    return lambda r: "{} {}".format(*r.choice(pairs))


def with_letter(letter, alphabet):
    """Lines that contain the letter about half the time."""
    def f(r):
        n = r.randint(0, 7)
        s = [r.choice(alphabet) for _ in range(n)]
        if n and r.random() < 0.6:
            for _ in range(r.randint(1, min(3, n))):
                s[r.randrange(n)] = letter
        return "".join(s)
    return f


def char_or(letter, others):
    return lambda r: letter if r.random() < 0.45 else r.choice(others)


# ---------------------------------------------------------------------------
# levels: each problem = (description, sampler, expected_fn, reference_code)
# ---------------------------------------------------------------------------

def level1():
    chars = "Q#w7~Vj%Kz"
    return [(f"Print the single character {c!r} and nothing else.", None, lambda s, c=c: c,
             const_code(c)) for c in chars]


def level2():
    phrases = ["moon lamp", "Go!", "rusty key", "7 up", "Blue Fox", "ping?", "ok ok", "Zed 42",
               "wave", "night owl"]
    return [(f"Print exactly {p!r} with no trailing newline.", None, lambda s, p=p: p,
             const_code(p)) for p in phrases]


def level3():
    one = one_of("ghijklmnopqrs")
    upper_in = one_of("GHIJKLMNOPQRS")
    return [
        ("Read one lowercase letter and print the letter 6 places later in the alphabet.",
         one, lambda s: chr(ord(s) + 6), "," + "+" * 6 + "."),
        ("Read one lowercase letter and print the letter 7 places later in the alphabet.",
         one, lambda s: chr(ord(s) + 7), "," + "+" * 7 + "."),
        ("Read one lowercase letter and print the letter 6 places earlier in the alphabet.",
         one, lambda s: chr(ord(s) - 6), "," + "-" * 6 + "."),
        ("Read one character and print it three times.", one, lambda s: s * 3, ",..."),
        ("Read one character and print it followed by an exclamation mark.", one,
         lambda s: s + "!", ",." + const_code_rel(0, "!", start=None)),
        ("Read one lowercase letter and print it in uppercase followed by a newline.", one,
         lambda s: s.upper() + "\n", "," + "-" * 32 + ">" + "+" * 10 + "<.>."),
        ("Read one character and print it twice with a space between.", one,
         lambda s: s + " " + s, ",.>" + "+" * 32 + ".<."),
        ("Read one lowercase letter and print the letter just before it in the alphabet.", one,
         lambda s: chr(ord(s) - 1), ",-."),
        ("Read one uppercase letter and print it in lowercase.", upper_in,
         lambda s: s.lower(), "," + "+" * 32 + "."),
        ("Read one lowercase letter and print the letter 2 places later, then the letter itself.",
         one, lambda s: chr(ord(s) + 2) + s, ",++.--."),
    ]


def const_code_rel(cell, text, start=None):
    """Code to print text from a fresh cell to the right of the current one, then come back."""
    a = Asm()
    print_text(a, 1, text)
    a.at(0)
    return a.out()


def level4():
    line = letters(0, 6, empty_ok=True)
    up = letters(0, 6, string.ascii_uppercase, empty_ok=True)
    safe = letters(0, 6, "fghijklmnopq", empty_ok=True)

    def sep_after(sep, transform="", tail=""):
        a = Asm()
        a.at(1, "+" * ord(sep))
        a.at(0, ","); a.at(0, "["); a.at(0, transform + "."); a.at(1, "."); a.at(0, ","); a.at(0, "]")
        if tail:
            print_text(a, 2, tail)
        return a.out()

    return [
        ("Read the whole input and print every character three times.", line,
         lambda s: "".join(c * 3 for c in s), ",[...,]"),
        ("Read the whole input and print every character with its byte value increased by 5.",
         safe, lambda s: "".join(chr(ord(c) + 5) for c in s), ",[+++++.,]"),
        ("Read the whole input and print every character with its byte value decreased by 4.",
         safe, lambda s: "".join(chr(ord(c) - 4) for c in s), ",[----.,]"),
        ("Read the whole input and print each character followed by '+'.", line,
         lambda s: "".join(c + "+" for c in s), sep_after("+")),
        ("Read the whole input and print each character followed by '='.", line,
         lambda s: "".join(c + "=" for c in s), sep_after("=")),
        ("Read the whole input (lowercase letters) and print each character in uppercase "
         "followed by '.'.", line, lambda s: "".join(c.upper() + "." for c in s),
         sep_after(".", "-" * 32)),
        ("Read the whole input, print every character twice, then print a newline.", line,
         lambda s: "".join(c * 2 for c in s) + "\n", ",[..,]" + "+" * 10 + "."),
        ("Read the whole input, print it unchanged, then print '#'.", line,
         lambda s: s + "#", ",[.,]" + "+" * 35 + "."),
        ("Read the whole input (uppercase letters) and print each character in lowercase "
         "followed by a space.", up, lambda s: "".join(c.lower() + " " for c in s),
         sep_after(" ", "+" * 32)),
        ("Read the whole input and print every character with its byte value increased by 1, "
         "twice.", safe, lambda s: "".join(chr(ord(c) + 1) * 2 for c in s), ",[+..,]"),
    ]


def level5():
    line = letters(1, 6)
    return [
        ("Read the whole input and print it followed by its reverse.", line,
         lambda s: s + s[::-1], ">,[.>,]<[.<]"),
        ("Read the whole input and print it reversed, then '!'.", line,
         lambda s: s[::-1] + "!", ">,[>,]<[.<]" + "+" * 33 + "."),
        ("Read the whole input and print it reversed with every byte value increased by 2.",
         line, lambda s: "".join(chr(ord(c) + 2) for c in s[::-1]), ">,[++>,]<[.<]"),
        ("Read the whole input, print it unchanged, then print its length as one digit.", line,
         lambda s: s + str(len(s)), ",[.>+<,]>" + "+" * 48 + "."),
        ("Read the whole input and print it reversed with every character doubled.", line,
         lambda s: "".join(c * 2 for c in s[::-1]), ">,[>,]<[..<]"),
        ("Read the whole input and print its last character followed by its first character.",
         line, lambda s: s[-1] + s[0], ">,[>,]<.[<]>."),
        ("Read the whole input (lowercase letters) and print it reversed in uppercase.", line,
         lambda s: s[::-1].upper(), ">,[" + "-" * 32 + ">,]<[.<]"),
        ("Read the whole input and print one '*' per character, then a newline.", line,
         lambda s: "*" * len(s) + "\n", ",[>+<,]>>" + "+" * 42 + "<[>.<-]>>" + "+" * 10 + "."),
        ("Read the whole input and print its first character, then the whole input reversed.",
         line, lambda s: s[0] + s[::-1], ">,.[>,]<[.<]"),
        ("Read the whole input, print it reversed, then a newline, then the input unchanged.",
         line, lambda s: s[::-1] + "\n" + s, ">,[>,]<[.<]" + "+" * 10 + ".>[.>]"),
    ]


def level6():
    d = one_of(string.digits)
    return [
        ("Read two digits separated by a space and print the first minus the second. The first "
         "is never smaller.", digit_pair(lambda x, y: x >= y),
         lambda s: str(int(s[0]) - int(s[2])), ",>,>,[-<<->>]<<" + "+" * 48 + "."),
        ("Read one digit and print that number plus 5. The result is at most 9.",
         one_of("01234"), lambda s: str(int(s) + 5), "," + "+" * 5 + "."),
        ("Read one digit and print three times its value. The result is at most 9.",
         one_of("0123"), lambda s: str(3 * int(s)),
         "," + "-" * 48 + "[>+++<-]>" + "+" * 48 + "."),
        ("Read three digits separated by spaces and print their sum. The sum is at most 9.",
         lambda r: (lambda t: f"{t[0]} {t[1]} {t[2]}")(
             r.choice([(x, y, z) for x in range(10) for y in range(10) for z in range(10)
                       if x + y + z <= 9])),
         lambda s: str(int(s[0]) + int(s[2]) + int(s[4])),
         ",>,>,>,>,[-<<<<+>>>>]<<[-<<+>>]<<" + "-" * 96 + "."),
        ("Read one digit and print that number minus 4. The digit is at least 4.",
         one_of("456789"), lambda s: str(int(s) - 4), "," + "-" * 4 + "."),
        ("Read one digit n and print 9 minus n.", d, lambda s: str(9 - int(s)),
         "," + "-" * 48 + ">" + "+" * 57 + "<[->-<]>."),
        ("Read two digits separated by a space and print their sum. The sum is at most 9.",
         digit_pair(lambda x, y: x + y <= 9), lambda s: str(int(s[0]) + int(s[2])),
         ",>,>,[-<<+>>]<<" + "-" * 48 + "."),
        ("Read one digit n (at most 8) and print n followed by n plus 1.", one_of("012345678"),
         lambda s: s + str(int(s) + 1), ",.+."),
        ("Read one digit n (at most 8) and print n plus 1 twice.", one_of("012345678"),
         lambda s: str(int(s) + 1) * 2, ",+.."),
        ("Read two digits separated by a space and print the first minus the second minus 1. "
         "The first is always larger.", digit_pair(lambda x, y: x > y),
         lambda s: str(int(s[0]) - int(s[2]) - 1), ",>,>,[-<<->>]<<" + "+" * 47 + "."),
    ]


def level7():
    """if / else on one character."""
    def branch(letter, then_text=None, else_text=None, then_char=False, else_char=False,
               then_upper=False):
        a = Asm()
        a.at(0, ",")
        copy(a, 0, 1, 2)
        a.at(1, "-" * ord(letter))
        a.at(3, "+")
        a.at(1, "["); a.at(1, "[-]"); a.at(3, "-")
        if else_char:
            a.at(0, ".")
        else:
            print_text(a, 4, else_text)
        a.at(1, "]")
        a.at(3, "["); a.at(3, "-")
        if then_char:
            a.at(0, "-" * 32 + "." if then_upper else ".")
        else:
            print_text(a, 4, then_text)
        a.at(3, "]")
        return a.out()

    others = "abcfijlnoqrstxyzAKMP"
    return [
        ("Read one character. Print 'Y' if it is 'k', otherwise print 'N'.",
         char_or("k", others), lambda s: "Y" if s == "k" else "N", branch("k", "Y", "N")),
        ("Read one character. Print 'yes' if it is 'm', otherwise print 'no'.",
         char_or("m", others), lambda s: "yes" if s == "m" else "no", branch("m", "yes", "no")),
        ("Read one character. Print '*' if it is 'p', otherwise print the character itself.",
         char_or("p", others), lambda s: "*" if s == "p" else s,
         branch("p", "*", else_char=True)),
        ("Read one character. Print 'U!' if it is 'u', otherwise print '-'.",
         char_or("u", others), lambda s: "U!" if s == "u" else "-", branch("u", "U!", "-")),
        ("Read one character. Print 'match' if it is 'w', otherwise print 'no'.",
         char_or("w", others), lambda s: "match" if s == "w" else "no",
         branch("w", "match", "no")),
        ("Read one character. Print '1' if it is 'b', otherwise print '0'.",
         char_or("b", others), lambda s: "1" if s == "b" else "0", branch("b", "1", "0")),
        ("Read one character. If it is 'd' print it in uppercase, otherwise print '?'.",
         char_or("d", others), lambda s: "D" if s == "d" else "?",
         branch("d", then_char=True, then_upper=True, else_text="?")),
        ("Read one character. Print the character itself if it is 'g', otherwise print '.'.",
         char_or("g", others), lambda s: "g" if s == "g" else ".",
         branch("g", then_char=True, else_text=".")),
        ("Read one character. Print 'hit' if it is 'h', otherwise print 'miss'.",
         char_or("h", others), lambda s: "hit" if s == "h" else "miss",
         branch("h", "hit", "miss")),
        ("Read one character. Print 'T' if it is 'v', otherwise print 'F'.",
         char_or("v", others), lambda s: "T" if s == "v" else "F", branch("v", "T", "F")),
    ]


def level8():
    """count / filter / replace over a line."""
    def scan(letter, on_match=None, on_other=None, replacement=None, count=False):
        a = Asm()
        a.at(0, ","); a.at(0, "[")
        copy(a, 0, 1, 2)
        a.at(1, "-" * ord(letter))
        a.at(3, "+"); a.at(4, "+")
        a.at(1, "["); a.at(1, "[-]"); a.at(3, "-"); a.at(1, "]")
        a.at(3, "["); a.at(4, "-")
        if count:
            a.at(5, "+")
        if on_match == "print":
            a.at(0, ".")
        if replacement is not None:
            print_text(a, 6, replacement)
        a.at(3, "-]")
        a.at(4, "[")
        if on_other == "print":
            a.at(0, ".")
        a.at(4, "-]")
        a.at(0, ","); a.at(0, "]")
        if count:
            a.at(5, "+" * 48 + ".")
        return a.out()

    alpha = "abcdeghklmnprstuvw"
    return [
        ("Read the whole input and print how many times the letter 'k' appears, as one digit.",
         with_letter("k", alpha), lambda s: str(s.count("k")), scan("k", count=True)),
        ("Read the whole input and print how many times the letter 'm' appears, as one digit.",
         with_letter("m", alpha), lambda s: str(s.count("m")), scan("m", count=True)),
        ("Read the whole input and print it with every 'p' removed.", with_letter("p", alpha),
         lambda s: s.replace("p", ""), scan("p", on_other="print")),
        ("Read the whole input and print it with every 'u' removed.", with_letter("u", alpha),
         lambda s: s.replace("u", ""), scan("u", on_other="print")),
        ("Read the whole input and print it with every 'w' replaced by '*'.",
         with_letter("w", alpha), lambda s: s.replace("w", "*"),
         scan("w", on_other="print", replacement="*")),
        ("Read the whole input and print it with every 'b' replaced by 'B'.",
         with_letter("b", alpha), lambda s: s.replace("b", "B"),
         scan("b", on_other="print", replacement="B")),
        ("Read the whole input and print how many times the letter 'd' appears, as one digit.",
         with_letter("d", alpha), lambda s: str(s.count("d")), scan("d", count=True)),
        ("Read the whole input and print it with every 'g' removed.", with_letter("g", alpha),
         lambda s: s.replace("g", ""), scan("g", on_other="print")),
        ("Read the whole input and print it with every 'h' replaced by '_'.",
         with_letter("h", alpha), lambda s: s.replace("h", "_"),
         scan("h", on_other="print", replacement="_")),
        ("Read the whole input and print only its 'v' characters.", with_letter("v", alpha),
         lambda s: "v" * s.count("v"), scan("v", on_match="print")),
    ]


def level9():
    """multi-digit numbers: parse, arithmetic, print in decimal."""
    P = 10

    def two_digit_plus(k):
        a = Asm()
        a.at(0, "," + "-" * 48); a.at(0, "[-"); a.at(P, "+" * 10); a.at(0, "]")
        a.at(1, "," + "-" * 48); a.at(1, "[-"); a.at(P, "+"); a.at(1, "]")
        a.at(P, "+" * k if k >= 0 else "-" * -k)
        print_num(a, P)
        return a.out()

    def digit_sum():
        a = Asm()
        a.at(0, "," + "-" * 48); a.at(0, "[-"); a.at(P, "+"); a.at(0, "]")
        a.at(1, "," + "-" * 48); a.at(1, "[-"); a.at(P, "+"); a.at(1, "]")
        print_num(a, P)
        return a.out()

    def double_digit():
        a = Asm()
        a.at(0, "," + "-" * 48); a.at(0, "[-"); a.at(P, "++"); a.at(0, "]")
        print_num(a, P)
        return a.out()

    def product(offset=0):
        a = Asm()
        a.at(0, "," + "-" * 48); a.at(1, ","); a.at(2, "," + "-" * 48)
        a.at(0, "[-")
        copy(a, 2, P, 3)
        a.at(0, "]")
        a.at(P, "+" * offset)
        print_num(a, P)
        return a.out()

    def digit_add():
        a = Asm()
        a.at(0, "," + "-" * 48); a.at(1, ","); a.at(2, "," + "-" * 48)
        a.at(0, "[-"); a.at(P, "+"); a.at(0, "]")
        a.at(2, "[-"); a.at(P, "+"); a.at(2, "]")
        print_num(a, P)
        return a.out()

    def square():
        a = Asm()
        a.at(0, "," + "-" * 48)
        copy(a, 0, 2, 3)
        a.at(0, "[-")
        copy(a, 2, P, 3)
        a.at(0, "]")
        print_num(a, P)
        return a.out()

    two = lambda lo, hi: (lambda r: str(r.randint(lo, hi)))
    return [
        ("Read a two-digit number n and print n + 1 in decimal.", two(10, 98),
         lambda s: str(int(s) + 1), two_digit_plus(1)),
        ("Read a two-digit number n and print n + 7 in decimal.", two(10, 92),
         lambda s: str(int(s) + 7), two_digit_plus(7)),
        ("Read a two-digit number n and print n + 10 in decimal.", two(10, 89),
         lambda s: str(int(s) + 10), two_digit_plus(10)),
        ("Read a two-digit number and print the sum of its two digits in decimal.", two(10, 99),
         lambda s: str(int(s[0]) + int(s[1])), digit_sum()),
        ("Read a two-digit number n (at least 20) and print n - 13 in decimal.", two(20, 99),
         lambda s: str(int(s) - 13), two_digit_plus(-13)),
        ("Read one digit and print twice its value in decimal.", one_of(string.digits),
         lambda s: str(2 * int(s)), double_digit()),
        ("Read two digits separated by a space and print their product in decimal.",
         digit_pair(lambda x, y: True), lambda s: str(int(s[0]) * int(s[2])), product()),
        ("Read two digits separated by a space and print their product plus 1 in decimal.",
         digit_pair(lambda x, y: True), lambda s: str(int(s[0]) * int(s[2]) + 1), product(1)),
        ("Read two digits separated by a space and print their sum in decimal.",
         digit_pair(lambda x, y: True), lambda s: str(int(s[0]) + int(s[2])), digit_add()),
        ("Read one digit and print its square in decimal.", one_of(string.digits),
         lambda s: str(int(s) ** 2), square()),
    ]


def level10():
    """comparisons of two digits 'a b'."""
    def core(a, need_a_copy=False):
        # a in c0, b in c2 (both as values 0-9); c4/c5 working copies; c7 = max(a-b,0), c5 = max(b-a,0)
        a.at(0, "," + "-" * 48); a.at(1, ","); a.at(2, "," + "-" * 48)
        if need_a_copy:
            copy(a, 0, 12, 6)
        copy(a, 0, 4, 6)
        copy(a, 2, 5, 6)
        a.at(4, "[-")
        a.at(8, "+")
        a.at(5, "["); a.at(5, "-"); a.at(8, "-")
        a.at(5, "[-"); a.at(9, "+"); a.at(5, "]")
        a.at(5, "]")
        a.at(9, "[-"); a.at(5, "+"); a.at(9, "]")
        a.at(8, "[-"); a.at(7, "+"); a.at(8, "]")
        a.at(4, "]")

    def larger():
        a = Asm(); core(a)
        a.at(5, "[-"); a.at(0, "+"); a.at(5, "]"); a.at(0, "+" * 48 + ".")
        return a.out()

    def smaller():
        a = Asm(); core(a)
        a.at(7, "[-"); a.at(0, "-"); a.at(7, "]"); a.at(0, "+" * 48 + ".")
        return a.out()

    def three_way(gt, lt, eq):
        a = Asm(); core(a)
        a.at(10, "+")
        a.at(7, "["); a.at(7, "[-]"); a.at(10, "-"); print_text(a, 11, gt); a.at(7, "]")
        a.at(5, "["); a.at(5, "[-]"); a.at(10, "-"); print_text(a, 11, lt); a.at(5, "]")
        a.at(10, "["); a.at(10, "-"); print_text(a, 11, eq); a.at(10, "]")
        return a.out()

    def larger_then_smaller(sep=""):
        a = Asm(); core(a, need_a_copy=True)
        a.at(5, "[-"); a.at(12, "+"); a.at(5, "]")           # c12 = a + max(b-a,0) = larger
        a.at(7, "[-"); a.at(0, "-"); a.at(7, "]")            # c0 = a - max(a-b,0) = smaller
        if sep:
            a.at(0, "+" * 48 + "."); print_text(a, 11, sep); a.at(12, "+" * 48 + ".")
        else:
            a.at(12, "+" * 48 + "."); a.at(0, "+" * 48 + ".")
        return a.out()

    def abs_diff():
        a = Asm(); core(a)
        a.at(7, "[-"); a.at(12, "+"); a.at(7, "]")
        a.at(5, "[-"); a.at(12, "+"); a.at(5, "]")
        a.at(12, "+" * 48 + ".")
        return a.out()

    any_pair = digit_pair(lambda x, y: True)

    def mostly_equal(r):
        if r.random() < 0.4:
            x = r.randint(0, 9)
            return f"{x} {x}"
        return any_pair(r)

    diff_pair = digit_pair(lambda x, y: x != y)
    return [
        ("Read two digits separated by a space and print the larger one.", any_pair,
         lambda s: str(max(int(s[0]), int(s[2]))), larger()),
        ("Read two digits separated by a space and print the smaller one.", any_pair,
         lambda s: str(min(int(s[0]), int(s[2]))), smaller()),
        ("Read two digits separated by a space. Print '=' if they are equal, otherwise '!='.",
         mostly_equal, lambda s: "=" if s[0] == s[2] else "!=", three_way("!=", "!=", "=")),
        ("Read two digits a and b separated by a space. Print '>' if a > b, '<' if a < b, "
         "'=' if equal.", mostly_equal,
         lambda s: ">" if s[0] > s[2] else "<" if s[0] < s[2] else "=", three_way(">", "<", "=")),
        ("Read two different digits separated by a space. Print 'first' if the first is larger, "
         "otherwise 'second'.", diff_pair, lambda s: "first" if s[0] > s[2] else "second",
         three_way("first", "second", "")),
        ("Read two digits separated by a space and print the larger one followed by the "
         "smaller one.", any_pair,
         lambda s: str(max(int(s[0]), int(s[2]))) + str(min(int(s[0]), int(s[2]))),
         larger_then_smaller()),
        ("Read two digits separated by a space and print the absolute difference.", any_pair,
         lambda s: str(abs(int(s[0]) - int(s[2]))), abs_diff()),
        ("Read two digits a and b separated by a space. Print 'yes' if a >= b, otherwise 'no'.",
         mostly_equal, lambda s: "yes" if s[0] >= s[2] else "no", three_way("yes", "no", "yes")),
        ("Read two digits separated by a space. Print 'Y' if they are equal, otherwise 'N'.",
         mostly_equal, lambda s: "Y" if s[0] == s[2] else "N", three_way("N", "N", "Y")),
        ("Read two digits separated by a space and print the smaller, then 'x', then the larger.",
         any_pair,
         lambda s: str(min(int(s[0]), int(s[2]))) + "x" + str(max(int(s[0]), int(s[2]))),
         larger_then_smaller("x")),
    ]


LEVELS = [
    ("print a character", level1), ("print text", level2), ("one character in", level3),
    ("loop over input", level4), ("several cells", level5), ("digit math", level6),
    ("if / else", level7), ("count and filter", level8), ("multi-digit numbers", level9),
    ("comparisons", level10),
]


def pick_inputs(rng, sampler, fn, tries=2000):
    """5 distinct inputs whose expected outputs are not all the same, so a program that
    ignores its input cannot pass. Falls back to repeats only when the input space is
    smaller than 5 (e.g. 'one digit from 0-3')."""
    inputs, outputs = [], set()
    for _ in range(tries):
        if len(inputs) == TESTS_PER_PROBLEM:
            break
        i = sampler(rng)
        if i in inputs:
            continue
        last_slot = len(inputs) == TESTS_PER_PROBLEM - 1
        if last_slot and len(outputs | {fn(i)}) < 2:
            continue
        inputs.append(i)
        outputs.add(fn(i))
    while len(inputs) < TESTS_PER_PROBLEM:   # tiny input space: allow repeats
        inputs.append(sampler(rng))
    return inputs


def check(code, inp, out):
    r = run_bf(code, stdin=inp.encode("latin-1"), config=STRICT)
    return not r.error and r.output == out.encode("latin-1"), r


def build(seed=20260915):
    rng = random.Random(seed)
    problems, failures = [], []
    for li, (lname, maker) in enumerate(LEVELS, start=1):
        for pi, (desc, sampler, fn, ref) in enumerate(maker(), start=1):
            pid = f"L{li:02d}-{pi:02d}"
            if sampler is None:
                inputs = [""]
                extra = []
            else:
                inputs = pick_inputs(rng, sampler, fn)
                extra = [sampler(rng) for _ in range(EXTRA_CHECKS)]
            tests = [[i, fn(i)] for i in inputs]
            for i in inputs + extra:
                ok, r = check(ref, i, fn(i))
                if not ok:
                    failures.append((pid, desc, i, fn(i), r.output, r.error))
                    break
            problems.append({"id": pid, "level": li, "level_name": lname, "description": desc,
                             "tests": tests, "reference": ref})
    return problems, failures


if __name__ == "__main__":
    problems, failures = build()
    for f in failures:
        print("REFERENCE FAILS:", f)
    assert len(problems) == 100
    if not failures:
        json.dump({"name": "bf-ladder", "version": 1, "tests_per_problem": TESTS_PER_PROBLEM,
                   "prompt": "Write a Brainfuck program that does the following.\n\n{description}",
                   "problems": problems},
                  open("bf_ladder_v1.json", "w", encoding="utf-8"), indent=1)
        print("wrote bf_ladder_v1.json: 100 problems, all references verified")
