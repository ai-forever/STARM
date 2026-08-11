import importlib
import inspect


def load_model_class(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    # Import the module
    module = importlib.import_module(prefix + module_path)
    cls = getattr(module, class_name)

    return cls


def get_model_source_path(identifier: str, prefix: str = "models."):
    module_path, class_name = identifier.split('@')

    module = importlib.import_module(prefix + module_path)
    return inspect.getsourcefile(module)


def validate_expression(line):
    """
    Validator for expressions of the form "digits=answersexpression"

    Returns a tuple: (valid: bool, error_message: str, expression: str)
    """
    # Basic parsing of the line
    if '=' not in line:
        return False, "Missing '=' character", ""

    left, right = line.split('=', 1)

    # Check that the left part consists only of digits
    if not left.isdigit():
        return False, f"Left part must contain only digits, got: '{left}'", ""

    target_digits = list(left)  # e.g. ['3', '9', '8', '5', '4', '6']

    # Look for 's' in the right part
    if 's' not in right:
        return False, f"Right part missing 's', got: '{right}'", ""

    expr = right.split('s', 1)[1].strip()
    if not expr:
        return False, "Expression is empty", ""

    # Allowed characters check
    allowed_chars = set('0123456789+-*/()')
    if any(c not in allowed_chars for c in expr):
        invalid = ''.join(c for c in expr if c not in allowed_chars)
        return False, f"Invalid characters in expression: '{invalid}'", expr

    # Check for digit concatenation and order
    extracted_digits = []
    prev_was_digit = False

    for i, c in enumerate(expr):
        if c.isdigit():
            # If previous char was also a digit → concatenation!
            if prev_was_digit:
                # Find the boundaries of the glued number for error reporting
                start = i - 1
                while start >= 0 and expr[start].isdigit():
                    start -= 1
                start += 1
                end = i + 1
                while end < len(expr) and expr[end].isdigit():
                    end += 1
                glued = expr[start:end]
                return False, f"Digit concatenation is not allowed: '{glued}' (positions {start}-{end - 1})", expr

            extracted_digits.append(c)
            prev_was_digit = True
        else:
            prev_was_digit = False

    # Check order and count of digits
    if extracted_digits != target_digits:
        if len(extracted_digits) != len(target_digits):
            return False, (
                f"Incorrect number of digits: expected {len(target_digits)} "
                f"({','.join(target_digits)}), got {len(extracted_digits)} "
                f"({','.join(extracted_digits)})"
            ), expr
        # Find the first position where they differ
        for idx, (exp, got) in enumerate(zip(target_digits, extracted_digits)):
            if exp != got:
                return False, (
                    f"Digit order violation: at position {idx + 1} expected '{exp}', "
                    f"got '{got}'"
                ), expr
        return False, "Digits do not match the left part", expr

    # Syntax analysis: check parentheses balance
    balance = 0
    for i, c in enumerate(expr):
        if c == '(':
            balance += 1
        elif c == ')':
            balance -= 1
            if balance < 0:
                return False, f"Extra closing parenthesis at position {i}", expr
    if balance != 0:
        return False, f"Mismatched parentheses: {balance} unclosed", expr

    # Syntax analysis: check operators
    operators = set('+-*/')

    # Expression cannot start with a binary operator (except unary minus)
    if expr[0] in '+*/':
        return False, f"Expression cannot start with '{expr[0]}'", expr

    # Expression cannot end with an operator or an opening parenthesis
    if expr[-1] in operators or expr[-1] == '(':
        return False, f"Expression cannot end with '{expr[-1]}'", expr

    # Walk through the expression for detailed checks
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]

        # Check '//' – must be exactly two slashes in a row
        if c == '/':
            if i + 1 < n and expr[i + 1] == '/':
                # This is '//', make sure there is no triple '///'
                if i + 2 < n and expr[i + 2] == '/':
                    return False, f"Invalid '///' construction at position {i}", expr
                # Check that after '//' there is not an operator or closing parenthesis
                if i + 2 < n and (expr[i + 2] in operators or expr[i + 2] == ')'):
                    return False, f"Operator after '//' at position {i}", expr
                i += 2  # skip both slashes
                continue
            else:
                return False, f"Single '/' is not allowed (use '//') at position {i}", expr

        # Check double operators (except unary minus after '(' or operator)
        if c in '+-*' and i + 1 < n and expr[i + 1] in operators:
            # Allow '-(' as unary minus, but not '--', '-+', '-*', etc.
            if c == '-' and expr[i + 1] == '(':
                i += 1
                continue
            return False, f"Double operator '{c}{expr[i + 1]}' at position {i}", expr

        # After an opening parenthesis, only digit, '(' or unary '-' is allowed
        if c == '(':
            if i + 1 < n:
                next_char = expr[i + 1]
                if next_char not in '0123456789(-':
                    return False, f"After '(' expected digit, '(' or '-', got '{next_char}' at position {i + 1}", expr

        # Before a closing parenthesis, there must be a digit or ')'
        if c == ')' and i > 0:
            prev_char = expr[i - 1]
            if prev_char not in '0123456789)':
                return False, f"Before ')' expected digit or ')', got '{prev_char}' at position {i - 1}", expr

        # After a digit, there may be an operator, closing parenthesis or end of string
        if c.isdigit() and i + 1 < n:
            next_char = expr[i + 1]
            if next_char.isdigit():  # already checked concatenation earlier, but just in case
                return False, f"Digit concatenation after position {i}", expr

        i += 1

    # Extra check: no empty parentheses '()'
    if '()' in expr:
        pos = expr.index('()')
        return False, f"Empty parentheses '()' at position {pos}", expr

    return True, "OK", expr


def safe_evaluate(expr):
    """
    Safely evaluate an expression where '//' is used instead of '/'
    Returns: (success: bool, result_or_error: int|str)
    """
    # Replace '//' with '/' for eval, but we already validated
    try:
        # Extra protection: remove all built-in functions
        result = eval(
            expr.replace('//', '/'),
            {"__builtins__": {}},
            {}
        )
        # Integer division in Python is //
        return True, int(result)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except ZeroDivisionError:
        return False, "Division by zero"
    except Exception as e:
        return False, f"Evaluation error: {type(e).__name__}: {e}"


CHARSET = "p1234567890()=+-/*s"
char2id = {x: e for e, x in enumerate(CHARSET)}
id2char = {v: k for k, v in char2id.items()}


def restore(arr):
    return ''.join([id2char[v] for v in arr])
