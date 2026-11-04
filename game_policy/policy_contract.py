"""Restricted policy syntax; Docker remains the OS isolation boundary."""
import ast
import builtins

BUILTINS = {name: getattr(builtins, name) for name in (
    'abs all any bool dict enumerate float int len list max min range reversed '
    'set sorted sum tuple zip').split()}
ALLOWED = set(('Module FunctionDef arguments arg Return Assign AugAssign If IfExp For While '
    'Break Continue Pass Expr Name Load Store Constant List Tuple Dict Set Subscript Slice '
    'BinOp UnaryOp BoolOp Compare Call Attribute keyword ListComp SetComp DictComp '
    'GeneratorExp comprehension Add Sub Mult Div FloorDiv Mod USub UAdd Not And Or '
    'Eq NotEq Lt LtE Gt GtE In NotIn Is IsNot').split())


def validate(code):
    if not isinstance(code, str) or len(code) > 20000:
        raise ValueError('Policy must be Python source of at most 20,000 characters')
    tree = ast.parse(code)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Return exactly one choose_action(state) function')
    function = tree.body[0]
    args = function.args
    if (function.name != 'choose_action' or len(args.args) != 1
            or args.args[0].arg != 'state' or args.posonlyargs or args.kwonlyargs
            or args.vararg or args.kwarg or args.defaults or args.kw_defaults):
        raise ValueError('Expected choose_action(state)')
    for node in ast.walk(tree):
        if type(node).__name__ not in ALLOWED:
            raise ValueError('Unsupported policy syntax: ' + type(node).__name__)
        if isinstance(node, ast.FunctionDef) and (
                node is not function or node.decorator_list or node.returns):
            raise ValueError('Nested functions, decorators and annotations are unsupported')
        if isinstance(node, ast.arg) and node.annotation:
            raise ValueError('Annotations are unsupported')
        if isinstance(node, ast.Name) and ((node.id.startswith('_') and node.id != '_') or node.id in {
                'open', 'exec', 'eval', 'compile', 'globals', 'locals', 'print',
                'getattr', 'setattr', 'input', 'help', 'breakpoint'}):
            raise ValueError('Policy references a forbidden name')
        if isinstance(node, ast.Attribute) and node.attr not in {'append', 'add'}:
            raise ValueError('Only append/add methods are supported; use builtins otherwise')
    return tree


def load(code):
    namespace = {'__builtins__': dict(BUILTINS)}
    exec(compile(validate(code), 'candidate.py', 'exec'), namespace)
    return namespace['choose_action']
